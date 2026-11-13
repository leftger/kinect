//! The scan session that runs behind the window.
//!
//! A scan is a synchronous, expensive loop and the GUI has to stay responsive,
//! so the whole pipeline runs on a worker thread. The two sides talk through one
//! mutexed [`SessionView`] for everything the window displays and one command
//! channel for everything the window asks for. The view carries the latest
//! preview as an `Arc`, so handing it over copies a pointer and not a mesh.
//!
//! The worker owns the [`Scanner`] for the entire session, including after the
//! last frame. That is what makes "stop, look at it, then export" possible: the
//! scanner is held in a small command loop until the window exports it or
//! resets it. Closing the window asks for an export first (see `mod.rs`).
//!
//! Pause, stop, export and reset are checked between frames. A live frame can be
//! up to a few hundred milliseconds of USB wait, so a command is honoured late
//! by at most one frame.

use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use geom::{Intrinsics, transform_point};
use nalgebra::{Isometry3, Vector3};

use crate::scanner::{FrameColor, Scanner};
use crate::synthetic;
use crate::{DEFAULT_FRAMES, Options, capture, recording};

use super::render::{PreviewLine, PreviewPoint, PreviewScene};

/// How many depth pixels to skip between preview points.
///
/// The mesh is the main preview. These points exist so that the first frame
/// shows something before the first surface is extracted, and so that a frame
/// whose pose is wrong is visible immediately.
const POINT_STRIDE: usize = 4;

/// Where a session's frames come from.
#[derive(Clone, Debug)]
pub(crate) enum Source {
    /// The attached Kinect v2.
    Live,
    /// The attached Kinect v2, writing a `.k2df` recording as it reconstructs.
    Record(PathBuf),
    /// A `.k2df` recording. Needs no sensor.
    Replay(PathBuf),
    /// The analytic room in `synthetic`. Needs no sensor.
    Synthetic,
}

impl Source {
    fn label(&self) -> String {
        match self {
            Self::Live => "live sensor".to_string(),
            Self::Record(path) => format!("record to {}", path.display()),
            Self::Replay(path) => format!("replay {}", path.display()),
            Self::Synthetic => "synthetic room".to_string(),
        }
    }
}

/// Everything needed to start one scan.
pub(crate) struct SessionConfig {
    pub source: Source,
    /// The same options the CLI uses, so the GUI cannot quietly diverge.
    pub options: Options,
    /// Extract and publish a mesh every N frames. One means every frame.
    pub preview_every: usize,
    /// Frames per second for replay and synthetic playback. Zero is unthrottled.
    pub playback_fps: f32,
    /// How many recent colour views the preview may paint from. The export
    /// still uses all of them.
    pub color_preview_views: usize,
}

/// What the worker is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Idle,
    Starting,
    Running,
    Paused,
    Stopped,
    Exporting,
    Exported,
    Error,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Starting => "starting",
            Self::Running => "scanning",
            Self::Paused => "paused",
            Self::Stopped => "stopped",
            Self::Exporting => "exporting",
            Self::Exported => "exported",
            Self::Error => "error",
        }
    }

    /// Whether the session is still making progress on its own.
    pub fn is_active(self) -> bool {
        matches!(
            self,
            Self::Starting | Self::Running | Self::Paused | Self::Exporting
        )
    }
}

/// What the last export wrote as a Gaussian-splatting dataset.
///
/// A small copy of the exporter's report, so the UI can show it without
/// holding the `dataset` module's type in every snapshot.
#[derive(Clone, Debug)]
pub(crate) struct DatasetSummary {
    pub dir: PathBuf,
    pub views: usize,
    pub points: usize,
    pub skipped: usize,
    pub warnings: Vec<String>,
}

/// The state the GUI reads every repaint.
pub(crate) struct SessionView {
    pub status: Status,
    pub total: usize,
    pub frames_done: usize,
    pub fused: usize,
    pub rejected: usize,
    pub submaps: usize,
    pub loops: usize,
    pub loops_dropped: usize,
    pub loop_attempts: usize,
    pub loop_points: usize,
    pub loop_rehydrations: usize,
    pub online_updates: usize,
    pub last_online_correction: f32,
    pub loop_pending: usize,
    pub graph_pending: bool,
    pub graph_updates: usize,
    pub chunks: usize,
    pub fusion_skipped: usize,
    pub position: [f32; 3],
    pub blocks: usize,
    pub allocated_bytes: usize,
    pub tracking_ms: f32,
    pub fusion_ms: f32,
    pub submap_ms: f32,
    pub scene: Option<Arc<PreviewScene>>,
    pub scene_id: u64,
    pub message: String,
    pub log: Vec<String>,
    pub error: Option<String>,
    pub output: Option<PathBuf>,
    /// Vertices and triangles of the last exported mesh, when it had a surface.
    pub mesh: Option<(usize, usize)>,
    /// The last dataset hand-off, when an export requested one.
    pub dataset: Option<DatasetSummary>,
    pub started: Option<Instant>,
    pub finished: Option<Instant>,
}

impl Default for SessionView {
    fn default() -> Self {
        Self {
            status: Status::Idle,
            total: 0,
            frames_done: 0,
            fused: 0,
            rejected: 0,
            submaps: 0,
            loops: 0,
            loops_dropped: 0,
            loop_attempts: 0,
            loop_points: 0,
            loop_rehydrations: 0,
            online_updates: 0,
            last_online_correction: 0.0,
            loop_pending: 0,
            graph_pending: false,
            graph_updates: 0,
            chunks: 0,
            fusion_skipped: 0,
            position: [0.0; 3],
            blocks: 0,
            allocated_bytes: 0,
            tracking_ms: 0.0,
            fusion_ms: 0.0,
            submap_ms: 0.0,
            scene: None,
            scene_id: 0,
            message: String::new(),
            log: Vec::new(),
            error: None,
            output: None,
            mesh: None,
            dataset: None,
            started: None,
            finished: None,
        }
    }
}

impl SessionView {
    fn snapshot(&self) -> SessionSnapshot {
        let elapsed = self
            .started
            .map(|started| {
                self.finished
                    .unwrap_or_else(Instant::now)
                    .duration_since(started)
                    .as_secs_f32()
            })
            .unwrap_or(0.0);

        SessionSnapshot {
            status: self.status,
            total: self.total,
            frames_done: self.frames_done,
            fused: self.fused,
            rejected: self.rejected,
            submaps: self.submaps,
            loops: self.loops,
            loops_dropped: self.loops_dropped,
            loop_attempts: self.loop_attempts,
            loop_points: self.loop_points,
            loop_rehydrations: self.loop_rehydrations,
            online_updates: self.online_updates,
            last_online_correction: self.last_online_correction,
            loop_pending: self.loop_pending,
            graph_pending: self.graph_pending,
            graph_updates: self.graph_updates,
            chunks: self.chunks,
            fusion_skipped: self.fusion_skipped,
            position: self.position,
            blocks: self.blocks,
            allocated_bytes: self.allocated_bytes,
            tracking_ms: self.tracking_ms,
            fusion_ms: self.fusion_ms,
            submap_ms: self.submap_ms,
            scene: self.scene.clone(),
            scene_id: self.scene_id,
            message: self.message.clone(),
            log: self.log.clone(),
            error: self.error.clone(),
            output: self.output.clone(),
            mesh: self.mesh,
            dataset: self.dataset.clone(),
            elapsed,
        }
    }

    fn log(&mut self, line: impl Into<String>) {
        let line = line.into();
        self.message = line.clone();
        self.log.push(line);
        // A scan can run for a long time and the window only shows the tail.
        if self.log.len() > 200 {
            self.log.drain(..self.log.len() - 200);
        }
    }
}

/// A cheap copy of the view for the UI thread. The scene is an `Arc`.
#[derive(Clone)]
pub(crate) struct SessionSnapshot {
    pub status: Status,
    pub total: usize,
    pub frames_done: usize,
    pub fused: usize,
    pub rejected: usize,
    pub submaps: usize,
    pub loops: usize,
    pub loops_dropped: usize,
    pub loop_attempts: usize,
    pub loop_points: usize,
    pub loop_rehydrations: usize,
    pub online_updates: usize,
    pub last_online_correction: f32,
    pub loop_pending: usize,
    pub graph_pending: bool,
    pub graph_updates: usize,
    pub chunks: usize,
    pub fusion_skipped: usize,
    pub position: [f32; 3],
    pub blocks: usize,
    pub allocated_bytes: usize,
    pub tracking_ms: f32,
    pub fusion_ms: f32,
    pub submap_ms: f32,
    pub scene: Option<Arc<PreviewScene>>,
    pub scene_id: u64,
    pub message: String,
    pub log: Vec<String>,
    pub error: Option<String>,
    pub output: Option<PathBuf>,
    /// Vertices and triangles of the last exported mesh.
    pub mesh: Option<(usize, usize)>,
    /// The last dataset hand-off, when one was requested and written.
    pub dataset: Option<DatasetSummary>,
    pub elapsed: f32,
}

impl Default for SessionSnapshot {
    fn default() -> Self {
        SessionView::default().snapshot()
    }
}

/// Commands from the window to the worker.
#[derive(Clone, Copy, Debug)]
enum Command {
    Pause,
    Resume,
    Stop,
    /// Stop the scan if it is still running, then write the scan out.
    Export,
    /// Stop, write the scan out, and end the worker. Used when the window
    /// closes, so `join` can never wait on a scanner held for a later export.
    Finish,
    /// Stop and throw the scan away, leaving the session idle.
    Reset,
}

/// Flags the scan loop polls between frames.
#[derive(Default)]
struct Control {
    paused: AtomicBool,
    stop: AtomicBool,
    export: AtomicBool,
    /// Export and leave the worker instead of holding the scanner.
    exit_after_export: AtomicBool,
    discard: AtomicBool,
}

/// The window's handle on a running session.
pub(crate) struct SessionHandle {
    shared: Arc<Mutex<SessionView>>,
    commands: Sender<Command>,
    join: Option<JoinHandle<()>>,
}

impl SessionHandle {
    pub fn snapshot(&self) -> SessionSnapshot {
        self.shared.lock().unwrap().snapshot()
    }

    pub fn pause(&self) {
        let _ = self.commands.send(Command::Pause);
    }

    pub fn resume(&self) {
        let _ = self.commands.send(Command::Resume);
    }

    pub fn stop(&self) {
        let _ = self.commands.send(Command::Stop);
    }

    pub fn export(&self) {
        let _ = self.commands.send(Command::Export);
    }

    pub fn reset(&self) {
        let _ = self.commands.send(Command::Reset);
    }

    /// Whether the worker has left its loop. The window uses this to join and
    /// drop the handle after a reset or an error.
    pub fn finished(&self) -> bool {
        self.join
            .as_ref()
            .map(|join| join.is_finished())
            .unwrap_or(true)
    }

    /// Export an unexported scan and end the worker. The window calls this on
    /// close; unlike `export`, it guarantees the thread can be joined.
    pub fn finish(&self) {
        let _ = self.commands.send(Command::Finish);
    }

    /// Wait for the worker. Called after reset, or when the window closes.
    pub fn join(&mut self) {
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }

    /// Wait for the worker, but do not hang the window forever if the sensor or
    /// a frame-to-model raycast is stuck. A `false` result means the handle was
    /// left detached; the process is about to exit anyway.
    pub fn join_timeout(&mut self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;

        while !self.finished() {
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(20));
        }

        self.join();
        true
    }
}

/// Start a session on its own thread.
pub(crate) fn spawn(config: SessionConfig) -> SessionHandle {
    let total = expected_total(&config);
    let started = Instant::now();

    let mut view = SessionView {
        status: Status::Starting,
        total,
        started: Some(started),
        ..SessionView::default()
    };
    view.log(format!("source: {}", config.source.label()));
    view.log(format!(
        "preview every {} frame(s); export to {}",
        config.preview_every.max(1),
        config
            .options
            .out
            .clone()
            .unwrap_or_else(|| PathBuf::from("scan.ply"))
            .display()
    ));

    let shared = Arc::new(Mutex::new(view));
    let control = Arc::new(Control::default());
    let (commands, receiver) = mpsc::channel();

    let worker_shared = Arc::clone(&shared);
    let worker_control = Arc::clone(&control);
    let join = thread::Builder::new()
        .name("scan-session".to_string())
        .spawn(move || worker(config, worker_shared, worker_control, receiver))
        .expect("spawning the scan worker");

    SessionHandle {
        shared,
        commands,
        join: Some(join),
    }
}

fn expected_total(config: &SessionConfig) -> usize {
    match &config.source {
        Source::Live | Source::Record(_) => config.options.frames.unwrap_or(DEFAULT_FRAMES),
        Source::Synthetic => config.options.frames.unwrap_or(24).max(2),
        // Unknown until the recording is opened.
        Source::Replay(_) => 0,
    }
}

fn worker(
    config: SessionConfig,
    shared: Arc<Mutex<SessionView>>,
    control: Arc<Control>,
    commands: Receiver<Command>,
) {
    match execute(&config, &shared, &control, &commands) {
        Ok(Some(mut scanner)) => {
            // Export was requested while the scan was still running.
            if control.export.load(Ordering::Relaxed) {
                export_scanner(&mut scanner, &config.options, &shared);
                return;
            }

            {
                let mut view = shared.lock().unwrap();
                view.status = Status::Stopped;
                view.finished = Some(Instant::now());
                view.log("scan finished; press Export to write it, or Reset to discard");
            }

            hold_scanner(scanner, &config, &shared, &commands);
        }
        Ok(None) => {
            let mut view = shared.lock().unwrap();
            view.status = Status::Idle;
            view.finished = Some(Instant::now());
            view.scene = None;
            view.mesh = None;
            view.dataset = None;
            view.log("scan discarded");
        }
        Err(error) => {
            let mut view = shared.lock().unwrap();
            view.status = Status::Error;
            view.finished = Some(Instant::now());
            view.error = Some(error.to_string());
            view.log(format!("error: {error}"));
        }
    }
}

/// Keep the finished scanner alive until the window exports or discards it.
fn hold_scanner(
    mut scanner: Scanner,
    config: &SessionConfig,
    shared: &Arc<Mutex<SessionView>>,
    commands: &Receiver<Command>,
) {
    loop {
        match commands.recv_timeout(Duration::from_millis(120)) {
            Ok(Command::Export) => export_scanner(&mut scanner, &config.options, shared),
            Ok(Command::Finish) => {
                export_scanner(&mut scanner, &config.options, shared);
                return;
            }
            Ok(Command::Reset) => {
                let mut view = shared.lock().unwrap();
                view.status = Status::Idle;
                view.scene = None;
                view.output = None;
                view.mesh = None;
                view.dataset = None;
                view.error = None;
                view.log("scan discarded");
                return;
            }
            // Meaningless once the scan has stopped.
            Ok(Command::Pause) | Ok(Command::Resume) | Ok(Command::Stop) => {}
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn export_scanner(scanner: &mut Scanner, options: &Options, shared: &Arc<Mutex<SessionView>>) {
    {
        let mut view = shared.lock().unwrap();
        view.status = Status::Exporting;
        view.log(format!(
            "exporting to {} ...",
            options
                .out
                .clone()
                .unwrap_or_else(|| PathBuf::from("scan.ply"))
                .display()
        ));
    }

    match crate::finish_scan_with_report(scanner, options) {
        Ok(summary) => {
            let written = summary.output.display().to_string();
            let warnings = summary.quality.warnings.clone();
            let level = summary.level;
            let dataset_log = summary.dataset.as_ref().map(|report| {
                format!(
                    "dataset: {} views, {} initial points in {}",
                    report.views,
                    report.points,
                    report.dir.display()
                )
            });

            let mut view = shared.lock().unwrap();
            view.status = Status::Exported;
            view.output = Some(summary.output);
            view.mesh = summary.mesh;
            view.loops = summary.loops;
            view.loops_dropped = summary.loops_dropped;
            view.loop_attempts = summary.loop_attempts;
            view.dataset = summary.dataset.map(|report| DatasetSummary {
                dir: report.dir,
                views: report.views,
                points: report.points,
                skipped: report.skipped,
                warnings: warnings.clone(),
            });
            view.finished = Some(Instant::now());
            view.error = None;
            view.log(format!("exported {written}"));
            match level {
                crate::LevelOutcome::Off => {}
                crate::LevelOutcome::Applied { tilt_degrees } => {
                    view.log(format!(
                        "levelled: rotated {tilt_degrees:.1} degrees so the floor is horizontal"
                    ));
                }
                crate::LevelOutcome::NotFound => {
                    view.log(
                        "level: no large horizontal surface found; exported as scanned"
                            .to_string(),
                    );
                }
            }
            if let Some(line) = dataset_log {
                view.log(line);
            }
            for warning in &warnings {
                view.log(format!("dataset warning: {warning}"));
            }
        }
        Err(error) => {
            // The window may be closing, so the log entry alone can be lost.
            eprintln!("scan gui: export failed: {error}");
            let mut view = shared.lock().unwrap();
            view.status = Status::Error;
            view.error = Some(error.to_string());
            view.log(format!("export failed: {error}"));
        }
    }
}

/// Run the chosen source to completion. `None` means the user reset the scan
/// before it ended, so there is nothing to export.
fn execute(
    config: &SessionConfig,
    shared: &Arc<Mutex<SessionView>>,
    control: &Control,
    commands: &Receiver<Command>,
) -> Result<Option<Scanner>, Box<dyn Error>> {
    match &config.source {
        Source::Live => run_live(config, shared, control, commands, None),
        Source::Record(path) => run_live(config, shared, control, commands, Some(path)),
        Source::Replay(path) => run_replay(path, config, shared, control, commands),
        Source::Synthetic => run_synthetic(config, shared, control, commands),
    }
}

/// End a scan that was stopped from the window.
///
/// A plain Stop keeps the scanner so the window can inspect and export it; only
/// Reset sets `discard`. Returning `None` for a plain Stop would throw the scan
/// away and make the Export button useless after it.
fn finish_session(
    state: SessionState,
    control: &Control,
) -> Result<Option<Scanner>, Box<dyn Error>> {
    if control.discard.load(Ordering::Relaxed) {
        Ok(None)
    } else {
        Ok(Some(state.scanner))
    }
}

/// Shared per-frame state. Keeps the scanner and the last frame's points.
struct SessionState {
    scanner: Scanner,
    options: Options,
    intrinsics: Intrinsics,
    width: usize,
    height: usize,
    frame: usize,
    last_points: Vec<PreviewPoint>,
    last_pose: Isometry3<f32>,
    preview_every: usize,
    color_preview_views: usize,
}

impl SessionState {
    fn new(intrinsics: Intrinsics, width: usize, height: usize, config: &SessionConfig) -> Self {
        Self {
            scanner: Scanner::new(intrinsics, crate::scanner_config(&config.options)),
            options: config.options.clone(),
            intrinsics,
            width,
            height,
            frame: 0,
            last_points: Vec::new(),
            last_pose: Isometry3::identity(),
            preview_every: config.preview_every.max(1),
            color_preview_views: config.color_preview_views,
        }
    }

    fn add(
        &mut self,
        depth: &[f32],
        color: Option<FrameColor<'_>>,
        shared: &Arc<Mutex<SessionView>>,
    ) {
        self.frame += 1;
        let report = self
            .scanner
            .add_frame_with_color(depth, self.width, self.height, color);

        self.last_pose = report.track.pose;
        self.last_points = build_points(
            depth,
            &self.last_pose,
            &self.intrinsics,
            self.width,
            self.height,
            self.options.unmirror,
        );

        let mut view = shared.lock().unwrap();
        view.frames_done = self.frame;
        view.fused = self.scanner.fused();
        view.rejected = self.scanner.rejected();
        view.submaps = self.scanner.submaps();
        view.loops = self.scanner.loops();
        view.loop_attempts = self.scanner.loop_attempts();
        view.loop_points = self.scanner.loop_keyframe_points();
        view.loop_rehydrations = self.scanner.loop_rehydrations();
        view.online_updates = self.scanner.online_loop_updates();
        view.last_online_correction = self.scanner.last_online_correction();
        view.loop_pending = self.scanner.loop_pending();
        view.graph_pending = self.scanner.graph_pending();
        view.graph_updates = self.scanner.background_graph_updates();
        view.chunks = self.scanner.chunks();
        view.fusion_skipped = self.scanner.fusion_skipped();
        view.blocks = report.blocks;
        view.allocated_bytes = report.allocated_bytes;

        let frames = self.scanner.frames().max(1) as f64;
        view.tracking_ms = (self.scanner.total_tracking().as_secs_f64() * 1000.0 / frames) as f32;
        view.fusion_ms = (self.scanner.total_fusion().as_secs_f64() * 1000.0 / frames) as f32;
        view.submap_ms = (self.scanner.total_submap().as_secs_f64() * 1000.0 / frames) as f32;

        let position = self.scanner.pose().translation.vector;
        view.position = if self.options.unmirror {
            [-position.x, position.y, position.z]
        } else {
            [position.x, position.y, position.z]
        };
    }

    fn maybe_publish(&self, shared: &Arc<Mutex<SessionView>>, force: bool) {
        if !force && self.frame > 1 && self.frame % self.preview_every != 0 {
            return;
        }
        self.publish(shared);
    }

    fn publish(&self, shared: &Arc<Mutex<SessionView>>) {
        let mut mesh = self.scanner.preview_mesh(self.color_preview_views);
        if self.options.unmirror {
            mesh.flip_x();
        }

        let mut lines = trajectory_lines(self.scanner.trajectory(), self.options.unmirror);
        lines.extend(sensor_lines(&self.last_pose, self.options.unmirror));

        let bounds = mesh.bounds();
        let scene = Arc::new(PreviewScene {
            mesh,
            points: self.last_points.clone(),
            lines,
            bounds,
        });

        let mut view = shared.lock().unwrap();
        view.scene = Some(scene);
        view.scene_id = view.scene_id.wrapping_add(1);
    }
}

fn run_live(
    config: &SessionConfig,
    shared: &Arc<Mutex<SessionView>>,
    control: &Control,
    commands: &Receiver<Command>,
    record_to: Option<&PathBuf>,
) -> Result<Option<Scanner>, Box<dyn Error>> {
    let frames = config.options.frames.unwrap_or(DEFAULT_FRAMES);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    runtime.block_on(async {
        let mut capture = capture::Capture::open(
            config.options.filters,
            config.options.color,
            config.options.gpu,
        )
        .await?;
        let intrinsics = capture.intrinsics();
        let (width, height) = capture.dimensions();

        // Open the recording before the first frame. If it cannot be created,
        // stop the capture and fail before a scan is spent.
        let mut writer = match record_to {
            Some(path) => match recording::FrameWriter::create_with_color(
                path,
                intrinsics,
                width,
                height,
                config.options.color,
            ) {
                Ok(writer) => Some(writer),
                Err(error) => {
                    if let Err(stop_error) = capture.stop().await {
                        shared
                            .lock()
                            .unwrap()
                            .log(format!("stopping the sensor failed: {stop_error}"));
                    }
                    return Err(
                        format!("cannot create recording {}: {error}", path.display()).into(),
                    );
                }
            },
            None => None,
        };
        let recording_path = record_to.map(|path| path.display().to_string());

        {
            let mut view = shared.lock().unwrap();
            view.status = Status::Running;
            view.total = frames;
            view.log(format!(
                "sensor open: {width}x{height}, fx={:.2} fy={:.2}",
                intrinsics.fx, intrinsics.fy
            ));
            if let Some(path) = &recording_path {
                view.log(format!("recording frames to {path}"));
            }
        }

        let mut state = SessionState::new(intrinsics, width, height, config);
        let recorded_from = Instant::now();

        for index in 1..=frames {
            if !pump(control, commands, shared) {
                finish_recording(writer.take(), recording_path.as_deref(), shared);
                // Stopping the streams failed should not cost the scan.
                if let Err(error) = capture.stop().await {
                    shared
                        .lock()
                        .unwrap()
                        .log(format!("stopping the sensor failed: {error}"));
                }
                return finish_session(state, control);
            }

            let depth = match capture.next_frame().await {
                Ok(frame) => frame.to_vec(),
                Err(error) => {
                    finish_recording(writer.take(), recording_path.as_deref(), shared);
                    // A sensor that drops out should not throw away the model
                    // built so far. Stop like a user Stop, and let the window
                    // export what it has.
                    shared
                        .lock()
                        .unwrap()
                        .log(format!("capture stopped: {error}"));
                    if let Err(error) = capture.stop().await {
                        shared
                            .lock()
                            .unwrap()
                            .log(format!("stopping the sensor failed: {error}"));
                    }
                    return finish_session(state, control);
                }
            };

            // Write the recording first, then hand the same frame to the
            // scanner. A failed write disables recording but keeps the scan.
            let mut write_failed = None;
            if let Some(active) = writer.as_mut() {
                let recorded_color = capture.color().map(|captured| FrameColor {
                    rgb: captured.rgb.as_slice(),
                    depth: captured.depth.as_slice(),
                    valid: captured.valid.as_slice(),
                    exposure: captured.exposure,
                    gain: captured.gain,
                    gamma: captured.gamma,
                });
                if let Err(error) = active.write_frame_with_color(
                    recorded_from.elapsed().as_secs_f32(),
                    &depth,
                    recorded_color,
                ) {
                    write_failed = Some(error);
                }
            }
            if let Some(error) = write_failed {
                shared
                    .lock()
                    .unwrap()
                    .log(format!("recording failed: {error}; continuing without it"));
                writer = None;
            }

            let color = capture.color().map(|captured| FrameColor {
                rgb: captured.rgb.as_slice(),
                depth: captured.depth.as_slice(),
                valid: captured.valid.as_slice(),
                exposure: captured.exposure,
                gain: captured.gain,
                gamma: captured.gamma,
            });

            state.add(&depth, color, shared);
            state.maybe_publish(shared, index == 1);
        }

        finish_recording(writer.take(), recording_path.as_deref(), shared);
        state.maybe_publish(shared, true);
        if let Err(error) = capture.stop().await {
            // A failure to shut the streams down must not discard the scan.
            shared
                .lock()
                .unwrap()
                .log(format!("stopping the sensor failed: {error}"));
        }
        Ok(Some(state.scanner))
    })
}

/// Close an optional recording and report what it wrote.
///
/// The writer owns the file, so every exit path from `run_live` calls this,
/// including Stop, Reset and a sensor that drops out.
fn finish_recording(
    writer: Option<recording::FrameWriter>,
    path: Option<&str>,
    shared: &Arc<Mutex<SessionView>>,
) {
    let Some(writer) = writer else {
        return;
    };
    match writer.finish() {
        Ok(count) => {
            if let Some(path) = path {
                shared
                    .lock()
                    .unwrap()
                    .log(format!("recorded {count} frames to {path}"));
            }
        }
        Err(error) => {
            shared
                .lock()
                .unwrap()
                .log(format!("finishing the recording failed: {error}"));
        }
    }
}

fn run_replay(
    path: &PathBuf,
    config: &SessionConfig,
    shared: &Arc<Mutex<SessionView>>,
    control: &Control,
    commands: &Receiver<Command>,
) -> Result<Option<Scanner>, Box<dyn Error>> {
    let recording = recording::read(path)?;
    if !recording.has_color && config.options.dataset.is_some() {
        // Fail before spending the replay. `write_dataset` now returns an error
        // too, but discovering this after a long scan is the wrong time.
        return Err(
            "cannot export dataset from this recording: it was captured without colour".into(),
        );
    }

    let limit = config.options.frames.unwrap_or(usize::MAX);
    let total = recording.frames.len().min(limit);

    {
        let mut view = shared.lock().unwrap();
        view.status = Status::Running;
        view.total = total;
        view.log(format!(
            "replay: {} frames of {}x{} (colour: {})",
            total,
            recording.width,
            recording.height,
            if recording.has_color { "yes" } else { "none" }
        ));
    }

    let mut state = SessionState::new(
        recording.intrinsics,
        recording.width,
        recording.height,
        config,
    );
    let mut pace = Pace::new(config.playback_fps);

    for frame in recording.frames.iter().take(total) {
        if !pump(control, commands, shared) {
            return finish_session(state, control);
        }

        let color = frame.color.as_ref().map(|color| FrameColor {
            rgb: &color.rgb,
            depth: &color.depth,
            valid: &color.valid,
            exposure: color.exposure,
            gain: color.gain,
            gamma: color.gamma,
        });

        state.add(&frame.depth, color, shared);
        state.maybe_publish(shared, state.frame == 1);
        pace.wait();
    }

    state.maybe_publish(shared, true);
    Ok(Some(state.scanner))
}

fn run_synthetic(
    config: &SessionConfig,
    shared: &Arc<Mutex<SessionView>>,
    control: &Control,
    commands: &Receiver<Command>,
) -> Result<Option<Scanner>, Box<dyn Error>> {
    let count = config.options.frames.unwrap_or(24).max(2);
    let intrinsics = synthetic::intrinsics();
    let shots = synthetic::shots(count);

    {
        let mut view = shared.lock().unwrap();
        view.status = Status::Running;
        view.total = count;
        view.log(format!(
            "synthetic room: {} shots of {}x{}",
            shots.len(),
            synthetic::WIDTH,
            synthetic::HEIGHT
        ));
    }

    let mut state = SessionState::new(intrinsics, synthetic::WIDTH, synthetic::HEIGHT, config);
    let mut pace = Pace::new(config.playback_fps);

    for shot in &shots {
        if !pump(control, commands, shared) {
            return finish_session(state, control);
        }

        let rendered = synthetic::render(shot, &intrinsics);
        let color = FrameColor {
            rgb: &rendered.rgb,
            depth: &rendered.depth,
            valid: &rendered.valid,
            exposure: shot.brightness,
            gain: 1.0,
            gamma: 1.0,
        };

        state.add(&rendered.depth, Some(color), shared);
        state.maybe_publish(shared, state.frame == 1);
        pace.wait();
    }

    state.maybe_publish(shared, true);
    Ok(Some(state.scanner))
}

/// A fixed-rate pacer for sources that are not a real sensor.
struct Pace {
    interval: Option<Duration>,
    next: Instant,
}

impl Pace {
    fn new(fps: f32) -> Self {
        let interval = if fps.is_finite() && fps > 0.0 {
            Some(Duration::from_secs_f32(1.0 / fps))
        } else {
            None
        };
        Self {
            interval,
            next: Instant::now(),
        }
    }

    fn wait(&mut self) {
        let Some(interval) = self.interval else {
            return;
        };

        self.next += interval;
        let now = Instant::now();
        if self.next > now {
            thread::sleep(self.next - now);
        } else {
            // The work overran the frame; do not build up a backlog.
            self.next = now;
        }
    }
}

/// Apply pending commands, then block while paused. Returns whether the scan
/// should keep going.
fn pump(control: &Control, commands: &Receiver<Command>, shared: &Arc<Mutex<SessionView>>) -> bool {
    loop {
        match commands.try_recv() {
            Ok(command) => apply(command, control, shared),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                control.stop.store(true, Ordering::Relaxed);
                break;
            }
        }
    }

    while control.paused.load(Ordering::Relaxed) && !control.stop.load(Ordering::Relaxed) {
        match commands.recv_timeout(Duration::from_millis(50)) {
            Ok(command) => apply(command, control, shared),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                control.stop.store(true, Ordering::Relaxed);
            }
        }
    }

    !control.stop.load(Ordering::Relaxed)
}

fn apply(command: Command, control: &Control, shared: &Arc<Mutex<SessionView>>) {
    match command {
        Command::Pause => {
            control.paused.store(true, Ordering::Relaxed);
            let mut view = shared.lock().unwrap();
            if view.status == Status::Running {
                view.status = Status::Paused;
                view.log("paused");
            }
        }
        Command::Resume => {
            control.paused.store(false, Ordering::Relaxed);
            let mut view = shared.lock().unwrap();
            if view.status == Status::Paused {
                view.status = Status::Running;
                view.log("resumed");
            }
        }
        Command::Stop => {
            control.stop.store(true, Ordering::Relaxed);
        }
        Command::Export => {
            control.stop.store(true, Ordering::Relaxed);
            control.export.store(true, Ordering::Relaxed);
        }
        Command::Finish => {
            control.stop.store(true, Ordering::Relaxed);
            control.export.store(true, Ordering::Relaxed);
            control.exit_after_export.store(true, Ordering::Relaxed);
        }
        Command::Reset => {
            control.stop.store(true, Ordering::Relaxed);
            control.discard.store(true, Ordering::Relaxed);
        }
    }
}

/// The latest frame's measured depth, as world-space points.
fn build_points(
    depth: &[f32],
    pose: &Isometry3<f32>,
    intrinsics: &Intrinsics,
    width: usize,
    height: usize,
    unmirror: bool,
) -> Vec<PreviewPoint> {
    let mut points = Vec::with_capacity((width / POINT_STRIDE + 1) * (height / POINT_STRIDE + 1));

    for y in (0..height).step_by(POINT_STRIDE) {
        for x in (0..width).step_by(POINT_STRIDE) {
            let index = y * width + x;
            let Some(&depth) = depth.get(index) else {
                continue;
            };
            if !depth.is_finite() || depth <= 0.0 {
                continue;
            }

            let camera = intrinsics.back_project(x as f32 + 0.5, y as f32 + 0.5, depth);
            let world = transform_point(pose, &camera);

            points.push(PreviewPoint {
                position: if unmirror {
                    [-world.x, world.y, world.z]
                } else {
                    [world.x, world.y, world.z]
                },
                color: depth_color(depth),
            });
        }
    }

    points
}

/// Warm near, cool far. The mesh carries real colour when there is any; these
/// points are a range cue and a warning that a frame saw something odd.
fn depth_color(depth: f32) -> [u8; 3] {
    let t = ((depth - 0.5) / 4.0).clamp(0.0, 1.0);
    let near = [255.0, 176.0, 76.0];
    let far = [74.0, 146.0, 255.0];

    [
        (near[0] + (far[0] - near[0]) * t) as u8,
        (near[1] + (far[1] - near[1]) * t) as u8,
        (near[2] + (far[2] - near[2]) * t) as u8,
    ]
}

fn trajectory_lines(trajectory: &[Vector3<f32>], unmirror: bool) -> Vec<PreviewLine> {
    let point = |p: &Vector3<f32>| {
        if unmirror {
            [-p.x, p.y, p.z]
        } else {
            [p.x, p.y, p.z]
        }
    };

    trajectory
        .windows(2)
        .map(|pair| PreviewLine {
            a: point(&pair[0]),
            b: point(&pair[1]),
            color: [80, 210, 235],
        })
        .collect()
}

fn sensor_lines(pose: &Isometry3<f32>, unmirror: bool) -> Vec<PreviewLine> {
    let corners = [
        Vector3::new(-0.07, -0.05, 0.20),
        Vector3::new(0.07, -0.05, 0.20),
        Vector3::new(0.07, 0.05, 0.20),
        Vector3::new(-0.07, 0.05, 0.20),
    ];
    let mirror = |p: Vector3<f32>| {
        if unmirror {
            [-p.x, p.y, p.z]
        } else {
            [p.x, p.y, p.z]
        }
    };

    let origin = mirror(transform_point(pose, &Vector3::zeros()));
    let corners: Vec<[f32; 3]> = corners
        .iter()
        .map(|corner| mirror(transform_point(pose, corner)))
        .collect();

    let mut lines = Vec::with_capacity(8);
    for corner in &corners {
        lines.push(PreviewLine {
            a: origin,
            b: *corner,
            color: [250, 220, 80],
        });
    }
    for index in 0..corners.len() {
        lines.push(PreviewLine {
            a: corners[index],
            b: corners[(index + 1) % corners.len()],
            color: [250, 220, 80],
        });
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use geom::tsdf::TsdfParams;

    fn synthetic_config(frames: usize) -> SessionConfig {
        let mut options = Options::default();
        options.frames = Some(frames);
        options.color = true;
        // Four-centimetre voxels keep this a test rather than a benchmark: the
        // default centimetre volume over a whole synthetic room is enormous in
        // an unoptimised build, and the preview only has to be non-empty.
        options.tsdf = TsdfParams {
            voxel_size: 0.04,
            truncation: 0.16,
            ..TsdfParams::default()
        };

        SessionConfig {
            source: Source::Synthetic,
            options,
            preview_every: 4,
            playback_fps: 0.0,
            color_preview_views: 4,
        }
    }

    #[test]
    fn a_synthetic_scan_fuses_frames_and_publishes_a_surface() {
        // Few shots keep this quick in an unoptimised build. The point is to
        // exercise tracking, fusion, publication and export, not to grade the
        // tracker.
        let mut config = synthetic_config(12);
        let out = std::env::temp_dir().join(format!("scan-gui-test-{}.ply", std::process::id()));
        let dataset = std::env::temp_dir().join(format!("scan-gui-test-ds-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dataset);
        config.options.out = Some(out.clone());
        config.options.dataset = Some(dataset.clone());

        let shared = Arc::new(Mutex::new(SessionView::default()));
        let control = Control::default();
        let (_commands, receiver) = mpsc::channel();

        let mut scanner = execute(&config, &shared, &control, &receiver)
            .expect("the synthetic room needs no sensor")
            .expect("the scan was not discarded");

        assert_eq!(scanner.frames(), 12);
        assert!(
            scanner.fused() >= 4,
            "only {} frames fused",
            scanner.fused()
        );

        {
            let view = shared.lock().unwrap();
            assert_eq!(view.frames_done, 12);
            assert_eq!(view.total, 12);
            assert!(view.log.iter().any(|line| line.contains("synthetic")));

            let scene = view.scene.as_ref().expect("a preview was published");
            assert!(!scene.mesh.is_empty(), "the synthetic room has a surface");
            assert!(
                !scene.points.is_empty(),
                "the last frame contributed points"
            );
            assert!(scene.bounds.is_some());
        }

        // The window's Drop path ends here: stop, then export the held scanner.
        export_scanner(&mut scanner, &config.options, &shared);
        assert!(out.exists(), "export did not write {}", out.display());
        {
            let view = shared.lock().unwrap();
            assert_eq!(view.status, Status::Exported);
            let mesh = view.mesh.expect("the export reported its mesh");
            assert!(mesh.0 > 0 && mesh.1 > 0);
            let summary = view
                .dataset
                .as_ref()
                .expect("the export reported its dataset");
            assert!(summary.views > 0);
            assert!(summary.points > 0);
            assert_eq!(summary.dir, dataset);
        }

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_dir_all(&dataset);
        let _ = std::fs::remove_file(out.with_extension("trajectory.ply"));
    }

    #[test]
    fn a_replay_that_cannot_be_opened_is_an_error() {
        let mut options = Options::default();
        options.frames = Some(4);
        let config = SessionConfig {
            source: Source::Replay(PathBuf::from("/nonexistent/capture.k2df")),
            options,
            preview_every: 1,
            playback_fps: 0.0,
            color_preview_views: 4,
        };

        let shared = Arc::new(Mutex::new(SessionView::default()));
        let control = Control::default();
        let (_commands, receiver) = mpsc::channel();

        assert!(execute(&config, &shared, &control, &receiver).is_err());
    }

    #[test]
    fn a_depth_only_replay_cannot_export_a_dataset() {
        let path =
            std::env::temp_dir().join(format!("scan-gui-depth-only-{}.k2df", std::process::id()));
        let intrinsics = synthetic::intrinsics();
        let mut writer =
            recording::FrameWriter::create(&path, intrinsics, 4, 4).expect("create recording");
        writer
            .write_frame(0.0, &vec![2.0f32; 16])
            .expect("write recording");
        writer.finish().expect("finish recording");

        let mut options = Options::default();
        options.frames = Some(1);
        options.dataset = Some(std::env::temp_dir().join("scan-gui-depth-only-ds"));
        let config = SessionConfig {
            source: Source::Replay(path.clone()),
            options,
            preview_every: 1,
            playback_fps: 0.0,
            color_preview_views: 4,
        };

        let shared = Arc::new(Mutex::new(SessionView::default()));
        let control = Control::default();
        let (_commands, receiver) = mpsc::channel();

        let error = match execute(&config, &shared, &control, &receiver) {
            Ok(_) => panic!("a depth-only replay cannot feed a dataset"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("without colour"), "{error}");

        let _ = std::fs::remove_file(&path);
        if let Some(dataset) = &config.options.dataset {
            let _ = std::fs::remove_dir_all(dataset);
        }
    }

    #[test]
    fn finish_recording_closes_the_file_and_reports_the_count() {
        let path = std::env::temp_dir().join(format!(
            "scan-gui-finish-record-{}.k2df",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let mut writer =
            recording::FrameWriter::create(&path, synthetic::intrinsics(), 4, 4).expect("create");
        writer
            .write_frame(0.0, &vec![2.0f32; 16])
            .expect("write frame");

        let shared = Arc::new(Mutex::new(SessionView::default()));
        finish_recording(Some(writer), Some("test.k2df"), &shared);

        let recording = recording::read(&path).expect("read the finished recording");
        assert_eq!(recording.frames.len(), 1);
        assert!(
            shared
                .lock()
                .unwrap()
                .log
                .iter()
                .any(|line| line.contains("recorded 1 frames to test.k2df"))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pause_blocks_until_resume_and_stop_wins() {
        let control = Control::default();
        let shared = Arc::new(Mutex::new(SessionView::default()));
        let (commands, receiver) = mpsc::channel();

        // `pump` blocks while paused, so the pause/resume transition is checked
        // through `apply`; `pump` is then checked with a stop.
        shared.lock().unwrap().status = Status::Running;

        apply(Command::Pause, &control, &shared);
        assert!(control.paused.load(Ordering::Relaxed));
        assert_eq!(shared.lock().unwrap().status, Status::Paused);

        apply(Command::Resume, &control, &shared);
        assert!(!control.paused.load(Ordering::Relaxed));
        assert_eq!(shared.lock().unwrap().status, Status::Running);

        commands.send(Command::Stop).unwrap();
        assert!(!pump(&control, &receiver, &shared));
    }

    #[test]
    fn a_reset_command_marks_the_scan_for_discard() {
        let control = Control::default();
        let shared = Arc::new(Mutex::new(SessionView::default()));
        let (commands, receiver) = mpsc::channel();

        commands.send(Command::Reset).unwrap();
        assert!(!pump(&control, &receiver, &shared));
        assert!(control.discard.load(Ordering::Relaxed));
    }

    #[test]
    fn a_stopped_scan_is_kept_for_export_but_a_reset_is_not() {
        let config = synthetic_config(2);

        let stopped = Control::default();
        stopped.stop.store(true, Ordering::Relaxed);
        let state = SessionState::new(synthetic::intrinsics(), 1, 1, &config);
        assert!(
            finish_session(state, &stopped).expect("no error").is_some(),
            "a plain Stop must keep the scanner so Export can write it"
        );

        let discarded = Control::default();
        discarded.discard.store(true, Ordering::Relaxed);
        let state = SessionState::new(synthetic::intrinsics(), 1, 1, &config);
        assert!(
            finish_session(state, &discarded)
                .expect("no error")
                .is_none(),
            "Reset is the only path that discards"
        );
    }

    #[test]
    fn export_stops_the_scan_before_exiting() {
        let control = Control::default();
        let shared = Arc::new(Mutex::new(SessionView::default()));

        apply(Command::Export, &control, &shared);

        assert!(control.stop.load(Ordering::Relaxed));
        assert!(control.export.load(Ordering::Relaxed));
        assert!(!control.discard.load(Ordering::Relaxed));
    }

    #[test]
    fn finish_ends_the_worker_without_hanging() {
        // This is the close-the-window path. `Finish` must export and leave
        // the worker, or `SessionHandle::join` would wait forever on a scanner
        // held for a later export.
        let mut config = synthetic_config(12);
        let out = std::env::temp_dir().join(format!("scan-gui-finish-{}.ply", std::process::id()));
        config.options.out = Some(out.clone());

        let mut session = spawn(config);
        session.finish();
        session.join();

        assert!(session.finished());
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(out.with_extension("trajectory.ply"));
    }

    #[test]
    fn depth_colours_run_from_warm_to_cool() {
        let near = depth_color(0.5);
        let far = depth_color(4.5);
        assert!(near[0] > far[0], "near should be warmer: {near:?} {far:?}");
        assert!(near[2] < far[2], "far should be cooler: {near:?} {far:?}");
    }
}
