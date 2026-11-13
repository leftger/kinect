//! The scanner window: a full application around the live preview.
//!
//! `scan gui` opens a window with a source picker (live sensor, a `.k2df`
//! recording, or the synthetic room), the reconstruction parameters, transport
//! controls, a 3D viewport, live telemetry and an explicit export. `live
//! --viewer` opens the same window with the live source started immediately.
//!
//! # Shape
//!
//! * [`session`] owns the scan worker and the scanner. It is where the frames
//!   are tracked, fused and turned into a preview mesh.
//! * [`render`] is the software rasteriser and orbit camera.
//! * This module is only UI: it copies a snapshot out of the worker, draws it,
//!   and sends commands back.
//!
//! # Closing
//!
//! Closing the window is a request to finish the scan, not a way to lose it. If
//! a scanner is still held, the app asks the worker to stop and export before it
//! joins. Reset is the explicit way to throw a scan away.

mod render;
mod session;
mod splat;
mod trainer;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;
use geom::coloring::ColorMode;
use geom::export::MeshFormat;
use geom::tsdf::TsdfParams;
use nalgebra::Vector3;

use crate::{DEFAULT_FRAMES, Options};

use render::{Camera, PreviewScene, Renderer};
use session::{SessionConfig, SessionHandle, SessionSnapshot, Source, Status};
use splat::SplatCloud;
use trainer::{
    BRUSH_RELEASES_URL, BrushInstall, TrainerConfig, TrainerHandle, find_brush, progress_summary,
};

/// Which source the controls are set to. The worker gets the resolved [`Source`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceChoice {
    Live,
    Record,
    Replay,
    Synthetic,
}

impl SourceChoice {
    fn label(self) -> &'static str {
        match self {
            Self::Live => "Live sensor",
            Self::Record => "Record",
            Self::Replay => "Replay",
            Self::Synthetic => "Synthetic",
        }
    }
}

/// Check that `path` (or its nearest existing ancestor) can be written to.
///
/// The window starts a long scan before it writes the dataset or recording, so
/// a typo or a read-only mount should be caught at Start, not after the capture.
/// The probe file is removed immediately.
fn check_directory_writable(path: &Path) -> Result<(), String> {
    let mut probe = path.to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => probe = parent.to_path_buf(),
            _ => {
                probe = PathBuf::from(".");
                break;
            }
        }
    }

    if !probe.is_dir() {
        return Err(format!("{} is not a directory", probe.display()));
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    let test = probe.join(format!(".scan-write-{}-{stamp}", std::process::id()));
    match std::fs::File::create(&test) {
        Ok(_) => {
            let _ = std::fs::remove_file(&test);
            Ok(())
        }
        Err(error) => Err(format!("cannot write in {}: {error}", probe.display())),
    }
}

/// The editable parameters. These mirror the CLI options; `start` copies them
/// onto a fresh `Options` so the worker and the CLI share one configuration type.
struct UiConfig {
    frames: usize,
    voxel: f32,
    filters: bool,
    color: bool,
    color_mode: ColorMode,
    color_depth_tolerance: f32,
    loop_closure: bool,
    online_loops: bool,
    loop_worker: bool,
    incremental_rebuild: bool,
    fusion_filter: bool,
    tsdf_chunk_frames: usize,
    relocalize_global: bool,
    loop_global: bool,
    match_score: f32,
    submap_graph: bool,
    submap_max_frames: usize,
    submap_max_translation: f32,
    graph_worker: bool,
    frame_to_model: bool,
    submaps: bool,
    gpu: bool,
    unmirror: bool,
    level: bool,
    out: String,
    dataset: String,
    preview_every: usize,
    playback_fps: f32,
}

impl UiConfig {
    fn from_options(options: &Options) -> Self {
        Self {
            frames: options.frames.unwrap_or(DEFAULT_FRAMES),
            voxel: options.tsdf.voxel_size,
            filters: options.filters,
            color: options.color,
            color_mode: options.color_mode,
            color_depth_tolerance: options.color_depth_tolerance,
            loop_closure: options.loop_closure,
            online_loops: options.online_loops,
            loop_worker: options.loop_worker,
            incremental_rebuild: options.incremental_rebuild,
            fusion_filter: options.fusion_filter,
            tsdf_chunk_frames: options.tsdf_chunk_frames,
            relocalize_global: options.relocalize_global,
            loop_global: options.loop_global,
            match_score: options.match_score,
            submap_graph: options.submap_graph,
            submap_max_frames: options.submap_max_frames,
            submap_max_translation: options.submap_max_translation,
            graph_worker: options.graph_worker,
            frame_to_model: options.frame_to_model,
            submaps: options.submaps,
            gpu: options.gpu,
            unmirror: options.unmirror,
            level: options.level,
            out: options
                .out
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "scan.ply".to_string()),
            dataset: options
                .dataset
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            preview_every: options.preview_every,
            playback_fps: options.playback_fps,
        }
    }
}

/// The Splats panel's editable state.
struct TrainerUi {
    /// Blank means "use the Brush found on PATH".
    program: String,
    dataset: String,
    output: String,
    iterations: usize,
    /// Brush writes a complete model every this many steps; the viewport shows
    /// each one as it lands.
    preview_every: usize,
    /// Load each preview into the viewport automatically.
    auto_preview: bool,
    /// Also open Brush's own live viewer in a separate window.
    viewer: bool,
    /// Start training as soon as an export writes a dataset.
    auto_train: bool,
    allow_unmerged: bool,
}

impl Default for TrainerUi {
    fn default() -> Self {
        Self {
            program: String::new(),
            dataset: String::new(),
            output: "out/splat-export".to_string(),
            iterations: 30_000,
            // Brush's own default. Each preview is a complete model, so a
            // smaller interval costs disk and a viewport reload each time.
            preview_every: 5_000,
            auto_preview: true,
            viewer: false,
            auto_train: false,
            allow_unmerged: false,
        }
    }
}

/// A preview model being read on a worker thread. A trained model can hold
/// millions of splats, and parsing one on the UI thread would freeze the
/// window for as long as it takes.
struct PreviewLoad {
    iteration: usize,
    path: PathBuf,
    receiver: mpsc::Receiver<Result<SplatCloud, String>>,
}

/// Open the scanner window.
pub(crate) fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    let source = if options.synthetic {
        SourceChoice::Synthetic
    } else if options.input.is_some() {
        SourceChoice::Replay
    } else {
        SourceChoice::Live
    };

    let replay_path = options
        .input
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_default();

    let app = ScannerApp::new(options, source, replay_path);

    eframe::run_native(
        "kinect scanner",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1280.0, 820.0])
                .with_min_inner_size([900.0, 600.0]),
            ..Default::default()
        },
        Box::new(move |_cc| Ok(Box::new(app))),
    )
    .map_err(|error| format!("scanner window: {error}").into())
}

struct ScannerApp {
    /// The CLI defaults, so a setting this window does not expose is not lost.
    base: Options,
    config: UiConfig,
    source: SourceChoice,
    /// Replay input, or the `.k2df` path a Record session writes.
    replay_path: String,
    record_path: String,
    session: Option<SessionHandle>,
    snapshot: SessionSnapshot,
    trainer: Option<TrainerHandle>,
    trainer_config: TrainerUi,
    trainer_error: Option<String>,
    /// Brush as found on `PATH`, looked up at start and on demand.
    brush: Option<BrushInstall>,
    preview_load: Option<PreviewLoad>,
    /// The training step of the preview the viewport is showing.
    loaded_preview: Option<usize>,
    /// The first preview of a run frames the camera and switches to splats;
    /// later ones only swap the model, so the operator's view is not reset.
    first_preview_pending: bool,
    /// True while a scan is in progress, so the next export is recognised as
    /// new rather than an old one still on screen.
    auto_train_armed: bool,
    scene: Option<Arc<PreviewScene>>,
    scene_id: u64,
    splat: Option<SplatCloud>,
    splat_path: String,
    splat_error: Option<String>,
    show_splats: bool,
    splat_scale: f32,
    splat_min_opacity: f32,
    camera: Camera,
    renderer: Renderer,
    texture: Option<egui::TextureHandle>,
    dirty: bool,
    auto_fit: bool,
    last_preview_at: Option<Instant>,
    preview_fps: f32,
    form_error: Option<String>,
}

impl ScannerApp {
    fn new(options: &Options, source: SourceChoice, replay_path: String) -> Self {
        let mut app = Self {
            base: options.clone(),
            config: UiConfig::from_options(options),
            source,
            replay_path,
            record_path: "capture.k2df".to_string(),
            session: None,
            snapshot: SessionSnapshot::default(),
            trainer: None,
            trainer_config: TrainerUi::default(),
            trainer_error: None,
            brush: find_brush(),
            preview_load: None,
            loaded_preview: None,
            first_preview_pending: true,
            auto_train_armed: false,
            scene: None,
            scene_id: 0,
            splat: None,
            splat_path: String::new(),
            splat_error: None,
            show_splats: false,
            splat_scale: 1.0,
            splat_min_opacity: 0.0,
            camera: Camera::default(),
            renderer: Renderer::new(800, 600),
            texture: None,
            dirty: true,
            auto_fit: true,
            last_preview_at: None,
            preview_fps: 0.0,
            form_error: None,
        };

        if options.auto_start {
            app.start();
        }

        app
    }

    /// Copy the worker's view into the window and notice a new preview.
    fn poll_session(&mut self) {
        let mut finished = false;

        if let Some(session) = self.session.as_mut() {
            self.snapshot = session.snapshot();
            finished = session.finished();
        }

        if finished {
            if let Some(mut session) = self.session.take() {
                session.join();
            }
        }

        if self.trainer_config.dataset.is_empty() {
            if let Some(dataset) = &self.snapshot.dataset {
                self.trainer_config.dataset = dataset.dir.display().to_string();
            }
        }

        if let Some(scene) = &self.snapshot.scene {
            if self.snapshot.scene_id != self.scene_id {
                self.scene = Some(Arc::clone(scene));
                self.scene_id = self.snapshot.scene_id;
                self.dirty = true;

                if self.auto_fit && !self.show_splats {
                    // Wait for the first scene that actually has a surface.
                    // An early frame can be empty, and consuming the flag then
                    // would leave the camera pointing at the origin forever.
                    if let Some((min, max)) = scene.bounds {
                        self.camera.fit(min, max);
                        self.auto_fit = false;
                    }
                }

                let now = Instant::now();
                if let Some(last) = self.last_preview_at {
                    let seconds = now.duration_since(last).as_secs_f32();
                    if seconds > 1e-3 {
                        let instant = 1.0 / seconds;
                        self.preview_fps = if self.preview_fps > 0.0 {
                            self.preview_fps * 0.8 + instant * 0.2
                        } else {
                            instant
                        };
                    }
                }
                self.last_preview_at = Some(now);
            }
        } else if self.session.is_none() && self.snapshot.status == Status::Idle {
            // A reset has cleared the view. Drop the last picture too.
            if self.scene.is_some() {
                self.scene = None;
                self.scene_id = 0;
                self.dirty = true;
            }
        }
    }

    /// Build an `Options` from the controls and start a worker.
    fn start(&mut self) {
        self.form_error = None;

        let mut options = self.base.clone();
        options.frames = Some(self.config.frames.max(2));
        options.tsdf = TsdfParams {
            voxel_size: self.config.voxel,
            truncation: (self.config.voxel * 4.0).max(1e-4),
            ..options.tsdf
        };
        options.filters = self.config.filters;
        options.color = self.config.color;
        options.color_mode = self.config.color_mode;
        options.color_depth_tolerance = self.config.color_depth_tolerance;
        options.loop_closure = self.config.loop_closure;
        // The dependent flags imply loop closure in the CLI too; keep the
        // window honest so a checkbox cannot silently do nothing.
        options.online_loops = self.config.online_loops;
        options.loop_worker = self.config.loop_worker;
        options.incremental_rebuild = self.config.incremental_rebuild;
        options.fusion_filter = self.config.fusion_filter;
        options.tsdf_chunk_frames = self.config.tsdf_chunk_frames;
        options.relocalize_global = self.config.relocalize_global;
        options.loop_global = self.config.loop_global;
        options.match_score = self.config.match_score;
        options.submap_graph = self.config.submap_graph;
        options.submap_max_frames = self.config.submap_max_frames;
        options.submap_max_translation = self.config.submap_max_translation;
        options.graph_worker = self.config.graph_worker;
        if options.online_loops
            || options.loop_worker
            || options.incremental_rebuild
            || options.loop_global
            || options.match_score > 0.0
            || options.submap_graph
            || options.graph_worker
        {
            options.loop_closure = true;
        }
        if options.relocalize_global {
            options.relocalize = true;
        }
        if options.graph_worker {
            options.online_loops = true;
        }
        options.frame_to_model = self.config.frame_to_model;
        options.submaps = self.config.submaps;
        options.gpu = self.config.gpu;
        options.unmirror = self.config.unmirror;
        options.level = self.config.level;
        options.viewer = false;
        options.auto_start = false;
        options.synthetic = false;
        options.input = None;

        options.out = if self.config.out.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(self.config.out.trim()))
        };

        // Validate the extension now. `finish` would reject it after the scan
        // has been taken, which is a frustrating place to find a typo.
        let output = options
            .out
            .clone()
            .unwrap_or_else(|| PathBuf::from("scan.ply"));
        if let Err(error) = MeshFormat::from_path(&output) {
            self.form_error = Some(error.to_string());
            return;
        }

        let dataset = self.config.dataset.trim();
        let needs_colour = matches!(self.source, SourceChoice::Live | SourceChoice::Record);
        if !dataset.is_empty() && needs_colour && !self.config.color {
            // A live or recorded dataset is built from the colour views, so it
            // needs the colour stream. Replay and synthetic may already have it.
            self.form_error =
                Some("a dataset needs colour on a live or recorded scan; tick Colour".to_string());
            return;
        }
        options.dataset = if dataset.is_empty() {
            None
        } else {
            Some(PathBuf::from(dataset))
        };
        if let Some(dir) = &options.dataset {
            // The exporter creates the directory, but only after the scan has
            // run. Check the nearest writable ancestor now instead.
            if let Err(error) = check_directory_writable(dir) {
                self.form_error = Some(error);
                return;
            }
        }

        let source = match self.source {
            SourceChoice::Live => Source::Live,
            SourceChoice::Record => {
                let path = self.record_path.trim();
                if path.is_empty() {
                    self.form_error = Some("choose a .k2df path for the recording".to_string());
                    return;
                }
                let path = PathBuf::from(path);
                let parent = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."));
                if !parent.is_dir() {
                    self.form_error = Some(format!(
                        "recording directory {} does not exist",
                        parent.display()
                    ));
                    return;
                }
                if let Err(error) = check_directory_writable(parent) {
                    self.form_error = Some(error);
                    return;
                }
                Source::Record(path)
            }
            SourceChoice::Replay => {
                let path = self.replay_path.trim();
                if path.is_empty() {
                    self.form_error = Some("choose a .k2df recording first".to_string());
                    return;
                }
                let path = PathBuf::from(path);
                if !path.exists() {
                    self.form_error = Some(format!("{} does not exist", path.display()));
                    return;
                }
                Source::Replay(path)
            }
            SourceChoice::Synthetic => Source::Synthetic,
        };

        let config = SessionConfig {
            source,
            options,
            preview_every: self.config.preview_every.max(1),
            playback_fps: self.config.playback_fps,
            color_preview_views: 12,
        };

        self.session = Some(session::spawn(config));
        self.snapshot = SessionSnapshot::default();
        self.scene = None;
        self.scene_id = 0;
        self.dirty = true;
        self.auto_fit = true;
        self.preview_fps = 0.0;
        self.last_preview_at = None;
    }

    fn fit_view(&mut self) {
        if self.show_splats {
            if let Some(splat) = &self.splat {
                let (min, max) = splat.bounds;
                self.camera.fit(min, max);
                self.dirty = true;
                return;
            }
        }

        let Some(scene) = &self.scene else {
            return;
        };
        let Some((min, max)) = scene.bounds else {
            return;
        };
        self.camera.fit(min, max);
        self.dirty = true;
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let status = self.snapshot.status;
        let editable = self.session.is_none();

        ui.heading("Source");
        ui.add_enabled_ui(editable, |ui| {
            ui.horizontal_wrapped(|ui| {
                for choice in [
                    SourceChoice::Live,
                    SourceChoice::Record,
                    SourceChoice::Replay,
                    SourceChoice::Synthetic,
                ] {
                    ui.selectable_value(&mut self.source, choice, choice.label());
                }
            });

            match self.source {
                SourceChoice::Live => {
                    ui.small("A Kinect v2 on a USB 3 port. Nothing else is needed.");
                }
                SourceChoice::Record => {
                    ui.small(
                        "Capture while reconstructing, and write a .k2df for later replays. \
                         Tick Colour to store the colour views a dataset needs.",
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut self.record_path).hint_text("capture.k2df"),
                    );
                }
                SourceChoice::Replay => {
                    ui.small(
                        "A .k2df recording. Make one with the Record source or `scan record`.",
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut self.replay_path)
                            .hint_text("/path/to/capture.k2df"),
                    );
                }
                SourceChoice::Synthetic => {
                    ui.small("The analytic room. Runs with no sensor attached.");
                }
            }
        });

        ui.separator();
        ui.heading("Reconstruction");
        ui.add_enabled_ui(editable, |ui| {
            egui::Grid::new("params")
                .num_columns(2)
                .spacing([10.0, 5.0])
                .show(ui, |ui| {
                    ui.label("Frames");
                    ui.add(
                        egui::DragValue::new(&mut self.config.frames)
                            .range(2..=100_000)
                            .speed(1.0),
                    );
                    ui.end_row();

                    ui.label("Voxel (m)");
                    ui.add(
                        egui::DragValue::new(&mut self.config.voxel)
                            .range(0.002..=0.1)
                            .speed(0.001)
                            .max_decimals(3),
                    );
                    ui.end_row();

                    ui.label("Colour");
                    match self.source {
                        SourceChoice::Live | SourceChoice::Record => {
                            ui.checkbox(&mut self.config.color, "capture and paint");
                        }
                        SourceChoice::Replay => {
                            ui.small("from the recording");
                        }
                        SourceChoice::Synthetic => {
                            ui.small("always on");
                        }
                    }
                    ui.end_row();

                    ui.label("Colour mode");
                    egui::ComboBox::from_id_salt("color-mode")
                        .selected_text(color_mode_label(self.config.color_mode))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.config.color_mode,
                                ColorMode::Best,
                                "best",
                            );
                            ui.selectable_value(
                                &mut self.config.color_mode,
                                ColorMode::Blend,
                                "blend",
                            );
                            ui.selectable_value(
                                &mut self.config.color_mode,
                                ColorMode::Average,
                                "average",
                            );
                        });
                    ui.end_row();

                    ui.label("Colour tolerance");
                    ui.add(
                        egui::DragValue::new(&mut self.config.color_depth_tolerance)
                            .range(0.001..=0.2)
                            .speed(0.001)
                            .max_decimals(3),
                    );
                    ui.end_row();

                    ui.label("Filters");
                    ui.checkbox(&mut self.config.filters, "decode filters");
                    ui.end_row();

                    ui.label("Loop closure");
                    ui.checkbox(&mut self.config.loop_closure, "correct revisits");
                    ui.end_row();

                    ui.label("Online loops");
                    ui.checkbox(&mut self.config.online_loops, "correct during the scan");
                    ui.end_row();

                    ui.label("Loop worker");
                    ui.checkbox(&mut self.config.loop_worker, "verify off the capture thread");
                    ui.end_row();

                    ui.label("Incremental");
                    ui.checkbox(
                        &mut self.config.incremental_rebuild,
                        "move submaps, do not re-fuse",
                    );
                    ui.end_row();

                    ui.label("Fusion filter");
                    ui.checkbox(&mut self.config.fusion_filter, "skip near-duplicate frames");
                    ui.end_row();

                    ui.label("TSDF chunk");
                    ui.add(
                        egui::DragValue::new(&mut self.config.tsdf_chunk_frames)
                            .range(0..=500)
                            .speed(1.0)
                            .suffix(" frames"),
                    );
                    ui.end_row();

                    ui.label("Global relocalize");
                    ui.checkbox(
                        &mut self.config.relocalize_global,
                        "branch-and-bound fallback",
                    );
                    ui.end_row();

                    ui.label("Global loops");
                    ui.checkbox(&mut self.config.loop_global, "branch-and-bound fallback");
                    ui.end_row();

                    ui.label("Match score");
                    ui.add(
                        egui::Slider::new(&mut self.config.match_score, 0.0..=1.0)
                            .text("min likelihood"),
                    );
                    ui.end_row();

                    ui.label("Submap graph");
                    ui.checkbox(&mut self.config.submap_graph, "submap nodes, rigid");
                    ui.end_row();

                    ui.label("Graph worker");
                    ui.checkbox(
                        &mut self.config.graph_worker,
                        "solve off the capture thread",
                    );
                    ui.end_row();

                    ui.label("Frame to model");
                    ui.checkbox(&mut self.config.frame_to_model, "slow, less drift");
                    ui.end_row();

                    ui.label("Submaps");
                    ui.checkbox(&mut self.config.submaps, "restart when tracking is lost");
                    ui.end_row();

                    ui.label("Submap frames");
                    ui.add(
                        egui::DragValue::new(&mut self.config.submap_max_frames)
                            .range(0..=500)
                            .speed(1.0),
                    );
                    ui.end_row();

                    ui.label("Submap metres");
                    ui.add(
                        egui::DragValue::new(&mut self.config.submap_max_translation)
                            .range(0.0..=20.0)
                            .speed(0.1),
                    );
                    ui.end_row();

                    ui.label("GPU decode");
                    ui.checkbox(&mut self.config.gpu, "wgpu feature");
                    ui.end_row();

                    ui.label("Mirror");
                    ui.checkbox(&mut self.config.unmirror, "un-mirror X");
                    ui.end_row();

                    ui.label("Level");
                    ui.checkbox(&mut self.config.level, "level on export")
                        .on_hover_text(
                            "Rotate the exported mesh and trajectory so the floor or \
                             ceiling is horizontal. Needs a large flat surface within \
                             25 degrees of level; otherwise the model is left as scanned. \
                             The live preview is not levelled.",
                        );
                    ui.end_row();

                    ui.label("Preview every");
                    ui.add(
                        egui::DragValue::new(&mut self.config.preview_every)
                            .range(1..=60)
                            .speed(1.0),
                    );
                    ui.end_row();

                    ui.label("Playback fps");
                    ui.add(
                        egui::DragValue::new(&mut self.config.playback_fps)
                            .range(0.0..=60.0)
                            .speed(0.5)
                            .max_decimals(1),
                    );
                    ui.end_row();
                });

            ui.small("Voxel size drives memory and fusion cost: halving it needs about eight times the memory.");
            if self.config.color_mode != ColorMode::Best {
                ui.small("Colour modes other than best affect per-vertex PLY colour.");
            }
        });

        ui.separator();
        ui.heading("Output");
        ui.add_enabled_ui(editable, |ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.config.out)
                    .hint_text("scan.ply, room.obj, room.glb ..."),
            );
            ui.small("The extension selects the format: .ply, .obj, .gltf or .glb.");
            ui.add(
                egui::TextEdit::singleline(&mut self.config.dataset)
                    .hint_text("optional dataset directory"),
            );
            ui.small(
                "A Nerfstudio/Brush dataset for Gaussian splatting. Needs colour views; \
                 leave empty to skip it.",
            );
        });

        ui.separator();
        ui.heading("Session");

        ui.horizontal(|ui| match self.session.as_ref() {
            None => {
                if ui.button("Start scan").clicked() {
                    self.start();
                }
            }
            Some(session) => {
                if status == Status::Paused {
                    if ui.button("Resume").clicked() {
                        session.resume();
                    }
                } else if ui
                    .add_enabled(status == Status::Running, egui::Button::new("Pause"))
                    .clicked()
                {
                    session.pause();
                }

                if ui
                    .add_enabled(
                        matches!(status, Status::Starting | Status::Running | Status::Paused),
                        egui::Button::new("Stop"),
                    )
                    .clicked()
                {
                    session.stop();
                }
            }
        });

        ui.horizontal(|ui| {
            let can_export = self.session.is_some()
                && matches!(
                    status,
                    Status::Running | Status::Paused | Status::Stopped | Status::Exported
                );
            if ui
                .add_enabled(can_export, egui::Button::new("Export"))
                .clicked()
            {
                if let Some(session) = &self.session {
                    session.export();
                }
            }

            if ui
                .add_enabled(self.session.is_some(), egui::Button::new("Reset"))
                .clicked()
            {
                if let Some(session) = &self.session {
                    session.reset();
                }
            }
        });

        ui.small("Export while scanning stops first. Reset discards the scan.");
        ui.small("Closing the window exports an unexported scan.");

        if let Some(error) = &self.form_error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }

        ui.separator();
        self.splats_panel(ui);
    }

    fn splats_panel(&mut self, ui: &mut egui::Ui) {
        let view = self
            .trainer
            .as_ref()
            .map(|trainer| trainer.snapshot())
            .unwrap_or_default();
        let last_dataset = self.snapshot.dataset.as_ref().map(|d| d.dir.clone());
        let selected_dataset = self.trainer_config.dataset.trim().to_string();
        let warnings = self
            .snapshot
            .dataset
            .as_ref()
            .filter(|dataset| dataset.dir.display().to_string() == selected_dataset)
            .map(|dataset| dataset.warnings.clone())
            .unwrap_or_default();

        ui.heading("Splats");
        ui.small(
            "Train a Gaussian-splat model from the exported dataset with Brush, and \
             watch it sharpen in the viewport as training runs. Brush is a child \
             process, separate from the scan worker.",
        );

        let mut redetect = false;
        ui.horizontal_wrapped(|ui| {
            match &self.brush {
                Some(brush) => {
                    let version = if brush.version.is_empty() {
                        "version unknown"
                    } else {
                        brush.version.as_str()
                    };
                    ui.colored_label(
                        egui::Color32::from_rgb(120, 200, 140),
                        format!("Brush found: {version}"),
                    );
                    ui.small(brush.path.display().to_string());
                }
                None => {
                    ui.colored_label(
                        egui::Color32::from_rgb(240, 200, 90),
                        "Brush not found on PATH.",
                    );
                    ui.hyperlink_to("Download a release", BRUSH_RELEASES_URL);
                    ui.small("and put it on PATH as `brush`, or type its full path below.");
                }
            }
            if ui.small_button("Re-detect").clicked() {
                redetect = true;
            }
        });
        if redetect {
            self.brush = find_brush();
        }

        let mut use_last = false;
        let mut load_splat = false;
        let mut use_trainer_output = false;
        egui::Grid::new("trainer-fields")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Trainer");
                ui.add(
                    egui::TextEdit::singleline(&mut self.trainer_config.program)
                        .hint_text("blank = the Brush found on PATH"),
                );
                ui.end_row();

                ui.label("Dataset");
                ui.vertical(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.trainer_config.dataset)
                            .hint_text("out/ds-synth"),
                    );
                    if last_dataset.is_some() && ui.small_button("use last export").clicked() {
                        use_last = true;
                    }
                });
                ui.end_row();

                ui.label("Output");
                ui.add(
                    egui::TextEdit::singleline(&mut self.trainer_config.output)
                        .hint_text("out/splat-export"),
                );
                ui.end_row();

                ui.label("Steps");
                ui.add(
                    egui::DragValue::new(&mut self.trainer_config.iterations)
                        .range(1..=1_000_000)
                        .speed(100.0),
                );
                ui.end_row();

                ui.label("Preview every");
                ui.add(
                    egui::DragValue::new(&mut self.trainer_config.preview_every)
                        .range(1..=1_000_000)
                        .speed(50.0)
                        .suffix(" steps"),
                )
                .on_hover_text(
                    "Brush writes a complete model this often. Each one can be shown in \
                     the viewport while training continues.",
                );
                ui.end_row();
            });

        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.trainer_config.auto_preview, "Live preview")
                .on_hover_text(
                    "Load each preview model into the viewport as Brush writes it.",
                );
            ui.checkbox(&mut self.trainer_config.viewer, "Brush window")
                .on_hover_text(
                    "Also open Brush's own live viewer in a separate window. Takes \
                     effect on the next Train.",
                );
            ui.checkbox(&mut self.trainer_config.auto_train, "Train after export")
                .on_hover_text(
                    "Start training as soon as a scan's dataset has been exported. \
                     Set a dataset directory in Output first.",
                );
        });

        ui.horizontal(|ui| {
            ui.label("Splat file");
            ui.add(
                egui::TextEdit::singleline(&mut self.splat_path)
                    .hint_text("export.ply from the trainer"),
            );
            if ui.button("Load").clicked() {
                load_splat = true;
            }
            if view.output.is_some() && ui.small_button("use trainer output").clicked() {
                use_trainer_output = true;
            }
        });
        if use_trainer_output {
            if let Some(output) = &view.output {
                self.splat_path = output.display().to_string();
                load_splat = true;
            }
        }
        if load_splat {
            self.load_splat();
        }
        if let Some(error) = &self.splat_error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        let splat_summary = self.splat.as_ref().map(|splat| (splat.len(), splat.bounds));
        if let Some((count, bounds)) = splat_summary {
            ui.horizontal(|ui| {
                ui.label(format!("{count} splats"));
                let before = self.show_splats;
                ui.checkbox(&mut self.show_splats, "Show splats");
                if before != self.show_splats {
                    self.dirty = true;
                }
                if ui.button("Fit splats").clicked() {
                    self.camera.fit(bounds.0, bounds.1);
                    self.dirty = true;
                }
                if ui.button("First-person").clicked() {
                    self.camera
                        .look_from(Vector3::zeros(), Vector3::new(0.0, 0.0, 2.0));
                    self.dirty = true;
                }
            });
            let before_scale = self.splat_scale;
            ui.add(
                egui::Slider::new(&mut self.splat_scale, 0.05..=2.0)
                    .logarithmic(true)
                    .text("splat size"),
            );
            if before_scale != self.splat_scale {
                self.dirty = true;
            }
            let before_opacity = self.splat_min_opacity;
            ui.add(egui::Slider::new(&mut self.splat_min_opacity, 0.0..=0.8).text("min opacity"));
            if before_opacity != self.splat_min_opacity {
                self.dirty = true;
            }
        }

        if use_last {
            if let Some(dir) = last_dataset {
                self.trainer_config.dataset = dir.display().to_string();
                self.trainer_error = None;
            }
        }

        if !warnings.is_empty() {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                "This exported dataset is not trainable:",
            );
            for warning in &warnings {
                ui.colored_label(
                    egui::Color32::from_rgb(240, 200, 90),
                    format!("  {warning}"),
                );
            }
            ui.checkbox(&mut self.trainer_config.allow_unmerged, "Train anyway");
        }

        let running = view.status.active();
        let mut start = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            if running {
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            } else if ui.button("Train").clicked() {
                start = true;
            }
            ui.label(format!("{}: {}", view.status.label(), view.message));
            if running {
                if let Some(started) = view.started {
                    ui.label(format!("{:.1} s", started.elapsed().as_secs_f32()));
                }
            }
        });
        if running || view.iteration > 0 {
            let elapsed = view
                .started
                .map(|started| view.finished.unwrap_or_else(Instant::now).duration_since(started))
                .unwrap_or_default();
            let text = progress_summary(view.iteration, view.total, view.splats, elapsed, running);
            ui.add(
                egui::ProgressBar::new(view.fraction().unwrap_or(0.0))
                    .animate(running)
                    .text(text),
            );

            let preview_note = match (self.loaded_preview, view.previews.last()) {
                (Some(shown), _) => {
                    let newer = if self.preview_load.is_some() {
                        "; loading a newer one"
                    } else {
                        ""
                    };
                    format!("viewport: step-{shown} preview{newer}")
                }
                (None, Some(latest)) if self.preview_load.is_some() => {
                    format!("loading the step-{} preview", latest.iteration)
                }
                (None, _) if running => format!(
                    "waiting for the first preview, written every {} steps",
                    self.trainer_config
                        .preview_every
                        .clamp(1, view.total.max(1))
                ),
                _ => String::new(),
            };
            if !preview_note.is_empty() {
                ui.small(preview_note);
            }
        }
        if let Some(output) = &view.output {
            ui.monospace(output.display().to_string());
        }
        if let Some(error) = &self.trainer_error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        if !view.log.is_empty() {
            egui::ScrollArea::vertical()
                .max_height(110.0)
                .show(ui, |ui| {
                    for line in &view.log {
                        ui.monospace(line);
                    }
                });
        }

        if cancel {
            if let Some(trainer) = &self.trainer {
                trainer.cancel();
            }
        }
        if start {
            self.start_trainer(&warnings);
        }
    }

    fn start_trainer(&mut self, warnings: &[String]) {
        self.trainer_error = None;
        if !warnings.is_empty() && !self.trainer_config.allow_unmerged {
            self.trainer_error = Some(
                "the exported dataset is not trainable; tick Train anyway to override".to_string(),
            );
            return;
        }

        let typed = self.trainer_config.program.trim().to_string();
        let dataset = self.trainer_config.dataset.trim().to_string();
        let output = self.trainer_config.output.trim().to_string();
        let program = if typed.is_empty() {
            match &self.brush {
                Some(brush) => brush.path.display().to_string(),
                None => {
                    self.trainer_error = Some(format!(
                        "Brush was not found on PATH. Install it from {BRUSH_RELEASES_URL} \
                         and put it on PATH as `brush`, or type its path in the Trainer field"
                    ));
                    return;
                }
            }
        } else {
            typed
        };
        if dataset.is_empty() {
            self.trainer_error = Some("choose a dataset directory".to_string());
            return;
        }
        if output.is_empty() {
            self.trainer_error = Some("choose an output directory".to_string());
            return;
        }

        // Brush resolves a relative `--export-path` against the dataset's
        // parent, which is surprising for a field the operator typed in the
        // window. Make it absolute so the reported path is where the file lands.
        let output = PathBuf::from(output);
        let output = if output.is_absolute() {
            output
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(&output))
                .unwrap_or(output)
        };

        let config = TrainerConfig {
            program,
            dataset: PathBuf::from(dataset),
            output,
            iterations: self.trainer_config.iterations,
            preview_every: self.trainer_config.preview_every,
            viewer: self.trainer_config.viewer,
        };
        match TrainerHandle::start(config) {
            Ok(handle) => {
                self.trainer = Some(handle);
                self.trainer_error = None;
                // A new run: its first preview should frame the camera again,
                // and an old run's preview must not count as already shown.
                self.loaded_preview = None;
                self.preview_load = None;
                self.first_preview_pending = true;
            }
            Err(error) => self.trainer_error = Some(error),
        }
    }

    /// Follow the trainer: start a run after an export when asked to, and show
    /// each preview model Brush writes.
    fn poll_trainer(&mut self) {
        self.maybe_auto_train();

        // Collect a model a worker thread has finished reading.
        let finished = self.preview_load.as_ref().map(|load| {
            (load.iteration, load.path.clone(), load.receiver.try_recv())
        });
        match finished {
            Some((iteration, path, Ok(result))) => {
                self.preview_load = None;
                match result {
                    Ok(cloud) => self.show_preview(iteration, &path, cloud),
                    Err(error) => {
                        // Give up on this one: the next preview, or the final
                        // model, will be tried on its own.
                        self.splat_error = Some(error);
                        self.loaded_preview = Some(iteration);
                    }
                }
            }
            Some((iteration, _, Err(mpsc::TryRecvError::Disconnected))) => {
                self.preview_load = None;
                self.loaded_preview = Some(iteration);
            }
            Some((_, _, Err(mpsc::TryRecvError::Empty))) | None => {}
        }

        if !self.trainer_config.auto_preview || self.preview_load.is_some() {
            return;
        }
        let Some(latest) = self
            .trainer
            .as_ref()
            .and_then(|trainer| trainer.snapshot().previews.last().cloned())
        else {
            return;
        };
        if self.loaded_preview.is_some_and(|shown| shown >= latest.iteration) {
            return;
        }

        let (sender, receiver) = mpsc::channel();
        let path = latest.path.clone();
        std::thread::spawn(move || {
            let _ = sender.send(SplatCloud::load(&path));
        });
        self.preview_load = Some(PreviewLoad {
            iteration: latest.iteration,
            path: latest.path,
            receiver,
        });
    }

    /// Start training the moment an export lands, if asked to. The decision is
    /// made once, at the transition into Exported, so ticking the box later
    /// does not retrain an export that was already on screen.
    fn maybe_auto_train(&mut self) {
        if self.snapshot.status != Status::Exported {
            self.auto_train_armed = true;
            return;
        }
        let fire = self.auto_train_armed && self.trainer_config.auto_train;
        self.auto_train_armed = false;
        if !fire {
            return;
        }

        let Some(dataset) = self.snapshot.dataset.clone() else {
            self.trainer_error = Some(
                "Train after export is on, but this export wrote no dataset; set a dataset \
                 directory in Output"
                    .to_string(),
            );
            return;
        };
        if self
            .trainer
            .as_ref()
            .is_some_and(|trainer| trainer.snapshot().status.active())
        {
            return;
        }
        self.trainer_config.dataset = dataset.dir.display().to_string();
        self.start_trainer(&dataset.warnings);
    }

    fn show_preview(&mut self, iteration: usize, path: &Path, cloud: SplatCloud) {
        if self.first_preview_pending {
            self.frame_splats(cloud.bounds);
            self.show_splats = true;
            self.first_preview_pending = false;
        }
        self.splat_path = path.display().to_string();
        self.splat_error = None;
        self.splat = Some(cloud);
        self.loaded_preview = Some(iteration);
        self.dirty = true;
    }

    /// Point the camera at a splat model: from inside it when it surrounds the
    /// origin (a room scan), otherwise framed from outside.
    fn frame_splats(&mut self, bounds: (Vector3<f32>, Vector3<f32>)) {
        let (min, max) = bounds;
        let contains_origin = min
            .iter()
            .zip(max.iter())
            .all(|(min, max)| *min <= 0.0 && *max >= 0.0);
        if contains_origin {
            self.camera
                .look_from(Vector3::zeros(), Vector3::new(0.0, 0.0, 2.0));
        } else {
            self.camera.fit(min, max);
        }
    }

    fn load_splat(&mut self) {
        self.splat_error = None;
        let path = self.splat_path.trim();
        if path.is_empty() {
            self.splat_error = Some("choose a splat .ply file".to_string());
            return;
        }

        match SplatCloud::load(Path::new(path)) {
            Ok(cloud) => {
                self.frame_splats(cloud.bounds);
                self.splat = Some(cloud);
                self.show_splats = true;
                self.dirty = true;
            }
            Err(error) => self.splat_error = Some(error),
        }
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        // The toolbar has to be laid out before the image claims all the
        // remaining space, or it ends up painted outside the panel.
        ui.horizontal(|ui| {
            if ui.button("Fit view").clicked() {
                self.fit_view();
            }
            if ui.button("Reset view").clicked() {
                self.camera = Camera::default();
                self.dirty = true;
            }
            if self.splat.is_some() {
                ui.separator();
                let before = self.show_splats;
                ui.selectable_value(&mut self.show_splats, false, "Mesh");
                ui.selectable_value(&mut self.show_splats, true, "Splats");
                if before != self.show_splats {
                    self.dirty = true;
                }
            }
            if self.show_splats {
                if let Some(splat) = &self.splat {
                    ui.separator();
                    match self.loaded_preview {
                        Some(step) => ui.small(format!(
                            "{} splats, training step {step}",
                            splat.len()
                        )),
                        None => ui.small(format!("{} splats", splat.len())),
                    };
                }
            } else if let Some(scene) = &self.scene {
                ui.separator();
                ui.small(format!(
                    "{} triangles, {} points",
                    scene.mesh.triangle_count(),
                    scene.points.len()
                ));
            }
        });

        let available = ui.available_size();
        if available.x < 4.0 || available.y < 4.0 {
            return;
        }

        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.camera.orbit(delta.x, delta.y);
            self.dirty = true;
        }
        if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let delta = response.drag_delta();
            self.camera.pan(delta.x, delta.y, rect.height());
            self.dirty = true;
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                self.camera.zoom(scroll);
                self.dirty = true;
            }
        }
        if response.double_clicked() {
            self.fit_view();
        }

        let width = (rect.width().round() as usize).clamp(1, 1600);
        let height = (rect.height().round() as usize).clamp(1, 1200);
        if self.renderer.width() != width || self.renderer.height() != height {
            self.renderer = Renderer::new(width, height);
            self.dirty = true;
        }

        if self.dirty {
            let image = if self.show_splats {
                if let Some(splat) = &self.splat {
                    // A million splats at full viewport resolution costs seconds
                    // per camera change. Halve the raster size for big clouds and
                    // let egui scale the texture; the model itself is the limit.
                    let (render_width, render_height) = if splat.len() > 250_000 {
                        ((width / 2).max(1), (height / 2).max(1))
                    } else {
                        (width, height)
                    };
                    let mut renderer = Renderer::new(render_width, render_height);
                    let rgba = renderer.render_splats(
                        splat,
                        &self.camera,
                        self.splat_scale,
                        self.splat_min_opacity,
                    );
                    egui::ColorImage::from_rgba_unmultiplied([render_width, render_height], rgba)
                } else {
                    let rgba = self.renderer.render(&PreviewScene::empty(), &self.camera);
                    egui::ColorImage::from_rgba_unmultiplied([width, height], rgba)
                }
            } else if let Some(scene) = &self.scene {
                let rgba = self.renderer.render(scene, &self.camera);
                egui::ColorImage::from_rgba_unmultiplied([width, height], rgba)
            } else {
                let rgba = self.renderer.render(&PreviewScene::empty(), &self.camera);
                egui::ColorImage::from_rgba_unmultiplied([width, height], rgba)
            };

            self.texture = Some(ui.ctx().load_texture(
                "scan-preview",
                image,
                egui::TextureOptions::LINEAR,
            ));
            self.dirty = false;
        }

        let painter = ui.painter_at(rect);
        if let Some(texture) = &self.texture {
            painter.image(
                texture.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }

        let hint = "drag: orbit   right-drag: pan   scroll: zoom   double-click: fit";
        painter.text(
            rect.left_top() + egui::vec2(10.0, 8.0),
            egui::Align2::LEFT_TOP,
            hint,
            egui::FontId::proportional(12.0),
            egui::Color32::from_gray(200),
        );

        if self.scene.is_none() && !self.snapshot.status.is_active() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "no preview yet - start a scan",
                egui::FontId::proportional(16.0),
                egui::Color32::from_gray(190),
            );
        } else if self.scene.is_none() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "waiting for the first frame ...",
                egui::FontId::proportional(16.0),
                egui::Color32::from_gray(190),
            );
        }
    }

    fn telemetry(&self, ui: &mut egui::Ui) {
        let snapshot = &self.snapshot;

        ui.horizontal_wrapped(|ui| {
            stat(
                ui,
                "frames",
                format!(
                    "{} / {}",
                    snapshot.frames_done,
                    snapshot.total.max(snapshot.frames_done)
                ),
            );
            stat(ui, "fused", snapshot.fused.to_string());
            stat(ui, "rejected", snapshot.rejected.to_string());
            stat(ui, "submaps", snapshot.submaps.to_string());
            let loop_text = if snapshot.loops_dropped > 0 {
                format!("{} ({} dropped)", snapshot.loops, snapshot.loops_dropped)
            } else {
                snapshot.loops.to_string()
            };
            stat(ui, "loops", loop_text);
            stat(ui, "loop ICP", snapshot.loop_attempts.to_string());
            stat(ui, "loop pts", snapshot.loop_points.to_string());
            stat(ui, "rehydrated", snapshot.loop_rehydrations.to_string());
            stat(
                ui,
                "online fixes",
                if snapshot.online_updates > 0 {
                    format!(
                        "{} ({:.2} m)",
                        snapshot.online_updates, snapshot.last_online_correction
                    )
                } else {
                    "0".to_string()
                },
            );
            if snapshot.loop_pending > 0 {
                stat(ui, "loop pending", snapshot.loop_pending.to_string());
            }
            if snapshot.fusion_skipped > 0 {
                stat(ui, "filtered", snapshot.fusion_skipped.to_string());
            }
            if snapshot.chunks > 0 {
                stat(ui, "chunks", snapshot.chunks.to_string());
            }
            if snapshot.graph_updates > 0 {
                stat(ui, "graph fixes", snapshot.graph_updates.to_string());
            }
            if snapshot.graph_pending {
                stat(ui, "graph pending", "1".to_string());
            }
            stat(
                ui,
                "position",
                format!(
                    "[{:+.2} {:+.2} {:+.2}] m",
                    snapshot.position[0], snapshot.position[1], snapshot.position[2]
                ),
            );
            stat(ui, "blocks", snapshot.blocks.to_string());
            stat(
                ui,
                "TSDF",
                format!("{:.1} MB", snapshot.allocated_bytes as f64 / 1e6),
            );
            stat(
                ui,
                "tracking",
                format!("{:.0} ms/frame", snapshot.tracking_ms),
            );
            stat(ui, "fusion", format!("{:.0} ms/frame", snapshot.fusion_ms));
            stat(ui, "submap", format!("{:.0} ms/frame", snapshot.submap_ms));
            if self.preview_fps > 0.1 {
                stat(ui, "preview", format!("{:.1} fps", self.preview_fps));
            }
        });

        if let Some(dataset) = &snapshot.dataset {
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("dataset").weak().small());
                ui.monospace(dataset.dir.display().to_string());
                ui.separator();
                ui.label(format!("{} views", dataset.views));
                ui.label(format!("{} initial points", dataset.points));
                if dataset.skipped > 0 {
                    ui.colored_label(
                        egui::Color32::from_rgb(240, 200, 90),
                        format!("{} views skipped", dataset.skipped),
                    );
                }
                if !dataset.warnings.is_empty() {
                    ui.colored_label(egui::Color32::LIGHT_RED, "NOT trainable");
                    for warning in &dataset.warnings {
                        ui.colored_label(
                            egui::Color32::from_rgb(240, 200, 90),
                            format!("warning: {warning}"),
                        );
                    }
                }
            });
        }

        if let Some((vertices, triangles)) = snapshot.mesh {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("exported mesh").weak().small());
                ui.monospace(format!("{vertices} vertices, {triangles} triangles"));
            });
        }

        if let Some(error) = &snapshot.error {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("message").weak());
            ui.label(&snapshot.message);
        });

        egui::CollapsingHeader::new("log").show(ui, |ui| {
            egui::ScrollArea::vertical()
                .max_height(110.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &snapshot.log {
                        ui.monospace(line);
                    }
                });
        });
    }
}

impl eframe::App for ScannerApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_session();
        self.poll_trainer();

        let interval = if self.snapshot.status.is_active() {
            Duration::from_millis(33)
        } else {
            Duration::from_millis(150)
        };
        ctx.request_repaint_after(interval);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top-bar").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("kinect scanner");
                ui.separator();
                ui.colored_label(
                    status_color(self.snapshot.status),
                    format!("\u{25cf} {}", self.snapshot.status.label()),
                );

                if self.snapshot.total > 0 {
                    let progress = (self.snapshot.frames_done as f32 / self.snapshot.total as f32)
                        .clamp(0.0, 1.0);
                    ui.add(
                        egui::ProgressBar::new(progress)
                            .desired_width(180.0)
                            .text(format!(
                                "{} / {}",
                                self.snapshot.frames_done, self.snapshot.total
                            )),
                    );
                } else if self.snapshot.frames_done > 0 {
                    ui.label(format!("{} frames", self.snapshot.frames_done));
                }

                if self.snapshot.elapsed > 0.0 {
                    ui.label(format!("{:.1} s", self.snapshot.elapsed));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(output) = &self.snapshot.output {
                        ui.label(format!("saved: {}", output.display()));
                    }
                });
            });
            ui.add_space(4.0);
        });

        egui::Panel::bottom("telemetry")
            .resizable(true)
            .default_size(165.0)
            .size_range(90.0..=320.0)
            .show(ui, |ui| {
                self.telemetry(ui);
            });

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(330.0)
            .size_range(280.0..=470.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.controls(ui);
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            self.viewport(ui);
        });
    }
}

impl Drop for ScannerApp {
    fn drop(&mut self) {
        if let Some(mut trainer) = self.trainer.take() {
            trainer.cancel();
            if !trainer.join_timeout(Duration::from_secs(5)) {
                eprintln!(
                    "scan gui: the trainer did not stop within 5 s; leaving it to process exit"
                );
            }
        }

        let Some(mut session) = self.session.take() else {
            return;
        };

        let snapshot = session.snapshot();

        if snapshot.status == Status::Error || snapshot.status == Status::Idle {
            // Nothing to export. The worker may already have left its loop, in
            // which case the reset simply fails to send.
            session.reset();
        } else if snapshot.output.is_none() && snapshot.fused > 0 {
            // Closing the window is a request to finish, not a way to lose the
            // scan. `finish` exports and leaves the worker; `stop` alone would
            // leave it holding the scanner, and `join` would wait forever.
            session.finish();
        } else {
            session.reset();
        }

        if !session.join_timeout(Duration::from_secs(5)) {
            eprintln!(
                "scan gui: the scan worker did not stop within 5 s; leaving it to process exit"
            );
        }
    }
}

fn stat(ui: &mut egui::Ui, label: &str, value: impl Into<String>) {
    ui.label(egui::RichText::new(label).weak().small());
    ui.label(egui::RichText::new(value.into()).monospace());
    ui.separator();
}

fn color_mode_label(mode: ColorMode) -> &'static str {
    match mode {
        ColorMode::Best => "best",
        ColorMode::Blend => "blend",
        ColorMode::Average => "average",
    }
}

fn status_color(status: Status) -> egui::Color32 {
    match status {
        Status::Running => egui::Color32::from_rgb(96, 220, 130),
        Status::Paused | Status::Starting | Status::Exporting => {
            egui::Color32::from_rgb(240, 200, 90)
        }
        Status::Stopped => egui::Color32::from_rgb(120, 180, 250),
        Status::Exported => egui::Color32::from_rgb(96, 220, 130),
        Status::Error => egui::Color32::from_rgb(250, 120, 110),
        Status::Idle => egui::Color32::from_gray(170),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_write_probe_accepts_a_directory_and_rejects_a_file() {
        let dir = std::env::temp_dir().join(format!("scan-gui-write-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");

        check_directory_writable(&dir).expect("the temp directory is writable");
        check_directory_writable(&dir.join("not/created/yet"))
            .expect("the nearest existing ancestor is writable");

        let file = dir.join("file");
        std::fs::write(&file, b"x").expect("write file");
        let error = check_directory_writable(&file.join("child")).expect_err("a file is not a dir");
        assert!(error.contains("not a directory"), "{error}");

        std::fs::remove_dir_all(&dir).expect("remove temp dir");
    }

    #[test]
    fn a_bad_record_or_dataset_path_is_rejected_before_a_worker_starts() {
        let dir = std::env::temp_dir().join(format!("scan-gui-start-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let file = dir.join("not-a-directory");
        std::fs::write(&file, b"x").expect("write file");

        let mut record = ScannerApp::new(&Options::default(), SourceChoice::Record, String::new());
        record.record_path = file.join("capture.k2df").display().to_string();
        record.start();
        assert!(record.form_error.is_some());
        assert!(record.session.is_none());

        let mut dataset = ScannerApp::new(&Options::default(), SourceChoice::Live, String::new());
        dataset.config.color = true;
        dataset.config.dataset = file.join("dataset").display().to_string();
        dataset.start();
        assert!(dataset.form_error.is_some());
        assert!(dataset.session.is_none());

        std::fs::remove_dir_all(&dir).expect("remove temp dir");
    }


    #[test]
    fn the_level_checkbox_starts_from_the_command_line_option() {
        // Off unless asked for, like the flag; and `--level` pre-ticks the box.
        assert!(!UiConfig::from_options(&Options::default()).level);

        let mut options = Options::default();
        options.level = true;
        assert!(UiConfig::from_options(&options).level);
    }

    // --- the Splats workflow: Brush detection, auto-train, and live previews ---

    use std::os::unix::fs::PermissionsExt;

    fn workflow_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("scan-gui-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// A stand-in for Brush: answers `--help` with the current flag, then runs
    /// `body` with the real arguments ($7 is the export path).
    fn fake_brush(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("brush");
        std::fs::write(
            &path,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--help\" ]; then echo '--total-steps'; exit 0; fi\n\
                 if [ \"$1\" = \"--version\" ]; then echo 'fake-brush 1.0'; exit 0; fi\n{body}\n"
            ),
        )
        .expect("write script");
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    /// A one-splat model in the layout Brush writes, so `SplatCloud::load` accepts it.
    fn tiny_splat_ply(path: &Path) {
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n\
            property float x\nproperty float y\nproperty float z\n\
            property float scale_0\nproperty float scale_1\nproperty float scale_2\n\
            property float opacity\n\
            property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n\
            property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nend_header\n";
        let mut bytes = header.as_bytes().to_vec();
        for value in [1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).expect("write ply");
    }

    fn app_with_brush(brush: Option<BrushInstall>) -> ScannerApp {
        let mut app = ScannerApp::new(&Options::default(), SourceChoice::Synthetic, String::new());
        app.brush = brush;
        app
    }

    fn exported_snapshot(dataset: &Path) -> SessionSnapshot {
        let mut snapshot = SessionSnapshot::default();
        snapshot.status = Status::Exported;
        snapshot.dataset = Some(session::DatasetSummary {
            dir: dataset.to_path_buf(),
            views: 3,
            points: 10,
            skipped: 0,
            warnings: Vec::new(),
        });
        snapshot
    }

    fn wait_until(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        done()
    }

    #[test]
    fn the_trainer_defaults_use_the_detected_brush_and_five_thousand_step_previews() {
        let ui = TrainerUi::default();
        assert!(ui.program.is_empty(), "blank means the Brush found on PATH");
        assert_eq!(ui.preview_every, 5_000);
        assert!(ui.auto_preview, "previews are shown unless turned off");
        assert!(!ui.viewer && !ui.auto_train, "the extras are opt-in");
    }

    #[test]
    fn training_without_brush_says_where_to_get_it_instead_of_failing_silently() {
        let dir = workflow_dir("nobrush");
        let mut app = app_with_brush(None);
        app.trainer_config.dataset = dir.display().to_string();
        app.trainer_config.output = dir.join("out").display().to_string();

        app.start_trainer(&[]);

        let error = app.trainer_error.clone().expect("an error is shown");
        assert!(error.contains("Brush was not found"), "{error}");
        assert!(error.contains(BRUSH_RELEASES_URL), "{error}");
        assert!(app.trainer.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_blank_trainer_field_runs_the_detected_brush() {
        let dir = workflow_dir("detected");
        let script = fake_brush(&dir, "printf x > \"$7/export_10.ply\"");
        let mut app = app_with_brush(Some(BrushInstall {
            path: script,
            version: "fake-brush 1.0".to_string(),
        }));
        app.trainer_config.dataset = dir.display().to_string();
        app.trainer_config.output = dir.join("out").display().to_string();
        app.trainer_config.iterations = 10;
        // Stale state from an earlier run must be cleared by a new one.
        app.loaded_preview = Some(99);
        app.first_preview_pending = false;

        app.start_trainer(&[]);

        assert!(app.trainer_error.is_none(), "{:?}", app.trainer_error);
        assert!(app.trainer.is_some(), "the blank field must resolve to the detected Brush");
        assert_eq!(app.loaded_preview, None, "a new run forgets the old preview");
        assert!(app.first_preview_pending, "and frames the camera again");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn training_starts_once_when_an_export_lands_and_only_if_asked() {
        let dir = workflow_dir("autotrain");
        let dataset = dir.join("dataset");
        std::fs::create_dir_all(&dataset).unwrap();
        let script = fake_brush(&dir, "printf x > \"$7/export_10.ply\"");
        let install = BrushInstall { path: script, version: String::new() };

        // Asked for: a scan is in progress, then an export lands.
        let mut app = app_with_brush(Some(install.clone()));
        app.trainer_config.auto_train = true;
        app.trainer_config.iterations = 10;
        app.trainer_config.output = dir.join("out").display().to_string();
        app.snapshot.status = Status::Running;
        app.maybe_auto_train();
        assert!(app.trainer.is_none(), "nothing to train while scanning");

        app.snapshot = exported_snapshot(&dataset);
        app.maybe_auto_train();
        assert!(app.trainer.is_some(), "the export should start training");
        assert_eq!(app.trainer_config.dataset, dataset.display().to_string());

        // The decision was made once: dropping the run and polling again, with the
        // same export still on screen, must not start another.
        app.trainer = None;
        app.maybe_auto_train();
        assert!(app.trainer.is_none(), "an export already on screen is not retrained");

        // Not asked for: the same sequence trains nothing, and ticking the box
        // afterwards does not retrain the export that is still on screen.
        let mut app = app_with_brush(Some(install));
        app.trainer_config.output = dir.join("out2").display().to_string();
        app.snapshot.status = Status::Running;
        app.maybe_auto_train();
        app.snapshot = exported_snapshot(&dataset);
        app.maybe_auto_train();
        assert!(app.trainer.is_none());
        app.trainer_config.auto_train = true;
        app.maybe_auto_train();
        assert!(app.trainer.is_none(), "ticking the box late must not retrain");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn auto_train_without_a_dataset_says_so() {
        let mut app = app_with_brush(None);
        app.trainer_config.auto_train = true;
        app.snapshot.status = Status::Running;
        app.maybe_auto_train();

        let mut snapshot = SessionSnapshot::default();
        snapshot.status = Status::Exported; // exported, but no dataset requested
        app.snapshot = snapshot;
        app.maybe_auto_train();

        let error = app.trainer_error.clone().expect("an explanation");
        assert!(error.contains("dataset"), "{error}");
        assert!(app.trainer.is_none());
    }

    #[test]
    fn the_first_preview_frames_the_view_and_later_ones_only_swap_the_model() {
        let cloud = |x: f32| SplatCloud {
            splats: vec![splat::Splat {
                position: Vector3::new(x, 0.0, 1.0),
                scale: Vector3::new(0.1, 0.1, 0.1),
                rotation: nalgebra::UnitQuaternion::identity(),
                opacity: 0.5,
                color: [0.5; 3],
            }],
            bounds: (Vector3::new(x, 0.0, 1.0), Vector3::new(x, 0.0, 1.0)),
        };
        let mut app = app_with_brush(None);
        assert!(!app.show_splats);

        app.show_preview(1000, Path::new("export_1000.ply"), cloud(1.0));
        assert!(app.show_splats, "the first preview switches to splats");
        assert!(!app.first_preview_pending);
        assert_eq!(app.loaded_preview, Some(1000));

        // The operator goes back to the mesh; a later preview must not yank them.
        app.show_splats = false;
        app.show_preview(2000, Path::new("export_2000.ply"), cloud(2.0));
        assert!(!app.show_splats, "later previews must not change the view mode");
        assert_eq!(app.loaded_preview, Some(2000));
        assert_eq!(app.splat_path, "export_2000.ply");
        assert_eq!(app.splat.as_ref().unwrap().splats[0].position.x, 2.0);
    }

    #[test]
    fn a_preview_the_trainer_writes_is_loaded_into_the_viewport_on_its_own() {
        let dir = workflow_dir("preview");
        let fixture = dir.join("fixture.ply");
        tiny_splat_ply(&fixture);
        let script = fake_brush(
            &dir,
            &format!("cp \"{}\" \"$7/export_5.ply\"", fixture.display()),
        );
        let mut app = app_with_brush(Some(BrushInstall { path: script, version: String::new() }));
        app.trainer_config.dataset = dir.display().to_string();
        app.trainer_config.output = dir.join("out").display().to_string();
        app.trainer_config.iterations = 5;
        app.trainer_config.preview_every = 5;

        app.start_trainer(&[]);
        assert!(app.trainer.is_some(), "{:?}", app.trainer_error);

        let shown = wait_until(Duration::from_secs(15), || {
            app.poll_trainer();
            app.splat.is_some()
        });
        assert!(shown, "no preview reached the viewport; error {:?}", app.splat_error);
        assert_eq!(app.loaded_preview, Some(5));
        assert!(app.show_splats, "the first preview switches the viewport to splats");
        assert_eq!(app.splat.as_ref().unwrap().len(), 1);
        assert!(app.preview_load.is_none(), "the background load is collected");

        // With Live preview off, the same export is not loaded again.
        app.splat = None;
        app.trainer_config.auto_preview = false;
        app.loaded_preview = None;
        app.poll_trainer();
        assert!(app.splat.is_none() && app.preview_load.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_preview_that_cannot_be_read_is_reported_and_not_retried_forever() {
        let dir = workflow_dir("badpreview");
        // Not a splat file: loading must fail once, be reported, and stop.
        let script = fake_brush(&dir, "printf 'not a ply' > \"$7/export_5.ply\"");
        let mut app = app_with_brush(Some(BrushInstall { path: script, version: String::new() }));
        app.trainer_config.dataset = dir.display().to_string();
        app.trainer_config.output = dir.join("out").display().to_string();
        app.trainer_config.iterations = 5;
        app.trainer_config.preview_every = 5;
        app.start_trainer(&[]);

        let reported = wait_until(Duration::from_secs(15), || {
            app.poll_trainer();
            app.splat_error.is_some()
        });
        assert!(reported, "the failure should reach the panel");
        assert_eq!(app.loaded_preview, Some(5), "the bad preview is marked handled");
        for _ in 0..5 {
            app.poll_trainer();
        }
        assert!(app.preview_load.is_none(), "it must not be retried every frame");
        assert!(app.splat.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}

