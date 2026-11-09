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

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;
use geom::coloring::ColorMode;
use geom::export::MeshFormat;
use geom::tsdf::TsdfParams;

use crate::{Options, DEFAULT_FRAMES};

use render::{Camera, PreviewScene, Renderer};
use session::{SessionConfig, SessionHandle, SessionSnapshot, Source, Status};

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
    frame_to_model: bool,
    submaps: bool,
    gpu: bool,
    unmirror: bool,
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
            frame_to_model: options.frame_to_model,
            submaps: options.submaps,
            gpu: options.gpu,
            unmirror: options.unmirror,
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
    scene: Option<Arc<PreviewScene>>,
    scene_id: u64,
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
            scene: None,
            scene_id: 0,
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

        if let Some(scene) = &self.snapshot.scene {
            if self.snapshot.scene_id != self.scene_id {
                self.scene = Some(Arc::clone(scene));
                self.scene_id = self.snapshot.scene_id;
                self.dirty = true;

                if self.auto_fit {
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
        options.frame_to_model = self.config.frame_to_model;
        options.submaps = self.config.submaps;
        options.gpu = self.config.gpu;
        options.unmirror = self.config.unmirror;
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

                    ui.label("Frame to model");
                    ui.checkbox(&mut self.config.frame_to_model, "slow, less drift");
                    ui.end_row();

                    ui.label("Submaps");
                    ui.checkbox(&mut self.config.submaps, "restart when tracking is lost");
                    ui.end_row();

                    ui.label("GPU decode");
                    ui.checkbox(&mut self.config.gpu, "wgpu feature");
                    ui.end_row();

                    ui.label("Mirror");
                    ui.checkbox(&mut self.config.unmirror, "un-mirror X");
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
            if let Some(scene) = &self.scene {
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
            let image = if let Some(scene) = &self.scene {
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
}
