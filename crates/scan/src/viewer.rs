//! A live window showing the scan as it builds.
//!
//! `scan live --viewer` opens a window that updates as frames arrive: the
//! reconstruction so far, plus the frame count, the tracked position and the
//! size of the model. It exists mainly to make *drift* legible -- a pose that
//! creeps shows up as a point cloud that smears, which is much easier to see than
//! a number in a log line at the end.
//!
//! # Why this is a picture rather than a scene
//!
//! The scan is rasterised into an RGBA buffer by the capture thread and handed
//! to the GUI as a texture. Rendering a real depth-tested point cloud would mean
//! a wgpu pipeline, a camera and its controls; the window here is a picture and
//! some numbers, which is most of the value for a fraction of the machinery. The
//! rasteriser is also plain CPU code that can be tested without a display.
//!
//! # Threading
//!
//! `winit` needs the main thread for its event loop, and the scan needs to keep
//! running, so the capture moves to a worker thread with its own async runtime
//! and the two communicate through a single-slot handoff: the GUI only ever wants
//! the *latest* preview, and a queue would just make it lag further behind.
//!
//! The worker owns everything it touches -- the device is opened inside it rather
//! than moved into it -- so nothing crosses a thread boundary except the finished
//! picture.

use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eframe::egui;
use geom::coloring::ColorView;
use nalgebra::Vector3;

use crate::Options;

/// One frame's worth of picture and numbers, ready for the GUI.
pub struct Preview {
    pub rgba: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub frame: usize,
    pub position: [f32; 3],
    pub blocks: usize,
    pub points: usize,
}

/// Accumulates coloured world-space points and rasterises them.
///
/// Colour views are absorbed incrementally: each new view contributes its pixels
/// once, and after that the accumulated cloud is just re-rendered. Absorbing is
/// the only expensive part, and it is proportional to the new frame rather than
/// to the size of the model.
pub struct Splatter {
    points: Vec<[f32; 3]>,
    colors: Vec<[u8; 3]>,
    /// How many views have been absorbed, so re-rendering never re-absorbs.
    consumed: usize,
    width: usize,
    height: usize,
    depth: Vec<f32>,
    rgba: Vec<u8>,
}

impl Splatter {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            points: Vec::new(),
            colors: Vec::new(),
            consumed: 0,
            width,
            height,
            depth: vec![f32::INFINITY; width * height],
            rgba: vec![0; width * height * 4],
        }
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Fold in every colour view not yet absorbed, at `stride` pixels.
    pub fn absorb(&mut self, views: &[ColorView], stride: usize) {
        for view in &views[self.consumed.min(views.len())..] {
            for y in (0..view.height).step_by(stride) {
                for x in (0..view.width).step_by(stride) {
                    let index = y * view.width + x;

                    let depth = view.depth[index];
                    if !depth.is_finite() || depth <= 0.0 {
                        continue;
                    }

                    let offset = index * 3;
                    if offset + 2 >= view.color.len() {
                        continue;
                    }

                    let camera = view.intrinsics.back_project(
                        x as f32 + 0.5,
                        y as f32 + 0.5,
                        depth,
                    );
                    // `transform_point`, not `pose * camera`: multiplying an
                    // isometry by a vector rotates it and silently drops the
                    // translation, which would place every view as if its camera
                    // were at the origin.
                    let world = geom::transform_point(&view.pose, &camera);

                    self.points.push([world.x, world.y, world.z]);
                    self.colors.push([
                        view.color[offset],
                        view.color[offset + 1],
                        view.color[offset + 2],
                    ]);
                }
            }
        }

        self.consumed = views.len();
    }

    /// Rasterise from `pose`, wide enough to show the model rather than just the
    /// current depth frame.
    pub fn render(&mut self, pose: &nalgebra::Isometry3<f32>) -> &[u8] {
        self.depth.fill(f32::INFINITY);
        self.rgba.fill(0);

        // A wider field of view than the sensor's, so the accumulated model is
        // visible around the current frame instead of only what is in front.
        let fov = 1.7f32;
        let focal = (self.width as f32 * 0.5) / (fov * 0.5).tan();
        let cx = self.width as f32 * 0.5;
        let cy = self.height as f32 * 0.5;

        let world_to_camera = pose.inverse();

        for (point, color) in self.points.iter().zip(&self.colors) {
            let p = Vector3::new(point[0], point[1], point[2]);
            // Same trap as in `absorb`: `*` would ignore the camera's position.
            let camera = geom::transform_point(&world_to_camera, &p);

            if camera.z <= 1e-3 {
                continue;
            }

            let u = focal * camera.x / camera.z + cx;
            let v = focal * camera.y / camera.z + cy;

            if u < 0.0 || v < 0.0 || u >= self.width as f32 || v >= self.height as f32 {
                continue;
            }

            let pixel = v as usize * self.width + u as usize;

            // Depth test, so nearer surfaces win rather than whichever point
            // happened to be written last.
            if camera.z >= self.depth[pixel] {
                continue;
            }
            self.depth[pixel] = camera.z;

            let offset = pixel * 4;
            self.rgba[offset] = color[0];
            self.rgba[offset + 1] = color[1];
            self.rgba[offset + 2] = color[2];
            self.rgba[offset + 3] = 255;
        }

        &self.rgba
    }
}

/// Run the scan on a worker thread and the window on this one.
pub fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    let latest: Arc<Mutex<Option<Preview>>> = Arc::new(Mutex::new(None));
    let stop = Arc::new(AtomicBool::new(false));
    let failure: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let worker = {
        let latest = Arc::clone(&latest);
        let stop = Arc::clone(&stop);
        let failure = Arc::clone(&failure);
        let options = options.clone();

        std::thread::spawn(move || {
            let runtime = match tokio::runtime::Runtime::new() {
                Ok(runtime) => runtime,
                Err(e) => {
                    *failure.lock().unwrap() = Some(format!("starting the scan runtime: {e}"));
                    return;
                }
            };

            let mut splatter = Splatter::new(640, 480);

            let scanned = runtime.block_on(crate::run_live(&options, |index, scanner| {
                // Only the pixels of views not yet absorbed are walked, so the
                // cost here is proportional to the new frame, not to the model.
                splatter.absorb(scanner.color_views(), 3);

                let rgba = splatter.render(&scanner.pose()).to_vec();
                let position = scanner.pose().translation.vector;

                *latest.lock().unwrap() = Some(Preview {
                    rgba,
                    width: splatter.width,
                    height: splatter.height,
                    frame: index,
                    position: [position.x, position.y, position.z],
                    blocks: scanner.block_count(),
                    points: splatter.point_count(),
                });

                // Stops when the window closes.
                !stop.load(Ordering::Relaxed)
            }));

            match scanned {
                // The window closing ends the scan, so write out whatever was
                // reconstructed rather than discarding it.
                Ok(scanner) => {
                    if let Err(e) = crate::finish(&scanner, &options) {
                        *failure.lock().unwrap() = Some(e.to_string());
                    }
                }
                Err(e) => *failure.lock().unwrap() = Some(e.to_string()),
            }
        })
    };

    let app = ViewerApp {
        latest: Arc::clone(&latest),
        stop: Arc::clone(&stop),
        failure: Arc::clone(&failure),
        texture: None,
        last_frame: 0,
        stats: None,
        size: (1, 1),
    };

    let result = eframe::run_native(
        "scan",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([720.0, 620.0]),
            ..Default::default()
        },
        Box::new(move |_cc| Ok(Box::new(app))),
    );

    // Closing the window ends the scan; the worker checks the flag between frames.
    stop.store(true, Ordering::Relaxed);
    let _ = worker.join();

    if let Some(message) = failure.lock().unwrap().take() {
        return Err(message.into());
    }

    result.map_err(|e| format!("viewer: {e}").into())
}

/// The numbers shown next to the picture, copied out of the last preview so the
/// handoff lock is not held while widgets are being built.
struct Stats {
    position: [f32; 3],
    blocks: usize,
    points: usize,
}

struct ViewerApp {
    latest: Arc<Mutex<Option<Preview>>>,
    stop: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
    texture: Option<egui::TextureHandle>,
    last_frame: usize,
    stats: Option<Stats>,
    size: (usize, usize),
}

impl eframe::App for ViewerApp {
    /// The `Context` is only reachable here, and the only thing this needs it for
    /// is to ask for the next repaint. The scan runs at a few frames a second, so
    /// there is no point spinning.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(50));
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Single-slot handoff: take whatever the worker last produced. Everything
        // drawn below is copied out first, so the lock is not held across the
        // widget calls.
        if let Some(preview) = self.latest.lock().unwrap().take() {
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [preview.width, preview.height],
                &preview.rgba,
            );
            self.texture = Some(ui.ctx().load_texture(
                "preview",
                image,
                egui::TextureOptions::NEAREST,
            ));

            self.last_frame = preview.frame;
            self.size = (preview.width, preview.height);
            self.stats = Some(Stats {
                position: preview.position,
                blocks: preview.blocks,
                points: preview.points,
            });
        }

        if let Some(message) = self.failure.lock().unwrap().clone() {
            ui.colored_label(egui::Color32::RED, message);
            return;
        }

        let Some(stats) = &self.stats else {
            ui.label("waiting for the first frame ...");
            return;
        };

        ui.horizontal(|ui| {
            ui.label(format!("frame {}", self.last_frame));
            ui.separator();
            ui.label(format!(
                "position [{:+.3} {:+.3} {:+.3}] m",
                stats.position[0], stats.position[1], stats.position[2]
            ));
            ui.separator();
            ui.label(format!("{} blocks", stats.blocks));
            ui.separator();
            ui.label(format!("{} points", stats.points));
        });

        if let Some(texture) = &self.texture {
            let available = ui.available_size();
            let scale = (available.x / self.size.0 as f32)
                .min(available.y / self.size.1 as f32)
                .max(0.01);
            let size = egui::vec2(
                self.size.0 as f32 * scale,
                self.size.1 as f32 * scale,
            );
            ui.image(egui::load::SizedTexture::new(texture.id(), size));
        }

        ui.label(
            "closing this window stops the scan. Frames arrive at roughly 2-4 per \
             second with colour enabled.",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion};

    /// A tiny view looking down +z: 8x8 pixels, everything 2 m away, one colour.
    fn view() -> ColorView {
        let (width, height) = (8, 8);
        ColorView {
            color: [200u8, 100, 50]
                .iter()
                .cycle()
                .take(width * height * 3)
                .cloned()
                .collect(),
            width,
            height,
            depth: vec![2.0; width * height],
            pose: Isometry3::identity(),
            intrinsics: geom::Intrinsics {
                fx: 8.0,
                fy: 8.0,
                cx: 4.0,
                cy: 4.0,
            },
        }
    }

    #[test]
    fn absorbing_a_view_draws_its_points_in_its_colour() {
        let mut splatter = Splatter::new(32, 32);
        splatter.absorb(&[view()], 1);

        assert_eq!(splatter.point_count(), 64, "8x8 at stride 1");

        let rgba = splatter.render(&Isometry3::identity());
        let painted: Vec<_> = rgba.chunks_exact(4).filter(|p| p[3] == 255).collect();

        assert!(!painted.is_empty(), "nothing was drawn at all");
        for pixel in painted {
            assert_eq!(
                [pixel[0], pixel[1], pixel[2]],
                [200, 100, 50],
                "a drawn pixel carried the wrong colour"
            );
        }
    }

    #[test]
    fn a_view_is_only_absorbed_once() {
        // The render loop calls absorb on every frame, so re-absorbing would
        // silently multiply the point count with every update.
        let mut splatter = Splatter::new(32, 32);
        let views = vec![view()];

        splatter.absorb(&views, 1);
        let after_first = splatter.point_count();
        splatter.absorb(&views, 1);

        assert_eq!(splatter.point_count(), after_first);
    }

    #[test]
    fn points_behind_the_camera_are_not_drawn() {
        let mut splatter = Splatter::new(32, 32);
        splatter.absorb(&[view()], 1);

        // 10 m along +z from points that sit at 2 m: all of them are behind.
        let behind = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 10.0),
            UnitQuaternion::identity(),
        );

        let rgba = splatter.render(&behind);
        let painted = rgba.chunks_exact(4).filter(|p| p[3] == 255).count();
        assert_eq!(painted, 0, "{painted} points behind the camera were drawn");
    }

    #[test]
    fn stride_reduces_the_point_count() {
        let mut full = Splatter::new(32, 32);
        full.absorb(&[view()], 1);

        let mut sparse = Splatter::new(32, 32);
        sparse.absorb(&[view()], 2);

        assert_eq!(sparse.point_count(), full.point_count() / 4);
    }
}
