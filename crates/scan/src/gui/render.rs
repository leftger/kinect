//! A software rasteriser and an orbit camera for the live preview.
//!
//! The preview deliberately does not need a GPU pipeline. The scan runs on a
//! worker thread at a few frames a second, and the window needs a picture it can
//! navigate. A depth-tested z-buffer over the extracted surface is enough, it is
//! plain code that can be tested without a display, and it does not pull a
//! second renderer in beside the one `wgpu-decode` already uses.
//!
//! The renderer is fed a [`PreviewScene`]: a mesh from the TSDF, the points of
//! the latest depth frame, and a few world-space lines for the trajectory and
//! the current sensor. Triangle colours are per-vertex when the mesh carries
//! them, and flat-lit otherwise.

use geom::mesh::Mesh;
use nalgebra::Vector3;

use super::splat::SplatCloud;

/// One point from the most recent depth frame, already in world space.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreviewPoint {
    pub position: [f32; 3],
    pub color: [u8; 3],
}

/// A world-space line segment, used for the trajectory and the sensor marker.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreviewLine {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [u8; 3],
}

/// Everything needed to draw one preview update.
pub(crate) struct PreviewScene {
    /// Surface extracted from the TSDF. Empty until the first surface appears.
    pub mesh: Mesh,
    /// The latest frame's depth samples, in world space. These show something
    /// before the first mesh exists, and make a bad frame visible immediately.
    pub points: Vec<PreviewPoint>,
    /// Camera path and the current sensor frustum.
    pub lines: Vec<PreviewLine>,
    /// Axis-aligned bounds of the mesh, for "fit the view".
    pub bounds: Option<(Vector3<f32>, Vector3<f32>)>,
}

impl PreviewScene {
    /// An empty scene, useful before the first frame and in tests.
    pub fn empty() -> Self {
        Self {
            mesh: Mesh::default(),
            points: Vec::new(),
            lines: Vec::new(),
            bounds: None,
        }
    }
}

/// A turntable camera with a target, two angles and a distance.
///
/// The sensor's world is y-down (the Kinect convention). The display uses
/// `-y` as up, so the picture is the right way up without mirroring the
/// geometry: a y reflection would also flip every triangle's winding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Camera {
    pub target: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vector3::zeros(),
            yaw: 0.60,
            pitch: 0.35,
            distance: 3.0,
            fov_y: 1.0,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vector3<f32> {
        let cos_pitch = self.pitch.cos();
        let direction = Vector3::new(
            cos_pitch * self.yaw.sin(),
            -self.pitch.sin(),
            cos_pitch * self.yaw.cos(),
        );
        self.target + direction * self.distance
    }

    /// Orbit around the target. Drag right turns the model, drag up lifts the
    /// camera above it.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.45, 1.45);
    }

    /// Move the target in the camera plane, so the scene follows the pointer.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let height = viewport_height.max(1.0);
        let (right, up, _) = self.basis();
        let world_per_pixel = 2.0 * self.distance * (self.fov_y * 0.5).tan() / height;

        self.target -= right * (dx * world_per_pixel);
        self.target += up * (dy * world_per_pixel);
    }

    /// Zoom by a scroll delta. Positive scroll (wheel up) moves closer.
    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (-scroll * 0.0015).exp()).clamp(0.05, 500.0);
    }

    /// Put the whole model in view.
    pub fn fit(&mut self, min: Vector3<f32>, max: Vector3<f32>) {
        let center = (min + max) * 0.5;
        let extent = (max - min).norm().max(0.1);
        let half_fov = (self.fov_y * 0.5).tan().max(1e-3);

        self.target = center;
        self.distance = (extent * 0.5 / half_fov * 1.25).clamp(0.2, 1_000.0);
    }

    fn basis(&self) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>) {
        let eye = self.eye();
        let mut forward = self.target - eye;
        let length = forward.norm();
        if length > 1e-6 {
            forward /= length;
        } else {
            forward = Vector3::new(0.0, 0.0, -1.0);
        }

        // Display up is -y. `up_hint.cross(forward)` gives +x to the right for
        // that choice, so the picture is not mirrored.
        let up_hint = Vector3::new(0.0, -1.0, 0.0);
        let mut right = up_hint.cross(&forward);
        if right.norm() < 1e-6 {
            right = Vector3::new(1.0, 0.0, 0.0);
        } else {
            right /= right.norm();
        }
        let up = forward.cross(&right);

        (right, up, forward)
    }

    fn view(&self, width: usize, height: usize) -> View {
        let (right, up, forward) = self.basis();
        let height_f = height.max(1) as f32;

        View {
            eye: self.eye(),
            right,
            up,
            forward,
            focal: (height_f * 0.5) / (self.fov_y * 0.5).tan().max(1e-3),
            cx: width as f32 * 0.5,
            cy: height_f * 0.5,
            width,
            height,
        }
    }
}

/// The camera basis and intrinsics for one render.
struct View {
    eye: Vector3<f32>,
    right: Vector3<f32>,
    up: Vector3<f32>,
    forward: Vector3<f32>,
    focal: f32,
    cx: f32,
    cy: f32,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy, Debug)]
struct Projected {
    x: f32,
    y: f32,
    inv_z: f32,
}

impl View {
    fn project(&self, point: &Vector3<f32>) -> Option<Projected> {
        let delta = point - self.eye;
        let z = delta.dot(&self.forward);
        if z <= 1e-4 {
            return None;
        }

        let x = delta.dot(&self.right);
        let y = delta.dot(&self.up);

        Some(Projected {
            x: self.cx + self.focal * x / z,
            y: self.cy - self.focal * y / z,
            inv_z: 1.0 / z,
        })
    }
}

/// A depth-tested RGBA rasteriser.
pub(crate) struct Renderer {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
    /// Inverse depth. Zero means "nothing drawn"; larger is nearer.
    depth: Vec<f32>,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        Self {
            width,
            height,
            rgba: vec![0; width * height * 4],
            depth: vec![0.0; width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn render(&mut self, scene: &PreviewScene, camera: &Camera) -> &[u8] {
        self.clear_background();

        let view = camera.view(self.width, self.height);

        // Axes at the world origin: x red, y green, z blue. The green axis
        // points down on screen because the sensor's +y is down.
        let origin = Vector3::zeros();
        let x = Vector3::new(0.4, 0.0, 0.0);
        let y = Vector3::new(0.0, 0.4, 0.0);
        let z = Vector3::new(0.0, 0.0, 0.4);
        self.line(&view, &origin, &x, [220, 80, 80]);
        self.line(&view, &origin, &y, [80, 220, 100]);
        self.line(&view, &origin, &z, [90, 130, 255]);

        // Points first: they are the latest frame and fill in before the first
        // surface exists. Where the mesh has a surface at the same depth, the
        // mesh's real colour wins.
        self.draw_points(&view, scene);
        self.draw_mesh(&view, scene);

        for line in &scene.lines {
            let a = Vector3::from(line.a);
            let b = Vector3::from(line.b);
            self.line(&view, &a, &b, line.color);
        }

        &self.rgba
    }

    /// Render a trained splat cloud as sorted, alpha-blended circular billboards.
    ///
    /// This is a deliberately conservative viewer: each splat uses its largest
    /// linear axis as an isotropic screen radius. It does not reconstruct the
    /// full anisotropic covariance or evaluate higher SH bands. The point is to
    /// see the trained result beside the mesh and check pose and scale, not to
    /// claim a production 3DGS renderer.
    pub fn render_splats(&mut self, cloud: &SplatCloud, camera: &Camera) -> &[u8] {
        self.clear_background();
        let view = camera.view(self.width, self.height);

        let mut order: Vec<(f32, usize)> = cloud
            .splats
            .iter()
            .enumerate()
            .filter_map(|(index, splat)| {
                let z = (splat.position - view.eye).dot(&view.forward);
                (z > 1e-3).then_some((z, index))
            })
            .collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0));

        for (z, index) in order {
            let splat = &cloud.splats[index];
            let delta = splat.position - view.eye;
            let sx = view.cx + view.focal * delta.dot(&view.right) / z;
            let sy = view.cy - view.focal * delta.dot(&view.up) / z;
            let sigma = (view.focal * splat.scale / z).clamp(0.5, 48.0);
            let radius = (sigma * 3.0).ceil();
            let sigma_inv = 1.0 / (sigma * sigma);

            let x0 = ((sx - radius).floor() as isize).max(0);
            let x1 = ((sx + radius).ceil() as isize).min(self.width as isize - 1);
            let y0 = ((sy - radius).floor() as isize).max(0);
            let y1 = ((sy + radius).ceil() as isize).min(self.height as isize - 1);
            if x0 > x1 || y0 > y1 {
                continue;
            }

            for py in y0..=y1 {
                let dy = py as f32 + 0.5 - sy;
                for px in x0..=x1 {
                    let dx = px as f32 + 0.5 - sx;
                    let alpha = splat.opacity * (-0.5 * (dx * dx + dy * dy) * sigma_inv).exp();
                    if alpha <= 0.001 {
                        continue;
                    }

                    let pixel = ((py as usize) * self.width + px as usize) * 4;
                    let inverse = 1.0 - alpha;
                    for channel in 0..3 {
                        let dst = self.rgba[pixel + channel] as f32 / 255.0;
                        self.rgba[pixel + channel] =
                            ((splat.color[channel] * alpha + dst * inverse) * 255.0)
                                .clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }

        &self.rgba
    }

    fn clear_background(&mut self) {
        for (index, pixel) in self.rgba.chunks_exact_mut(4).enumerate() {
            let y = index / self.width;
            let t = y as f32 / self.height.max(1) as f32;
            pixel[0] = (18.0 + 12.0 * t) as u8;
            pixel[1] = (20.0 + 13.0 * t) as u8;
            pixel[2] = (26.0 + 16.0 * t) as u8;
            pixel[3] = 255;
        }
        self.depth.fill(0.0);
    }

    fn draw_mesh(&mut self, view: &View, scene: &PreviewScene) {
        let colors = scene.mesh.colors.as_ref();

        for triangle in &scene.mesh.triangles {
            let (Some(a), Some(b), Some(c)) = (
                scene.mesh.vertices.get(triangle[0] as usize),
                scene.mesh.vertices.get(triangle[1] as usize),
                scene.mesh.vertices.get(triangle[2] as usize),
            ) else {
                continue;
            };

            let vertex_colors = colors.and_then(|colors| {
                Some([
                    *colors.get(triangle[0] as usize)?,
                    *colors.get(triangle[1] as usize)?,
                    *colors.get(triangle[2] as usize)?,
                ])
            });

            self.triangle(view, a, b, c, vertex_colors);
        }
    }

    fn draw_points(&mut self, view: &View, scene: &PreviewScene) {
        for point in &scene.points {
            self.point(view, &Vector3::from(point.position), point.color);
        }
    }

    fn triangle(
        &mut self,
        view: &View,
        a: &Vector3<f32>,
        b: &Vector3<f32>,
        c: &Vector3<f32>,
        colors: Option<[[u8; 3]; 3]>,
    ) {
        let normal = (b - a).cross(&(c - a));
        if normal.norm_squared() < 1e-12 || !normal.norm_squared().is_finite() {
            return;
        }

        // Back-face culling. The extracted surface is consistently wound, so a
        // triangle whose outward normal points away from the eye is interior.
        let centroid = (a + b + c) / 3.0;
        if normal.dot(&(view.eye - centroid)) <= 0.0 {
            return;
        }

        let normal = normal / normal.norm();
        // A key light from above (-y) and to the right, plus ambient. `abs`
        // keeps an inconsistently wound triangle visible rather than black.
        let light: Vector3<f32> = Vector3::new(0.35, -0.82, 0.45);
        let light = light / light.norm();
        let shade = 0.32 + 0.68 * normal.dot(&light).abs();

        let (Some(p0), Some(p1), Some(p2)) = (view.project(a), view.project(b), view.project(c))
        else {
            return;
        };

        let min_x = p0.x.min(p1.x).min(p2.x).floor().max(0.0) as usize;
        let max_x = p0.x.max(p1.x).max(p2.x).ceil().min(view.width as f32 - 1.0) as usize;
        let min_y = p0.y.min(p1.y).min(p2.y).floor().max(0.0) as usize;
        let max_y =
            p0.y.max(p1.y)
                .max(p2.y)
                .ceil()
                .min(view.height as f32 - 1.0) as usize;

        if min_x > max_x || min_y > max_y {
            return;
        }

        // Barycentric denominators can be positive or negative with the same
        // formula; the inside test below accepts both.
        let denom = (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y);
        if denom.abs() < 1e-6 {
            return;
        }
        let inv_denom = 1.0 / denom;

        let base = colors.unwrap_or([[196, 201, 210]; 3]);
        let base = [
            base[0].map(|v| v as f32),
            base[1].map(|v| v as f32),
            base[2].map(|v| v as f32),
        ];

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let px = x as f32 + 0.5;
                let py = y as f32 + 0.5;

                let w0 = ((p1.y - p2.y) * (px - p2.x) + (p2.x - p1.x) * (py - p2.y)) * inv_denom;
                let w1 = ((p2.y - p0.y) * (px - p2.x) + (p0.x - p2.x) * (py - p2.y)) * inv_denom;
                let w2 = 1.0 - w0 - w1;

                if w0 < -1e-4 || w1 < -1e-4 || w2 < -1e-4 {
                    continue;
                }

                let inv_z = w0 * p0.inv_z + w1 * p1.inv_z + w2 * p2.inv_z;
                let index = y * self.width + x;
                // `>=` at equal depth so the scene can paint over the
                // orientation axes, which are drawn first at the origin.
                if inv_z < self.depth[index] {
                    continue;
                }
                self.depth[index] = inv_z;

                let rgb = [
                    (w0 * base[0][0] + w1 * base[1][0] + w2 * base[2][0]) * shade,
                    (w0 * base[0][1] + w1 * base[1][1] + w2 * base[2][1]) * shade,
                    (w0 * base[0][2] + w1 * base[1][2] + w2 * base[2][2]) * shade,
                ];

                let offset = index * 4;
                self.rgba[offset] = rgb[0].clamp(0.0, 255.0) as u8;
                self.rgba[offset + 1] = rgb[1].clamp(0.0, 255.0) as u8;
                self.rgba[offset + 2] = rgb[2].clamp(0.0, 255.0) as u8;
                self.rgba[offset + 3] = 255;
            }
        }
    }

    fn point(&mut self, view: &View, position: &Vector3<f32>, color: [u8; 3]) {
        let Some(projected) = view.project(position) else {
            return;
        };

        let x = projected.x.floor() as i32;
        let y = projected.y.floor() as i32;
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }

        for dy in 0..2 {
            for dx in 0..2 {
                let px = x + dx;
                let py = y + dy;
                if px < 0 || py < 0 || px >= self.width as i32 || py >= self.height as i32 {
                    continue;
                }

                let index = py as usize * self.width + px as usize;
                if projected.inv_z < self.depth[index] {
                    continue;
                }
                self.depth[index] = projected.inv_z;

                let offset = index * 4;
                self.rgba[offset] = color[0];
                self.rgba[offset + 1] = color[1];
                self.rgba[offset + 2] = color[2];
                self.rgba[offset + 3] = 255;
            }
        }
    }

    fn line(&mut self, view: &View, a: &Vector3<f32>, b: &Vector3<f32>, color: [u8; 3]) {
        let (Some(qa), Some(qb)) = (view.project(a), view.project(b)) else {
            return;
        };

        let dx = qb.x - qa.x;
        let dy = qb.y - qa.y;
        let steps = dx.abs().max(dy.abs()).ceil() as i32;
        if steps <= 0 {
            return;
        }
        let steps = steps.min(8_192);

        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let x = qa.x + dx * t;
            let y = qa.y + dy * t;
            // 1/z is linear in screen space along a straight 3D line.
            let inv_z = qa.inv_z + (qb.inv_z - qa.inv_z) * t;

            let x = x.floor() as i32;
            let y = y.floor() as i32;
            if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
                continue;
            }

            let index = y as usize * self.width + x as usize;
            if inv_z < self.depth[index] {
                continue;
            }
            self.depth[index] = inv_z;

            let offset = index * 4;
            self.rgba[offset] = color[0];
            self.rgba[offset + 1] = color[1];
            self.rgba[offset + 2] = color[2];
            self.rgba[offset + 3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::splat::Splat;

    fn camera_looking_at(target: Vector3<f32>, distance: f32) -> Camera {
        Camera {
            target,
            yaw: 0.0,
            pitch: 0.0,
            distance,
            fov_y: 1.0,
        }
    }

    fn quad(z: f32, color: [u8; 3]) -> Mesh {
        Mesh {
            vertices: vec![
                Vector3::new(-1.0, -1.0, z),
                Vector3::new(1.0, -1.0, z),
                Vector3::new(1.0, 1.0, z),
                Vector3::new(-1.0, 1.0, z),
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            colors: Some(vec![color; 4]),
        }
    }

    #[test]
    fn the_target_projects_to_the_centre_of_the_view() {
        let camera = camera_looking_at(Vector3::new(1.0, 2.0, 3.0), 4.0);
        let view = camera.view(100, 80);

        let projected = view.project(&camera.target).expect("target is in front");
        assert!((projected.x - 50.0).abs() < 1e-3, "{projected:?}");
        assert!((projected.y - 40.0).abs() < 1e-3, "{projected:?}");
    }

    #[test]
    fn points_behind_the_camera_are_not_projected() {
        let camera = camera_looking_at(Vector3::zeros(), 3.0);
        let view = camera.view(100, 100);

        // The camera looks along -z from +z=3, so +4 is behind it.
        assert!(view.project(&Vector3::new(0.0, 0.0, 4.0)).is_none());
    }

    #[test]
    fn the_nearer_surface_wins_the_depth_test() {
        let camera = camera_looking_at(Vector3::zeros(), 3.0);
        let mut renderer = Renderer::new(64, 64);

        let near = quad(0.0, [20, 40, 220]);
        let far = quad(-1.0, [220, 40, 20]);
        let mut vertices = near.vertices.clone();
        vertices.extend(far.vertices.iter().copied());
        let triangles = near
            .triangles
            .iter()
            .map(|t| [t[0], t[1], t[2]])
            .chain(far.triangles.iter().map(|t| [t[0] + 4, t[1] + 4, t[2] + 4]))
            .collect();

        let scene = PreviewScene {
            mesh: Mesh {
                vertices,
                triangles,
                colors: Some(
                    near.colors
                        .unwrap()
                        .into_iter()
                        .chain(far.colors.unwrap())
                        .collect(),
                ),
            },
            ..PreviewScene::empty()
        };

        let rgba = renderer.render(&scene, &camera);
        let centre = (32 * 64 + 32) * 4;
        assert!(
            rgba[centre + 2] > rgba[centre],
            "the near blue quad should cover the far red one: {:?}",
            &rgba[centre..centre + 3]
        );
    }

    #[test]
    fn an_empty_scene_is_still_an_opaque_picture() {
        let camera = Camera::default();
        let mut renderer = Renderer::new(32, 24);
        let rgba = renderer.render(&PreviewScene::empty(), &camera);

        assert_eq!(rgba.len(), 32 * 24 * 4);
        assert!(rgba.chunks_exact(4).all(|pixel| pixel[3] == 255));
    }

    #[test]
    fn fitting_places_the_target_and_leaves_the_model_in_front() {
        let mut camera = camera_looking_at(Vector3::zeros(), 100.0);
        camera.fit(Vector3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0));

        assert!((camera.target - Vector3::zeros()).norm() < 1e-6);
        assert!(camera.distance > 0.2 && camera.distance < 100.0);
    }

    #[test]
    fn a_splat_in_front_of_the_camera_is_visible() {
        let position = Vector3::zeros();
        let cloud = SplatCloud {
            splats: vec![Splat {
                position,
                scale: 0.1,
                opacity: 1.0,
                color: [1.0, 0.0, 0.0],
            }],
            bounds: (position, position),
        };
        let camera = camera_looking_at(Vector3::zeros(), 3.0);
        let mut renderer = Renderer::new(64, 64);
        let rgba = renderer.render_splats(&cloud, &camera);
        let centre = (32 * 64 + 32) * 4;

        assert!(
            rgba[centre] > 200 && rgba[centre + 1] < 50,
            "a red splat should cover the centre: {:?}",
            &rgba[centre..centre + 3]
        );
    }

    #[test]
    fn a_splat_behind_the_camera_is_not_drawn() {
        let position = Vector3::new(0.0, 0.0, 10.0);
        let cloud = SplatCloud {
            splats: vec![Splat {
                position,
                scale: 0.1,
                opacity: 1.0,
                color: [1.0, 0.0, 0.0],
            }],
            bounds: (position, position),
        };
        let camera = camera_looking_at(Vector3::zeros(), 3.0);
        let mut renderer = Renderer::new(64, 64);
        let rgba = renderer.render_splats(&cloud, &camera);
        let centre = (32 * 64 + 32) * 4;

        assert!(
            rgba[centre] < 80,
            "a splat behind the camera should leave the background: {:?}",
            &rgba[centre..centre + 3]
        );
    }

    #[test]
    #[ignore = "set SCAN_SPLAT_PLY to a trained Brush export"]
    fn renders_a_real_brush_export() {
        use std::time::Instant;

        let path = std::env::var("SCAN_SPLAT_PLY").expect("SCAN_SPLAT_PLY");
        let cloud = SplatCloud::load(std::path::Path::new(&path)).expect("load Brush PLY");
        let mut camera = Camera::default();
        camera.fit(cloud.bounds.0, cloud.bounds.1);

        let mut renderer = Renderer::new(320, 240);
        let started = Instant::now();
        let rgba = renderer.render_splats(&cloud, &camera);
        println!(
            "rendered {} splats at 320x240 in {:?}",
            cloud.len(),
            started.elapsed()
        );

        assert_eq!(rgba.len(), 320 * 240 * 4);
        assert!(rgba.chunks_exact(4).all(|pixel| pixel[3] == 255));
    }
}
