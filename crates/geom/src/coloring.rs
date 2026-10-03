//! Painting a fused mesh with colour from the frames that saw it.
//!
//! The TSDF is geometry only: fusing colour into the volume itself would mean a
//! second volume and a second set of update rules. Colouring the finished mesh is
//! far less machinery, and it has a real advantage -- once the model is final,
//! *every* frame that saw a surface can contribute to its colour, rather than
//! only the frames that happened to arrive while that surface was being fused.
//!
//! # What a view is
//!
//! A `ColorView` is a frame's colour already registered into the *depth* camera's
//! grid, which is what `Registration::undistort_depth_and_color` produces. That
//! registration is the whole reason this is tractable: with the colour in the
//! depth grid, projecting a vertex only needs the depth intrinsics, and the
//! depth image from the same frame is directly usable for the visibility test.
//!
//! The alternative, projecting from the depth camera into the colour camera, needs
//! the stereo extrinsics for every sample and is what registration already did
//! once, properly, for the whole image.
//!
//! # Visibility is the hard part
//!
//! Sampling the wrong colour is easy: a vertex on a far wall, seen through a
//! doorway from some other pose, would be painted with whatever is in front of
//! it. So a sample is only taken where the view's own depth measurement agrees
//! with how far away the vertex actually is. That is the same occlusion test the
//! projective odometry uses, applied to colour instead of geometry.
//!
//! # What this does not do
//!
//! It averages every view that passed the test, weighted equally. It does not
//! weight by how squarely a view saw the surface, because that needs per-vertex
//! normals; and it does not correct exposure, even though `ColorFrame` carries
//! exposure, gain and gamma per frame, so surfaces seen at different exposures
//! will show seams. Both are the obvious next improvements.

use nalgebra::Vector3;

use crate::{transform_point, Intrinsics};

/// One frame's colour, registered into the depth camera's grid.
pub struct ColorView {
    /// RGB, `width * height * 3`, row-major.
    pub color: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// Depth in metres from the same frame and the same grid, for the visibility
    /// test. Distances *along the camera axis*, which is what `project` gives.
    pub depth: Vec<f32>,
    /// World-from-camera pose of the frame the colour came from.
    pub pose: nalgebra::Isometry3<f32>,
    /// Depth-camera intrinsics, since the colour is in the depth grid.
    pub intrinsics: Intrinsics,
}

#[derive(Clone, Copy, Debug)]
pub struct ColoringParams {
    /// How far a vertex may disagree with a view's measured depth and still count
    /// as visible from it, in metres. Too tight and nothing is ever visible; too
    /// loose and colour bleeds through walls.
    pub depth_tolerance: f32,
    /// Colour given to vertices no view could see.
    pub fallback: [u8; 3],
}

impl Default for ColoringParams {
    fn default() -> Self {
        Self {
            depth_tolerance: 0.05,
            fallback: [128, 128, 128],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ColoringReport {
    pub vertices: usize,
    /// Vertices no view could see, left at the fallback colour.
    pub unobserved: usize,
    /// `(vertex, view)` pairs that passed the visibility test.
    pub samples: usize,
}

impl ColoringReport {
    /// Mean number of views that contributed to each coloured vertex.
    pub fn mean_samples(&self) -> f32 {
        let coloured = self.vertices - self.unobserved;
        if coloured == 0 {
            return 0.0;
        }
        self.samples as f32 / coloured as f32
    }
}

/// Average the views that can see each vertex.
pub fn colorize(
    vertices: &[Vector3<f32>],
    views: &[ColorView],
    params: &ColoringParams,
) -> (Vec<[u8; 3]>, ColoringReport) {
    let mut colors = vec![params.fallback; vertices.len()];
    let mut unobserved = 0;
    let mut samples = 0;

    // Scratch, reused so the inner loop allocates nothing.
    let mut sum = [0.0f32; 3];
    let mut count = 0.0f32;

    for (index, vertex) in vertices.iter().enumerate() {
        sum = [0.0; 3];
        count = 0.0;

        for view in views {
            let Some(rgb) = sample(view, vertex, params.depth_tolerance) else {
                continue;
            };

            sum[0] += rgb[0] as f32;
            sum[1] += rgb[1] as f32;
            sum[2] += rgb[2] as f32;
            count += 1.0;
            samples += 1;
        }

        if count > 0.0 {
            colors[index] = [
                (sum[0] / count).round().clamp(0.0, 255.0) as u8,
                (sum[1] / count).round().clamp(0.0, 255.0) as u8,
                (sum[2] / count).round().clamp(0.0, 255.0) as u8,
            ];
        } else {
            unobserved += 1;
        }
    }

    (
        colors,
        ColoringReport {
            vertices: vertices.len(),
            unobserved,
            samples,
        },
    )
}

/// The colour `view` gives this world point, or `None` if it could not see it.
fn sample(view: &ColorView, world: &Vector3<f32>, tolerance: f32) -> Option<[u8; 3]> {
    let camera = transform_point(&view.pose.inverse(), world);

    // Behind the camera, or so close the projection is meaningless.
    if camera.z <= 1e-3 {
        return None;
    }

    let (u, v) = view.intrinsics.project(&camera)?;

    let x = u.floor();
    let y = v.floor();
    if x < 0.0 || y < 0.0 || x >= view.width as f32 || y >= view.height as f32 {
        return None;
    }

    let (x, y) = (x as usize, y as usize);
    let pixel = y * view.width + x;

    // The occlusion test. If the view measured a surface much nearer or much
    // further than this vertex is, then this vertex is not what that pixel is
    // looking at.
    let measured = view.depth[pixel];
    if !measured.is_finite() || measured <= 0.0 {
        return None;
    }
    if (measured - camera.z).abs() > tolerance {
        return None;
    }

    let offset = pixel * 3;
    if offset + 2 >= view.color.len() {
        return None;
    }

    Some([
        view.color[offset],
        view.color[offset + 1],
        view.color[offset + 2],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Translation3};

    const SIDE: usize = 64;

    fn intrinsics() -> Intrinsics {
        Intrinsics {
            fx: 60.0,
            fy: 60.0,
            cx: SIDE as f32 / 2.0,
            cy: SIDE as f32 / 2.0,
        }
    }

    /// A view that sees the plane z = 2.0 from the origin, filled with one colour
    /// except for a single pixel.
    fn view_at(pose: Isometry3<f32>, fill: [u8; 3]) -> ColorView {
        ColorView {
            color: fill
                .iter()
                .cycle()
                .take(SIDE * SIDE * 3)
                .cloned()
                .collect(),
            width: SIDE,
            height: SIDE,
            depth: vec![2.0; SIDE * SIDE],
            pose,
            intrinsics: intrinsics(),
        }
    }

    fn at(x: f32, y: f32, z: f32) -> Isometry3<f32> {
        Isometry3::from_parts(Translation3::new(x, y, z), nalgebra::UnitQuaternion::identity())
    }

    #[test]
    fn a_vertex_takes_the_colour_of_the_view_that_saw_it() {
        let views = vec![view_at(Isometry3::identity(), [10, 20, 30])];
        let vertex = Vector3::new(0.0, 0.0, 2.0);

        let (colors, report) = colorize(&[vertex], &views, &ColoringParams::default());

        assert_eq!(colors[0], [10, 20, 30]);
        assert_eq!(report.unobserved, 0);
        assert_eq!(report.samples, 1);
    }

    #[test]
    fn a_view_that_cannot_see_the_vertex_does_not_paint_it() {
        // One view whose depth says the surface is 2 m away, and one that says
        // 0.5 m. The vertex sits at 2 m, so only the first may colour it. Without
        // the visibility test the second would bleed its colour in.
        let mut wrong = view_at(at(0.0, 0.0, 0.0), [255, 0, 0]);
        wrong.depth = vec![0.5; SIDE * SIDE];

        let views = vec![view_at(Isometry3::identity(), [0, 255, 0]), wrong];
        let vertex = Vector3::new(0.0, 0.0, 2.0);

        let (colors, report) = colorize(&[vertex], &views, &ColoringParams::default());

        assert_eq!(colors[0], [0, 255, 0], "the occluded view leaked colour");
        assert_eq!(report.samples, 1);
    }

    #[test]
    fn a_vertex_no_view_saw_keeps_the_fallback() {
        // Empty views, and a vertex outside the image of the only one there is.
        let param = ColoringParams {
            fallback: [1, 2, 3],
            ..ColoringParams::default()
        };

        let (colors, report) = colorize(
            &[Vector3::new(50.0, 50.0, 2.0)],
            &[view_at(Isometry3::identity(), [9, 9, 9])],
            &param,
        );

        assert_eq!(colors[0], [1, 2, 3]);
        assert_eq!(report.unobserved, 1);
        assert_eq!(report.samples, 0);
    }

    #[test]
    fn colour_is_averaged_across_every_view_that_saw_the_vertex() {
        let views = vec![
            view_at(Isometry3::identity(), [0, 0, 0]),
            view_at(Isometry3::identity(), [100, 200, 30]),
        ];
        let vertex = Vector3::new(0.0, 0.0, 2.0);

        let (colors, report) = colorize(&[vertex], &views, &ColoringParams::default());

        assert_eq!(colors[0], [50, 100, 15]);
        assert_eq!(report.samples, 2);
        assert!((report.mean_samples() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn a_vertex_behind_the_camera_is_not_coloured() {
        let views = vec![view_at(Isometry3::identity(), [7, 7, 7])];
        let vertex = Vector3::new(0.0, 0.0, -2.0);

        let (colors, report) = colorize(&[vertex], &views, &ColoringParams::default());

        assert_eq!(colors[0], ColoringParams::default().fallback);
        assert_eq!(report.unobserved, 1);
    }
}
