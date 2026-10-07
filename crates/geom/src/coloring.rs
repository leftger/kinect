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
//! Registration does not fill every depth pixel. Where it wrote nothing the
//! colour buffer stays zero, which is also a real black sample, so each view
//! carries the mask from registration and a sample is taken only where that mask
//! says the pixel was copied.
//!
//! # Visibility, then a score
//!
//! Sampling the wrong colour is easy: a vertex on a far wall, seen through a
//! doorway from some other pose, would be painted with whatever is in front of
//! it. A sample is kept only when the view's own depth measurement agrees with
//! how far away the vertex is, the pixel was actually registered, the sample
//! clears the image margin, and a known normal faces the camera. That is the
//! same occlusion test the projective odometry uses, applied to colour.
//!
//! Samples that survive are scored, and every mode uses that same score:
//!
//! ```text
//! score = incidence × centrality × proximity × agreement × tracking
//! ```
//!
//! `incidence` is how squarely the camera looks at the surface (`n · view`, or 1
//! when the vertex has no normal). `centrality` is the distance to the nearest
//! image border, divided by half the shorter side, so the middle of the frame
//! outranks the edge. `proximity` is `1 / (1 + distance)`. `agreement` is how
//! much of the depth tolerance is still left. `tracking` is the inlier ratio.
//!
//! `best` keeps the highest score. `blend` mixes the top three in linear light,
//! weighted by those scores. `average` is the old painter: an equal mean of the
//! stored bytes, still gated by the same visibility test, and it does not apply
//! exposure correction. Those three modes are the per-vertex PLY painter. A
//! texture atlas uses this visibility test and then keeps one source view per
//! triangle; see [`crate::texturing`].
//!
//! # Exposure
//!
//! The colour camera's exposure, gain and gamma ride along on the view for
//! diagnostics. They are not treated as a photometric model — the Kinect footer
//! does not document units that would make that honest. Instead each view gets
//! one scalar, estimated from vertices that view shares with the
//! best-observed view, and the scalar is applied in linear light. A correction
//! needs several consistent overlaps, ignores near-black samples (their ratio is
//! noise), and is clamped so a bad estimate cannot amplify a dark pixel by more
//! than the configured range.

use std::cmp::Ordering;

use nalgebra::Vector3;

use crate::{transform_point, Intrinsics};

/// One frame's colour, registered into the depth camera's grid.
///
/// On the Kinect v2 depth grid (512×424) the buffers are RGB, one `f32` depth
/// per pixel, and a validity byte per pixel: about 1.66 MiB. Six hundred kept
/// views are about 1 GiB. Loop closure retains the depth frames as well, about
/// 850 KB each, so a colour scan pays both.
pub struct ColorView {
    /// RGB, `width * height * 3`, row-major. Treated as sRGB.
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
    /// One byte per depth pixel. Non-zero where `color` was copied from the
    /// colour camera; zero where registration left the pixel untouched.
    pub valid: Vec<u8>,
    /// Colour-camera exposure, gain and gamma for the frame. Kept so a caller
    /// can diagnose the capture. [`colorize`] does not read them; exposure
    /// correction is estimated from overlapping geometry instead.
    pub exposure: f32,
    pub gain: f32,
    pub gamma: f32,
    /// Arrival index of the depth frame this colour belongs to. That is the
    /// index into the trajectory, including frames that were not fused, so a
    /// corrected pose can be written back after loop closure.
    pub frame_index: usize,
    /// Inlier ratio from tracking this frame. `1.0` for the first frame, which
    /// is fused without a predecessor.
    pub tracking_quality: f32,
}

/// Which visible samples become the vertex colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// The single highest-scoring sample, after its exposure correction.
    #[default]
    Best,
    /// The top three scores, weighted together in linear light, after each
    /// sample's exposure correction.
    Blend,
    /// Equal mean of every visible sample, in the stored byte values.
    ///
    /// This is the original painter. Ranking and exposure are ignored so a
    /// caller can still ask for that average; the visibility test is not.
    Average,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColoringParams {
    /// How far a vertex may disagree with a view's measured depth and still count
    /// as visible from it, in metres. Too tight and nothing is ever visible; too
    /// loose and colour bleeds through walls.
    pub depth_tolerance: f32,
    /// Colour given to vertices no view could see.
    pub fallback: [u8; 3],
    pub mode: ColorMode,
    /// Samples closer to the image border than this, in pixels, are not visible.
    /// The score still prefers the centre when this is zero.
    pub image_margin: f32,
    /// Lower clamp for a view's linear-light exposure scale.
    pub exposure_min: f32,
    /// Upper clamp for a view's linear-light exposure scale. This is what stops
    /// a dark, noisy overlap from being amplified without limit.
    pub exposure_max: f32,
}

impl Default for ColoringParams {
    fn default() -> Self {
        Self {
            // Two centimetres. The old painter used five, which let colour
            // through thin structure; two is about the depth noise at typical
            // range and still tolerates a vertex that sits between voxels.
            depth_tolerance: 0.02,
            fallback: [128, 128, 128],
            mode: ColorMode::Best,
            image_margin: 0.0,
            exposure_min: 0.5,
            exposure_max: 2.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColoringReport {
    pub vertices: usize,
    /// Vertices no view could see, left at the fallback colour.
    pub unobserved: usize,
    /// `(vertex, view)` pairs that passed the visibility test.
    pub samples: usize,
    /// Linear-light multiplier for each view, in view order. `1` means that
    /// view was left alone. [`ColorMode::Average`] reports the same estimate
    /// and then does not apply it.
    pub exposure_scales: Vec<f32>,
}

impl ColoringReport {
    /// Vertices that received a sample. The rest were left at the fallback.
    pub fn painted(&self) -> usize {
        self.vertices.saturating_sub(self.unobserved)
    }

    /// Share of vertices that were painted. `0` when the mesh has none.
    pub fn coverage(&self) -> f32 {
        if self.vertices == 0 {
            return 0.0;
        }
        self.painted() as f32 / self.vertices as f32
    }

    /// Mean number of views that contributed to each coloured vertex.
    pub fn mean_samples(&self) -> f32 {
        let coloured = self.painted();
        if coloured == 0 {
            return 0.0;
        }
        self.samples as f32 / coloured as f32
    }
}

/// A sample that passed the visibility test, before exposure correction.
///
/// Texturing assigns whole faces with these same samples, so the fields are
/// visible inside the crate. The score is the one described on [`colorize`].
pub(crate) struct Observation {
    pub(crate) view: usize,
    pub(crate) rgb: [u8; 3],
    pub(crate) score: f32,
    /// Rec. 709 luminance in linear light, before the exposure scale.
    pub(crate) luminance: f32,
    /// Continuous pixel coordinates of the projection, before they are snapped
    /// to the depth pixel that was tested.
    pub(crate) pixel: (f32, f32),
}

/// Overlaps below this are not an exposure estimate, they are an anecdote.
pub(crate) const MIN_EXPOSURE_OVERLAPS: usize = 8;
/// Linear luminance under this is too close to the noise floor to form a ratio.
const EXPOSURE_LUMA_FLOOR: f32 = 0.02;
/// `q3 / q1` above this means the two views do not differ by one exposure.
const EXPOSURE_MAX_SPREAD: f32 = 1.35;
/// Ratios inside this band of 1 are left alone, so quantization is not "corrected".
const EXPOSURE_DEADZONE: f32 = 0.05;

/// Paint `vertices` from `views`.
///
/// `normals` is parallel to `vertices`. A missing or zero normal does not cull
/// the vertex: incidence is treated as neutral, because an extracted mesh can
/// contain a vertex that no finite face claimed. A non-zero normal that faces
/// away from the camera does cull.
pub fn colorize(
    vertices: &[Vector3<f32>],
    normals: &[Vector3<f32>],
    views: &[ColorView],
    params: &ColoringParams,
) -> (Vec<[u8; 3]>, ColoringReport) {
    let observed = collect_observations(vertices, normals, views, params);
    let exposure_scales = estimate_exposure(views.len(), &observed, params);

    let mut colors = Vec::with_capacity(vertices.len());
    let mut unobserved = 0;
    let mut samples = 0;

    for vertex_samples in &observed {
        samples += vertex_samples.len();
        if vertex_samples.is_empty() {
            unobserved += 1;
            colors.push(params.fallback);
        } else {
            colors.push(paint(vertex_samples, &exposure_scales, params));
        }
    }

    (
        colors,
        ColoringReport {
            vertices: vertices.len(),
            unobserved,
            samples,
            exposure_scales,
        },
    )
}

/// Every view that can see each vertex, in view order.
///
/// This is the shared input of the vertex painter and of exposure estimation.
/// Texturing looks corners up here so a face is visible only when the painter
/// would have accepted each of its corners.
pub(crate) fn collect_observations(
    vertices: &[Vector3<f32>],
    normals: &[Vector3<f32>],
    views: &[ColorView],
    params: &ColoringParams,
) -> Vec<Vec<Observation>> {
    let mut observed = Vec::with_capacity(vertices.len());

    for (index, vertex) in vertices.iter().enumerate() {
        let normal = normals.get(index).copied().unwrap_or_else(Vector3::zeros);
        let mut samples = Vec::new();
        for (view_index, view) in views.iter().enumerate() {
            if let Some(sample) = observe(view, view_index, vertex, &normal, params) {
                samples.push(sample);
            }
        }
        observed.push(samples);
    }

    observed
}

/// The visibility test and the score, together. `None` means this view must not
/// paint the vertex at all.
pub(crate) fn observe(
    view: &ColorView,
    view_index: usize,
    world: &Vector3<f32>,
    normal: &Vector3<f32>,
    params: &ColoringParams,
) -> Option<Observation> {
    if view.width == 0 || view.height == 0 {
        return None;
    }

    let camera = transform_point(&view.pose.inverse(), world);
    // Behind the camera, or so close the projection is meaningless.
    if !camera.z.is_finite() || camera.z <= 1e-3 {
        return None;
    }

    let (u, v) = view.intrinsics.project(&camera)?;

    let margin = params.image_margin.max(0.0);
    if u < margin
        || v < margin
        || u >= view.width as f32 - margin
        || v >= view.height as f32 - margin
    {
        return None;
    }

    let x = u.floor();
    let y = v.floor();
    if x < 0.0 || y < 0.0 || x >= view.width as f32 || y >= view.height as f32 {
        return None;
    }

    let (x, y) = (x as usize, y as usize);
    let pixel = y * view.width + x;

    // Unregistered colour is stored as black, which is also a real sample. The
    // mask is what says this pixel was copied from the colour camera.
    if view.valid.get(pixel).copied().unwrap_or(0) == 0 {
        return None;
    }

    // The occlusion test. If the view measured a surface much nearer or much
    // further than this vertex is, then this vertex is not what that pixel is
    // looking at.
    let measured = *view.depth.get(pixel)?;
    if !measured.is_finite() || measured <= 0.0 {
        return None;
    }
    let residual = (measured - camera.z).abs();
    if residual > params.depth_tolerance {
        return None;
    }

    let offset = pixel * 3;
    if offset + 2 >= view.color.len() {
        return None;
    }
    let rgb = [
        view.color[offset],
        view.color[offset + 1],
        view.color[offset + 2],
    ];

    let to_camera = view.pose.translation.vector - world;
    let distance = to_camera.norm();
    if !distance.is_finite() || distance <= 1e-4 {
        return None;
    }
    let direction = to_camera / distance;

    let normal_length = normal.norm();
    let incidence = if !normal_length.is_finite() || normal_length <= 1e-6 {
        // No orientation: do not invent a back-face, and do not punish the view.
        1.0
    } else {
        let cosine = (normal / normal_length).dot(&direction);
        if cosine <= 0.0 {
            return None;
        }
        cosine
    };

    let border = u
        .min(v)
        .min(view.width as f32 - u)
        .min(view.height as f32 - v)
        .max(0.0);
    let half_short_side = (view.width.min(view.height) as f32) * 0.5;
    let centrality = if half_short_side <= 0.0 {
        0.0
    } else {
        (border / half_short_side).clamp(0.0, 1.0)
    };

    let proximity = 1.0 / (1.0 + distance);
    let agreement = if params.depth_tolerance <= 1e-8 {
        1.0
    } else {
        (1.0 - residual / params.depth_tolerance).clamp(0.0, 1.0)
    };
    let tracking = if view.tracking_quality.is_finite() {
        view.tracking_quality.clamp(0.0, 1.0)
    } else {
        return None;
    };

    let score = incidence * centrality * proximity * agreement * tracking;
    if !score.is_finite() {
        return None;
    }

    let linear = [
        srgb_to_linear(rgb[0]),
        srgb_to_linear(rgb[1]),
        srgb_to_linear(rgb[2]),
    ];
    let luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];

    Some(Observation {
        view: view_index,
        rgb,
        score,
        luminance,
        pixel: (u, v),
    })
}

/// Scalar per view that makes overlapping observations of the same vertices
/// agree, pulled toward 1 wherever the evidence is thin or inconsistent.
///
/// The reference is the view with the most samples above the luminance floor
/// (lowest index wins a tie) and its scale stays 1. Every other view is the
/// median ratio of reference luminance over its own, on vertices both can see.
/// One bad vertex must not move the scale, and neither must a pair of views
/// that simply saw different materials, so a wide interquartile spread is
/// refused outright.
pub(crate) fn estimate_exposure(
    view_count: usize,
    observed: &[Vec<Observation>],
    params: &ColoringParams,
) -> Vec<f32> {
    let mut counts = vec![0usize; view_count];
    for samples in observed {
        for sample in samples {
            if sample.luminance >= EXPOSURE_LUMA_FLOOR {
                counts[sample.view] += 1;
            }
        }
    }

    let mut reference = 0;
    let mut most = 0usize;
    for (index, count) in counts.iter().enumerate() {
        if *count > most {
            most = *count;
            reference = index;
        }
    }

    let mut ratios = vec![Vec::new(); view_count];
    for samples in observed {
        let Some(reference_luma) = samples
            .iter()
            .find(|sample| sample.view == reference)
            .map(|sample| sample.luminance)
        else {
            continue;
        };
        if reference_luma < EXPOSURE_LUMA_FLOOR {
            continue;
        }
        for sample in samples {
            if sample.view == reference || sample.luminance < EXPOSURE_LUMA_FLOOR {
                continue;
            }
            ratios[sample.view].push(reference_luma / sample.luminance);
        }
    }

    let lo = params.exposure_min.min(params.exposure_max);
    let hi = params.exposure_min.max(params.exposure_max);
    let mut scales = vec![1.0f32; view_count];

    for view in 0..view_count {
        if view == reference {
            continue;
        }
        let ratio = &mut ratios[view];
        if ratio.len() < MIN_EXPOSURE_OVERLAPS {
            continue;
        }
        ratio.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        let median = ratio[ratio.len() / 2];
        let lower_quartile = ratio[ratio.len() / 4];
        let upper_quartile = ratio[ratio.len() * 3 / 4];
        if !(lower_quartile > 0.0
            && median.is_finite()
            && upper_quartile / lower_quartile <= EXPOSURE_MAX_SPREAD)
        {
            continue;
        }
        if (median - 1.0).abs() < EXPOSURE_DEADZONE {
            continue;
        }
        scales[view] = median.clamp(lo, hi);
    }

    scales
}

fn paint(samples: &[Observation], scales: &[f32], params: &ColoringParams) -> [u8; 3] {
    match params.mode {
        ColorMode::Average => average_bytes(samples),
        ColorMode::Best => {
            let best = samples
                .iter()
                .max_by(|left, right| rank(left, right))
                .expect("paint is only called with a visible sample");
            encode(scale_linear(best.rgb, scale_of(scales, best.view)))
        }
        ColorMode::Blend => blend_top(samples, scales, 3),
    }
}

/// Higher score first. An equal score keeps the earlier view, so two identical
/// observations do not flicker with hash order.
fn rank(left: &Observation, right: &Observation) -> Ordering {
    match left.score.partial_cmp(&right.score) {
        Some(Ordering::Equal) | None => right.view.cmp(&left.view),
        Some(order) => order,
    }
}

fn average_bytes(samples: &[Observation]) -> [u8; 3] {
    let mut sum = [0.0f32; 3];
    for sample in samples {
        sum[0] += sample.rgb[0] as f32;
        sum[1] += sample.rgb[1] as f32;
        sum[2] += sample.rgb[2] as f32;
    }
    let count = samples.len() as f32;
    [
        (sum[0] / count).round().clamp(0.0, 255.0) as u8,
        (sum[1] / count).round().clamp(0.0, 255.0) as u8,
        (sum[2] / count).round().clamp(0.0, 255.0) as u8,
    ]
}

fn blend_top(samples: &[Observation], scales: &[f32], limit: usize) -> [u8; 3] {
    let mut ranked: Vec<&Observation> = samples.iter().collect();
    ranked.sort_by(|left, right| rank(right, left));
    let chosen = &ranked[..ranked.len().min(limit)];

    let weight_sum: f32 = chosen.iter().map(|sample| sample.score).sum();
    let mut acc = [0.0f32; 3];

    if weight_sum <= 1e-8 {
        // Every survivor scored nothing — tracking was zero — but they did
        // pass visibility, so they still have a colour. Weight them equally.
        let share = 1.0 / chosen.len() as f32;
        for sample in chosen {
            let linear = scale_linear(sample.rgb, scale_of(scales, sample.view));
            for channel in 0..3 {
                acc[channel] += share * linear[channel];
            }
        }
    } else {
        for sample in chosen {
            let linear = scale_linear(sample.rgb, scale_of(scales, sample.view));
            let weight = sample.score / weight_sum;
            for channel in 0..3 {
                acc[channel] += weight * linear[channel];
            }
        }
    }

    encode(acc)
}

fn scale_of(scales: &[f32], view: usize) -> f32 {
    scales.get(view).copied().unwrap_or(1.0)
}

pub(crate) fn scale_linear(rgb: [u8; 3], scale: f32) -> [f32; 3] {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    [
        srgb_to_linear(rgb[0]) * scale,
        srgb_to_linear(rgb[1]) * scale,
        srgb_to_linear(rgb[2]) * scale,
    ]
}

pub(crate) fn encode(linear: [f32; 3]) -> [u8; 3] {
    [
        linear_to_srgb(linear[0]),
        linear_to_srgb(linear[1]),
        linear_to_srgb(linear[2]),
    ]
}

pub(crate) fn srgb_to_linear(channel: u8) -> f32 {
    let encoded = channel as f32 / 255.0;
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn linear_to_srgb(linear: f32) -> u8 {
    let linear = linear.clamp(0.0, 1.0);
    let encoded = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform_point;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    const SIDE: usize = 64;

    #[test]
    fn a_kinect_colour_view_matches_the_documented_size() {
        // 512×424 depth grid: RGB, f32 depth, and one validity byte.
        let pixels = 512 * 424;
        let bytes = pixels * 3 + pixels * 4 + pixels;
        let mib = bytes as f64 / (1024.0 * 1024.0);
        assert!(
            (mib - 1.66).abs() < 0.01,
            "{bytes} bytes is {mib:.3} MiB, documented as ~1.66"
        );
        let gib = bytes as f64 * 600.0 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            (gib - 1.0).abs() < 0.04,
            "600 views are {gib:.3} GiB, documented as ~1"
        );
        // Loop closure keeps the depth buffer on its own, about 850 KB.
        let depth_kib = (pixels * 4) as f64 / 1024.0;
        assert!(
            (depth_kib - 850.0).abs() < 20.0,
            "retained depth is {depth_kib:.1} KiB"
        );
    }

    fn intrinsics() -> Intrinsics {
        Intrinsics {
            fx: 60.0,
            fy: 60.0,
            cx: SIDE as f32 / 2.0,
            cy: SIDE as f32 / 2.0,
        }
    }

    /// A view that sees the plane z = 2 from the origin, filled with one colour.
    fn view_at(pose: Isometry3<f32>, fill: [u8; 3]) -> ColorView {
        ColorView {
            color: fill.iter().copied().cycle().take(SIDE * SIDE * 3).collect(),
            width: SIDE,
            height: SIDE,
            depth: vec![2.0; SIDE * SIDE],
            pose,
            intrinsics: intrinsics(),
            valid: vec![1; SIDE * SIDE],
            exposure: 1.0,
            gain: 1.0,
            gamma: 1.0,
            frame_index: 0,
            tracking_quality: 1.0,
        }
    }

    fn at(x: f32, y: f32, z: f32) -> Isometry3<f32> {
        Isometry3::from_parts(Translation3::new(x, y, z), UnitQuaternion::identity())
    }

    fn paint(
        vertices: &[Vector3<f32>],
        normals: &[Vector3<f32>],
        views: &[ColorView],
        params: &ColoringParams,
    ) -> (Vec<[u8; 3]>, ColoringReport) {
        colorize(vertices, normals, views, params)
    }

    fn vertex() -> Vector3<f32> {
        Vector3::new(0.0, 0.0, 2.0)
    }

    /// Eight copies of the on-axis point. Exposure will not move on fewer.
    fn repeated_vertex() -> Vec<Vector3<f32>> {
        vec![vertex(); MIN_EXPOSURE_OVERLAPS]
    }

    fn facing_camera() -> Vector3<f32> {
        Vector3::new(0.0, 0.0, -1.0)
    }

    /// Half the linear value of each channel, re-encoded. Quantization means the
    /// ratio is only approximately two; tests assert against the scale that was
    /// actually estimated.
    fn half_linear(rgb: [u8; 3]) -> [u8; 3] {
        [
            linear_to_srgb(srgb_to_linear(rgb[0]) * 0.5),
            linear_to_srgb(srgb_to_linear(rgb[1]) * 0.5),
            linear_to_srgb(srgb_to_linear(rgb[2]) * 0.5),
        ]
    }

    fn luma(rgb: [u8; 3]) -> f32 {
        0.2126 * srgb_to_linear(rgb[0])
            + 0.7152 * srgb_to_linear(rgb[1])
            + 0.0722 * srgb_to_linear(rgb[2])
    }

    #[test]
    fn defaults_are_best_mode_and_two_centimetres() {
        let params = ColoringParams::default();
        assert_eq!(params.mode, ColorMode::Best);
        assert!((params.depth_tolerance - 0.02).abs() < 1e-6);
        assert!((params.exposure_min - 0.5).abs() < 1e-6);
        assert!((params.exposure_max - 2.0).abs() < 1e-6);
    }

    #[test]
    fn srgb_roundtrip_holds_for_a_mid_grey() {
        for channel in [0u8, 10, 40, 128, 180, 255] {
            let back = linear_to_srgb(srgb_to_linear(channel));
            assert!(
                (back as i32 - channel as i32).abs() <= 1,
                "{channel} came back as {back}"
            );
        }
    }

    #[test]
    fn a_vertex_takes_the_colour_of_the_view_that_saw_it() {
        let views = vec![view_at(Isometry3::identity(), [10, 20, 30])];

        let (colors, report) = paint(&[vertex()], &[], &views, &ColoringParams::default());

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

        let (colors, report) = paint(&[vertex()], &[], &views, &ColoringParams::default());

        assert_eq!(colors[0], [0, 255, 0], "the occluded view leaked colour");
        assert_eq!(report.samples, 1);
    }

    #[test]
    fn average_mode_uses_the_same_visibility_test() {
        let mut wrong = view_at(Isometry3::identity(), [255, 0, 0]);
        wrong.depth = vec![0.5; SIDE * SIDE];
        let views = vec![view_at(Isometry3::identity(), [0, 0, 0]), wrong];
        let params = ColoringParams {
            mode: ColorMode::Average,
            ..ColoringParams::default()
        };

        let (colors, report) = paint(&[vertex()], &[], &views, &params);

        assert_eq!(colors[0], [0, 0, 0]);
        assert_eq!(report.samples, 1);
    }

    #[test]
    fn a_vertex_no_view_saw_keeps_the_fallback() {
        let param = ColoringParams {
            fallback: [1, 2, 3],
            ..ColoringParams::default()
        };

        let (colors, report) = paint(
            &[Vector3::new(50.0, 50.0, 2.0)],
            &[],
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
        let params = ColoringParams {
            mode: ColorMode::Average,
            ..ColoringParams::default()
        };

        let (colors, report) = paint(&[vertex()], &[], &views, &params);

        assert_eq!(colors[0], [50, 100, 15]);
        assert_eq!(report.samples, 2);
        assert!((report.mean_samples() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn best_is_the_default_and_does_not_average() {
        // Same inputs as the average test. Tied scores keep the earlier view,
        // which is black, not the mean.
        let views = vec![
            view_at(Isometry3::identity(), [0, 0, 0]),
            view_at(Isometry3::identity(), [100, 200, 30]),
        ];

        let (colors, _) = paint(&[vertex()], &[], &views, &ColoringParams::default());

        assert_eq!(colors[0], [0, 0, 0]);
    }

    #[test]
    fn an_unregistered_pixel_does_not_paint_the_vertex() {
        // Depth agrees and the stored colour is bright red, but registration
        // never wrote that pixel. Without the mask the vertex would come out red.
        let mut view = view_at(Isometry3::identity(), [255, 0, 0]);
        let pixel = (SIDE / 2) * SIDE + SIDE / 2;
        view.valid[pixel] = 0;

        let (colors, report) = paint(&[vertex()], &[], &[view], &ColoringParams::default());

        assert_eq!(colors[0], ColoringParams::default().fallback);
        assert_eq!(report.unobserved, 1);
        assert_eq!(report.samples, 0);
    }

    #[test]
    fn a_vertex_behind_the_camera_is_not_coloured() {
        let views = vec![view_at(Isometry3::identity(), [7, 7, 7])];

        let (colors, report) = paint(
            &[Vector3::new(0.0, 0.0, -2.0)],
            &[],
            &views,
            &ColoringParams::default(),
        );

        assert_eq!(colors[0], ColoringParams::default().fallback);
        assert_eq!(report.unobserved, 1);
    }

    #[test]
    fn a_short_depth_buffer_is_unobserved_rather_than_a_panic() {
        let mut view = view_at(Isometry3::identity(), [7, 7, 7]);
        view.depth.clear();

        let (colors, report) = paint(&[vertex()], &[], &[view], &ColoringParams::default());

        assert_eq!(colors[0], ColoringParams::default().fallback);
        assert_eq!(report.samples, 0);
    }

    #[test]
    fn the_default_tolerance_is_two_centimetres() {
        let mut just_inside = view_at(Isometry3::identity(), [4, 5, 6]);
        just_inside.depth = vec![2.015; SIDE * SIDE];
        let mut just_outside = view_at(Isometry3::identity(), [4, 5, 6]);
        just_outside.depth = vec![2.03; SIDE * SIDE];
        let mut on_the_boundary = view_at(Isometry3::identity(), [4, 5, 6]);
        on_the_boundary.depth = vec![2.02; SIDE * SIDE];

        let (inside, _) = paint(&[vertex()], &[], &[just_inside], &ColoringParams::default());
        let (outside, outside_report) = paint(
            &[vertex()],
            &[],
            &[just_outside],
            &ColoringParams::default(),
        );
        let (boundary, _) = paint(
            &[vertex()],
            &[],
            &[on_the_boundary],
            &ColoringParams::default(),
        );

        assert_eq!(inside[0], [4, 5, 6]);
        assert_eq!(
            boundary[0],
            [4, 5, 6],
            "a residual equal to the tolerance still counts"
        );
        assert_eq!(outside[0], ColoringParams::default().fallback);
        assert_eq!(outside_report.samples, 0);

        // The old five-centimetre tolerance would have accepted the 3 cm miss.
        let loose = ColoringParams {
            depth_tolerance: 0.05,
            ..ColoringParams::default()
        };
        let mut previously_accepted = view_at(Isometry3::identity(), [4, 5, 6]);
        previously_accepted.depth = vec![2.03; SIDE * SIDE];
        let (loose_colors, _) = paint(&[vertex()], &[], &[previously_accepted], &loose);
        assert_eq!(loose_colors[0], [4, 5, 6]);
    }

    #[test]
    fn the_image_margin_drops_samples_near_the_border() {
        // An on-axis point projects to the principal point, so moving cx moves
        // the sample and nothing else about the ray.
        let mut edge = view_at(Isometry3::identity(), [9, 9, 9]);
        edge.intrinsics.cx = 5.0;
        let params = ColoringParams {
            image_margin: 10.0,
            ..ColoringParams::default()
        };

        let (rejected, report) = paint(&[vertex()], &[], &[edge], &params);
        assert_eq!(rejected[0], params.fallback);
        assert_eq!(report.samples, 0);

        let mut on_the_margin = view_at(Isometry3::identity(), [9, 9, 9]);
        on_the_margin.intrinsics.cx = 10.0;
        let (kept, _) = paint(&[vertex()], &[], &[on_the_margin], &params);
        assert_eq!(kept[0], [9, 9, 9]);
    }

    #[test]
    fn a_known_back_face_is_invisible_and_an_unknown_normal_is_not() {
        let views = vec![view_at(Isometry3::identity(), [3, 4, 5])];
        let away = Vector3::new(0.0, 0.0, 1.0);

        let (back, report) = paint(&[vertex()], &[away], &views, &ColoringParams::default());
        assert_eq!(back[0], ColoringParams::default().fallback);
        assert_eq!(report.samples, 0);

        let (unknown, _) = paint(
            &[vertex()],
            &[Vector3::zeros()],
            &views,
            &ColoringParams::default(),
        );
        assert_eq!(unknown[0], [3, 4, 5]);

        let (toward, _) = paint(
            &[vertex()],
            &[facing_camera()],
            &views,
            &ColoringParams::default(),
        );
        assert_eq!(toward[0], [3, 4, 5]);
    }

    /// A camera placed `distance` metres from `point` along `direction` (unit),
    /// with the principal point shifted so `point` still lands on the image
    /// centre. That holds centrality fixed while incidence and distance change.
    fn view_from(
        point: Vector3<f32>,
        direction_from_point: Vector3<f32>,
        distance: f32,
        fill: [u8; 3],
    ) -> ColorView {
        let direction = direction_from_point.normalize();
        let camera_at = point + direction * distance;
        let pose = at(camera_at.x, camera_at.y, camera_at.z);
        let camera = transform_point(&pose.inverse(), &point);
        assert!(camera.z > 0.05, "the point must sit in front of the camera");

        let mut view = view_at(pose, fill);
        view.intrinsics.cx = SIDE as f32 / 2.0 - camera.x / camera.z * view.intrinsics.fx;
        view.intrinsics.cy = SIDE as f32 / 2.0 - camera.y / camera.z * view.intrinsics.fy;
        view.depth = vec![camera.z; SIDE * SIDE];
        view
    }

    #[test]
    fn a_more_frontal_view_outranks_a_grazing_one() {
        let point = vertex();
        let frontal = view_from(point, facing_camera(), 2.0, [10, 20, 30]);
        let grazing = view_from(point, Vector3::new(0.766, 0.0, -0.643), 2.0, [200, 0, 0]);

        let (colors, report) = paint(
            &[point],
            &[facing_camera()],
            &[grazing, frontal],
            &ColoringParams::default(),
        );

        assert_eq!(colors[0], [10, 20, 30]);
        assert_eq!(
            report.samples, 2,
            "the grazing view should still be visible"
        );
    }

    #[test]
    fn a_sample_nearer_the_image_centre_outranks_one_near_the_edge() {
        let mut edge = view_at(Isometry3::identity(), [200, 0, 0]);
        edge.intrinsics.cx = 8.0;
        let centre = view_at(Isometry3::identity(), [10, 20, 30]);

        let (colors, _) = paint(
            &[vertex()],
            &[facing_camera()],
            &[edge, centre],
            &ColoringParams::default(),
        );

        assert_eq!(colors[0], [10, 20, 30]);
    }

    #[test]
    fn a_closer_camera_outranks_a_farther_one() {
        let point = vertex();
        let near = view_from(point, facing_camera(), 2.0, [10, 20, 30]);
        let far = view_from(point, facing_camera(), 4.0, [200, 0, 0]);

        let (colors, report) = paint(
            &[point],
            &[facing_camera()],
            &[far, near],
            &ColoringParams::default(),
        );

        assert_eq!(colors[0], [10, 20, 30]);
        assert_eq!(report.samples, 2);
    }

    #[test]
    fn a_tighter_depth_residual_outranks_a_looser_one() {
        let mut exact = view_at(Isometry3::identity(), [10, 20, 30]);
        exact.depth = vec![2.0; SIDE * SIDE];
        let mut loose = view_at(Isometry3::identity(), [200, 0, 0]);
        loose.depth = vec![2.015; SIDE * SIDE];

        let (colors, report) = paint(
            &[vertex()],
            &[facing_camera()],
            &[loose, exact],
            &ColoringParams::default(),
        );

        assert_eq!(colors[0], [10, 20, 30]);
        assert_eq!(report.samples, 2);
    }

    #[test]
    fn higher_tracking_confidence_outranks_a_weaker_track() {
        let mut weak = view_at(Isometry3::identity(), [200, 0, 0]);
        weak.tracking_quality = 0.2;
        let mut strong = view_at(Isometry3::identity(), [10, 20, 30]);
        strong.tracking_quality = 0.9;

        let (colors, _) = paint(
            &[vertex()],
            &[facing_camera()],
            &[weak, strong],
            &ColoringParams::default(),
        );

        assert_eq!(colors[0], [10, 20, 30]);
    }

    #[test]
    fn a_tied_score_keeps_the_earlier_view() {
        let views = vec![
            view_at(Isometry3::identity(), [1, 2, 3]),
            view_at(Isometry3::identity(), [9, 9, 9]),
        ];

        let (colors, _) = paint(&[vertex()], &[], &views, &ColoringParams::default());

        assert_eq!(colors[0], [1, 2, 3]);
    }

    #[test]
    fn the_only_visible_sample_is_kept_even_when_its_score_is_zero() {
        let mut view = view_at(Isometry3::identity(), [6, 7, 8]);
        view.tracking_quality = 0.0;

        let (colors, report) = paint(&[vertex()], &[], &[view], &ColoringParams::default());

        assert_eq!(colors[0], [6, 7, 8]);
        assert_eq!(report.samples, 1);
    }

    #[test]
    fn blend_mixes_the_top_three_in_linear_light_by_score() {
        // Colours stay inside the linear piece of the sRGB curve, so a weighted
        // mean in linear light is the same weighted mean of the bytes. A fourth
        // view would move blue if it were included.
        let mut views = Vec::new();
        for (quality, fill) in [
            (1.0, [10, 0, 0]),
            (0.5, [0, 10, 0]),
            (0.25, [0, 0, 10]),
            (0.05, [10, 10, 10]),
        ] {
            let mut view = view_at(Isometry3::identity(), fill);
            view.tracking_quality = quality;
            views.push(view);
        }
        let params = ColoringParams {
            mode: ColorMode::Blend,
            ..ColoringParams::default()
        };

        let (colors, report) = paint(&[vertex()], &[], &views, &params);

        let weight = 1.0 + 0.5 + 0.25;
        let expected = [
            ((10.0f32) / weight).round() as u8,
            ((5.0f32) / weight).round() as u8,
            ((2.5f32) / weight).round() as u8,
        ];
        assert_eq!(colors[0], expected);
        assert_eq!(
            report.samples, 4,
            "the fourth view is visible, just not blended"
        );
        assert_ne!(
            colors[0][2], 2,
            "including the fourth view would have raised blue"
        );
    }

    #[test]
    fn blend_of_black_and_white_is_not_the_encoded_midpoint() {
        let views = vec![
            view_at(Isometry3::identity(), [0, 0, 0]),
            view_at(Isometry3::identity(), [255, 255, 255]),
        ];
        let params = ColoringParams {
            mode: ColorMode::Blend,
            ..ColoringParams::default()
        };

        let (colors, _) = paint(&[vertex()], &[], &views, &params);

        let linear_mid = linear_to_srgb(0.5);
        assert_eq!(colors[0], [linear_mid; 3]);
        assert_ne!(
            colors[0][0], 128,
            "that would be a byte average, not linear light"
        );
        assert!(linear_mid > 180);
    }

    #[test]
    fn a_zero_score_does_not_dilute_a_blend_that_has_a_real_weight() {
        let mut tracked = view_at(Isometry3::identity(), [10, 0, 0]);
        tracked.tracking_quality = 1.0;
        let mut untracked = view_at(Isometry3::identity(), [0, 10, 0]);
        untracked.tracking_quality = 0.0;
        let params = ColoringParams {
            mode: ColorMode::Blend,
            ..ColoringParams::default()
        };

        let (colors, _) = paint(&[vertex()], &[], &[tracked, untracked], &params);

        assert_eq!(colors[0], [10, 0, 0]);
    }

    #[test]
    fn exposure_brightens_a_consistently_darker_view_in_linear_light() {
        let reference_rgb = [200, 160, 80];
        let darker = half_linear(reference_rgb);
        let mut reference = view_at(Isometry3::identity(), reference_rgb);
        reference.tracking_quality = 0.3;
        // Absurd footer values. If these were the photometric model the scale
        // would not be a plain luminance ratio.
        reference.exposure = 0.001;
        reference.gain = 1.0;
        reference.gamma = 1.0;
        let mut dark_view = view_at(Isometry3::identity(), darker);
        dark_view.tracking_quality = 1.0;
        dark_view.exposure = 80.0;
        dark_view.gain = 16.0;
        dark_view.gamma = 2.2;

        let (colors, report) = paint(
            &repeated_vertex(),
            &[],
            &[reference, dark_view],
            &ColoringParams::default(),
        );

        let scale = report.exposure_scales[1];
        assert!((report.exposure_scales[0] - 1.0).abs() < 1e-6);
        assert!(scale > 1.5 && scale <= 2.0, "scale {scale}");
        let expected = encode(scale_linear(darker, scale));
        assert_eq!(colors[0], expected);
        assert_ne!(
            expected[0], expected[1],
            "a scalar must not wash the tint out"
        );
        // The footer disagreed violently and the pixels only disagreed by a
        // factor of two. The estimate followed the pixels.
        assert!(scale < 4.0);
    }

    #[test]
    fn exposure_scales_clamp_instead_of_amplifying_a_huge_ratio() {
        let bright = [220, 220, 220];
        let barely = (0u8..=255)
            .find(|&channel| srgb_to_linear(channel) >= EXPOSURE_LUMA_FLOOR)
            .expect("some byte clears the floor");
        let dark = [barely; 3];
        assert!(
            luma(bright) / luma(dark) > 2.0,
            "the fixture must exceed the clamp"
        );

        let mut reference = view_at(Isometry3::identity(), bright);
        reference.tracking_quality = 0.2;
        let mut dark_view = view_at(Isometry3::identity(), dark);
        dark_view.tracking_quality = 1.0;

        let (colors, report) = paint(
            &repeated_vertex(),
            &[],
            &[reference, dark_view],
            &ColoringParams::default(),
        );

        assert!((report.exposure_scales[1] - 2.0).abs() < 1e-4);
        assert_eq!(colors[0], encode(scale_linear(dark, 2.0)));
        assert!(
            colors[0][0] < 180,
            "the clamp must leave the sample darker than the reference"
        );

        let limited = ColoringParams {
            exposure_max: 1.25,
            ..ColoringParams::default()
        };
        let mut reference = view_at(Isometry3::identity(), bright);
        reference.tracking_quality = 0.2;
        let mut half = view_at(Isometry3::identity(), half_linear(bright));
        half.tracking_quality = 1.0;
        let (limited_colors, limited_report) =
            paint(&repeated_vertex(), &[], &[reference, half], &limited);
        assert!((limited_report.exposure_scales[1] - 1.25).abs() < 1e-4);
        assert_eq!(
            limited_colors[0],
            encode(scale_linear(half_linear(bright), 1.25))
        );
    }

    #[test]
    fn exposure_is_not_estimated_from_too_few_or_inconsistent_overlaps() {
        let bright = [200, 200, 200];
        let darker = half_linear(bright);
        let mut reference = view_at(Isometry3::identity(), bright);
        reference.tracking_quality = 0.2;
        let mut dark_view = view_at(Isometry3::identity(), darker);
        dark_view.tracking_quality = 1.0;

        let (colors, report) = paint(
            &vec![vertex(); MIN_EXPOSURE_OVERLAPS - 1],
            &[],
            &[reference, dark_view],
            &ColoringParams::default(),
        );
        assert!((report.exposure_scales[1] - 1.0).abs() < 1e-6);
        assert_eq!(colors[0], darker);

        // Eight vertices, but the ratio flips sign across them, so this is not
        // one exposure. The dark view still wins best; it must keep its raw bytes.
        let mut steady = view_at(Isometry3::identity(), bright);
        steady.tracking_quality = 0.2;
        let mut uneven = view_at(Isometry3::identity(), bright);
        uneven.tracking_quality = 1.0;
        let points = spread_vertices(MIN_EXPOSURE_OVERLAPS);
        for (index, point) in points.iter().enumerate() {
            let camera = transform_point(&uneven.pose.inverse(), point);
            let (u, v) = uneven.intrinsics.project(&camera).unwrap();
            let pixel = v.floor() as usize * SIDE + u.floor() as usize;
            let rgb = if index % 2 == 0 {
                darker
            } else {
                // Brighter than the reference, so the ratios straddle 1.
                encode([
                    srgb_to_linear(bright[0]) * 1.8,
                    srgb_to_linear(bright[1]) * 1.8,
                    srgb_to_linear(bright[2]) * 1.8,
                ])
            };
            uneven.color[pixel * 3..pixel * 3 + 3].copy_from_slice(&rgb);
        }

        let (colors, report) = paint(&points, &[], &[steady, uneven], &ColoringParams::default());
        assert!(
            (report.exposure_scales[1] - 1.0).abs() < 1e-6,
            "scale {}",
            report.exposure_scales[1]
        );
        assert_eq!(colors[0], darker);
    }

    #[test]
    fn a_small_brightness_difference_is_left_alone() {
        let bright = [200, 200, 200];
        let slight = encode([
            srgb_to_linear(bright[0]) / 1.03,
            srgb_to_linear(bright[1]) / 1.03,
            srgb_to_linear(bright[2]) / 1.03,
        ]);
        let ratio = luma(bright) / luma(slight);
        assert!(
            (ratio - 1.0).abs() < EXPOSURE_DEADZONE,
            "fixture ratio {ratio} is outside the deadzone"
        );

        let mut reference = view_at(Isometry3::identity(), bright);
        reference.tracking_quality = 0.2;
        let mut other = view_at(Isometry3::identity(), slight);
        other.tracking_quality = 1.0;

        let (colors, report) = paint(
            &repeated_vertex(),
            &[],
            &[reference, other],
            &ColoringParams::default(),
        );

        assert!((report.exposure_scales[1] - 1.0).abs() < 1e-6);
        assert_eq!(colors[0], slight);
    }

    #[test]
    fn near_black_overlaps_are_not_amplified() {
        let mut reference = view_at(Isometry3::identity(), [180, 180, 180]);
        reference.tracking_quality = 0.2;
        let mut black = view_at(Isometry3::identity(), [1, 1, 1]);
        black.tracking_quality = 1.0;
        assert!(luma([1, 1, 1]) < EXPOSURE_LUMA_FLOOR);

        let (colors, report) = paint(
            &repeated_vertex(),
            &[],
            &[reference, black],
            &ColoringParams::default(),
        );

        assert!((report.exposure_scales[1] - 1.0).abs() < 1e-6);
        assert_eq!(colors[0], [1, 1, 1]);
    }

    #[test]
    fn a_view_that_does_not_overlap_the_reference_is_not_corrected() {
        let front = spread_vertices(MIN_EXPOSURE_OVERLAPS);
        let behind = Vector3::new(0.0, 0.0, -2.0);
        let bright = [200, 180, 140];
        let darker = half_linear(bright);

        let mut reference = view_at(Isometry3::identity(), bright);
        reference.tracking_quality = 0.3;
        let mut overlapping = view_at(Isometry3::identity(), darker);
        overlapping.tracking_quality = 0.3;

        // Looking at the point behind the reference camera. Its depth matches
        // that point and not the front set, so it shares no vertex with the
        // reference and must stay at scale 1.
        let mut isolated = view_from(behind, Vector3::new(0.0, 0.0, -1.0), 2.0, [30, 10, 10]);
        isolated.tracking_quality = 1.0;

        let mut vertices = front;
        vertices.push(behind);
        let (colors, report) = paint(
            &vertices,
            &[],
            &[reference, overlapping, isolated],
            &ColoringParams::default(),
        );

        assert!(report.exposure_scales[1] > 1.5);
        assert!(
            (report.exposure_scales[2] - 1.0).abs() < 1e-6,
            "isolated scale {}",
            report.exposure_scales[2]
        );
        assert_eq!(*colors.last().unwrap(), [30, 10, 10]);
    }

    #[test]
    fn identical_pixels_are_not_corrected_just_because_the_footer_differs() {
        let mut first = view_at(Isometry3::identity(), [140, 130, 120]);
        first.exposure = 0.01;
        first.gain = 1.0;
        first.gamma = 1.0;
        let mut second = view_at(Isometry3::identity(), [140, 130, 120]);
        second.exposure = 0.9;
        second.gain = 8.0;
        second.gamma = 2.2;

        let (colors, report) = paint(
            &repeated_vertex(),
            &[],
            &[first, second],
            &ColoringParams::default(),
        );

        assert!(report
            .exposure_scales
            .iter()
            .all(|scale| (scale - 1.0).abs() < 1e-6));
        assert_eq!(colors[0], [140, 130, 120]);
    }

    #[test]
    fn average_mode_does_not_apply_the_exposure_it_estimated() {
        let bright = [200, 160, 80];
        let darker = half_linear(bright);
        let reference = view_at(Isometry3::identity(), bright);
        let dark_view = view_at(Isometry3::identity(), darker);
        let params = ColoringParams {
            mode: ColorMode::Average,
            ..ColoringParams::default()
        };

        let (colors, report) = paint(&repeated_vertex(), &[], &[reference, dark_view], &params);

        assert!(
            report.exposure_scales[1] > 1.5,
            "the estimate should still be reported"
        );
        let expected = [
            ((bright[0] as u16 + darker[0] as u16) as f32 / 2.0).round() as u8,
            ((bright[1] as u16 + darker[1] as u16) as f32 / 2.0).round() as u8,
            ((bright[2] as u16 + darker[2] as u16) as f32 / 2.0).round() as u8,
        ];
        assert_eq!(colors[0], expected);
        let corrected = encode(scale_linear(darker, report.exposure_scales[1]));
        let corrected_mean = [
            ((bright[0] as u16 + corrected[0] as u16) as f32 / 2.0).round() as u8,
            ((bright[1] as u16 + corrected[1] as u16) as f32 / 2.0).round() as u8,
            ((bright[2] as u16 + corrected[2] as u16) as f32 / 2.0).round() as u8,
        ];
        assert_ne!(
            colors[0], corrected_mean,
            "average must stay in the stored bytes"
        );
    }

    #[test]
    fn coverage_is_the_painted_share_and_samples_average_over_it() {
        let report = ColoringReport {
            vertices: 4,
            unobserved: 1,
            samples: 6,
            exposure_scales: Vec::new(),
        };
        assert_eq!(report.painted(), 3);
        assert!((report.coverage() - 0.75).abs() < 1e-6);
        assert!((report.mean_samples() - 2.0).abs() < 1e-6);

        let empty = ColoringReport {
            vertices: 0,
            unobserved: 0,
            samples: 0,
            exposure_scales: Vec::new(),
        };
        assert_eq!(empty.painted(), 0);
        assert_eq!(empty.coverage(), 0.0);
        assert_eq!(empty.mean_samples(), 0.0);
    }

    /// Points along x at z = 2 that all project inside the default view.
    fn spread_vertices(count: usize) -> Vec<Vector3<f32>> {
        (0..count)
            .map(|index| {
                let x = if count == 1 {
                    0.0
                } else {
                    -0.7 + 1.4 * index as f32 / (count - 1) as f32
                };
                Vector3::new(x, 0.0, 2.0)
            })
            .collect()
    }
}
