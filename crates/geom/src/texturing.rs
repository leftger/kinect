//! A texture atlas for a fused mesh, taken from the same views as [`colorize`].
//!
//! Per-vertex colour throws away everything between the samples. An atlas keeps
//! the pixels, but only from a view that really saw the face: each triangle is
//! given to the single view that passes the vertex painter's visibility test on
//! all three corners and on the triangle centroid, and has the best score. The
//! centroid check is what rejects a view whose corners are clear while something
//! nearer covers the middle of the face. The score is the minimum of the three
//! corner scores, so a view that merely grazes one corner loses to one that
//! sees the whole face, and a tie keeps the earlier view. Atlas assignment is
//! one source view per triangle. Per-vertex PLY colour is what `--color-mode`
//! (`best`, `blend`, `average`) selects.
//!
//! Charts are the bounding boxes of the faces assigned to one view, not the
//! whole frame. They are shelf-packed with a replicated gutter so a bilinear
//! lookup on the border repeats the chart instead of reading its neighbour.
//! The atlas is never larger than [`MAX_ATLAS_SIZE`]. When the native charts
//! would overflow that square they are downscaled together, gutters included
//! only while a pixel of content still fits.
//!
//! A vertex shared by two charts cannot carry both texture coordinates, so it
//! is duplicated at a view boundary. Faces no view could see all land on one
//! fallback texel. [`TexturedMesh::flip_x`] is applied after the UVs exist: it
//! mirrors positions, normals and winding, and leaves the atlas mapping alone.

use std::cmp::Ordering;
use std::collections::HashMap;

use nalgebra::Vector3;

use crate::coloring::{
    collect_observations, encode, estimate_exposure, observe, scale_linear, ColorView,
    ColoringParams, Observation,
};
use crate::mesh::Mesh;

/// Hard cap on both atlas dimensions. Larger requests are clamped to this.
pub const MAX_ATLAS_SIZE: usize = 8192;

/// Knobs for [`texture`]. Visibility and exposure come from [`ColoringParams`]
/// so a face is accepted under the same rules as a vertex.
#[derive(Clone, Debug, PartialEq)]
pub struct TexturingParams {
    pub coloring: ColoringParams,
    /// Replicated pixels around every chart, including the fallback texel.
    /// Shrunk only when the atlas cap would otherwise leave no content pixel.
    pub gutter: usize,
    /// Requested maximum width and height. Clamped to `1..=`[`MAX_ATLAS_SIZE`].
    pub max_atlas_size: usize,
}

impl Default for TexturingParams {
    fn default() -> Self {
        Self {
            coloring: ColoringParams::default(),
            gutter: 2,
            max_atlas_size: MAX_ATLAS_SIZE,
        }
    }
}

/// Indexed triangle mesh plus the atlas those UVs address.
///
/// `atlas` is RGB8, row-major, three bytes per pixel. UV `(0, 0)` is the
/// top-left pixel of that buffer and V grows downward, so
/// `y = floor(v * atlas_height)` is the row. Positions, normals and `indices`
/// are parallel to a normal indexed mesh: every three indices are one triangle.
#[derive(Clone, Debug, Default)]
pub struct TexturedMesh {
    pub positions: Vec<Vector3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub atlas: Vec<u8>,
    pub atlas_width: usize,
    pub atlas_height: usize,
}

impl TexturedMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Mirror X, as [`Mesh::flip_x`] does, after UVs have been built.
    ///
    /// Positions and normals take the reflection (a normal's X component flips
    /// with the inverse-transpose of the mirror). Winding swaps so the stored
    /// normal still agrees with the face. Texture coordinates stay put: the
    /// atlas was painted in the views' images, which were not mirrored.
    pub fn flip_x(&mut self) {
        for position in &mut self.positions {
            position.x = -position.x;
        }
        for normal in &mut self.normals {
            normal.x = -normal.x;
        }
        for triangle in self.indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
}

/// What [`texture`] did, in view order for the exposure scales.
#[derive(Clone, Debug)]
pub struct TexturingReport {
    pub triangles: usize,
    /// Triangles that failed the corner-and-centroid test and were mapped to
    /// the fallback texel.
    pub unobserved: usize,
    /// Packed charts, counting the fallback chart when any face needed it.
    pub charts: usize,
    pub atlas_width: usize,
    pub atlas_height: usize,
    /// True when any photographed chart was stored smaller than its crop.
    pub downscaled: bool,
    /// The same per-view scales [`colorize`](crate::coloring::colorize) would
    /// report for these vertices. They have already been applied to the atlas.
    pub exposure_scales: Vec<f32>,
}

impl TexturingReport {
    /// Triangles assigned to a photographed chart.
    pub fn painted(&self) -> usize {
        self.triangles.saturating_sub(self.unobserved)
    }

    /// Share of triangles that landed on a photographed chart. `0` when the
    /// mesh has none.
    pub fn coverage(&self) -> f32 {
        if self.triangles == 0 {
            return 0.0;
        }
        self.painted() as f32 / self.triangles as f32
    }
}

/// Build an atlas mesh. `mesh.colors` is ignored: colour comes from `views`.
pub fn texture(
    mesh: &Mesh,
    views: &[ColorView],
    params: &TexturingParams,
) -> (TexturedMesh, TexturingReport) {
    let limit = atlas_limit(params.max_atlas_size);
    let normals = mesh.vertex_normals();
    let observed = collect_observations(&mesh.vertices, &normals, views, &params.coloring);
    let exposure_scales = estimate_exposure(views.len(), &observed, &params.coloring);

    let mut assigned = Vec::with_capacity(mesh.triangles.len());
    let mut members = vec![Vec::new(); views.len()];
    let mut missed = Vec::new();

    for (face_index, triangle) in mesh.triangles.iter().enumerate() {
        if let Some(hit) = best_view(mesh, *triangle, &observed, views, &params.coloring) {
            members[hit.view].push(face_index);
            assigned.push(Some(hit));
        } else {
            missed.push(face_index);
            assigned.push(None);
        }
    }

    let unobserved = missed.len();
    let mut charts = Vec::new();
    let mut chart_of_view = vec![None; views.len()];

    for (view_index, faces) in members.iter().enumerate() {
        if faces.is_empty() {
            continue;
        }
        let mut pixels = Vec::with_capacity(faces.len() * 3);
        for &face in faces {
            if let Some(hit) = &assigned[face] {
                pixels.extend_from_slice(&hit.pixels);
            }
        }
        let (src_x0, src_y0, src_w, src_h) =
            crop_rect(&pixels, views[view_index].width, views[view_index].height);
        chart_of_view[view_index] = Some(charts.len());
        charts.push(Chart {
            view: Some(view_index),
            src_x0,
            src_y0,
            src_w,
            src_h,
            content_w: src_w,
            content_h: src_h,
            origin_x: 0,
            origin_y: 0,
            gutter: 0,
        });
    }

    let fallback_chart = if missed.is_empty() {
        None
    } else {
        let index = charts.len();
        charts.push(Chart {
            view: None,
            src_x0: 0,
            src_y0: 0,
            src_w: 1,
            src_h: 1,
            content_w: 1,
            content_h: 1,
            origin_x: 0,
            origin_y: 0,
            gutter: 0,
        });
        Some(index)
    };

    let downscaled = pack_charts(&mut charts, params.gutter, limit);
    let (atlas, atlas_width, atlas_height) =
        bake_atlas(&charts, views, &exposure_scales, params.coloring.fallback);

    let mut chart_of_face = Vec::with_capacity(assigned.len());
    for hit in &assigned {
        let chart = match hit {
            Some(hit) => chart_of_view[hit.view].expect("a used view has a chart"),
            None => fallback_chart.expect("an unobserved face has a fallback chart"),
        };
        chart_of_face.push(chart);
    }

    let mut textured = TexturedMesh {
        positions: Vec::new(),
        normals: Vec::new(),
        uvs: Vec::new(),
        indices: Vec::with_capacity(mesh.triangles.len() * 3),
        atlas,
        atlas_width,
        atlas_height,
    };
    emit_vertices(
        mesh,
        &normals,
        &assigned,
        &chart_of_face,
        &charts,
        &mut textured,
    );

    let report = TexturingReport {
        triangles: mesh.triangles.len(),
        unobserved,
        charts: charts.len(),
        atlas_width,
        atlas_height,
        downscaled,
        exposure_scales,
    };
    (textured, report)
}

struct Assignment {
    view: usize,
    pixels: [(f32, f32); 3],
}

struct Chart {
    /// `None` is the single fallback texel shared by every unobserved face.
    view: Option<usize>,
    src_x0: usize,
    src_y0: usize,
    src_w: usize,
    src_h: usize,
    content_w: usize,
    content_h: usize,
    origin_x: usize,
    origin_y: usize,
    gutter: usize,
}

fn atlas_limit(requested: usize) -> usize {
    requested.clamp(1, MAX_ATLAS_SIZE)
}

/// Highest minimum corner score among views that also see the centroid.
///
/// A view that misses any corner is not a candidate, which stops an occluder
/// in front of one vertex from painting the whole face. The centroid uses the
/// same visibility test, so an occluder in the interior is rejected too. The
/// score stays the minimum of the three corners.
fn best_view(
    mesh: &Mesh,
    triangle: [u32; 3],
    observed: &[Vec<Observation>],
    views: &[ColorView],
    params: &ColoringParams,
) -> Option<Assignment> {
    let centroid = mesh.centroid(&triangle);
    let face_normal = mesh.triangle_normal(&triangle);
    let mut best: Option<Assignment> = None;
    let mut best_score = f32::NEG_INFINITY;

    for view in 0..views.len() {
        let mut pixels = [(0.0f32, 0.0f32); 3];
        let mut min_score = f32::INFINITY;
        let mut visible = true;
        for (corner, index) in triangle.iter().enumerate() {
            let Some(sample) = sample_of(observed, *index, view) else {
                visible = false;
                break;
            };
            pixels[corner] = sample.pixel;
            min_score = min_score.min(sample.score);
        }
        if !visible || !min_score.is_finite() {
            continue;
        }
        if observe(&views[view], view, &centroid, &face_normal, params).is_none() {
            continue;
        }
        let replace = match &best {
            None => true,
            Some(current) => match min_score.partial_cmp(&best_score) {
                Some(Ordering::Greater) => true,
                Some(Ordering::Equal) => view < current.view,
                _ => false,
            },
        };
        if replace {
            best_score = min_score;
            best = Some(Assignment { view, pixels });
        }
    }

    best
}

fn sample_of(observed: &[Vec<Observation>], vertex: u32, view: usize) -> Option<&Observation> {
    observed
        .get(vertex as usize)?
        .iter()
        .find(|sample| sample.view == view)
}

/// Inclusive pixel bounds of the projected corners, clipped to the view.
fn crop_rect(pixels: &[(f32, f32)], width: usize, height: usize) -> (usize, usize, usize, usize) {
    if width == 0 || height == 0 {
        return (0, 0, 1, 1);
    }
    let mut min_u = f32::MAX;
    let mut min_v = f32::MAX;
    let mut max_u = f32::MIN;
    let mut max_v = f32::MIN;
    for &(u, v) in pixels {
        if u.is_finite() && v.is_finite() {
            min_u = min_u.min(u);
            min_v = min_v.min(v);
            max_u = max_u.max(u);
            max_v = max_v.max(v);
        }
    }
    if !min_u.is_finite() {
        return (0, 0, 1.min(width), 1.min(height));
    }
    let x0 = (min_u.floor() as isize).clamp(0, width as isize - 1) as usize;
    let y0 = (min_v.floor() as isize).clamp(0, height as isize - 1) as usize;
    let x1 = ((max_u.floor() as isize) + 1).clamp(x0 as isize + 1, width as isize) as usize;
    let y1 = ((max_v.floor() as isize) + 1).clamp(y0 as isize + 1, height as isize) as usize;
    (x0, y0, x1 - x0, y1 - y0)
}

/// Shelf-pack `charts` into a square of side `limit`. Returns whether any
/// photographed chart had to shrink.
fn pack_charts(charts: &mut [Chart], gutter_request: usize, limit: usize) -> bool {
    if charts.is_empty() {
        return false;
    }

    // A chart needs one content pixel. Gutter takes the rest, symmetrically.
    let mut gutter = gutter_request.min(limit.saturating_sub(1) / 2);
    loop {
        if let Some(placed) = search_scale(charts, gutter, limit) {
            let mut downscaled = false;
            for (chart, place) in charts.iter_mut().zip(placed.placements) {
                if chart.view.is_some()
                    && (place.content_w < chart.src_w || place.content_h < chart.src_h)
                {
                    downscaled = true;
                }
                chart.content_w = place.content_w;
                chart.content_h = place.content_h;
                chart.origin_x = place.origin_x;
                chart.origin_y = place.origin_y;
                chart.gutter = gutter;
            }
            return downscaled;
        }
        if gutter == 0 {
            break;
        }
        gutter -= 1;
    }

    // More minimum charts than the square can hold. Keep the cap and fold the
    // overflow onto the last pixel rather than grow past it.
    force_into_limit(charts, limit);
    true
}

#[derive(Clone)]
struct Place {
    content_w: usize,
    content_h: usize,
    origin_x: usize,
    origin_y: usize,
}

struct Placed {
    placements: Vec<Place>,
}

fn search_scale(charts: &[Chart], gutter: usize, limit: usize) -> Option<Placed> {
    if let Some(placed) = try_pack(charts, 1.0, gutter, limit) {
        return Some(placed);
    }
    let mut best: Option<Placed> = None;
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for _ in 0..24 {
        let mid = (lo + hi) * 0.5;
        if let Some(placed) = try_pack(charts, mid, gutter, limit) {
            best = Some(placed);
            lo = mid;
        } else {
            hi = mid;
        }
    }
    best
}

fn content_dims(src_w: usize, src_h: usize, scale: f32) -> (usize, usize) {
    let src_w = src_w.max(1);
    let src_h = src_h.max(1);
    if scale >= 1.0 {
        return (src_w, src_h);
    }
    let width = ((src_w as f32) * scale).floor().max(1.0) as usize;
    let height = ((src_h as f32) * scale).floor().max(1.0) as usize;
    (width.min(src_w), height.min(src_h))
}

fn try_pack(charts: &[Chart], scale: f32, gutter: usize, limit: usize) -> Option<Placed> {
    let mut padded = Vec::with_capacity(charts.len());
    for chart in charts {
        let (content_w, content_h) = content_dims(chart.src_w, chart.src_h, scale);
        let rect_w = content_w + 2 * gutter;
        let rect_h = content_h + 2 * gutter;
        if rect_w > limit || rect_h > limit {
            return None;
        }
        padded.push((content_w, content_h, rect_w, rect_h));
    }

    let mut order: Vec<usize> = (0..charts.len()).collect();
    order.sort_by(|&left, &right| {
        padded[right]
            .3
            .cmp(&padded[left].3)
            .then(padded[right].2.cmp(&padded[left].2))
            .then(left.cmp(&right))
    });

    let mut placements = vec![
        Place {
            content_w: 0,
            content_h: 0,
            origin_x: 0,
            origin_y: 0,
        };
        charts.len()
    ];
    let mut x = 0usize;
    let mut y = 0usize;
    let mut shelf_h = 0usize;

    for index in order {
        let (content_w, content_h, rect_w, rect_h) = padded[index];
        if x > 0 && x + rect_w > limit {
            y += shelf_h;
            x = 0;
            shelf_h = 0;
        }
        if y.saturating_add(rect_h) > limit {
            return None;
        }
        placements[index] = Place {
            content_w,
            content_h,
            origin_x: x,
            origin_y: y,
        };
        x += rect_w;
        shelf_h = shelf_h.max(rect_h);
    }

    Some(Placed { placements })
}

fn force_into_limit(charts: &mut [Chart], limit: usize) {
    let mut x = 0usize;
    let mut y = 0usize;
    for chart in charts.iter_mut() {
        if x >= limit {
            x = 0;
            y = y.saturating_add(1).min(limit.saturating_sub(1));
        }
        chart.content_w = 1;
        chart.content_h = 1;
        chart.gutter = 0;
        chart.origin_x = x.min(limit.saturating_sub(1));
        chart.origin_y = y;
        x += 1;
    }
}

fn bake_atlas(
    charts: &[Chart],
    views: &[ColorView],
    scales: &[f32],
    fallback: [u8; 3],
) -> (Vec<u8>, usize, usize) {
    if charts.is_empty() {
        return (Vec::new(), 0, 0);
    }
    let mut width = 1usize;
    let mut height = 1usize;
    for chart in charts {
        let right = chart.origin_x + chart.content_w + 2 * chart.gutter;
        let bottom = chart.origin_y + chart.content_h + 2 * chart.gutter;
        width = width.max(right);
        height = height.max(bottom);
    }
    let mut atlas = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        atlas.extend_from_slice(&fallback);
    }

    for chart in charts {
        let content = raster_chart(chart, views, scales, fallback);
        blit_replicated(&mut atlas, width, chart, &content);
    }

    (atlas, width, height)
}

fn raster_chart(chart: &Chart, views: &[ColorView], scales: &[f32], fallback: [u8; 3]) -> Vec<u8> {
    let mut pixels = vec![0u8; chart.content_w * chart.content_h * 3];
    let Some(view_index) = chart.view else {
        for chunk in pixels.chunks_exact_mut(3) {
            chunk.copy_from_slice(&fallback);
        }
        return pixels;
    };
    let Some(view) = views.get(view_index) else {
        for chunk in pixels.chunks_exact_mut(3) {
            chunk.copy_from_slice(&fallback);
        }
        return pixels;
    };
    let scale = scales.get(view_index).copied().unwrap_or(1.0);

    for y in 0..chart.content_h {
        for x in 0..chart.content_w {
            let su = chart.src_x0 as f32
                + (x as f32 + 0.5) * chart.src_w as f32 / chart.content_w as f32;
            let sv = chart.src_y0 as f32
                + (y as f32 + 0.5) * chart.src_h as f32 / chart.content_h as f32;
            let sx = (su.floor() as isize).clamp(
                chart.src_x0 as isize,
                chart.src_x0 as isize + chart.src_w as isize - 1,
            ) as usize;
            let sy = (sv.floor() as isize).clamp(
                chart.src_y0 as isize,
                chart.src_y0 as isize + chart.src_h as isize - 1,
            ) as usize;
            let rgb = sample_exposed(view, sx, sy, scale, fallback);
            let dst = (y * chart.content_w + x) * 3;
            pixels[dst..dst + 3].copy_from_slice(&rgb);
        }
    }
    pixels
}

fn sample_exposed(view: &ColorView, x: usize, y: usize, scale: f32, fallback: [u8; 3]) -> [u8; 3] {
    if x >= view.width || y >= view.height {
        return fallback;
    }
    let pixel = y * view.width + x;
    if view.valid.get(pixel).copied().unwrap_or(0) == 0 {
        return fallback;
    }
    let offset = pixel * 3;
    if offset + 2 >= view.color.len() {
        return fallback;
    }
    let rgb = [
        view.color[offset],
        view.color[offset + 1],
        view.color[offset + 2],
    ];
    // Scale 1 is the estimate "leave this view alone". Re-encoding it would
    // only move bytes by quantization, which is not a photometric change.
    if !scale.is_finite() || (scale - 1.0).abs() <= 1e-6 {
        return rgb;
    }
    encode(scale_linear(rgb, scale))
}

/// Copy `content` into the chart and repeat its edge pixels through the gutter.
fn blit_replicated(atlas: &mut [u8], atlas_width: usize, chart: &Chart, content: &[u8]) {
    let rect_w = chart.content_w + 2 * chart.gutter;
    let rect_h = chart.content_h + 2 * chart.gutter;
    for y in 0..rect_h {
        for x in 0..rect_w {
            let cx = (x as isize - chart.gutter as isize).clamp(0, chart.content_w as isize - 1)
                as usize;
            let cy = (y as isize - chart.gutter as isize).clamp(0, chart.content_h as isize - 1)
                as usize;
            let src = (cy * chart.content_w + cx) * 3;
            let dx = chart.origin_x + x;
            let dy = chart.origin_y + y;
            let dst = (dy * atlas_width + dx) * 3;
            atlas[dst..dst + 3].copy_from_slice(&content[src..src + 3]);
        }
    }
}

fn emit_vertices(
    mesh: &Mesh,
    normals: &[Vector3<f32>],
    assigned: &[Option<Assignment>],
    chart_of_face: &[usize],
    charts: &[Chart],
    out: &mut TexturedMesh,
) {
    let mut key_to_index: HashMap<(u32, usize), u32> = HashMap::new();

    for (face_index, triangle) in mesh.triangles.iter().enumerate() {
        let chart_index = chart_of_face[face_index];
        let chart = &charts[chart_index];
        let mut indices = [0u32; 3];
        for corner in 0..3 {
            let vertex = triangle[corner];
            let key = (vertex, chart_index);
            if let Some(&existing) = key_to_index.get(&key) {
                indices[corner] = existing;
                continue;
            }
            let index = out.positions.len() as u32;
            let source = vertex as usize;
            out.positions.push(
                mesh.vertices
                    .get(source)
                    .copied()
                    .unwrap_or_else(Vector3::zeros),
            );
            out.normals
                .push(normals.get(source).copied().unwrap_or_else(Vector3::zeros));
            let uv = match (&assigned[face_index], chart.view) {
                (Some(hit), Some(_)) => uv_for_pixel(hit.pixels[corner], chart, out),
                _ => uv_for_fallback(chart, out),
            };
            out.uvs.push(uv);
            key_to_index.insert(key, index);
            indices[corner] = index;
        }
        out.indices.extend(indices);
    }
}

fn uv_for_pixel(pixel: (f32, f32), chart: &Chart, mesh: &TexturedMesh) -> [f32; 2] {
    let lx = if chart.src_w == 0 {
        0.0
    } else {
        (pixel.0 - chart.src_x0 as f32) / chart.src_w as f32 * chart.content_w as f32
    };
    let ly = if chart.src_h == 0 {
        0.0
    } else {
        (pixel.1 - chart.src_y0 as f32) / chart.src_h as f32 * chart.content_h as f32
    };
    // The last thousandth of a pixel stays in the content. The gutter is only
    // there for filtering; a vertex should not sit on the neighbouring chart.
    let lx = lx.clamp(0.0, (chart.content_w as f32 - 1e-3).max(0.0));
    let ly = ly.clamp(0.0, (chart.content_h as f32 - 1e-3).max(0.0));
    let ax = chart.origin_x as f32 + chart.gutter as f32 + lx;
    let ay = chart.origin_y as f32 + chart.gutter as f32 + ly;
    normalize_uv(ax, ay, mesh.atlas_width, mesh.atlas_height)
}

fn uv_for_fallback(chart: &Chart, mesh: &TexturedMesh) -> [f32; 2] {
    let ax = chart.origin_x as f32 + chart.gutter as f32 + 0.5;
    let ay = chart.origin_y as f32 + chart.gutter as f32 + 0.5;
    normalize_uv(ax, ay, mesh.atlas_width, mesh.atlas_height)
}

fn normalize_uv(x: f32, y: f32, width: usize, height: usize) -> [f32; 2] {
    [
        if width == 0 {
            0.0
        } else {
            (x / width as f32).clamp(0.0, 1.0)
        },
        if height == 0 {
            0.0
        } else {
            (y / height as f32).clamp(0.0, 1.0)
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coloring::{
        colorize, linear_to_srgb, srgb_to_linear, ColorMode, ColoringParams, MIN_EXPOSURE_OVERLAPS,
    };
    use crate::transform_point;
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    const SIDE: usize = 64;
    const FX: f32 = 60.0;

    fn intrinsics() -> crate::Intrinsics {
        crate::Intrinsics {
            fx: FX,
            fy: FX,
            cx: SIDE as f32 / 2.0,
            cy: SIDE as f32 / 2.0,
        }
    }

    fn at(x: f32, y: f32, z: f32) -> Isometry3<f32> {
        Isometry3::from_parts(Translation3::new(x, y, z), UnitQuaternion::identity())
    }

    fn solid(pose: Isometry3<f32>, fill: [u8; 3], depth: f32) -> ColorView {
        ColorView {
            color: fill.iter().copied().cycle().take(SIDE * SIDE * 3).collect(),
            width: SIDE,
            height: SIDE,
            depth: vec![depth; SIDE * SIDE],
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

    /// Camera-space Z of a point on the plane `world z = 2` seen by `pose`,
    /// which is a pure translation. Every such point shares that Z.
    fn plane_depth(pose: &Isometry3<f32>) -> f32 {
        let camera = transform_point(&pose.inverse(), &Vector3::new(0.0, 0.0, 2.0));
        camera.z
    }

    fn project(pose: &Isometry3<f32>, point: &Vector3<f32>) -> (f32, f32) {
        let camera = transform_point(&pose.inverse(), point);
        intrinsics()
            .project(&camera)
            .expect("in front of the camera")
    }

    fn at_pixel(u: f32, v: f32) -> Vector3<f32> {
        let z = 2.0;
        let k = intrinsics();
        Vector3::new((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z)
    }

    /// A camera `distance` metres from `point` along `direction` (from the
    /// point toward the camera), with the principal point shifted so `point`
    /// still projects to the image centre.
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
        let mut view = solid(pose, fill, camera.z);
        view.intrinsics.cx = SIDE as f32 / 2.0 - camera.x / camera.z * FX;
        view.intrinsics.cy = SIDE as f32 / 2.0 - camera.y / camera.z * FX;
        view
    }

    fn exposure_views(bright: [u8; 3], darker: [u8; 3]) -> Vec<ColorView> {
        let mut reference = solid(Isometry3::identity(), bright, 2.0);
        reference.tracking_quality = 0.3;
        let mut dark = solid(Isometry3::identity(), darker, 2.0);
        dark.tracking_quality = 1.0;
        vec![reference, dark]
    }

    fn facing() -> Vector3<f32> {
        Vector3::new(0.0, 0.0, -1.0)
    }

    fn triangle_mesh(vertices: Vec<Vector3<f32>>, triangles: Vec<[u32; 3]>) -> Mesh {
        Mesh {
            vertices,
            triangles,
            colors: None,
        }
    }

    /// One face in the plane z = 2, wound so its normal points at the origin.
    fn front_triangle() -> Mesh {
        triangle_mesh(
            vec![
                at_pixel(28.0, 28.0),
                at_pixel(36.0, 28.0),
                at_pixel(32.0, 38.0),
            ],
            vec![[0, 2, 1]],
        )
    }

    fn params() -> TexturingParams {
        TexturingParams::default()
    }

    fn texel(mesh: &TexturedMesh, x: usize, y: usize) -> [u8; 3] {
        let index = (y * mesh.atlas_width + x) * 3;
        [
            mesh.atlas[index],
            mesh.atlas[index + 1],
            mesh.atlas[index + 2],
        ]
    }

    fn sample(mesh: &TexturedMesh, uv: [f32; 2]) -> [u8; 3] {
        assert!(mesh.atlas_width > 0 && mesh.atlas_height > 0);
        let x = ((uv[0] * mesh.atlas_width as f32).floor() as usize).min(mesh.atlas_width - 1);
        let y = ((uv[1] * mesh.atlas_height as f32).floor() as usize).min(mesh.atlas_height - 1);
        texel(mesh, x, y)
    }

    fn assert_unit_uvs(mesh: &TexturedMesh) {
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert_eq!(mesh.positions.len(), mesh.uvs.len());
        assert_eq!(mesh.indices.len() % 3, 0);
        for uv in &mesh.uvs {
            assert!(
                (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]),
                "uv {uv:?} left the atlas"
            );
        }
    }

    #[test]
    fn a_more_frontal_view_wins_and_a_tie_keeps_the_earlier_view() {
        let mesh = front_triangle();
        let point = Vector3::new(0.0, 0.0, 2.0);
        let frontal = view_from(point, facing(), 2.0, [10, 20, 30]);
        let grazing = view_from(point, Vector3::new(0.766, 0.0, -0.643), 2.0, [200, 0, 0]);

        let (textured, report) = texture(&mesh, &[grazing, frontal], &params());

        assert_eq!(report.unobserved, 0);
        assert_unit_uvs(&textured);
        for uv in &textured.uvs {
            assert_eq!(sample(&textured, *uv), [10, 20, 30]);
        }

        let first = solid(Isometry3::identity(), [1, 2, 3], 2.0);
        let second = solid(Isometry3::identity(), [9, 9, 9], 2.0);
        let (tied, _) = texture(&mesh, &[first, second], &params());
        assert_eq!(sample(&tied, tied.uvs[0]), [1, 2, 3]);
    }

    #[test]
    fn a_view_must_see_every_corner_and_an_occluder_is_rejected() {
        let mesh = front_triangle();
        let mut blocked = solid(Isometry3::identity(), [255, 0, 0], 0.4);
        blocked.tracking_quality = 1.0;
        let mut clear = solid(Isometry3::identity(), [0, 255, 0], 2.0);
        clear.tracking_quality = 0.2;

        let (textured, report) = texture(&mesh, &[blocked, clear], &params());

        assert_eq!(report.unobserved, 0, "the clear view still sees the face");
        assert_eq!(sample(&textured, textured.uvs[0]), [0, 255, 0]);

        // The frontal camera would win on score, but one corner is behind an
        // occluder, so the whole face goes to the grazing camera instead.
        let point = Vector3::new(0.0, 0.0, 2.0);
        let mut frontal = view_from(point, facing(), 2.0, [255, 0, 0]);
        let grazing = view_from(point, Vector3::new(0.766, 0.0, -0.643), 2.0, [0, 0, 255]);
        let corner = mesh.vertices[0];
        let (u, v) = {
            let camera = transform_point(&frontal.pose.inverse(), &corner);
            frontal.intrinsics.project(&camera).unwrap()
        };
        let pixel = v.floor() as usize * frontal.width + u.floor() as usize;
        frontal.depth[pixel] = 0.3;

        let (textured, report) = texture(&mesh, &[frontal, grazing], &params());
        assert_eq!(report.unobserved, 0);
        assert_eq!(
            sample(&textured, textured.uvs[0]),
            [0, 0, 255],
            "one failed corner must drop the otherwise better view"
        );
    }

    #[test]
    fn an_interior_occluder_rejects_a_view_whose_corners_are_all_visible() {
        let mesh = front_triangle();
        let pose = Isometry3::identity();
        let centroid = mesh.centroid(&mesh.triangles[0]);
        let centre = project(&pose, &centroid);
        let centre_pixel = (centre.0.floor() as usize, centre.1.floor() as usize);
        for corner in &mesh.vertices {
            let (u, v) = project(&pose, corner);
            assert_ne!(
                (u.floor() as usize, v.floor() as usize),
                centre_pixel,
                "the occluder has to sit inside the face, not on a corner"
            );
        }

        let mut blocked = solid(pose, [255, 0, 0], plane_depth(&pose));
        blocked.tracking_quality = 1.0;
        obscure(&mut blocked, &pose, &centroid);

        let mut clear = solid(pose, [0, 255, 0], plane_depth(&pose));
        clear.tracking_quality = 0.2;

        let (textured, report) = texture(&mesh, &[blocked, clear], &params());
        assert_eq!(report.unobserved, 0, "the clear view still sees the face");
        assert_eq!(
            sample(&textured, textured.uvs[0]),
            [0, 255, 0],
            "a nearer surface over the centroid must drop the better-scoring view"
        );

        let mut only = solid(pose, [255, 0, 0], plane_depth(&pose));
        obscure(&mut only, &pose, &centroid);
        let (alone, alone_report) = texture(&mesh, &[only], &params());
        assert_eq!(alone_report.unobserved, 1);
        assert_eq!(sample(&alone, alone.uvs[0]), params().coloring.fallback);
    }

    #[test]
    fn vertices_are_duplicated_only_at_view_seams() {
        // A quad split into two triangles that share an edge.
        let vertices = vec![
            at_pixel(12.4, 12.2),
            at_pixel(28.6, 13.1),
            at_pixel(13.2, 30.4),
            at_pixel(27.8, 29.5),
        ];
        let mesh = triangle_mesh(vertices.clone(), vec![[0, 2, 1], [1, 2, 3]]);
        let pose = Isometry3::identity();
        let shared = solid(pose, [20, 40, 60], plane_depth(&pose));

        let (together, report) = texture(&mesh, &[shared], &params());

        assert_eq!(report.unobserved, 0);
        assert_eq!(
            together.positions.len(),
            4,
            "one view has one UV per vertex"
        );
        assert_eq!(together.indices.len(), 6);
        let seam = shared_edge_indices(&together);
        assert_eq!(seam.len(), 2, "the shared edge is two vertices, once each");

        let mut red = solid(pose, [255, 0, 0], plane_depth(&pose));
        let mut blue = solid(pose, [0, 0, 255], plane_depth(&pose));
        // Hide the far corner of each triangle from the other view.
        obscure(&mut red, &pose, &vertices[3]);
        obscure(&mut blue, &pose, &vertices[0]);

        let (split, report) = texture(&mesh, &[red, blue], &params());
        assert_eq!(report.unobserved, 0);
        assert_eq!(
            split.positions.len(),
            6,
            "the seam vertices must exist once per view"
        );
        let duplicated = duplicated_positions(&split);
        assert_eq!(duplicated, 2, "exactly the shared edge is duplicated");
        assert!(
            seam_uvs_differ(&split),
            "the two copies of a seam vertex need different UVs"
        );
        assert_eq!(sample(&split, split.uvs[0]), [255, 0, 0]);
        let blue_corner = split
            .uvs
            .iter()
            .copied()
            .find(|uv| sample(&split, *uv) == [0, 0, 255])
            .expect("the second face keeps the blue view");
        assert_eq!(sample(&split, blue_corner), [0, 0, 255]);
    }

    fn obscure(view: &mut ColorView, pose: &Isometry3<f32>, point: &Vector3<f32>) {
        let (u, v) = project(pose, point);
        let pixel = v.floor() as usize * view.width + u.floor() as usize;
        view.depth[pixel] = 0.25;
    }

    fn shared_edge_indices(mesh: &TexturedMesh) -> Vec<u32> {
        let first: Vec<u32> = mesh.indices[..3].to_vec();
        let second: Vec<u32> = mesh.indices[3..6].to_vec();
        first
            .into_iter()
            .filter(|index| second.contains(index))
            .collect()
    }

    fn duplicated_positions(mesh: &TexturedMesh) -> usize {
        let mut seen = Vec::new();
        let mut dups = 0usize;
        for position in &mesh.positions {
            let key = (
                position.x.to_bits(),
                position.y.to_bits(),
                position.z.to_bits(),
            );
            if seen.contains(&key) {
                dups += 1;
            } else {
                seen.push(key);
            }
        }
        dups
    }

    fn seam_uvs_differ(mesh: &TexturedMesh) -> bool {
        let mut by_position: Vec<((u32, u32, u32), [f32; 2])> = Vec::new();
        for (position, uv) in mesh.positions.iter().zip(mesh.uvs.iter()) {
            let key = (
                position.x.to_bits(),
                position.y.to_bits(),
                position.z.to_bits(),
            );
            if let Some((_, other)) = by_position.iter().find(|(found, _)| *found == key) {
                let delta = (uv[0] - other[0]).abs() + (uv[1] - other[1]).abs();
                if delta > 1e-4 {
                    return true;
                }
            } else {
                by_position.push((key, *uv));
            }
        }
        false
    }

    #[test]
    fn uvs_stay_inside_the_atlas_and_sample_the_assigned_chart() {
        let mesh = front_triangle();
        let view = solid(Isometry3::identity(), [8, 16, 32], 2.0);
        let (textured, report) = texture(&mesh, &[view], &params());

        assert_eq!(report.unobserved, 0);
        assert_unit_uvs(&textured);
        assert!(
            textured.uvs.windows(2).any(|pair| pair[0] != pair[1]),
            "the three corners of a real face must not collapse to one UV"
        );
        for uv in &textured.uvs {
            assert_eq!(sample(&textured, *uv), [8, 16, 32]);
        }
        assert!(textured.atlas_width <= MAX_ATLAS_SIZE);
        assert!(textured.atlas_height <= MAX_ATLAS_SIZE);
        assert!(!report.downscaled);
    }

    #[test]
    fn gutters_replicate_the_chart_edge() {
        let pose = Isometry3::identity();
        let corners = [
            at_pixel(8.2, 8.4),
            at_pixel(24.7, 9.3),
            at_pixel(10.6, 26.8),
        ];
        let mesh = triangle_mesh(corners.to_vec(), vec![[0, 2, 1]]);
        let pixels = [
            project(&pose, &corners[0]),
            project(&pose, &corners[2]),
            project(&pose, &corners[1]),
        ];
        let (x0, y0, width, height) = crop_rect(&pixels, SIDE, SIDE);
        assert!(width >= 3 && height >= 3, "the crop needs an interior");

        let mut view = solid(pose, [0, 0, 0], plane_depth(&pose));
        for y in y0..y0 + height {
            for x in x0..x0 + width {
                let edge = x == x0 || y == y0 || x + 1 == x0 + width || y + 1 == y0 + height;
                let rgb = if edge { [255, 0, 0] } else { [0, 255, 0] };
                let offset = (y * SIDE + x) * 3;
                view.color[offset..offset + 3].copy_from_slice(&rgb);
            }
        }

        let mut settings = params();
        settings.gutter = 2;
        let (textured, report) = texture(&mesh, &[view], &settings);

        assert!(!report.downscaled);
        assert_eq!(report.charts, 1);
        assert_eq!(textured.atlas_width, width + 4);
        assert_eq!(textured.atlas_height, height + 4);
        // The chart is the only one, so it sits at the origin. The gutter
        // repeats the crop's red edge; the middle of the chart stays green.
        assert_eq!(texel(&textured, 0, settings.gutter), [255, 0, 0]);
        assert_eq!(
            texel(&textured, settings.gutter, settings.gutter),
            [255, 0, 0],
            "the content edge has to match the gutter that copies it"
        );
        let interior = texel(
            &textured,
            settings.gutter + width / 2,
            settings.gutter + height / 2,
        );
        assert_eq!(interior, [0, 255, 0]);
    }

    #[test]
    fn an_unobserved_face_uses_the_fallback_texel_including_its_gutter() {
        let hidden = triangle_mesh(vec![Vector3::new(0.0, 0.0, -2.0); 3], vec![[0, 1, 2]]);
        let mut settings = params();
        settings.gutter = 2;
        settings.coloring.fallback = [1, 2, 3];

        let (only_hidden, report) = texture(
            &hidden,
            &[solid(Isometry3::identity(), [9, 9, 9], 2.0)],
            &settings,
        );

        assert_eq!(report.unobserved, 1);
        assert_eq!(report.charts, 1);
        assert!(only_hidden
            .atlas
            .chunks_exact(3)
            .all(|pixel| pixel == [1, 2, 3]));
        assert_unit_uvs(&only_hidden);
        for uv in &only_hidden.uvs {
            assert_eq!(sample(&only_hidden, *uv), [1, 2, 3]);
        }

        let mut vertices = front_triangle().vertices;
        vertices.push(Vector3::new(0.0, 0.0, -2.0));
        vertices.push(Vector3::new(0.1, 0.0, -2.0));
        vertices.push(Vector3::new(0.0, 0.1, -2.0));
        let mixed = triangle_mesh(vertices, vec![[0, 2, 1], [3, 4, 5]]);
        let (textured, report) = texture(
            &mixed,
            &[solid(Isometry3::identity(), [255, 0, 0], 2.0)],
            &settings,
        );

        assert_eq!(report.unobserved, 1);
        assert_eq!(report.charts, 2);
        assert_unit_uvs(&textured);
        let seen = sample(&textured, textured.uvs[0]);
        let unseen = sample(&textured, textured.uvs[3]);
        assert_eq!(seen, [255, 0, 0]);
        assert_eq!(unseen, [1, 2, 3]);
        assert_ne!(seen, unseen);
    }

    #[test]
    fn the_atlas_downscales_to_stay_within_the_cap() {
        let corners = [at_pixel(1.5, 1.5), at_pixel(60.5, 4.0), at_pixel(6.0, 60.5)];
        let mesh = triangle_mesh(corners.to_vec(), vec![[0, 2, 1]]);
        let pixels = [
            project(&Isometry3::identity(), &corners[0]),
            project(&Isometry3::identity(), &corners[2]),
            project(&Isometry3::identity(), &corners[1]),
        ];
        let (_, _, crop_w, crop_h) = crop_rect(&pixels, SIDE, SIDE);
        let mut settings = params();
        settings.gutter = 2;
        settings.max_atlas_size = 24;
        assert!(crop_w + 4 > settings.max_atlas_size || crop_h + 4 > settings.max_atlas_size);

        let (textured, report) = texture(
            &mesh,
            &[solid(Isometry3::identity(), [0, 0, 255], 2.0)],
            &settings,
        );

        assert!(report.downscaled);
        assert!(textured.atlas_width <= 24, "width {}", textured.atlas_width);
        assert!(
            textured.atlas_height <= 24,
            "height {}",
            textured.atlas_height
        );
        assert!(textured.atlas_width <= MAX_ATLAS_SIZE);
        assert!(textured.atlas_height <= MAX_ATLAS_SIZE);
        assert_eq!(
            textured.atlas.len(),
            textured.atlas_width * textured.atlas_height * 3
        );
        assert_eq!(sample(&textured, textured.uvs[0]), [0, 0, 255]);
        assert_ne!(
            sample(&textured, textured.uvs[0]),
            settings.coloring.fallback
        );
    }

    #[test]
    fn the_requested_atlas_cap_cannot_exceed_8192() {
        assert_eq!(MAX_ATLAS_SIZE, 8192);
        assert_eq!(atlas_limit(usize::MAX), 8192);
        assert_eq!(atlas_limit(0), 1);
        assert_eq!(atlas_limit(100), 100);

        let mut settings = params();
        settings.max_atlas_size = usize::MAX;
        let (textured, _) = texture(
            &front_triangle(),
            &[solid(Isometry3::identity(), [4, 5, 6], 2.0)],
            &settings,
        );
        assert!(textured.atlas_width <= MAX_ATLAS_SIZE);
        assert!(textured.atlas_height <= MAX_ATLAS_SIZE);
        assert!(
            textured.atlas_width < 64,
            "a small face must not be inflated to the cap"
        );
    }

    #[test]
    fn flip_x_mirrors_positions_normals_and_winding_but_not_uvs() {
        // Tilted so the area-weighted normal has a nonzero X component.
        let mesh = triangle_mesh(
            vec![
                Vector3::new(0.2, 0.0, 2.0),
                Vector3::new(0.9, 0.1, 2.35),
                Vector3::new(0.15, 0.7, 2.05),
            ],
            vec![[0, 2, 1]],
        );
        let mut view = solid(Isometry3::identity(), [30, 10, 10], 2.0);
        let centroid = mesh.centroid(&mesh.triangles[0]);
        for point in mesh.vertices.iter().chain(std::iter::once(&centroid)) {
            let camera = transform_point(&view.pose.inverse(), point);
            let (u, v) = view.intrinsics.project(&camera).unwrap();
            let pixel = v.floor() as usize * SIDE + u.floor() as usize;
            view.depth[pixel] = camera.z;
        }

        let (mut textured, report) = texture(&mesh, &[view], &params());
        assert_eq!(report.unobserved, 0);
        assert!(
            textured.normals.iter().any(|normal| normal.x.abs() > 0.1),
            "the fixture normal must leave the YZ plane, got {:?}",
            textured.normals
        );

        let uvs = textured.uvs.clone();
        let positions = textured.positions.clone();
        let normals = textured.normals.clone();
        let indices = textured.indices.clone();
        textured.flip_x();

        assert_eq!(textured.uvs, uvs);
        for (before, after) in positions.iter().zip(textured.positions.iter()) {
            assert!((after.x + before.x).abs() < 1e-6);
            assert!((after.y - before.y).abs() < 1e-6);
            assert!((after.z - before.z).abs() < 1e-6);
        }
        for (before, after) in normals.iter().zip(textured.normals.iter()) {
            assert!((after.x + before.x).abs() < 1e-5, "{before:?} -> {after:?}");
            assert!((after.y - before.y).abs() < 1e-5);
            assert!((after.z - before.z).abs() < 1e-5);
        }
        assert_eq!(textured.indices[0], indices[0]);
        assert_eq!(textured.indices[1], indices[2]);
        assert_eq!(textured.indices[2], indices[1]);

        let a = textured.positions[textured.indices[0] as usize];
        let b = textured.positions[textured.indices[1] as usize];
        let c = textured.positions[textured.indices[2] as usize];
        let face = (b - a).cross(&(c - a)).normalize();
        let stored = textured.normals[textured.indices[0] as usize].normalize();
        assert!(
            face.dot(&stored) > 0.99,
            "stored {stored:?} diverged from the wound face {face:?}"
        );
    }

    #[test]
    fn exposure_scales_match_the_vertex_painter_and_are_applied() {
        let bright = [200, 160, 80];
        let darker = [
            linear_to_srgb(srgb_to_linear(bright[0]) * 0.5),
            linear_to_srgb(srgb_to_linear(bright[1]) * 0.5),
            linear_to_srgb(srgb_to_linear(bright[2]) * 0.5),
        ];
        let mut vertices = Vec::new();
        for index in 0..MIN_EXPOSURE_OVERLAPS {
            let shift = index as f32 * 0.4;
            vertices.push(at_pixel(24.0 + shift, 30.0));
        }
        // The face uses three of those vertices. The rest exist so the overlap
        // count is the same one the vertex painter requires.
        let mesh = triangle_mesh(vertices, vec![[0, 1, 2]]);
        let coloring = ColoringParams {
            mode: ColorMode::Best,
            ..ColoringParams::default()
        };
        let normals = mesh.vertex_normals();
        let (_, color_report) = colorize(
            &mesh.vertices,
            &normals,
            &exposure_views(bright, darker),
            &coloring,
        );
        let settings = TexturingParams {
            coloring,
            ..params()
        };
        let (textured, report) = texture(&mesh, &exposure_views(bright, darker), &settings);

        assert_eq!(report.exposure_scales, color_report.exposure_scales);
        let scale = report.exposure_scales[1];
        assert!(scale > 1.5 && scale <= 2.0, "scale {scale}");
        let expected = encode(scale_linear(darker, scale));
        assert_eq!(sample(&textured, textured.uvs[0]), expected);
        assert_ne!(expected, darker, "the dark view must not be copied raw");
    }

    #[test]
    fn coverage_is_the_share_of_triangles_assigned_to_a_chart() {
        let report = TexturingReport {
            triangles: 8,
            unobserved: 2,
            charts: 3,
            atlas_width: 64,
            atlas_height: 32,
            downscaled: false,
            exposure_scales: Vec::new(),
        };
        assert_eq!(report.painted(), 6);
        assert!((report.coverage() - 0.75).abs() < 1e-6);

        let empty = TexturingReport {
            triangles: 0,
            unobserved: 0,
            charts: 0,
            atlas_width: 0,
            atlas_height: 0,
            downscaled: false,
            exposure_scales: Vec::new(),
        };
        assert_eq!(empty.coverage(), 0.0);
    }
}
