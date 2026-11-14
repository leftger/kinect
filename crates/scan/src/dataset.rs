//! Write a finished scan out as a dataset a Gaussian-splatting trainer can read.
//!
//! The scanner and a splatting trainer want opposite things from the same
//! capture. The scanner recovers geometry and poses from depth, and paints what
//! it can onto a triangle mesh. A trainer takes posed colour images, treats the
//! poses as given, and optimises appearance from them. The overlap is exactly the
//! part the scanner already has in hand: for every fused frame it holds the
//! registered colour *and* the world-from-camera pose, in the same structure.
//!
//! This module writes that pair out in the Nerfstudio convention, which is one
//! of the two dataset layouts Brush accepts, together with an initial point
//! cloud and the registration mask.
//!
//! # What goes where
//!
//! ```text
//! <dir>/transforms.json   intrinsics, poses, and the path of the point cloud
//! <dir>/images/00000.png  the registered colour, one file per kept view
//! <dir>/masks/00000.png   white where a colour sample was registered
//! <dir>/depth/00000.png   16-bit depth in millimetres (0 = invalid)
//! <dir>/init.ply          mesh vertices, coloured, as initial Gaussian means
//! ```
//!
//! # The two things that are easy to get wrong
//!
//! **The pose convention is not the scanner's.** The scanner's pose is
//! world-from-camera with x right, y down, z forward, because that is what
//! `Intrinsics::back_project` produces. The dataset formats store a camera-to-
//! world matrix in the OpenGL convention, where the camera looks down -z with y
//! up. [`camera_to_world`] negates the camera's y and z axes, which is precisely
//! the inverse of the conversion the trainer applies when it reads the file, so
//! the round trip through a trainer recovers the pose the scanner tracked.
//!
//! **The initial point cloud must carry no Gaussian attributes.** A trainer
//! reads `x/y/z`, takes `red/green/blue` as colour, and fills in scale, rotation
//! and opacity with its own defaults when the properties are absent. Writing
//! `scale_0` or `opacity` alongside the positions does not refine those
//! defaults, it replaces them with zeros, leaving metre-wide transparent
//! Gaussians. Colour is the one attribute worth supplying, and
//! [`geom::mesh::write_colored_points_ply`] writes exactly that and no more.
//!
//! # Resolution
//!
//! The colour is the *registered* colour: registration copied each depth pixel's
//! colour into the 512x424 depth grid, so these images are at depth resolution,
//! not the colour sensor's 1920x1080. That keeps the poses and the intrinsics
//! exact, because the depth and colour grids are then the same grid and no
//! stereo extrinsics are needed. Exporting the native colour frame instead would
//! need the depth-to-colour extrinsics and a second camera model; until that
//! exists, the depth grid is the ceiling and it is stated rather than
//! discovered.
//!
//! # The mirror
//!
//! The Kinect v2 reads out mirrored frames, so the sensor frame is a reflection
//! of the room: left and right are swapped. The exported mesh is flipped back by
//! default (`--mirror` keeps it raw), and a dataset has to follow it, or a
//! trainer builds a perfectly consistent but mirrored world that no longer lines
//! up with the mesh.
//!
//! The danger is doing it by halves: flipped images with unflipped poses, or a
//! flipped point cloud with unflipped images, train to nothing. So the reflection
//! is applied to everything at once, as the single mirror `M = diag(-1, 1, 1)`
//! of the world, which is also the first camera's frame:
//!
//! * each image and mask is flipped left to right, which is the camera seeing
//!   the mirrored scene;
//! * each pose becomes `M * T * M`, a proper rotation, so the file stays a rigid
//!   transform and the trainer's maths is untouched;
//! * the principal point moves to `width - cx`, because pixel columns reversed;
//! * the seed cloud's x is negated.
//!
//! A point the camera sees at column `u` is then seen at `width - u`, which is the
//! property the tests check. With `unmirror` false everything is written as the
//! sensor saw it, again all at once.

use geom::Intrinsics;
use geom::coloring::ColorView;
use geom::mesh::save_colored_points_ply;
use geom::png;
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use std::collections::HashSet;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use crate::scanner::Scanner;

/// A full TSDF surface can have millions of vertices. Training one Gaussian per
/// vertex makes Brush slow and over-dense, so the seed cloud is voxel-decimated
/// to this many points at most.
const MAX_INIT_POINTS: usize = 250_000;

/// Images live here. The trainer resolves each frame's `file_path` against the
/// directory holding `transforms.json`.
const IMAGE_DIR: &str = "images";
/// Masks live here, mirroring the image paths. The trainer finds a mask by
/// looking for a directory named `masks` whose path past that point matches the
/// image's, and matching the stem.
const MASK_DIR: &str = "masks";
const DEPTH_DIR: &str = "depth";
const TRANSFORMS: &str = "transforms.json";
const INIT_PLY: &str = "init.ply";
/// Brush 0.3.0 reads exactly 16,387 bytes from each image with `read_exact` to
/// learn its format and size, so a smaller file fails with "early eof". Every
/// image is padded past that (see `geom::png::pad_to_min_size`). Masks are read
/// whole, so they are left alone.
const MIN_IMAGE_BYTES: usize = 16_400;

/// What [`write_dataset`] wrote.
#[derive(Debug)]
pub struct DatasetReport {
    pub dir: PathBuf,
    /// Views written, one image and one mask each.
    pub views: usize,
    /// Points in the initial cloud. Zero means no `init.ply` was written.
    pub points: usize,
    /// Views left out because their pose could not be described in the file.
    pub skipped: usize,
}

/// One frame's entry in `transforms.json`.
#[derive(Debug)]
struct Frame {
    file_path: String,
    depth_path: String,
    matrix: [[f32; 4]; 4],
}

/// Write the scan's colour views, poses and surface to `dir`.
///
/// Needs colour views: `live --color`, a colour `.k2df` replay, or the synthetic
/// room. A depth-only scan has nothing to export, which is an error rather than
/// an empty dataset.
///
/// With `unmirror` the whole dataset is reflected to match the un-mirrored mesh
/// (see the module docs); without it everything stays in the sensor's frame.
pub fn write_dataset(scanner: &Scanner, dir: &Path, unmirror: bool) -> io::Result<DatasetReport> {
    let views = scanner.color_views();
    if views.is_empty() {
        return Err(invalid(
            "no colour views to export: a dataset needs `live --color`, a colour \
             recording, or the synthetic room",
        ));
    }

    let image_dir = dir.join(IMAGE_DIR);
    let mask_dir = dir.join(MASK_DIR);
    let depth_dir = dir.join(DEPTH_DIR);

    let (frames, skipped) = write_views(views, &image_dir, &mask_dir, &depth_dir, unmirror)?;

    // Every view carries the same intrinsics: the scanner copies one `Intrinsics`
    // onto each. Taking the first is therefore not a sample, it is the value.
    let intrinsics = views[0].intrinsics;
    let width = views[0].width;
    let height = views[0].height;

    // The mesh is in the sensor frame, like the poses: `Scanner::mesh` extracts
    // from the volume without applying the export mirror. Its colours come from
    // the same views that were just written, so the cloud starts near the colour
    // the trainer is about to optimise towards.
    //
    // This extracts and paints the surface a second time, the first being the
    // mesh export. Reusing the exported mesh would work only if its mirror
    // matched this dataset's, and one function owning both is simpler than
    // keeping two in step.
    let mesh = scanner.mesh();
    let source_points = mesh.vertices.len();
    let (seed, seed_colors) =
        decimate_init(&mesh.vertices, mesh.colors.as_deref(), MAX_INIT_POINTS);
    let seed: Vec<Vector3<f32>> = if unmirror {
        seed.iter().map(mirror_point).collect()
    } else {
        seed
    };
    let points = seed.len();
    if source_points > points {
        println!(
            "[scan] init cloud: {points} of {source_points} vertices (capped at {MAX_INIT_POINTS})"
        );
    }
    if points > 0 {
        save_colored_points_ply(&dir.join(INIT_PLY), &seed, seed_colors.as_deref())?;
    } else {
        println!("[scan] no surface to seed the splats with; the trainer will start from its own");
    }

    // Columns were reversed, so the principal point moves with them.
    let intrinsics = if unmirror {
        mirror_intrinsics(&intrinsics, width)
    } else {
        intrinsics
    };
    let transforms = build_transforms(&intrinsics, width, height, &frames, points > 0);
    fs::write(dir.join(TRANSFORMS), transforms)?;

    Ok(DatasetReport {
        dir: dir.to_path_buf(),
        views: frames.len(),
        points,
        skipped,
    })
}

/// Keep at most `max_points` seed vertices by taking the first vertex in each
/// voxel. The voxel starts at 2 cm and is enlarged until the cloud fits.
fn decimate_init(
    points: &[Vector3<f32>],
    colors: Option<&[[u8; 3]]>,
    max_points: usize,
) -> (Vec<Vector3<f32>>, Option<Vec<[u8; 3]>>) {
    let max_points = max_points.max(1);
    let colors = colors.filter(|colors| colors.len() == points.len());
    if points.len() <= max_points {
        return (points.to_vec(), colors.map(|colors| colors.to_vec()));
    }

    let mut voxel = 0.02f32;
    for _ in 0..12 {
        let inverse = 1.0 / voxel;
        let mut seen = HashSet::new();
        let mut kept = Vec::new();
        let mut kept_colors = colors.map(|_| Vec::new());

        for (index, point) in points.iter().enumerate() {
            let key = [
                (point.x * inverse).floor() as i32,
                (point.y * inverse).floor() as i32,
                (point.z * inverse).floor() as i32,
            ];
            if !seen.insert(key) {
                continue;
            }
            kept.push(*point);
            if let (Some(colors), Some(kept_colors)) = (colors, kept_colors.as_mut()) {
                kept_colors.push(colors[index]);
            }
        }

        if kept.len() <= max_points || kept.len() == points.len() {
            return (kept, kept_colors);
        }
        voxel *= 1.25;
    }

    // A pathological cloud can still miss the target after enlarging the
    // voxels. A uniform stride is coarse, but it is bounded and never worse
    // than handing the trainer millions of points.
    let stride = points.len().div_ceil(max_points);
    let mut kept = Vec::new();
    let mut kept_colors = colors.map(|_| Vec::new());
    for (index, point) in points.iter().enumerate().step_by(stride) {
        kept.push(*point);
        if let (Some(colors), Some(kept_colors)) = (colors, kept_colors.as_mut()) {
            kept_colors.push(colors[index]);
        }
    }
    (kept, kept_colors)
}

/// Write one image and one mask per view, and describe each as a frame entry.
///
/// Split out from [`write_dataset`] so the per-view logic can be exercised
/// without a scanner, and therefore without a sensor.
fn write_views(
    views: &[ColorView],
    image_dir: &Path,
    mask_dir: &Path,
    depth_dir: &Path,
    unmirror: bool,
) -> io::Result<(Vec<Frame>, usize)> {
    let Some(first) = views.first() else {
        return Err(invalid(
            "no colour views to export: a dataset needs `live --color`, a colour \
             recording, or the synthetic room",
        ));
    };

    // One camera model has to describe every frame in the file, so a mixed set
    // of image sizes cannot be written honestly.
    let (width, height) = (first.width, first.height);
    if views
        .iter()
        .any(|view| view.width != width || view.height != height)
    {
        return Err(invalid(
            "colour views do not share one image size, so a single camera model cannot describe them",
        ));
    }

    // Created here rather than by the caller so this is usable on its own.
    fs::create_dir_all(image_dir)?;
    fs::create_dir_all(mask_dir)?;
    fs::create_dir_all(depth_dir)?;

    let mut frames: Vec<Frame> = Vec::with_capacity(views.len());
    let mut skipped = 0;

    for view in views {
        // `NaN` is not JSON, and a view that cannot be described is worse than a
        // view that is left out. Nothing upstream should produce one; this only
        // keeps the file well-formed if something does.
        if !pose_is_finite(&view.pose) {
            skipped += 1;
            continue;
        }

        // Numbered by position among the frames actually written, so the
        // sequence has no gaps even when a view above was skipped.
        let name = format!("{:05}.png", frames.len());

        // A real 512x424 frame is far larger than Brush's header sniff, but a
        // small capture (the synthetic room) is not, and would fail to load.
        let (color, valid, pose);
        let (color, valid, pose) = if unmirror {
            color = flip_rows(width, height, 3, &view.color);
            valid = flip_rows(width, height, 1, &view.valid);
            pose = mirror_pose(&view.pose);
            (&color[..], &valid[..], &pose)
        } else {
            (&view.color[..], &view.valid[..], &view.pose)
        };

        let image = png::encode_rgb8(width as u32, height as u32, color)?;
        let image = png::pad_to_min_size(image, MIN_IMAGE_BYTES);
        fs::write(image_dir.join(&name), image)?;

        let mask = validity_mask(width, height, valid);
        let mask = png::encode_rgb8(width as u32, height as u32, &mask)?;
        fs::write(mask_dir.join(&name), mask)?;

        let mut depth_mm = depth_to_mm(&view.depth, width * height);
        if unmirror {
            depth_mm = flip_rows(width, height, 1, &depth_mm);
        }
        let depth_png = png::encode_gray16(width as u32, height as u32, &depth_mm)?;
        fs::write(depth_dir.join(&name), depth_png)?;

        frames.push(Frame {
            file_path: format!("{IMAGE_DIR}/{name}"),
            depth_path: format!("{DEPTH_DIR}/{name}"),
            matrix: camera_to_world(pose),
        });
    }

    if frames.is_empty() {
        return Err(invalid(
            "every colour view had an unusable pose, so nothing was written",
        ));
    }

    Ok((frames, skipped))
}

/// Reverse every row of a `width` x `height` image with `channels` elements per
/// pixel: the picture a camera would take of the mirrored scene.
fn flip_rows<T: Copy>(width: usize, height: usize, channels: usize, data: &[T]) -> Vec<T> {
    let row = width * channels;
    let mut out = data.to_vec();
    for y in 0..height {
        let start = y * row;
        let Some(line) = out.get_mut(start..start + row) else {
            break;
        };
        for x in 0..width / 2 {
            for c in 0..channels {
                line.swap(x * channels + c, (width - 1 - x) * channels + c);
            }
        }
    }
    out
}

/// The same camera after its image columns are reversed: only the principal
/// point's x moves, to `width - cx`. The focal lengths and `cy` are untouched.
///
/// This is not a no-op even though it looks like one for a centred lens: the
/// Kinect's depth camera has `cx` about 261 on a 512-wide image, so skipping it
/// would shift every ray by around ten pixels.
fn mirror_intrinsics(intrinsics: &Intrinsics, width: usize) -> Intrinsics {
    Intrinsics {
        cx: width as f32 - intrinsics.cx,
        ..*intrinsics
    }
}

/// Reflect a point through the plane `x = 0`.
fn mirror_point(point: &Vector3<f32>) -> Vector3<f32> {
    Vector3::new(-point.x, point.y, point.z)
}

/// The pose of the same camera in the mirrored world: `M * T * M` with
/// `M = diag(-1, 1, 1)`.
///
/// Conjugating by a reflection gives a rotation again, not a reflection (the
/// determinant is `(-1) * 1 * (-1) = 1`), so the result is an ordinary rigid
/// transform. Element-wise, `R'[i][j] = m_i * m_j * R[i][j]` with `m = (-1, 1, 1)`,
/// and the translation's x is negated.
fn mirror_pose(pose: &Isometry3<f32>) -> Isometry3<f32> {
    let r = pose.rotation.to_rotation_matrix();
    let m = [-1.0f32, 1.0, 1.0];
    let mut mirrored = nalgebra::Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            mirrored[(i, j)] = m[i] * m[j] * r[(i, j)];
        }
    }
    Isometry3::from_parts(
        Translation3::from(mirror_point(&pose.translation.vector)),
        UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(mirrored)),
    )
}

/// The camera-to-world matrix in the convention the dataset formats use.
///
/// `pose` is world-from-camera: its rotation's columns are the camera's x, y and
/// z axes in world coordinates, in the computer-vision convention the scanner
/// works in (x right, y down, z forward). The file wants the OpenGL convention
/// (x right, y up, z back), so the camera's y and z axes are negated. The
/// translation is the camera centre either way and is copied.
///
/// Negating two axes is a rotation, not a reflection -- the determinant stays
/// +1 -- so the matrix handed over is still a rigid transform.
fn camera_to_world(pose: &Isometry3<f32>) -> [[f32; 4]; 4] {
    let rotation = pose.rotation.to_rotation_matrix();
    let m = rotation.matrix();
    let t = pose.translation.vector;

    [
        [m[(0, 0)], -m[(0, 1)], -m[(0, 2)], t.x],
        [m[(1, 0)], -m[(1, 1)], -m[(1, 2)], t.y],
        [m[(2, 0)], -m[(2, 1)], -m[(2, 2)], t.z],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn pose_is_finite(pose: &Isometry3<f32>) -> bool {
    let t = pose.translation.vector;
    t.x.is_finite()
        && t.y.is_finite()
        && t.z.is_finite()
        && pose.rotation.as_vector().iter().all(|c| c.is_finite())
}

/// White where registration copied a colour sample, black where it did not.
///
/// Registration leaves the colour buffer at zero wherever it wrote nothing, and
/// zero is also a real black pixel, so the buffer alone cannot say which pixels
/// are real. The mask is the only record, and handing it over lets the trainer
/// ignore the holes instead of training against black.
///
/// Written as RGB rather than greyscale so it goes out through the same encoder
/// as the images. A binary image deflates to almost nothing either way.
fn validity_mask(width: usize, height: usize, valid: &[u8]) -> Vec<u8> {
    let pixels = width * height;
    let mut mask = vec![0u8; pixels * 3];

    for (index, flag) in valid.iter().take(pixels).enumerate() {
        if *flag != 0 {
            mask[index * 3..index * 3 + 3].fill(255);
        }
    }

    mask
}

/// Convert metric depth (`m`) to 16-bit unsigned millimetres (`0` = invalid).
fn depth_to_mm(depth: &[f32], pixels: usize) -> Vec<u16> {
    depth
        .iter()
        .take(pixels)
        .map(|&d| {
            if d.is_finite() && d > 0.0 {
                (d * 1000.0).round().clamp(1.0, u16::MAX as f32) as u16
            } else {
                0
            }
        })
        .collect()
}

/// Build `transforms.json`.
///
/// `camera_model` is deliberately absent rather than spelled out. The registered
/// colour is already undistorted and lives on the depth camera's grid, so a
/// pinhole with the depth intrinsics is exact and no distortion coefficients
/// apply. Absent is the layout's documented "pinhole" spelling; the obvious
/// `"PINHOLE"` string is not one the trainer accepts, and a file carrying it is
/// rejected rather than misread.
fn build_transforms(
    intrinsics: &Intrinsics,
    width: usize,
    height: usize,
    frames: &[Frame],
    has_init: bool,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"fl_x\": {},\n", intrinsics.fx));
    out.push_str(&format!("  \"fl_y\": {},\n", intrinsics.fy));
    out.push_str(&format!("  \"cx\": {},\n", intrinsics.cx));
    out.push_str(&format!("  \"cy\": {},\n", intrinsics.cy));
    out.push_str(&format!("  \"w\": {width},\n"));
    out.push_str(&format!("  \"h\": {height},\n"));
    out.push_str("  \"depth_unit_scale_factor\": 0.001,\n");
    if has_init {
        out.push_str(&format!("  \"ply_file_path\": \"{INIT_PLY}\",\n"));
    }

    out.push_str("  \"frames\": [\n");
    for (index, frame) in frames.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"file_path\": \"{}\",\n", frame.file_path));
        out.push_str(&format!(
            "      \"depth_file_path\": \"{}\",\n",
            frame.depth_path
        ));
        out.push_str("      \"transform_matrix\": [\n");
        for (row, values) in frame.matrix.iter().enumerate() {
            let comma = if row == 3 { "" } else { "," };
            out.push_str(&format!(
                "        [{}, {}, {}, {}]{comma}\n",
                values[0], values[1], values[2], values[3]
            ));
        }
        out.push_str("      ]\n");
        let comma = if index + 1 == frames.len() { "" } else { "," };
        out.push_str(&format!("    }}{comma}\n"));
    }
    out.push_str("  ]\n}\n");

    out
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

/// Write a small dataset from the synthetic room, for tests that need a real
/// one on disk (including the opt-in run against an installed Brush). Short
/// enough that every frame tracks.
#[cfg(test)]
pub(crate) fn write_synthetic_dataset(dir: &Path, shots: usize) -> DatasetReport {
    write_synthetic_dataset_with(dir, shots, true)
}

/// As [`write_synthetic_dataset`], choosing the orientation.
#[cfg(test)]
pub(crate) fn write_synthetic_dataset_with(
    dir: &Path,
    shots: usize,
    unmirror: bool,
) -> DatasetReport {
    write_dataset(&synthetic_scanner(shots), dir, unmirror).expect("write synthetic dataset")
}

/// A scanner that has seen the synthetic room, for tests that write more than
/// one dataset from the same scan. Two separate scanners do not agree on vertex
/// order (the volume's hash map iterates differently per instance), so only one
/// scanner makes two datasets comparable point by point.
#[cfg(test)]
pub(crate) fn synthetic_scanner(shots: usize) -> Scanner {
    use crate::scanner::{FrameColor, ScannerConfig};
    use geom::tsdf::TsdfParams;

    let intrinsics = crate::synthetic::intrinsics();
    let mut scanner = Scanner::new(
        intrinsics,
        ScannerConfig {
            tsdf: TsdfParams {
                voxel_size: 0.03,
                truncation: 0.12,
                ..TsdfParams::default()
            },
            ..ScannerConfig::default()
        },
    );
    for shot in &crate::synthetic::shots(shots) {
        let rendered = crate::synthetic::render(shot, &intrinsics);
        scanner.add_frame_with_color(
            &rendered.depth,
            crate::synthetic::WIDTH,
            crate::synthetic::HEIGHT,
            Some(FrameColor {
                rgb: &rendered.rgb,
                depth: &rendered.depth,
                valid: &rendered.valid,
                exposure: shot.brightness,
                gain: 1.0,
                gamma: 1.0,
            }),
        );
    }
    scanner
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scanner::{FrameColor, ScannerConfig};
    use geom::tsdf::TsdfParams;
    use nalgebra::{Matrix3, Rotation3, Translation3, Unit, UnitQuaternion, Vector3};
    use std::path::Path;

    const WIDTH: usize = 48;
    const HEIGHT: usize = 40;

    fn intrinsics() -> Intrinsics {
        Intrinsics {
            fx: 40.0,
            fy: 40.0,
            cx: WIDTH as f32 / 2.0,
            cy: HEIGHT as f32 / 2.0,
        }
    }

    fn pose(axis: Vector3<f32>, angle: f32, translation: Vector3<f32>) -> Isometry3<f32> {
        Isometry3::from_parts(
            Translation3::from(translation),
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle),
        )
    }

    /// A view on the depth grid, with `valid` all set except for the first
    /// `holes` pixels, as registration would leave them.
    fn view(at: Isometry3<f32>, holes: usize) -> ColorView {
        let pixels = WIDTH * HEIGHT;
        let mut valid = vec![1u8; pixels];
        for flag in valid.iter_mut().take(holes) {
            *flag = 0;
        }

        ColorView {
            color: vec![30u8; pixels * 3],
            width: WIDTH,
            height: HEIGHT,
            depth: vec![2.0f32; pixels],
            pose: at,
            intrinsics: intrinsics(),
            valid,
            exposure: 0.01,
            gain: 1.0,
            gamma: 2.2,
            frame_index: 0,
            tracking_quality: 1.0,
        }
    }

    /// Removes its directory when it goes out of scope, so a failing assertion
    /// does not leave a dataset behind.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!("scan-dataset-{}-{tag}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("temp dir");
            TempDir(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// The conversion the trainer applies when it reads a `transform_matrix`:
    /// negate the camera's y and z axes, then read the rotation and translation
    /// back out. Reproduced here so the round trip is tested against the actual
    /// convention rather than against a second copy of our own arithmetic.
    fn recover_pose(matrix: [[f32; 4]; 4]) -> Isometry3<f32> {
        let column = |j: usize| Vector3::new(matrix[0][j], matrix[1][j], matrix[2][j]);

        let rotation = Matrix3::from_columns(&[column(0), -column(1), -column(2)]);
        let translation = Vector3::new(matrix[0][3], matrix[1][3], matrix[2][3]);

        Isometry3::from_parts(
            Translation3::from(translation),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation)),
        )
    }

    fn assert_same_pose(expected: &Isometry3<f32>, actual: &Isometry3<f32>) {
        let delta = expected.inverse() * actual;
        assert!(
            delta.translation.vector.norm() < 1e-5,
            "translation drifted by {:?}",
            delta.translation.vector
        );
        assert!(
            delta.rotation.angle() < 1e-5,
            "rotation drifted by {} rad",
            delta.rotation.angle()
        );
    }

    const PNG_SIGNATURE: &[u8] = &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

    /// Width and height out of a PNG's IHDR chunk, which follows the 8-byte
    /// signature and its 4-byte length and 4-byte type.
    fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
        assert_eq!(&bytes[12..16], b"IHDR", "not a PNG with a leading IHDR");
        let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
        (width, height)
    }

    #[test]
    fn the_written_matrix_recovers_the_tracked_pose() {
        let poses = [
            Isometry3::identity(),
            pose(
                Vector3::z_axis().into_inner(),
                0.0,
                Vector3::new(1.0, -2.0, 3.0),
            ),
            pose(
                Vector3::z_axis().into_inner(),
                0.7,
                Vector3::new(0.3, 0.1, -0.2),
            ),
            pose(
                Vector3::x_axis().into_inner(),
                -1.2,
                Vector3::new(-0.4, 0.9, 0.05),
            ),
            // A compound rotation: no single axis, so a wrong-handed conversion
            // cannot pass by symmetry.
            pose(
                Vector3::new(1.0, 2.0, -3.0),
                2.1,
                Vector3::new(0.7, -0.3, 1.4),
            ),
        ];

        for original in poses {
            let recovered = recover_pose(camera_to_world(&original));
            assert_same_pose(&original, &recovered);
        }
    }

    #[test]
    fn the_matrix_is_a_rigid_transform() {
        // Negating two axes is a rotation, not a reflection. If it were a
        // reflection the trainer would recover a mirrored camera and every
        // splat would land on the wrong side of the room.
        let matrix = camera_to_world(&pose(
            Vector3::new(1.0, -1.0, 0.5),
            1.1,
            Vector3::new(2.0, 3.0, 4.0),
        ));

        let column = |j: usize| Vector3::new(matrix[0][j], matrix[1][j], matrix[2][j]);
        let (a, b, c) = (column(0), column(1), column(2));

        for (name, axis) in [("x", a), ("y", b), ("z", c)] {
            assert!(
                (axis.norm() - 1.0).abs() < 1e-5,
                "{name} axis is not unit: {}",
                axis.norm()
            );
        }
        assert!(a.dot(&b).abs() < 1e-5);
        assert!(a.dot(&c).abs() < 1e-5);
        assert!(b.dot(&c).abs() < 1e-5);
        assert!(
            a.cross(&b).dot(&c) > 0.0,
            "the camera axes are left-handed, so the matrix is a reflection"
        );
    }

    #[test]
    fn every_written_frame_matches_the_view_it_came_from() {
        // The images were captured in these poses. A file whose poses disagree
        // with the frames cannot train, however well-formed the JSON is.
        let poses = [
            pose(
                Vector3::z_axis().into_inner(),
                0.4,
                Vector3::new(0.0, 0.0, 0.0),
            ),
            pose(
                Vector3::y_axis().into_inner(),
                -0.9,
                Vector3::new(0.5, -0.1, 0.2),
            ),
        ];
        let views: Vec<_> = poses.iter().map(|p| view(*p, 0)).collect();

        let dir = TempDir::new("poses");
        let (frames, skipped) = write_views(
            &views,
            &dir.path().join("images"),
            &dir.path().join("masks"),
            &dir.path().join("depth"),
            false,
        )
        .expect("write");

        assert_eq!(skipped, 0);
        assert_eq!(frames.len(), 2);

        for (frame, expected) in frames.iter().zip(poses) {
            assert_same_pose(&expected, &recover_pose(frame.matrix));
        }
    }

    #[test]
    fn the_mask_is_white_only_where_registration_copied_colour() {
        let mut valid = vec![0u8; 4];
        valid[1] = 1;
        valid[3] = 1;

        let mask = validity_mask(2, 2, &valid);

        assert_eq!(&mask[0..3], &[0, 0, 0], "unregistered pixel 0");
        assert_eq!(&mask[3..6], &[255, 255, 255], "registered pixel 1");
        assert_eq!(&mask[6..9], &[0, 0, 0], "unregistered pixel 2");
        assert_eq!(&mask[9..12], &[255, 255, 255], "registered pixel 3");
    }

    #[test]
    fn views_are_numbered_without_gaps_when_one_is_dropped() {
        let good = pose(Vector3::z_axis().into_inner(), 0.2, Vector3::zeros());
        let broken = Isometry3::from_parts(
            Translation3::new(f32::NAN, 0.0, 0.0),
            UnitQuaternion::identity(),
        );

        let views = vec![view(good, 0), view(broken, 0), view(good, 0)];

        let dir = TempDir::new("gaps");
        let (frames, skipped) = write_views(
            &views,
            &dir.path().join("images"),
            &dir.path().join("masks"),
            &dir.path().join("depth"),
            false,
        )
        .expect("write");

        assert_eq!(skipped, 1, "the unusable pose should have been skipped");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].file_path, "images/00000.png");
        assert_eq!(frames[0].depth_path, "depth/00000.png");
        assert_eq!(frames[1].file_path, "images/00001.png");
        assert_eq!(frames[1].depth_path, "depth/00001.png");

        // Both surviving frames, and nothing for the dropped one.
        assert!(dir.path().join("images/00000.png").exists());
        assert!(dir.path().join("images/00001.png").exists());
        assert!(!dir.path().join("images/00002.png").exists());
        assert!(dir.path().join("depth/00000.png").exists());
        assert!(dir.path().join("depth/00001.png").exists());
        assert!(!dir.path().join("depth/00002.png").exists());
    }

    #[test]
    fn a_pose_that_cannot_be_written_is_detected() {
        assert!(pose_is_finite(&Isometry3::identity()));
        assert!(pose_is_finite(&pose(
            Vector3::new(1.0, 1.0, 1.0),
            0.5,
            Vector3::new(1.0, 2.0, 3.0)
        )));

        let nan_translation = Isometry3::from_parts(
            Translation3::new(f32::NAN, 0.0, 0.0),
            UnitQuaternion::identity(),
        );
        assert!(!pose_is_finite(&nan_translation));

        let infinite_rotation = Isometry3::from_parts(
            Translation3::identity(),
            Unit::new_unchecked(nalgebra::Quaternion::new(f32::INFINITY, 0.0, 0.0, 0.0)),
        );
        assert!(!pose_is_finite(&infinite_rotation));
    }

    #[test]
    fn mixed_image_sizes_are_refused_rather_than_described_by_one_camera() {
        let good = pose(Vector3::z_axis().into_inner(), 0.0, Vector3::zeros());
        let mut resized = view(good, 0);
        resized.height = HEIGHT + 8;
        resized.color = vec![0u8; WIDTH * (HEIGHT + 8) * 3];
        resized.depth = vec![2.0f32; WIDTH * (HEIGHT + 8)];
        resized.valid = vec![1u8; WIDTH * (HEIGHT + 8)];

        let dir = TempDir::new("mixed");
        let error = write_views(
            &[view(good, 0), resized],
            &dir.path().join("images"),
            &dir.path().join("masks"),
            &dir.path().join("depth"),
            false,
        )
        .expect_err("mixed sizes");

        assert!(error.to_string().contains("one image size"), "{error}");
    }

    #[test]
    fn the_transforms_file_has_the_fields_a_loader_requires() {
        let frames = vec![
            Frame {
                file_path: "images/00000.png".to_string(),
                depth_path: "depth/00000.png".to_string(),
                matrix: camera_to_world(&Isometry3::identity()),
            },
            Frame {
                file_path: "images/00001.png".to_string(),
                depth_path: "depth/00001.png".to_string(),
                matrix: camera_to_world(&pose(
                    Vector3::z_axis().into_inner(),
                    0.5,
                    Vector3::new(0.1, 0.2, 0.3),
                )),
            },
        ];

        let json = build_transforms(&intrinsics(), WIDTH, HEIGHT, &frames, true);

        assert!(json.contains("\"fl_x\": 40"), "{json}");
        assert!(json.contains("\"fl_y\": 40"), "{json}");
        assert!(json.contains(&format!("\"w\": {WIDTH}")), "{json}");
        assert!(json.contains(&format!("\"h\": {HEIGHT}")), "{json}");
        assert!(json.contains("\"depth_unit_scale_factor\": 0.001"), "{json}");
        assert!(json.contains("\"ply_file_path\": \"init.ply\""), "{json}");
        assert!(json.contains("\"transform_matrix\""), "{json}");
        assert!(json.contains("\"images/00000.png\""), "{json}");
        assert!(json.contains("\"images/00001.png\""), "{json}");
        assert!(json.contains("\"depth/00000.png\""), "{json}");
        assert!(json.contains("\"depth/00001.png\""), "{json}");

        // Two frames, so exactly one separating comma between them and no
        // trailing comma before `]`, which JSON does not allow.
        assert_eq!(json.matches("\"file_path\"").count(), 2, "{json}");
        assert!(!json.contains(",\n  ]"), "trailing comma in frames: {json}");

        // A loader rejects an unknown `camera_model` outright, and the obvious
        // name for a pinhole is one it does not accept.
        assert!(!json.contains("camera_model"), "{json}");
    }

    #[test]
    fn without_an_initial_cloud_the_file_does_not_reference_one() {
        let frames = vec![Frame {
            file_path: "images/00000.png".to_string(),
            depth_path: "depth/00000.png".to_string(),
            matrix: camera_to_world(&Isometry3::identity()),
        }];

        let json = build_transforms(&intrinsics(), WIDTH, HEIGHT, &frames, false);
        assert!(!json.contains("ply_file_path"), "{json}");
    }

    #[test]
    fn the_json_is_balanced_enough_to_parse() {
        // No JSON parser is in the dependency tree, so check the shape the
        // loader actually needs: the braces and brackets balance, and the depth
        // of nesting matches a single `frames` array of objects.
        let frames: Vec<_> = (0..3)
            .map(|i| Frame {
                file_path: format!("images/{i:05}.png"),
                depth_path: format!("depth/{i:05}.png"),
                matrix: camera_to_world(&pose(
                    Vector3::new(1.0, 1.0, 0.0),
                    0.3 * i as f32,
                    Vector3::new(i as f32, 0.0, 0.0),
                )),
            })
            .collect();

        let json = build_transforms(&intrinsics(), WIDTH, HEIGHT, &frames, true);

        let mut depth = 0i32;
        for ch in json.chars() {
            match ch {
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0, "closed a bracket that was never opened: {json}");
        }
        assert_eq!(depth, 0, "unbalanced brackets: {json}");
        assert_eq!(json.matches("\"transform_matrix\"").count(), 3, "{json}");
    }

    /// A flat wall two metres away. The first frame is always fused, and a plane
    /// gives a surface for the initial point cloud without needing a second pose.
    fn scanner_with_color() -> Scanner {
        let mut scanner = Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.05,
                    truncation: 0.2,
                    ..TsdfParams::default()
                },
                ..ScannerConfig::default()
            },
        );

        let depth = vec![2.0f32; WIDTH * HEIGHT];
        let pixels = WIDTH * HEIGHT;
        let rgb = vec![30u8; pixels * 3];
        let mut valid = vec![1u8; pixels];
        valid[0] = 0;
        valid[1] = 0;

        scanner.add_frame_with_color(
            &depth,
            WIDTH,
            HEIGHT,
            Some(FrameColor {
                rgb: &rgb,
                depth: &depth,
                valid: &valid,
                exposure: 0.01,
                gain: 1.0,
                gamma: 2.2,
            }),
        );

        scanner
    }

    #[test]
    fn a_scan_writes_images_masks_transforms_and_an_initial_cloud() {
        let scanner = scanner_with_color();
        let dir = TempDir::new("round-trip");

        let report = write_dataset(&scanner, dir.path(), false).expect("write");

        assert_eq!(report.views, 1);
        assert_eq!(report.skipped, 0);
        assert!(
            report.points > 0,
            "the wall should have extracted a surface to seed from"
        );

        let transforms = fs::read_to_string(dir.path().join("transforms.json")).expect("json");
        assert!(transforms.contains("\"images/00000.png\""), "{transforms}");
        assert!(transforms.contains("\"ply_file_path\""), "{transforms}");

        // `png::decode_rgb8` is test-only inside `geom`, so it is not visible
        // here. The IHDR chunk carries the dimensions in the clear, which is
        // enough to show the image really is the depth grid and not a stub.
        let image = fs::read(dir.path().join("images/00000.png")).expect("image");
        assert_eq!(&image[..8], PNG_SIGNATURE);
        assert_eq!(png_dimensions(&image), (WIDTH as u32, HEIGHT as u32));

        let mask = fs::read(dir.path().join("masks/00000.png")).expect("mask");
        assert_eq!(&mask[..8], PNG_SIGNATURE);
        assert_eq!(png_dimensions(&mask), (WIDTH as u32, HEIGHT as u32));

        let ply = fs::read(dir.path().join("init.ply")).expect("ply");
        let header_end = ply
            .windows(11)
            .position(|w| w == b"end_header\n")
            .expect("ply header")
            + 11;
        let header = std::str::from_utf8(&ply[..header_end]).expect("utf8 header");

        assert!(header.contains("element vertex"), "{header}");
        assert!(header.contains("property uchar red"), "{header}");
        assert!(
            !header.contains("element face"),
            "a trainer reads this as a point cloud, so it must not carry faces"
        );
        // Gaussian attributes would override the loader's defaults with zeros.
        for forbidden in ["scale_0", "opacity", "rot_0", "f_dc_0"] {
            assert!(
                !header.contains(forbidden),
                "the initial cloud must not carry `{forbidden}`: {header}"
            );
        }

        // The cloud shares the poses' frame, so a wall seen from the origin sits
        // at positive z. A mirrored cloud would not.
        let mut positive_z = 0;
        for chunk in ply[header_end..].chunks_exact(15) {
            let z = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
            assert!(z.is_finite(), "the cloud carries a non-finite coordinate");
            if z > 1.0 {
                positive_z += 1;
            }
        }
        assert!(
            positive_z > 0,
            "the wall should sit in front of the camera at +z"
        );
    }

    #[test]
    fn a_scan_with_no_colour_cannot_produce_a_dataset() {
        let mut scanner = Scanner::new(intrinsics(), ScannerConfig::default());
        scanner.add_frame(&vec![2.0f32; WIDTH * HEIGHT], WIDTH, HEIGHT);

        let dir = TempDir::new("no-color");
        let error = write_dataset(&scanner, dir.path(), false).expect_err("no views");

        assert!(
            error.to_string().contains("live --color"),
            "the message should say how to get colour views: {error}"
        );
        assert!(!dir.path().join("transforms.json").exists());
    }

    #[test]
    fn no_views_at_all_is_reported_as_a_missing_colour_stream() {
        let dir = TempDir::new("empty");

        let error = write_views(
            &[],
            &dir.path().join("images"),
            &dir.path().join("masks"),
            &dir.path().join("depth"),
            false,
        )
        .expect_err("no views");

        assert!(error.to_string().contains("live --color"), "{error}");
    }

    #[test]
    fn views_that_are_all_unusable_are_an_error_not_an_empty_file() {
        let dir = TempDir::new("all-nan");
        let broken = Isometry3::from_parts(
            Translation3::new(f32::NAN, 0.0, 0.0),
            UnitQuaternion::identity(),
        );

        let error = write_views(
            &[view(broken, 0)],
            &dir.path().join("images"),
            &dir.path().join("masks"),
            &dir.path().join("depth"),
            false,
        )
        .expect_err("nothing usable");

        assert!(error.to_string().contains("unusable pose"), "{error}");
        assert!(
            !dir.path().join("images/00000.png").exists(),
            "a view with no usable pose must not leave an image behind"
        );
    }

    /// The whole exporter, driven by a scene with no sensor behind it.
    ///
    /// Every other test here builds a `Scanner` by hand out of synthetic arrays
    /// that were chosen to exercise one branch. This one walks a real capture
    /// through it -- tracking, fusion, registration, the colour views, the mesh,
    /// the poses -- so it is the only test that covers the path `scan live
    /// --color --dataset` actually takes, and it is the only one that can run to
    /// completion without a Kinect attached.
    #[test]
    fn a_synthetic_room_produces_a_dataset_a_trainer_can_read() {
        let intrinsics = crate::synthetic::intrinsics();
        let mut scanner = Scanner::new(
            intrinsics,
            ScannerConfig {
                // Finer than the default so a small room still extracts a
                // surface worth seeding from.
                tsdf: TsdfParams {
                    voxel_size: 0.03,
                    truncation: 0.12,
                    ..TsdfParams::default()
                },
                ..ScannerConfig::default()
            },
        );

        // Short enough that every frame tracks: the tracker is known to lose a
        // viewpoint that turns onto a bare wall around frame 20, and this test is
        // about the exporter, not about that.
        let shots = crate::synthetic::shots(16);
        for shot in &shots {
            let rendered = crate::synthetic::render(shot, &intrinsics);
            scanner.add_frame_with_color(
                &rendered.depth,
                crate::synthetic::WIDTH,
                crate::synthetic::HEIGHT,
                Some(FrameColor {
                    rgb: &rendered.rgb,
                    depth: &rendered.depth,
                    valid: &rendered.valid,
                    exposure: shot.brightness,
                    gain: 1.0,
                    gamma: 1.0,
                }),
            );
        }

        let dir = TempDir::new("synthetic");
        let report = write_dataset(&scanner, dir.path(), false).expect("write");

        assert_eq!(report.skipped, 0, "no synthetic frame should be dropped");
        assert!(
            report.views >= shots.len() - 1,
            "kept {} views out of {} shots",
            report.views,
            shots.len()
        );
        assert!(
            report.points > 0,
            "the room should have extracted a surface to seed from"
        );

        // Every view gets an image and a mask that agree with it.
        for index in 0..report.views {
            let name = format!("{index:05}.png");
            let image = fs::read(dir.path().join("images").join(&name)).expect("image");
            let mask = fs::read(dir.path().join("masks").join(&name)).expect("mask");

            assert_eq!(
                png_dimensions(&image),
                (
                    crate::synthetic::WIDTH as u32,
                    crate::synthetic::HEIGHT as u32
                )
            );
            assert_eq!(
                png_dimensions(&mask),
                (
                    crate::synthetic::WIDTH as u32,
                    crate::synthetic::HEIGHT as u32
                )
            );
            assert_ne!(
                image, mask,
                "the mask must not be a copy of the image at {name}"
            );
        }

        // The poses describe one scene at a metric scale: the room is 3 x 2.4 m,
        // so a frame that wandered metres away would mean the convention or the
        // anchoring is wrong, and one that never moved would mean the poses are
        // all the same matrix.
        let transforms = fs::read_to_string(dir.path().join("transforms.json")).expect("json");
        assert!(transforms.contains("\"ply_file_path\""), "{transforms}");

        let translations: Vec<f32> = scanner.trajectory().iter().map(|p| p.norm()).collect();
        let furthest = translations.iter().copied().fold(0.0f32, f32::max);
        assert!(
            furthest > 0.2,
            "the capture moved but the furthest pose is only {furthest:.3} m from the anchor"
        );
        assert!(
            furthest < 3.0,
            "a pose {furthest:.3} m from the anchor is outside the room"
        );
    }

    #[test]
    fn the_init_cloud_is_capped_by_voxels() {
        let mut points = Vec::new();
        let mut colors = Vec::new();
        for x in 0..50 {
            for y in 0..20 {
                for z in 0..10 {
                    points.push(Vector3::new(
                        x as f32 * 0.01,
                        y as f32 * 0.01,
                        z as f32 * 0.01,
                    ));
                    colors.push([x as u8, y as u8, z as u8]);
                }
            }
        }

        let (seed, seed_colors) = decimate_init(&points, Some(&colors), 1000);

        assert!(!seed.is_empty());
        assert!(seed.len() <= 1000, "kept {} points", seed.len());
        assert_eq!(seed_colors.as_ref().unwrap().len(), seed.len());
    }

    /// Regression: Brush 0.3.0 reads exactly 16,387 bytes from each image to
    /// learn its format and fails any shorter file with "early eof". A small
    /// capture's PNGs are well under that, so every image must be padded past it
    /// while still decoding to the same pixels.
    #[test]
    fn every_image_is_long_enough_for_brushs_header_sniff() {
        let dir = TempDir::new("brush-sniff");
        let report = write_synthetic_dataset(dir.path(), 8);
        assert!(report.views >= 7, "kept {} views", report.views);

        let mut checked = 0;
        for entry in fs::read_dir(dir.path().join("images")).expect("images") {
            let path = entry.expect("entry").path();
            let bytes = fs::read(&path).expect("read image");
            assert!(
                bytes.len() >= 16_387,
                "{} is {} bytes, below Brush's 16,387-byte header read",
                path.display(),
                bytes.len()
            );
            // `geom`'s decoder is test-only there, so read the structure directly:
            // signature, IHDR dimensions, and IEND still closing the file.
            assert_eq!(&bytes[..8], &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']);
            let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap()) as usize;
            let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap()) as usize;
            assert_eq!((width, height), (crate::synthetic::WIDTH, crate::synthetic::HEIGHT));
            assert_eq!(&bytes[bytes.len() - 8..bytes.len() - 4], b"IEND", "IEND must stay last");
            checked += 1;
        }
        assert_eq!(checked, report.views);
    }

    // --- the mirror: the dataset follows the un-mirrored mesh ---

    #[test]
    fn flip_rows_reverses_each_row_whatever_the_channel_count() {
        // 3 x 2 RGB: rows [a b c] [d e f], each pixel three distinct bytes.
        let rgb: Vec<u8> = (0..18).collect();
        let flipped = flip_rows(3, 2, 3, &rgb);
        assert_eq!(
            flipped,
            vec![
                6, 7, 8, 3, 4, 5, 0, 1, 2, //
                15, 16, 17, 12, 13, 14, 9, 10, 11
            ]
        );
        // One channel, even width, and flipping twice is the identity.
        assert_eq!(flip_rows(4, 1, 1, &[1, 2, 3, 4]), vec![4, 3, 2, 1]);
        assert_eq!(flip_rows(3, 2, 3, &flipped), rgb);
        // A buffer shorter than the image is left alone rather than indexed out of range.
        assert_eq!(flip_rows(3, 2, 3, &rgb[..6]), flip_rows(3, 2, 3, &rgb[..6]));
    }

    #[test]
    fn a_mirrored_pose_is_a_proper_rotation_that_undoes_itself() {
        let original = pose(
            Vector3::new(0.3, -0.7, 0.5),
            0.9,
            Vector3::new(1.0, 2.0, 3.0),
        );
        let mirrored = mirror_pose(&original);

        // Translation: only x changes sign.
        assert!((mirrored.translation.vector - Vector3::new(-1.0, 2.0, 3.0)).norm() < 1e-6);
        // Still a unit rotation, not a reflection or a scaled one.
        assert!((mirrored.rotation.quaternion().norm() - 1.0).abs() < 1e-5);
        assert!(
            (mirrored.rotation.to_rotation_matrix().matrix().determinant() - 1.0).abs() < 1e-5
        );
        // M * (M * T * M) * M is T again.
        assert_same_pose(&original, &mirror_pose(&mirrored));

        // A turn about y reverses (the mirror swaps left and right); a turn about
        // x does not, because x is the axis the mirror acts along.
        let yaw = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3),
        );
        assert!((mirror_pose(&yaw).rotation.angle() - 0.3).abs() < 1e-5);
        let yaw_axis = mirror_pose(&yaw).rotation.axis().unwrap();
        assert!((yaw_axis.into_inner() - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-5);
        let pitch = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.0),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.3),
        );
        assert_same_pose(&pitch, &mirror_pose(&pitch));
    }

    /// The property the whole mirror rests on: a point the original camera sees
    /// at column `u` must be seen at `width - u` by the mirrored camera looking at
    /// the mirrored world, in the same row, at the same depth.
    #[test]
    fn a_mirrored_dataset_sees_every_point_at_the_reflected_column() {
        // Off-centre on purpose, like the real sensor's: with a centred lens
        // `width - cx` equals `cx` and a missing correction would go unnoticed.
        let k = Intrinsics {
            cx: WIDTH as f32 / 2.0 + 3.7,
            ..intrinsics()
        };
        let project = |at: &Isometry3<f32>, cx: f32, p: &Vector3<f32>| {
            let c = at.inverse() * nalgebra::Point3::from(*p);
            (k.fx * c.x / c.z + cx, k.fy * c.y / c.z + k.cy, c.z)
        };

        let mirrored_cx = mirror_intrinsics(&k, WIDTH).cx;
        for (axis, angle, translation) in [
            (Vector3::new(0.0, 1.0, 0.0), 0.0, Vector3::new(0.0, 0.0, 0.0)),
            (Vector3::new(0.0, 1.0, 0.0), 0.4, Vector3::new(0.5, -0.1, 0.2)),
            (Vector3::new(0.3, -0.7, 0.5), 0.9, Vector3::new(-0.8, 0.3, 0.6)),
        ] {
            let camera = pose(axis, angle, translation);
            let mirrored_camera = mirror_pose(&camera);
            for point in [
                Vector3::new(0.2, 0.1, 2.5),
                Vector3::new(-0.4, -0.3, 3.0),
                Vector3::new(1.1, 0.5, 4.0),
            ] {
                let (u, v, z) = project(&camera, k.cx, &point);
                let (mu, mv, mz) = project(&mirrored_camera, mirrored_cx, &mirror_point(&point));
                assert!((mu - (WIDTH as f32 - u)).abs() < 1e-3, "column {mu} vs {}", WIDTH as f32 - u);
                assert!((mv - v).abs() < 1e-3, "row changed: {mv} vs {v}");
                assert!((mz - z).abs() < 1e-4, "depth changed: {mz} vs {z}");
            }
        }
    }

    /// A view whose pixels are all different and whose validity mask is not
    /// symmetric, so a missing or doubled flip cannot go unnoticed.
    fn asymmetric_view(at: Isometry3<f32>) -> ColorView {
        let mut v = view(at, 0);
        v.color = (0..WIDTH * HEIGHT * 3).map(|i| (i * 7 % 251) as u8).collect();
        // Left third invalid.
        v.valid = (0..WIDTH * HEIGHT)
            .map(|i| u8::from(i % WIDTH >= WIDTH / 3))
            .collect();
        v
    }

    fn decoded(path: &Path) -> Vec<u8> {
        geom::png::decode_rgb8(&fs::read(path).expect("read png"))
            .expect("decode png")
            .2
    }

    #[test]
    fn unmirroring_flips_images_masks_and_poses_together_and_only_when_asked() {
        let original = pose(
            Vector3::new(0.3, -0.7, 0.5),
            0.9,
            Vector3::new(1.0, 2.0, 3.0),
        );
        let source = asymmetric_view(original);

        for unmirror in [false, true] {
            let dir = TempDir::new(if unmirror { "flipped" } else { "raw" });
            let (frames, _) = write_views(
                std::slice::from_ref(&source),
                &dir.path().join("images"),
                &dir.path().join("masks"),
                &dir.path().join("depth"),
                unmirror,
            )
            .expect("write");

            let image = decoded(&dir.path().join("images/00000.png"));
            let mask = decoded(&dir.path().join("masks/00000.png"));
            let (expected_image, expected_valid, expected_pose) = if unmirror {
                (
                    flip_rows(WIDTH, HEIGHT, 3, &source.color),
                    flip_rows(WIDTH, HEIGHT, 1, &source.valid),
                    mirror_pose(&original),
                )
            } else {
                (source.color.clone(), source.valid.clone(), original)
            };
            assert_eq!(image, expected_image, "image (unmirror = {unmirror})");
            assert_eq!(
                mask,
                validity_mask(WIDTH, HEIGHT, &expected_valid),
                "mask (unmirror = {unmirror})"
            );
            assert_eq!(
                frames[0].matrix,
                camera_to_world(&expected_pose),
                "pose (unmirror = {unmirror})"
            );
        }

        // The two orientations really do differ, or the test above proves nothing.
        assert_ne!(
            flip_rows(WIDTH, HEIGHT, 3, &source.color),
            source.color,
            "the fixture must not be left-right symmetric"
        );
    }

    /// Numbers in a JSON-ish fragment, in order. Enough to read `transforms.json`
    /// without a JSON dependency.
    fn numbers(text: &str) -> Vec<f32> {
        text.split(|c: char| !(c.is_ascii_digit() || matches!(c, '-' | '.' | 'e' | 'E' | '+')))
            .filter_map(|token| token.parse().ok())
            .collect()
    }

    fn scalar(json: &str, key: &str) -> f32 {
        let after = json.split(&format!("\"{key}\":")).nth(1).expect("key present");
        numbers(after.lines().next().unwrap())[0]
    }

    fn matrices(json: &str) -> Vec<Vec<f32>> {
        json.split("\"transform_matrix\":")
            .skip(1)
            .map(|chunk| numbers(chunk)[..16].to_vec())
            .collect()
    }

    #[test]
    fn a_whole_dataset_is_mirrored_consistently_images_poses_principal_point_and_seed() {
        let raw = TempDir::new("whole-raw");
        let flipped = TempDir::new("whole-flipped");
        // One scan, written both ways, so the two datasets describe the same surface.
        let scanner = synthetic_scanner(6);
        let a = write_dataset(&scanner, raw.path(), false).expect("raw");
        let b = write_dataset(&scanner, flipped.path(), true).expect("flipped");
        assert_eq!((a.views, a.points), (b.views, b.points));

        let ja = fs::read_to_string(raw.path().join("transforms.json")).unwrap();
        let jb = fs::read_to_string(flipped.path().join("transforms.json")).unwrap();

        // Columns reversed, so the principal point reflects; nothing else about the camera moves.
        let width = scalar(&ja, "w");
        assert!((scalar(&ja, "cx") + scalar(&jb, "cx") - width).abs() < 1e-3);
        assert_eq!(scalar(&ja, "cy"), scalar(&jb, "cy"));
        assert_eq!(scalar(&ja, "fl_x"), scalar(&jb, "fl_x"));

        // Every camera centre is reflected through x = 0: only x changes sign.
        let (ma, mb) = (matrices(&ja), matrices(&jb));
        assert_eq!(ma.len(), mb.len());
        for (frame, (x, y)) in ma.iter().zip(&mb).enumerate() {
            assert!((x[3] + y[3]).abs() < 1e-5, "frame {frame}: x {} vs {}", x[3], y[3]);
            assert!((x[7] - y[7]).abs() < 1e-5, "frame {frame}: y changed");
            assert!((x[11] - y[11]).abs() < 1e-5, "frame {frame}: z changed");
        }

        // The seed cloud is the same points, reflected.
        let pa = geom::mesh::read_points_ply(&raw.path().join("init.ply")).unwrap();
        let pb = geom::mesh::read_points_ply(&flipped.path().join("init.ply")).unwrap();
        assert_eq!(pa.len(), pb.len());
        assert!(!pa.is_empty());
        for (index, (x, y)) in pa.iter().zip(&pb).enumerate() {
            assert!(
                (mirror_point(x) - y).norm() < 1e-6,
                "seed point {index}: raw {x:?} mirrored should be {:?}, got {y:?}",
                mirror_point(x)
            );
        }

        // And the images are the same pictures, reversed left to right.
        let image_a = decoded(&raw.path().join("images/00000.png"));
        let image_b = decoded(&flipped.path().join("images/00000.png"));
        assert_eq!(
            flip_rows(crate::synthetic::WIDTH, crate::synthetic::HEIGHT, 3, &image_a),
            image_b
        );
    }

    #[test]
    fn mirroring_moves_only_the_principal_point_x_and_uses_the_real_sensors_off_centre_value() {
        // The Kinect v2 depth camera, as the scanner reports it.
        let kinect = Intrinsics {
            fx: 367.13,
            fy: 367.13,
            cx: 261.08,
            cy: 211.21,
        };
        let mirrored = mirror_intrinsics(&kinect, 512);

        assert!((mirrored.cx - 250.92).abs() < 1e-3, "cx {}", mirrored.cx);
        assert!((kinect.cx - 256.0).abs() > 5.0, "the real lens is not centred");
        assert_eq!(mirrored.fx, kinect.fx);
        assert_eq!(mirrored.fy, kinect.fy);
        assert_eq!(mirrored.cy, kinect.cy);
        // Mirroring twice returns the original.
        assert!((mirror_intrinsics(&mirrored, 512).cx - kinect.cx).abs() < 1e-3);
    }
}

