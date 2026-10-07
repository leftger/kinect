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
//! `--mirror` and `--unmirror` flip X on the exported mesh so real-world left and
//! right line up. A dataset must not do that: the images, the poses and the
//! point cloud all have to stay in one frame, and the frame they are all in is
//! the sensor's. A reflected scan with reflected poses would train just as well,
//! but mixing a flipped point cloud with unflipped images would not train at
//! all. So this module writes everything unflipped, and says so.

use geom::coloring::ColorView;
use geom::mesh::save_colored_points_ply;
use geom::png;
use geom::Intrinsics;
use nalgebra::Isometry3;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use crate::scanner::Scanner;

/// Images live here. The trainer resolves each frame's `file_path` against the
/// directory holding `transforms.json`.
const IMAGE_DIR: &str = "images";
/// Masks live here, mirroring the image paths. The trainer finds a mask by
/// looking for a directory named `masks` whose path past that point matches the
/// image's, and matching the stem.
const MASK_DIR: &str = "masks";
const TRANSFORMS: &str = "transforms.json";
const INIT_PLY: &str = "init.ply";

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
    matrix: [[f32; 4]; 4],
}

/// Write the scan's colour views, poses and surface to `dir`.
///
/// Needs `scanner.color_views()`: the dataset is built from the colour stream, so
/// only a `live --color` scan has one. A depth-only scan has nothing to export.
pub fn write_dataset(scanner: &Scanner, dir: &Path) -> io::Result<DatasetReport> {
    let views = scanner.color_views();

    let image_dir = dir.join(IMAGE_DIR);
    let mask_dir = dir.join(MASK_DIR);

    let (frames, skipped) = write_views(views, &image_dir, &mask_dir)?;

    // Every view carries the same intrinsics: the scanner copies one `Intrinsics`
    // onto each. Taking the first is therefore not a sample, it is the value.
    let intrinsics = views[0].intrinsics;
    let width = views[0].width;
    let height = views[0].height;

    // The mesh is in the same frame as the poses: `Scanner::mesh` extracts from
    // the volume without applying the export mirror. Its colours come from the
    // same views that were just written, so the cloud starts near the colour the
    // trainer is about to optimise towards.
    //
    // This extracts and paints the surface a second time, the first being the
    // mesh export. That is deliberate: reusing the exported mesh would mean
    // un-mirroring it again first, and the mirror is exactly the thing this
    // module exists to keep away from the dataset.
    let mesh = scanner.mesh();
    let points = mesh.vertices.len();
    if points > 0 {
        save_colored_points_ply(&dir.join(INIT_PLY), &mesh.vertices, mesh.colors.as_deref())?;
    } else {
        println!("[scan] no surface to seed the splats with; the trainer will start from its own");
    }

    let transforms = build_transforms(&intrinsics, width, height, &frames, points > 0);
    fs::write(dir.join(TRANSFORMS), transforms)?;

    Ok(DatasetReport {
        dir: dir.to_path_buf(),
        views: frames.len(),
        points,
        skipped,
    })
}

/// Write one image and one mask per view, and describe each as a frame entry.
///
/// Split out from [`write_dataset`] so the per-view logic can be exercised
/// without a scanner, and therefore without a sensor.
fn write_views(
    views: &[ColorView],
    image_dir: &Path,
    mask_dir: &Path,
) -> io::Result<(Vec<Frame>, usize)> {
    let Some(first) = views.first() else {
        return Err(invalid(
            "no colour views to export: a dataset is built from the colour stream, \
             so capture it with `live --color`",
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

        let image = png::encode_rgb8(width as u32, height as u32, &view.color)?;
        fs::write(image_dir.join(&name), image)?;

        let mask = validity_mask(width, height, &view.valid);
        let mask = png::encode_rgb8(width as u32, height as u32, &mask)?;
        fs::write(mask_dir.join(&name), mask)?;

        frames.push(Frame {
            file_path: format!("{IMAGE_DIR}/{name}"),
            matrix: camera_to_world(&view.pose),
        });
    }

    if frames.is_empty() {
        return Err(invalid(
            "every colour view had an unusable pose, so nothing was written",
        ));
    }

    Ok((frames, skipped))
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
    if has_init {
        out.push_str(&format!("  \"ply_file_path\": \"{INIT_PLY}\",\n"));
    }

    out.push_str("  \"frames\": [\n");
    for (index, frame) in frames.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"file_path\": \"{}\",\n", frame.file_path));
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
            pose(Vector3::new(1.0, 2.0, -3.0), 2.1, Vector3::new(0.7, -0.3, 1.4)),
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
        let (frames, skipped) =
            write_views(&views, &dir.path().join("images"), &dir.path().join("masks"))
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
        let (frames, skipped) =
            write_views(&views, &dir.path().join("images"), &dir.path().join("masks"))
                .expect("write");

        assert_eq!(skipped, 1, "the unusable pose should have been skipped");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].file_path, "images/00000.png");
        assert_eq!(frames[1].file_path, "images/00001.png");

        // Both surviving frames, and nothing for the dropped one.
        assert!(dir.path().join("images/00000.png").exists());
        assert!(dir.path().join("images/00001.png").exists());
        assert!(!dir.path().join("images/00002.png").exists());
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
        )
        .expect_err("mixed sizes");

        assert!(error.to_string().contains("one image size"), "{error}");
    }

    #[test]
    fn the_transforms_file_has_the_fields_a_loader_requires() {
        let frames = vec![
            Frame {
                file_path: "images/00000.png".to_string(),
                matrix: camera_to_world(&Isometry3::identity()),
            },
            Frame {
                file_path: "images/00001.png".to_string(),
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
        assert!(json.contains("\"ply_file_path\": \"init.ply\""), "{json}");
        assert!(json.contains("\"transform_matrix\""), "{json}");
        assert!(json.contains("\"images/00000.png\""), "{json}");
        assert!(json.contains("\"images/00001.png\""), "{json}");

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

        let report = write_dataset(&scanner, dir.path()).expect("write");

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
        let error = write_dataset(&scanner, dir.path()).expect_err("no views");

        assert!(
            error.to_string().contains("live --color"),
            "the message should say how to get colour views: {error}"
        );
        assert!(!dir.path().join("transforms.json").exists());
    }

    #[test]
    fn no_views_at_all_is_reported_as_a_missing_colour_stream() {
        let dir = TempDir::new("empty");

        let error = write_views(&[], &dir.path().join("images"), &dir.path().join("masks"))
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
        )
        .expect_err("nothing usable");

        assert!(error.to_string().contains("unusable pose"), "{error}");
        assert!(
            !dir.path().join("images/00000.png").exists(),
            "a view with no usable pose must not leave an image behind"
        );
    }
}
