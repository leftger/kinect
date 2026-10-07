//! A synthetic coloured room, so the whole pipeline can be exercised without a
//! sensor.
//!
//! Everything downstream of `add_frame_with_color` -- tracking, fusion,
//! registration, the colour views, the mesh, and the dataset exporter -- is
//! reachable from a depth frame plus a colour frame plus a pose. Supplying those
//! from an analytic scene rather than a camera buys two things that no capture
//! can:
//!
//! * **Ground truth.** The pose of every frame is known exactly, so the poses in
//!   a written dataset can be *falsified* rather than eyeballed. Nothing else in
//!   this repo can do that: a real capture has no reference trajectory, so every
//!   pose figure is unfalsifiable.
//! * **No sensor.** The exporter can be run to completion on a machine with no
//!   Kinect attached, which is the difference between a format that has been
//!   reasoned about and one that has produced a file.
//!
//! The geometry is a box interior traced analytically, the same shape
//! `odometry`'s tests use, because a single plane leaves three degrees of
//! freedom unobservable and leaves holes wherever a ray misses -- and normals
//! come from neighbouring pixels, so holes poison the normal map.
//!
//! Colour is a 3D checker rather than flat wall colours. Flat walls are
//! trivially textureless, which makes a radiance field both uninteresting to
//! train and useless for spotting a pose that is subtly wrong; the checker gives
//! every surface structure that only lines up when the poses do.

use geom::Intrinsics;
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

/// Small on purpose: a synthetic capture is a fixture, not a scene.
pub const WIDTH: usize = 160;
pub const HEIGHT: usize = 120;

const MIN_X: f32 = -1.5;
const MAX_X: f32 = 1.5;
const MIN_Y: f32 = -1.2;
const MAX_Y: f32 = 1.2;
const MIN_Z: f32 = 0.0;
const MAX_Z: f32 = 4.0;

/// Checker cell size, in metres. Small enough that a wrong pose shows up as
/// smeared structure rather than a smooth blur.
const CHECKER: f32 = 0.25;

/// One rendered frame: what `FrameColor` needs, plus the depth.
pub struct Rendered {
    /// Camera-space depth in metres, `NaN` where a ray hit nothing.
    pub depth: Vec<f32>,
    /// RGB, `WIDTH * HEIGHT * 3`.
    pub rgb: Vec<u8>,
    /// One byte per pixel, non-zero where colour is usable.
    pub valid: Vec<u8>,
}

/// A frame's pose and the exposure the render was made at.
#[derive(Clone, Copy)]
pub struct Shot {
    /// Camera-to-world, the same convention the scanner uses internally.
    pub pose: Isometry3<f32>,
    /// Linear-light brightness multiplier, standing in for the colour camera's
    /// auto-exposure. `1.0` is the reference exposure.
    pub brightness: f32,
}

pub fn intrinsics() -> Intrinsics {
    Intrinsics {
        fx: 120.0,
        fy: 120.0,
        cx: WIDTH as f32 / 2.0,
        cy: HEIGHT as f32 / 2.0,
    }
}

/// A short handheld-looking path: a shallow arc across the room while panning to
/// keep the far wall in frame.
///
/// Consecutive shots overlap heavily, because the tracker has to be able to
/// follow this at all -- a fixture that cannot be tracked tests nothing about
/// the exporter.
pub fn shots(count: usize) -> Vec<Shot> {
    assert!(count > 1, "a path needs at least two shots");

    (0..count)
        .map(|index| {
            let t = index as f32 / (count - 1) as f32;

            let eye = Vector3::new(
                -0.9 + 1.8 * t,
                0.15 * (t * std::f32::consts::TAU).sin(),
                1.6,
            );

            // Camera looks down +z, with y down, so yaw is about y and pitch is
            // about x. A gentle pan keeps every view on the far wall.
            let yaw = -0.30 + 0.60 * t;
            let pitch = -0.04;
            let rotation = UnitQuaternion::from_euler_angles(pitch, yaw, 0.0);

            // Auto-exposure drift: the colour camera does not hold one exposure,
            // and neither does this.
            let brightness = 1.0 + 0.30 * (t * 6.0).sin();

            Shot {
                pose: Isometry3::from_parts(Translation3::from(eye), rotation),
                brightness,
            }
        })
        .collect()
}

/// Ray-trace the room interior from `shot`, returning depth and colour.
pub fn render(shot: &Shot, intrinsics: &Intrinsics) -> Rendered {
    let rotation = shot.pose.rotation;
    let eye = shot.pose.translation.vector;

    let mut depth = vec![f32::NAN; WIDTH * HEIGHT];
    let mut rgb = vec![0u8; WIDTH * HEIGHT * 3];

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let direction = Vector3::new(
                (x as f32 + 0.5 - intrinsics.cx) / intrinsics.fx,
                (y as f32 + 0.5 - intrinsics.cy) / intrinsics.fy,
                1.0,
            );

            let Some((distance, hit, base)) = cast(&eye, &(rotation * direction)) else {
                continue;
            };

            // `direction.z` is 1, so the ray parameter *is* camera-space depth.
            let index = y * WIDTH + x;
            depth[index] = distance;

            let linear = shade(&hit, base, shot.brightness);
            rgb[index * 3] = linear[0];
            rgb[index * 3 + 1] = linear[1];
            rgb[index * 3 + 2] = linear[2];
        }
    }

    Rendered {
        depth,
        rgb,
        valid: mask(),
    }
}

/// Nearest hit in the room, as (camera-space depth, world point, surface colour).
fn cast(eye: &Vector3<f32>, direction: &Vector3<f32>) -> Option<(f32, Vector3<f32>, [u8; 3])> {
    let mut nearest = f32::INFINITY;
    let mut colour = [128u8, 128, 128];

    // The eye is inside a convex box, so the ray leaves through exactly one face
    // and that is the smallest positive crossing. Testing each crossing for "is
    // this point inside the box" looks equivalent and is not: the crossing is
    // computed *at* the face, so roundoff puts `hit.z` a hair past `MAX_Z` and
    // the face that was actually hit gets rejected. Rows 0 and 239 of the first
    // shot came back empty that way. The smallest positive crossing needs no
    // tolerance and cannot reject the right answer.
    for (normal, offset) in walls() {
        let denominator = normal.dot(direction);
        if denominator.abs() < 1e-6 {
            continue;
        }

        // n . (eye + t * direction) = offset
        let t = (offset - normal.dot(eye)) / denominator;
        if t <= 0.0 || t >= nearest {
            continue;
        }

        nearest = t;
        colour = wall_colour(&(eye + direction * t));
    }

    for block in furniture() {
        if let Some(t) = cast_block(eye, direction, &block) {
            if t < nearest {
                nearest = t;
                colour = block.colour;
            }
        }
    }

    nearest
        .is_finite()
        .then(|| (nearest, eye + direction * nearest, colour))
}

fn walls() -> [(Vector3<f32>, f32); 6] {
    [
        (Vector3::new(1.0, 0.0, 0.0), MIN_X),
        (Vector3::new(-1.0, 0.0, 0.0), -MAX_X),
        (Vector3::new(0.0, 1.0, 0.0), MIN_Y),
        (Vector3::new(0.0, -1.0, 0.0), -MAX_Y),
        (Vector3::new(0.0, 0.0, 1.0), MIN_Z),
        (Vector3::new(0.0, 0.0, -1.0), -MAX_Z),
    ]
}

/// Furniture: axis-aligned boxes standing in the room.
///
/// A bare box room is *degenerate* for point-to-plane ICP. Three orthogonal
/// planes pin down the three translations but leave sliding along a flat wall
/// nearly free, and point-to-plane has no term that resists it. So a perfect,
/// noise-free render of an empty room tracks worse than a noisy real capture --
/// which is backwards for a fixture, and is not the tracker's fault.
///
/// A real room is trackable because it has clutter, so this one does too. Four
/// boxes, varied in size and depth, on the floor and the ceiling.
///
/// Placement is not free. An object only has to sit a few centimetres from the
/// camera path for the camera to sweep right over it, and a silhouette crossing
/// the whole frame between two shots is a discontinuity the tracker gets blamed
/// for. Every box here keeps at least 0.4 m of clearance from the path, which
/// still leaves them well inside the view.
struct Block {
    min: Vector3<f32>,
    max: Vector3<f32>,
    colour: [u8; 3],
}

fn furniture() -> [Block; 4] {
    [
        Block {
            min: Vector3::new(-1.35, 0.55, 2.60),
            max: Vector3::new(-0.70, 1.20, 3.30),
            colour: [200, 120, 40],
        },
        Block {
            min: Vector3::new(0.30, 0.55, 0.50),
            max: Vector3::new(1.05, 1.20, 1.20),
            colour: [110, 90, 200],
        },
        Block {
            min: Vector3::new(-0.55, -1.20, 2.20),
            max: Vector3::new(0.15, -0.60, 2.90),
            colour: [60, 180, 190],
        },
        Block {
            min: Vector3::new(0.45, -1.20, 0.40),
            max: Vector3::new(1.10, -0.65, 1.10),
            colour: [190, 180, 60],
        },
    ]
}

/// Slab test. `None` where the ray misses, or the camera is inside the box.
fn cast_block(eye: &Vector3<f32>, direction: &Vector3<f32>, block: &Block) -> Option<f32> {
    let mut enter = f32::NEG_INFINITY;
    let mut leave = f32::INFINITY;

    for axis in 0..3 {
        let d = direction[axis];
        let (low, high) = (block.min[axis], block.max[axis]);

        if d.abs() < 1e-6 {
            // Parallel to this pair of faces: between them always, or never.
            if eye[axis] < low || eye[axis] > high {
                return None;
            }
            continue;
        }

        let mut near = (low - eye[axis]) / d;
        let mut far = (high - eye[axis]) / d;
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }

        enter = enter.max(near);
        leave = leave.min(far);
        if enter > leave {
            return None;
        }
    }

    if enter <= 0.0 {
        return None;
    }

    Some(enter)
}

/// Colour of a surface point: the surface's own hue, cut by a 3D checker so
/// every face carries structure in two directions.
fn shade(hit: &Vector3<f32>, base: [u8; 3], brightness: f32) -> [u8; 3] {
    let parity = (hit.x / CHECKER).floor() + (hit.y / CHECKER).floor() + (hit.z / CHECKER).floor();
    let dark = (parity as i64).rem_euclid(2) == 1;
    let scale = brightness * if dark { 0.45 } else { 1.0 };

    let channel = |value: u8| ((value as f32 * scale).clamp(0.0, 255.0)) as u8;
    [channel(base[0]), channel(base[1]), channel(base[2])]
}

fn wall_colour(hit: &Vector3<f32>) -> [u8; 3] {
    if hit.x < MIN_X + 1e-3 {
        [220, 70, 70]
    } else if hit.x > MAX_X - 1e-3 {
        [70, 200, 120]
    } else if hit.y < MIN_Y + 1e-3 {
        [180, 170, 150]
    } else if hit.y > MAX_Y - 1e-3 {
        [170, 180, 200]
    } else if hit.z < MIN_Z + 1e-3 {
        [90, 120, 230]
    } else {
        [230, 200, 90]
    }
}

/// The registration mask.
///
/// Registration always loses a border -- the colour frame has to be resampled
/// onto the depth grid and the outermost pixels have no source -- so the border
/// is dropped here too. The point is that the exporter is handed a non-trivial
/// mask rather than an all-white one; a mask that keeps everything would let a
/// masking bug pass unnoticed.
fn mask() -> Vec<u8> {
    let inset_x = (WIDTH as f32 * 0.02).ceil() as usize;
    let inset_y = (HEIGHT as f32 * 0.02).ceil() as usize;

    let mut valid = vec![255u8; WIDTH * HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if x < inset_x || y < inset_y || x >= WIDTH - inset_x || y >= HEIGHT - inset_y {
                valid[y * WIDTH + x] = 0;
            }
        }
    }

    valid
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consecutive_shots_render_to_a_continuous_view() {
        let intrinsics = intrinsics();
        let shots = shots(72);

        for (index, pair) in shots.windows(2).enumerate() {
            let before = render(&pair[0], &intrinsics);
            let after = render(&pair[1], &intrinsics);

            // A camera walking through a static scene cannot make a sliver of
            // the view jump by metres between frames: surfaces move by the step
            // and silhouettes sweep. Anything larger means the path clips
            // through geometry, and the tracker would be blamed for it.
            let jumped = before
                .depth
                .iter()
                .zip(&after.depth)
                .filter(|(a, b)| (**a - **b).abs() > 0.30)
                .count();
            let fraction = jumped as f32 / before.depth.len() as f32;

            assert!(
                fraction < 0.05,
                "shots {}->{} change {:.1}% of the view by over 30 cm, so the path \
                 passes through geometry",
                index + 1,
                index + 2,
                fraction * 100.0
            );
        }
    }

    #[test]
    fn the_room_fills_every_pixel_of_every_shot() {
        let intrinsics = intrinsics();

        for (index, shot) in shots(8).into_iter().enumerate() {
            let rendered = render(&shot, &intrinsics);
            let missing: Vec<usize> = rendered
                .depth
                .iter()
                .enumerate()
                .filter(|(_, d)| !d.is_finite())
                .map(|(i, _)| i)
                .collect();

            let first = missing
                .first()
                .map(|i| format!("at x={} y={}", i % WIDTH, i / WIDTH))
                .unwrap_or_default();

            assert_eq!(
                missing.len(),
                0,
                "shot {}: {} rays escaped the room {first}, so normals would be empty",
                index + 1,
                missing.len()
            );
            assert!(rendered.depth.iter().all(|d| *d > 0.0));
        }
    }

    #[test]
    fn every_shot_stays_inside_the_room() {
        for shot in shots(12) {
            let eye = shot.pose.translation.vector;
            assert!(eye.x > MIN_X && eye.x < MAX_X, "eye left the room in x");
            assert!(eye.y > MIN_Y && eye.y < MAX_Y, "eye left the room in y");
            assert!(eye.z > MIN_Z && eye.z < MAX_Z, "eye left the room in z");
        }
    }

    #[test]
    fn the_path_is_walkable_so_the_tracker_can_follow_it() {
        // The path is 1.8 m long and fixed, so the sampling has to be fine
        // enough that a frame-to-frame step is plausible. The odometry default
        // is tuned for about 4 cm per frame at the sensor's ~5 fps, and 20 shots
        // over this path is 9.5 cm -- fast enough that the tracker loses it, and
        // the fixture would then be measuring its own path rather than the
        // exporter.
        let shots = shots(60);

        for pair in shots.windows(2) {
            let step = (pair[1].pose.translation.vector - pair[0].pose.translation.vector).norm();
            assert!(step < 0.05, "a {step:.3} m step is not a handheld move");
        }
    }

    #[test]
    fn the_checker_gives_each_surface_structure() {
        let intrinsics = intrinsics();

        for (index, shot) in shots(12).iter().enumerate() {
            let rendered = render(shot, &intrinsics);

            // A checker quantises each surface to two shades, so the whole frame
            // holds at most twelve distinct colours -- a count in the hundreds
            // would mean the shading was continuous, which it is not. What
            // matters is that the frame is not one flat colour per surface:
            // structure is what the tracker and a radiance field hold on to.
            let mut counts = std::collections::HashMap::new();
            for pixel in rendered.rgb.chunks_exact(3) {
                *counts.entry([pixel[0], pixel[1], pixel[2]]).or_insert(0usize) += 1;
            }

            let pixels = WIDTH * HEIGHT;
            let commonest = counts.values().copied().max().unwrap_or(pixels);
            let share = commonest as f32 / pixels as f32;

            assert!(
                counts.len() >= 4,
                "shot {} has only {} distinct colours",
                index + 1,
                counts.len()
            );
            assert!(
                share < 0.80,
                "shot {} is {:.0}% one colour, so the room is close to a flat wash",
                index + 1,
                share * 100.0
            );
        }
    }

    #[test]
    fn brightness_scales_the_render_so_exposure_is_recoverable() {
        let intrinsics = intrinsics();
        let mut dim = shots(2).remove(0);
        let mut bright = dim;
        dim.brightness = 0.5;
        bright.brightness = 1.0;

        let dim = render(&dim, &intrinsics);
        let bright = render(&bright, &intrinsics);

        // Pick a pixel that is not clipped in either render.
        let index = dim
            .rgb
            .chunks_exact(3)
            .zip(bright.rgb.chunks_exact(3))
            .position(|(d, b)| d[0] > 20 && d[0] < 100 && b[0] > 40 && b[0] < 200)
            .expect("the room should contain a usable mid-tone");

        let ratio = bright.rgb[index * 3] as f32 / dim.rgb[index * 3] as f32;
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "a 2x exposure should read as 2x, got {ratio:.3}"
        );
    }

    #[test]
    fn the_mask_drops_a_border_and_keeps_the_interior() {
        let valid = mask();

        assert_eq!(valid[0], 0, "the corner should be masked");
        assert_eq!(valid[(HEIGHT / 2) * WIDTH + WIDTH / 2], 255, "the middle should be kept");

        let kept = valid.iter().filter(|v| **v != 0).count();
        let fraction = kept as f32 / valid.len() as f32;
        assert!(fraction > 0.9, "a {fraction:.3} keep rate is not a border");
        assert!(fraction < 1.0, "the mask should not keep everything");
    }
}
