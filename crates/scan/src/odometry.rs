//! Depth odometry: recover the sensor's motion from consecutive depth frames.
//!
//! Tracking is *projective*: correspondences come from projecting a point into
//! the previous depth image rather than searching a spatial index. See
//! [`geom::projective`] for why. The spatial-index ICP is still in `geom::icp`
//! for the cases projective association cannot serve (aligning keyframes that are
//! far apart, which is what loop closure will need).

use geom::icp::IcpParams;
use geom::projective::{align_level, DepthPyramid};
use geom::{DepthImage, Intrinsics};
use nalgebra::Isometry3;

/// One resolution of the alignment pyramid.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    /// Correspondence rejection distance at this scale, in metres. Doubles as the
    /// depth-discontinuity threshold for the occlusion test, so it trades off
    /// "can bridge this much motion" against "will not match across an edge".
    pub max_distance: f32,
}

impl Level {
    /// Coarse-to-fine ladder tuned for handheld motion at ~5 fps: at 0.2 m/s the
    /// sensor travels ~4 cm per frame, so the coarsest level still accepts 20 cm
    /// of misalignment while the finest resolves well under the sensor's own
    /// ~5 mm noise.
    ///
    /// These are correspondence distances, not voxel sizes: the pyramid supplies
    /// the resolutions and `geom::projective` computes normals from the depth
    /// image itself, so there is nothing else to tune per level.
    pub fn handheld_default() -> Vec<Level> {
        vec![
            // Finest: the default rejection distance.
            Level { max_distance: 0.05 },
            Level { max_distance: 0.10 },
            // Coarsest (quarter resolution): the basin the initial guess needs.
            Level { max_distance: 0.20 },
        ]
    }
}

#[derive(Clone, Debug)]
pub struct OdometryConfig {
    pub levels: Vec<Level>,
    pub huber_delta: f32,
    pub max_iterations: usize,
    /// A frame with fewer points than this is treated as unusable (sensor
    /// covered, or staring at something out of range).
    pub min_points: usize,

    // --- acceptance thresholds -------------------------------------------------
    // A scan is only as good as its worst fused frame: one bad pose smears
    // geometry through the model, and if that frame also becomes the reference
    // for the next one the error compounds. These gates decide when to refuse.
    /// Minimum fraction of the frame that must match the previous frame.
    pub min_inlier_ratio: f32,
    /// Maximum RMS point-to-plane residual to accept, in metres.
    pub max_rmse: f32,
    /// Maximum plausible sensor translation between two frames, in metres.
    /// At ~5 fps even a brisk 0.5 m/s is 10 cm, so 20 cm means something went
    /// wrong rather than that the operator moved quickly.
    pub max_translation: f32,
    /// Maximum plausible rotation between two frames, in radians.
    pub max_rotation: f32,
}

impl Default for OdometryConfig {
    fn default() -> Self {
        Self {
            levels: Level::handheld_default(),
            huber_delta: 0.02,
            // Per level. Point-to-plane ICP on this geometry settles well inside
            // 10 iterations; the previous 20 was mostly burning CPU re-deriving
            // the same solution, since the step stops shrinking at the noise floor.
            max_iterations: 10,
            min_points: 500,
            min_inlier_ratio: 0.60,
            max_rmse: 0.020,
            max_translation: 0.20,
            max_rotation: 0.35,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrackReport {
    /// Camera-to-world pose after this frame. Unchanged if the frame was rejected.
    pub pose: Isometry3<f32>,
    /// True when the pose was accepted and may be fused.
    pub accepted: bool,
    /// Why a tracked frame was rejected, for reporting.
    pub rejection: Option<&'static str>,
    /// Fraction of the frame's points that matched the previous frame.
    pub inlier_ratio: f32,
    /// RMS point-to-plane residual, in metres.
    pub rmse: f32,
    pub iterations: usize,
    /// Motion recovered from this frame, in metres. Zero when rejected.
    pub translation: f32,
}

/// Frame-to-frame ICP odometry.
///
/// This tracks each frame against the *previous* frame only. That is the simplest
/// thing that works and it keeps the error model easy to reason about, but it
/// means error accumulates frame by frame: expect visible drift over a long scan
/// and fix it with loop closure and pose-graph optimisation, not by tweaking
/// these parameters.
pub struct Odometry {
    intrinsics: Intrinsics,
    config: OdometryConfig,
    /// The previous frame, kept as a pyramid so it can serve as the target for
    /// the next alignment without rebuilding anything.
    previous: Option<DepthPyramid>,
    /// Previous motion, used as the initial guess (constant-velocity model).
    guess: Isometry3<f32>,
    pose: Isometry3<f32>,
}

impl Odometry {
    pub fn new(intrinsics: Intrinsics, config: OdometryConfig) -> Self {
        assert!(!config.levels.is_empty(), "need at least one scale");

        Self {
            intrinsics,
            config,
            previous: None,
            guess: Isometry3::identity(),
            pose: Isometry3::identity(),
        }
    }

    /// Current camera-to-world pose.
    pub fn pose(&self) -> Isometry3<f32> {
        self.pose
    }

    /// Track a depth frame against the previous one.
    pub fn track(&mut self, depth: &DepthImage<'_>) -> TrackReport {
        let current = DepthPyramid::from_image(depth, self.intrinsics, self.config.levels.len());

        if current.trackable_points() < self.config.min_points {
            return TrackReport {
                pose: self.pose,
                accepted: false,
                rejection: Some("too little geometry in view"),
                inlier_ratio: 0.0,
                rmse: f32::INFINITY,
                iterations: 0,
                translation: 0.0,
            };
        }

        let Some(previous) = self.previous.as_ref() else {
            self.previous = Some(current);
            return TrackReport {
                pose: self.pose,
                accepted: false,
                rejection: None,
                inlier_ratio: 1.0,
                rmse: 0.0,
                iterations: 0,
                translation: 0.0,
            };
        };

        let mut transform = self.guess;
        let mut inlier_ratio = 0.0;
        let mut rmse = f32::INFINITY;
        let mut iterations = 0;

        // Coarse to fine: `levels[0]` is the finest resolution, so walk backwards.
        // The coarse levels absorb the bulk of the motion cheaply and stop the
        // fine level from settling into a wrong local minimum -- the failure that
        // showed up on real data as bursts of "too few inliers".
        let count = self.config.levels.len().min(current.levels.len());
        for index in (0..count).rev() {
            let params = IcpParams {
                max_correspondence_distance: self.config.levels[index].max_distance,
                huber_delta: self.config.huber_delta,
                max_iterations: self.config.max_iterations,
                ..IcpParams::default()
            };

            let result = align_level(
                &current.levels[index],
                &previous.levels[index],
                transform,
                &params,
            );

            iterations += result.iterations;
            if result.correspondences > 0 {
                transform = result.transform;
                inlier_ratio = result.inlier_ratio;
                rmse = result.rmse;
            }
        }

        let translation = transform.translation.vector.norm();
        let rotation = transform.rotation.angle();

        let rejection = if inlier_ratio < self.config.min_inlier_ratio {
            Some("too few inliers")
        } else if rmse > self.config.max_rmse {
            Some("residual too large")
        } else if translation > self.config.max_translation {
            Some("implausible translation")
        } else if rotation > self.config.max_rotation {
            Some("implausible rotation")
        } else {
            None
        };

        if let Some(reason) = rejection {
            // The pose does not move: the recovered motion is not trustworthy, so
            // recording a gap is more honest than inventing a pose. The frame is
            // simply not fused, and `pose`/`guess` stay as they were.
            //
            // The *reference cloud* does advance, and that detail matters. Holding
            // the last good frame looks safer -- don't track against junk -- but
            // it is a trap: the sensor keeps moving while we hold, so the next
            // frame sits even further from the stale reference, exceeds the
            // correspondence radius, and fails too. That is a permanent lock-out,
            // and it reproduced exactly on real data: one rejected frame at 33 cm
            // of motion took the scan from 47% inliers to 3% and never recovered.
            //
            // A reference cloud is only ever used in its own camera frame, so a
            // fresh one is always the better target regardless of whether the pose
            // attached to it is trusted.
            self.previous = Some(current);

            return TrackReport {
                pose: self.pose,
                accepted: false,
                rejection: Some(reason),
                inlier_ratio,
                rmse,
                iterations,
                translation,
            };
        }

        // `transform` maps this frame's camera coords into the previous frame's,
        // so chaining it onto the previous pose gives the new world pose.
        self.pose *= transform;
        self.guess = transform;
        self.previous = Some(current);

        TrackReport {
            pose: self.pose,
            accepted: true,
            rejection: None,
            inlier_ratio,
            rmse,
            iterations,
            translation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// Reduced-resolution stand-in for the sensor's optics.
    const WIDTH: usize = 96;
    const HEIGHT: usize = 80;

    fn intrinsics() -> Intrinsics {
        Intrinsics {
            fx: 92.0,
            fy: 92.0,
            cx: WIDTH as f32 / 2.0,
            cy: HEIGHT as f32 / 2.0,
        }
    }

    fn odometry() -> Odometry {
        Odometry::new(intrinsics(), OdometryConfig::default())
    }

    /// Render a rectangular room interior as seen from `eye`, ray-tracing the
    /// walls analytically.
    ///
    /// A room rather than a single plane, for two reasons. A plane leaves three
    /// degrees of freedom unobservable, so it cannot exercise all six. And the
    /// render has to be *dense* -- every ray hitting something -- because normals
    /// come from neighbouring pixels, so holes would leave the normal map empty.
    /// A box interior gives both, and it is what the scanner really looks at.
    fn render_room(eye: Vector3<f32>) -> Vec<f32> {
        const MIN_X: f32 = -1.5;
        const MAX_X: f32 = 1.5;
        const MIN_Y: f32 = -1.2;
        const MAX_Y: f32 = 1.2;
        const MAX_Z: f32 = 4.0;

        // Plane as (normal, offset) with n . p = d.
        let planes = [
            (Vector3::new(1.0, 0.0, 0.0), -MIN_X),
            (Vector3::new(-1.0, 0.0, 0.0), -MAX_X),
            (Vector3::new(0.0, 1.0, 0.0), -MIN_Y),
            (Vector3::new(0.0, -1.0, 0.0), -MAX_Y),
            (Vector3::new(0.0, 0.0, -1.0), -MAX_Z),
        ];

        let intrinsics = intrinsics();
        let mut depth = vec![f32::NAN; WIDTH * HEIGHT];

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let direction = Vector3::new(
                    (x as f32 + 0.5 - intrinsics.cx) / intrinsics.fx,
                    (y as f32 + 0.5 - intrinsics.cy) / intrinsics.fy,
                    1.0,
                );

                let mut nearest = f32::INFINITY;
                for (normal, offset) in &planes {
                    let denominator = normal.dot(&direction);
                    if denominator.abs() < 1e-6 {
                        continue;
                    }

                    // Ray from the eye, not the origin: moving the camera is the
                    // only thing these tests do.
                    let t = (offset - normal.dot(&eye)) / denominator;
                    if t <= 0.0 || t >= nearest {
                        continue;
                    }

                    let hit = eye + direction * t;
                    let inside = hit.x >= MIN_X
                        && hit.x <= MAX_X
                        && hit.y >= MIN_Y
                        && hit.y <= MAX_Y
                        && hit.z >= 0.0
                        && hit.z <= MAX_Z;
                    if inside {
                        nearest = t;
                    }
                }

                if nearest.is_finite() {
                    // `direction.z` is 1, so t is already the camera-space depth.
                    depth[y * WIDTH + x] = nearest;
                }
            }
        }

        depth
    }

    fn frame(depth: &[f32]) -> DepthImage<'_> {
        DepthImage::new(WIDTH, HEIGHT, depth)
    }

    #[test]
    fn first_frame_defines_the_world_frame_but_is_not_tracked() {
        let mut odometry = odometry();
        let room = render_room(Vector3::zeros());
        let report = odometry.track(&frame(&room));

        assert!(!report.accepted, "there is nothing to track against yet");
        assert!(
            report.rejection.is_none(),
            "not an error, just the first frame"
        );
        assert!(report.pose.translation.vector.norm() < 1e-6);
    }

    #[test]
    fn stationary_input_keeps_the_pose_fixed() {
        let mut odometry = odometry();
        let room = render_room(Vector3::zeros());

        odometry.track(&frame(&room));
        for _ in 0..5 {
            let report = odometry.track(&frame(&room));
            assert!(report.accepted, "rejected: {:?}", report.rejection);
        }

        let pose = odometry.pose();
        assert!(
            pose.translation.vector.norm() < 2e-3,
            "drifted {} m while stationary",
            pose.translation.vector.norm()
        );
        assert!(pose.rotation.angle() < 0.01);
    }

    #[test]
    fn accumulated_pose_tracks_a_known_motion() {
        let mut odometry = odometry();
        let start = render_room(Vector3::zeros());
        odometry.track(&frame(&start));

        // The camera steps 3 cm to the right each frame; each view is the same
        // room seen from the new position, so the recovered pose should match.
        for step in 1..=4 {
            let eye = Vector3::new(0.03 * step as f32, 0.0, 0.0);
            let room = render_room(eye);

            let report = odometry.track(&frame(&room));
            assert!(
                report.accepted,
                "step {step} rejected: {:?}",
                report.rejection
            );
            assert!(
                report.inlier_ratio > 0.5,
                "poor convergence at step {step}: {}",
                report.inlier_ratio
            );
        }

        let pose = odometry.pose();
        assert!(
            (pose.translation.vector.x - 0.12).abs() < 0.02,
            "expected ~0.12 m in x, got {:.4}",
            pose.translation.vector.x
        );
        assert!(pose.translation.vector.y.abs() < 0.02);
        assert!(pose.translation.vector.z.abs() < 0.03);
    }

    #[test]
    fn a_frame_with_too_few_points_is_rejected() {
        let mut odometry = odometry();
        let room = render_room(Vector3::zeros());
        odometry.track(&frame(&room));

        let empty = vec![f32::NAN; WIDTH * HEIGHT];
        let report = odometry.track(&frame(&empty));
        assert!(!report.accepted);
        assert_eq!(report.rejection, Some("too little geometry in view"));
    }

    #[test]
    fn a_jump_to_an_unrelated_view_is_rejected_and_does_not_lock_out() {
        let mut odometry = odometry();

        let room = render_room(Vector3::zeros());
        odometry.track(&frame(&room));
        assert!(odometry.track(&frame(&room)).accepted);

        let pose_before = odometry.pose();

        // The same room seen from 2 m further back. Nothing can match: every
        // pixel's depth differs by far more than the correspondence radius.
        let far = render_room(Vector3::new(0.0, 0.0, -2.0));
        let rejected = odometry.track(&frame(&far));
        assert!(!rejected.accepted, "a 2 m jump must not be accepted");
        assert!(rejected.rejection.is_some());

        // The pose must be untouched: the skipped motion is a gap, not a pose we
        // get to invent.
        assert!(
            (odometry.pose().translation.vector - pose_before.translation.vector).norm() < 1e-9,
            "a rejected frame moved the pose"
        );

        // But the reference must have advanced to the rejected frame. Holding the
        // last good one is a lock-out: the sensor keeps moving, so every later
        // frame would be measured against an ever-staler view until nothing
        // matches. Tracking the same view again must therefore succeed.
        let settled = odometry.track(&frame(&far));
        assert!(settled.accepted, "locked out after a rejected frame");
        assert!(
            (odometry.pose().translation.vector - pose_before.translation.vector).norm() < 1e-9,
            "pose moved on a frame that was never trusted"
        );
    }

    #[test]
    fn an_identical_frame_aligns_almost_perfectly() {
        let mut odometry = odometry();
        let room = render_room(Vector3::zeros());
        odometry.track(&frame(&room));

        let report = odometry.track(&frame(&room));
        assert!(report.accepted);
        assert!(report.inlier_ratio > 0.9, "{}", report.inlier_ratio);
        assert!(report.rmse < 1e-3, "{}", report.rmse);
        assert!(report.iterations > 0);
    }
}
