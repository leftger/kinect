//! The scan pipeline: depth frame -> odometry -> TSDF fusion.

use geom::mesh::Mesh;
use geom::tsdf::{IntegrationStats, TsdfParams, TsdfVolume};
use geom::{DepthImage, Intrinsics};
use nalgebra::{Isometry3, Vector3};
use std::time::{Duration, Instant};

use crate::odometry::{Odometry, OdometryConfig, TrackReport};

#[derive(Clone, Debug)]
pub struct ScannerConfig {
    /// Ignore measurements nearer than this, in metres.
    pub depth_min: f32,
    /// Ignore measurements farther than this, in metres.
    pub depth_max: f32,
    pub tsdf: TsdfParams,
    /// The acceptance thresholds live here, not duplicated on this struct, so
    /// there is exactly one place that decides whether a pose is trustworthy.
    pub odometry: OdometryConfig,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            // The sensor's rated range; the decoder already clips to the same
            // band, this is belt-and-braces.
            depth_min: 0.5,
            depth_max: 4.5,
            tsdf: TsdfParams::default(),
            odometry: OdometryConfig::default(),
        }
    }
}

/// What happened to one frame.
pub struct FrameReport {
    /// Points that survived range clipping.
    pub points: usize,
    pub track: TrackReport,
    /// Present when the frame was fused; carries what the integration did.
    pub integration: Option<IntegrationStats>,
    pub blocks: usize,
    pub allocated_bytes: usize,
}

/// Run frames through odometry and fuse the well-tracked ones.
pub struct Scanner {
    intrinsics: Intrinsics,
    config: ScannerConfig,
    odometry: Odometry,
    volume: TsdfVolume,
    frames: usize,
    fused: usize,
    rejected: usize,
    trajectory: Vec<Vector3<f32>>,
    points: Vec<Vector3<f32>>,
    /// Cumulative time in tracking and in fusion, for the end-of-scan breakdown.
    /// Tracking and fusion are the two costs that scale with the scan, and they
    /// fail in completely different ways, so they are measured separately.
    total_tracking: Duration,
    total_fusion: Duration,
}

impl Scanner {
    pub fn new(intrinsics: Intrinsics, config: ScannerConfig) -> Self {
        // The fusion range has to match the range the point cloud is clipped to.
        // Otherwise measurements the tracker deliberately ignored still get fused
        // into the model, which is how junk turns into "geometry".
        let volume = TsdfVolume::new(TsdfParams {
            min_depth: config.depth_min,
            max_depth: config.depth_max,
            ..config.tsdf
        });
        let odometry = Odometry::new(intrinsics, config.odometry.clone());

        Self {
            intrinsics,
            config,
            odometry,
            volume,
            frames: 0,
            fused: 0,
            rejected: 0,
            trajectory: Vec::new(),
            points: Vec::new(),
            total_tracking: Duration::ZERO,
            total_fusion: Duration::ZERO,
        }
    }

    /// Feed one depth frame, in metres, row-major, `width * height` long.
    ///
    /// Anywhere the depth is `NaN` or non-positive is treated as no measurement.
    pub fn add_frame(&mut self, depth_metres: &[f32], width: usize, height: usize) -> FrameReport {
        let intrinsics = self.intrinsics;
        let image = DepthImage::new(width, height, depth_metres);

        // Back-project, dropping anything outside the working range.
        self.points.clear();
        for y in 0..height {
            for x in 0..width {
                let depth = image.at(x, y);
                if !depth.is_finite() || depth <= 0.0 {
                    continue;
                }
                if depth < self.config.depth_min || depth > self.config.depth_max {
                    continue;
                }
                self.points
                    .push(intrinsics.back_project(x as f32 + 0.5, y as f32 + 0.5, depth));
            }
        }

        let first = self.frames == 0;

        let started = Instant::now();
        // Tracking consumes the depth image directly: the odometry builds its own
        // multi-resolution pyramid from it, so the back-projected cloud below is
        // only needed for fusion.
        let track = self.odometry.track(&image);
        self.total_tracking += started.elapsed();

        // The first frame defines the world frame and is always fused. After
        // that, only a pose the odometry actually accepted may touch the model.
        let fuse = first || track.accepted;

        let integration = if fuse {
            self.fused += 1;
            self.trajectory.push(track.pose.translation.vector);

            let started = Instant::now();
            let stats = self.volume.integrate(&image, &intrinsics, &track.pose);
            self.total_fusion += started.elapsed();

            Some(stats)
        } else {
            self.rejected += 1;
            None
        };

        self.frames += 1;

        FrameReport {
            points: self.points.len(),
            track,
            integration,
            blocks: self.volume.block_count(),
            allocated_bytes: self.volume.allocated_bytes(),
        }
    }

    pub fn mesh(&self) -> Mesh {
        self.volume.extract_mesh()
    }

    pub fn trajectory(&self) -> &[Vector3<f32>] {
        &self.trajectory
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn fused(&self) -> usize {
        self.fused
    }

    pub fn rejected(&self) -> usize {
        self.rejected
    }

    pub fn block_count(&self) -> usize {
        self.volume.block_count()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.volume.allocated_bytes()
    }

    /// Cumulative time spent tracking frames.
    pub fn total_tracking(&self) -> Duration {
        self.total_tracking
    }

    /// Cumulative time spent fusing frames into the TSDF.
    pub fn total_fusion(&self) -> Duration {
        self.total_fusion
    }

    pub fn pose(&self) -> Isometry3<f32> {
        self.odometry.pose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reduced-resolution version of the real sensor's optics: the same ~70 degree
    /// field of view with far fewer pixels, so the tests stay quick.
    const WIDTH: usize = 128;
    const HEIGHT: usize = 106;

    fn intrinsics() -> Intrinsics {
        Intrinsics {
            fx: 92.0,
            fy: 92.0,
            cx: WIDTH as f32 / 2.0,
            cy: HEIGHT as f32 / 2.0,
        }
    }

    /// Render a rectangular room interior from the origin, ray-tracing the five
    /// walls analytically. `NaN` where a ray escapes.
    ///
    /// A flat wall would not do: a single plane leaves three degrees of freedom
    /// unobservable, and a small patch also collapses to a handful of points at
    /// the coarse ICP level. A box interior is what the scanner actually expects
    /// to see, and it has plenty of structure in all three directions.
    fn room_frame(width: usize, height: usize, intrinsics: &Intrinsics) -> Vec<f32> {
        const MIN_X: f32 = -1.0;
        const MAX_X: f32 = 1.0;
        const MIN_Y: f32 = -0.8;
        const MAX_Y: f32 = 1.2;
        const MAX_Z: f32 = 3.0;

        // (normal, plane offset) with the plane defined by n . p = d.
        let planes = [
            (Vector3::new(1.0, 0.0, 0.0), -MIN_X),
            (Vector3::new(-1.0, 0.0, 0.0), -MAX_X),
            (Vector3::new(0.0, 1.0, 0.0), -MIN_Y),
            (Vector3::new(0.0, -1.0, 0.0), -MAX_Y),
            (Vector3::new(0.0, 0.0, -1.0), -MAX_Z),
        ];

        let mut frame = vec![f32::NAN; width * height];

        for y in 0..height {
            for x in 0..width {
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

                    let t = offset / denominator;
                    if t <= 0.0 || t >= nearest {
                        continue;
                    }

                    let hit = direction * t;
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
                    // dir.z is 1, so t is already the z of the hit point.
                    frame[y * width + x] = nearest;
                }
            }
        }

        frame
    }

    fn scanner() -> Scanner {
        Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    // Coarser than the 1 cm default purely to keep the tests
                    // quick; the fusion logic does not depend on the resolution.
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                ..ScannerConfig::default()
            },
        )
    }

    #[test]
    fn a_single_room_frame_produces_a_surface() {
        let mut scanner = scanner();
        let frame = room_frame(WIDTH, HEIGHT, &intrinsics());

        let report = scanner.add_frame(&frame, WIDTH, HEIGHT);

        assert!(
            report.integration.is_some(),
            "the first frame should always be fused"
        );
        assert!(report.points > 5000, "only {} points", report.points);

        let mesh = scanner.mesh();
        assert!(!mesh.is_empty(), "no surface extracted");

        let (min, max) = mesh.bounds().expect("bounds");
        // The reconstruction should fill the room rather than collapsing onto a
        // single wall.
        assert!(
            min.z > 0.4 && max.z < 3.05,
            "depth extent {min:?}..{max:?} does not match the room"
        );
        assert!(max.x - min.x > 1.5, "width only {}", max.x - min.x);
        assert!(max.y - min.y > 1.0, "height only {}", max.y - min.y);
    }

    #[test]
    fn out_of_range_measurements_are_dropped() {
        let mut scanner = scanner();

        // Everything is beyond the working range: no points, and critically
        // nothing fused either.
        let frame = vec![9.0f32; WIDTH * HEIGHT];
        let report = scanner.add_frame(&frame, WIDTH, HEIGHT);

        assert_eq!(report.points, 0);
        assert_eq!(scanner.block_count(), 0);
        assert!(scanner.mesh().is_empty());

        // The first frame is always *offered* to fusion, so `integration` is
        // present -- but with nothing in range it must have integrated nothing.
        let stats = report.integration.expect("first frame is always offered");
        assert_eq!(stats.updated_voxels, 0);
        assert_eq!(stats.new_blocks, 0);
    }

    #[test]
    fn repeated_identical_frames_stay_registered() {
        let mut scanner = scanner();
        let frame = room_frame(WIDTH, HEIGHT, &intrinsics());

        for _ in 0..4 {
            let report = scanner.add_frame(&frame, WIDTH, HEIGHT);
            assert!(report.integration.is_some(), "frame was skipped");
            assert!(
                report.track.pose.translation.vector.norm() < 0.01,
                "pose drifted to {:?}",
                report.track.pose.translation.vector
            );
        }

        assert_eq!(scanner.frames(), 4);
        assert_eq!(scanner.fused(), 4);
        assert_eq!(scanner.rejected(), 0);
    }

    #[test]
    fn frame_dimensions_must_match_the_buffer() {
        let mut scanner = scanner();
        let frame = vec![1.0f32; 32 * 32];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            scanner.add_frame(&frame, 64, 64);
        }));

        assert!(result.is_err(), "mismatched dimensions must be rejected");
    }
}
