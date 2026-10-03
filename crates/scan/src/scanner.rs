//! The scan pipeline: depth frame -> odometry -> TSDF fusion.

use geom::mesh::Mesh;
use geom::tsdf::{IntegrationStats, TsdfParams, TsdfVolume};
use geom::coloring::{colorize, ColoringParams, ColorView};
use geom::pose_graph::{PoseGraph, PoseGraphParams};
use geom::{DepthImage, Intrinsics};
use nalgebra::{Isometry3, Vector3};
use std::time::{Duration, Instant};

use crate::loop_closure::{LoopClosureConfig, LoopFinder};
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
    /// Revisit detection. `None` disables it and, more importantly, skips
    /// buffering the frames -- which is the memory cost of being able to correct
    /// them after the fact.
    pub loop_closure: Option<LoopClosureConfig>,
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
            loop_closure: None,
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

/// A frame kept so the model can be rebuilt once the poses are corrected.
///
/// Rebuilding needs the measurements again. The TSDF was integrated with the
/// drifted poses, so correcting them invalidates the volume outright rather than
/// nudging it -- there is no way to move geometry that has already been fused.
/// This is the memory cost of loop closure, and the reason it is opt-in.
struct KeptFrame {
    depth: Vec<f32>,
    /// Whether this frame was fused the first time round. A rebuild has to make
    /// the same choice, or the model gains geometry the tracker refused.
    fused: bool,
}

/// The most any single pose may be moved by loop closure before the whole
/// correction is discarded.
///
/// A pose graph will fold a map in half to satisfy one wrong edge, and a wrong
/// edge is exactly what a false revisit produces. The ICP verification in
/// `LoopFinder` and the robust kernel are the primary defences; this is the
/// backstop, set well above the drift a real handheld scan accumulates.
pub const MAX_CORRECTION_METRES: f32 = 2.0;

/// The equivalent backstop for rotation, in radians (about 29 degrees).
pub const MAX_CORRECTION_RADIANS: f32 = 0.5;

/// What loop closure did.
#[derive(Clone, Copy, Debug)]
pub struct ClosureReport {
    pub nodes: usize,
    pub odometry_edges: usize,
    pub loop_edges: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Largest pose change the optimisation asked for, in metres.
    pub max_correction: f32,
    /// Largest rotation change, in radians. Guarded separately: a pose can be
    /// badly rotated while its translation barely moves.
    pub max_rotation: f32,
    /// Set when the correction exceeded `MAX_CORRECTION_METRES` and was thrown
    /// away. The trajectory and the model are then exactly as odometry left them.
    pub refused: bool,
    /// Whether the volume was actually rebuilt with corrected poses.
    pub rebuilt: bool,
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

    /// Pose-graph nodes, one per frame in arrival order. Consecutive poses
    /// already encode the odometry measurement: the tracked pose *is* the
    /// accumulation, so the relative motion is just `T[i]^-1 * T[i+1]`.
    poses: Vec<Isometry3<f32>>,
    /// Frames retained so the model can be rebuilt; parallel to `poses`.
    kept: Vec<KeptFrame>,
    loops: Option<LoopFinder>,
    /// Colour from the frames that saw the surface, for texturing the mesh.
    color_views: Vec<ColorView>,
    coloring: ColoringParams,
    dimensions: Option<(usize, usize)>,
    closure: Option<ClosureReport>,
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
        let loops = config.loop_closure.map(LoopFinder::new);

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
            poses: Vec::new(),
            kept: Vec::new(),
            loops,
            color_views: Vec::new(),
            coloring: ColoringParams::default(),
            dimensions: None,
            closure: None,
        }
    }

    /// Feed one depth frame, in metres, row-major, `width * height` long.
    ///
    /// Anywhere the depth is `NaN` or non-positive is treated as no measurement.
    pub fn add_frame(&mut self, depth_metres: &[f32], width: usize, height: usize) -> FrameReport {
        self.add_frame_with_color(depth_metres, width, height, None)
    }

    /// As `add_frame`, but also keeping this frame's colour for texturing.
    ///
    /// `color` is the registered colour and the undistorted depth from the same
    /// frame; the pose is not needed because it is the pose this frame is about
    /// to be tracked to.
    pub fn add_frame_with_color(
        &mut self,
        depth_metres: &[f32],
        width: usize,
        height: usize,
        color: Option<(&[u8], &[f32])>,
    ) -> FrameReport {
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

        self.poses.push(track.pose);
        self.dimensions.get_or_insert((width, height));

        if let Some((rgb, depth)) = color {
            self.color_views.push(ColorView {
                color: rgb.to_vec(),
                width,
                height,
                depth: depth.to_vec(),
                pose: track.pose,
                intrinsics,
            });
        }

        // Offer this frame as somewhere the sensor might return to. The cloud is
        // the one already back-projected for fusion, so the extra cost is the
        // voxel hashing and, when a candidate turns up, one ICP alignment.
        if let Some(finder) = self.loops.as_mut() {
            if fuse {
                finder.consider(self.frames, track.pose, &self.points);
            }
        }

        if self.loops.is_some() {
            self.kept.push(KeptFrame {
                depth: depth_metres.to_vec(),
                fused: fuse,
            });
        }

        self.frames += 1;

        FrameReport {
            points: self.points.len(),
            track,
            integration,
            blocks: self.volume.block_count(),
            allocated_bytes: self.volume.allocated_bytes(),
        }
    }

    /// Optimise the trajectory against the revisits that were verified, and
    /// rebuild the model if the correction is worth having.
    ///
    /// Call once, after the last frame. A no-op when loop closure is disabled, or
    /// when no revisit survived verification.
    pub fn close_loops(&mut self) -> ClosureReport {
        let loop_edges = self
            .loops
            .as_ref()
            .map(|finder| finder.edges())
            .unwrap_or_default();

        let nodes = self.poses.len();
        let odometry_edges = nodes.saturating_sub(1);

        let nothing = ClosureReport {
            nodes,
            odometry_edges,
            loop_edges: loop_edges.len(),
            cost_before: 0.0,
            cost_after: 0.0,
            iterations: 0,
            converged: true,
            max_correction: 0.0,
            max_rotation: 0.0,
            refused: false,
            rebuilt: false,
        };

        // With no revisit there is nothing to redistribute, so the odometry
        // trajectory stands unaltered.
        if nodes < 2 || loop_edges.is_empty() {
            self.closure = Some(nothing);
            return nothing;
        }

        let mut graph = PoseGraph::new(self.poses[0]);
        for pose in &self.poses[1..] {
            graph.add_node(*pose);
        }

        // The odometry edges are free: the tracked pose is the accumulation, so
        // the measurement between neighbours is already encoded in the poses.
        for index in 0..nodes - 1 {
            let measurement = self.poses[index].inverse() * self.poses[index + 1];
            graph.add_edge(index, index + 1, measurement, 1.0);
        }
        for edge in &loop_edges {
            graph.add_edge(edge.from, edge.to, edge.measurement, edge.weight);
        }

        let optimised = graph.optimize(&PoseGraphParams::default());

        let mut max_correction = 0.0f32;
        let mut max_rotation = 0.0f32;
        for (before, after) in self.poses.iter().zip(graph.nodes()) {
            let delta = before.inverse() * after;
            max_correction = max_correction.max(delta.translation.vector.norm());
            max_rotation = max_rotation.max(delta.rotation.angle());
        }

        let mut closure = ClosureReport {
            nodes,
            odometry_edges,
            loop_edges: loop_edges.len(),
            cost_before: optimised.cost_before,
            cost_after: optimised.cost_after,
            iterations: optimised.iterations,
            converged: optimised.converged,
            max_correction,
            max_rotation,
            refused: false,
            rebuilt: false,
        };

        // Simplicity beats quiet corruption. If the graph wants to move a pose
        // further than the odometry could plausibly be wrong, believe the
        // odometry and leave the model alone.
        if max_correction > MAX_CORRECTION_METRES || max_rotation > MAX_CORRECTION_RADIANS {
            closure.refused = true;
            self.closure = Some(closure);
            return closure;
        }

        self.poses = graph.nodes().to_vec();
        self.rebuild();
        closure.rebuilt = true;

        self.closure = Some(closure);
        closure
    }

    /// Re-integrate every kept frame with the corrected poses.
    ///
    /// A fresh volume, because geometry already fused at the old poses cannot be
    /// moved -- only discarded and laid down again.
    fn rebuild(&mut self) {
        let Some((width, height)) = self.dimensions else {
            return;
        };
        let intrinsics = self.intrinsics;

        self.volume = TsdfVolume::new(TsdfParams {
            min_depth: self.config.depth_min,
            max_depth: self.config.depth_max,
            ..self.config.tsdf
        });
        self.trajectory.clear();

        let mut fused = 0;
        for index in 0..self.kept.len() {
            if !self.kept[index].fused {
                continue;
            }

            let pose = self.poses[index];
            let image = DepthImage::new(width, height, &self.kept[index].depth);
            self.volume.integrate(&image, &intrinsics, &pose);
            self.trajectory.push(pose.translation.vector);
            fused += 1;
        }

        self.fused = fused;
    }

    /// What loop closure did, once `close_loops` has run.
    pub fn closure(&self) -> Option<&ClosureReport> {
        self.closure.as_ref()
    }

    /// How many revisits were verified, whether or not they were used.
    pub fn loops(&self) -> usize {
        self.loops
            .as_ref()
            .map(|finder| finder.loops().len())
            .unwrap_or(0)
    }

    pub fn mesh(&self) -> Mesh {
        let mut mesh = self.volume.extract_mesh();

        if !self.color_views.is_empty() {
            let (colors, _report) = colorize(&mesh.vertices, &self.color_views, &self.coloring);
            mesh.colors = Some(colors);
        }

        mesh
    }

    /// The frames that contributed colour, for the live preview.
    pub fn color_views(&self) -> &[ColorView] {
        &self.color_views
    }

    /// How many frames contributed colour. Zero when texturing is off.
    pub fn color_view_count(&self) -> usize {
        self.color_views.len()
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

    /// A scanner with revisit detection on, and a small temporal guard: the
    /// synthetic scans here are only a handful of frames long, so the default
    /// gap of 40 frames would reject every candidate and test nothing.
    fn scanner_with_loops() -> Scanner {
        Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                loop_closure: Some(LoopClosureConfig {
                    min_index_gap: 2,
                    ..LoopClosureConfig::default()
                }),
                ..ScannerConfig::default()
            },
        )
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

    #[test]
    fn loop_closure_does_not_damage_a_consistent_scan() {
        // A stationary scan: every frame is a revisit of every earlier one, so
        // the finder has plenty to verify while the pose graph has nothing that
        // needs correcting.
        //
        // The property under test is that a *correct* trajectory survives the
        // machinery untouched. That is the one that matters, because the failure
        // mode of a pose graph is not failing to correct -- it is happily folding
        // the map in half to satisfy a wrong edge.
        let mut scanner = scanner_with_loops();
        let frame = room_frame(WIDTH, HEIGHT, &intrinsics());

        for _ in 0..6 {
            scanner.add_frame(&frame, WIDTH, HEIGHT);
        }

        assert!(
            scanner.loops() > 0,
            "no revisits were verified, so nothing is being tested"
        );

        let report = scanner.close_loops();

        assert!(report.loop_edges > 0, "no loop edge reached the graph");
        assert!(
            !report.refused,
            "a consistent scan was refused: moved {:.3} m",
            report.max_correction
        );
        assert!(report.rebuilt, "the model was not rebuilt");
        assert!(
            report.max_correction < 0.02,
            "loop closure moved an already-correct scan by {:.3} m",
            report.max_correction
        );
        assert!(
            report.cost_after <= report.cost_before,
            "optimisation increased the cost"
        );

        // A rebuild throws the volume away and lays it down again, so it has to
        // actually produce a model rather than an empty one.
        assert!(!scanner.mesh().is_empty(), "the rebuild produced no surface");
        assert_eq!(scanner.fused(), 6, "the rebuild dropped frames");
    }

    #[test]
    fn loop_closure_is_off_by_default_and_costs_nothing() {
        let mut scanner = scanner();
        let frame = room_frame(WIDTH, HEIGHT, &intrinsics());

        for _ in 0..3 {
            scanner.add_frame(&frame, WIDTH, HEIGHT);
        }

        let report = scanner.close_loops();

        assert_eq!(report.loop_edges, 0);
        assert!(!report.rebuilt);
        assert_eq!(scanner.loops(), 0);
        assert_eq!(scanner.fused(), 3);
    }
}
