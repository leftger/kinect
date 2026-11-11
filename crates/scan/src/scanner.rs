//! The scan pipeline: depth frame -> odometry -> TSDF fusion.

use geom::coloring::{ColorView, ColoringParams, ColoringReport, colorize};
use geom::icp::IcpCloud;
use geom::mesh::Mesh;
use geom::pose_graph::{Edge, PoseGraph, PoseGraphParams};
use geom::texturing::{TexturedMesh, TexturingParams, TexturingReport, texture};
use geom::tsdf::{IntegrationStats, TsdfParams, TsdfVolume};
use geom::{DepthImage, Intrinsics, transform_point};
use nalgebra::{Isometry3, Translation3, Vector3};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::loop_closure::{LoopClosureConfig, LoopFinder, VisualDescriptor};
use crate::odometry::{Odometry, OdometryConfig, TrackReport};

/// A fusion motion filter, adapted to a frame-counted scanner.
///
/// A frame that moved less than this since the last *fused* frame is tracked
/// but not integrated. Fusion is the largest per-frame cost, and a stationary
/// or slow handheld scan produces many near-duplicate frames that add nothing
/// to the surface. `max_interval_frames` forces one through periodically so a
/// stationary model still accumulates some averaging.
#[derive(Clone, Copy, Debug)]
pub struct MotionFilterConfig {
    pub max_distance: f32,
    pub max_angle: f32,
    pub max_interval_frames: usize,
}

impl Default for MotionFilterConfig {
    fn default() -> Self {
        Self {
            max_distance: 0.01,
            max_angle: 0.005,
            max_interval_frames: 5,
        }
    }
}

/// Why a frame begins a new tracking submap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    /// No boundary; the previous frame continues the same submap.
    None,
    /// The tracker could not place the frame, so the boundary pose is a guess.
    Failure,
    /// The active submap reached its frame or distance budget. Odometry is
    /// continuous across this boundary, so the relative pose is trusted.
    Budget,
}

impl BoundaryKind {
    pub fn is_boundary(self) -> bool {
        !matches!(self, Self::None)
    }
}

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
    /// Per-vertex PLY colour. `live` copies `--color-mode` and
    /// `--color-depth-tolerance` onto `mode` and `depth_tolerance`. The other
    /// fields stay at [`ColoringParams::default`] (best sample, 2 cm). Atlas
    /// export uses the same visibility rules and assigns each triangle to one
    /// best source view.
    pub coloring: ColoringParams,
    /// Start a new submap when a frame cannot be placed in the current one.
    ///
    /// The frame becomes the new submap's first frame, fused at the best guess
    /// the failed alignment produced. Separate volumes keep each fragment clean
    /// instead of smearing it onto the old model; the boundary is a weak pose
    /// edge, so loop closure can move the whole submap later.
    pub submaps: bool,
    /// Finish a tracking submap after this many fused frames. Zero disables the
    /// frame budget. Odometry continues across the boundary, so the relative
    /// pose is a measurement, not a guess.
    pub submap_max_frames: usize,
    /// Finish a tracking submap once the camera has moved this far from its
    /// first pose, in metres. Zero disables the distance budget.
    pub submap_max_translation: f32,
    /// Apply a verified loop as soon as it arrives, rather than waiting for
    /// `close_loops` after the last frame. The keyframe graph is optimised and
    /// the corrected poses are used for every subsequent frame; the fused
    /// submaps are moved rigidly, and the final rebuild is still available.
    /// Requires `loop_closure`.
    pub online_loops: bool,
    /// On the final correction, move the cached submap surfaces with the
    /// corrected poses instead of throwing the volume away and re-integrating
    /// every frame. Much cheaper and the LIO-SAM keyframe-map trade-off, but a
    /// rigid submap move cannot fix drift *inside* a submap. The full rebuild
    /// remains the accuracy default.
    pub incremental_rebuild: bool,
    /// Skip fusion for frames that barely moved. Tracking, loop closure, and the
    /// retained depth still see them; only the TSDF integration is skipped.
    pub fusion_filter: Option<MotionFilterConfig>,
    /// Finish the active TSDF into a cached chunk every N fused frames. Zero
    /// disables chunking. This bounds the active volume and the per-frame
    /// fusion cost without resetting the odometry frame or creating a pose
    /// boundary: chunks are a memory policy, not a new submap.
    pub tsdf_chunk_frames: usize,
    /// Optimise a graph whose nodes are tracking submaps rather than
    /// keyframes, a sparse pose-adjustment shape. A correction
    /// moves each submap rigidly, so this is cheaper and matches the
    /// incremental path, but it cannot deform inside a submap. Off by default;
    /// the keyframe graph is the accuracy path.
    pub submap_graph: bool,
    /// Run the online submap-graph solve on a worker thread instead of on the
    /// capture thread. Local SLAM and global SLAM stay on
    /// separate threads for the same reason: the capture loop must not stall
    /// behind a dense solve. The final `close_loops` pass is still synchronous
    /// and flushes any online job first.
    pub graph_worker: bool,
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
            coloring: ColoringParams::default(),
            submaps: true,
            submap_max_frames: 0,
            submap_max_translation: 0.0,
            online_loops: false,
            incremental_rebuild: false,
            fusion_filter: None,
            tsdf_chunk_frames: 0,
            submap_graph: false,
            graph_worker: false,
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
    /// The submap this frame belongs to, zero-based.
    pub submap: usize,
    /// True when this frame began a new submap.
    pub new_submap: bool,
    /// Why the tracker rejected the frame that started this submap, if any.
    pub submap_reason: Option<&'static str>,
}

/// A frame kept so the model can be rebuilt once the poses are corrected.
///
/// Rebuilding needs the measurements again. The TSDF was integrated with the
/// drifted poses, so correcting them invalidates the volume outright rather than
/// nudging it -- there is no way to move geometry that has already been fused.
/// The retained cost is the depth buffer, about 850 KB at 512×424. Colour
/// views are stored separately at about 1.66 MiB each. This is why loop
/// closure is opt-in.
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
    /// Keyframe nodes in the optimised graph.
    pub keyframe_nodes: usize,
    /// Submap nodes in the optimised graph. Non-zero only for the submap-graph
    /// mode; the keyframe graph has none.
    pub submap_nodes: usize,
    pub odometry_edges: usize,
    /// Loop edges used by the final optimisation.
    pub loop_edges: usize,
    /// Verified loops discarded because they made the correction implausible.
    pub dropped_loops: usize,
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
    /// Whether the model was actually corrected. A full rebuild, an incremental
    /// submap move, and an online update all count; the trajectory and the
    /// geometry both agree with the corrected poses afterwards.
    pub rebuilt: bool,
    /// True when this report came from an in-scan correction rather than the
    /// end-of-scan pass.
    pub online: bool,
}

/// A pre-training sanity check of the reconstruction.
///
/// This is not a quality metric: it only distinguishes a scan that can be
/// handed to a trainer from one that is obviously broken. The warnings are
/// shown before a dataset is launched, not used to silently discard data.
#[derive(Clone, Debug)]
pub struct ScanQuality {
    pub frames: usize,
    pub fused: usize,
    pub rejected: usize,
    pub submaps: usize,
    pub loop_edges: usize,
    pub rebuilt: bool,
    pub refused: bool,
    pub final_distance: f32,
    pub warnings: Vec<String>,
}

impl ScanQuality {
    /// True when a trainer has a coherent reconstruction to start from.
    pub fn trainable(&self) -> bool {
        self.warnings.is_empty()
    }
}

/// Colour for one frame, already registered into the depth grid.
///
/// Borrowed because the capture that produced it still owns the buffers. The
/// scanner copies what it keeps.
pub struct FrameColor<'a> {
    /// RGB, `width * height * 3`.
    pub rgb: &'a [u8],
    /// Undistorted depth in metres, same grid as `rgb`.
    pub depth: &'a [f32],
    /// One byte per depth pixel. Non-zero where `rgb` was copied by registration.
    pub valid: &'a [u8],
    pub exposure: f32,
    pub gain: f32,
    pub gamma: f32,
}

/// What [`Scanner::export_mesh`] decided to write.
pub enum ExportMesh {
    /// Positions, normals and, for PLY, per-vertex colour. No atlas.
    ///
    /// The report is present when colour views were painted onto the mesh.
    Geometry(Mesh, Option<ColoringReport>),
    /// Atlas mesh. The mirror, when requested, has already been applied.
    Atlas(TexturedMesh, TexturingReport),
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
    /// Frames that tracked but the fusion motion filter skipped.
    fusion_skipped: usize,
    /// Pose of the last frame actually integrated, for the motion filter.
    last_fused: Option<Isometry3<f32>>,
    /// Frames since the last integration, for the forced-refresh interval.
    frames_since_fused: usize,
    /// First frame of the active TSDF chunk.
    chunk_start: usize,
    /// Fused frames in the active TSDF chunk.
    chunk_frames: usize,
    /// Fused frames in the active tracking submap.
    submap_fused: usize,
    /// The next frame begins a budget boundary.
    pending_budget_boundary: bool,
    trajectory: Vec<Vector3<f32>>,
    points: Vec<Vector3<f32>>,
    /// Cumulative time in tracking and in fusion, for the end-of-scan breakdown.
    /// Tracking and fusion are the two costs that scale with the scan, and they
    /// fail in completely different ways, so they are measured separately.
    total_tracking: Duration,
    total_fusion: Duration,
    /// Time spent finalising a finished submap's surface at a boundary.
    total_submap: Duration,

    /// Pose of every frame in arrival order. The keyframe graph's nodes are a
    /// subset of these; the in-between frames are corrected by interpolating
    /// between the keyframe corrections. Consecutive poses already encode the
    /// odometry measurement: the tracked pose *is* the accumulation, so the
    /// relative motion is just `T[i]^-1 * T[i+1]`.
    poses: Vec<Isometry3<f32>>,
    /// Frames retained so the model can be rebuilt; parallel to `poses`.
    kept: Vec<KeptFrame>,
    loops: Option<LoopFinder>,
    /// Optional background solver for the online submap graph.
    graph_worker: Option<GraphWorker>,
    graph_pending: bool,
    graph_total_loops: usize,
    /// How many online corrections the worker delivered.
    graph_updates: usize,
    /// Colour from the frames that were fused. A rejected frame did not
    /// contribute geometry, so its colour is not kept either. After an accepted
    /// loop closure each view's pose is the corrected pose of its frame.
    color_views: Vec<ColorView>,
    /// Frames that arrived with a colour buffer, including ones the tracker
    /// rejected. [`Self::color_view_count`] is the subset that was kept.
    color_supplied: usize,
    dimensions: Option<(usize, usize)>,
    closure: Option<ClosureReport>,

    /// Surfaces from submaps that were finished at a lost track, kept in their
    /// submap frame with the transform that places them in the world. A
    /// correction updates the transform instead of re-integrating the volume,
    /// which is the cheap, LIO-SAM-style way to move already-fused geometry.
    finished_submaps: Vec<FinishedSubmap>,
    /// World-from-current-submap. Identity for the first submap.
    world_from_submap: Isometry3<f32>,
    /// Parallel to `poses`: the reason the edge from the previous pose starts a
    /// new submap, if it does.
    pose_boundaries: Vec<BoundaryKind>,
    /// Arrival index of each submap's first frame.
    submap_starts: Vec<usize>,
    /// True once an incremental correction has placed every piece in one world
    /// frame, so the scan reports one model even though the pieces are stored
    /// separately.
    model_merged: bool,
    /// In-scan corrections applied.
    online_updates: usize,
    /// Largest pose move from the most recent online correction, in metres.
    last_online_correction: f32,
}

/// One finished submap surface and where it sits in the scan's world.
struct FinishedSubmap {
    /// Untransformed surface in the submap's own frame.
    mesh: Mesh,
    /// World-from-submap for this surface.
    transform: Isometry3<f32>,
    /// Arrival index of this chunk's first frame, used to pick its correction.
    start: usize,
    /// Tracking submap this chunk belongs to. Several chunks can share one.
    submap: usize,
}

/// What optimising the graph produced, projected onto every frame.
struct GraphOutcome {
    poses: Vec<Isometry3<f32>>,
    nodes: usize,
    keyframe_nodes: usize,
    submap_nodes: usize,
    /// New submap poses when the submap graph was used.
    submap_poses: Option<Vec<Isometry3<f32>>>,
    odometry_edges: usize,
    max_correction: f32,
    max_rotation: f32,
    cost_before: f64,
    cost_after: f64,
    iterations: usize,
    converged: bool,
}

/// A snapshot of everything the submap graph needs. It is deliberately owned
/// data with no borrow of the scanner, so it can be moved to a worker thread.
struct SubmapGraphJob {
    poses: Vec<Isometry3<f32>>,
    loop_edges: Vec<Edge>,
    submap_starts: Vec<usize>,
    boundaries: Vec<BoundaryKind>,
}

/// A snapshot of everything the keyframe graph needs.
struct KeyframeGraphJob {
    poses: Vec<Isometry3<f32>>,
    boundaries: Vec<BoundaryKind>,
    keyframes: Vec<(usize, Isometry3<f32>)>,
    loop_edges: Vec<Edge>,
}

enum GraphJob {
    Submap(SubmapGraphJob),
    Keyframe(KeyframeGraphJob),
}

/// Optional pose-graph worker. Owned by [`Scanner`]; dropping it closes the job
/// channel and joins the thread.
struct GraphWorker {
    sender: Option<Sender<GraphJob>>,
    receiver: Receiver<Option<(GraphOutcome, usize, bool)>>,
    handle: Option<JoinHandle<()>>,
}

impl GraphWorker {
    fn spawn() -> Self {
        let (job_sender, job_receiver) = mpsc::channel::<GraphJob>();
        let (outcome_sender, outcome_receiver) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("scan-pose-graph".to_string())
            .spawn(move || {
                while let Ok(job) = job_receiver.recv() {
                    let result = match job {
                        GraphJob::Submap(job) => solve_submap_graph(&job),
                        GraphJob::Keyframe(job) => solve_keyframe_graph(&job),
                    };
                    if outcome_sender.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn pose-graph worker");

        Self {
            sender: Some(job_sender),
            receiver: outcome_receiver,
            handle: Some(handle),
        }
    }
}

impl Drop for GraphWorker {
    fn drop(&mut self) {
        // Closing the sender lets the worker leave its receive loop; joining it
        // before the receiver disappears keeps shutdown deterministic.
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Optimise keyframe nodes. Free function so it runs synchronously or on the
/// worker thread.
fn solve_keyframe_graph(job: &KeyframeGraphJob) -> Option<(GraphOutcome, usize, bool)> {
    let keyframes = &job.keyframes;
    if keyframes.len() < 2 {
        return None;
    }

    let mut loop_edges = job.loop_edges.clone();
    loop {
        if loop_edges.is_empty() {
            return None;
        }

        let mut graph = PoseGraph::new(keyframes[0].1);
        for (_, pose) in &keyframes[1..] {
            graph.add_node(*pose);
        }

        // Only keyframes are nodes. The relative pose between consecutive
        // keyframes is read from the snapshot poses. A failure boundary is a
        // guess, so its edge is omitted and a loop edge is free to move the
        // whole submap. A budget boundary is continuous odometry, so its edge
        // is kept.
        let mut odometry_edges = 0;
        for window in 0..keyframes.len() - 1 {
            let from = keyframes[window].0;
            let to = keyframes[window + 1].0;
            let failed = ((from + 1)..=to)
                .any(|frame| job.boundaries.get(frame).copied() == Some(BoundaryKind::Failure));
            if failed {
                continue;
            }
            let measurement = keyframes[window].1.inverse() * keyframes[window + 1].1;
            graph.add_edge(window, window + 1, measurement, 1.0);
            odometry_edges += 1;
        }

        let node_of = |frame: usize| {
            keyframes
                .binary_search_by_key(&frame, |(index, _)| *index)
                .ok()
        };
        let mut added = 0;
        for edge in &loop_edges {
            let (Some(from), Some(to)) = (node_of(edge.from), node_of(edge.to)) else {
                continue;
            };
            if from == to {
                continue;
            }
            graph.add_edge(from, to, edge.measurement, edge.weight);
            added += 1;
        }
        if added == 0 {
            return None;
        }

        let optimised = graph.optimize(&PoseGraphParams::default());
        let keyframe_poses = graph.nodes().to_vec();
        let poses =
            project_keyframe_corrections(&job.poses, &job.boundaries, keyframes, &keyframe_poses);

        let mut max_correction = 0.0f32;
        let mut max_rotation = 0.0f32;
        for (before, after) in job.poses.iter().zip(&poses) {
            let delta = before.inverse() * after;
            max_correction = max_correction.max(delta.translation.vector.norm());
            max_rotation = max_rotation.max(delta.rotation.angle());
        }
        let within =
            max_correction <= MAX_CORRECTION_METRES && max_rotation <= MAX_CORRECTION_RADIANS;

        let outcome = GraphOutcome {
            poses,
            nodes: keyframes.len(),
            keyframe_nodes: keyframes.len(),
            submap_nodes: 0,
            submap_poses: None,
            odometry_edges,
            max_correction,
            max_rotation,
            cost_before: optimised.cost_before,
            cost_after: optimised.cost_after,
            iterations: optimised.iterations,
            converged: optimised.converged,
        };

        let kept = loop_edges.len();
        if within || kept <= 1 {
            return Some((outcome, kept, within));
        }

        // `edges_ranked` is best-first, so the weakest constraint is last.
        loop_edges.pop();
    }
}

/// Optimise submap nodes with the keyframes rigidly attached.
///
/// Free function, not a method, so the exact same solve runs synchronously or
/// on [`GraphWorker`].
fn solve_submap_graph(job: &SubmapGraphJob) -> Option<(GraphOutcome, usize, bool)> {
    let submaps = job.submap_starts.len();
    if submaps < 2 {
        return None;
    }
    let submap_poses: Vec<Isometry3<f32>> = job
        .submap_starts
        .iter()
        .map(|start| job.poses[*start])
        .collect();

    let submap_of = |frame: usize| -> usize {
        job.submap_starts
            .partition_point(|start| *start <= frame)
            .saturating_sub(1)
    };

    let mut loop_edges = job.loop_edges.clone();
    loop {
        if loop_edges.is_empty() {
            return None;
        }

        let mut graph = PoseGraph::new(submap_poses[0]);
        for pose in &submap_poses[1..] {
            graph.add_node(*pose);
        }

        // A failure boundary is a guess, so its chain edge is weak and a loop
        // edge is free to move the submap away from it. A budget boundary is
        // continuous odometry, so its edge is a real measurement.
        let mut odometry_edges = 0;
        for index in 0..submaps - 1 {
            let kind = job
                .boundaries
                .get(job.submap_starts[index + 1])
                .copied()
                .unwrap_or(BoundaryKind::None);
            let weight = if kind == BoundaryKind::Failure {
                0.1
            } else {
                1.0
            };
            let measurement = submap_poses[index].inverse() * submap_poses[index + 1];
            graph.add_edge(index, index + 1, measurement, weight);
            odometry_edges += 1;
        }

        let mut added = 0;
        for edge in &loop_edges {
            let from = submap_of(edge.from);
            let to = submap_of(edge.to);
            if from == to {
                continue;
            }
            // T_sub_from^-1 T_sub_to, built from the keyframe loop and the
            // keyframes' current offsets inside their submaps.
            let from_rel = submap_poses[from].inverse() * job.poses[edge.from];
            let to_rel = submap_poses[to].inverse() * job.poses[edge.to];
            let measurement = from_rel * edge.measurement * to_rel.inverse();
            graph.add_edge(from, to, measurement, edge.weight);
            added += 1;
        }
        if added == 0 {
            return None;
        }

        let optimised = graph.optimize(&PoseGraphParams::default());
        let new_submap_poses = graph.nodes().to_vec();
        let poses: Vec<Isometry3<f32>> = job
            .poses
            .iter()
            .enumerate()
            .map(|(frame, pose)| {
                let submap = submap_of(frame);
                let delta = new_submap_poses[submap] * submap_poses[submap].inverse();
                delta * pose
            })
            .collect();

        let mut max_correction = 0.0f32;
        let mut max_rotation = 0.0f32;
        for (before, after) in job.poses.iter().zip(&poses) {
            let delta = before.inverse() * after;
            max_correction = max_correction.max(delta.translation.vector.norm());
            max_rotation = max_rotation.max(delta.rotation.angle());
        }
        let within =
            max_correction <= MAX_CORRECTION_METRES && max_rotation <= MAX_CORRECTION_RADIANS;

        let outcome = GraphOutcome {
            poses,
            nodes: submaps,
            keyframe_nodes: 0,
            submap_nodes: submaps,
            submap_poses: Some(new_submap_poses),
            odometry_edges,
            max_correction,
            max_rotation,
            cost_before: optimised.cost_before,
            cost_after: optimised.cost_after,
            iterations: optimised.iterations,
            converged: optimised.converged,
        };

        let kept = loop_edges.len();
        if within || kept <= 1 {
            return Some((outcome, kept, within));
        }

        loop_edges.pop();
    }
}

/// How an accepted correction is put into the model.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CorrectionMode {
    /// In-scan: move the cached surfaces and carry on. No re-integration.
    Online,
    /// End of scan, accuracy path: discard the volume and re-fuse every frame.
    FullRebuild,
    /// End of scan, speed path: move the cached submap surfaces rigidly.
    Incremental,
}

/// Interpolate two left-multiplied corrections. Translation and rotation are
/// interpolated separately, which is exact for the small corrections a verified
/// loop produces and keeps the in-between frames on a smooth path.
fn interpolate_correction(a: Isometry3<f32>, b: Isometry3<f32>, t: f32) -> Isometry3<f32> {
    let translation = a.translation.vector.lerp(&b.translation.vector, t);
    let rotation = a.rotation.slerp(&b.rotation, t);
    Isometry3::from_parts(Translation3::from(translation), rotation)
}

/// Spread keyframe corrections over the frames between them.
///
/// The graph only has keyframe nodes; every other frame still needs a pose for
/// fusion, colour, and export. Inside one submap a frame between keyframes `a`
/// and `b` gets the interpolation of their left corrections, clamped to the
/// submap's own keyframes. Corrections are never blended across a submap
/// boundary: the two sides are separate rigid pieces, and interpolation would
/// shear a submap that the correction is supposed to move as a whole.
fn project_keyframe_corrections(
    poses: &[Isometry3<f32>],
    boundaries: &[BoundaryKind],
    keyframes: &[(usize, Isometry3<f32>)],
    optimised: &[Isometry3<f32>],
) -> Vec<Isometry3<f32>> {
    let deltas: Vec<Isometry3<f32>> = keyframes
        .iter()
        .zip(optimised)
        .map(|((_, old), new)| new * old.inverse())
        .collect();

    // The submap that contains each frame. Both failure and budget boundaries
    // are rigid segments for correction projection.
    let mut starts = Vec::with_capacity(poses.len());
    let mut segment_start = 0;
    for frame in 0..poses.len() {
        if boundaries
            .get(frame)
            .copied()
            .unwrap_or(BoundaryKind::None)
            .is_boundary()
        {
            segment_start = frame;
        }
        starts.push(segment_start);
    }

    let mut ends = vec![poses.len(); poses.len()];
    let mut next_boundary = poses.len();
    for frame in (0..poses.len()).rev() {
        // `ends[frame]` is the next boundary strictly after this frame, so a
        // boundary frame starts a non-empty segment rather than ending one.
        ends[frame] = next_boundary;
        if boundaries
            .get(frame)
            .copied()
            .unwrap_or(BoundaryKind::None)
            .is_boundary()
        {
            next_boundary = frame;
        }
    }

    poses
        .iter()
        .enumerate()
        .map(|(frame, pose)| {
            let start = starts[frame];
            let end = ends[frame];

            let left = keyframes.partition_point(|(index, _)| *index < start);
            let right = keyframes.partition_point(|(index, _)| *index < end);

            let delta = if left == right {
                // A submap shorter than the keyframe policy has no node of its
                // own; use the nearest keyframe rather than inventing one.
                deltas[left.min(keyframes.len() - 1)]
            } else if frame <= keyframes[left].0 {
                deltas[left]
            } else if frame >= keyframes[right - 1].0 {
                deltas[right - 1]
            } else {
                let after = keyframes.partition_point(|(index, _)| *index <= frame);
                let b = after.min(right - 1);
                let a = b.saturating_sub(1).max(left);
                let (index_a, _) = keyframes[a];
                let (index_b, _) = keyframes[b];
                let t = (frame - index_a) as f32 / (index_b - index_a) as f32;
                interpolate_correction(deltas[a], deltas[b], t)
            };
            delta * pose
        })
        .collect()
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
        let graph_worker = if config.graph_worker && config.loop_closure.is_some() {
            Some(GraphWorker::spawn())
        } else {
            None
        };
        let loops = config.loop_closure.map(LoopFinder::new);

        Self {
            intrinsics,
            config,
            odometry,
            volume,
            frames: 0,
            fused: 0,
            rejected: 0,
            fusion_skipped: 0,
            last_fused: None,
            frames_since_fused: 0,
            chunk_start: 0,
            chunk_frames: 0,
            submap_fused: 0,
            pending_budget_boundary: false,
            trajectory: Vec::new(),
            points: Vec::new(),
            total_tracking: Duration::ZERO,
            total_fusion: Duration::ZERO,
            total_submap: Duration::ZERO,
            poses: Vec::new(),
            kept: Vec::new(),
            loops,
            graph_worker,
            graph_pending: false,
            graph_total_loops: 0,
            graph_updates: 0,
            color_views: Vec::new(),
            color_supplied: 0,
            dimensions: None,
            closure: None,
            finished_submaps: Vec::new(),
            world_from_submap: Isometry3::identity(),
            pose_boundaries: Vec::new(),
            submap_starts: Vec::new(),
            model_merged: false,
            online_updates: 0,
            last_online_correction: 0.0,
        }
    }

    fn fresh_volume(&self) -> TsdfVolume {
        TsdfVolume::new(TsdfParams {
            min_depth: self.config.depth_min,
            max_depth: self.config.depth_max,
            ..self.config.tsdf
        })
    }

    /// Finish the current submap and start a new one at `world_from_submap`.
    ///
    /// Extracting the finished surface now, rather than keeping every TSDF
    /// volume alive, frees the voxel blocks at the boundary. The mesh stays in
    /// the submap's own frame with its world transform beside it, so a later
    /// correction can move it without re-integrating anything. A full rebuild
    /// still regenerates the surface from the retained frames when accuracy
    /// matters more than time.
    fn begin_submap(&mut self, world_from_submap: Isometry3<f32>) {
        let mesh = self.volume.extract_mesh();
        let submap = self.submap_starts.len().saturating_sub(1);
        if !mesh.is_empty() {
            self.finished_submaps.push(FinishedSubmap {
                mesh,
                transform: self.world_from_submap,
                start: self.chunk_start,
                submap,
            });
        }
        self.volume = self.fresh_volume();
        self.world_from_submap = world_from_submap;
        if self.submap_starts.last().copied() != Some(self.frames) {
            self.submap_starts.push(self.frames);
        }
        self.chunk_start = self.frames;
        self.chunk_frames = 0;
        self.submap_fused = 0;
        self.pending_budget_boundary = false;
        // A new rigid piece means the model is no longer one merged world, even
        // if an earlier correction had collapsed the count to one.
        self.model_merged = false;
    }

    /// Finish the active tracking submap at a budget boundary and start a new
    /// one at the next frame. Odometry and `world_from_submap` continue
    /// unchanged, so the relative pose across the boundary is trusted.
    fn begin_continuous_submap(&mut self) {
        let mesh = self.volume.extract_mesh();
        if !mesh.is_empty() {
            let submap = self.submap_starts.len().saturating_sub(1);
            self.finished_submaps.push(FinishedSubmap {
                mesh,
                transform: self.world_from_submap,
                start: self.chunk_start,
                submap,
            });
        }
        self.volume = self.fresh_volume();
        self.chunk_start = self.frames + 1;
        self.chunk_frames = 0;
        self.submap_fused = 0;
        self.pending_budget_boundary = true;
        self.model_merged = false;
    }

    /// Finish the active TSDF into a cached chunk and start a fresh volume.
    ///
    /// The odometry frame and `world_from_submap` do not change: a chunk is a
    /// memory boundary inside one tracking submap, not a pose boundary. That is
    /// the difference between a submap budget and your submap
    /// failure path, and it is why this costs no trajectory accuracy.
    fn finish_chunk(&mut self) {
        let mesh = self.volume.extract_mesh();
        if !mesh.is_empty() {
            let submap = self.submap_starts.len().saturating_sub(1);
            self.finished_submaps.push(FinishedSubmap {
                mesh,
                transform: self.world_from_submap,
                start: self.chunk_start,
                submap,
            });
        }
        self.volume = self.fresh_volume();
        self.chunk_start = self.frames + 1;
        self.chunk_frames = 0;
    }

    /// The union of every submap's surface, in the world frame.
    fn geometry(&self) -> Mesh {
        let mut mesh = Mesh::default();
        for submap in &self.finished_submaps {
            append_transformed(&mut mesh, submap.mesh.clone(), submap.transform);
        }
        append_transformed(
            &mut mesh,
            self.volume.extract_mesh(),
            self.world_from_submap,
        );
        mesh
    }

    /// Feed one depth frame, in metres, row-major, `width * height` long.
    ///
    /// Anywhere the depth is `NaN` or non-positive is treated as no measurement.
    pub fn add_frame(&mut self, depth_metres: &[f32], width: usize, height: usize) -> FrameReport {
        self.add_frame_with_color(depth_metres, width, height, None)
    }

    /// As `add_frame`, but also keeping this frame's colour for texturing.
    ///
    /// `color` is the registered colour, its validity mask and the undistorted
    /// depth from the same frame, plus the colour camera's exposure settings.
    /// The pose is not passed in: it is the pose this frame is about to be
    /// tracked to. Colour is kept only when the frame is fused.
    pub fn add_frame_with_color(
        &mut self,
        depth_metres: &[f32],
        width: usize,
        height: usize,
        color: Option<FrameColor<'_>>,
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
        if first {
            self.submap_starts.push(0);
        } else if self.pending_budget_boundary
            && self.submap_starts.last().copied() != Some(self.frames)
        {
            // The previous frame finished the old submap. Register this frame as
            // the new node only when it actually arrives, so a scan that ends on
            // a budget boundary does not leave a trailing empty submap.
            self.submap_starts.push(self.frames);
        }

        let started = Instant::now();
        // Tracking consumes the depth image directly: the odometry builds its own
        // multi-resolution pyramid from it, so the back-projected cloud below is
        // only needed for fusion.
        //
        // The volume goes along as well so frame-to-model mode can render the
        // surface it has fused so far and align against that instead of against
        // the previous frame. `Odometry` ignores it unless the mode is on.
        let mut track = self.odometry.track(&image, Some(&self.volume));
        self.total_tracking += started.elapsed();

        // A frame the projective tracker cannot place has no overlap with the
        // current submap. Restart the odometry and make it the first frame of a
        // new submap. The origin is the candidate the failed alignment produced:
        // it is local to the old submap, so lift it into the world frame.
        let mut new_submap = first;
        let mut submap_reason = None;
        if !first
            && self.config.submaps
            && !track.accepted
            && track.rejection != Some("too little geometry in view")
        {
            submap_reason = track.rejection;
            let guess = self.world_from_submap * track.candidate;
            let finalizing = Instant::now();
            self.begin_submap(guess);
            self.total_submap += finalizing.elapsed();
            self.odometry.restart();
            let retry = Instant::now();
            track = self.odometry.track(&image, Some(&self.volume));
            self.total_tracking += retry.elapsed();
            new_submap = true;
        }

        // A background correction changes `world_from_submap` and every earlier
        // pose. Apply it before this frame's world pose is derived, or the frame
        // would be placed in the pre-correction frame.
        self.poll_graph();

        // The odometry pose is local to the current submap. Everything the
        // scanner stores, colours, or exports is in the world frame.
        let local_pose = track.pose;
        let pose = self.world_from_submap * local_pose;
        let candidate = self.world_from_submap * track.candidate;

        // The first frame of the scan or of a submap defines that submap's
        // origin and is always fused: there is no target to accept it against.
        let mut fuse = first || new_submap || track.accepted;
        let mut fusion_skipped = false;

        // Fusion motion filter. The frame is still tracked, still
        // retained for loop closure, and still contributes to the pose graph;
        // it just does not spend a fusion pass on a near-duplicate view. The
        // interval forces one through so a stationary model keeps averaging.
        if fuse && !first && !new_submap {
            if let (Some(filter), Some(last)) = (self.config.fusion_filter, self.last_fused) {
                let motion = last.inverse() * pose;
                if motion.translation.vector.norm() < filter.max_distance
                    && motion.rotation.angle() < filter.max_angle
                    && self.frames_since_fused < filter.max_interval_frames.max(1)
                {
                    fuse = false;
                    fusion_skipped = true;
                }
            }
        }

        let integration = if fuse {
            self.fused += 1;
            self.trajectory.push(pose.translation.vector);
            self.last_fused = Some(pose);
            self.frames_since_fused = 0;
            self.chunk_frames += 1;
            self.submap_fused += 1;

            let started = Instant::now();
            let stats = self.volume.integrate(&image, &intrinsics, &local_pose);
            self.total_fusion += started.elapsed();

            Some(stats)
        } else {
            if fusion_skipped {
                self.fusion_skipped += 1;
            } else {
                self.rejected += 1;
            }
            self.frames_since_fused += 1;
            None
        };

        // Bound the active volume. This is a memory boundary inside the current
        // tracking submap, so it deliberately does not touch the odometry pose,
        // `world_from_submap`, or `pose_boundaries`.
        if fuse
            && self.config.tsdf_chunk_frames > 0
            && self.chunk_frames >= self.config.tsdf_chunk_frames
        {
            let started = Instant::now();
            self.finish_chunk();
            self.total_submap += started.elapsed();
        }

        let boundary = if self.pending_budget_boundary {
            self.pending_budget_boundary = false;
            BoundaryKind::Budget
        } else if new_submap && !first {
            BoundaryKind::Failure
        } else {
            BoundaryKind::None
        };
        self.poses.push(pose);
        self.pose_boundaries.push(boundary);
        let frame_submap = self.submap_starts.len().saturating_sub(1);
        self.dimensions.get_or_insert((width, height));

        // Retain the frame before loop verification. An online correction
        // rebuilds the fused trajectory from this list, so it has to include
        // the frame being added; the rehydration callback only ever asks for
        // older keyframes, so seeing this one changes nothing there.
        if self.loops.is_some() {
            self.kept.push(KeptFrame {
                depth: depth_metres.to_vec(),
                fused: fuse,
            });
        }

        // Same rule as fusion: the first frame, a submap's first frame, or a
        // pose the tracker accepted. A rejected frame's colour would paint the
        // model from a pose that was never used to build it. It is still
        // counted, so the summary can say how many colour frames were dropped.
        //
        // The appearance signature is computed before `color` is consumed below.
        // It lets a loop candidate be proposed by what the camera saw, not only
        // by where the drifted pose says it is -- the RTAB-Map idea, with a
        // tiny descriptor standing in for a visual vocabulary.
        let appearance = if self.loops.is_some() {
            color
                .as_ref()
                .and_then(|color| VisualDescriptor::from_rgb(color.rgb, width, height))
        } else {
            None
        };
        if let Some(color) = color {
            self.color_supplied += 1;
            if fuse {
                self.color_views.push(ColorView {
                    color: color.rgb.to_vec(),
                    width,
                    height,
                    depth: color.depth.to_vec(),
                    pose,
                    intrinsics,
                    valid: color.valid.to_vec(),
                    exposure: color.exposure,
                    gain: color.gain,
                    gamma: color.gamma,
                    frame_index: self.frames,
                    tracking_quality: track.inlier_ratio,
                });
            }
        }

        // Offer this frame as somewhere the sensor might return to. The cloud is
        // the one already back-projected for fusion, so the extra cost is the
        // voxel hashing and, when a candidate turns up, one ICP alignment. An
        // old candidate can be rebuilt from its retained depth before that
        // alignment, so decimation costs memory rather than loop quality.
        let mut loop_activity = false;
        if let Some(finder) = self.loops.as_mut() {
            if fuse {
                let loop_config = finder.config();
                let kept = &self.kept;
                let intrinsics = self.intrinsics;
                let depth_min = self.config.depth_min;
                let depth_max = self.config.depth_max;
                loop_activity |= finder
                    .consider_with_rehydration(
                        self.frames,
                        pose,
                        &self.points,
                        appearance.as_ref(),
                        |index| {
                            let frame = kept.get(index)?;
                            if frame.depth.len() != width * height {
                                return None;
                            }
                            let image = DepthImage::new(width, height, &frame.depth);
                            let mut points = Vec::new();
                            for y in 0..height {
                                for x in 0..width {
                                    let depth = image.at(x, y);
                                    if !depth.is_finite() || depth <= 0.0 {
                                        continue;
                                    }
                                    if depth < depth_min || depth > depth_max {
                                        continue;
                                    }
                                    points.push(intrinsics.back_project(
                                        x as f32 + 0.5,
                                        y as f32 + 0.5,
                                        depth,
                                    ));
                                }
                            }
                            if points.is_empty() {
                                return None;
                            }
                            Some(IcpCloud::build(
                                &points,
                                loop_config.voxel,
                                loop_config.normal_radius,
                                loop_config.max_distance,
                                &Vector3::zeros(),
                            ))
                        },
                    )
                    .is_some();
            }

            // In background mode verification finishes a frame or two after
            // the keyframe was offered. Drain whatever is ready so the final
            // pass and online correction both see it.
            loop_activity |= !finder.poll().is_empty();
        }

        if loop_activity && self.config.online_loops {
            self.optimize_online();
        }

        // A budget boundary is a real submap node, but odometry continues
        // across it. That keeps the relative pose a measurement instead of a
        // guess, so the keyframe graph can keep the edge.
        if fuse && self.config.submaps && !new_submap {
            let current_pose = self.poses.last().copied().unwrap_or(pose);
            let start_pose = self
                .submap_starts
                .last()
                .and_then(|start| self.poses.get(*start))
                .copied();
            let frames_budget = self.config.submap_max_frames > 0
                && self.submap_fused >= self.config.submap_max_frames;
            let distance_budget = self.config.submap_max_translation > 0.0
                && start_pose.is_some_and(|start| {
                    (current_pose.translation.vector - start.translation.vector).norm()
                        >= self.config.submap_max_translation
                });
            if frames_budget || distance_budget {
                let started = Instant::now();
                self.begin_continuous_submap();
                self.total_submap += started.elapsed();
            }
        }

        self.frames += 1;

        // The caller sees world poses, not the odometry's local ones. Read the
        // committed pose back after the online correction, so a frame that was
        // just moved reports where it actually sits.
        track.pose = self.poses.last().copied().unwrap_or(pose);
        track.candidate = candidate;

        FrameReport {
            points: self.points.len(),
            track,
            integration,
            blocks: self.block_count(),
            allocated_bytes: self.allocated_bytes(),
            submap: frame_submap,
            new_submap,
            submap_reason,
        }
    }

    /// Optimise every verified loop and rebuild or move the model to the
    /// corrected poses.
    ///
    /// Call once, after the last frame. When online loops are enabled some
    /// corrections will already have been applied; this pass is still the final
    /// global one, and it waits for any background verification still in flight.
    pub fn close_loops(&mut self) -> ClosureReport {
        // An online worker job may still be running. Apply its result before
        // reading the loops, so the final pass sees the same trajectory the
        // capture thread would have.
        self.flush_graph();
        if let Some(finder) = self.loops.as_mut() {
            finder.flush();
        }

        let loop_edges = self
            .loops
            .as_ref()
            .map(|finder| finder.edges_ranked())
            .unwrap_or_default();
        let total_loops = loop_edges.len();
        let nodes = self
            .loops
            .as_ref()
            .map(|finder| finder.keyframes().len())
            .unwrap_or(0);

        let nothing = ClosureReport {
            nodes,
            keyframe_nodes: nodes,
            submap_nodes: 0,
            odometry_edges: 0,
            loop_edges: total_loops,
            dropped_loops: 0,
            cost_before: 0.0,
            cost_after: 0.0,
            iterations: 0,
            converged: true,
            max_correction: 0.0,
            max_rotation: 0.0,
            refused: false,
            rebuilt: false,
            online: false,
        };

        // With no revisit there is nothing to redistribute, so the odometry
        // trajectory stands unaltered.
        if nodes < 2 || loop_edges.is_empty() {
            self.closure = Some(nothing);
            return nothing;
        }

        let Some((outcome, kept, within)) = self.optimise_keyframe_graph(loop_edges) else {
            self.closure = Some(nothing);
            return nothing;
        };

        // Simplicity beats quiet corruption. If even the strongest loop wants
        // to fold the map, believe the odometry and leave the model alone.
        if !within {
            let closure = self.report_from_outcome(&outcome, 0, total_loops, true, false, false);
            self.closure = Some(closure);
            return closure;
        }

        let mode = if self.config.incremental_rebuild {
            CorrectionMode::Incremental
        } else {
            CorrectionMode::FullRebuild
        };
        let closure = self.apply_corrections(&outcome, mode, kept, total_loops, false);
        self.closure = Some(closure);
        closure
    }

    /// Apply a verified loop during the scan.
    ///
    /// Returns `None` when there is nothing to do, or when the correction was
    /// refused. A refusal leaves the trajectory exactly as odometry had it; the
    /// final pass can still retry once more loops have arrived.
    fn optimize_online(&mut self) -> Option<ClosureReport> {
        let loop_edges = self.loops.as_ref()?.edges_ranked();
        let total_loops = loop_edges.len();
        if total_loops == 0 {
            return None;
        }

        if self.config.graph_worker {
            if self.graph_pending {
                return None;
            }
            let job = if self.config.submap_graph {
                GraphJob::Submap(SubmapGraphJob {
                    poses: self.poses.clone(),
                    loop_edges,
                    submap_starts: self.submap_starts.clone(),
                    boundaries: self.pose_boundaries.clone(),
                })
            } else {
                GraphJob::Keyframe(self.keyframe_job(loop_edges)?)
            };
            let Some(worker) = self.graph_worker.as_ref() else {
                return None;
            };
            let Some(sender) = worker.sender.as_ref() else {
                return None;
            };
            if sender.send(job).is_ok() {
                self.graph_pending = true;
                self.graph_total_loops = total_loops;
            }
            return None;
        }

        let (outcome, kept, within) = self.optimise_keyframe_graph(loop_edges)?;
        if !within {
            let closure = self.report_from_outcome(&outcome, 0, total_loops, true, false, true);
            self.closure = Some(closure);
            return Some(closure);
        }
        self.online_updates += 1;
        if outcome.max_correction > 0.0 {
            // A later re-solve can legitimately be zero because the first loop
            // already fixed the trajectory. Keep the last non-zero move so the
            // telemetry still shows what online correction did.
            self.last_online_correction = outcome.max_correction;
        }
        let closure =
            self.apply_corrections(&outcome, CorrectionMode::Online, kept, total_loops, true);
        self.closure = Some(closure);
        Some(closure)
    }

    /// Drain a finished background graph result, if one is ready.
    fn poll_graph(&mut self) {
        if !self.graph_pending {
            return;
        }
        let Some(worker) = self.graph_worker.as_mut() else {
            self.graph_pending = false;
            return;
        };
        match worker.receiver.try_recv() {
            Ok(result) => {
                self.graph_pending = false;
                self.accept_graph_result(result);
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => self.graph_pending = false,
        }
    }

    /// Wait for an in-flight graph job before the final pass rewrites the
    /// trajectory. A 30 s backstop matches the loop worker's.
    fn flush_graph(&mut self) {
        if !self.graph_pending {
            return;
        }
        let Some(worker) = self.graph_worker.as_mut() else {
            self.graph_pending = false;
            return;
        };
        match worker.receiver.recv_timeout(Duration::from_secs(30)) {
            Ok(result) => {
                self.graph_pending = false;
                self.accept_graph_result(result);
            }
            Err(_) => self.graph_pending = false,
        }
    }

    fn accept_graph_result(&mut self, result: Option<(GraphOutcome, usize, bool)>) {
        let Some((outcome, kept, within)) = result else {
            return;
        };
        if within {
            self.graph_updates += 1;
            self.online_updates += 1;
            if outcome.max_correction > 0.0 {
                self.last_online_correction = outcome.max_correction;
            }
            let closure = self.apply_corrections(
                &outcome,
                CorrectionMode::Online,
                kept,
                self.graph_total_loops,
                true,
            );
            self.closure = Some(closure);
        } else {
            // Refuse the correction, but keep the report so the caller can see
            // that the worker ran and why it was rejected.
            let closure =
                self.report_from_outcome(&outcome, 0, self.graph_total_loops, true, false, true);
            self.closure = Some(closure);
        }
    }

    /// Build and optimise the keyframe graph, dropping the weakest loop while
    /// the requested correction stays implausible.
    ///
    /// Returns the projected correction, how many loop edges were kept, and
    /// whether the result is within the plausibility bounds. `None` when there
    /// is no keyframe graph to optimise.
    fn optimise_keyframe_graph(
        &self,
        loop_edges: Vec<Edge>,
    ) -> Option<(GraphOutcome, usize, bool)> {
        // A sparse pose-adjustment shape, when asked for: submap nodes with the
        // keyframes rigidly attached. If none of the loops actually cross a
        // submap, fall through to the keyframe graph rather than returning no
        // correction at all.
        if self.config.submap_graph {
            if let Some(result) = self.optimise_submap_graph(loop_edges.clone()) {
                return Some(result);
            }
        }

        let job = self.keyframe_job(loop_edges)?;
        solve_keyframe_graph(&job)
    }

    /// Snapshot the keyframe graph inputs so the same solve can run synchronously
    /// or on [`GraphWorker`].
    fn keyframe_job(&self, loop_edges: Vec<Edge>) -> Option<KeyframeGraphJob> {
        let finder = self.loops.as_ref()?;
        let keyframes: Vec<(usize, Isometry3<f32>)> = finder
            .keyframes()
            .iter()
            .map(|keyframe| (keyframe.index, keyframe.pose))
            .collect();
        if keyframes.len() < 2 {
            return None;
        }
        Some(KeyframeGraphJob {
            poses: self.poses.clone(),
            boundaries: self.pose_boundaries.clone(),
            keyframes,
            loop_edges,
        })
    }

    /// Optimise submap nodes with the keyframes rigidly attached.
    ///
    /// This is a sparse pose-adjustment shape. The graph is much
    /// smaller than the keyframe graph, and a correction moves a whole submap,
    /// which matches the cached surfaces and the incremental path. It cannot
    /// deform inside a submap, so it is opt-in.
    fn optimise_submap_graph(&self, loop_edges: Vec<Edge>) -> Option<(GraphOutcome, usize, bool)> {
        solve_submap_graph(&SubmapGraphJob {
            poses: self.poses.clone(),
            loop_edges,
            submap_starts: self.submap_starts.clone(),
            boundaries: self.pose_boundaries.clone(),
        })
    }

    fn report_from_outcome(
        &self,
        outcome: &GraphOutcome,
        kept_loops: usize,
        total_loops: usize,
        refused: bool,
        rebuilt: bool,
        online: bool,
    ) -> ClosureReport {
        ClosureReport {
            nodes: outcome.nodes,
            keyframe_nodes: outcome.keyframe_nodes,
            submap_nodes: outcome.submap_nodes,
            odometry_edges: outcome.odometry_edges,
            loop_edges: kept_loops,
            dropped_loops: total_loops.saturating_sub(kept_loops),
            cost_before: outcome.cost_before,
            cost_after: outcome.cost_after,
            iterations: outcome.iterations,
            converged: outcome.converged,
            max_correction: outcome.max_correction,
            max_rotation: outcome.max_rotation,
            refused,
            rebuilt,
            online,
        }
    }

    fn apply_corrections(
        &mut self,
        outcome: &GraphOutcome,
        mode: CorrectionMode,
        kept_loops: usize,
        total_loops: usize,
        online: bool,
    ) -> ClosureReport {
        // The solve can be older than the current trajectory when it ran on the
        // worker. Build one correction per *current* frame, extending the last
        // correction to frames captured while the solve was in flight. That
        // keeps the local odometry intact instead of truncating the trajectory.
        let old_poses = self.poses.clone();
        let mut corrections = vec![Isometry3::<f32>::identity(); old_poses.len()];

        if let Some(new_submap_poses) = &outcome.submap_poses {
            let old_submap_poses: Vec<Isometry3<f32>> = self
                .submap_starts
                .iter()
                .filter_map(|start| old_poses.get(*start).copied())
                .collect();
            let submap_corrections: Vec<Isometry3<f32>> = new_submap_poses
                .iter()
                .zip(&old_submap_poses)
                .map(|(new, old)| new * old.inverse())
                .collect();
            let last = submap_corrections
                .last()
                .copied()
                .unwrap_or_else(Isometry3::identity);
            for (frame, correction) in corrections.iter_mut().enumerate() {
                let submap = self
                    .submap_starts
                    .partition_point(|start| *start <= frame)
                    .saturating_sub(1);
                *correction = submap_corrections.get(submap).copied().unwrap_or(last);
            }
        } else {
            let snapshot = outcome.poses.len().min(old_poses.len());
            let last = if snapshot > 0 {
                outcome.poses[snapshot - 1] * old_poses[snapshot - 1].inverse()
            } else {
                Isometry3::identity()
            };
            for (frame, correction) in corrections.iter_mut().enumerate() {
                *correction = if frame < snapshot {
                    outcome.poses[frame] * old_poses[frame].inverse()
                } else {
                    last
                };
            }
        }

        self.poses = old_poses
            .iter()
            .zip(&corrections)
            .map(|(pose, correction)| correction * pose)
            .collect();

        // Keyframe graph: move a finished chunk by the correction at its first
        // frame. Submap graph: every frame in a submap shares that correction.
        // Either way this is cheap and correct for a rigid piece, but it cannot
        // fix drift within one.
        for submap in &mut self.finished_submaps {
            if let Some(correction) = corrections.get(submap.start) {
                submap.transform = *correction * submap.transform;
            }
        }

        // The odometry's local frame did not move; the world transform did.
        // Anchor the current submap on the last corrected world pose so the next
        // frame continues from the corrected trajectory.
        if let Some(last) = self.poses.last() {
            self.world_from_submap = *last * self.odometry.pose().inverse();
        }

        self.trajectory.clear();
        for (index, kept) in self.kept.iter().enumerate() {
            if kept.fused {
                if let Some(pose) = self.poses.get(index) {
                    self.trajectory.push(pose.translation.vector);
                }
            }
        }

        self.update_color_view_poses();
        if let Some(finder) = self.loops.as_mut() {
            finder.update_poses(&self.poses);
        }

        match mode {
            CorrectionMode::Online => {}
            CorrectionMode::Incremental => self.model_merged = true,
            CorrectionMode::FullRebuild => self.rebuild(),
        }

        self.report_from_outcome(outcome, kept_loops, total_loops, false, true, online)
    }

    /// Re-integrate every kept frame with the corrected poses.
    ///
    /// A fresh volume, because geometry already fused at the old poses cannot be
    /// moved -- only discarded and laid down again. This is the accuracy path;
    /// `CorrectionMode::Incremental` is the cheap one.
    fn rebuild(&mut self) {
        let Some((width, height)) = self.dimensions else {
            return;
        };
        let intrinsics = self.intrinsics;

        self.volume = self.fresh_volume();
        // The rebuild lays every fused frame into one global volume at the
        // corrected poses, so the cached submap surfaces are now redundant.
        self.finished_submaps.clear();
        self.world_from_submap = Isometry3::identity();
        self.submap_starts.clear();
        if self.frames > 0 {
            self.submap_starts.push(0);
        }
        self.chunk_start = 0;
        self.chunk_frames = 0;
        self.submap_fused = 0;
        self.pending_budget_boundary = false;
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
        self.model_merged = true;
    }

    /// Point each kept colour view at the corrected pose of the frame it came from.
    ///
    /// Views are stored only for fused frames, so they are not parallel to
    /// `poses`. `frame_index` is the arrival index, which is.
    fn update_color_view_poses(&mut self) {
        for view in &mut self.color_views {
            if let Some(pose) = self.poses.get(view.frame_index).copied() {
                view.pose = pose;
            }
        }
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

    /// How many spatial ICP attempts the loop finder has spent.
    pub fn loop_attempts(&self) -> usize {
        self.loops
            .as_ref()
            .map(|finder| finder.attempts())
            .unwrap_or(0)
    }

    /// Points still held by the loop keyframe index.
    pub fn loop_keyframe_points(&self) -> usize {
        self.loops
            .as_ref()
            .map(|finder| finder.keyframe_points())
            .unwrap_or(0)
    }

    /// How many long-term keyframe clouds were rebuilt for verification.
    pub fn loop_rehydrations(&self) -> usize {
        self.loops
            .as_ref()
            .map(|finder| finder.rehydrations())
            .unwrap_or(0)
    }

    /// A pre-training sanity check. Call after `close_loops` so the submap
    /// count reflects any correction that was actually rebuilt.
    pub fn quality(&self) -> ScanQuality {
        let closure = self.closure();
        let loop_edges = closure.map(|report| report.loop_edges).unwrap_or(0);
        let rebuilt = closure.map(|report| report.rebuilt).unwrap_or(false);
        let refused = closure.map(|report| report.refused).unwrap_or(false);

        let mut warnings = Vec::new();
        if self.frames == 0 {
            warnings.push("no frames were captured".to_string());
        } else if self.fused == 0 {
            warnings.push("no frames were fused".to_string());
        } else if (self.fused + self.fusion_skipped) * 2 < self.frames {
            warnings.push(format!(
                "only {} of {} frames were fused ({} motion-filtered)",
                self.fused, self.frames, self.fusion_skipped
            ));
        }
        if self.frames > 0 && self.rejected > 0 && self.rejected * 3 >= self.frames {
            warnings.push(format!(
                "{} of {} frames were rejected",
                self.rejected, self.frames
            ));
        }
        if self.submaps() > 1 {
            warnings.push(format!(
                "{} submaps remain unmerged; a trainer will see disjoint fragments",
                self.submaps()
            ));
        }
        if refused {
            warnings.push(
                "loop closure found a correction but refused it as implausible; \
                 the poses remain uncorrected"
                    .to_string(),
            );
        }

        ScanQuality {
            frames: self.frames,
            fused: self.fused,
            rejected: self.rejected,
            submaps: self.submaps(),
            loop_edges,
            rebuilt,
            refused,
            final_distance: self.pose().translation.vector.norm(),
            warnings,
        }
    }

    pub fn mesh(&self) -> Mesh {
        self.painted_mesh().0
    }

    /// Geometry for the live preview, optionally painted from the most recent
    /// `max_color_views` colour views.
    ///
    /// The full painter walks every stored view for every vertex, which is the
    /// right trade for a file and the wrong one for a preview that has to keep up
    /// with capture: the cost would grow with the length of the scan. Limiting
    /// the view window keeps the preview bounded. The exported mesh is still
    /// painted from every view by [`Self::export_mesh`].
    ///
    /// `max_color_views == 0` returns geometry only. A reconstruction with no
    /// surface, or no colour views, does the same.
    pub fn preview_mesh(&self, max_color_views: usize) -> Mesh {
        let mut mesh = self.geometry();
        if mesh.is_empty() || self.color_views.is_empty() || max_color_views == 0 {
            return mesh;
        }

        let start = self.color_views.len().saturating_sub(max_color_views);
        let views = &self.color_views[start..];
        let normals = mesh.vertex_normals();
        let (colors, _) = colorize(&mesh.vertices, &normals, views, &self.config.coloring);
        mesh.colors = Some(colors);
        mesh
    }

    /// The extracted mesh, plus the painter's report when colour views exist.
    fn painted_mesh(&self) -> (Mesh, Option<ColoringReport>) {
        let mut mesh = self.geometry();
        if self.color_views.is_empty() {
            return (mesh, None);
        }

        let normals = mesh.vertex_normals();
        let (colors, report) = colorize(
            &mesh.vertices,
            &normals,
            &self.color_views,
            &self.config.coloring,
        );
        mesh.colors = Some(colors);
        (mesh, Some(report))
    }

    /// Atlas for the colour views, in the scanner's world frame.
    ///
    /// `None` when texturing is off or the reconstruction has no surface. The
    /// result is not mirrored: views were captured in this frame, so a caller
    /// that wants real-world left and right has to [`TexturedMesh::flip_x`]
    /// afterwards. [`export_mesh`] does that, and only after the atlas exists.
    pub fn textured_mesh(&self) -> Option<(TexturedMesh, TexturingReport)> {
        if self.color_views.is_empty() {
            return None;
        }
        let mesh = self.geometry();
        if mesh.is_empty() {
            return None;
        }
        let params = TexturingParams {
            coloring: self.config.coloring,
            ..TexturingParams::default()
        };
        Some(texture(&mesh, &self.color_views, &params))
    }

    /// The mesh `finish` writes, with the mirror applied after texturing.
    ///
    /// `atlas` is how the chosen file carries colour. An atlas is built from
    /// the unflipped surface and only then mirrored, so positions, normals and
    /// winding move together while the UVs stay pointed at the image that was
    /// just painted. PLY passes `atlas: false` and keeps per-vertex colour.
    /// With no colour views the result is geometry either way.
    pub fn export_mesh(&self, unmirror: bool, atlas: bool) -> Option<ExportMesh> {
        if atlas {
            if let Some((mut textured, report)) = self.textured_mesh() {
                if unmirror {
                    textured.flip_x();
                }
                return Some(ExportMesh::Atlas(textured, report));
            }
        }

        let (mut mesh, coloring) = self.painted_mesh();
        if mesh.is_empty() {
            return None;
        }
        if unmirror {
            mesh.flip_x();
        }
        Some(ExportMesh::Geometry(mesh, coloring))
    }

    /// The frames that contributed colour, for the live preview.
    pub fn color_views(&self) -> &[ColorView] {
        &self.color_views
    }

    /// How many frames contributed colour. Zero when texturing is off.
    pub fn color_view_count(&self) -> usize {
        self.color_views.len()
    }

    /// How many frames arrived with a colour buffer, kept or not.
    pub fn color_supplied(&self) -> usize {
        self.color_supplied
    }

    /// Colour buffers that were not kept because the frame was rejected.
    pub fn color_dropped(&self) -> usize {
        self.color_supplied.saturating_sub(self.color_views.len())
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

    /// Frames that tracked but the fusion motion filter skipped.
    pub fn fusion_skipped(&self) -> usize {
        self.fusion_skipped
    }

    /// Cached TSDF chunks. Chunks are memory boundaries inside tracking
    /// submaps, so this can exceed [`Self::submaps`].
    pub fn chunks(&self) -> usize {
        self.finished_submaps.len()
    }

    pub fn block_count(&self) -> usize {
        self.volume.block_count()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.volume.allocated_bytes() + self.submap_surface_bytes()
    }

    /// Memory held by the cached finished-submap surfaces, in bytes.
    pub fn submap_surface_bytes(&self) -> usize {
        self.finished_submaps
            .iter()
            .map(|submap| mesh_bytes(&submap.mesh))
            .sum()
    }

    /// Cumulative time spent tracking frames.
    pub fn total_tracking(&self) -> Duration {
        self.total_tracking
    }

    /// Cumulative time spent fusing frames into the TSDF.
    pub fn total_fusion(&self) -> Duration {
        self.total_fusion
    }

    /// Cumulative time spent finalising finished submap surfaces.
    pub fn total_submap(&self) -> Duration {
        self.total_submap
    }

    /// How many submaps the frames have been split across.
    ///
    /// An incremental or full correction places every piece in one world frame,
    /// so the scan reports one model even when the pieces are stored separately.
    pub fn submaps(&self) -> usize {
        if self.model_merged {
            1
        } else {
            self.submap_starts.len()
        }
    }

    /// How many loop corrections were applied during the scan.
    pub fn online_loop_updates(&self) -> usize {
        self.online_updates
    }

    /// Largest pose move from the most recent online correction, in metres.
    pub fn last_online_correction(&self) -> f32 {
        self.last_online_correction
    }

    /// True while a background graph solve is in flight.
    pub fn graph_pending(&self) -> bool {
        self.graph_pending
    }

    /// How many online corrections the background graph worker applied.
    pub fn background_graph_updates(&self) -> usize {
        self.graph_updates
    }

    /// Loop verifications still in flight on the background worker.
    pub fn loop_pending(&self) -> usize {
        self.loops
            .as_ref()
            .map(|finder| finder.pending())
            .unwrap_or(0)
    }

    /// Zero-based arrival index of each submap's first frame.
    pub fn submap_starts(&self) -> &[usize] {
        &self.submap_starts
    }

    /// The current frame's pose in the scan's world frame.
    pub fn pose(&self) -> Isometry3<f32> {
        self.poses
            .last()
            .copied()
            .unwrap_or_else(|| self.world_from_submap * self.odometry.pose())
    }
}

/// Append `mesh`, transformed, to `target`.
///
/// Indices are rebased; colours are left out because the caller paints the
/// merged surface from the colour views after every submap is in place.
fn append_transformed(target: &mut Mesh, mesh: Mesh, transform: Isometry3<f32>) {
    let base = target.vertices.len() as u32;
    target.vertices.extend(
        mesh.vertices
            .into_iter()
            .map(|vertex| transform_point(&transform, &vertex)),
    );
    target.triangles.extend(
        mesh.triangles
            .into_iter()
            .map(|triangle| [triangle[0] + base, triangle[1] + base, triangle[2] + base]),
    );
}

fn mesh_bytes(mesh: &Mesh) -> usize {
    mesh.vertices.len() * std::mem::size_of::<Vector3<f32>>()
        + mesh.triangles.len() * std::mem::size_of::<[u32; 3]>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geom::coloring::ColorMode;

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

    /// Render a rectangular room interior from the origin.
    fn room_frame(width: usize, height: usize, intrinsics: &Intrinsics) -> Vec<f32> {
        room_frame_from(width, height, intrinsics, &Vector3::zeros(), 0.0)
    }

    /// Render the room interior from `eye`, yawed about the vertical axis.
    ///
    /// The stored depth is the camera-frame z of the hit, so this is a valid
    /// depth image for a pose whose rotation is that yaw.
    fn room_frame_from(
        width: usize,
        height: usize,
        intrinsics: &Intrinsics,
        eye: &Vector3<f32>,
        yaw: f32,
    ) -> Vec<f32> {
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
        let rotation = nalgebra::UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);

        let mut frame = vec![f32::NAN; width * height];

        for y in 0..height {
            for x in 0..width {
                let local = Vector3::new(
                    (x as f32 + 0.5 - intrinsics.cx) / intrinsics.fx,
                    (y as f32 + 0.5 - intrinsics.cy) / intrinsics.fy,
                    1.0,
                );
                let direction = rotation * local;

                let mut nearest = f32::INFINITY;
                for (normal, offset) in &planes {
                    let denominator = normal.dot(&direction);
                    if denominator.abs() < 1e-6 {
                        continue;
                    }

                    let t = (offset - normal.dot(eye)) / denominator;
                    if t <= 0.0 || t >= nearest {
                        continue;
                    }

                    let hit = *eye + direction * t;
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
                    // The local ray has z = 1, so t is the camera-frame depth.
                    frame[y * width + x] = nearest;
                }
            }
        }

        frame
    }

    /// Loop settings for the short synthetic scans. The default 5 cm keyframe
    /// policy and 40-frame gap are right for a real room but would produce one
    /// node and no candidates here.
    fn short_scan_loops() -> LoopClosureConfig {
        LoopClosureConfig {
            min_index_gap: 2,
            // The old default policy, still appropriate for a handful of
            // synthetic frames. The shipped real-scan defaults are coarser.
            keyframe_min_translation: 0.02,
            keyframe_min_rotation: 0.05,
            keyframe_max_interval: 5,
            ..LoopClosureConfig::default()
        }
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
                loop_closure: Some(short_scan_loops()),
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

        let quality = scanner.quality();
        assert!(quality.trainable(), "{:?}", quality.warnings);
        assert_eq!(quality.submaps, 1);
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
    fn a_lost_view_starts_a_second_submap() {
        let mut scanner = scanner();
        let front = room_frame(WIDTH, HEIGHT, &intrinsics());
        let side = room_frame_from(
            WIDTH,
            HEIGHT,
            &intrinsics(),
            &Vector3::zeros(),
            std::f32::consts::FRAC_PI_2,
        );

        let first = scanner.add_frame(&front, WIDTH, HEIGHT);
        assert!(first.integration.is_some());

        let transition = scanner.add_frame(&side, WIDTH, HEIGHT);
        assert!(
            transition.new_submap,
            "the side view should not track the front view"
        );
        assert_eq!(transition.submap, 1);
        assert!(
            transition.integration.is_some(),
            "the submap origin is fused"
        );

        let confirmed = scanner.add_frame(&side, WIDTH, HEIGHT);
        assert!(!confirmed.new_submap);
        assert!(
            confirmed.integration.is_some(),
            "the side view should track itself"
        );

        assert_eq!(scanner.frames(), 3);
        assert_eq!(scanner.submaps(), 2);
        assert_eq!(scanner.fused(), 3);
        assert_eq!(scanner.rejected(), 0);
        assert!(!scanner.mesh().is_empty());

        let quality = scanner.quality();
        assert!(!quality.trainable());
        assert_eq!(quality.submaps, 2);
        assert!(
            quality
                .warnings
                .iter()
                .any(|warning| warning.contains("unmerged")),
            "{:?}",
            quality.warnings
        );
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
    fn a_scan_rehydrates_a_long_term_keyframe_from_its_depth() {
        let mut scanner = Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                loop_closure: Some(LoopClosureConfig {
                    min_index_gap: 1,
                    keyframe_min_translation: 0.0,
                    keyframe_min_rotation: 0.0,
                    keyframe_max_interval: 1,
                    working_memory_keyframes: 1,
                    long_term_decimation: 2,
                    ..LoopClosureConfig::default()
                }),
                ..ScannerConfig::default()
            },
        );
        let frame = room_frame(WIDTH, HEIGHT, &intrinsics());

        for _ in 0..4 {
            scanner.add_frame(&frame, WIDTH, HEIGHT);
        }

        assert!(
            scanner.loop_attempts() > 0,
            "no candidate was verified, so rehydration was never exercised"
        );
        assert!(
            scanner.loop_rehydrations() > 0,
            "the long-term cloud was never rebuilt from depth"
        );
        assert!(
            scanner.loop_keyframe_points() > 0,
            "the loop index lost its remaining clouds"
        );
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
        assert!(
            !scanner.mesh().is_empty(),
            "the rebuild produced no surface"
        );
        assert_eq!(scanner.fused(), 6, "the rebuild dropped frames");
    }

    #[test]
    fn a_revisit_merges_submaps_at_the_corrected_pose() {
        let mut scanner = scanner_with_loops();
        let front = room_frame(WIDTH, HEIGHT, &intrinsics());
        let side = room_frame_from(
            WIDTH,
            HEIGHT,
            &intrinsics(),
            &Vector3::zeros(),
            std::f32::consts::FRAC_PI_2,
        );

        // The return view is the front view again, but from 20 cm to the right.
        // Its submap is born at the failed alignment's guess (here, the origin),
        // so only the loop edge can put it in the right place.
        let returned = room_frame_from(
            WIDTH,
            HEIGHT,
            &intrinsics(),
            &Vector3::new(0.2, 0.0, 0.0),
            0.0,
        );

        let mut boundaries = Vec::new();
        for _ in 0..2 {
            boundaries.push(scanner.add_frame(&front, WIDTH, HEIGHT).new_submap);
        }
        for _ in 0..2 {
            boundaries.push(scanner.add_frame(&side, WIDTH, HEIGHT).new_submap);
        }
        for _ in 0..3 {
            boundaries.push(scanner.add_frame(&returned, WIDTH, HEIGHT).new_submap);
        }

        assert_eq!(
            boundaries,
            [true, false, true, false, true, false, false],
            "the side and return views should each start a submap"
        );
        assert_eq!(scanner.submaps(), 3);
        assert_eq!(scanner.fused(), 7);
        assert_eq!(scanner.rejected(), 0);
        assert!(
            scanner.loops() > 0,
            "the return view was not verified as a revisit"
        );

        let report = scanner.close_loops();

        assert!(report.loop_edges > 0, "no loop edge reached the pose graph");
        assert!(report.rebuilt, "the corrected model was not rebuilt");
        assert!(!report.refused, "the correction was refused");
        assert!(
            (report.max_correction - 0.2).abs() < 0.05,
            "the submap moved {:.3} m, expected about 0.2 m",
            report.max_correction
        );
        assert_eq!(scanner.submaps(), 1, "the rebuild did not merge");
        assert_eq!(scanner.fused(), 7, "the rebuild dropped frames");
        assert_eq!(scanner.rejected(), 0);

        let position = scanner.pose().translation.vector;
        assert!(
            (position - Vector3::new(0.2, 0.0, 0.0)).norm() < 0.05,
            "the corrected return pose is {position:?}"
        );
        assert!(!scanner.mesh().is_empty(), "the merged model is empty");

        let quality = scanner.quality();
        assert!(quality.trainable(), "{:?}", quality.warnings);
        assert_eq!(quality.submaps, 1);
        assert!(quality.loop_edges > 0);
        assert!(quality.rebuilt);
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

    fn frame_color<'a>(
        rgb: &'a [u8],
        depth: &'a [f32],
        valid: &'a [u8],
        exposure: f32,
    ) -> FrameColor<'a> {
        FrameColor {
            rgb,
            depth,
            valid,
            exposure,
            gain: 1.5,
            gamma: 2.2,
        }
    }

    #[test]
    fn colour_is_kept_only_for_fused_frames_and_records_where_it_came_from() {
        let mut scanner = scanner();
        let room = room_frame(WIDTH, HEIGHT, &intrinsics());
        let pixels = WIDTH * HEIGHT;
        let rgb = vec![40u8; pixels * 3];
        let color_depth = vec![2.0f32; pixels];
        let mut valid = vec![1u8; pixels];
        valid[0] = 0;

        let first = scanner.add_frame_with_color(
            &room,
            WIDTH,
            HEIGHT,
            Some(frame_color(&rgb, &color_depth, &valid, 0.01)),
        );
        assert!(first.integration.is_some());
        assert_eq!(scanner.color_view_count(), 1);
        assert_eq!(scanner.color_views()[0].frame_index, 0);
        assert!(
            (scanner.color_views()[0].tracking_quality - first.track.inlier_ratio).abs() < 1e-6
        );
        assert!((scanner.color_views()[0].tracking_quality - 1.0).abs() < 1e-6);
        assert_eq!(scanner.color_views()[0].exposure, 0.01);
        assert_eq!(scanner.color_views()[0].gain, 1.5);
        assert_eq!(scanner.color_views()[0].gamma, 2.2);
        assert_eq!(scanner.color_views()[0].valid[0], 0);
        assert_eq!(scanner.color_views()[0].valid[1], 1);

        // Nothing in view: the tracker refuses it, and so does the colour log.
        let empty = vec![f32::NAN; pixels];
        let rejected = scanner.add_frame_with_color(
            &empty,
            WIDTH,
            HEIGHT,
            Some(frame_color(&rgb, &color_depth, &valid, 0.5)),
        );
        assert!(rejected.integration.is_none());
        assert!(!rejected.track.accepted);
        assert_eq!(scanner.color_view_count(), 1);
        assert_eq!(scanner.color_supplied(), 2);
        assert_eq!(scanner.color_dropped(), 1);
        assert_eq!(scanner.rejected(), 1);

        let third = scanner.add_frame_with_color(
            &room,
            WIDTH,
            HEIGHT,
            Some(frame_color(&rgb, &color_depth, &valid, 0.02)),
        );
        assert!(
            third.track.accepted,
            "the room should match the first frame"
        );
        assert_eq!(scanner.color_view_count(), 2);
        assert_eq!(scanner.fused(), 2);
        assert_eq!(scanner.color_views()[1].frame_index, 2);
        assert!(
            (scanner.color_views()[1].tracking_quality - third.track.inlier_ratio).abs() < 1e-6
        );
        assert_eq!(scanner.color_views()[1].exposure, 0.02);
        assert_eq!(scanner.color_supplied(), 3);
        assert_eq!(scanner.color_dropped(), 1);
    }

    #[test]
    fn an_accepted_loop_closure_moves_colour_views_onto_the_corrected_poses() {
        let mut scanner = scanner_with_loops();
        let room = room_frame(WIDTH, HEIGHT, &intrinsics());
        let pixels = WIDTH * HEIGHT;
        let rgb = vec![10u8; pixels * 3];
        let color_depth = vec![2.0f32; pixels];
        let valid = vec![1u8; pixels];

        for _ in 0..6 {
            scanner.add_frame_with_color(
                &room,
                WIDTH,
                HEIGHT,
                Some(frame_color(&rgb, &color_depth, &valid, 0.01)),
            );
        }

        assert!(scanner.loops() > 0, "no revisits were verified");
        assert_eq!(scanner.color_view_count(), 6);

        // A stale pose, as if colour were still sitting where odometry left a
        // frame that the graph has since moved. Closing the loops has to
        // overwrite it; leaving it would paint the mesh from the wrong camera.
        scanner.color_views[0].pose = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(5.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
        );

        let report = scanner.close_loops();
        assert!(report.rebuilt, "the correction was not applied");
        assert!(!report.refused);

        for view in scanner.color_views() {
            let corrected = scanner.poses[view.frame_index];
            let delta = view.pose.inverse() * corrected;
            assert!(
                delta.translation.vector.norm() < 1e-5,
                "view {} stayed at {:?}, corrected pose is {:?}",
                view.frame_index,
                view.pose.translation.vector,
                corrected.translation.vector
            );
        }
        assert!(
            scanner.color_views()[0].pose.translation.vector.norm() < 0.05,
            "the sabotaged view was not replaced"
        );
    }

    #[test]
    fn scanner_config_defaults_to_best_colour_and_two_centimetres() {
        let coloring = ScannerConfig::default().coloring;
        assert_eq!(coloring.mode, ColorMode::Best);
        assert!((coloring.depth_tolerance - 0.02).abs() < 1e-6);
    }

    #[test]
    fn coloring_params_on_the_config_reach_the_mesh() {
        // The mask is empty, so every vertex is unobserved and must come back
        // as the fallback this config asked for. That is the whole check: the
        // painter ran with `ScannerConfig::coloring`, not with a private default.
        let mut scanner = Scanner::new(
            intrinsics(),
            ScannerConfig {
                coloring: ColoringParams {
                    fallback: [9, 8, 7],
                    ..ColoringParams::default()
                },
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                ..ScannerConfig::default()
            },
        );
        let room = room_frame(WIDTH, HEIGHT, &intrinsics());
        let pixels = WIDTH * HEIGHT;
        let rgb = vec![40u8; pixels * 3];
        let color_depth = vec![2.0f32; pixels];
        let valid = vec![0u8; pixels];

        scanner.add_frame_with_color(
            &room,
            WIDTH,
            HEIGHT,
            Some(frame_color(&rgb, &color_depth, &valid, 0.01)),
        );

        let colors = scanner.mesh().colors.expect("a colour view was kept");
        assert!(!colors.is_empty(), "the room produced no vertices");
        assert!(
            colors.iter().all(|color| *color == [9, 8, 7]),
            "mesh colour ignored ColoringParams on the config"
        );
    }

    #[test]
    fn colour_view_poses_stay_when_loop_closure_does_nothing() {
        // No revisit detector, so close_loops has nothing to apply and must not
        // rewrite the poses the views were captured with.
        let mut scanner = scanner();
        let room = room_frame(WIDTH, HEIGHT, &intrinsics());
        let pixels = WIDTH * HEIGHT;
        let rgb = vec![10u8; pixels * 3];
        let color_depth = vec![2.0f32; pixels];
        let valid = vec![1u8; pixels];

        scanner.add_frame_with_color(
            &room,
            WIDTH,
            HEIGHT,
            Some(frame_color(&rgb, &color_depth, &valid, 0.01)),
        );
        scanner.add_frame_with_color(
            &room,
            WIDTH,
            HEIGHT,
            Some(frame_color(&rgb, &color_depth, &valid, 0.01)),
        );

        let stale = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::new(5.0, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
        );
        scanner.color_views[0].pose = stale;

        let report = scanner.close_loops();
        assert!(!report.rebuilt);
        assert!(
            (scanner.color_views()[0].pose.translation.vector - stale.translation.vector).norm()
                < 1e-6
        );
    }

    #[test]
    fn export_without_colour_is_geometry_and_mirrors_after_extraction() {
        let mut scanner = scanner();
        let room = room_frame(WIDTH, HEIGHT, &intrinsics());
        scanner.add_frame(&room, WIDTH, HEIGHT);

        let unflipped = scanner.mesh();
        let exported = scanner
            .export_mesh(true, true)
            .expect("the room has a surface");
        let ExportMesh::Geometry(mirrored, coloring) = exported else {
            panic!("an atlas was built with no colour views");
        };
        assert!(coloring.is_none());
        assert!(mirrored.colors.is_none());
        assert!(
            unflipped
                .vertices
                .iter()
                .zip(&mirrored.vertices)
                .any(|(before, after)| (after.x + before.x).abs() < 1e-5 && before.x.abs() > 1e-3),
            "mirroring did not negate a vertex that had an x coordinate"
        );
    }

    #[test]
    fn the_atlas_is_built_before_the_mirror() {
        let mut scanner = Scanner::new(
            intrinsics(),
            ScannerConfig {
                coloring: ColoringParams {
                    // The extracted surface sits a voxel or so off the raw
                    // depth. Wide enough that the view really paints a face,
                    // which is what makes the UVs depend on the unflipped pose.
                    depth_tolerance: 0.5,
                    ..ColoringParams::default()
                },
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                ..ScannerConfig::default()
            },
        );
        let room = room_frame(WIDTH, HEIGHT, &intrinsics());
        let pixels = WIDTH * HEIGHT;
        let rgb = vec![40u8; pixels * 3];
        let valid = vec![1u8; pixels];
        scanner.add_frame_with_color(
            &room,
            WIDTH,
            HEIGHT,
            Some(frame_color(&rgb, &room, &valid, 0.01)),
        );

        let (raw, report) = scanner.textured_mesh().expect("atlas");
        assert!(report.triangles > 0);
        assert!(
            report.unobserved < report.triangles,
            "no face was observed, so the UV check would not catch a mirror that ran too early"
        );

        let exported = scanner.export_mesh(true, true).expect("export");
        let ExportMesh::Atlas(textured, _) = exported else {
            panic!("colour views did not produce an atlas");
        };

        let mut expected = raw.clone();
        expected.flip_x();
        assert_eq!(textured.uvs, expected.uvs);
        assert_eq!(textured.indices, expected.indices);
        assert_eq!(textured.atlas, expected.atlas);
        for (before, after) in expected.positions.iter().zip(&textured.positions) {
            assert!((after - before).norm() < 1e-5);
        }
        assert!(
            raw.positions
                .iter()
                .zip(&textured.positions)
                .any(|(before, after)| (after.x + before.x).abs() < 1e-4 && before.x.abs() > 1e-3),
            "export left the atlas in the unflipped frame"
        );
    }

    fn scanner_with_loop_flags(online: bool, incremental: bool, background: bool) -> Scanner {
        Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                loop_closure: Some(LoopClosureConfig {
                    background,
                    ..short_scan_loops()
                }),
                online_loops: online,
                incremental_rebuild: incremental,
                ..ScannerConfig::default()
            },
        )
    }

    fn scanner_with_submap_graph() -> Scanner {
        Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                loop_closure: Some(short_scan_loops()),
                submap_graph: true,
                ..ScannerConfig::default()
            },
        )
    }

    fn scanner_with_background_graph(submap_graph: bool) -> Scanner {
        Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                loop_closure: Some(short_scan_loops()),
                submap_graph,
                graph_worker: true,
                online_loops: true,
                ..ScannerConfig::default()
            },
        )
    }

    /// Front, side, and a return to the front 20 cm to the right. The return
    /// starts a submap at the failed alignment's guess, so only a loop edge can
    /// put it back where it belongs.
    fn add_revisit_sequence(scanner: &mut Scanner) {
        let front = room_frame(WIDTH, HEIGHT, &intrinsics());
        let side = room_frame_from(
            WIDTH,
            HEIGHT,
            &intrinsics(),
            &Vector3::zeros(),
            std::f32::consts::FRAC_PI_2,
        );
        let returned = room_frame_from(
            WIDTH,
            HEIGHT,
            &intrinsics(),
            &Vector3::new(0.2, 0.0, 0.0),
            0.0,
        );

        for _ in 0..2 {
            scanner.add_frame(&front, WIDTH, HEIGHT);
        }
        for _ in 0..2 {
            scanner.add_frame(&side, WIDTH, HEIGHT);
        }
        for _ in 0..3 {
            scanner.add_frame(&returned, WIDTH, HEIGHT);
        }
    }

    #[test]
    fn the_keyframe_graph_uses_keyframes_not_every_frame() {
        let mut scanner = scanner_with_loops();
        add_revisit_sequence(&mut scanner);

        let report = scanner.close_loops();
        assert!(report.rebuilt);
        assert!(
            report.nodes >= 2,
            "a graph needs at least two keyframes, got {}",
            report.nodes
        );
        assert!(
            report.nodes < scanner.frames(),
            "the graph used {} nodes for {} frames",
            report.nodes,
            scanner.frames()
        );
    }

    #[test]
    fn online_loops_apply_a_revisit_during_the_scan() {
        let mut scanner = scanner_with_loop_flags(true, false, false);
        add_revisit_sequence(&mut scanner);

        assert!(scanner.loops() > 0, "no revisit was verified");
        assert!(
            scanner.online_loop_updates() > 0,
            "the verified revisit was not applied during the scan"
        );
        assert!(
            scanner.last_online_correction() > 0.05,
            "the online correction was only {:.3} m",
            scanner.last_online_correction()
        );

        let closure = scanner.closure().copied().expect("an online report");
        assert!(closure.online, "the report lost its in-scan origin");
        assert!(
            closure.rebuilt,
            "the online correction did not move the model"
        );

        let position = scanner.pose().translation.vector;
        assert!(
            (position - Vector3::new(0.2, 0.0, 0.0)).norm() < 0.1,
            "the tracked return pose is {position:?} before the final pass"
        );

        let report = scanner.close_loops();
        assert!(report.rebuilt);
        assert!(!report.online, "the final pass reported itself as online");
        assert_eq!(scanner.submaps(), 1, "the final rebuild did not merge");
        assert_eq!(scanner.fused(), 7, "the final rebuild dropped frames");
        assert!(!scanner.mesh().is_empty());
    }

    #[test]
    fn incremental_rebuild_moves_cached_submaps_without_refusing() {
        let mut scanner = scanner_with_loop_flags(false, true, false);
        add_revisit_sequence(&mut scanner);

        assert!(
            scanner.submap_surface_bytes() > 0,
            "there are no cached submap surfaces to move"
        );

        let report = scanner.close_loops();
        assert!(report.rebuilt);
        assert!(!report.refused);
        assert_eq!(
            scanner.submaps(),
            1,
            "the pieces are not in one world frame"
        );
        assert_eq!(scanner.fused(), 7);
        assert!(
            scanner.submap_surface_bytes() > 0,
            "incremental correction discarded the cached pieces and re-fused"
        );
        assert!(!scanner.mesh().is_empty());
    }

    #[test]
    fn background_loops_are_flushed_by_close_loops() {
        let mut scanner = scanner_with_loop_flags(false, false, true);
        let frame = room_frame(WIDTH, HEIGHT, &intrinsics());

        for _ in 0..6 {
            scanner.add_frame(&frame, WIDTH, HEIGHT);
        }
        assert!(
            scanner.loop_pending() > 0 || scanner.loops() > 0,
            "no verification was ever submitted"
        );

        let report = scanner.close_loops();
        assert!(
            report.loop_edges > 0,
            "close_loops did not wait for the worker's loop"
        );
        assert!(report.rebuilt);
        assert_eq!(scanner.loop_pending(), 0);
    }

    #[test]
    fn corrections_are_interpolated_within_a_submap_and_not_across_one() {
        let at = |x: f32| {
            Isometry3::from_parts(
                Translation3::new(x, 0.0, 0.0),
                nalgebra::UnitQuaternion::identity(),
            )
        };
        let poses = vec![at(0.0), at(0.1), at(0.2), at(0.3), at(0.4)];
        let keyframes = vec![(0, at(0.0)), (2, at(0.2)), (4, at(0.4))];

        // No boundary: the corrections blend across the keyframes. The
        // corrections are +0.2 at frame 2 and +0.4 at frame 4, so frame 1
        // moves by +0.1 on top of its own 0.1 m pose, and frame 3 by +0.3 on
        // top of 0.3 m.
        let corrected = project_keyframe_corrections(
            &poses,
            &[BoundaryKind::None; 5],
            &keyframes,
            &[at(0.0), at(0.4), at(0.8)],
        );
        assert!((corrected[1].translation.x - 0.2).abs() < 1e-5);
        assert!((corrected[3].translation.x - 0.6).abs() < 1e-5);

        // Boundaries at frames 2 and 4: frame 3 belongs to the second submap,
        // whose only keyframe is frame 2 and did not move. It must not pick up
        // half of the third submap's correction.
        let corrected = project_keyframe_corrections(
            &poses,
            &[
                BoundaryKind::None,
                BoundaryKind::None,
                BoundaryKind::Budget,
                BoundaryKind::None,
                BoundaryKind::Budget,
            ],
            &keyframes,
            &[at(0.0), at(0.2), at(0.6)],
        );
        assert!(
            (corrected[1].translation.x - 0.1).abs() < 1e-5,
            "frame 1 moved inside the first submap: {}",
            corrected[1].translation.x
        );
        assert!(
            (corrected[2].translation.x - 0.2).abs() < 1e-5,
            "the second submap's keyframe moved: {}",
            corrected[2].translation.x
        );
        assert!(
            (corrected[3].translation.x - 0.3).abs() < 1e-5,
            "frame 3 was sheared across the boundary: {}",
            corrected[3].translation.x
        );
        assert!((corrected[4].translation.x - 0.6).abs() < 1e-5);
    }

    #[test]
    fn the_fusion_motion_filter_skips_near_duplicate_frames() {
        let mut scanner = Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                fusion_filter: Some(MotionFilterConfig {
                    max_distance: 0.02,
                    max_angle: 0.02,
                    max_interval_frames: 100,
                }),
                ..ScannerConfig::default()
            },
        );
        let frame = room_frame(WIDTH, HEIGHT, &intrinsics());

        for _ in 0..6 {
            scanner.add_frame(&frame, WIDTH, HEIGHT);
        }

        assert_eq!(scanner.fused(), 1, "only the first frame should fuse");
        assert_eq!(scanner.fusion_skipped(), 5);
        assert_eq!(scanner.rejected(), 0, "filtered frames are not rejections");
        assert!(!scanner.mesh().is_empty());
    }

    #[test]
    fn tsdf_chunks_bound_the_active_volume_without_a_pose_boundary() {
        let mut scanner = Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                tsdf_chunk_frames: 2,
                ..ScannerConfig::default()
            },
        );
        let frame = room_frame(WIDTH, HEIGHT, &intrinsics());

        for _ in 0..6 {
            scanner.add_frame(&frame, WIDTH, HEIGHT);
        }

        assert_eq!(scanner.fused(), 6);
        assert_eq!(scanner.submaps(), 1, "chunks must not become pose submaps");
        assert_eq!(scanner.chunks(), 3, "six frames at two per chunk");
        assert!(scanner.submap_surface_bytes() > 0);
        assert!(!scanner.mesh().is_empty(), "the cached chunks are missing");
    }

    #[test]
    fn the_submap_graph_uses_submap_nodes_and_moves_them_rigidly() {
        let mut scanner = scanner_with_submap_graph();
        add_revisit_sequence(&mut scanner);
        assert_eq!(scanner.submaps(), 3);

        let report = scanner.close_loops();
        assert!(
            report.rebuilt,
            "the submap graph did not apply a correction"
        );
        assert_eq!(report.keyframe_nodes, 0);
        assert_eq!(report.submap_nodes, 3);
        assert_eq!(report.nodes, 3);
        assert_eq!(report.odometry_edges, 2);

        let last = scanner.pose();
        assert!(
            (last.translation.x - 0.2).abs() < 0.08,
            "the final submap was not moved back to the revisit: {:.3}",
            last.translation.x
        );
    }

    #[test]
    fn the_background_submap_graph_worker_applies_an_online_correction() {
        let mut scanner = scanner_with_background_graph(true);
        add_revisit_sequence(&mut scanner);

        // The last offer may still be in flight; close_loops has to flush it.
        let report = scanner.close_loops();
        assert!(report.rebuilt);
        assert!(!scanner.graph_pending());
        assert!(
            scanner.background_graph_updates() > 0,
            "the background solve never applied a correction"
        );
        assert!(scanner.online_loop_updates() > 0);

        let last = scanner.pose();
        assert!(
            (last.translation.x - 0.2).abs() < 0.08,
            "the worker's correction missed the revisit: {:.3}",
            last.translation.x
        );
    }

    #[test]
    fn the_background_keyframe_graph_worker_applies_an_online_correction() {
        let mut scanner = scanner_with_background_graph(false);
        add_revisit_sequence(&mut scanner);

        let report = scanner.close_loops();
        assert!(report.rebuilt);
        assert!(!scanner.graph_pending());
        assert!(
            report.keyframe_nodes >= 2,
            "the worker should report keyframe nodes, got {}",
            report.keyframe_nodes
        );
        assert!(
            scanner.background_graph_updates() > 0,
            "the background keyframe solve never applied a correction"
        );
        assert!(scanner.online_loop_updates() > 0);

        let last = scanner.pose();
        assert!(
            (last.translation.x - 0.2).abs() < 0.08,
            "the keyframe worker's correction missed the revisit: {:.3}",
            last.translation.x
        );
    }

    #[test]
    fn a_submap_budget_finishes_submaps_without_losing_tracking() {
        let mut scanner = Scanner::new(
            intrinsics(),
            ScannerConfig {
                tsdf: TsdfParams {
                    voxel_size: 0.04,
                    truncation: 0.16,
                    ..TsdfParams::default()
                },
                submap_max_frames: 3,
                ..ScannerConfig::default()
            },
        );

        let mut submaps = Vec::new();
        for step in 0..8 {
            let eye = Vector3::new(0.05 * step as f32, 0.0, 0.0);
            let frame = room_frame_from(WIDTH, HEIGHT, &intrinsics(), &eye, 0.0);
            let report = scanner.add_frame(&frame, WIDTH, HEIGHT);
            assert!(
                report.integration.is_some(),
                "frame {step} was not fused: {:?}",
                report.track.rejection
            );
            submaps.push(report.submap);
        }

        assert_eq!(scanner.fused(), 8);
        assert_eq!(scanner.rejected(), 0);
        assert_eq!(submaps, vec![0, 0, 0, 1, 1, 1, 2, 2]);
        assert_eq!(scanner.submaps(), 3);
        assert!(
            (scanner.pose().translation.x - 0.35).abs() < 0.03,
            "the budget boundary broke continuity: {:.3}",
            scanner.pose().translation.x
        );
    }
}
