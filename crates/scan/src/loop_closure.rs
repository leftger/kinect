//! Loop closure: noticing that the sensor has returned somewhere it has been.
//!
//! Odometry edges alone leave drift unbounded. This module supplies the other
//! kind of edge: a relative-pose measurement between two frames that are *far
//! apart in time but close in space*, which is what tells the pose graph that the
//! trajectory is a loop rather than an open curve.
//!
//! # How a candidate is found, and why it is verified
//!
//! Candidates come from the estimated poses: anything within `search_radius`, and
//! at least `min_index_gap` frames back so that temporal neighbours -- which
//! always align well and say nothing new -- are ignored. Pose proximity is only a
//! *filter*, though, because the poses are exactly what has drifted; it narrows
//! thousands of pairs down to a handful worth testing.
//!
//! Colour scans have a second, appearance-based path inspired by RTAB-Map's
//! bag-of-words loop detection: if the pose-proximity pass verifies nothing, a
//! compact colour signature can propose an earlier frame *outside* the pose
//! radius. That is only a proposal. The same spatial ICP has to accept it before
//! it becomes an edge, so a repeated-looking wall cannot fold the map by itself.
//!
//! Each surviving candidate is then aligned geometrically with point-to-plane ICP
//! (`geom::icp::align`, the spatial-index path, since two keyframes that are far
//! apart share no viewpoint and projective association does not apply). Only an
//! alignment that meets both an inlier-ratio and a residual threshold is accepted.
//!
//! That verification is not optional. A pose graph will happily fold a map in half
//! to satisfy a single bad edge, so a false loop is far worse than a missed one.

use geom::global_search::{self, GlobalSearchParams, ScoreGrid};
use geom::icp::{IcpCloud, IcpParams, align};
use geom::pose_graph::Edge;
use geom::transform_point;
use nalgebra::{Isometry3, Vector3};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// A compact appearance signature for one colour frame.
///
/// This is a deliberately small stand-in for RTAB-Map's visual bag of words:
/// an 8×8 grayscale image and a 4×4×4 RGB histogram, each normalised and
/// concatenated. It is cheap enough to compute on every frame and good enough
/// to *propose* a loop candidate; the spatial ICP below still has to verify it.
/// Without that second stage an appearance match could fold two similar-looking
/// walls together.
#[derive(Clone, Debug)]
pub struct VisualDescriptor {
    values: Vec<f32>,
}

impl VisualDescriptor {
    pub fn from_rgb(rgb: &[u8], width: usize, height: usize) -> Option<Self> {
        if width == 0 || height == 0 || rgb.len() < width * height * 3 {
            return None;
        }

        let mut gray = [0.0f32; 64];
        for gy in 0..8 {
            for gx in 0..8 {
                let x0 = gx * width / 8;
                let x1 = ((gx + 1) * width / 8).max(x0 + 1).min(width);
                let y0 = gy * height / 8;
                let y1 = ((gy + 1) * height / 8).max(y0 + 1).min(height);
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let index = (y * width + x) * 3;
                        sum += 0.299 * rgb[index] as f32
                            + 0.587 * rgb[index + 1] as f32
                            + 0.114 * rgb[index + 2] as f32;
                        count += 1;
                    }
                }
                gray[gy * 8 + gx] = sum / count.max(1) as f32;
            }
        }

        let mean = gray.iter().sum::<f32>() / 64.0;
        let variance = gray
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / 64.0;
        let deviation = variance.sqrt();
        let mut values = Vec::with_capacity(128);
        if deviation < 1e-4 {
            values.extend(std::iter::repeat_n(0.0, 64));
        } else {
            values.extend(gray.iter().map(|value| (value - mean) / deviation / 8.0));
        }

        let mut color = [0.0f32; 64];
        let pixels = width * height;
        for index in (0..pixels * 3).step_by(3) {
            let r = rgb[index] as f32 / 255.0;
            let g = rgb[index + 1] as f32 / 255.0;
            let b = rgb[index + 2] as f32 / 255.0;
            let ri = (r * 4.0).floor().min(3.0) as usize;
            let gi = (g * 4.0).floor().min(3.0) as usize;
            let bi = (b * 4.0).floor().min(3.0) as usize;
            color[(ri * 4 + gi) * 4 + bi] += 1.0;
        }
        let color_norm = color.iter().map(|value| value * value).sum::<f32>().sqrt();
        values.extend(color.iter().map(|value| value / color_norm.max(1e-6)));

        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        let norm = norm.max(1e-6);
        for value in &mut values {
            *value /= norm;
        }
        Some(Self { values })
    }

    /// Cosine similarity: 1 for the same view, lower for a different one.
    pub fn similarity(&self, other: &Self) -> f32 {
        if self.values.len() != other.values.len() {
            return 0.0;
        }
        self.values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| a * b)
            .sum::<f32>()
            .clamp(-1.0, 1.0)
    }
}

/// A frame kept as a place the sensor might return to.
pub struct Keyframe {
    /// Index of this frame within the scan, which is what edges refer to.
    pub index: usize,
    pub pose: Isometry3<f32>,
    /// Appearance signature, when the scan had colour.
    appearance: Option<VisualDescriptor>,
    cloud: IcpCloud,
    /// True once the cloud is the cheaper long-term copy.
    decimated: bool,
}

impl Keyframe {
    pub fn point_count(&self) -> usize {
        self.cloud.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoopClosureConfig {
    /// Candidate poses must be at least this close, in metres. Not tight: drift is
    /// what makes the estimate wrong in the first place.
    pub search_radius: f32,
    /// Never match against a frame this recent.
    pub min_index_gap: usize,
    /// Build the next loop keyframe only after this much translation, in metres,
    /// or `keyframe_min_rotation`. RTAB-Map's linear/angular update thresholds.
    pub keyframe_min_translation: f32,
    pub keyframe_min_rotation: f32,
    /// Force a keyframe after this many frames even if the camera barely moved.
    /// Submap boundaries reset the pose, so motion alone is not enough.
    pub keyframe_max_interval: usize,
    /// How many candidates to spend ICP on per frame.
    pub max_candidates: usize,
    /// Recent keyframes whose clouds stay at full resolution. Older keyframes
    /// move to long-term memory, still searchable but decimated.
    pub working_memory_keyframes: usize,
    /// Keep every Nth point when a keyframe leaves working memory.
    pub long_term_decimation: usize,
    /// Appearance similarity that can propose a candidate whose pose is outside
    /// `search_radius`. Purely a proposal: ICP still has to accept it.
    pub appearance_min_similarity: f32,
    /// ICP acceptance: fraction of points that must find a match...
    pub min_inlier_ratio: f32,
    /// ...and the RMS residual they must achieve, in metres.
    pub max_rmse: f32,
    /// A verified loop must also align in reverse. These bound the disagreement
    /// between `T_ab` and the inverse of the reverse alignment.
    pub reciprocal_max_translation: f32,
    pub reciprocal_max_rotation: f32,
    /// Confidence given to a loop edge, relative to an odometry edge.
    pub edge_weight: f32,
    /// Resolution to keep keyframes at. Coarse keeps the memory per keyframe small
    /// and gives wide-baseline ICP a usable basin of convergence.
    pub voxel: f32,
    pub normal_radius: f32,
    pub max_distance: f32,
    pub max_iterations: usize,
    /// When ICP from the drifted initial guess fails, run the 4-DOF
    /// branch-and-bound search before giving up on the candidate. This is the
    /// global half of loop closure: a wider basin, still followed
    /// by the same reciprocal verification.
    pub global_search: bool,
    /// Translation half-width for that search, in metres.
    pub global_search_radius: f32,
    /// Yaw half-width for that search, in radians.
    pub global_search_yaw: f32,
    /// Minimum mean likelihood for the global search to be trusted.
    pub global_search_min_score: f32,
    /// Minimum mean likelihood of the final alignment against the target's
    /// surface grid. Zero disables the gate; the ICP inlier and reciprocal
    /// checks remain the primary defences.
    pub min_match_score: f32,
    /// How many keyframes on each side of a loop candidate to merge into the
    /// ICP target. LIO-SAM builds its loop target from a window of history
    /// keyframes rather than a single scan; the wider surface makes a narrow
    /// single view much less likely to miss a real revisit. Zero keeps the
    /// original single-keyframe target.
    pub loop_target_keyframes: usize,
    /// Run loop verification on a worker thread instead of on the capture
    /// thread. Keyframe indexing and candidate selection stay synchronous;
    /// only the ICP alignment moves off-thread. The trade is one extra cloud
    /// clone per keyframe, and results arrive a frame or two later.
    pub background: bool,
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            search_radius: 1.5,
            min_index_gap: 40,
            keyframe_min_translation: 0.02,
            keyframe_min_rotation: 0.05,
            keyframe_max_interval: 5,
            max_candidates: 3,
            working_memory_keyframes: 80,
            long_term_decimation: 2,
            appearance_min_similarity: 0.78,
            min_inlier_ratio: 0.5,
            max_rmse: 0.03,
            reciprocal_max_translation: 0.05,
            reciprocal_max_rotation: 0.10,
            // Below odometry on purpose: a loop is a hypothesis, an odometry edge
            // is a measurement of something we actually saw move.
            edge_weight: 0.5,
            voxel: 0.04,
            normal_radius: 0.12,
            max_distance: 0.25,
            max_iterations: 20,
            global_search: false,
            global_search_radius: 0.55,
            global_search_yaw: 0.55,
            global_search_min_score: 0.30,
            min_match_score: 0.0,
            // A three-keyframe window each way: LIO-SAM uses 25, but a depth
            // camera already sees a dense surface and indoor rooms are small,
            // so a handful is enough to widen the basin without burying the
            // candidate in redundant geometry.
            loop_target_keyframes: 2,
            background: false,
        }
    }
}

/// An accepted loop closure.
#[derive(Clone, Copy, Debug)]
pub struct Loop {
    /// Earlier frame.
    pub from: usize,
    /// Later frame.
    pub to: usize,
    /// `T_from^-1 T_to`, as recovered by ICP rather than by the drifted poses.
    pub measurement: Isometry3<f32>,
    pub inlier_ratio: f32,
    pub rmse: f32,
}

impl Loop {
    /// The constraint, ready for a pose graph.
    ///
    /// The base loop weight is scaled by both correspondence coverage and how
    /// close the residual is to the acceptance limit. A loop that barely passed
    /// still gets a voice, but a much quieter one than a tight fit.
    pub fn edge(&self, base_weight: f32, max_rmse: f32) -> Edge {
        let residual_quality = (max_rmse / self.rmse.max(1e-6)).clamp(0.0, 1.0);
        let quality = (0.5 * self.inlier_ratio + 0.5 * residual_quality).clamp(0.1, 1.0);
        Edge {
            from: self.from,
            to: self.to,
            measurement: self.measurement,
            weight: base_weight * quality,
        }
    }
}

/// One candidate prepared for verification.
///
/// The target is a small local submap in the candidate keyframe's frame, not
/// just one keyframe's cloud. See [`LoopClosureConfig::loop_target_keyframes`].
struct LoopCandidate {
    from: usize,
    target: IcpCloud,
    initial: Isometry3<f32>,
    /// Pose-proximity candidate, as opposed to an appearance-only proposal.
    spatial: bool,
}

/// Verify prepared candidates against `source`, returning the best accepted
/// loop and how many alignments were spent.
///
/// Pure on purpose: the capture thread and the background worker run exactly
/// the same verification, so turning the worker on cannot quietly change which
/// loops are accepted.
fn verify_candidates(
    source: &IcpCloud,
    candidates: &[LoopCandidate],
    config: &LoopClosureConfig,
    index: usize,
    debug: bool,
) -> (Option<Loop>, usize) {
    let params = IcpParams {
        max_correspondence_distance: config.max_distance,
        max_iterations: config.max_iterations,
        ..IcpParams::default()
    };

    let mut best: Option<Loop> = None;
    let mut attempts = 0;

    for candidate in candidates {
        // A verified spatial loop is enough; only fall back to appearance when
        // the entire pose-proximity pass produced nothing.
        if !candidate.spatial && best.is_some() {
            break;
        }

        attempts += 1;
        let mut result = align(source, &candidate.target, candidate.initial, &params);

        // If the drifted prior is too far out for ICP, give the 4-DOF global
        // search one chance before rejecting the candidate. The verification
        // below is unchanged, so a global match still has to survive reverse
        // alignment and the likelihood gate.
        if (result.inlier_ratio < config.min_inlier_ratio || result.rmse > config.max_rmse)
            && config.global_search
        {
            let global = GlobalSearchParams {
                voxel: config.voxel,
                sigma: config.normal_radius,
                translation_radius: config.global_search_radius,
                yaw_radius: config.global_search_yaw,
                min_score: config.global_search_min_score,
                ..GlobalSearchParams::default()
            };
            if let Some(found) = global_search::search(
                &candidate.target.points,
                &source.points,
                candidate.initial,
                &global,
            ) {
                result = align(source, &candidate.target, found.transform, &params);
                if debug {
                    eprintln!(
                        "[loop] global search rescued kf {} -> {index} at score {:.3}",
                        candidate.from, found.score
                    );
                }
            }
        }

        if debug {
            eprintln!(
                "[loop] align kf {} -> {index}: inliers {:.3} rmse {:.4} corr {}",
                candidate.from, result.inlier_ratio, result.rmse, result.correspondences
            );
        }

        if result.inlier_ratio < config.min_inlier_ratio || result.rmse > config.max_rmse {
            continue;
        }

        // A real place match survives being aligned in the other direction.
        // Reverse ICP starts from the inverse of the forward result, so a
        // one-sided surface fit shows up as a large round-trip error.
        let reverse = align(
            &candidate.target,
            source,
            result.transform.inverse(),
            &params,
        );
        let round_trip = reverse.transform * result.transform;
        // Reverse RMSE is not a gate: a slightly noisier reverse fit with a
        // tiny round-trip error is exactly what a correct loop looks like.
        if reverse.inlier_ratio < config.min_inlier_ratio
            || round_trip.translation.vector.norm() > config.reciprocal_max_translation
            || round_trip.rotation.angle() > config.reciprocal_max_rotation
        {
            if debug {
                eprintln!(
                    "[loop] reject kf {} -> {index}: reverse inliers {:.3} rmse {:.4}, round-trip {:.3} m {:.3} rad",
                    candidate.from,
                    reverse.inlier_ratio,
                    reverse.rmse,
                    round_trip.translation.vector.norm(),
                    round_trip.rotation.angle()
                );
            }
            continue;
        }

        // Likelihood gate. ICP inliers say "points found neighbours"; this says
        // "they landed near the observed surface", which is a different signal
        // and catches repetitive geometry that happens to fit.
        if config.min_match_score > 0.0 {
            let score_params = GlobalSearchParams {
                voxel: config.voxel,
                sigma: config.normal_radius,
                translation_radius: 0.0,
                yaw_radius: 0.0,
                levels: 1,
                max_points: source.points.len().max(1),
                max_nodes: 1,
                min_score: 0.0,
                ..GlobalSearchParams::default()
            };
            if let Some(grid) = ScoreGrid::build(&candidate.target.points, &score_params) {
                let score = grid.score_transform(&source.points, &result.transform);
                if debug {
                    eprintln!(
                        "[loop] kf {} -> {index} match score {score:.3}",
                        candidate.from
                    );
                }
                if score < config.min_match_score {
                    continue;
                }
            }
        }

        let loop_closure = Loop {
            from: candidate.from,
            to: index,
            measurement: result.transform,
            inlier_ratio: result.inlier_ratio,
            rmse: result.rmse,
        };

        // Keep the best of the verified candidates rather than the first.
        let better = match &best {
            None => true,
            Some(previous) => loop_closure.inlier_ratio > previous.inlier_ratio,
        };
        if better {
            best = Some(loop_closure);
        }
    }

    (best, attempts)
}

/// A verification handed to the background worker.
struct LoopJob {
    source: IcpCloud,
    candidates: Vec<LoopCandidate>,
    index: usize,
    config: LoopClosureConfig,
}

/// The worker's answer. `attempts` is reported per job so the capture thread
/// can keep its telemetry honest without sharing the candidate list.
struct LoopOutcome {
    found: Option<Loop>,
    attempts: usize,
}

/// Optional ICP worker. Owned by [`LoopFinder`]; dropping it closes the job
/// channel and joins the thread.
struct LoopWorker {
    sender: Option<Sender<LoopJob>>,
    receiver: Receiver<LoopOutcome>,
    handle: Option<JoinHandle<()>>,
}

impl LoopWorker {
    fn spawn() -> Self {
        let (job_sender, job_receiver) = mpsc::channel::<LoopJob>();
        let (outcome_sender, outcome_receiver) = mpsc::channel::<LoopOutcome>();
        let handle = thread::Builder::new()
            .name("scan-loop-icp".to_string())
            .spawn(move || {
                let debug = std::env::var_os("SCAN_DEBUG_LOOPS").is_some();
                while let Ok(job) = job_receiver.recv() {
                    let (found, attempts) = verify_candidates(
                        &job.source,
                        &job.candidates,
                        &job.config,
                        job.index,
                        debug,
                    );
                    if outcome_sender
                        .send(LoopOutcome { found, attempts })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("spawn loop verification worker");

        Self {
            sender: Some(job_sender),
            receiver: outcome_receiver,
            handle: Some(handle),
        }
    }

    fn submit(&self, job: LoopJob) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(job);
        }
    }

    fn try_recv(&self) -> Option<LoopOutcome> {
        match self.receiver.try_recv() {
            Ok(outcome) => Some(outcome),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
        }
    }

    fn recv_timeout(&self, timeout: Duration) -> Option<LoopOutcome> {
        self.receiver.recv_timeout(timeout).ok()
    }
}

impl Drop for LoopWorker {
    fn drop(&mut self) {
        // Close the job channel first, then wait for the thread to finish the
        // job it already has. A scan that ends with a loop in flight still gets
        // the chance to use it; `flush` is the normal path, this is the backstop.
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Collects keyframes and the loop closures between them.
pub struct LoopFinder {
    config: LoopClosureConfig,
    keyframes: Vec<Keyframe>,
    loops: Vec<Loop>,
    attempts: usize,
    rehydrations: usize,
    last_keyframe: Option<Isometry3<f32>>,
    /// Index of the first keyframe still in full-resolution working memory.
    working_start: usize,
    /// Verification worker, present only when `config.background` is set.
    worker: Option<LoopWorker>,
    /// Jobs submitted to the worker but not answered yet.
    pending: usize,
}

impl LoopFinder {
    pub fn new(config: LoopClosureConfig) -> Self {
        let worker = config.background.then(LoopWorker::spawn);
        Self {
            config,
            keyframes: Vec::new(),
            loops: Vec::new(),
            attempts: 0,
            rehydrations: 0,
            last_keyframe: None,
            working_start: 0,
            worker,
            pending: 0,
        }
    }

    /// True when verification is running on a worker thread.
    pub fn is_background(&self) -> bool {
        self.worker.is_some()
    }

    /// Jobs submitted to the worker that have not been answered yet.
    pub fn pending(&self) -> usize {
        self.pending
    }

    /// Take any completed background results. Synchronous mode always returns
    /// an empty vector, because loops are already in the index.
    pub fn poll(&mut self) -> Vec<Loop> {
        let mut outcomes = Vec::new();
        if let Some(worker) = &self.worker {
            while let Some(outcome) = worker.try_recv() {
                outcomes.push(outcome);
            }
        }

        let mut found = Vec::new();
        for outcome in outcomes {
            if let Some(loop_closure) = self.absorb(outcome) {
                found.push(loop_closure);
            }
        }
        found
    }

    /// Wait for every in-flight verification so a final optimisation sees all
    /// the loops the scan produced. Returns the loops that arrived meanwhile.
    ///
    /// The timeout is a backstop against a wedged worker, not a normal path:
    /// one ICP alignment is bounded by `max_iterations`, and 30 s is far above
    /// anything a real keyframe pair costs. If it does expire, the scan keeps
    /// the loops it already has rather than hanging.
    pub fn flush(&mut self) -> Vec<Loop> {
        let mut found = self.poll();
        let mut outcomes = Vec::new();
        if let Some(worker) = &self.worker {
            while self.pending > 0 {
                match worker.recv_timeout(Duration::from_secs(30)) {
                    Some(outcome) => outcomes.push(outcome),
                    None => break,
                }
            }
        }
        for outcome in outcomes {
            if let Some(loop_closure) = self.absorb(outcome) {
                found.push(loop_closure);
            }
        }
        found
    }

    fn absorb(&mut self, outcome: LoopOutcome) -> Option<Loop> {
        self.pending = self.pending.saturating_sub(1);
        self.attempts += outcome.attempts;
        if let Some(loop_closure) = outcome.found {
            self.loops.push(loop_closure);
            Some(loop_closure)
        } else {
            None
        }
    }

    /// Point every keyframe at its corrected world pose.
    ///
    /// `poses` is indexed by arrival frame, exactly like `Scanner::poses`. The
    /// clouds are camera-frame data and do not move; only the poses used for
    /// candidate search and relative measurements need updating. Without this,
    /// an online correction would leave the loop index proposing candidates
    /// from the drifted trajectory it just corrected.
    pub fn update_poses(&mut self, poses: &[Isometry3<f32>]) {
        for keyframe in &mut self.keyframes {
            if let Some(pose) = poses.get(keyframe.index) {
                keyframe.pose = *pose;
            }
        }
        self.last_keyframe = self.keyframes.last().map(|keyframe| keyframe.pose);
    }

    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    /// The loop configuration, copied for callers that need to rebuild a cloud.
    pub fn config(&self) -> LoopClosureConfig {
        self.config
    }

    /// Points still held by the loop index, after long-term decimation.
    pub fn keyframe_points(&self) -> usize {
        self.keyframes
            .iter()
            .map(|keyframe| keyframe.cloud.len())
            .sum()
    }

    /// Move keyframes that have fallen out of working memory into the cheaper
    /// long-term representation. Their pose and appearance signature stay, so
    /// they can still be proposed and verified.
    fn manage_memory(&mut self) {
        let budget = self.config.working_memory_keyframes.max(1);
        while self.keyframes.len().saturating_sub(self.working_start) > budget {
            let index = self.working_start;
            let stride = self.config.long_term_decimation.max(1);
            self.keyframes[index].cloud = self.keyframes[index].cloud.decimated(stride);
            self.keyframes[index].decimated = true;
            self.working_start += 1;
        }
    }

    pub fn loops(&self) -> &[Loop] {
        &self.loops
    }

    /// How many ICP alignments were spent, accepted or not. A high number with no
    /// loops means the candidate filter is not narrowing enough.
    pub fn attempts(&self) -> usize {
        self.attempts
    }

    /// How many long-term clouds were rebuilt for verification.
    pub fn rehydrations(&self) -> usize {
        self.rehydrations
    }

    /// Offer a frame. `points` are in this frame's camera space, and `pose` is the
    /// odometry estimate of where the camera was.
    ///
    /// The frame is always retained as a future candidate; a loop is returned only
    /// if one was found and verified.
    pub fn consider(
        &mut self,
        index: usize,
        pose: Isometry3<f32>,
        points: &[Vector3<f32>],
    ) -> Option<Loop> {
        self.consider_with_appearance(index, pose, points, None)
    }

    /// As [`Self::consider`], with an optional appearance signature.
    ///
    /// The signature can retrieve a candidate whose pose has drifted outside
    /// `search_radius`; ICP still decides whether it is a real loop. Scans
    /// without colour simply pass `None` and keep the pose-only path.
    pub fn consider_with_appearance(
        &mut self,
        index: usize,
        pose: Isometry3<f32>,
        points: &[Vector3<f32>],
        appearance: Option<&VisualDescriptor>,
    ) -> Option<Loop> {
        self.consider_inner(index, pose, points, appearance, None)
    }

    /// Offer a frame and rebuild a decimated long-term cloud on demand.
    ///
    /// The callback receives the frame index of a candidate that has left
    /// working memory and returns a full-resolution cloud for that index. It is
    /// only called while that candidate is being verified, so the scanner keeps
    /// one temporary full cloud rather than every old frame's full cloud.
    pub fn consider_with_rehydration<F>(
        &mut self,
        index: usize,
        pose: Isometry3<f32>,
        points: &[Vector3<f32>],
        appearance: Option<&VisualDescriptor>,
        mut rehydrate: F,
    ) -> Option<Loop>
    where
        F: FnMut(usize) -> Option<IcpCloud>,
    {
        self.consider_inner(index, pose, points, appearance, Some(&mut rehydrate))
    }

    fn consider_inner(
        &mut self,
        index: usize,
        pose: Isometry3<f32>,
        points: &[Vector3<f32>],
        appearance: Option<&VisualDescriptor>,
        rehydrate: Option<&mut dyn FnMut(usize) -> Option<IcpCloud>>,
    ) -> Option<Loop> {
        let due = match self.keyframes.last() {
            None => true,
            Some(last) => {
                index.saturating_sub(last.index) >= self.config.keyframe_max_interval.max(1)
            }
        };
        if !due {
            if let Some(last) = self.last_keyframe {
                let motion = last.inverse() * pose;
                if motion.translation.vector.norm() < self.config.keyframe_min_translation
                    && motion.rotation.angle() < self.config.keyframe_min_rotation
                {
                    return None;
                }
            }
        }

        let origin = Vector3::zeros();
        let cloud = IcpCloud::build(
            points,
            self.config.voxel,
            self.config.normal_radius,
            self.config.max_distance,
            &origin,
        );

        let candidates = self.prepare_candidates(index, &pose, appearance, rehydrate);

        // The frame becomes a keyframe even when it produced no candidates, so
        // later frames can match against it.
        if self.worker.is_some() {
            self.keyframes.push(Keyframe {
                index,
                pose,
                appearance: appearance.cloned(),
                cloud: cloud.clone(),
                decimated: false,
            });
            self.manage_memory();
            self.last_keyframe = Some(pose);

            if candidates.is_empty() {
                return None;
            }

            self.pending += 1;
            self.worker
                .as_ref()
                .expect("worker was checked above")
                .submit(LoopJob {
                    source: cloud,
                    candidates,
                    index,
                    config: self.config,
                });
            return None;
        }

        let debug = std::env::var_os("SCAN_DEBUG_LOOPS").is_some();
        let (found, attempts) = verify_candidates(&cloud, &candidates, &self.config, index, debug);
        self.attempts += attempts;

        self.keyframes.push(Keyframe {
            index,
            pose,
            appearance: appearance.cloned(),
            cloud,
            decimated: false,
        });
        self.manage_memory();
        self.last_keyframe = Some(pose);

        if let Some(loop_closure) = found {
            self.loops.push(loop_closure);
        }

        found
    }

    /// Select candidates and build each one's local target submap.
    ///
    /// The keyframe window follows LIO-SAM's `loopFindNearKeyframes`: the
    /// candidate plus `loop_target_keyframes` on either side, all transformed
    /// into the candidate's frame and merged into one cloud. A single depth
    /// view is narrow, so a nearby keyframe often supplies the surface that
    /// makes a real revisit verifiable.
    fn prepare_candidates(
        &mut self,
        index: usize,
        pose: &Isometry3<f32>,
        appearance: Option<&VisualDescriptor>,
        mut rehydrate: Option<&mut dyn FnMut(usize) -> Option<IcpCloud>>,
    ) -> Vec<LoopCandidate> {
        let debug = std::env::var_os("SCAN_DEBUG_LOOPS").is_some();

        // Spatial candidates first: they are the normal path and keep colourless
        // scans, and scans that already track well, behaving exactly as before.
        let mut spatial: Vec<(f32, usize)> = self
            .keyframes
            .iter()
            .enumerate()
            .filter(|(_, keyframe)| {
                index.saturating_sub(keyframe.index) >= self.config.min_index_gap
            })
            .filter_map(|(position, keyframe)| {
                let distance = (keyframe.pose.translation.vector - pose.translation.vector).norm();
                (distance <= self.config.search_radius).then_some((distance, position))
            })
            .collect();
        spatial.sort_by(|a, b| a.0.total_cmp(&b.0));
        spatial.truncate(self.config.max_candidates);
        if debug {
            for (distance, position) in &spatial {
                eprintln!(
                    "[loop] {index} spatial candidate kf {} at {distance:.2} m",
                    self.keyframes[*position].index
                );
            }
        }

        // Appearance is the fallback: only candidates *outside* the pose radius
        // are retrieved. This is the RTAB-Map "appearance proposes, geometry
        // verifies" split, without letting a generic view reorder the normal
        // spatial candidates.
        let mut appearance_only: Vec<(f32, usize)> = self
            .keyframes
            .iter()
            .enumerate()
            .filter(|(_, keyframe)| {
                index.saturating_sub(keyframe.index) >= self.config.min_index_gap
            })
            .filter_map(|(position, keyframe)| {
                let current = appearance?;
                let previous = keyframe.appearance.as_ref()?;
                let distance =
                    (keyframe.pose.translation.vector - pose.translation.vector).norm();
                if distance <= self.config.search_radius {
                    return None;
                }
                let similarity = current.similarity(previous);
                if similarity < self.config.appearance_min_similarity {
                    return None;
                }
                if debug {
                    eprintln!(
                        "[loop] {index} appearance candidate kf {} at {distance:.2} m, similarity {similarity:.3}",
                        keyframe.index
                    );
                }
                Some((similarity, position))
            })
            .collect();
        appearance_only.sort_by(|a, b| b.0.total_cmp(&a.0));
        appearance_only.truncate(self.config.max_candidates);

        let candidates: Vec<(bool, usize)> = spatial
            .into_iter()
            .map(|(_, position)| (true, position))
            .chain(
                appearance_only
                    .into_iter()
                    .map(|(_, position)| (false, position)),
            )
            .collect();

        let mut prepared = Vec::with_capacity(candidates.len());
        let window = self.config.loop_target_keyframes;

        for (spatial, position) in candidates {
            let keyframe_index = self.keyframes[position].index;
            let keyframe_pose = self.keyframes[position].pose;
            let decimated = self.keyframes[position].decimated;
            let initial = keyframe_pose.inverse() * pose;

            // Rehydrate only the primary candidate. Neighbours are used at
            // whatever resolution they are already held at; they are context
            // for the alignment, not the thing being measured.
            let mut rehydrated = None;
            if decimated {
                if let Some(build) = rehydrate.as_mut() {
                    rehydrated = build(keyframe_index);
                    if rehydrated.is_some() {
                        self.rehydrations += 1;
                    }
                }
            }

            let start = position.saturating_sub(window);
            let end = (position + window).min(self.keyframes.len().saturating_sub(1));

            let target = if start == end && rehydrated.is_none() {
                self.keyframes[position].cloud.clone()
            } else {
                let mut points = Vec::new();
                let mut normals = Vec::new();
                for neighbour in start..=end {
                    let frame = &self.keyframes[neighbour];
                    let relative = keyframe_pose.inverse() * frame.pose;
                    let cloud = if neighbour == position {
                        rehydrated.as_ref().unwrap_or(&frame.cloud)
                    } else {
                        &frame.cloud
                    };
                    points.extend(
                        cloud
                            .points
                            .iter()
                            .map(|point| transform_point(&relative, point)),
                    );
                    normals.extend(
                        cloud
                            .normals
                            .iter()
                            .map(|normal| relative.rotation * *normal),
                    );
                }
                if points.is_empty() {
                    continue;
                }
                IcpCloud::new(points, normals, self.config.max_distance)
            };

            if debug {
                eprintln!(
                    "[loop] {index} candidate kf {} window kf {}..={} target {} points",
                    keyframe_index,
                    self.keyframes[start].index,
                    self.keyframes[end].index,
                    target.len()
                );
            }

            prepared.push(LoopCandidate {
                from: keyframe_index,
                target,
                initial,
                spatial,
            });
        }

        prepared
    }

    /// Turn every accepted loop into a pose-graph edge.
    pub fn edges(&self) -> Vec<Edge> {
        self.loops
            .iter()
            .map(|loop_closure| loop_closure.edge(self.config.edge_weight, self.config.max_rmse))
            .collect()
    }

    /// Loop edges best-first, so the scanner can drop weak constraints when the
    /// pose graph asks for an implausible correction.
    pub fn edges_ranked(&self) -> Vec<Edge> {
        let mut loops: Vec<&Loop> = self.loops.iter().collect();
        loops.sort_by(|a, b| {
            b.inlier_ratio
                .total_cmp(&a.inlier_ratio)
                .then_with(|| a.rmse.total_cmp(&b.rmse))
        });
        loops
            .into_iter()
            .map(|loop_closure| loop_closure.edge(self.config.edge_weight, self.config.max_rmse))
            .collect()
    }
}

/// Camera-frame points for a world point cloud seen from `pose`.
///
/// Tests and replay use this to synthesise views; the scanner gets the same thing
/// from the depth frame.
pub fn view(world: &[Vector3<f32>], pose: &Isometry3<f32>) -> Vec<Vector3<f32>> {
    let world_to_camera = pose.inverse();
    world
        .iter()
        .map(|point| transform_point(&world_to_camera, point))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    /// A wavy surface. Not a plane: a plane leaves in-plane motion unobservable, so
    /// ICP could "succeed" while sliding along it.
    fn scene() -> Vec<Vector3<f32>> {
        let mut points = Vec::new();
        for i in 0..60 {
            for j in 0..60 {
                let x = -0.8 + 1.6 * i as f32 / 59.0;
                let y = -0.6 + 1.2 * j as f32 / 59.0;
                let z = 2.0 + 0.08 * (5.0 * x).sin() * (5.0 * y).cos();
                points.push(Vector3::new(x, y, z));
            }
        }
        points
    }

    fn at(x: f32, y: f32, z: f32) -> Isometry3<f32> {
        Isometry3::from_parts(Translation3::new(x, y, z), UnitQuaternion::identity())
    }

    fn finder() -> LoopFinder {
        LoopFinder::new(LoopClosureConfig {
            min_index_gap: 30,
            ..LoopClosureConfig::default()
        })
    }

    fn solid(width: usize, height: usize, rgb: [u8; 3]) -> Vec<u8> {
        let mut pixels = vec![0u8; width * height * 3];
        for pixel in pixels.chunks_exact_mut(3) {
            pixel.copy_from_slice(&rgb);
        }
        pixels
    }

    #[test]
    fn appearance_similarity_prefers_the_same_view() {
        let red =
            VisualDescriptor::from_rgb(&solid(32, 32, [220, 40, 40]), 32, 32).expect("descriptor");
        let blue =
            VisualDescriptor::from_rgb(&solid(32, 32, [40, 40, 220]), 32, 32).expect("descriptor");

        assert!(red.similarity(&red) > 0.99);
        assert!(red.similarity(&blue) < 0.5);
    }

    #[test]
    fn appearance_can_propose_a_loop_outside_the_pose_radius() {
        let world = scene();
        let config = LoopClosureConfig {
            search_radius: 0.1,
            min_index_gap: 30,
            max_distance: 0.3,
            ..LoopClosureConfig::default()
        };

        let start = at(0.0, 0.0, 0.0);
        let returned = at(0.2, 0.0, 0.0);
        let first = view(&world, &start);
        let second = view(&world, &returned);

        // Without appearance, 0.2 m is outside the 0.1 m search radius.
        let mut plain = LoopFinder::new(config);
        plain.consider(0, start, &first);
        assert!(plain.consider(50, returned, &second).is_none());
        assert_eq!(plain.attempts(), 0);

        let signature =
            VisualDescriptor::from_rgb(&solid(32, 32, [220, 40, 40]), 32, 32).expect("descriptor");
        let mut appearance = LoopFinder::new(config);
        appearance.consider_with_appearance(0, start, &first, Some(&signature));
        let found = appearance.consider_with_appearance(50, returned, &second, Some(&signature));

        assert!(
            found.is_some(),
            "appearance should retrieve the candidate and ICP should verify it"
        );
        assert_eq!(appearance.attempts(), 1);
    }

    #[test]
    fn nearby_frames_are_not_loop_keyframes() {
        let world = scene();
        let config = LoopClosureConfig {
            keyframe_min_translation: 0.1,
            keyframe_min_rotation: 0.2,
            ..LoopClosureConfig::default()
        };
        let mut finder = LoopFinder::new(config);

        finder.consider(0, at(0.0, 0.0, 0.0), &view(&world, &at(0.0, 0.0, 0.0)));
        assert_eq!(finder.keyframes().len(), 1);

        // Too little motion: this frame is not worth a cloud or a search.
        finder.consider(1, at(0.02, 0.0, 0.0), &view(&world, &at(0.02, 0.0, 0.0)));
        assert_eq!(finder.keyframes().len(), 1);

        // Enough motion from the last keyframe: keep it.
        finder.consider(2, at(0.2, 0.0, 0.0), &view(&world, &at(0.2, 0.0, 0.0)));
        assert_eq!(finder.keyframes().len(), 2);
    }

    #[test]
    fn old_keyframes_are_decimated_after_leaving_working_memory() {
        let config = LoopClosureConfig {
            keyframe_min_translation: 0.0,
            keyframe_min_rotation: 0.0,
            keyframe_max_interval: 1,
            working_memory_keyframes: 1,
            long_term_decimation: 2,
            ..LoopClosureConfig::default()
        };
        let world = scene();
        let mut finder = LoopFinder::new(config);

        finder.consider(0, at(0.0, 0.0, 0.0), &view(&world, &at(0.0, 0.0, 0.0)));
        let full = finder.keyframes()[0].point_count();
        assert!(full > 1);

        finder.consider(1, at(0.5, 0.0, 0.0), &view(&world, &at(0.5, 0.0, 0.0)));
        let old = finder.keyframes()[0].point_count();
        let new = finder.keyframes()[1].point_count();

        assert_eq!(old, full.div_ceil(2), "the old cloud was not decimated");
        assert!(new > old, "the newest keyframe should stay full-resolution");
        assert_eq!(finder.keyframe_points(), old + new);
    }

    #[test]
    fn a_decimated_keyframe_is_rehydrated_for_verification() {
        let config = LoopClosureConfig {
            keyframe_min_translation: 0.0,
            keyframe_min_rotation: 0.0,
            keyframe_max_interval: 1,
            working_memory_keyframes: 1,
            long_term_decimation: 2,
            min_index_gap: 1,
            ..LoopClosureConfig::default()
        };
        let world = scene();
        let mut finder = LoopFinder::new(config);

        finder.consider(0, at(0.0, 0.0, 0.0), &view(&world, &at(0.0, 0.0, 0.0)));
        finder.consider(1, at(0.5, 0.0, 0.0), &view(&world, &at(0.5, 0.0, 0.0)));
        let old = finder.keyframes()[0].point_count();

        let mut rebuilt = Vec::new();
        finder.consider_with_rehydration(
            2,
            at(1.0, 0.0, 0.0),
            &view(&world, &at(1.0, 0.0, 0.0)),
            None,
            |index| {
                rebuilt.push(index);
                Some(IcpCloud::build(
                    &view(&world, &at(0.0, 0.0, 0.0)),
                    config.voxel,
                    config.normal_radius,
                    config.max_distance,
                    &Vector3::zeros(),
                ))
            },
        );

        assert_eq!(
            rebuilt,
            vec![0],
            "only the long-term candidate should rebuild"
        );
        assert_eq!(
            finder.keyframes()[0].point_count(),
            old,
            "the temporary full cloud must not be stored back"
        );
    }

    #[test]
    fn loop_edges_are_ranked_by_quality_and_weighted_by_residual() {
        let config = LoopClosureConfig::default();
        let mut finder = LoopFinder::new(config);
        finder.loops.push(Loop {
            from: 4,
            to: 40,
            measurement: Isometry3::identity(),
            inlier_ratio: 0.60,
            rmse: 0.028,
        });
        finder.loops.push(Loop {
            from: 1,
            to: 50,
            measurement: Isometry3::identity(),
            inlier_ratio: 0.95,
            rmse: 0.005,
        });

        let edges = finder.edges_ranked();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].from, 1, "the stronger loop should rank first");
        assert!(
            edges[0].weight > edges[1].weight,
            "the tighter residual should carry more weight"
        );
        assert!(edges[0].weight <= config.edge_weight + 1e-6);
    }

    #[test]
    fn returning_to_a_place_is_detected_and_re_measured() {
        let world = scene();
        let mut finder = finder();

        // First keyframe: camera at the origin.
        assert!(
            finder
                .consider(
                    0,
                    Isometry3::identity(),
                    &view(&world, &Isometry3::identity())
                )
                .is_none()
        );

        // Much later, the camera is genuinely at (0.4, 0, 0), but odometry thinks
        // it is at (0.55, 0.05, 0) because of accumulated drift.
        let truth = at(0.4, 0.0, 0.0);
        let estimate = at(0.55, 0.05, 0.0);

        let found = finder
            .consider(50, estimate, &view(&world, &truth))
            .expect("the return to a previous place was not detected");

        assert_eq!(found.from, 0);
        assert_eq!(found.to, 50);
        assert!(
            found.inlier_ratio > 0.8,
            "weak overlap: {}",
            found.inlier_ratio
        );

        // The whole point: the measurement comes from the geometry, not from the
        // drifted poses, so it should land on the truth rather than the estimate.
        let recovered = found.measurement.translation.vector;
        assert!(
            (recovered - truth.translation.vector).norm() < 0.03,
            "recovered {recovered:?}, expected {:?}",
            truth.translation.vector
        );
        assert!(
            (recovered - estimate.translation.vector).norm() > 0.05,
            "the measurement just echoed the drifted pose estimate"
        );
    }

    #[test]
    fn a_different_place_is_not_accepted() {
        let world = scene();
        let mut finder = finder();

        finder.consider(
            0,
            Isometry3::identity(),
            &view(&world, &Isometry3::identity()),
        );

        // Same nominal pose, but the surface is 1.5 m further away, so nothing can
        // match within the correspondence distance.
        let mut moved = world.clone();
        for point in &mut moved {
            point.z += 1.5;
        }

        let found = finder.consider(50, at(0.1, 0.0, 0.0), &view(&moved, &at(0.1, 0.0, 0.0)));
        assert!(
            found.is_none(),
            "matched against unrelated geometry: {found:?}"
        );
    }

    #[test]
    fn recent_frames_are_never_candidates() {
        let world = scene();
        let mut finder = finder();

        finder.consider(
            0,
            Isometry3::identity(),
            &view(&world, &Isometry3::identity()),
        );

        // The very next frames are the same place, but a temporal neighbour says
        // nothing about drift, so the index gap must reject them.
        for index in 1..30 {
            let pose = at(0.001 * index as f32, 0.0, 0.0);
            let found = finder.consider(index, pose, &view(&world, &pose));
            assert!(found.is_none(), "frame {index} matched a neighbour");
        }
        assert_eq!(finder.attempts(), 0, "neighbours reached ICP at all");
    }

    #[test]
    fn loops_become_pose_graph_edges() {
        let world = scene();
        let mut finder = finder();

        finder.consider(
            0,
            Isometry3::identity(),
            &view(&world, &Isometry3::identity()),
        );
        finder
            .consider(50, at(0.4, 0.0, 0.0), &view(&world, &at(0.4, 0.0, 0.0)))
            .expect("loop");

        let edges = finder.edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, 0);
        assert_eq!(edges[0].to, 50);
        assert!(edges[0].weight > 0.0);
    }

    /// The left or right half of the wavy surface, in world coordinates.
    fn half_scene(left: bool) -> Vec<Vector3<f32>> {
        scene()
            .into_iter()
            .filter(|point| (point.x < 0.0) == left)
            .collect()
    }

    #[test]
    fn a_keyframe_window_widens_a_narrow_loop_target() {
        // Two complementary half-views from the same place, then a full view
        // later. One keyframe alone only covers half the source, which is below
        // the acceptance gate; the merged window covers all of it. This is the
        // LIO-SAM `loopFindNearKeyframes` idea, expressed as a behaviour rather
        // than a point count.
        let left = half_scene(true);
        let right = half_scene(false);
        let full = scene();

        let base = LoopClosureConfig {
            search_radius: 1.5,
            min_index_gap: 30,
            keyframe_min_translation: 0.0,
            keyframe_min_rotation: 0.0,
            keyframe_max_interval: 100,
            max_candidates: 1,
            min_inlier_ratio: 0.7,
            ..LoopClosureConfig::default()
        };

        let run = |window: usize| {
            let mut finder = LoopFinder::new(LoopClosureConfig {
                loop_target_keyframes: window,
                ..base
            });
            finder.consider(0, Isometry3::identity(), &left);
            finder.consider(1, Isometry3::identity(), &right);
            finder.consider(50, at(0.1, 0.0, 0.0), &full)
        };

        assert!(
            run(0).is_none(),
            "a single half-view should not clear the inlier gate"
        );
        assert!(
            run(1).is_some(),
            "the merged keyframe window should verify the same revisit"
        );
    }

    #[test]
    fn background_verification_delivers_the_same_loop() {
        let world = scene();
        let config = LoopClosureConfig {
            min_index_gap: 30,
            background: true,
            ..LoopClosureConfig::default()
        };
        let mut finder = LoopFinder::new(config);
        assert!(finder.is_background());

        finder.consider(
            0,
            Isometry3::identity(),
            &view(&world, &Isometry3::identity()),
        );

        let returned = at(0.4, 0.0, 0.0);
        assert!(
            finder
                .consider(50, returned, &view(&world, &returned))
                .is_none(),
            "background verification must not block the offering frame"
        );
        assert_eq!(finder.pending(), 1);

        let found = finder.flush();
        assert_eq!(found.len(), 1, "the worker's loop was not collected");
        assert_eq!(finder.loops().len(), 1);
        assert!(finder.attempts() > 0);
        assert_eq!(finder.pending(), 0);
        assert_eq!(found[0].from, 0);
        assert_eq!(found[0].to, 50);
    }

    #[test]
    fn the_likelihood_gate_can_reject_an_icp_verified_loop() {
        let world = scene();
        let returned = at(0.4, 0.0, 0.0);

        // A valid loop passes a normal likelihood threshold.
        let mut finder = LoopFinder::new(LoopClosureConfig {
            min_index_gap: 30,
            min_match_score: 0.5,
            ..LoopClosureConfig::default()
        });
        finder.consider(
            0,
            Isometry3::identity(),
            &view(&world, &Isometry3::identity()),
        );
        assert!(
            finder
                .consider(50, returned, &view(&world, &returned))
                .is_some(),
            "a valid loop failed the normal likelihood gate"
        );

        // An impossible threshold must reject the same loop even though ICP
        // and the reciprocal check both pass.
        let mut finder = LoopFinder::new(LoopClosureConfig {
            min_index_gap: 30,
            min_match_score: 1.01,
            ..LoopClosureConfig::default()
        });
        finder.consider(
            0,
            Isometry3::identity(),
            &view(&world, &Isometry3::identity()),
        );
        assert!(
            finder
                .consider(50, returned, &view(&world, &returned))
                .is_none(),
            "the likelihood gate was not applied"
        );
    }

    #[test]
    fn global_search_rescues_a_loop_from_a_bad_prior() {
        let world = scene();
        let mut finder = LoopFinder::new(LoopClosureConfig {
            min_index_gap: 30,
            // The drifted estimate is 0.55 m beyond the truth, which is outside
            // the ICP correspondence radius but inside the global search.
            max_distance: 0.12,
            min_inlier_ratio: 0.80,
            global_search: true,
            global_search_radius: 0.65,
            global_search_yaw: 0.40,
            global_search_min_score: 0.20,
            ..LoopClosureConfig::default()
        });

        finder.consider(
            0,
            Isometry3::identity(),
            &view(&world, &Isometry3::identity()),
        );

        let truth = at(0.4, 0.0, 0.0);
        let estimate = at(0.95, 0.0, 0.0);
        let found = finder
            .consider(50, estimate, &view(&world, &truth))
            .expect("the global search did not rescue the loop");
        assert_eq!(found.from, 0);
        assert_eq!(found.to, 50);
        assert!(found.inlier_ratio > 0.8);
    }
}
