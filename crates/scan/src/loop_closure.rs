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
//! Each surviving candidate is then aligned geometrically with point-to-plane ICP
//! (`geom::icp::align`, the spatial-index path, since two keyframes that are far
//! apart share no viewpoint and projective association does not apply). Only an
//! alignment that meets both an inlier-ratio and a residual threshold is accepted.
//!
//! That verification is not optional. A pose graph will happily fold a map in half
//! to satisfy a single bad edge, so a false loop is far worse than a missed one.

use geom::icp::{align, IcpCloud, IcpParams};
use geom::pose_graph::Edge;
use geom::transform_point;
use nalgebra::{Isometry3, Vector3};

/// A frame kept as a place the sensor might return to.
pub struct Keyframe {
    /// Index of this frame within the scan, which is what edges refer to.
    pub index: usize,
    pub pose: Isometry3<f32>,
    cloud: IcpCloud,
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
    /// How many candidates to spend ICP on per frame.
    pub max_candidates: usize,
    /// ICP acceptance: fraction of points that must find a match...
    pub min_inlier_ratio: f32,
    /// ...and the RMS residual they must achieve, in metres.
    pub max_rmse: f32,
    /// Confidence given to a loop edge, relative to an odometry edge.
    pub edge_weight: f32,
    /// Resolution to keep keyframes at. Coarse keeps the memory per keyframe small
    /// and gives wide-baseline ICP a usable basin of convergence.
    pub voxel: f32,
    pub normal_radius: f32,
    pub max_distance: f32,
    pub max_iterations: usize,
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            search_radius: 1.5,
            min_index_gap: 40,
            max_candidates: 3,
            min_inlier_ratio: 0.5,
            max_rmse: 0.03,
            // Below odometry on purpose: a loop is a hypothesis, an odometry edge
            // is a measurement of something we actually saw move.
            edge_weight: 0.5,
            voxel: 0.04,
            normal_radius: 0.12,
            max_distance: 0.25,
            max_iterations: 20,
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
    pub fn edge(&self, weight: f32) -> Edge {
        Edge {
            from: self.from,
            to: self.to,
            measurement: self.measurement,
            weight,
        }
    }
}

/// Collects keyframes and the loop closures between them.
pub struct LoopFinder {
    config: LoopClosureConfig,
    keyframes: Vec<Keyframe>,
    loops: Vec<Loop>,
    attempts: usize,
}

impl LoopFinder {
    pub fn new(config: LoopClosureConfig) -> Self {
        Self {
            config,
            keyframes: Vec::new(),
            loops: Vec::new(),
            attempts: 0,
        }
    }

    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
    }

    pub fn loops(&self) -> &[Loop] {
        &self.loops
    }

    /// How many ICP alignments were spent, accepted or not. A high number with no
    /// loops means the candidate filter is not narrowing enough.
    pub fn attempts(&self) -> usize {
        self.attempts
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
        let origin = Vector3::zeros();
        let cloud = IcpCloud::build(
            points,
            self.config.voxel,
            self.config.normal_radius,
            self.config.max_distance,
            &origin,
        );

        let found = self.search(index, &pose, &cloud);

        self.keyframes.push(Keyframe { index, pose, cloud });

        if let Some(ref loop_closure) = found {
            self.loops.push(*loop_closure);
        }

        found
    }

    /// Align the new frame against the most plausible earlier ones.
    fn search(&mut self, index: usize, pose: &Isometry3<f32>, cloud: &IcpCloud) -> Option<Loop> {
        let mut candidates: Vec<(f32, usize)> = self
            .keyframes
            .iter()
            .enumerate()
            .filter(|(_, keyframe)| {
                index.saturating_sub(keyframe.index) >= self.config.min_index_gap
            })
            .map(|(position, keyframe)| {
                (
                    (keyframe.pose.translation.vector - pose.translation.vector).norm(),
                    position,
                )
            })
            .filter(|(distance, _)| *distance <= self.config.search_radius)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Closest first: the pose estimate is wrong, but not arbitrarily wrong.
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        candidates.truncate(self.config.max_candidates);

        let params = IcpParams {
            max_correspondence_distance: self.config.max_distance,
            max_iterations: self.config.max_iterations,
            ..IcpParams::default()
        };

        let mut best: Option<Loop> = None;

        for (_, position) in candidates {
            let keyframe = &self.keyframes[position];

            // Starting guess: where the poses say the earlier frame is, relative
            // to this one. ICP then moves it onto the geometry.
            let initial = keyframe.pose.inverse() * pose;

            self.attempts += 1;
            let result = align(cloud, &keyframe.cloud, initial, &params);

            if result.inlier_ratio < self.config.min_inlier_ratio
                || result.rmse > self.config.max_rmse
            {
                continue;
            }

            let candidate = Loop {
                from: keyframe.index,
                to: index,
                measurement: result.transform,
                inlier_ratio: result.inlier_ratio,
                rmse: result.rmse,
            };

            // Keep the best of the verified candidates rather than the first.
            let better = match &best {
                None => true,
                Some(previous) => candidate.inlier_ratio > previous.inlier_ratio,
            };
            if better {
                best = Some(candidate);
            }
        }

        best
    }

    /// Turn every accepted loop into a pose-graph edge.
    pub fn edges(&self) -> Vec<Edge> {
        self.loops
            .iter()
            .map(|loop_closure| loop_closure.edge(self.config.edge_weight))
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

    #[test]
    fn returning_to_a_place_is_detected_and_re_measured() {
        let world = scene();
        let mut finder = finder();

        // First keyframe: camera at the origin.
        assert!(finder
            .consider(
                0,
                Isometry3::identity(),
                &view(&world, &Isometry3::identity())
            )
            .is_none());

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
}
