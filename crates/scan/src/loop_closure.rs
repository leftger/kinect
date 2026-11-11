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

use geom::icp::{IcpCloud, IcpParams, align};
use geom::pose_graph::Edge;
use geom::transform_point;
use nalgebra::{Isometry3, Vector3};

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

/// Collects keyframes and the loop closures between them.
pub struct LoopFinder {
    config: LoopClosureConfig,
    keyframes: Vec<Keyframe>,
    loops: Vec<Loop>,
    attempts: usize,
    last_keyframe: Option<Isometry3<f32>>,
    /// Index of the first keyframe still in full-resolution working memory.
    working_start: usize,
}

impl LoopFinder {
    pub fn new(config: LoopClosureConfig) -> Self {
        Self {
            config,
            keyframes: Vec::new(),
            loops: Vec::new(),
            attempts: 0,
            last_keyframe: None,
            working_start: 0,
        }
    }

    pub fn keyframes(&self) -> &[Keyframe] {
        &self.keyframes
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

        let found = self.search(index, &pose, &cloud, appearance);

        self.keyframes.push(Keyframe {
            index,
            pose,
            appearance: appearance.cloned(),
            cloud,
        });
        self.manage_memory();
        self.last_keyframe = Some(pose);

        if let Some(ref loop_closure) = found {
            self.loops.push(*loop_closure);
        }

        found
    }

    /// Align the new frame against the most plausible earlier ones.
    fn search(
        &mut self,
        index: usize,
        pose: &Isometry3<f32>,
        cloud: &IcpCloud,
        appearance: Option<&VisualDescriptor>,
    ) -> Option<Loop> {
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

        if candidates.is_empty() {
            return None;
        }

        let params = IcpParams {
            max_correspondence_distance: self.config.max_distance,
            max_iterations: self.config.max_iterations,
            ..IcpParams::default()
        };

        let mut best: Option<Loop> = None;

        for (spatial, position) in candidates {
            // A verified spatial loop is enough; only fall back to appearance
            // when the entire pose-proximity pass produced nothing.
            if !spatial && best.is_some() {
                break;
            }
            let keyframe = &self.keyframes[position];

            // Starting guess: where the poses say the earlier frame is, relative
            // to this one. ICP then moves it onto the geometry.
            let initial = keyframe.pose.inverse() * pose;

            self.attempts += 1;
            let result = align(cloud, &keyframe.cloud, initial, &params);
            if debug {
                eprintln!(
                    "[loop] align kf {} -> {index}: inliers {:.3} rmse {:.4} corr {}",
                    keyframe.index, result.inlier_ratio, result.rmse, result.correspondences
                );
            }

            if result.inlier_ratio < self.config.min_inlier_ratio
                || result.rmse > self.config.max_rmse
            {
                continue;
            }

            // A real place match survives being aligned in the other direction.
            // Reverse ICP starts from the inverse of the forward result, so a
            // one-sided surface fit shows up as a large round-trip error.
            let reverse = align(&keyframe.cloud, cloud, result.transform.inverse(), &params);
            let round_trip = reverse.transform * result.transform;
            // Reverse RMSE is not a gate: a slightly noisier reverse fit with a
            // tiny round-trip error is exactly what a correct loop looks like.
            if reverse.inlier_ratio < self.config.min_inlier_ratio
                || round_trip.translation.vector.norm() > self.config.reciprocal_max_translation
                || round_trip.rotation.angle() > self.config.reciprocal_max_rotation
            {
                if debug {
                    eprintln!(
                        "[loop] reject kf {} -> {index}: reverse inliers {:.3} rmse {:.4}, round-trip {:.3} m {:.3} rad",
                        keyframe.index,
                        reverse.inlier_ratio,
                        reverse.rmse,
                        round_trip.translation.vector.norm(),
                        round_trip.rotation.angle()
                    );
                }
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
}
