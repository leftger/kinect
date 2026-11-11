//! Localisation against a frozen prior map.
//!
//! A pure-localisation mode keeps the map fixed and tracks a new
//! trajectory inside it. For a depth camera: a prior
//! point cloud is indexed once, each query frame gets a 4-DOF global search for
//! an initial guess, and point-to-plane ICP refines it. The map is never
//! modified; the caller only receives a pose.
//!
//! This is deliberately separate from [`crate::tsdf`]: a prior mesh or point
//! cloud is already a surface, and re-fusing it would both lose the prior's
//! accuracy and pay a needless integration cost.

use crate::global_search::{self, GlobalSearchParams, ScoreGrid};
use crate::icp::{self, IcpCloud, IcpParams};
use nalgebra::{Isometry3, Vector3};

/// Tuning for a frozen map and its per-frame localisation.
#[derive(Clone, Copy, Debug)]
pub struct FrozenMapParams {
    /// Voxel size used to downsample both clouds, in metres.
    pub voxel: f32,
    /// Normal-estimation radius, in metres.
    pub normal_radius: f32,
    /// ICP correspondence radius, in metres.
    pub max_distance: f32,
    /// Minimum ICP inlier ratio for a pose to be accepted.
    pub min_inlier_ratio: f32,
    /// Maximum ICP point-to-plane RMSE, in metres.
    pub max_rmse: f32,
    /// Branch-and-bound search settings. Its `translation_radius` and
    /// `yaw_radius` bound how far the prior pose may be wrong before the search
    /// gives up.
    pub global_search: GlobalSearchParams,
}

impl Default for FrozenMapParams {
    fn default() -> Self {
        Self {
            // A prior mesh can have millions of vertices. Six centimetres keeps
            // the map index and the per-frame ICP affordable; the global grid
            // still uses its own finer resolution.
            voxel: 0.06,
            normal_radius: 0.18,
            max_distance: 0.35,
            // A prior mesh and a live depth frame are sampled differently, so
            // these are looser than the odometry gates. The trajectory is not
            // fused, so a false accept costs a pose, not the map.
            min_inlier_ratio: 0.40,
            max_rmse: 0.05,
            global_search: GlobalSearchParams {
                min_score: 0.25,
                ..GlobalSearchParams::default()
            },
        }
    }
}

/// One localisation result.
#[derive(Clone, Copy, Debug)]
pub struct FrozenMatch {
    /// World-from-camera pose in the frozen map's frame.
    pub pose: Isometry3<f32>,
    /// Global-search likelihood, or zero when ICP started from the prior.
    pub score: f32,
    pub inlier_ratio: f32,
    pub rmse: f32,
}

/// A point cloud treated as an immutable map.
pub struct FrozenMap {
    cloud: IcpCloud,
    /// Built once; reused by every localisation query.
    grid: Option<ScoreGrid>,
}

impl FrozenMap {
    /// Index a prior point cloud. Returns `None` when the cloud is too small to
    /// align against after downsampling.
    ///
    /// Non-finite points and the large sentinel values that surface-nets writes
    /// for unmeshed vertices are discarded before the voxel hash sees them.
    pub fn from_points(points: &[Vector3<f32>], params: &FrozenMapParams) -> Option<Self> {
        let points: Vec<Vector3<f32>> = points
            .iter()
            .copied()
            .filter(|point| {
                point
                    .iter()
                    .all(|value| value.is_finite() && value.abs() < 1.0e6)
            })
            .collect();
        if points.len() < 16 {
            return None;
        }
        let cloud = IcpCloud::build(
            &points,
            params.voxel,
            params.normal_radius,
            params.max_distance,
            &Vector3::zeros(),
        );
        if cloud.points.len() < 16 {
            return None;
        }
        // The grid depends only on the map, not on the query, so build it once
        // here. Rebuilding it per frame turned a room-sized localisation into a
        // multi-minute operation.
        let grid = ScoreGrid::build(&cloud.points, &params.global_search);
        Some(Self { cloud, grid })
    }

    pub fn points(&self) -> &[Vector3<f32>] {
        &self.cloud.points
    }

    pub fn len(&self) -> usize {
        self.cloud.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cloud.points.is_empty()
    }

    /// Localise one query cloud. `prior` is the previous pose, or identity on
    /// the first frame.
    ///
    /// ICP from the prior is the fast path. Only when it fails the inlier/RMSE
    /// gates does the cached global grid supply an initial guess for a second
    /// ICP pass. Either way the same gates decide whether the pose is returned.
    pub fn localize(
        &self,
        query: &[Vector3<f32>],
        prior: Isometry3<f32>,
        params: &FrozenMapParams,
    ) -> Option<FrozenMatch> {
        let source = IcpCloud::build(
            query,
            params.voxel,
            params.normal_radius,
            params.max_distance,
            &Vector3::zeros(),
        );
        if source.points.len() < 16 {
            return None;
        }

        let icp_params = IcpParams {
            max_correspondence_distance: params.max_distance,
            huber_delta: 0.03,
            // Live localisation needs a bounded per-frame cost. A mesh prior is
            // coarser than an odometry target, so it does not need 30 sweeps.
            max_iterations: 12,
            ..IcpParams::default()
        };

        // Normal tracking: refine from the previous pose. This is the fast path
        // and the one a live session takes almost every frame.
        let result = icp::align(&source, &self.cloud, prior, &icp_params);
        if result.inlier_ratio >= params.min_inlier_ratio && result.rmse <= params.max_rmse {
            return Some(FrozenMatch {
                pose: result.transform,
                score: 0.0,
                inlier_ratio: result.inlier_ratio,
                rmse: result.rmse,
            });
        }

        // Lost: only now pay for the global search. The grid was built once with
        // the map, so this is the branch-and-bound cost, not a grid rebuild.
        let Some(grid) = &self.grid else {
            return None;
        };
        let found =
            global_search::search_with_grid(grid, &source.points, prior, &params.global_search)?;
        let result = icp::align(&source, &self.cloud, found.transform, &icp_params);
        if result.inlier_ratio < params.min_inlier_ratio || result.rmse > params.max_rmse {
            return None;
        }

        Some(FrozenMatch {
            pose: result.transform,
            score: found.score,
            inlier_ratio: result.inlier_ratio,
            rmse: result.rmse,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform_point;
    use nalgebra::{Translation3, UnitQuaternion};

    fn wavy_surface() -> Vec<Vector3<f32>> {
        let mut points = Vec::new();
        for i in 0..50 {
            for j in 0..50 {
                let x = -0.6 + 1.2 * i as f32 / 49.0;
                let y = -0.4 + 0.8 * j as f32 / 49.0;
                let z = 2.0 + 0.06 * (6.0 * x).sin() * (5.0 * y).cos();
                points.push(Vector3::new(x, y, z));
            }
        }
        points
    }

    fn params() -> FrozenMapParams {
        FrozenMapParams {
            voxel: 0.05,
            normal_radius: 0.12,
            max_distance: 0.25,
            min_inlier_ratio: 0.55,
            max_rmse: 0.03,
            global_search: GlobalSearchParams {
                voxel: 0.05,
                levels: 4,
                sigma: 0.10,
                translation_radius: 0.30,
                yaw_radius: 0.30,
                yaw_step: 0.10,
                max_points: 300,
                max_nodes: 6_000,
                min_score: 0.30,
            },
        }
    }

    #[test]
    fn localizes_a_query_against_a_frozen_surface() {
        let target = wavy_surface();
        let map = FrozenMap::from_points(&target, &params()).expect("frozen map");

        let truth = Isometry3::from_parts(
            Translation3::new(0.10, -0.04, 0.02),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.08),
        );
        let query: Vec<Vector3<f32>> = target
            .iter()
            .map(|point| transform_point(&truth.inverse(), point))
            .collect();

        let found = map
            .localize(&query, Isometry3::identity(), &params())
            .expect("localized");
        let error = truth.inverse() * found.pose;
        assert!(
            error.translation.vector.norm() < 0.08,
            "translation error {:.3} m",
            error.translation.vector.norm()
        );
        assert!(
            error.rotation.angle() < 0.12,
            "rotation error {:.3} rad",
            error.rotation.angle()
        );
        assert!(found.inlier_ratio > 0.5);
    }

    #[test]
    fn rejects_a_query_that_does_not_belong_to_the_map() {
        let target = wavy_surface();
        let map = FrozenMap::from_points(&target, &params()).expect("frozen map");
        let unrelated: Vec<Vector3<f32>> = target
            .iter()
            .map(|point| point + Vector3::new(0.0, 0.0, 3.0))
            .collect();
        assert!(
            map.localize(&unrelated, Isometry3::identity(), &params())
                .is_none(),
            "an unrelated cloud was localised"
        );
    }
}
