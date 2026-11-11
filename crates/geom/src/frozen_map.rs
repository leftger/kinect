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

use crate::global_search::{self, GlobalSearchParams};
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
            voxel: 0.04,
            normal_radius: 0.12,
            max_distance: 0.25,
            min_inlier_ratio: 0.55,
            max_rmse: 0.03,
            global_search: GlobalSearchParams::default(),
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
}

impl FrozenMap {
    /// Index a prior point cloud. Returns `None` when the cloud is too small to
    /// align against after downsampling.
    pub fn from_points(points: &[Vector3<f32>], params: &FrozenMapParams) -> Option<Self> {
        if points.len() < 16 {
            return None;
        }
        let cloud = IcpCloud::build(
            points,
            params.voxel,
            params.normal_radius,
            params.max_distance,
            &Vector3::zeros(),
        );
        if cloud.points.len() < 16 {
            return None;
        }
        Some(Self { cloud })
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
    /// The global search is tried first; if it finds nothing, ICP still gets a
    /// chance from the prior. Either way the same inlier/RMSE gates decide
    /// whether the pose is returned.
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

        let mut initial = prior;
        let mut score = None;
        if let Some(found) = global_search::search(
            &self.cloud.points,
            &source.points,
            prior,
            &params.global_search,
        ) {
            initial = found.transform;
            score = Some(found.score);
        }

        let result = icp::align(
            &source,
            &self.cloud,
            initial,
            &IcpParams {
                max_correspondence_distance: params.max_distance,
                ..IcpParams::default()
            },
        );
        if result.inlier_ratio < params.min_inlier_ratio || result.rmse > params.max_rmse {
            return None;
        }

        Some(FrozenMatch {
            pose: result.transform,
            score: score.unwrap_or(0.0),
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
