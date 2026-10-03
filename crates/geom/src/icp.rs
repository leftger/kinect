//! Point-to-plane ICP: the scanner's odometry.
//!
//! Each frame is aligned to the previous one to recover the sensor's motion.
//! Point-to-plane is used rather than point-to-point because it converges in far
//! fewer iterations on the planar surfaces that dominate indoor scenes, and it
//! tolerates sliding along a wall (which genuinely is unobservable from a
//! single depth view) instead of fighting it.

use crate::spatial::VoxelHash;
use crate::{normals, transform_point, voxel};
use nalgebra::{Isometry3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};

#[derive(Clone, Copy, Debug)]
pub struct IcpParams {
    /// Correspondences farther apart than this are discarded.
    pub max_correspondence_distance: f32,
    /// Huber threshold: residuals larger than this are down-weighted, which is
    /// what keeps moving objects and depth outliers from dragging the solution.
    pub huber_delta: f32,
    /// Stop once an iteration moves the cloud less than this (units of the twist
    /// vector, so metres and radians mixed).
    pub convergence_epsilon: f32,
    pub max_iterations: usize,
}

impl Default for IcpParams {
    fn default() -> Self {
        Self {
            max_correspondence_distance: 0.10,
            // Sensor noise measured at ~5 mm RMS on planar surfaces, so residuals
            // beyond a couple of centimetres are outliers, not signal.
            huber_delta: 0.02,
            // 0.1 mm of translation / 0.1 mrad of rotation. Tighter than this is
            // below the sensor's own 5 mm noise, so demanding it just means every
            // frame "fails to converge" after burning the whole iteration budget.
            convergence_epsilon: 1e-4,
            max_iterations: 30,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct IcpResult {
    /// Maps the source cloud into the target cloud's frame.
    pub transform: Isometry3<f32>,
    /// Source points that found a correspondence on the final iteration.
    pub correspondences: usize,
    /// `correspondences / source_len`.
    pub inlier_ratio: f32,
    /// RMS point-to-plane residual over those correspondences, in metres.
    pub rmse: f32,
    pub iterations: usize,
    /// False when too little of the source matched to trust the result.
    pub converged: bool,
}

impl IcpResult {
    /// A result that claims nothing: no correspondences, no convergence. Used
    /// both internally and by other correspondence strategies (see
    /// [`crate::projective`]) that need to report "could not align".
    pub fn failed(transform: Isometry3<f32>) -> Self {
        Self {
            transform,
            correspondences: 0,
            inlier_ratio: 0.0,
            rmse: f32::INFINITY,
            iterations: 0,
            converged: false,
        }
    }
}

/// A cloud prepared for alignment: downsampled, with normals, plus the spatial
/// index used for correspondence search.
///
/// Build these once per frame per scale — the downsample and normal estimation
/// are far more expensive than the ICP iterations themselves.
pub struct IcpCloud {
    pub points: Vec<Vector3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    hash: VoxelHash,
    /// The radius this cloud was indexed for; the correspondence distance any
    /// query against it must use.
    query_radius: f32,
}

impl IcpCloud {
    /// From points and normals that are already known (used by tests and by
    /// replay).
    pub fn new(points: Vec<Vector3<f32>>, normals: Vec<Vector3<f32>>, query_radius: f32) -> Self {
        assert_eq!(
            points.len(),
            normals.len(),
            "every point needs exactly one normal"
        );

        // Measured, and the opposite of what the volumes suggest: a *smaller*
        // cell is slower. A radius-r query over cells of size c visits
        // (1 + 2r/c)^3 cells, and the dominant cost is the hash lookup per cell
        // (SipHash over the cell key), not the distance test per point. Cell = r
        // visits 27 cells; cell = r/2 visits 125 and measured 3x slower overall
        // (tracking 638 ms -> 1981 ms per frame). Leave the cell at the query
        // radius.
        let hash = VoxelHash::build(query_radius, &points);

        Self {
            points,
            normals,
            hash,
            query_radius,
        }
    }

    /// Downsample, estimate normals, and index.
    ///
    /// `query_radius` should be the largest correspondence distance that will be
    /// queried against this cloud.
    pub fn build(
        raw_points: &[Vector3<f32>],
        voxel_size: f32,
        normal_radius: f32,
        query_radius: f32,
        viewpoint: &Vector3<f32>,
    ) -> Self {
        let points = voxel::downsample(raw_points, voxel_size);
        let normals = normals::estimate(&points, normal_radius, normals::MIN_NEIGHBOURS, viewpoint);
        Self::new(points, normals, query_radius)
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Number of points that actually carry a usable normal.
    pub fn normals_present(&self) -> usize {
        self.normals
            .iter()
            .filter(|n| n.norm_squared() > 0.0)
            .count()
    }
}

/// Finds the surface point and normal that a transformed source point matches.
///
/// Point-to-plane ICP does not care *how* the match was found, only that it was,
/// so the solver is separated from the search. That lets one already-tested
/// solver drive both a spatial-index lookup ([`IcpCloud`]) and projective data
/// association against a depth image ([`crate::projective`]) -- the latter being
/// much cheaper, because it replaces a 27-cell hash probe per correspondence
/// with a single projection.
pub trait CorrespondenceSource {
    /// Match `moved`, already transformed into the target frame.
    fn find(&self, moved: &Vector3<f32>) -> Option<(Vector3<f32>, Vector3<f32>)>;
}

impl CorrespondenceSource for IcpCloud {
    fn find(&self, moved: &Vector3<f32>) -> Option<(Vector3<f32>, Vector3<f32>)> {
        let (index, _) = self
            .hash
            .nearest_within(&self.points, moved, self.query_radius)?;

        let index = index as usize;
        let normal = self.normals[index];
        if normal.norm_squared() == 0.0 {
            return None;
        }

        Some((self.points[index], normal))
    }
}

/// Align `source` onto `target` with point-to-plane ICP.
///
/// `initial` is the starting guess mapping source into the target frame; pass the
/// previous frame's motion for a good one, or `Isometry3::identity()`.
pub fn align(
    source: &IcpCloud,
    target: &IcpCloud,
    initial: Isometry3<f32>,
    params: &IcpParams,
) -> IcpResult {
    if source.is_empty() || target.is_empty() {
        return IcpResult::failed(initial);
    }

    let usable_source: Vec<Vector3<f32>> = source
        .points
        .iter()
        .zip(&source.normals)
        .filter(|(_, n)| n.norm_squared() > 0.0)
        .map(|(p, _)| *p)
        .collect();

    if usable_source.is_empty() {
        return IcpResult::failed(initial);
    }

    align_points(&usable_source, target, initial, params)
}

/// Point-to-plane ICP over an explicit list of source points.
///
/// This is the solver. [`align`] and
/// [`crate::projective::align_level`] are the two ways of supplying it with
/// correspondences.
pub fn align_points(
    source: &[Vector3<f32>],
    target: &impl CorrespondenceSource,
    initial: Isometry3<f32>,
    params: &IcpParams,
) -> IcpResult {
    if source.is_empty() {
        return IcpResult::failed(initial);
    }

    let mut transform = initial;
    let mut converged = false;
    let mut last_inliers = 0usize;
    let mut last_rmse = f32::INFINITY;
    let mut iterations = 0usize;

    for iteration in 0..params.max_iterations {
        iterations = iteration + 1;

        let mut hessian = Matrix6::<f32>::zeros();
        let mut gradient = Vector6::<f32>::zeros();
        let mut inliers = 0usize;
        let mut squared_residual = 0.0f64;

        for point in source {
            let moved = transform_point(&transform, point);

            let Some((target_point, target_normal)) = target.find(&moved) else {
                continue;
            };

            let residual = (moved - target_point).dot(&target_normal);
            let magnitude = residual.abs();

            // Huber: quadratic near zero, linear in the tails.
            let weight = if magnitude <= params.huber_delta {
                1.0
            } else {
                params.huber_delta / magnitude
            };

            // Linearising T <- exp(xi) * T gives
            //     r' = r + omega . (moved x n) + n . v
            // so the Jacobian row is [ (moved x n) | n ].
            let moment = moved.cross(&target_normal);
            let jacobian = Vector6::new(
                moment.x,
                moment.y,
                moment.z,
                target_normal.x,
                target_normal.y,
                target_normal.z,
            );

            hessian += weight * jacobian * jacobian.transpose();
            gradient += weight * jacobian * residual;

            inliers += 1;
            squared_residual += (residual as f64) * (residual as f64);
        }

        last_inliers = inliers;
        last_rmse = if inliers > 0 {
            (squared_residual / inliers as f64).sqrt() as f32
        } else {
            f32::INFINITY
        };

        if inliers < 6 {
            // Under-constrained (a rigid transform needs at least 6 independent
            // constraints) -- return what we started with rather than nonsense.
            return IcpResult {
                transform,
                correspondences: inliers,
                inlier_ratio: inliers as f32 / source.len() as f32,
                rmse: last_rmse,
                iterations,
                converged: false,
            };
        }

        // Slight Levenberg damping keeps the 6x6 system solvable when the
        // geometry is weakly constrained (e.g. staring at a single flat wall).
        let damping = 1e-6 + 1e-4 * hessian.diagonal().max();
        let damped = hessian + Matrix6::identity() * damping;

        let Some(step) = damped.lu().solve(&(-gradient)) else {
            break;
        };

        let omega = Vector3::new(step[0], step[1], step[2]);
        let translation = Vector3::new(step[3], step[4], step[5]);

        let delta = Isometry3::from_parts(
            Translation3::from(translation),
            UnitQuaternion::from_scaled_axis(omega),
        );
        transform = delta * transform;

        if omega.norm() < params.convergence_epsilon
            && translation.norm() < params.convergence_epsilon
        {
            converged = true;
            break;
        }
    }

    IcpResult {
        transform,
        correspondences: last_inliers,
        inlier_ratio: last_inliers as f32 / source.len() as f32,
        rmse: last_rmse,
        iterations,
        // Whether the step actually fell below tolerance, rather than merely
        // running without error. Exhausting the iteration budget with the residual
        // still moving is exactly the case a caller needs to know about.
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gently wavy surface. Deliberately not a plane: a plane leaves in-plane
    /// translation and rotation unobservable, so it cannot test all six degrees
    /// of freedom.
    fn wavy_surface(side: usize) -> (Vec<Vector3<f32>>, Vec<Vector3<f32>>) {
        let amplitude = 0.05f32;
        let frequency = 4.0f32;
        let extent = 0.6f32;

        let mut points = Vec::new();
        let mut normals = Vec::new();

        for i in 0..side {
            for j in 0..side {
                let x = -extent + 2.0 * extent * i as f32 / (side - 1) as f32;
                let y = -extent + 2.0 * extent * j as f32 / (side - 1) as f32;

                let z = amplitude * (frequency * x).sin() * (frequency * y).cos();
                let dz_dx = amplitude * frequency * (frequency * x).cos() * (frequency * y).cos();
                let dz_dy = -amplitude * frequency * (frequency * x).sin() * (frequency * y).sin();

                points.push(Vector3::new(x, y, z));
                normals.push(Vector3::new(-dz_dx, -dz_dy, 1.0).normalize());
            }
        }

        (points, normals)
    }

    fn perturbed(truth: &Isometry3<f32>, points: &[Vector3<f32>], normals: &[Vector3<f32>]) -> (Vec<Vector3<f32>>, Vec<Vector3<f32>>) {
        (
            points.iter().map(|p| transform_point(truth, p)).collect(),
            normals.iter().map(|n| truth.rotation * n).collect(),
        )
    }

    /// Angle between two rotations, in degrees.
    fn angle_between(a: &UnitQuaternion<f32>, b: &UnitQuaternion<f32>) -> f32 {
        (a * b.inverse()).angle().to_degrees()
    }

    #[test]
    fn recovers_a_known_transform() {
        let (target_points, target_normals) = wavy_surface(45);
        let target = IcpCloud::new(target_points, target_normals, 0.15);

        // 3 degrees about an arbitrary axis, 3 cm translation.
        let truth = Isometry3::from_parts(
            Translation3::new(0.028, -0.021, 0.017),
            UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(Vector3::new(0.3, -0.7, 0.6)),
                3.0f32.to_radians(),
            ),
        );

        let (raw_source_points, raw_source_normals) = perturbed(
            &truth,
            &target.points,
            &target.normals,
        );
        let source = IcpCloud::new(raw_source_points, raw_source_normals, 0.15);

        let result = align(&source, &target, Isometry3::identity(), &IcpParams {
            max_correspondence_distance: 0.15,
            ..IcpParams::default()
        });

        assert!(result.converged, "ICP should converge on this input");
        assert!(result.inlier_ratio > 0.9, "inlier ratio {}", result.inlier_ratio);

        // Aligning source onto target must undo `truth`.
        let expected = truth.inverse();
        let rotation_error = angle_between(&result.transform.rotation, &expected.rotation);
        let translation_error =
            (result.transform.translation.vector - expected.translation.vector).norm();

        assert!(rotation_error < 0.5, "rotation error {rotation_error} deg");
        assert!(
            translation_error < 0.008,
            "translation error {translation_error} m"
        );
    }

    #[test]
    fn identity_input_gives_identity() {
        let (points, normals) = wavy_surface(30);
        let cloud = IcpCloud::new(points.clone(), normals.clone(), 0.1);

        let result = align(&cloud, &cloud, Isometry3::identity(), &IcpParams::default());

        assert!(result.transform.translation.vector.norm() < 1e-4);
        assert!(result.transform.rotation.angle() < 1e-3);
        assert!(result.rmse < 1e-4, "rmse {}", result.rmse);
    }

    #[test]
    fn disjoint_clouds_report_failure() {
        let (points, normals) = wavy_surface(30);
        let target = IcpCloud::new(points, normals, 0.1);

        let far_points: Vec<Vector3<f32>> = (0..500)
            .map(|i| Vector3::new(20.0 + i as f32 * 0.01, 20.0, 20.0))
            .collect();
        let far_normals = vec![Vector3::new(0.0, 0.0, 1.0); far_points.len()];
        let source = IcpCloud::new(far_points, far_normals, 0.1);

        let result = align(&source, &target, Isometry3::identity(), &IcpParams::default());

        assert!(!result.converged);
        assert_eq!(result.correspondences, 0);
        assert_eq!(result.inlier_ratio, 0.0);
    }

    #[test]
    fn empty_input_is_reported_as_failure() {
        let empty = IcpCloud::new(Vec::new(), Vec::new(), 0.1);
        let (points, normals) = wavy_surface(10);
        let target = IcpCloud::new(points, normals, 0.1);

        assert!(!align(&empty, &target, Isometry3::identity(), &IcpParams::default()).converged);
        assert!(!align(&target, &empty, Isometry3::identity(), &IcpParams::default()).converged);
    }

    /// Noise at the level measured from the real sensor (5 mm RMS) must not
    /// wreck the estimate.
    #[test]
    fn tolerates_sensor_level_noise() {
        let (target_points, target_normals) = wavy_surface(45);
        let target = IcpCloud::new(target_points, target_normals, 0.15);

        let truth = Isometry3::from_parts(
            Translation3::new(0.02, 0.015, -0.01),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 2.0f32.to_radians()),
        );

        let mut state = 987_654_321u64;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0 - 0.5) * 0.01
        };

        let (noisy_points, noisy_normals) = perturbed(&truth, &target.points, &target.normals);
        let noisy_points: Vec<Vector3<f32>> = noisy_points
            .into_iter()
            .map(|p| p + Vector3::new(noise(), noise(), noise()))
            .collect();
        let source = IcpCloud::new(noisy_points, noisy_normals, 0.15);

        let result = align(&source, &target, Isometry3::identity(), &IcpParams {
            max_correspondence_distance: 0.15,
            ..IcpParams::default()
        });

        let expected = truth.inverse();
        let rotation_error = angle_between(&result.transform.rotation, &expected.rotation);
        let translation_error =
            (result.transform.translation.vector - expected.translation.vector).norm();

        assert!(rotation_error < 1.0, "rotation error {rotation_error} deg");
        assert!(translation_error < 0.015, "translation error {translation_error} m");
    }
}
