//! Surface normal estimation by local PCA.

use crate::spatial::VoxelHash;
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;

/// Fewer neighbours than this and a PCA normal is meaningless, so the point is
/// reported as having no normal (zero vector).
pub const MIN_NEIGHBOURS: usize = 6;

/// Per-point normals from the smallest principal axis of the local neighbourhood.
///
/// Normals are oriented to face `viewpoint` (normally the camera position), so
/// the sign is consistent across a scan rather than flipping arbitrarily with
/// the eigenvector solver.
///
/// Points with too few neighbours get a zero normal; callers are expected to
/// skip those rather than treat zero as a direction.
pub fn estimate(
    points: &[Vector3<f32>],
    radius: f32,
    min_neighbours: usize,
    viewpoint: &Vector3<f32>,
) -> Vec<Vector3<f32>> {
    // NOTE: do not shrink this cell. See the measurement note in
    // `IcpCloud::new`: fewer cells with more points each beats more cells with
    // fewer, because the per-cell hash lookup dominates the per-point work.
    let hash = VoxelHash::build(radius, points);

    points
        .par_iter()
        .map(|point| {
            let mut neighbours = Vec::new();
            hash.indices_within(points, point, radius, &mut neighbours);

            if neighbours.len() < min_neighbours {
                return Vector3::zeros();
            }

            let mut mean = Vector3::zeros();
            for &index in &neighbours {
                mean += points[index as usize];
            }
            mean /= neighbours.len() as f32;

            let mut covariance = Matrix3::<f32>::zeros();
            for &index in &neighbours {
                let offset = points[index as usize] - mean;
                covariance += offset * offset.transpose();
            }
            covariance /= neighbours.len() as f32;

            // The surface normal is the direction of least variance.
            // `SymmetricEigen` does not sort its eigenvalues, so find the minimum.
            let eigen = SymmetricEigen::new(covariance);
            let mut smallest = 0;
            for axis in 1..3 {
                if eigen.eigenvalues[axis] < eigen.eigenvalues[smallest] {
                    smallest = axis;
                }
            }

            let mut normal = Vector3::new(
                eigen.eigenvectors[(0, smallest)],
                eigen.eigenvectors[(1, smallest)],
                eigen.eigenvectors[(2, smallest)],
            );

            let norm = normal.norm();
            if norm <= 0.0 || !norm.is_finite() {
                return Vector3::zeros();
            }
            normal /= norm;

            if normal.dot(&(viewpoint - point)) < 0.0 {
                normal = -normal;
            }

            normal
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat patch in the z = 0 plane, offset so the viewpoint test is meaningful.
    fn plane_points() -> Vec<Vector3<f32>> {
        let mut points = Vec::new();
        for i in 0..30 {
            for j in 0..30 {
                points.push(Vector3::new(
                    -0.3 + 0.02 * i as f32,
                    -0.3 + 0.02 * j as f32,
                    0.0,
                ));
            }
        }
        points
    }

    #[test]
    fn flat_patch_normals_are_plus_or_minus_z() {
        let points = plane_points();
        // Viewpoint above the patch: normals should point at it, i.e. +z.
        let normals = estimate(&points, 0.06, MIN_NEIGHBOURS, &Vector3::new(0.0, 0.0, 1.0));

        let interior: Vec<_> = normals.iter().filter(|n| n.norm_squared() > 0.0).collect();

        assert!(interior.len() > 400, "expected most points to get a normal");

        for normal in &interior {
            assert!(
                (normal.z - 1.0).abs() < 1e-3,
                "expected +z normal, got {normal:?}"
            );
        }
    }

    #[test]
    fn viewpoint_flips_the_sign() {
        let points = plane_points();

        let from_above = estimate(&points, 0.06, MIN_NEIGHBOURS, &Vector3::new(0.0, 0.0, 1.0));
        let from_below = estimate(&points, 0.06, MIN_NEIGHBOURS, &Vector3::new(0.0, 0.0, -1.0));

        // Find a point that got a normal in both.
        for (a, b) in from_above.iter().zip(from_below.iter()) {
            if a.norm_squared() > 0.0 && b.norm_squared() > 0.0 {
                assert!(
                    (a + b).norm() < 1e-3,
                    "expected opposite signs, got {a:?} {b:?}"
                );
                return;
            }
        }

        panic!("no point received a normal in both orientations");
    }

    #[test]
    fn isolated_points_get_no_normal() {
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(0.0, 5.0, 0.0),
        ];

        let normals = estimate(&points, 0.01, MIN_NEIGHBOURS, &Vector3::zeros());
        assert!(normals.iter().all(|n| n.norm_squared() == 0.0));
    }
}
