//! Voxel-grid downsampling.

use nalgebra::Vector3;
use std::collections::HashMap;

/// Replace the cloud with one averaged point per occupied cell.
///
/// A depth frame's raw point density falls off with distance *and* grows with
/// the square of the range, so a far wall can outnumber a near object by an
/// order of magnitude. Left alone that skews ICP (the far wall dominates the
/// error) and the TSDF. Downsampling on a fixed grid makes density uniform in
/// world space instead of in image space.
///
/// Output order is sorted, so repeated runs give identical results.
pub fn downsample(points: &[Vector3<f32>], cell: f32) -> Vec<Vector3<f32>> {
    assert!(cell > 0.0, "voxel size must be positive");

    let mut cells: HashMap<[i32; 3], (Vector3<f32>, u32)> = HashMap::new();

    for point in points {
        let key = crate::spatial::cell_of(point, cell);
        let entry = cells.entry(key).or_insert((Vector3::zeros(), 0));
        entry.0 += point;
        entry.1 += 1;
    }

    let mut out: Vec<Vector3<f32>> = cells
        .into_values()
        .map(|(sum, count)| sum / count as f32)
        .collect();

    out.sort_by(|a, b| {
        a.x.total_cmp(&b.x)
            .then(a.y.total_cmp(&b.y))
            .then(a.z.total_cmp(&b.z))
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collinear_points_fold_into_one_cell() {
        let points: Vec<Vector3<f32>> = (0..100)
            .map(|i| Vector3::new(0.001 * i as f32, 0.0, 0.0))
            .collect();

        let out = downsample(&points, 0.1);
        assert_eq!(out.len(), 1, "all points lie in the first 0.1 m cell");
        // Mean of 0.000..0.099 in 0.001 steps is 0.0495.
        assert!((out[0].x - 0.0495).abs() < 1e-6);
    }

    #[test]
    fn separated_points_survive() {
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),
            Vector3::new(0.0, 5.0, 0.0),
        ];

        assert_eq!(downsample(&points, 0.1).len(), 3);
    }

    #[test]
    fn averaging_preserves_the_voxel_centre_of_mass() {
        let points = vec![
            Vector3::new(0.01, 0.01, 0.01),
            Vector3::new(0.02, 0.02, 0.02),
            Vector3::new(10.0, 10.0, 10.0),
        ];

        let out = downsample(&points, 0.5);
        assert_eq!(out.len(), 2);
        // The two near points average to (0.015, 0.015, 0.015) and sort first.
        assert!((out[0].x - 0.015).abs() < 1e-6);
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(downsample(&[], 0.05).is_empty());
    }
}
