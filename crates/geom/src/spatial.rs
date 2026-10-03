//! Uniform voxel hash for neighbour queries.

use nalgebra::Vector3;
use std::collections::HashMap;

/// Integer grid coordinate.
pub type Cell = [i32; 3];

/// Which cell of a grid with the given cell size a point falls in.
#[inline]
pub fn cell_of(point: &Vector3<f32>, cell: f32) -> Cell {
    [
        (point.x / cell).floor() as i32,
        (point.y / cell).floor() as i32,
        (point.z / cell).floor() as i32,
    ]
}

/// A uniform voxel hash over a fixed set of points.
///
/// Points are bucketed into cubic cells of `cell` metres. Queries visit only the
/// cells that overlap the query sphere, so a radius search costs roughly the
/// number of points actually in range rather than the size of the cloud.
///
/// Cell size does not affect correctness, only speed: the cell span scanned is
/// `ceil(2 * radius / cell)` per axis. Setting `cell` near the query radius
/// keeps that to a handful of cells.
pub struct VoxelHash {
    cell: f32,
    buckets: HashMap<Cell, Vec<u32>>,
}

impl VoxelHash {
    pub fn build(cell: f32, points: &[Vector3<f32>]) -> Self {
        assert!(cell > 0.0, "voxel hash cell size must be positive");

        let mut buckets: HashMap<Cell, Vec<u32>> = HashMap::new();
        for (index, point) in points.iter().enumerate() {
            buckets.entry(cell_of(point, cell)).or_default().push(index as u32);
        }

        Self { cell, buckets }
    }

    pub fn cell(&self) -> f32 {
        self.cell
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    fn for_each_within(
        &self,
        points: &[Vector3<f32>],
        query: &Vector3<f32>,
        radius: f32,
        mut visit: impl FnMut(u32, f32),
    ) {
        let radius2 = radius * radius;
        let min = cell_of(&(query - Vector3::repeat(radius)), self.cell);
        let max = cell_of(&(query + Vector3::repeat(radius)), self.cell);

        for x in min[0]..=max[0] {
            for y in min[1]..=max[1] {
                for z in min[2]..=max[2] {
                    let Some(indices) = self.buckets.get(&[x, y, z]) else {
                        continue;
                    };

                    for &index in indices {
                        let distance2 = (points[index as usize] - query).norm_squared();
                        if distance2 <= radius2 {
                            visit(index, distance2);
                        }
                    }
                }
            }
        }
    }

    /// Nearest point within `radius`, as `(index, distance)`.
    pub fn nearest_within(
        &self,
        points: &[Vector3<f32>],
        query: &Vector3<f32>,
        radius: f32,
    ) -> Option<(u32, f32)> {
        let mut best: Option<(u32, f32)> = None;

        self.for_each_within(points, query, radius, |index, distance2| {
            if best.is_none_or(|(_, best2)| distance2 < best2) {
                best = Some((index, distance2));
            }
        });

        best.map(|(index, distance2)| (index, distance2.sqrt()))
    }

    /// Indices of every point within `radius`. `out` is cleared first.
    pub fn indices_within(
        &self,
        points: &[Vector3<f32>],
        query: &Vector3<f32>,
        radius: f32,
        out: &mut Vec<u32>,
    ) {
        out.clear();
        self.for_each_within(points, query, radius, |index, _| out.push(index));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_of_handles_negative_coordinates() {
        // Truncation would fold -0.5 and +0.5 into the same cell; floor must not.
        assert_eq!(cell_of(&Vector3::new(-0.5, 0.5, -1.5), 1.0), [-1, 0, -2]);
    }

    #[test]
    fn radius_search_matches_brute_force() {
        let mut state = 12345u64;
        let points: Vec<Vector3<f32>> = (0..2000)
            .map(|_| {
                let mut next = || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / 16_777_216.0 - 0.5
                };
                Vector3::new(next(), next(), next()) * 2.0
            })
            .collect();

        let query = Vector3::new(0.13, -0.27, 0.41);
        let radius = 0.3;

        let hash = VoxelHash::build(radius, &points);

        let mut found = Vec::new();
        hash.indices_within(&points, &query, radius, &mut found);
        found.sort_unstable();

        let mut expected: Vec<u32> = points
            .iter()
            .enumerate()
            .filter(|(_, p)| (*p - query).norm() <= radius)
            .map(|(i, _)| i as u32)
            .collect();
        expected.sort_unstable();

        assert_eq!(found, expected);
    }

    #[test]
    fn nearest_within_matches_brute_force() {
        let points: Vec<Vector3<f32>> = (0..500)
            .map(|i| {
                let t = i as f32 * 0.037;
                Vector3::new(t.sin() * 1.5, (t * 1.7).cos() * 1.5, t * 0.01)
            })
            .collect();

        let hash = VoxelHash::build(0.25, &points);
        let query = Vector3::new(0.4, -0.2, 0.03);

        let (index, distance) = hash.nearest_within(&points, &query, 0.5).expect("nearby point");

        let brute = points
            .iter()
            .map(|p| (*p - query).norm())
            .fold(f32::INFINITY, f32::min);

        assert!((distance - brute).abs() < 1e-5);
        assert!((points[index as usize] - query).norm() - brute < 1e-5);
    }

    #[test]
    fn returns_none_when_nothing_in_range() {
        let points = vec![Vector3::new(10.0, 10.0, 10.0)];
        let hash = VoxelHash::build(0.5, &points);
        assert!(hash.nearest_within(&points, &Vector3::zeros(), 1.0).is_none());
    }
}
