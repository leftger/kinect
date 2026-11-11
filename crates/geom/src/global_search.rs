//! Global 4-DOF scan matching: yaw plus translation, branch and bound.
//!
//! A fast correlative matcher is the reference here. The idea is
//! not to replace ICP: it is to supply an initial guess when the drifted pose is
//! too wrong for ICP's local basin. A multi-resolution likelihood grid is built
//! from the target cloud; a branch-and-bound search over yaw and translation
//! prunes whole boxes with an upper bound, then the result is refined by the
//! ordinary point-to-plane ICP and the existing verification gates.
//!
//! The search is deliberately 4-DOF (yaw, x, y, z), not 6-DOF. A handheld
//! scanner is usually upright, and a six-dimensional correlative search is much
//! too expensive for a depth frame. Callers that have a gravity/floor prior can
//! use this directly; callers that do not should leave it off.

use crate::transform_point;
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

/// Tuning for the global search.
#[derive(Clone, Copy, Debug)]
pub struct GlobalSearchParams {
    /// Base likelihood-grid resolution, in metres.
    pub voxel: f32,
    /// Branch-and-bound depth. The leaf box is roughly
    /// `2 * translation_radius / 2^levels` wide.
    pub levels: usize,
    /// Width of the distance-to-surface likelihood kernel, in metres.
    pub sigma: f32,
    /// Translation search half-width, in metres.
    pub translation_radius: f32,
    /// Yaw search half-width, in radians.
    pub yaw_radius: f32,
    /// Coarse yaw step, in radians.
    pub yaw_step: f32,
    /// Cap on query points, for a bounded per-node cost.
    pub max_points: usize,
    /// Cap on expanded branch-and-bound nodes.
    pub max_nodes: usize,
    /// Minimum mean likelihood to return a match.
    pub min_score: f32,
}

impl Default for GlobalSearchParams {
    fn default() -> Self {
        Self {
            voxel: 0.05,
            levels: 4,
            sigma: 0.10,
            translation_radius: 0.45,
            yaw_radius: 0.45,
            yaw_step: 0.17,
            max_points: 350,
            max_nodes: 6_000,
            min_score: 0.35,
        }
    }
}

/// A global match: the source-to-target transform and its mean likelihood.
#[derive(Clone, Copy, Debug)]
pub struct GlobalMatch {
    pub transform: Isometry3<f32>,
    pub score: f32,
}

#[derive(Clone)]
struct Level {
    cell: f32,
    origin: Vector3<f32>,
    dims: [usize; 3],
    values: Vec<f32>,
}

/// A multi-resolution likelihood grid built from a target point cloud.
pub struct ScoreGrid {
    levels: Vec<Level>,
}

impl ScoreGrid {
    /// Build a grid from target points in the target frame.
    ///
    /// Occupied cells score 1; empty cells score a Gaussian of their distance
    /// to the nearest occupied cell. Coarser levels hold the maximum of their
    /// children, which is exactly the upper bound branch-and-bound needs.
    pub fn build(points: &[Vector3<f32>], params: &GlobalSearchParams) -> Option<Self> {
        if points.len() < 8 {
            return None;
        }

        let voxel = params.voxel.max(1e-4);
        let margin =
            params.translation_radius.max(0.0) + params.sigma.max(voxel) * 3.0 + voxel * 2.0;

        let mut min = Vector3::repeat(f32::INFINITY);
        let mut max = Vector3::repeat(f32::NEG_INFINITY);
        for point in points {
            if point.iter().all(|value| value.is_finite()) {
                min = min.inf(point);
                max = max.sup(point);
            }
        }
        if !min.iter().all(|value| value.is_finite()) {
            return None;
        }
        min -= Vector3::repeat(margin);
        max += Vector3::repeat(margin);

        let dims = [
            (((max.x - min.x) / voxel).ceil() as usize).saturating_add(1),
            (((max.y - min.y) / voxel).ceil() as usize).saturating_add(1),
            (((max.z - min.z) / voxel).ceil() as usize).saturating_add(1),
        ];
        let count = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
        // A 4 cm grid over a large room grows quickly; refuse rather than
        // silently allocating gigabytes. The caller can lower `max_points` or
        // use a larger `voxel`.
        if count == 0 || count > 4_000_000 {
            return None;
        }

        let mut occupied = vec![false; count];
        for point in points {
            if let Some(index) = cell_index(&min, voxel, dims, point) {
                occupied[index] = true;
            }
        }
        if !occupied.iter().any(|value| *value) {
            return None;
        }

        // Multi-source BFS gives a 26-neighbour distance transform. Diagonal
        // steps cost one cell rather than sqrt(3), which is fine for a
        // likelihood kernel and keeps the queue cheap.
        let mut distance = vec![u32::MAX; count];
        let mut queue = VecDeque::new();
        for (index, occupied) in occupied.iter().enumerate() {
            if *occupied {
                distance[index] = 0;
                queue.push_back(index as u32);
            }
        }
        while let Some(index) = queue.pop_front() {
            let index = index as usize;
            let current = distance[index];
            let [x, y, z] = unravel(index, dims);
            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if nx < 0
                            || ny < 0
                            || nz < 0
                            || nx >= dims[0] as i32
                            || ny >= dims[1] as i32
                            || nz >= dims[2] as i32
                        {
                            continue;
                        }
                        let neighbour = ravel([nx as usize, ny as usize, nz as usize], dims);
                        if distance[neighbour] == u32::MAX {
                            distance[neighbour] = current.saturating_add(1);
                            queue.push_back(neighbour as u32);
                        }
                    }
                }
            }
        }

        let sigma_cells = (params.sigma.max(voxel) / voxel).max(1.0);
        let mut values = vec![0.0f32; count];
        for (value, distance) in values.iter_mut().zip(&distance) {
            let d = *distance as f32;
            *value = (-0.5 * (d / sigma_cells).powi(2)).exp();
        }

        let base = Level {
            cell: voxel,
            origin: min,
            dims,
            values,
        };
        let mut levels = vec![base];

        for _ in 1..params.levels.max(1) {
            let previous = levels.last().expect("at least the base level");
            let dims = [
                previous.dims[0].div_ceil(2),
                previous.dims[1].div_ceil(2),
                previous.dims[2].div_ceil(2),
            ];
            let count = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
            let mut values = vec![0.0f32; count];
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        let mut best = 0.0f32;
                        for dz in 0..2 {
                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let child = [x * 2 + dx, y * 2 + dy, z * 2 + dz];
                                    if child[0] >= previous.dims[0]
                                        || child[1] >= previous.dims[1]
                                        || child[2] >= previous.dims[2]
                                    {
                                        continue;
                                    }
                                    best = best.max(previous.values[ravel(child, previous.dims)]);
                                }
                            }
                        }
                        values[ravel([x, y, z], dims)] = best;
                    }
                }
            }
            levels.push(Level {
                cell: previous.cell * 2.0,
                origin: previous.origin,
                dims,
                values,
            });
        }

        Some(Self { levels })
    }

    /// Likelihood at a target-frame point. Outside the grid is zero.
    pub fn value(&self, point: &Vector3<f32>) -> f32 {
        let base = &self.levels[0];
        let Some(index) = cell_index(&base.origin, base.cell, base.dims, point) else {
            return 0.0;
        };
        base.values[index]
    }

    /// Upper bound on the likelihood anywhere in a target-frame box.
    fn max_over_box(&self, level: usize, min: &Vector3<f32>, max: &Vector3<f32>) -> f32 {
        let level = &self.levels[level.min(self.levels.len() - 1)];
        let low = [
            ((min.x - level.origin.x) / level.cell).floor() as i32,
            ((min.y - level.origin.y) / level.cell).floor() as i32,
            ((min.z - level.origin.z) / level.cell).floor() as i32,
        ];
        let high = [
            ((max.x - level.origin.x) / level.cell).floor() as i32,
            ((max.y - level.origin.y) / level.cell).floor() as i32,
            ((max.z - level.origin.z) / level.cell).floor() as i32,
        ];

        let mut best = 0.0f32;
        for z in low[2].max(0)..=high[2].min(level.dims[2] as i32 - 1) {
            for y in low[1].max(0)..=high[1].min(level.dims[1] as i32 - 1) {
                for x in low[0].max(0)..=high[0].min(level.dims[0] as i32 - 1) {
                    let index = ravel([x as usize, y as usize, z as usize], level.dims);
                    best = best.max(level.values[index]);
                }
            }
        }
        best
    }

    /// Mean likelihood of source points under a source-to-target transform.
    pub fn score_transform(&self, points: &[Vector3<f32>], transform: &Isometry3<f32>) -> f32 {
        if points.is_empty() {
            return 0.0;
        }
        let sum: f32 = points
            .iter()
            .map(|point| self.value(&transform_point(transform, point)))
            .sum();
        sum / points.len() as f32
    }
}

fn cell_index(
    origin: &Vector3<f32>,
    cell: f32,
    dims: [usize; 3],
    point: &Vector3<f32>,
) -> Option<usize> {
    if !point.iter().all(|value| value.is_finite()) {
        return None;
    }
    let x = ((point.x - origin.x) / cell).floor();
    let y = ((point.y - origin.y) / cell).floor();
    let z = ((point.z - origin.z) / cell).floor();
    if x < 0.0 || y < 0.0 || z < 0.0 {
        return None;
    }
    let x = x as usize;
    let y = y as usize;
    let z = z as usize;
    if x >= dims[0] || y >= dims[1] || z >= dims[2] {
        return None;
    }
    Some(ravel([x, y, z], dims))
}

fn ravel([x, y, z]: [usize; 3], dims: [usize; 3]) -> usize {
    (z * dims[1] + y) * dims[0] + x
}

fn unravel(index: usize, dims: [usize; 3]) -> [usize; 3] {
    let x = index % dims[0];
    let y = (index / dims[0]) % dims[1];
    let z = index / (dims[0] * dims[1]);
    [x, y, z]
}

#[derive(Clone, Copy)]
struct SearchNode {
    min: Vector3<f32>,
    max: Vector3<f32>,
    bound: f32,
    depth: usize,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.bound == other.bound
    }
}
impl Eq for SearchNode {}
impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound.total_cmp(&other.bound)
    }
}

/// Search for the yaw-plus-translation correction that best explains `source`
/// under `prior`, scored against `target`.
///
/// Returns `None` when the grid cannot be built, the query is too small, or the
/// best score is below `min_score`. The caller is expected to refine the result
/// with ICP and apply its own verification gates.
pub fn search(
    target: &[Vector3<f32>],
    source: &[Vector3<f32>],
    prior: Isometry3<f32>,
    params: &GlobalSearchParams,
) -> Option<GlobalMatch> {
    let grid = ScoreGrid::build(target, params)?;
    let query = sample(source, params.max_points);
    if query.len() < 8 {
        return None;
    }

    let mut best = GlobalMatch {
        transform: prior,
        score: grid.score_transform(&query, &prior),
    };
    let mut best_yaw = 0.0f32;

    let coarse_steps = (params.yaw_radius / params.yaw_step.max(1e-3)).floor() as i32;
    for step in -coarse_steps..=coarse_steps {
        let yaw = step as f32 * params.yaw_step;
        let yawed = yaw_query(&query, &prior, yaw);
        let (offset, score) = branch_and_bound(&grid, &yawed, params);
        if score > best.score {
            best.score = score;
            best.transform = yaw_transform(&prior, yaw, offset);
            best_yaw = yaw;
        }
    }

    // Refine around the best coarse yaw. The coarse pass is what makes the
    // search affordable; this pass is what makes the result accurate enough for
    // ICP to take over.
    let fine_step = params.yaw_step / 5.0;
    for step in -5..=5 {
        let yaw = best_yaw + step as f32 * fine_step;
        let yawed = yaw_query(&query, &prior, yaw);
        let (offset, score) = branch_and_bound(&grid, &yawed, params);
        if score > best.score {
            best.score = score;
            best.transform = yaw_transform(&prior, yaw, offset);
        }
    }

    if best.score >= params.min_score {
        Some(best)
    } else {
        None
    }
}

fn sample(points: &[Vector3<f32>], max_points: usize) -> Vec<Vector3<f32>> {
    if points.len() <= max_points.max(1) {
        return points.to_vec();
    }
    let stride = points.len().div_ceil(max_points.max(1));
    points.iter().step_by(stride.max(1)).copied().collect()
}

fn yaw_query(query: &[Vector3<f32>], prior: &Isometry3<f32>, yaw: f32) -> Vec<Vector3<f32>> {
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
    let center = prior.translation.vector;
    query
        .iter()
        .map(|point| {
            let world = transform_point(prior, point);
            center + rotation * (world - center)
        })
        .collect()
}

fn yaw_transform(prior: &Isometry3<f32>, yaw: f32, offset: Vector3<f32>) -> Isometry3<f32> {
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
    let center = prior.translation.vector;
    Isometry3::from_parts(Translation3::from(center + offset), rotation)
        * Isometry3::from_parts(Translation3::from(-center), UnitQuaternion::identity())
        * prior
}

fn branch_and_bound(
    grid: &ScoreGrid,
    query: &[Vector3<f32>],
    params: &GlobalSearchParams,
) -> (Vector3<f32>, f32) {
    let radius = params.translation_radius.max(0.0);
    if radius <= 0.0 || query.is_empty() {
        return (
            Vector3::zeros(),
            grid.score_transform(query, &Isometry3::identity()),
        );
    }

    let root = SearchNode {
        min: Vector3::repeat(-radius),
        max: Vector3::repeat(radius),
        bound: 1.0,
        depth: 0,
    };
    let mut heap = BinaryHeap::new();
    heap.push(root);

    let mut best_offset = Vector3::zeros();
    let mut best_score = 0.0f32;
    let mut expanded = 0usize;

    while let Some(node) = heap.pop() {
        if node.bound <= best_score {
            break;
        }
        expanded += 1;
        if expanded > params.max_nodes {
            break;
        }

        let side = (node.max.x - node.min.x)
            .max(node.max.y - node.min.y)
            .max(node.max.z - node.min.z);
        let leaf = node.depth >= params.levels.max(1) || side <= params.voxel * 0.5;

        if leaf {
            let center = (node.min + node.max) * 0.5;
            let score = mean_score(grid, query, &center);
            if score > best_score {
                best_score = score;
                best_offset = center;
            }
            continue;
        }

        let mid = (node.min + node.max) * 0.5;
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let min = Vector3::new(
                        if x == 0 { node.min.x } else { mid.x },
                        if y == 0 { node.min.y } else { mid.y },
                        if z == 0 { node.min.z } else { mid.z },
                    );
                    let max = Vector3::new(
                        if x == 0 { mid.x } else { node.max.x },
                        if y == 0 { mid.y } else { node.max.y },
                        if z == 0 { mid.z } else { node.max.z },
                    );
                    let bound = box_bound(grid, query, &min, &max, params.voxel);
                    if bound > best_score {
                        heap.push(SearchNode {
                            min,
                            max,
                            bound,
                            depth: node.depth + 1,
                        });
                    }
                }
            }
        }
    }

    (best_offset, best_score)
}

fn mean_score(grid: &ScoreGrid, query: &[Vector3<f32>], offset: &Vector3<f32>) -> f32 {
    let sum: f32 = query
        .iter()
        .map(|point| grid.value(&(point + offset)))
        .sum();
    sum / query.len() as f32
}

fn box_bound(
    grid: &ScoreGrid,
    query: &[Vector3<f32>],
    min: &Vector3<f32>,
    max: &Vector3<f32>,
    voxel: f32,
) -> f32 {
    let side = (max.x - min.x).max(max.y - min.y).max(max.z - min.z);
    let relative = (side / voxel.max(1e-4)).max(1.0).log2().ceil() as usize;
    let level = relative.min(grid.levels.len() - 1);

    let sum: f32 = query
        .iter()
        .map(|point| grid.max_over_box(level, &(point + min), &(point + max)))
        .sum();
    sum / query.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn small_params() -> GlobalSearchParams {
        GlobalSearchParams {
            voxel: 0.05,
            levels: 4,
            sigma: 0.10,
            translation_radius: 0.30,
            yaw_radius: 0.30,
            yaw_step: 0.10,
            max_points: 300,
            max_nodes: 6_000,
            min_score: 0.30,
        }
    }

    #[test]
    fn the_likelihood_grid_peaks_on_the_surface() {
        let points = wavy_surface();
        let grid = ScoreGrid::build(&points, &small_params()).expect("grid");
        let on_surface = Vector3::new(0.0, 0.0, 2.0 + 0.06 * 0.0);
        let empty = Vector3::new(0.0, 0.0, 3.5);
        assert!(grid.value(&on_surface) > 0.9, "{}", grid.value(&on_surface));
        assert!(grid.value(&empty) < grid.value(&on_surface));
    }

    #[test]
    fn global_search_recovers_a_known_yaw_and_translation() {
        let target = wavy_surface();
        let truth = Isometry3::from_parts(
            Translation3::new(0.12, -0.05, 0.03),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.12),
        );
        let source: Vec<Vector3<f32>> = target
            .iter()
            .map(|point| transform_point(&truth.inverse(), point))
            .collect();

        let found =
            search(&target, &source, Isometry3::identity(), &small_params()).expect("a match");
        assert!(found.score > 0.5, "score {:.3}", found.score);

        let delta = truth.inverse() * found.transform;
        assert!(
            delta.translation.vector.norm() < 0.08,
            "translation error {:.3} m",
            delta.translation.vector.norm()
        );
        assert!(
            delta.rotation.angle() < 0.12,
            "rotation error {:.3} rad",
            delta.rotation.angle()
        );
    }

    #[test]
    fn global_search_rejects_an_unrelated_view() {
        let target = wavy_surface();
        let source: Vec<Vector3<f32>> = target
            .iter()
            .map(|point| point + Vector3::new(0.0, 0.0, 3.0))
            .collect();
        assert!(
            search(&target, &source, Isometry3::identity(), &small_params()).is_none(),
            "a disjoint surface produced a match"
        );
    }
}
