//! Pose-graph optimisation, for closing loops.
//!
//! Frame-to-frame odometry has no way to notice that it has returned somewhere it
//! has already been, so drift accumulates without bound: the 150-frame handheld
//! capture in this repo reconstructs 12 m of room. The fix is to detect those
//! revisits (loop closure, which lives in the scanner) and then *redistribute* the
//! accumulated error over the whole trajectory rather than leaving it piled up at
//! the end. This module is the redistribution.
//!
//! # The model
//!
//! Nodes are camera poses `T_i` (world-from-camera). Edges are relative-pose
//! measurements `Z_ij = T_i^-1 T_j`: an odometry edge between consecutive
//! keyframes, or a loop-closure edge between two keyframes that were recognised as
//! the same place. Each edge contributes a residual
//!
//! ```text
//! E_ij = Z_ij^-1 (T_i^-1 T_j)      e_ij = [ E.translation ; log(E.rotation) ]
//! ```
//!
//! and the optimiser minimises the weighted sum of `|e_ij|^2`.
//!
//! # The residual approximation
//!
//! **The residual is not the exact SE(3) log map.** It stacks the translation and
//! the rotation vector rather than applying the coupling `V^-1` factor. The
//! difference vanishes as the residual goes to zero, which is where an optimiser
//! ends up, and it removes the small-angle special cases that are easy to get
//! subtly wrong.
//!
//! # Sparse normal equations
//!
//! The Jacobians are analytic 6x6 blocks. Each edge touches only two nodes, so
//! the normal equations are assembled as a block-sparse system and solved with
//! block-SSOR-preconditioned conjugate gradients. A long scan therefore does not
//! allocate an `(6N x 6N)` dense matrix, and the per-edge work is a handful of
//! fixed-size products. The rotation blocks use the first-order SO(3) Jacobian,
//! consistent with the small residuals a verified graph converges to; a
//! finite-difference Jacobian remains in the tests as a cross-check.
//!
//! # What makes it safe
//!
//! A pose graph happily folds the map in half to satisfy a single wrong loop
//! edge, so the *caller* must verify candidates geometrically before adding them
//! (see the scanner: ICP fitness thresholds). On top of that, a Huber kernel here
//! stops any one edge that survives verification from dominating.

use nalgebra::{Isometry3, Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};
use std::collections::HashMap;

/// An observed relative pose between two nodes.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    /// `T_from^-1 * T_to` as measured.
    pub measurement: Isometry3<f32>,
    /// Relative confidence. Loop closures are normally weighted below odometry,
    /// because they come from matching a candidate that might not be the same
    /// place at all.
    pub weight: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct PoseGraphParams {
    pub iterations: usize,
    /// Stop once the largest node update falls below this.
    pub convergence_epsilon: f32,
    /// Huber threshold on each edge's residual norm, in metres-ish. Zero disables
    /// the robust kernel and makes this plain least squares.
    pub robust_delta: f32,
    /// Levenberg damping, so the normal equations stay solvable when the graph is
    /// weakly constrained.
    pub damping: f32,
    /// Relative residual tolerance for the sparse conjugate-gradient solve.
    pub cg_tolerance: f32,
    /// Cap on conjugate-gradient iterations per Gauss-Newton step.
    pub cg_max_iterations: usize,
}

impl Default for PoseGraphParams {
    fn default() -> Self {
        Self {
            iterations: 30,
            convergence_epsilon: 1e-5,
            // Errors past this are treated as outliers rather than as a large
            // correction to follow.
            robust_delta: 0.5,
            damping: 1e-6,
            cg_tolerance: 1e-8,
            cg_max_iterations: 200,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OptimizationReport {
    pub cost_before: f64,
    pub cost_after: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Total conjugate-gradient iterations spent across all Gauss-Newton steps.
    pub linear_solver_iterations: usize,
}

/// A trajectory of poses plus the measurements that constrain them.
pub struct PoseGraph {
    nodes: Vec<Isometry3<f32>>,
    edges: Vec<Edge>,
}

impl PoseGraph {
    /// Start a graph with its first node; node 0 anchors the map.
    pub fn new(first: Isometry3<f32>) -> Self {
        Self {
            nodes: vec![first],
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, pose: Isometry3<f32>) -> usize {
        self.nodes.push(pose);
        self.nodes.len() - 1
    }

    /// Add a measurement. `measurement` must be the observed motion *from* `from`
    /// *to* `to`, i.e. `T_from^-1 T_to`.
    pub fn add_edge(&mut self, from: usize, to: usize, measurement: Isometry3<f32>, weight: f32) {
        assert!(from < self.nodes.len(), "edge references a missing node");
        assert!(to < self.nodes.len(), "edge references a missing node");
        assert_ne!(from, to, "an edge from a node to itself constrains nothing");

        self.edges.push(Edge {
            from,
            to,
            measurement,
            weight,
        });
    }

    pub fn nodes(&self) -> &[Isometry3<f32>] {
        &self.nodes
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn set_pose(&mut self, node: usize, pose: Isometry3<f32>) {
        self.nodes[node] = pose;
    }

    /// Residual of one edge under the current poses.
    pub fn edge_error(&self, edge: &Edge) -> Vector6<f32> {
        error_between(
            &edge.measurement,
            &self.nodes[edge.from],
            &self.nodes[edge.to],
        )
    }

    /// Weighted sum of squared residuals.
    pub fn cost(&self) -> f64 {
        self.edges
            .iter()
            .map(|edge| {
                let error = self.edge_error(edge);
                edge.weight as f64 * error.norm_squared() as f64
            })
            .sum()
    }

    /// Largest residual over the loop-closure edges only.
    ///
    /// Odometry edges are expected to carry some error after optimisation -- that
    /// is the whole point, the drift gets spread across them -- so the loop edges
    /// are the meaningful check on whether the map came back together.
    pub fn largest_loop_error(&self, odometry_edge_count: usize) -> f32 {
        self.edges
            .iter()
            .skip(odometry_edge_count)
            .map(|edge| self.edge_error(edge).norm())
            .fold(0.0, f32::max)
    }

    /// Gauss-Newton over the whole graph, in place.
    pub fn optimize(&mut self, params: &PoseGraphParams) -> OptimizationReport {
        let node_count = self.nodes.len();
        let cost_before = self.cost();

        if node_count < 2 || self.edges.is_empty() {
            return OptimizationReport {
                cost_before,
                cost_after: cost_before,
                iterations: 0,
                converged: true,
                linear_solver_iterations: 0,
            };
        }

        // Node 0 is the anchor. The cost is invariant to a global rigid motion of
        // the whole trajectory, so without fixing something the normal equations
        // are singular in six directions and the solve is arbitrary.
        let mut converged = false;
        let mut iterations = 0;
        let mut linear_solver_iterations = 0;

        for iteration in 0..params.iterations {
            iterations = iteration + 1;

            // Block-sparse normal equations: one 6x6 block per (row node,
            // column node). An edge touches only two nodes, so the assembly is
            // linear in the edge count and never allocates an N x N matrix.
            let mut blocks: Vec<HashMap<usize, Matrix6<f32>>> = vec![HashMap::new(); node_count];
            let mut gradient = vec![Vector6::<f32>::zeros(); node_count];

            for edge in &self.edges {
                let (j_from, j_to, error) = self.linearise(edge);

                // Huber: quadratic near zero, linear in the tails, so one edge that
                // survived verification but is still wrong cannot drag the map.
                let mut weight = edge.weight;
                let norm = error.norm();
                if params.robust_delta > 0.0 && norm > params.robust_delta {
                    weight *= params.robust_delta / norm;
                }

                let h_ff = weight * (j_from.transpose() * j_from);
                let h_ft = weight * (j_from.transpose() * j_to);
                let h_tf = weight * (j_to.transpose() * j_from);
                let h_tt = weight * (j_to.transpose() * j_to);

                add_block(&mut blocks, edge.from, edge.from, h_ff);
                add_block(&mut blocks, edge.from, edge.to, h_ft);
                add_block(&mut blocks, edge.to, edge.from, h_tf);
                add_block(&mut blocks, edge.to, edge.to, h_tt);

                gradient[edge.from] += weight * (j_from.transpose() * error);
                gradient[edge.to] += weight * (j_to.transpose() * error);
            }

            for node in 1..node_count {
                add_block(
                    &mut blocks,
                    node,
                    node,
                    params.damping * Matrix6::<f32>::identity(),
                );
            }
            gradient[0] = Vector6::zeros();

            let (step, cg_iterations) = sparse_solve(&blocks, &gradient, params);
            linear_solver_iterations += cg_iterations;

            let mut largest_update = 0.0f32;
            for (node, delta) in step.iter().enumerate().skip(1) {
                let translation = Vector3::new(delta[0], delta[1], delta[2]);
                let rotation = Vector3::new(delta[3], delta[4], delta[5]);

                // Right-multiply: the increment is expressed in the node's own
                // frame, which is what the Jacobian perturbed.
                self.nodes[node] *= Isometry3::from_parts(
                    Translation3::from(translation),
                    UnitQuaternion::from_scaled_axis(rotation),
                );

                largest_update = largest_update.max(translation.norm().max(rotation.norm()));
            }

            if largest_update < params.convergence_epsilon {
                converged = true;
                break;
            }
        }

        OptimizationReport {
            cost_before,
            cost_after: self.cost(),
            iterations,
            converged,
            linear_solver_iterations,
        }
    }

    /// Analytic Jacobians of one edge's residual with respect to its two nodes.
    ///
    /// The perturbation is right-multiplied on each node, matching the update in
    /// `optimize`. The rotation blocks use the first-order SO(3) Jacobian, which
    /// is accurate where a verified graph converges; the tests compare this
    /// against central differences.
    fn linearise(&self, edge: &Edge) -> (Matrix6<f32>, Matrix6<f32>, Vector6<f32>) {
        let from = self.nodes[edge.from];
        let to = self.nodes[edge.to];
        let relative = from.inverse() * to;
        let error_pose = edge.measurement.inverse() * relative;
        let error = error_vector(&error_pose);

        let t = error_pose.translation.vector;
        let r = error_pose.rotation.scaled_axis();

        // Left and right first-order SO(3) Jacobian inverses.
        let j_left_rot = Matrix3::<f32>::identity() + 0.5 * skew(&r);
        let j_right_rot = Matrix3::<f32>::identity() - 0.5 * skew(&r);

        let mut j_left = Matrix6::<f32>::zeros();
        j_left
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::<f32>::identity());
        j_left.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-skew(&t)));
        j_left.fixed_view_mut::<3, 3>(3, 3).copy_from(&j_left_rot);

        let rotation = error_pose.rotation.to_rotation_matrix();
        let mut j_right = Matrix6::<f32>::zeros();
        j_right
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(rotation.matrix());
        j_right.fixed_view_mut::<3, 3>(3, 3).copy_from(&j_right_rot);

        // The residual lives in the measurement frame, so a perturbation of the
        // "from" node is conjugated by the measurement's inverse adjoint.
        let j_from = -(j_left * adjoint_inverse(&edge.measurement));
        (j_from, j_right, error)
    }

    /// Finite-difference Jacobians, retained as the analytic check in tests.
    #[cfg(test)]
    fn linearise_finite(
        &self,
        edge: &Edge,
        step: f32,
    ) -> (Matrix6<f32>, Matrix6<f32>, Vector6<f32>) {
        let from = self.nodes[edge.from];
        let to = self.nodes[edge.to];
        let base = error_between(&edge.measurement, &from, &to);

        let mut j_from = Matrix6::<f32>::zeros();
        let mut j_to = Matrix6::<f32>::zeros();

        for k in 0..6 {
            let bump = twist(k, step);
            let back = twist(k, -step);

            let plus = error_between(&edge.measurement, &(from * bump), &to);
            let minus = error_between(&edge.measurement, &(from * back), &to);
            j_from.set_column(k, &((plus - minus) / (2.0 * step)));

            let plus = error_between(&edge.measurement, &from, &(to * bump));
            let minus = error_between(&edge.measurement, &from, &(to * back));
            j_to.set_column(k, &((plus - minus) / (2.0 * step)));
        }

        (j_from, j_to, base)
    }
}

fn add_block(
    blocks: &mut [HashMap<usize, Matrix6<f32>>],
    row: usize,
    column: usize,
    block: Matrix6<f32>,
) {
    *blocks[row]
        .entry(column)
        .or_insert_with(Matrix6::<f32>::zeros) += block;
}

fn matvec(blocks: &[HashMap<usize, Matrix6<f32>>], vector: &[Vector6<f32>]) -> Vec<Vector6<f32>> {
    let mut result = vec![Vector6::<f32>::zeros(); vector.len()];
    for row in 1..vector.len() {
        let mut sum = Vector6::<f32>::zeros();
        for (column, block) in &blocks[row] {
            if *column == 0 {
                continue;
            }
            sum += block * vector[*column];
        }
        result[row] = sum;
    }
    result
}

fn precondition(
    blocks: &[HashMap<usize, Matrix6<f32>>],
    diagonal: &[Matrix6<f32>],
    diagonal_inverse: &[Matrix6<f32>],
    residual: &[Vector6<f32>],
) -> Vec<Vector6<f32>> {
    let nodes = residual.len();
    let mut forward = vec![Vector6::<f32>::zeros(); nodes];

    // Block forward substitution: (D + L) y = r.
    for row in 1..nodes {
        let mut sum = residual[row];
        for (column, block) in &blocks[row] {
            if *column < row && *column != 0 {
                sum -= block * forward[*column];
            }
        }
        forward[row] = diagonal_inverse[row] * sum;
    }

    // Scale by D, then block backward substitution: (D + L)^T z = D y.
    let mut scaled = vec![Vector6::<f32>::zeros(); nodes];
    for row in 1..nodes {
        scaled[row] = diagonal[row] * forward[row];
    }
    let mut result = vec![Vector6::<f32>::zeros(); nodes];
    for row in (1..nodes).rev() {
        let mut sum = scaled[row];
        for (column, block) in &blocks[row] {
            if *column > row {
                sum -= block.transpose() * result[*column];
            }
        }
        result[row] = diagonal_inverse[row] * sum;
    }
    result
}

fn dot(left: &[Vector6<f32>], right: &[Vector6<f32>]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.dot(right))
        .sum()
}

/// Block-SSOR-preconditioned conjugate gradients over the block-sparse normal
/// equations. Node 0 is the anchor, so its row and column are skipped.
fn sparse_solve(
    blocks: &[HashMap<usize, Matrix6<f32>>],
    gradient: &[Vector6<f32>],
    params: &PoseGraphParams,
) -> (Vec<Vector6<f32>>, usize) {
    let nodes = blocks.len();
    let mut step = vec![Vector6::<f32>::zeros(); nodes];
    if nodes <= 1 {
        return (step, 0);
    }

    let diagonal: Vec<Matrix6<f32>> = (0..nodes)
        .map(|node| {
            blocks[node]
                .get(&node)
                .copied()
                .unwrap_or_else(Matrix6::<f32>::identity)
        })
        .collect();
    let diagonal_inverse: Vec<Matrix6<f32>> = diagonal
        .iter()
        .map(|block| block.try_inverse().unwrap_or_else(Matrix6::<f32>::identity))
        .collect();

    let mut residual: Vec<Vector6<f32>> = gradient.iter().map(|value| -*value).collect();
    residual[0] = Vector6::zeros();
    let mut preconditioned = precondition(blocks, &diagonal, &diagonal_inverse, &residual);
    let mut direction = preconditioned.clone();
    let mut rz = dot(&residual, &preconditioned);
    let initial_norm = dot(&residual, &residual).sqrt();

    if initial_norm <= params.cg_tolerance {
        return (step, 0);
    }

    let mut iterations = 0;
    for _ in 0..params.cg_max_iterations {
        iterations += 1;
        let applied = matvec(blocks, &direction);
        let denominator = dot(&direction, &applied);
        if !denominator.is_finite() || denominator <= 0.0 {
            break;
        }

        let alpha = rz / denominator;
        for node in 1..nodes {
            step[node] += alpha * direction[node];
            residual[node] -= alpha * applied[node];
        }

        if dot(&residual, &residual).sqrt() <= params.cg_tolerance * initial_norm.max(1.0) {
            break;
        }

        preconditioned = precondition(blocks, &diagonal, &diagonal_inverse, &residual);
        let next_rz = dot(&residual, &preconditioned);
        if !next_rz.is_finite() || next_rz <= 0.0 {
            break;
        }
        let beta = next_rz / rz;
        for node in 1..nodes {
            direction[node] = preconditioned[node] + beta * direction[node];
        }
        rz = next_rz;
    }

    (step, iterations)
}

#[inline]
fn skew(vector: &Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(
        0.0, -vector.z, vector.y, vector.z, 0.0, -vector.x, -vector.y, vector.x, 0.0,
    )
}

/// Adjoint of the inverse of a rigid transform, mapping a twist in the
/// transform's child frame into the parent frame.
fn adjoint_inverse(transform: &Isometry3<f32>) -> Matrix6<f32> {
    let rotation = transform.rotation.to_rotation_matrix();
    let rotation_t = rotation.matrix().transpose();
    let translation = transform.translation.vector;

    let mut adjoint = Matrix6::<f32>::zeros();
    adjoint.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation_t);
    adjoint
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation_t * skew(&translation)));
    adjoint.fixed_view_mut::<3, 3>(3, 3).copy_from(&rotation_t);
    adjoint
}

#[inline]
fn error_vector(error: &Isometry3<f32>) -> Vector6<f32> {
    let translation = error.translation.vector;
    let rotation = error.rotation.scaled_axis();
    Vector6::new(
        translation.x,
        translation.y,
        translation.z,
        rotation.x,
        rotation.y,
        rotation.z,
    )
}

/// Residual of the edge `measurement` under the two poses given.
#[inline]
fn error_between(
    measurement: &Isometry3<f32>,
    from: &Isometry3<f32>,
    to: &Isometry3<f32>,
) -> Vector6<f32> {
    error_vector(&(measurement.inverse() * (from.inverse() * to)))
}

/// A unit-magnitude twist: translation for the first three directions, rotation
/// for the last three. This is the basis the finite-difference test perturbation
/// lives in.
#[cfg(test)]
#[inline]
fn twist(index: usize, magnitude: f32) -> Isometry3<f32> {
    let mut translation = Vector3::zeros();
    let mut rotation = Vector3::zeros();

    if index < 3 {
        translation[index] = magnitude;
    } else {
        rotation[index - 3] = magnitude;
    }

    Isometry3::from_parts(
        Translation3::from(translation),
        UnitQuaternion::from_scaled_axis(rotation),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translate(x: f32, y: f32) -> Isometry3<f32> {
        Isometry3::from_parts(Translation3::new(x, y, 0.0), UnitQuaternion::identity())
    }

    fn relative(from: &Isometry3<f32>, to: &Isometry3<f32>) -> Isometry3<f32> {
        from.inverse() * *to
    }

    #[test]
    fn a_graph_that_already_agrees_is_left_alone() {
        let truth = [
            translate(0.0, 0.0),
            translate(1.0, 0.0),
            translate(1.0, 1.0),
            translate(0.0, 1.0),
        ];

        let mut graph = PoseGraph::new(truth[0]);
        for pose in &truth[1..] {
            graph.add_node(*pose);
        }
        for i in 0..3 {
            graph.add_edge(i, i + 1, relative(&truth[i], &truth[i + 1]), 1.0);
        }
        graph.add_edge(3, 0, relative(&truth[3], &truth[0]), 1.0);

        let report = graph.optimize(&PoseGraphParams::default());

        assert!(
            report.cost_after <= report.cost_before + 1e-6,
            "cost rose from {} to {}",
            report.cost_before,
            report.cost_after
        );
        assert!(report.cost_after < 1e-6, "residuals should be ~zero");

        for (original, optimised) in truth.iter().zip(graph.nodes()) {
            assert!(
                (original.translation.vector - optimised.translation.vector).norm() < 1e-3,
                "a consistent graph moved: {original:?} -> {optimised:?}"
            );
        }
    }

    #[test]
    fn a_loop_edge_pulls_a_drifted_path_back_together() {
        // Odometry for a square walk that returns to its start, but every step is
        // slightly wrong, so integrating it leaves the sensor away from where it
        // actually is. This is exactly what the real 150-frame capture does.
        let drifted_steps = [
            translate(1.05, 0.02),
            translate(0.03, 0.95),
            translate(-0.98, 0.04),
            translate(0.02, -1.03),
        ];

        let mut graph = PoseGraph::new(Isometry3::identity());
        let mut pose = Isometry3::identity();
        for step in &drifted_steps {
            pose *= *step;
            graph.add_node(pose);
        }
        for (i, step) in drifted_steps.iter().enumerate() {
            graph.add_edge(i, i + 1, *step, 1.0);
        }

        // The loop closure: this place *is* the start, so the measured relative
        // pose between the last keyframe and the first is the identity. The graph
        // currently disagrees by the accumulated drift.
        graph.add_edge(4, 0, Isometry3::identity(), 1.0);

        let odometry_edges = 4;
        let error_before = graph.largest_loop_error(odometry_edges);
        let report = graph.optimize(&PoseGraphParams::default());
        let error_after = graph.largest_loop_error(odometry_edges);

        assert!(
            error_before > 0.1,
            "the test needs real drift to correct, got {error_before}"
        );
        assert!(
            error_after < error_before * 0.5,
            "loop barely closed: {error_before:.4} -> {error_after:.4}"
        );
        assert!(
            report.cost_after < report.cost_before * 0.5,
            "cost barely improved: {} -> {}",
            report.cost_before,
            report.cost_after
        );

        // The error must be shared out, not dumped on one edge: that is the
        // difference between a pose graph and simply teleporting the last node.
        let last = graph.nodes()[4].translation.vector;
        assert!(
            last.norm() > 1e-3,
            "the whole correction landed on the final node"
        );
    }

    #[test]
    fn the_anchor_node_never_moves() {
        let mut graph = PoseGraph::new(translate(3.0, -2.0));
        graph.add_node(translate(4.0, -2.0));
        graph.add_node(translate(4.0, -1.0));

        // Deliberately inconsistent measurements.
        graph.add_edge(0, 1, translate(2.0, 0.0), 1.0);
        graph.add_edge(1, 2, translate(0.0, 1.0), 1.0);
        graph.add_edge(2, 0, translate(-1.0, -1.0), 1.0);

        let anchor = graph.nodes()[0];
        graph.optimize(&PoseGraphParams::default());

        assert_eq!(
            graph.nodes()[0],
            anchor,
            "the anchor moved, so the gauge is not fixed"
        );
    }

    #[test]
    fn a_wildly_inconsistent_edge_is_down_weighted_by_the_robust_kernel() {
        // Three consistent nodes plus one loop closure that is a lie. Without the
        // robust kernel the lie would be treated as a large correction to follow.
        let mut graph = PoseGraph::new(Isometry3::identity());
        graph.add_node(translate(1.0, 0.0));
        graph.add_node(translate(2.0, 0.0));

        graph.add_edge(0, 1, translate(1.0, 0.0), 1.0);
        graph.add_edge(1, 2, translate(1.0, 0.0), 1.0);
        // Claims node 2 is 50 m away from node 0, when it is 2 m away.
        graph.add_edge(2, 0, translate(50.0, 0.0), 1.0);

        graph.optimize(&PoseGraphParams::default());

        // The honest chain should still be recognisably a chain: node 2 must not
        // have been flung 25 m to split the difference.
        let end = graph.nodes()[2].translation.vector - graph.nodes()[0].translation.vector;
        assert!(
            end.norm() < 10.0,
            "the outlier dragged the map to {} m",
            end.norm()
        );
    }

    #[test]
    fn an_empty_or_single_node_graph_is_not_an_error() {
        let mut graph = PoseGraph::new(Isometry3::identity());
        let report = graph.optimize(&PoseGraphParams::default());
        assert_eq!(report.iterations, 0);
        assert_eq!(report.cost_after, 0.0);
    }

    #[test]
    fn analytic_jacobians_match_finite_differences() {
        let from = Isometry3::from_parts(
            Translation3::new(0.3, -0.2, 0.1),
            UnitQuaternion::from_euler_angles(0.2, -0.1, 0.15),
        );
        let to = Isometry3::from_parts(
            Translation3::new(1.1, 0.4, -0.2),
            UnitQuaternion::from_euler_angles(-0.1, 0.25, 0.05),
        );
        let truth = from.inverse() * to;
        let measurement = truth
            * Isometry3::from_parts(
                Translation3::new(0.01, -0.005, 0.02),
                UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.015)),
            );

        let mut graph = PoseGraph::new(from);
        graph.add_node(to);
        graph.add_edge(0, 1, measurement, 1.0);
        let edge = graph.edges()[0];

        let (analytic_from, analytic_to, _) = graph.linearise(&edge);
        let (finite_from, finite_to, _) = graph.linearise_finite(&edge, 1e-5);

        for row in 0..6 {
            for column in 0..6 {
                let from_difference =
                    (analytic_from[(row, column)] - finite_from[(row, column)]).abs();
                let to_difference = (analytic_to[(row, column)] - finite_to[(row, column)]).abs();
                assert!(
                    from_difference < 0.05,
                    "from Jacobian ({row},{column}) differs by {from_difference}: {} vs {}",
                    analytic_from[(row, column)],
                    finite_from[(row, column)]
                );
                assert!(
                    to_difference < 0.05,
                    "to Jacobian ({row},{column}) differs by {to_difference}: {} vs {}",
                    analytic_to[(row, column)],
                    finite_to[(row, column)]
                );
            }
        }
    }

    #[test]
    fn a_long_chain_uses_the_sparse_solver() {
        let mut graph = PoseGraph::new(Isometry3::identity());
        let mut pose = Isometry3::identity();
        let step = translate(0.1, 0.002);
        for _ in 0..120 {
            pose *= step;
            graph.add_node(pose);
        }
        for index in 0..120 {
            graph.add_edge(index, index + 1, step, 1.0);
        }
        // The chain has accumulated 0.24 m of lateral drift; the loop says the
        // return is straight, so the graph has to redistribute it.
        graph.add_edge(120, 0, translate(-12.0, 0.0), 1.0);

        // A deliberately small budget keeps this a solver regression test rather
        // than a benchmark. The full default budget is exercised by the real
        // scans, where the graph is much shorter.
        let report = graph.optimize(&PoseGraphParams {
            iterations: 8,
            convergence_epsilon: 1e-4,
            cg_tolerance: 1e-6,
            cg_max_iterations: 60,
            ..PoseGraphParams::default()
        });
        assert!(report.cost_after < report.cost_before);
        assert!(report.linear_solver_iterations > 0);
    }
}
