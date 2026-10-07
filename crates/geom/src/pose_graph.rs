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
//! # Two deliberate simplifications
//!
//! **The residual is not the exact SE(3) log map.** It stacks the translation and
//! the rotation vector rather than applying the coupling `V^-1` factor. The
//! difference vanishes as the residual goes to zero, which is where an optimiser
//! ends up, and it removes the small-angle special cases that are easy to get
//! subtly wrong.
//!
//! **The Jacobians are numerical, not analytic.** A finite-difference Jacobian
//! over six perturbation directions per edge is a few thousand small matrix
//! products per iteration at the sizes this scanner produces, which is
//! milliseconds -- and it cannot silently disagree with the residual the way a
//! hand-derived Jacobian can. If the graph ever grows past a few hundred
//! keyframes, this and the dense solve below are the two things to revisit.
//!
//! # What makes it safe
//!
//! A pose graph happily folds the map in half to satisfy a single wrong loop
//! edge, so the *caller* must verify candidates geometrically before adding them
//! (see the scanner: ICP fitness thresholds). On top of that, a Huber kernel here
//! stops any one edge that survives verification from dominating.

use nalgebra::{DMatrix, Isometry3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};

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
    /// Step used for the finite-difference Jacobian.
    pub finite_difference_step: f32,
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
            finite_difference_step: 1e-5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OptimizationReport {
    pub cost_before: f64,
    pub cost_after: f64,
    pub iterations: usize,
    pub converged: bool,
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
            };
        }

        // Node 0 is the anchor. The cost is invariant to a global rigid motion of
        // the whole trajectory, so without fixing something the normal equations
        // are singular in six directions and the solve is arbitrary.
        let dimension = 6 * node_count;
        let mut converged = false;
        let mut iterations = 0;

        for iteration in 0..params.iterations {
            iterations = iteration + 1;

            let mut hessian = DMatrix::<f32>::zeros(dimension, dimension);
            let mut gradient = DMatrix::<f32>::zeros(dimension, 1);

            for edge in &self.edges {
                let (j_from, j_to, error) = self.linearise(edge, params.finite_difference_step);

                // Huber: quadratic near zero, linear in the tails, so one edge that
                // survived verification but is still wrong cannot drag the map.
                let mut weight = edge.weight;
                let norm = error.norm();
                if params.robust_delta > 0.0 && norm > params.robust_delta {
                    weight *= params.robust_delta / norm;
                }

                let i = 6 * edge.from;
                let j = 6 * edge.to;

                for a in 0..6 {
                    for k in 0..6 {
                        gradient[(i + k, 0)] += weight * j_from[(a, k)] * error[a];
                        gradient[(j + k, 0)] += weight * j_to[(a, k)] * error[a];

                        for l in 0..6 {
                            hessian[(i + k, i + l)] += weight * j_from[(a, k)] * j_from[(a, l)];
                            hessian[(i + k, j + l)] += weight * j_from[(a, k)] * j_to[(a, l)];
                            hessian[(j + k, i + l)] += weight * j_to[(a, k)] * j_from[(a, l)];
                            hessian[(j + k, j + l)] += weight * j_to[(a, k)] * j_to[(a, l)];
                        }
                    }
                }
            }

            for k in 0..dimension {
                hessian[(k, k)] += params.damping;
            }

            // Pin node 0: zero its row and column, unit diagonal, zero gradient.
            for k in 0..6 {
                for l in 0..dimension {
                    hessian[(k, l)] = 0.0;
                    hessian[(l, k)] = 0.0;
                }
            }
            for k in 0..6 {
                hessian[(k, k)] = 1.0;
                gradient[(k, 0)] = 0.0;
            }

            let Some(step) = hessian.lu().solve(&(-gradient)) else {
                break;
            };

            let mut largest_update = 0.0f32;
            for node in 1..node_count {
                let mut translation = Vector3::zeros();
                let mut rotation = Vector3::zeros();
                for k in 0..3 {
                    translation[k] = step[(6 * node + k, 0)];
                    rotation[k] = step[(6 * node + 3 + k, 0)];
                }

                // Right-multiply: the increment is expressed in the node's own
                // frame, which is what the numerical Jacobian perturbed.
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
        }
    }

    /// Jacobians of one edge's residual with respect to its two nodes, by central
    /// differences. Columns are the six perturbation directions of that node.
    fn linearise(&self, edge: &Edge, step: f32) -> (Matrix6<f32>, Matrix6<f32>, Vector6<f32>) {
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

/// Residual of the edge `measurement` under the two poses given.
#[inline]
fn error_between(
    measurement: &Isometry3<f32>,
    from: &Isometry3<f32>,
    to: &Isometry3<f32>,
) -> Vector6<f32> {
    let error = measurement.inverse() * (from.inverse() * to);

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

/// A unit-magnitude twist: translation for the first three directions, rotation
/// for the last three. This is the basis the perturbations live in.
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
}
