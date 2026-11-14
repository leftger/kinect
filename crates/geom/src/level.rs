//! Level a reconstruction against its dominant horizontal plane.
//!
//! The scanner's world frame is the *first camera frame*, and there is no IMU
//! to say which way is down. A scan that starts with the sensor pitched or
//! rolled a few degrees therefore comes out tilted: the floor and ceiling are
//! still flat and parallel, but they lie across a range of heights instead of
//! at two. That is harmless to the geometry and bad for everything that wants a
//! room to stand upright.
//!
//! This finds the floor or ceiling and returns the smallest rotation that makes
//! it perpendicular to the sensor's own vertical (`y`) axis. It is a
//! Manhattan-style assumption, stated plainly: the scan has a large, flat,
//! roughly horizontal surface and the sensor was held roughly upright. When
//! either is false it returns `None` rather than guess.
//!
//! The search is a deterministic scan over candidate normals inside a cone
//! around `y`, scoring each by how many points fall into one thin slab. A random
//! three-point RANSAC would need thousands of draws to hit a plane holding a few
//! percent of the vertices; the cone is only two degrees of freedom, so an
//! exhaustive scan is both cheaper to reason about and repeatable.

use nalgebra::{Matrix3, Rotation3, SymmetricEigen, Vector3};
use rayon::prelude::*;

/// Tuning for [`estimate`].
#[derive(Clone, Copy, Debug)]
pub struct LevelParams {
    /// Widest tilt, in radians, between the sensor's `y` axis and a plane's
    /// normal that still counts as "horizontal". Walls are far outside this.
    pub max_tilt: f32,
    /// Half-thickness of the slab that counts as one plane, in metres.
    pub inlier_band: f32,
    /// The plane must hold at least this fraction of the sampled points.
    pub min_inlier_fraction: f32,
    /// ... and that fraction must be at least this many times what a perfectly
    /// uniform spread of heights would put in the same slab. A vertical wall
    /// spreads its points evenly over height, so a thin slab through one can
    /// hold a few percent of a scan without being a floor; only a surface that
    /// piles points onto one height beats the uniform expectation by a margin.
    pub min_contrast: f32,
    /// Points are strided down to about this many before the search.
    pub max_samples: usize,
}

impl Default for LevelParams {
    fn default() -> Self {
        Self {
            max_tilt: 25.0f32.to_radians(),
            inlier_band: 0.03,
            min_inlier_fraction: 0.02,
            min_contrast: 3.0,
            max_samples: 20_000,
        }
    }
}

/// A rotation that levels a reconstruction, and how it was found.
#[derive(Clone, Copy, Debug)]
pub struct Leveling {
    /// Rotation about the origin taking the plane's normal onto `+y`.
    pub rotation: Rotation3<f32>,
    /// Angle, in radians, the plane was tilted from horizontal.
    pub tilt: f32,
    /// Fraction of the sampled points inside the plane's slab.
    pub inlier_fraction: f32,
}

impl Leveling {
    pub fn apply(&self, point: &Vector3<f32>) -> Vector3<f32> {
        self.rotation * point
    }
}

/// Coarse angular step of the normal search, in radians (0.5 degrees).
const COARSE_STEP: f64 = 0.5 * std::f64::consts::PI / 180.0;
/// Fine step for the refinement around the best coarse direction (0.1 degrees).
const FINE_STEP: f64 = 0.1 * std::f64::consts::PI / 180.0;
/// Half-width of the refinement window, in fine steps.
const FINE_HALF_WIDTH: i32 = 10;
/// Fewer points than this cannot support a plane estimate.
const MIN_POINTS: usize = 200;

/// Find the dominant near-horizontal plane in `points` and the rotation that
/// levels it, or `None` when there is not one.
pub fn estimate(points: &[Vector3<f32>], params: &LevelParams) -> Option<Leveling> {
    let step = (points.len() / params.max_samples.max(1)).max(1);
    let sample: Vec<Vector3<f64>> = points
        .iter()
        .step_by(step)
        .filter(|p| p.iter().all(|c| c.is_finite()))
        .map(|p| p.cast::<f64>())
        .collect();
    if sample.len() < MIN_POINTS {
        return None;
    }

    let band = f64::from(params.inlier_band).max(1.0e-4);
    let max_tilt = f64::from(params.max_tilt);
    let radius = sample
        .iter()
        .map(|p| p.norm())
        .fold(0.0f64, f64::max)
        .max(band);
    let slabs = Slabs::new(radius, band);

    // Coarse: every direction in the cone around +y.
    let coarse = cone(&Vector3::y(), max_tilt, COARSE_STEP);
    let (mut best, best_count) = best_direction(&sample, &coarse, &slabs);

    // Fine: a small window around the winner, so a plane that falls between
    // coarse samples is not scored down by a degree of misalignment.
    let (e1, e2) = perpendicular_basis(&best);
    let mut fine = Vec::new();
    for a in -FINE_HALF_WIDTH..=FINE_HALF_WIDTH {
        for b in -FINE_HALF_WIDTH..=FINE_HALF_WIDTH {
            let candidate =
                (best + e1 * (f64::from(a) * FINE_STEP) + e2 * (f64::from(b) * FINE_STEP))
                    .normalize();
            if candidate.y >= max_tilt.cos() {
                fine.push(candidate);
            }
        }
    }
    let (refined, refined_count) = best_direction(&sample, &fine, &slabs);
    if refined_count >= best_count {
        best = refined;
    }

    // Refit on the slab's own points: a least-squares plane is sharper than any
    // grid direction.
    let offset = slabs.peak_offset(&sample, &best);
    let inliers: Vec<&Vector3<f64>> = sample
        .iter()
        .filter(|p| (best.dot(p) - offset).abs() <= band)
        .collect();
    if inliers.len() < MIN_POINTS {
        return None;
    }
    let normal = fit_normal(&inliers)?;
    let normal = if normal.y < 0.0 { -normal } else { normal };

    // The refit can walk off the grid; it still has to be a floor or ceiling.
    if normal.y < max_tilt.cos() {
        return None;
    }
    let inlier_fraction = inliers.len() as f32 / sample.len() as f32;
    if inlier_fraction < params.min_inlier_fraction {
        return None;
    }

    // Compare against a uniform spread of heights over the scan's own extent
    // along this normal (1st to 99th percentile, so stray points do not stretch
    // it). A real floor concentrates points far beyond that expectation.
    let mut heights: Vec<f64> = sample.iter().map(|p| normal.dot(p)).collect();
    heights.sort_by(|a, b| a.total_cmp(b));
    let low = heights[heights.len() / 100];
    let high = heights[heights.len() - 1 - heights.len() / 100];
    let extent = (high - low).max(2.0 * band);
    let uniform = (2.0 * band / extent).min(1.0);
    if f64::from(inlier_fraction) < f64::from(params.min_contrast) * uniform {
        return None;
    }

    let rotation = Rotation3::rotation_between(&normal, &Vector3::y())?;
    Some(Leveling {
        rotation: rotation.cast::<f32>(),
        tilt: normal.y.clamp(-1.0, 1.0).acos() as f32,
        inlier_fraction,
    })
}

/// Histogram of plane offsets along one direction, with a fixed slab width.
struct Slabs {
    radius: f64,
    band: f64,
    bins: usize,
}

impl Slabs {
    fn new(radius: f64, band: f64) -> Self {
        let bins = ((2.0 * radius) / band).ceil() as usize + 2;
        Self { radius, band, bins }
    }

    fn bin(&self, height: f64) -> usize {
        (((height + self.radius) / self.band) as usize).min(self.bins - 1)
    }

    /// The most-populated slab along `normal`, as a point count.
    fn peak(&self, points: &[Vector3<f64>], normal: &Vector3<f64>) -> usize {
        let mut counts = vec![0u32; self.bins];
        for point in points {
            counts[self.bin(normal.dot(point))] += 1;
        }
        counts.into_iter().max().unwrap_or(0) as usize
    }

    /// The centre height of that slab.
    fn peak_offset(&self, points: &[Vector3<f64>], normal: &Vector3<f64>) -> f64 {
        let mut counts = vec![0u32; self.bins];
        for point in points {
            counts[self.bin(normal.dot(point))] += 1;
        }
        let index = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| **count)
            .map_or(0, |(index, _)| index);
        (index as f64 + 0.5) * self.band - self.radius
    }
}

fn best_direction(
    points: &[Vector3<f64>],
    directions: &[Vector3<f64>],
    slabs: &Slabs,
) -> (Vector3<f64>, usize) {
    directions
        .par_iter()
        .map(|direction| (*direction, slabs.peak(points, direction)))
        .reduce(|| (Vector3::y(), 0), |a, b| if b.1 > a.1 { b } else { a })
}

/// Unit directions within `max_angle` of `axis`, spaced about `step` apart.
fn cone(axis: &Vector3<f64>, max_angle: f64, step: f64) -> Vec<Vector3<f64>> {
    let (e1, e2) = perpendicular_basis(axis);
    let rings = (max_angle / step).ceil() as usize;
    let mut directions = vec![*axis];
    for ring in 1..=rings {
        let tilt = (ring as f64 * step).min(max_angle);
        let count = ((std::f64::consts::TAU * tilt.sin()) / step)
            .ceil()
            .max(1.0) as usize;
        for index in 0..count {
            let azimuth = std::f64::consts::TAU * index as f64 / count as f64;
            directions
                .push(axis * tilt.cos() + (e1 * azimuth.cos() + e2 * azimuth.sin()) * tilt.sin());
        }
    }
    directions
}

fn perpendicular_basis(axis: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    // Cross with whichever world axis is least aligned, to stay well-conditioned.
    let helper = if axis.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::z()
    };
    let e1 = axis.cross(&helper).normalize();
    let e2 = axis.cross(&e1);
    (e1, e2)
}

/// Least-squares plane normal: the eigenvector of the smallest covariance
/// eigenvalue.
fn fit_normal(points: &[&Vector3<f64>]) -> Option<Vector3<f64>> {
    let centroid = points.iter().fold(Vector3::zeros(), |sum, p| sum + **p) / points.len() as f64;
    let mut covariance = Matrix3::zeros();
    for point in points {
        let d = **point - centroid;
        covariance += d * d.transpose();
    }
    let eigen = SymmetricEigen::new(covariance);
    let smallest = (0..3).min_by(|&a, &b| {
        eigen.eigenvalues[a]
            .partial_cmp(&eigen.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    let normal: Vector3<f64> = eigen.eigenvectors.column(smallest).into();
    let norm = normal.norm();
    (norm > 0.5).then(|| normal / norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise in `[-amplitude, amplitude]`, so a test never flakes.
    fn jitter(seed: u32, amplitude: f32) -> f32 {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9e37_79b9);
        x ^= x >> 15;
        x = x.wrapping_mul(2_246_822_519);
        x ^= x >> 13;
        (x as f32 / u32::MAX as f32 * 2.0 - 1.0) * amplitude
    }

    /// A room as a surface point cloud: floor at `y = 1.4`, ceiling at
    /// `y = -1.2` (sensor `y` points down), and four walls.
    fn room(with_horizontals: bool) -> Vec<Vector3<f32>> {
        let mut points = Vec::new();
        let mut seed = 0;
        let mut noise = |amplitude: f32| {
            seed += 1;
            jitter(seed, amplitude)
        };

        for i in 0..70 {
            for j in 0..50 {
                let u = -3.0 + 6.0 * i as f32 / 69.0;
                let v = 0.5 + 5.0 * j as f32 / 49.0;
                if with_horizontals {
                    points.push(Vector3::new(u, 1.4 + noise(0.004), v));
                    points.push(Vector3::new(u, -1.2 + noise(0.004), v));
                }
            }
        }
        for i in 0..70 {
            for j in 0..50 {
                let u = -3.0 + 6.0 * i as f32 / 69.0;
                let h = -1.2 + 2.6 * j as f32 / 49.0;
                points.push(Vector3::new(u, h, 5.5 + noise(0.004)));
                points.push(Vector3::new(u, h, 0.5 + noise(0.004)));
            }
        }
        for i in 0..50 {
            for j in 0..50 {
                let v = 0.5 + 5.0 * i as f32 / 49.0;
                let h = -1.2 + 2.6 * j as f32 / 49.0;
                points.push(Vector3::new(-3.0 + noise(0.004), h, v));
                points.push(Vector3::new(3.0 + noise(0.004), h, v));
            }
        }
        points
    }

    fn tilt(points: &[Vector3<f32>], rx: f32, rz: f32) -> Vec<Vector3<f32>> {
        let rotation = Rotation3::from_euler_angles(rx, 0.0, rz);
        points.iter().map(|p| rotation * p).collect()
    }

    #[test]
    fn a_tilted_room_is_levelled_back_to_its_floor_and_ceiling() {
        let level = room(true);
        let tilted = tilt(&level, 7.0f32.to_radians(), -3.0f32.to_radians());

        let found = estimate(&tilted, &LevelParams::default()).expect("a floor exists");
        assert!(
            (found.tilt.to_degrees() - 7.6).abs() < 0.7,
            "tilt {:.2} deg, expected about 7.6",
            found.tilt.to_degrees()
        );

        // Follow the floor and ceiling points through the tilt by their original
        // height (the walls also have points with y on either side of zero), and
        // require each set to collapse back onto one height.
        let heights_after = |wanted: f32| -> Vec<f32> {
            level
                .iter()
                .zip(&tilted)
                .filter(|(original, _)| (original.y - wanted).abs() < 0.02)
                .map(|(_, moved)| found.apply(moved).y)
                .collect()
        };
        for (name, wanted) in [("floor", 1.4f32), ("ceiling", -1.2f32)] {
            let heights = heights_after(wanted);
            assert!(
                heights.len() > 1000,
                "{name}: only {} points",
                heights.len()
            );
            let mean = heights.iter().sum::<f32>() / heights.len() as f32;
            let low = heights.iter().cloned().fold(f32::MAX, f32::min);
            let high = heights.iter().cloned().fold(f32::MIN, f32::max);
            assert!(
                (mean - wanted).abs() < 0.08,
                "{name} sits at {mean:.3}, not {wanted}"
            );
            assert!(high - low < 0.08, "{name} still spans {:.3} m", high - low);
        }
    }

    #[test]
    fn an_already_level_scan_is_left_essentially_alone() {
        let found = estimate(&room(true), &LevelParams::default()).expect("a floor exists");
        assert!(
            found.tilt.to_degrees() < 0.5,
            "tilt {:.2} deg",
            found.tilt.to_degrees()
        );
        let moved = found.apply(&Vector3::new(1.0, 1.0, 1.0)) - Vector3::new(1.0, 1.0, 1.0);
        assert!(
            moved.norm() < 0.02,
            "moved a level scan by {}",
            moved.norm()
        );
    }

    #[test]
    fn a_scan_with_no_horizontal_surface_is_not_levelled() {
        // Walls only: the largest planes are vertical, outside the cone.
        assert!(estimate(&room(false), &LevelParams::default()).is_none());
    }

    #[test]
    fn a_floor_tilted_past_the_limit_is_not_trusted() {
        let tilted = tilt(&room(true), 40.0f32.to_radians(), 0.0);
        assert!(estimate(&tilted, &LevelParams::default()).is_none());
    }

    #[test]
    fn too_few_points_return_none() {
        let few = vec![Vector3::new(0.0, 1.0, 1.0); 50];
        assert!(estimate(&few, &LevelParams::default()).is_none());
        assert!(estimate(&[], &LevelParams::default()).is_none());
    }

    #[test]
    fn non_finite_points_are_ignored() {
        let mut points = tilt(&room(true), 5.0f32.to_radians(), 0.0);
        points.push(Vector3::new(f32::NAN, 0.0, 0.0));
        points.push(Vector3::new(f32::INFINITY, 0.0, 1.0));
        assert!(estimate(&points, &LevelParams::default()).is_some());
    }
}
