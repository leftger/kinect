//! Driver-independent 3D geometry for the Kinect scanner.
//!
//! Nothing in this crate knows about the Kinect or USB: it operates on point
//! clouds, depth images and camera intrinsics, so the whole pipeline can be
//! unit-tested against synthetic data without hardware attached.
//!
//! The pieces, in the order a scan uses them:
//!
//! 1. [`voxel`] — make point density independent of range.
//! 2. [`normals`] — local PCA normals, needed by point-to-plane ICP.
//! 3. [`icp`] — point-to-plane alignment, i.e. the odometry.
//! 4. [`projective`] — the fast correspondence strategy the odometry uses.
//! 5. [`tsdf`] — fused truncated signed distance field with surface extraction.
//! 6. [`mesh`] — triangle mesh, including binary PLY.
//! 7. [`texturing`] — atlas built from the colour views.
//! 8. [`export`] — write that mesh as PLY, OBJ, glTF or GLB.
//! 9. [`png`] — the compressed RGB image those textured formats embed or sit beside.
//! 10. [`level`] — optionally rotate a finished model so its floor is horizontal.

pub mod coloring;
pub mod export;
pub mod frozen_map;
pub mod global_search;
pub mod icp;
pub mod level;
pub mod mesh;
pub mod normals;
pub mod png;
pub mod pose_graph;
pub mod projective;
pub mod spatial;
pub mod texturing;
pub mod tsdf;
pub mod voxel;

use nalgebra::{Isometry3, Vector3};

/// Apply a rigid transform to a position vector.
///
/// Spelled out rather than relying on `Isometry3 * Vector3`, because nalgebra's
/// operator semantics for vectors-vs-points are easy to misread and this is the
/// single most-used operation in the pipeline.
#[inline]
pub fn transform_point(transform: &Isometry3<f32>, point: &Vector3<f32>) -> Vector3<f32> {
    transform.rotation * point + transform.translation.vector
}

/// OpenCV / Brown-Conrady lens distortion coefficients `(k1, k2, p1, p2)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Distortion {
    pub k1: f32,
    pub k2: f32,
    pub p1: f32,
    pub p2: f32,
}

impl Distortion {
    /// True when every coefficient is zero (exact pinhole).
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0
    }

    /// Forward Brown-Conrady distortion mapping normalized undistorted rays `(x, y)`
    /// to distorted normalized coordinates.
    #[inline]
    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let x2 = x * x;
        let y2 = y * y;
        let xy = x * y;
        let r2 = x2 + y2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        let dx = 2.0 * self.p1 * xy + self.p2 * (r2 + 2.0 * x2);
        let dy = self.p1 * (r2 + 2.0 * y2) + 2.0 * self.p2 * xy;
        (x * radial + dx, y * radial + dy)
    }

    /// Distortion coefficients after mirroring images horizontally (`x -> -x`).
    /// Radial terms and vertical tangential term `p1` stay invariant; `p2` negates.
    #[inline]
    pub fn mirrored_x(&self) -> Self {
        Self {
            k1: self.k1,
            k2: self.k2,
            p1: self.p1,
            p2: -self.p2,
        }
    }
}

/// Pinhole camera intrinsics: the parameters that turn pixels into rays and back.
///
/// The Kinect v2 depth camera reports `fx = fy ~= 367`, `cx ~= 261`, `cy ~= 211`
/// for its 512x424 image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    /// Focal length in pixels, x.
    pub fx: f32,
    /// Focal length in pixels, y.
    pub fy: f32,
    /// Principal point, x, in pixels.
    pub cx: f32,
    /// Principal point, y, in pixels.
    pub cy: f32,
}

impl Intrinsics {
    /// Camera-space point for a pixel and a depth (metres).
    ///
    /// `u` and `v` are continuous pixel coordinates: add 0.5 to an integer index
    /// to address the centre of that pixel, which matters at this noise level.
    #[inline]
    pub fn back_project(&self, u: f32, v: f32, depth: f32) -> Vector3<f32> {
        Vector3::new(
            (u - self.cx) / self.fx * depth,
            (v - self.cy) / self.fy * depth,
            depth,
        )
    }

    /// Pixel coordinates of a camera-space point, or `None` if it is not in
    /// front of the camera.
    #[inline]
    pub fn project(&self, point: &Vector3<f32>) -> Option<(f32, f32)> {
        if point.z <= 0.0 {
            return None;
        }
        Some((
            point.x / point.z * self.fx + self.cx,
            point.y / point.z * self.fy + self.cy,
        ))
    }
}

/// A depth image in metres, row-major.
///
/// `NaN` or any non-positive value means "no measurement here" — the convention
/// the driver already uses, and the one [`TsdfVolume::integrate`] expects.
///
/// [`TsdfVolume::integrate`]: tsdf::TsdfVolume::integrate
pub struct DepthImage<'a> {
    pub width: usize,
    pub height: usize,
    pub depth: &'a [f32],
}

impl<'a> DepthImage<'a> {
    pub fn new(width: usize, height: usize, depth: &'a [f32]) -> Self {
        assert_eq!(
            depth.len(),
            width * height,
            "depth buffer does not match {width}x{height}"
        );
        Self {
            width,
            height,
            depth,
        }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.depth[y * self.width + x]
    }

    /// Points for every measured pixel, in camera space (metres).
    ///
    /// Unlike a TSDF this does not need the points to be ordered, so the output
    /// is just a flat cloud suitable for [`icp::IcpCloud::build`].
    ///
    /// [`icp::IcpCloud::build`]: icp::IcpCloud::build
    pub fn to_points(&self, intrinsics: &Intrinsics) -> Vec<Vector3<f32>> {
        let mut points = Vec::new();

        for y in 0..self.height {
            for x in 0..self.width {
                let depth = self.at(x, y);
                if depth.is_finite() && depth > 0.0 {
                    points.push(intrinsics.back_project(x as f32 + 0.5, y as f32 + 0.5, depth));
                }
            }
        }

        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intrinsics() -> Intrinsics {
        Intrinsics {
            fx: 367.0,
            fy: 367.0,
            cx: 261.0,
            cy: 211.0,
        }
    }

    #[test]
    fn principal_point_maps_to_the_optical_axis() {
        let k = intrinsics();
        let point = k.back_project(k.cx, k.cy, 2.0);
        assert!((point - Vector3::new(0.0, 0.0, 2.0)).norm() < 1e-6);
    }

    #[test]
    fn project_inverts_back_project() {
        let k = intrinsics();

        for &(u, v, d) in &[
            (0.0f32, 0.0f32, 1.0f32),
            (511.5, 423.5, 3.2),
            (100.0, 300.0, 0.7),
        ] {
            let point = k.back_project(u, v, d);
            let (pu, pv) = k.project(&point).expect("in front of camera");
            assert!((pu - u).abs() < 1e-3, "{pu} vs {u}");
            assert!((pv - v).abs() < 1e-3, "{pv} vs {v}");
        }
    }

    #[test]
    fn project_rejects_points_behind_the_camera() {
        assert!(
            intrinsics()
                .project(&Vector3::new(0.0, 0.0, -1.0))
                .is_none()
        );
        assert!(intrinsics().project(&Vector3::new(0.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn to_points_skips_missing_measurements() {
        let mut depth = vec![1.0f32; 9];
        depth[4] = f32::NAN;
        depth[0] = 0.0;

        let image = DepthImage::new(3, 3, &depth);
        assert_eq!(image.to_points(&intrinsics()).len(), 7);
    }

    #[test]
    fn transform_point_matches_manual_rotation() {
        let transform = Isometry3::from_parts(
            nalgebra::Translation3::new(1.0, 2.0, 3.0),
            nalgebra::UnitQuaternion::from_axis_angle(
                &Vector3::z_axis(),
                std::f32::consts::FRAC_PI_2,
            ),
        );

        let moved = transform_point(&transform, &Vector3::new(1.0, 0.0, 0.0));
        assert!(
            (moved - Vector3::new(1.0, 3.0, 3.0)).norm() < 1e-6,
            "{moved:?}"
        );
    }

    #[test]
    fn distortion_mirrored_x_matches_horizontal_flip() {
        let d = Distortion {
            k1: -0.08,
            k2: 0.02,
            p1: 0.003,
            p2: -0.004,
        };
        let (x_d, y_d) = d.apply(0.35, -0.22);
        let (x_m, y_m) = d.mirrored_x().apply(-0.35, -0.22);
        assert!((x_m + x_d).abs() < 1e-6, "{x_m} vs -{x_d}");
        assert!((y_m - y_d).abs() < 1e-6, "{y_m} vs {y_d}");
    }
}
