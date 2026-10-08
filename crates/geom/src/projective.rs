//! Projective data association: find ICP correspondences by projecting into a
//! depth image instead of searching a spatial index.
//!
//! Classic ICP answers "which target point is nearest to this moved point?" with
//! a spatial query -- here a hash probe over the 27 cells around the query, which
//! measured as the most expensive thing in the whole pipeline (~600 ms/frame, and
//! 74% of the per-frame cost).
//!
//! Projective association answers it differently. The target *was* a depth image,
//! so a moved point corresponds to whichever pixel it lands on when transformed
//! into the target camera. That is one projection and one array read: no index,
//! no search, and no cell size to tune.
//!
//! The price is that it only works when the two frames are close enough that the
//! projection is meaningful, and when the initial guess is roughly right -- which
//! is exactly the situation frame-to-frame odometry is in. It is not a drop-in
//! replacement for `icp::align`, which stays available for aligning
//! widely-separated clouds (keyframes, loop closure).

use nalgebra::{Isometry3, Vector3};

use crate::icp::{align_points, CorrespondenceSource, IcpParams, IcpResult};
use crate::{DepthImage, Intrinsics};

/// One resolution level of a depth pyramid, with surface normals precomputed.
pub struct DepthLevel {
    pub width: usize,
    pub height: usize,
    /// Metres, row-major. `NaN` means no measurement.
    pub depth: Vec<f32>,
    /// Unit normals oriented towards the camera; zero where unknown.
    normals: Vec<Vector3<f32>>,
    pub intrinsics: Intrinsics,
}

impl DepthLevel {
    fn new(width: usize, height: usize, depth: Vec<f32>, intrinsics: Intrinsics) -> Self {
        let mut level = Self {
            width,
            height,
            depth,
            normals: vec![Vector3::zeros(); width * height],
            intrinsics,
        };
        level.compute_normals();
        level
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.depth[y * self.width + x]
    }

    #[inline]
    fn valid(&self, x: usize, y: usize) -> bool {
        let d = self.at(x, y);
        d.is_finite() && d > 0.0
    }

    /// Camera-space point at an integer pixel index.
    #[inline]
    fn point(&self, x: usize, y: usize) -> Vector3<f32> {
        self.intrinsics
            .back_project(x as f32 + 0.5, y as f32 + 0.5, self.at(x, y))
    }

    /// Per-pixel normals straight from the depth image: the cross product of the
    /// local tangents. Adjacent pixels are the natural neighbourhood here, so
    /// this is a handful of flops per pixel and needs no search at all.
    fn compute_normals(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let index = y * self.width + x;
                if !self.valid(x, y) {
                    continue;
                }

                // Central differences need all four neighbours. At the border, or
                // across a depth discontinuity, a tangent would be a lie, so leave
                // the normal unset and let the correspondence be rejected.
                if x == 0 || y == 0 || x + 1 >= self.width || y + 1 >= self.height {
                    continue;
                }
                if !self.valid(x - 1, y)
                    || !self.valid(x + 1, y)
                    || !self.valid(x, y - 1)
                    || !self.valid(x, y + 1)
                {
                    continue;
                }

                let centre = self.point(x, y);
                let tangent_x = self.point(x + 1, y) - self.point(x - 1, y);
                let tangent_y = self.point(x, y + 1) - self.point(x, y - 1);

                let cross = tangent_x.cross(&tangent_y);
                let length = cross.norm();
                if length < 1e-9 {
                    continue;
                }

                let mut normal = cross / length;
                // Orient towards the camera, which sits at the origin. Without
                // this the sign is arbitrary and the residuals flip for half the
                // surface.
                if normal.dot(&centre) > 0.0 {
                    normal = -normal;
                }

                self.normals[index] = normal;
            }
        }
    }

    /// Half-resolution copy, averaging the valid depth of each 2x2 block.
    fn halved(&self) -> DepthLevel {
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut depth = vec![f32::NAN; width * height];

        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                        if sx < self.width && sy < self.height && self.valid(sx, sy) {
                            sum += self.at(sx, sy);
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    depth[y * width + x] = sum / count as f32;
                }
            }
        }

        // Halving a pixel grid moves the principal point by half a pixel as well
        // as halving the focal length. `(c + 0.5) / 2 - 0.5` is the exact
        // relation between pixel *centres* at the two scales, and getting it
        // wrong biases every projection by a quarter pixel.
        let intrinsics = Intrinsics {
            fx: self.intrinsics.fx / 2.0,
            fy: self.intrinsics.fy / 2.0,
            cx: (self.intrinsics.cx + 0.5) / 2.0 - 0.5,
            cy: (self.intrinsics.cy + 0.5) / 2.0 - 0.5,
        };

        DepthLevel::new(width, height, depth, intrinsics)
    }

    /// Every pixel with a valid measurement, in camera space.
    fn points(&self) -> Vec<Vector3<f32>> {
        let mut points = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                if self.valid(x, y) {
                    points.push(self.point(x, y));
                }
            }
        }
        points
    }

    /// Pixels carrying a usable normal. Used to decide whether a frame has
    /// enough structure to track against at all.
    pub fn normals_present(&self) -> usize {
        self.normals
            .iter()
            .filter(|n| n.norm_squared() > 0.0)
            .count()
    }
}

/// A depth frame at several resolutions, finest first.
pub struct DepthPyramid {
    pub levels: Vec<DepthLevel>,
}

impl DepthPyramid {
    /// Build `count` levels, each half the resolution of the last.
    ///
    /// Building the pyramid once per frame and keeping it means the frame serves
    /// as the source now and as the target next frame, so its normals are
    /// computed exactly once.
    pub fn build(
        depth: &[f32],
        width: usize,
        height: usize,
        intrinsics: Intrinsics,
        count: usize,
    ) -> Self {
        assert!(count >= 1, "a pyramid needs at least one level");
        assert_eq!(depth.len(), width * height, "depth buffer size mismatch");

        let mut levels = Vec::with_capacity(count);
        let mut level = DepthLevel::new(width, height, depth.to_vec(), intrinsics);
        for _ in 1..count {
            let next = level.halved();
            levels.push(level);
            level = next;
        }
        levels.push(level);

        Self { levels }
    }

    /// Convenience for building from an existing depth image.
    pub fn from_image(image: &DepthImage<'_>, intrinsics: Intrinsics, count: usize) -> Self {
        Self::build(image.depth, image.width, image.height, intrinsics, count)
    }

    /// Build a pyramid with precomputed surface normals at the finest level.
    ///
    /// Used by frame-to-model alignment to supply the TSDF's analytic gradient
    /// normals directly instead of estimating them from discrete depth differences.
    pub fn from_image_with_normals(
        image: &DepthImage<'_>,
        normals: Vec<Vector3<f32>>,
        intrinsics: Intrinsics,
        count: usize,
    ) -> Self {
        assert!(count >= 1, "a pyramid needs at least one level");
        assert_eq!(image.depth.len(), image.width * image.height, "depth size mismatch");
        assert_eq!(normals.len(), image.width * image.height, "normals size mismatch");

        let mut levels = Vec::with_capacity(count);
        let mut level = DepthLevel {
            width: image.width,
            height: image.height,
            depth: image.depth.to_vec(),
            normals,
            intrinsics,
        };
        for _ in 1..count {
            let next = level.halved();
            levels.push(level);
            level = next;
        }
        levels.push(level);

        Self { levels }
    }

    /// Finest level.
    pub fn finest(&self) -> &DepthLevel {
        &self.levels[0]
    }

    /// Pixels with a usable normal at the finest level.
    pub fn trackable_points(&self) -> usize {
        self.finest().normals_present()
    }
}

/// Projective correspondences against one level of a target pyramid.
struct ProjectiveTarget<'a> {
    level: &'a DepthLevel,
    max_distance: f32,
}

impl CorrespondenceSource for ProjectiveTarget<'_> {
    fn find(&self, moved: &Vector3<f32>) -> Option<(Vector3<f32>, Vector3<f32>)> {
        let (u, v) = self.level.intrinsics.project(moved)?;

        // `u`/`v` are continuous pixel coordinates, so truncation picks the pixel
        // that actually contains the projection.
        let fx = u.floor();
        let fy = v.floor();
        if fx < 0.0 || fy < 0.0 || fx >= self.level.width as f32 || fy >= self.level.height as f32 {
            return None;
        }
        let (x, y) = (fx as usize, fy as usize);

        if !self.level.valid(x, y) {
            return None;
        }
        let measured = self.level.at(x, y);

        // The occlusion test, and the reason projective association is safe at
        // all. If the target pixel's depth disagrees with the moved point's own
        // depth, these are different surfaces -- the point has passed behind
        // something, or a nearer surface has appeared in front of it -- and
        // matching them would drag the solution towards the wrong surface. It is
        // also what makes this robust to moving objects.
        if (measured - moved.z).abs() > self.max_distance {
            return None;
        }

        let normal = self.level.normals[y * self.level.width + x];
        if normal.norm_squared() == 0.0 {
            return None;
        }

        Some((self.level.point(x, y), normal))
    }
}

/// Align two depth frames at one pyramid level, projectively.
///
/// `initial` maps source camera coordinates into the target camera frame.
/// Callers run this coarse-to-fine over a [`DepthPyramid`]: the coarse levels
/// take up the bulk of the motion cheaply and keep the fine level from settling
/// into a wrong local minimum.
pub fn align_level(
    source: &DepthLevel,
    target: &DepthLevel,
    initial: Isometry3<f32>,
    params: &IcpParams,
) -> IcpResult {
    let target_normals = target.normals_present();
    if target_normals == 0 {
        return IcpResult::failed(initial);
    }

    let points = source.points();
    if points.is_empty() {
        return IcpResult::failed(initial);
    }

    let finder = ProjectiveTarget {
        level: target,
        max_distance: params.max_correspondence_distance,
    };

    let mut result = align_points(&points, &finder, initial, params);
    let achievable = points.len().min(target_normals);
    if achievable > 0 {
        result.inlier_ratio = (result.correspondences as f32 / achievable as f32).min(1.0);
    }
    result
}
