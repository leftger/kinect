//! Voxel-hashed truncated signed distance field (TSDF) with surface extraction.
//!
//! Each frame is integrated into a weighted-average distance field, so repeated
//! observations of the same surface average down the ~5 mm depth noise instead of
//! leaving the speckle a plain point-cloud merge would. Only blocks near an
//! observed surface are ever allocated, which is what makes room-scale scanning
//! fit in memory: a 10 m room at 1 cm resolution is far too many voxels to store
//! densely, but only the ~100 m² of surface needs to exist.

use crate::mesh::Mesh;
use crate::spatial::Cell;
use crate::{transform_point, DepthImage, Intrinsics};
use nalgebra::{Isometry3, Vector3};
use std::collections::{HashMap, HashSet};

/// Voxels along one edge of a block.
pub const BLOCK: i32 = 8;
const VOXELS_PER_BLOCK: usize = (BLOCK * BLOCK * BLOCK) as usize;

#[derive(Clone, Copy, Debug)]
pub struct TsdfParams {
    /// Edge length of one voxel, in metres.
    pub voxel_size: f32,
    /// Distance beyond which a measurement is not trusted, in metres. Typically
    /// a few voxels: larger keeps more of the noise, smaller clips thin geometry.
    pub truncation: f32,
    /// Measurements nearer than this are ignored, in metres.
    pub min_depth: f32,
    /// Measurements farther than this are ignored, in metres.
    ///
    /// This matters: the decoder emits junk far past the sensor's rated range
    /// (values past 15 m were measured with the filters off), and without a bound
    /// that junk becomes real geometry in the model.
    pub max_depth: f32,
}

impl Default for TsdfParams {
    fn default() -> Self {
        // 1 cm voxels with a 4 cm band is a reasonable room-scale default: fine
        // enough for furniture edges, coarse enough to stay in memory.
        Self {
            voxel_size: 0.01,
            truncation: 0.04,
            min_depth: 0.5,
            max_depth: 4.5,
        }
    }
}

#[derive(Clone)]
struct Block {
    /// Weighted mean of the truncated signed distance, in [-1, 1].
    distance: [f32; VOXELS_PER_BLOCK],
    /// Observation count; zero means "never seen".
    weight: [f32; VOXELS_PER_BLOCK],
}

impl Default for Block {
    fn default() -> Self {
        Self {
            distance: [0.0; VOXELS_PER_BLOCK],
            weight: [0.0; VOXELS_PER_BLOCK],
        }
    }
}

/// One-entry cache over the block map, for the ray march.
///
/// A ray samples the field every few centimetres, and a block is `BLOCK` voxels
/// on a side -- 16 cm at the default 2 cm voxels -- so consecutive samples land
/// in the same block most of the time. Without this every sample paid a SipHash
/// of the cell key and a probe into a map whose values are 4 KB, which is what
/// made a full-resolution render take tens of seconds.
struct BlockCache<'a> {
    blocks: &'a HashMap<Cell, Block>,
    cell: Cell,
    block: Option<&'a Block>,
}

impl<'a> BlockCache<'a> {
    fn new(blocks: &'a HashMap<Cell, Block>) -> Self {
        Self {
            blocks,
            // No real block can sit here, so the first query misses the cache
            // rather than reading whatever was left in it.
            cell: [i32::MIN; 3],
            block: None,
        }
    }

    /// The block at `cell`, or `None` if it was never allocated.
    ///
    /// Returning `&'a Block` rather than `&self`'s borrow lets the caller hold
    /// the block across further `get` calls, which is what makes the cache worth
    /// having.
    #[inline]
    fn get(&mut self, cell: Cell) -> Option<&'a Block> {
        if cell == self.cell {
            return self.block;
        }
        let block = self.blocks.get(&cell);
        self.cell = cell;
        self.block = block;
        block
    }
}

/// Statistics from one [`TsdfVolume::integrate`] call.
#[derive(Clone, Copy, Debug, Default)]
pub struct IntegrationStats {
    /// Voxels whose distance was actually updated this frame.
    pub updated_voxels: usize,
    /// Voxels examined (i.e. inside blocks touched by this frame).
    pub examined_voxels: usize,
    /// Blocks allocated for the first time this frame.
    pub new_blocks: usize,
}

/// What [`TsdfVolume::raycast`] rendered.
pub struct RaycastImage {
    pub width: usize,
    pub height: usize,
    /// Camera-space z in metres, row-major, in the same convention as
    /// [`DepthImage`]. `NaN` where no surface was found along the ray.
    pub depth: Vec<f32>,
    /// Camera-space unit normals oriented towards the camera, derived directly
    /// from the continuous TSDF gradient at the surface crossing.
    pub normals: Vec<Vector3<f32>>,
}

impl RaycastImage {
    /// Pixels that found a surface.
    pub fn hits(&self) -> usize {
        self.depth.iter().filter(|d| d.is_finite()).count()
    }

    /// The render as a depth image, for anything that already consumes one.
    pub fn as_depth_image(&self) -> DepthImage<'_> {
        DepthImage::new(self.width, self.height, &self.depth)
    }
}

pub struct TsdfVolume {
    params: TsdfParams,
    blocks: HashMap<Cell, Block>,
}

impl TsdfVolume {
    pub fn new(params: TsdfParams) -> Self {
        assert!(params.voxel_size > 0.0, "voxel size must be positive");
        assert!(params.truncation > 0.0, "truncation must be positive");

        Self {
            params,
            blocks: HashMap::new(),
        }
    }

    pub fn params(&self) -> TsdfParams {
        self.params
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Approximate memory footprint, for keeping an eye on room-scale scans.
    pub fn allocated_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Block>()
    }

    /// Integrate one depth view observed from `camera_to_world`.
    pub fn integrate(
        &mut self,
        depth: &DepthImage<'_>,
        intrinsics: &Intrinsics,
        camera_to_world: &Isometry3<f32>,
    ) -> IntegrationStats {
        let mut stats = IntegrationStats::default();

        // Which blocks could this view possibly affect? Anything within the
        // truncation band of an observed surface point.
        let mut touched: HashSet<Cell> = HashSet::new();
        for y in 0..depth.height {
            for x in 0..depth.width {
                let d = depth.at(x, y);
                if !Self::is_usable(self.params.min_depth, self.params.max_depth, d) {
                    continue;
                }

                let point_camera = intrinsics.back_project(x as f32 + 0.5, y as f32 + 0.5, d);
                let point_world = transform_point(camera_to_world, &point_camera);

                // Convert the truncation band straight into *block* coordinates.
                //
                // Walking the voxels of the band and hashing each voxel's block
                // produces the identical set of blocks (block coordinates are
                // monotonic in voxel coordinates, so ranging over the band's
                // corners covers it), but does 8^3 times the work: at 0.03 m
                // voxels that was 217k pixels x 729 hashes = ~158M hash
                // insertions per frame, measured at 4.4 s per frame of pure
                // fusion. Ranging over blocks instead is a handful per pixel.
                let low = self.block_of(
                    self.voxel_of(&(point_world - Vector3::repeat(self.params.truncation))),
                );
                let high = self.block_of(
                    self.voxel_of(&(point_world + Vector3::repeat(self.params.truncation))),
                );

                for bx in low[0]..=high[0] {
                    for by in low[1]..=high[1] {
                        for bz in low[2]..=high[2] {
                            touched.insert([bx, by, bz]);
                        }
                    }
                }
            }
        }

        for cell in &touched {
            if !self.blocks.contains_key(cell) {
                self.blocks.insert(*cell, Block::default());
                stats.new_blocks += 1;
            }
        }

        // Update every voxel of every touched block by projecting it back into
        // this depth image -- the standard projective TSDF update.
        //
        // The parameter values are copied out first: `self.blocks` is borrowed
        // mutably inside the loop, so touching `self` again would not compile.
        let world_to_camera = camera_to_world.inverse();
        let truncation = self.params.truncation;
        let voxel_size = self.params.voxel_size;
        let (min_depth, max_depth) = (self.params.min_depth, self.params.max_depth);

        for cell in &touched {
            let block = self.blocks.get_mut(cell).expect("block just allocated");

            for local in 0..VOXELS_PER_BLOCK {
                let lx = (local % BLOCK as usize) as i32;
                let ly = ((local / BLOCK as usize) % BLOCK as usize) as i32;
                let lz = (local / (BLOCK as usize * BLOCK as usize)) as i32;

                let voxel = [
                    cell[0] * BLOCK + lx,
                    cell[1] * BLOCK + ly,
                    cell[2] * BLOCK + lz,
                ];
                stats.examined_voxels += 1;

                let world = Vector3::new(
                    voxel[0] as f32 * voxel_size,
                    voxel[1] as f32 * voxel_size,
                    voxel[2] as f32 * voxel_size,
                );
                let camera = transform_point(&world_to_camera, &world);

                // Behind the camera, or outside the image: no information.
                if camera.z <= 1e-4 {
                    continue;
                }
                let Some((u, v)) = intrinsics.project(&camera) else {
                    continue;
                };
                let (u, v) = (u.floor() as i64, v.floor() as i64);
                if u < 0 || v < 0 || u >= depth.width as i64 || v >= depth.height as i64 {
                    continue;
                }

                let measured = depth.at(u as usize, v as usize);
                if !Self::is_usable(min_depth, max_depth, measured) {
                    continue;
                }

                // Distance along the camera ray, approximated by the z difference.
                // Exact for a fronto-parallel surface, and the truncation band
                // keeps the error small elsewhere.
                let signed_distance = measured - camera.z;
                if signed_distance < -truncation || signed_distance > truncation {
                    continue;
                }

                let value = signed_distance / truncation;
                let weight = block.weight[local];
                let new_weight = weight + 1.0;
                block.distance[local] = (block.distance[local] * weight + value) / new_weight;
                block.weight[local] = new_weight;

                stats.updated_voxels += 1;
            }
        }

        stats
    }

    #[inline]
    fn voxel_of(&self, point: &Vector3<f32>) -> [i32; 3] {
        [
            (point.x / self.params.voxel_size).floor() as i32,
            (point.y / self.params.voxel_size).floor() as i32,
            (point.z / self.params.voxel_size).floor() as i32,
        ]
    }

    #[inline]
    fn block_of(&self, voxel: [i32; 3]) -> Cell {
        [
            voxel[0].div_euclid(BLOCK),
            voxel[1].div_euclid(BLOCK),
            voxel[2].div_euclid(BLOCK),
        ]
    }

    /// Distance at a voxel corner, or `None` if it was never observed.
    #[inline]
    fn voxel_distance(&self, voxel: [i32; 3]) -> Option<f32> {
        let cell = self.block_of(voxel);
        let block = self.blocks.get(&cell)?;

        let lx = voxel[0].rem_euclid(BLOCK) as usize;
        let ly = voxel[1].rem_euclid(BLOCK) as usize;
        let lz = voxel[2].rem_euclid(BLOCK) as usize;
        let index = lx + ly * BLOCK as usize + lz * BLOCK as usize * BLOCK as usize;

        (block.weight[index] > 0.0).then_some(block.distance[index])
    }

    /// Minimum observation count needed for a voxel to count as observed.
    pub fn min_weight_for_extraction(&self) -> f32 {
        0.0
    }

    /// Distance to the nearest surface at `point`, in metres, trilinearly
    /// interpolated between voxel corners. `None` only when *no* surrounding
    /// corner was observed.
    ///
    /// Voxel values belong to *corners*, not centres: `integrate` evaluates the
    /// field at `voxel_size * index`, the cell's minimum corner. Reading the
    /// value of whichever voxel happens to contain a query therefore attributes
    /// one corner's distance to the whole cell -- and a ray only ever samples
    /// points *after* the corner it landed in, so that error is one-directional
    /// rather than noise. Measured at 2 cm voxels it was +18 mm on average and
    /// 60 mm at worst, which matters because the render is the target every pose
    /// is measured against. Interpolating between the corners removes it.
    ///
    /// Corners that were never observed are left out of the average rather than
    /// failing the sample. Demanding all eight shrinks the rendered surface to
    /// the cells fully enclosed by the truncation band, which on a real frame
    /// costs about half the pixels and leaves the alignment too little to match
    /// against. Weighting by the observed corners is what the weight array is
    /// for, and a cell with nothing observed still reads as unknown so the march
    /// keeps striding.
    pub fn distance_at(&self, point: &Vector3<f32>) -> Option<f32> {
        let mut cache = BlockCache::new(&self.blocks);
        self.sample(&mut cache, point)
    }

    /// [`Self::distance_at`], re-using `cache` across calls.
    ///
    /// A ray march samples the field every few centimetres, so threading the
    /// cache through is the difference between one hash probe per sample and one
    /// per block.
    fn sample(&self, cache: &mut BlockCache<'_>, point: &Vector3<f32>) -> Option<f32> {
        let size = self.params.voxel_size;
        let base = self.voxel_of(point);

        // Position inside the cell, in [0, 1). `voxel_of` floors, so no
        // component can be negative.
        let fraction = Vector3::new(
            point.x / size - base[0] as f32,
            point.y / size - base[1] as f32,
            point.z / size - base[2] as f32,
        );

        // The eight corners occupy the 2x2x2 voxel neighbourhood at `base`. That
        // neighbourhood sits inside a single block unless `base` lies on the
        // block's far face -- unlikely on all three axes at once -- so usually
        // one probe covers all eight corners instead of eight probes covering
        // them.
        let cell = self.block_of(base);
        let local = [
            base[0].rem_euclid(BLOCK),
            base[1].rem_euclid(BLOCK),
            base[2].rem_euclid(BLOCK),
        ];
        let single_block = local.iter().all(|axis| *axis < BLOCK - 1);

        let single = if single_block { cache.get(cell) } else { None };
        if single_block && single.is_none() {
            // The whole neighbourhood is unobserved, so nothing here is known.
            return None;
        }

        let mut weighted = 0.0f32;
        let mut total_weight = 0.0f32;

        for index in 0..8usize {
            let offset = [
                (index & 1) as i32,
                ((index >> 1) & 1) as i32,
                ((index >> 2) & 1) as i32,
            ];

            // Trilinear weight: a corner on the far side of the query along an
            // axis is weighted by how far past it the query lies.
            let weight = [fraction.x, fraction.y, fraction.z]
                .iter()
                .enumerate()
                .map(|(axis, f)| if offset[axis] == 1 { *f } else { 1.0 - *f })
                .product::<f32>();

            let value = match single {
                Some(block) => {
                    let index = ((local[0] + offset[0])
                        + (local[1] + offset[1]) * BLOCK
                        + (local[2] + offset[2]) * BLOCK * BLOCK)
                        as usize;
                    if block.weight[index] <= 0.0 {
                        continue;
                    }
                    block.distance[index]
                }
                None => {
                    // Straddling a block face: this corner resolves through its
                    // own block.
                    let corner = [
                        base[0] + offset[0],
                        base[1] + offset[1],
                        base[2] + offset[2],
                    ];
                    let Some(block) = cache.get(self.block_of(corner)) else {
                        continue;
                    };
                    let index = (corner[0].rem_euclid(BLOCK)
                        + corner[1].rem_euclid(BLOCK) * BLOCK
                        + corner[2].rem_euclid(BLOCK) * BLOCK * BLOCK)
                        as usize;
                    if block.weight[index] <= 0.0 {
                        continue;
                    }
                    block.distance[index]
                }
            };

            weighted += value * weight;
            total_weight += weight;
        }

        if total_weight <= 0.0 {
            return None;
        }

        Some(weighted / total_weight * self.params.truncation)
    }

    /// Render the fused surface as a depth image viewed from `camera_to_world`.
    ///
    /// This is what makes frame-to-model tracking possible: aligning a live frame
    /// to a *render of the model* instead of to the previous frame is what stops
    /// tracking error accumulating frame by frame. The output is an ordinary
    /// depth image in the same convention as [`DepthImage`] -- camera-space z, in
    /// metres, `NaN` where no surface was found -- so the existing projective
    /// alignment can be pointed straight at it.
    ///
    /// Cost is per pixel and independent of how large the model has grown, which
    /// matters: real scans reach 150k blocks and 600 MB, so anything that walked
    /// the volume once per frame could not keep up.
    ///
    /// # How a ray is marched
    ///
    /// The integrator only ever writes a voxel within `truncation` of a measured
    /// surface, so a voxel that was never observed proves the nearest surface is
    /// *further* than the truncation band. Two consequences are used here:
    ///
    /// * In unobserved space the ray may stride a full truncation band without
    ///   stepping over anything, which is what keeps empty space cheap.
    /// * When the march finally meets a negative sample, the last positive sample
    ///   is either a real distance or the band width, and either is a sound lower
    ///   bound for interpolating where the crossing lies.
    ///
    /// Inside the band the stored value is a genuine (truncated) distance, so
    /// stepping by the sample itself -- sphere tracing -- lands on the surface
    /// from the first positive sample onwards.
    pub fn raycast(
        &self,
        intrinsics: &Intrinsics,
        camera_to_world: &Isometry3<f32>,
        width: usize,
        height: usize,
    ) -> RaycastImage {
        let truncation = self.params.truncation;
        // Half a band, not a voxel. A voxel-sized floor forces up to
        // `range / voxel_size` steps -- 400 at the defaults -- and it exists
        // only to stop a distance near zero stalling the march. Overshooting the
        // surface by up to half a band is harmless: the sample lands inside the
        // band on the far side, reads negative, and the crossing is then placed
        // by interpolation between the last two samples.
        let min_step = truncation * 0.5;
        let (near, far) = (self.params.min_depth, self.params.max_depth);
        let h = self.params.voxel_size;

        let mut depth = vec![f32::NAN; width * height];
        let mut normals = vec![Vector3::zeros(); width * height];
        let mut normal_cache = BlockCache::new(&self.blocks);

        for y in 0..height {
            for x in 0..width {
                // `direction.z` is 1, so the ray parameter *is* the camera-space
                // depth, and no separate division is needed at the hit.
                let direction = intrinsics.back_project(x as f32 + 0.5, y as f32 + 0.5, 1.0);
                if let Some(hit) =
                    self.march(camera_to_world, &direction, near, far, truncation, min_step)
                {
                    let index = y * width + x;
                    depth[index] = hit;

                    let cam_point = direction * hit;
                    let world_point = transform_point(camera_to_world, &cam_point);
                    if let Some(world_normal) =
                        self.normal_at_world(&mut normal_cache, &world_point, h)
                    {
                        let mut cam_normal = camera_to_world.rotation.inverse() * world_normal;
                        if cam_normal.dot(&cam_point) > 0.0 {
                            cam_normal = -cam_normal;
                        }
                        normals[index] = cam_normal;
                    }
                }
            }
        }

        RaycastImage {
            width,
            height,
            depth,
            normals,
        }
    }

    /// Unit normal in world space from central differences of the TSDF distance field.
    fn normal_at_world(
        &self,
        cache: &mut BlockCache<'_>,
        world: &Vector3<f32>,
        h: f32,
    ) -> Option<Vector3<f32>> {
        let dx = self.sample(cache, &(world + Vector3::new(h, 0.0, 0.0)))?
            - self.sample(cache, &(world - Vector3::new(h, 0.0, 0.0)))?;
        let dy = self.sample(cache, &(world + Vector3::new(0.0, h, 0.0)))?
            - self.sample(cache, &(world - Vector3::new(0.0, h, 0.0)))?;
        let dz = self.sample(cache, &(world + Vector3::new(0.0, 0.0, h)))?
            - self.sample(cache, &(world - Vector3::new(0.0, 0.0, h)))?;

        let grad = Vector3::new(dx, dy, dz);
        let length = grad.norm();
        if length > 1e-6 {
            Some(grad / length)
        } else {
            None
        }
    }

    /// One ray, from the near plane to the far plane. Returns the camera-space
    /// depth of the first surface crossing, or `None`.
    fn march(
        &self,
        camera_to_world: &Isometry3<f32>,
        direction: &Vector3<f32>,
        near: f32,
        far: f32,
        truncation: f32,
        min_step: f32,
    ) -> Option<f32> {
        // One cache for the whole ray: successive samples are centimetres apart
        // and a block is 16 cm on a side, so most steps re-use the same block.
        let mut cache = BlockCache::new(&self.blocks);
        let mut t = near;

        // The sample before the current one. Before the ray reaches the band
        // nothing is known, but "at least a band away" is known, and it is
        // exactly the value that makes the first crossing interpolation honest.
        let mut previous_t = near;
        let mut previous_distance = truncation;

        while t <= far {
            let world = transform_point(camera_to_world, &(direction * t));
            let observed = self.sample(&mut cache, &world);

            // An unobserved voxel is not "distance zero": it means nothing came
            // within a band of this point, so the surface is at least that far.
            let distance = observed.unwrap_or(truncation);

            if distance <= 0.0 {
                let denominator = previous_distance - distance;
                let hit = if denominator.abs() > f32::EPSILON {
                    previous_t + (t - previous_t) * (previous_distance / denominator)
                } else {
                    t
                };
                return Some(hit.clamp(near, far));
            }

            previous_t = t;
            previous_distance = distance;

            // In the band, step by the distance itself. Outside it, stride.
            t += match observed {
                Some(distance) => distance.max(min_step),
                None => truncation,
            };
        }

        None
    }

    /// Whether a depth reading is a usable measurement for this field.
    ///
    /// Both bounds matter. Anything outside the sensor's working range is either
    /// absent or junk, and junk becomes real geometry once it is fused in.
    #[inline]
    fn is_usable(min_depth: f32, max_depth: f32, depth: f32) -> bool {
        depth.is_finite() && depth >= min_depth && depth <= max_depth
    }

    /// Extract a triangle mesh by naive surface nets.
    ///
    /// Surface nets places one vertex per sign-changing cube (at the average of
    /// its edge crossings) and joins neighbouring vertices into quads. It needs
    /// no case tables and produces far fewer, better-shaped triangles than
    /// marching cubes on the same field.
    pub fn extract_mesh(&self) -> Mesh {
        // Keyed by the cube's minimum corner, in voxel coordinates.
        let mut cube_vertices: HashMap<[i32; 3], u32> = HashMap::new();
        let mut mesh = Mesh::default();

        let voxel_size = self.params.voxel_size;

        // Pass 1: one vertex per sign-changing cube.
        for cell in self.blocks.keys() {
            for local in 0..VOXELS_PER_BLOCK {
                let lx = (local % BLOCK as usize) as i32;
                let ly = ((local / BLOCK as usize) % BLOCK as usize) as i32;
                let lz = (local / (BLOCK as usize * BLOCK as usize)) as i32;

                let cube = [
                    cell[0] * BLOCK + lx,
                    cell[1] * BLOCK + ly,
                    cell[2] * BLOCK + lz,
                ];

                let Some(position) = self.cube_vertex(cube, voxel_size) else {
                    continue;
                };

                mesh.vertices.push(position);
                cube_vertices.insert(cube, (mesh.vertices.len() - 1) as u32);
            }
        }

        // Pass 2: for each grid edge with a sign change, join the four cubes
        // around it. This is what turns isolated vertices into a surface.
        let push_quad = |mesh: &mut Mesh, quad: [u32; 4], flip: bool| {
            let [a, b, c, d] = if flip {
                [quad[3], quad[2], quad[1], quad[0]]
            } else {
                quad
            };
            mesh.triangles.push([a, b, c]);
            mesh.triangles.push([a, c, d]);
        };

        for cell in self.blocks.keys() {
            for local in 0..VOXELS_PER_BLOCK {
                let lx = (local % BLOCK as usize) as i32;
                let ly = ((local / BLOCK as usize) % BLOCK as usize) as i32;
                let lz = (local / (BLOCK as usize * BLOCK as usize)) as i32;

                let voxel = [
                    cell[0] * BLOCK + lx,
                    cell[1] * BLOCK + ly,
                    cell[2] * BLOCK + lz,
                ];

                let Some(here) = self.voxel_distance(voxel) else {
                    continue;
                };
                let outside = here >= 0.0;

                // +x edge
                if let Some(next) = self.voxel_distance([voxel[0] + 1, voxel[1], voxel[2]]) {
                    if (next >= 0.0) != outside {
                        let quad = [
                            [voxel[0], voxel[1] - 1, voxel[2] - 1],
                            [voxel[0], voxel[1], voxel[2] - 1],
                            [voxel[0], voxel[1], voxel[2]],
                            [voxel[0], voxel[1] - 1, voxel[2]],
                        ];
                        if let Some(indices) = lookup(&cube_vertices, &quad) {
                            push_quad(&mut mesh, indices, outside);
                        }
                    }
                }

                // +y edge
                if let Some(next) = self.voxel_distance([voxel[0], voxel[1] + 1, voxel[2]]) {
                    if (next >= 0.0) != outside {
                        let quad = [
                            [voxel[0] - 1, voxel[1], voxel[2] - 1],
                            [voxel[0] - 1, voxel[1], voxel[2]],
                            [voxel[0], voxel[1], voxel[2]],
                            [voxel[0], voxel[1], voxel[2] - 1],
                        ];
                        if let Some(indices) = lookup(&cube_vertices, &quad) {
                            push_quad(&mut mesh, indices, outside);
                        }
                    }
                }

                // +z edge
                if let Some(next) = self.voxel_distance([voxel[0], voxel[1], voxel[2] + 1]) {
                    if (next >= 0.0) != outside {
                        let quad = [
                            [voxel[0] - 1, voxel[1] - 1, voxel[2]],
                            [voxel[0], voxel[1] - 1, voxel[2]],
                            [voxel[0], voxel[1], voxel[2]],
                            [voxel[0] - 1, voxel[1], voxel[2]],
                        ];
                        if let Some(indices) = lookup(&cube_vertices, &quad) {
                            push_quad(&mut mesh, indices, outside);
                        }
                    }
                }
            }
        }

        mesh
    }

    /// Surface-nets vertex for the cube whose minimum corner is `cube`, in world
    /// metres. `None` unless every corner was observed and the cube straddles a
    /// sign change.
    fn cube_vertex(&self, cube: [i32; 3], voxel_size: f32) -> Option<Vector3<f32>> {
        let mut corners = [0.0f32; 8];

        for (index, corner) in corners.iter_mut().enumerate() {
            let offset = [
                (index & 1) as i32,
                ((index >> 1) & 1) as i32,
                ((index >> 2) & 1) as i32,
            ];
            *corner = self.voxel_distance([
                cube[0] + offset[0],
                cube[1] + offset[1],
                cube[2] + offset[2],
            ])?;
        }

        let all_outside = corners.iter().all(|c| *c >= 0.0);
        let all_inside = corners.iter().all(|c| *c < 0.0);
        if all_outside || all_inside {
            return None;
        }

        // Average the crossings on the 12 cube edges.
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7), // along x
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7), // along y
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // along z
        ];

        let mut sum = Vector3::zeros();
        let mut crossings = 0;

        for (a, b) in EDGES {
            let (da, db) = (corners[a], corners[b]);
            if (da >= 0.0) == (db >= 0.0) {
                continue;
            }

            let denominator = da - db;
            let t = if denominator.abs() > f32::EPSILON {
                da / denominator
            } else {
                0.5
            };

            let pa = corner_offset(a);
            let pb = corner_offset(b);
            sum += pa + (pb - pa) * t;
            crossings += 1;
        }

        if crossings == 0 {
            return None;
        }

        let local = sum / crossings as f32;

        Some(Vector3::new(
            (cube[0] as f32 + local.x) * voxel_size,
            (cube[1] as f32 + local.y) * voxel_size,
            (cube[2] as f32 + local.z) * voxel_size,
        ))
    }
}

#[inline]
fn corner_offset(corner: usize) -> Vector3<f32> {
    Vector3::new(
        (corner & 1) as f32,
        ((corner >> 1) & 1) as f32,
        ((corner >> 2) & 1) as f32,
    )
}

fn lookup(vertices: &HashMap<[i32; 3], u32>, cubes: &[[i32; 3]; 4]) -> Option<[u32; 4]> {
    Some([
        *vertices.get(&cubes[0])?,
        *vertices.get(&cubes[1])?,
        *vertices.get(&cubes[2])?,
        *vertices.get(&cubes[3])?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Translation3;

    fn intrinsics(width: usize, height: usize) -> Intrinsics {
        Intrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
        }
    }

    /// Analytically render the nearest intersection of each pixel ray with a
    /// sphere, returning depth in metres (`NaN` where there is no hit).
    fn render_sphere(
        width: usize,
        height: usize,
        intrinsics: &Intrinsics,
        center: Vector3<f32>,
        radius: f32,
    ) -> Vec<f32> {
        let mut depth = vec![f32::NAN; width * height];

        for v in 0..height {
            for u in 0..width {
                let direction = Vector3::new(
                    (u as f32 + 0.5 - intrinsics.cx) / intrinsics.fx,
                    (v as f32 + 0.5 - intrinsics.cy) / intrinsics.fy,
                    1.0,
                );

                let to_center = -center;
                let b = to_center.dot(&direction);
                let c = to_center.dot(&to_center) - radius * radius;
                let a = direction.dot(&direction);
                let discriminant = b * b - a * c;
                if discriminant < 0.0 {
                    continue;
                }

                let t = (-b - discriminant.sqrt()) / a;
                if t > 0.0 {
                    // dir.z is 1, so t is already the z of the hit point.
                    depth[v * width + u] = t;
                }
            }
        }

        depth
    }

    #[test]
    fn integrates_a_fronto_parallel_plane_at_the_right_depth() {
        let width = 64;
        let height = 64;
        let intrinsics = intrinsics(width, height);
        let plane_depth = 1.5f32;

        let depth = vec![plane_depth; width * height];
        let image = DepthImage::new(width, height, &depth);

        let mut volume = TsdfVolume::new(TsdfParams {
            voxel_size: 0.01,
            truncation: 0.04,
            ..TsdfParams::default()
        });

        let stats = volume.integrate(&image, &intrinsics, &Isometry3::identity());
        assert!(stats.updated_voxels > 0, "nothing was integrated");
        assert!(stats.new_blocks > 0);

        let mesh = volume.extract_mesh();
        assert!(!mesh.is_empty(), "no surface extracted");

        let (min, max) = mesh.bounds().expect("bounds");
        assert!(
            (min.z - plane_depth).abs() < 0.02 && (max.z - plane_depth).abs() < 0.02,
            "surface should sit at z ~ {plane_depth}, got {min:?}..{max:?}"
        );
        // The plane is 1.5 m away and the image is 64 px wide at fx=200, so it
        // should span roughly 64*1.5/200 = 0.48 m. Allow slack for the border.
        let span = max.x - min.x;
        assert!(span > 0.3 && span <= 0.5, "unexpected width {span}");
    }

    #[test]
    fn averages_repeated_observations_towards_the_mean() {
        let width = 32;
        let height = 32;
        let intrinsics = intrinsics(width, height);

        let mut volume = TsdfVolume::new(TsdfParams {
            voxel_size: 0.01,
            truncation: 0.04,
            ..TsdfParams::default()
        });

        // Two noisy observations either side of 1.0 m.
        for offset in [0.008f32, -0.008] {
            let depth = vec![1.0 + offset; width * height];
            volume.integrate(
                &DepthImage::new(width, height, &depth),
                &intrinsics,
                &Isometry3::identity(),
            );
        }

        let mesh = volume.extract_mesh();
        let (min, max) = mesh.bounds().expect("bounds");
        let mid = (min.z + max.z) * 0.5;

        // Averaging should land near the true 1.0 m, not on either sample.
        assert!(
            (mid - 1.0).abs() < 0.006,
            "expected averaged surface near 1.0 m, got {mid}"
        );
    }

    #[test]
    fn reconstructed_sphere_normals_point_outward() {
        let width = 160;
        let height = 160;
        let intrinsics = intrinsics(width, height);

        let center = Vector3::new(0.0, 0.0, 0.7);
        let radius = 0.15f32;

        // Integrate the same sphere from several viewpoints so the field closes.
        let mut volume = TsdfVolume::new(TsdfParams {
            voxel_size: 0.01,
            truncation: 0.05,
            ..TsdfParams::default()
        });

        let views = [
            Isometry3::identity(),
            Isometry3::from_parts(
                Translation3::new(0.0, 0.0, 2.0 * center.z),
                nalgebra::UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f32::consts::PI),
            ),
            Isometry3::from_parts(
                Translation3::new(0.0, -2.0 * center.z, center.z),
                nalgebra::UnitQuaternion::from_axis_angle(
                    &Vector3::x_axis(),
                    std::f32::consts::FRAC_PI_2,
                ),
            ),
            Isometry3::from_parts(
                Translation3::new(0.0, 2.0 * center.z, center.z),
                nalgebra::UnitQuaternion::from_axis_angle(
                    &Vector3::x_axis(),
                    -std::f32::consts::FRAC_PI_2,
                ),
            ),
        ];

        for view in &views {
            // Express the sphere centre in this view's camera frame.
            let center_camera = transform_point(&view.inverse(), &center);
            let depth = render_sphere(width, height, &intrinsics, center_camera, radius);
            volume.integrate(&DepthImage::new(width, height, &depth), &intrinsics, view);
        }

        let mesh = volume.extract_mesh();
        assert!(
            mesh.triangle_count() > 1000,
            "expected a dense sphere, got {}",
            mesh.triangle_count()
        );

        // Every triangle should face away from the sphere centre.
        let mut outward = 0;
        for triangle in &mesh.triangles {
            let normal = mesh.triangle_normal(triangle);
            if normal.norm_squared() == 0.0 {
                continue;
            }
            let centroid = mesh.centroid(triangle);
            if normal.dot(&(centroid - center)) > 0.0 {
                outward += 1;
            }
        }

        let ratio = outward as f32 / mesh.triangle_count() as f32;
        assert!(
            ratio > 0.98,
            "only {:.1}% of triangles face outward; winding is wrong",
            ratio * 100.0
        );

        // Vertices should sit on the sphere within about a voxel.
        let worst = mesh
            .vertices
            .iter()
            .map(|v| ((*v - center).norm() - radius).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02, "worst vertex radius error {worst} m");
    }

    #[test]
    fn empty_volume_extracts_an_empty_mesh() {
        let volume = TsdfVolume::new(TsdfParams::default());
        let mesh = volume.extract_mesh();
        assert!(mesh.is_empty());
        assert_eq!(volume.block_count(), 0);
    }

    #[test]
    fn depth_behind_the_camera_is_ignored() {
        let width = 16;
        let height = 16;
        let intrinsics = intrinsics(width, height);

        let mut depth = vec![f32::NAN; width * height];
        depth[0] = -1.0; // nonsense
        depth[1] = 0.0; // missing
        depth[2] = f32::INFINITY; // nonsense

        let mut volume = TsdfVolume::new(TsdfParams::default());
        let stats = volume.integrate(
            &DepthImage::new(width, height, &depth),
            &intrinsics,
            &Isometry3::identity(),
        );

        assert_eq!(stats.updated_voxels, 0);
        assert_eq!(volume.block_count(), 0);
    }

    /// A volume holding one fronto-parallel plane at `plane` metres, observed
    /// from the origin.
    fn volume_with_plane(width: usize, height: usize, plane: f32) -> (TsdfVolume, Intrinsics) {
        let intrinsics = intrinsics(width, height);
        let depth = vec![plane; width * height];

        let mut volume = TsdfVolume::new(TsdfParams {
            voxel_size: 0.01,
            truncation: 0.04,
            ..TsdfParams::default()
        });
        volume.integrate(
            &DepthImage::new(width, height, &depth),
            &intrinsics,
            &Isometry3::identity(),
        );

        (volume, intrinsics)
    }

    #[test]
    fn an_empty_volume_raycasts_nothing() {
        let volume = TsdfVolume::new(TsdfParams::default());
        let image = volume.raycast(&intrinsics(32, 32), &Isometry3::identity(), 32, 32);

        assert_eq!(image.hits(), 0, "an empty field cannot be rendered");
        assert!(image.depth.iter().all(|d| d.is_nan()));
    }

    #[test]
    fn raycast_recovers_the_plane_it_integrated() {
        let (width, height) = (64, 64);
        let plane = 1.5f32;
        let (volume, intrinsics) = volume_with_plane(width, height, plane);

        let image = volume.raycast(&intrinsics, &Isometry3::identity(), width, height);
        assert!(
            image.hits() > width * height / 2,
            "only {} of {} rays found the plane",
            image.hits(),
            width * height
        );

        // Border rays graze the edge of the integrated patch, so the middle is
        // what is checked. A hit should land within about a voxel.
        let mut worst = 0.0f32;
        for y in height / 4..(3 * height) / 4 {
            for x in width / 4..(3 * width) / 4 {
                let depth = image.depth[y * width + x];
                assert!(depth.is_finite(), "no surface at ({x}, {y})");
                worst = worst.max((depth - plane).abs());
            }
        }
        assert!(worst < 0.02, "worst depth error {worst} m");
    }

    #[test]
    fn raycast_returns_the_nearest_surface_not_the_farthest() {
        // Two bands, at 1 m and 2 m. A ray from the camera meets the near one
        // first; returning the far one would put the predicted model behind the
        // real surface and drive tracking in the wrong direction.
        let (width, height) = (48, 48);
        let (mut volume, intrinsics) = volume_with_plane(width, height, 1.0);

        let far = vec![2.0f32; width * height];
        volume.integrate(
            &DepthImage::new(width, height, &far),
            &intrinsics,
            &Isometry3::identity(),
        );

        let image = volume.raycast(&intrinsics, &Isometry3::identity(), width, height);
        let middle = image.depth[(height / 2) * width + width / 2];
        assert!(
            (middle - 1.0).abs() < 0.02,
            "expected the near plane at 1 m, got {middle}"
        );
    }

    #[test]
    fn a_pose_that_never_saw_the_surface_raycasts_nothing() {
        let (width, height) = (32, 32);
        let (volume, intrinsics) = volume_with_plane(width, height, 1.5);

        // Far past the model, still looking away from it.
        let away = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 50.0),
            nalgebra::UnitQuaternion::identity(),
        );

        let image = volume.raycast(&intrinsics, &away, width, height);
        assert_eq!(image.hits(), 0, "rays that never reach the band must miss");
    }

    #[test]
    fn a_small_motion_still_renders_the_surface() {
        // The model render is used at a *predicted* pose, which is never the pose
        // the surface was integrated from. It has to survive that offset.
        let (width, height) = (48, 48);
        let plane = 1.5f32;
        let (volume, intrinsics) = volume_with_plane(width, height, plane);

        let nudged = Isometry3::from_parts(
            Translation3::new(0.03, 0.0, 0.0),
            nalgebra::UnitQuaternion::identity(),
        );

        let image = volume.raycast(&intrinsics, &nudged, width, height);
        assert!(image.hits() > 100, "only {} hits", image.hits());

        // Viewed from 3 cm to the side, a fronto-parallel plane is still 1.5 m
        // away along the optical axis.
        let middle = image.depth[(height / 2) * width + width / 2];
        assert!(
            (middle - plane).abs() < 0.02,
            "expected ~{plane} m from the nudged pose, got {middle}"
        );
    }

    #[test]
    fn distance_at_is_none_where_nothing_was_observed() {
        let (width, height) = (32, 32);
        let (volume, _) = volume_with_plane(width, height, 1.5);

        // On the plane, in the band: a real distance.
        assert!(volume.distance_at(&Vector3::new(0.0, 0.0, 1.5)).is_some());
        // Nowhere near anything: unobserved.
        assert!(volume.distance_at(&Vector3::new(5.0, 5.0, 5.0)).is_none());
    }
}
