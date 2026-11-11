# How it works

The scanner has no IMU and no GPS. Motion is recovered from depth, and drift is
corrected by recognising places the sensor has already visited. An IMU would
help a little; GPS is useless indoors, and the Kinect v2 has neither.

```text
depth frame from USB
  -> decode            raw packet to metres          (vendored libfreenect2 port)
  -> odometry          pose relative to the last frame
  -> TSDF fusion       integrate into a voxel-hashed volume
  -> mesh extraction   naive surface nets, then PLY, OBJ, glTF, or GLB
  -> optional colour   register views, paint the mesh or atlas
  -> optional dataset  posed images, masks, and init.ply for a trainer
```

## Odometry

Odometry uses projective data association: each point in the current frame is
projected into the previous depth image to find its correspondence, with an
occlusion test on depth disagreement, rather than searching a spatial index.

That change alone took tracking from 928 ms to 81 ms per frame with no loss of
accuracy. Point-to-plane ICP with Huber weighting and a coarse-to-fine pyramid
does the solving.

Frames whose solution fails acceptance thresholds are dropped rather than fused,
and the reference frame still advances so that one bad frame cannot lock out the
rest of the scan.

Frame-to-model tracking is also implemented. It renders the fused surface and
aligns against that instead of the previous frame. On the closed-loop capture it
did worse than frame-to-frame tracking, so it is optional.

## Fusion

Fusion uses a voxel-hashed TSDF, one hash entry per 8x8x8 block. Each block is
enumerated once per frame and the projective update runs over its voxels.

Memory scales with the inverse cube of `--voxel`. Fusion is the largest
remaining per-frame cost on a large model; see [Performance](performance.md).

## Mesh extraction

Mesh extraction is naive surface nets: one vertex per sign change, then
connectivity is recovered by, for each grid edge with a sign change, joining the
four surrounding cubes.

It is far less code than marching cubes and produces cleaner quads at the cost
of some detail.

## Loop closure

Loop closure detects revisits and rebuilds the model from corrected poses.

1. A new frame is considered a keyframe after 2 cm / 3 degrees of motion, or
   every five frames.
2. Pose proximity proposes nearby keyframes. On colour scans, a compact
   appearance signature (8x8 grayscale plus RGB histogram) can also propose a
   candidate outside the pose radius.
3. Spatial ICP verifies the candidate. A reverse alignment and round-trip gate
   must also pass.
4. Accepted edges are weighted by inlier coverage and residual.
5. A pose graph optimises odometry edges plus loop edges.
6. If the correction is implausible, the weakest loop is dropped and the graph
   is retried.
7. The model is rebuilt from the corrected poses.

The loop index has bounded working memory. The most recent 80 keyframes keep
full-resolution clouds; older signatures keep every second point. When a
candidate points at an old keyframe, its cloud is rebuilt from retained depth for
that verification and then dropped again. Telemetry reports loops, attempts,
dropped loops, points held, and rehydrated clouds.

Loop closure keeps every depth frame (about 850 KB each), so a 200-frame scan
holds around 170 MB of depth. Colour views add about 1.66 MiB each.

## Submaps

Submaps are on by default. When the tracker cannot place a frame and the spatial
search does not recover it, that frame starts a new submap at the failed
alignment's candidate pose. Tracking and fusion continue locally.

On `out/room.k2df` this turns 77 fused / 73 skipped into 150 / 0 across 19
fragments, using 220 MB instead of 464 MB. No revisit is verified on that
recording, so the fragments stay at guessed origins.

Finalising each submap extracts its surface, about a second at each boundary.
`--no-submaps` restores the single model and the skipped frames.

## Colour

Colour is not fused into the TSDF. Accepted views are registered, visibility
tested, scored, and painted onto the finished mesh or packed into a texture
atlas. See [Colour](colour.md).

## Further reading

- [Command line](cli.md)
- [Colour](colour.md)
- [Performance](performance.md)
- [Gaussian-splatting plan](../GAUSSIAN_SPLATTING.md)
