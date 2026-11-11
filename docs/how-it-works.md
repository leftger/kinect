# How it works

The scanner has no IMU and no GPS. Motion is recovered from depth, and drift is
corrected by recognising places the sensor has already visited. An IMU would
help a little; GPS is useless indoors, and the Kinect v2 has neither.

```text
depth frame from USB
  -> decode            raw packet to metres          (vendored libfreenect2 port)
  -> odometry          pose relative to the last frame
  -> global search     optional 4-DOF branch-and-bound fallback
  -> TSDF fusion       integrate into a voxel-hashed volume, optionally chunked
  -> mesh extraction   naive surface nets, then PLY, OBJ, glTF, or GLB
  -> optional colour   register views, paint the mesh or atlas
  -> optional dataset  posed images, masks, and init.ply for a trainer
  -> optional localize frozen prior map, trajectory only
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

When a frame is rejected, the spatial fallback does not align against one stale
frame. `--relocalize` keeps a ring of the most recent accepted keyframes and
merges the ones near the failed candidate into a local map, then runs the
spatial ICP against that. This extends LIO-SAM's scan-to-map step to a depth
camera: several nearby views cover the surface that a single narrow view misses,
which matters most when the sensor turns in place. The ring is bounded
(`--relocalize-keyframes`, default 6) and only built when the spatial retry is
actually needed.

`--relocalize-global` adds the missing global half: when that ICP still
fails, a multi-resolution likelihood grid is built from the local map and a
branch-and-bound search tries yaw plus translation over a bounded window. The
search is 4-DOF, not 6-DOF, because a handheld scanner is usually upright; a
full six-dimensional correlative search is too expensive for a depth frame. The
result is only an initial guess: point-to-plane ICP refines it and the ordinary
inlier and residual gates still decide whether the frame is accepted.

## Fusion

Fusion uses a voxel-hashed TSDF, one hash entry per 8x8x8 block. Each block is
enumerated once per frame and the projective update runs over its voxels.

`--fusion-filter` applies a motion filter: a frame that moved less
than 1 cm / 0.005 rad since the last *fused* frame is still tracked and retained
for loop closure, but it is not integrated. A forced refresh every five frames
keeps a stationary model averaging. Fusion is the largest per-frame cost, so
this is the cheapest way to recover time on a slow or stationary scan.

`--tsdf-chunk N` finishes the active volume into a cached chunk every N fused
frames and starts a fresh one. The odometry frame and `world_from_submap` do not
change, so this is a memory boundary inside a tracking submap, not a new pose
boundary. It bounds the active volume and the per-frame fusion cost without
introducing a boundary guess.

Memory scales with the inverse cube of `--voxel`. Fusion is the largest
remaining per-frame cost on a large model; see [Performance](performance.md).

## Mesh extraction

Mesh extraction is naive surface nets: one vertex per sign change, then
connectivity is recovered by, for each grid edge with a sign change, joining the
four surrounding cubes.

It is far less code than marching cubes and produces cleaner quads at the cost
of some detail.

## Loop closure

Loop closure detects revisits and corrects the model. It is split into
detection, verification, and correction, following LIO-SAM's keyframe
factor-graph structure rather than a single align-and-rebuild pass.

1. A new frame becomes a loop keyframe after 2 cm / 3 degrees of motion, or
   every five frames. Those keyframes are the graph's nodes; ordinary frames are
   corrected by interpolating between the keyframes on either side.
2. Pose proximity proposes nearby keyframes. On colour scans, a compact
   appearance signature (8x8 grayscale plus RGB histogram) can also propose a
   candidate outside the pose radius.
3. The verification target is a window of `--loop-window` keyframes on each
   side of the candidate (default 2), transformed into the candidate's frame and
   merged. This is LIO-SAM's `loopFindNearKeyframes`: one depth view is narrow,
   and the extra surface turns a marginal revisit into a verifiable one.
4. Spatial ICP verifies the candidate. A reverse alignment and round-trip gate
   must also pass.
5. Accepted edges are weighted by inlier coverage and residual.
6. A keyframe pose graph optimises keyframe odometry plus loop edges. A submap
   boundary has no odometry edge, so a loop edge can move the whole submap.
7. If the correction is implausible, the weakest loop is dropped and the graph
   is retried.
8. The correction is applied online, incrementally, or by a full rebuild.

`--loop-global` uses the same 4-DOF branch-and-bound search as relocalisation
when the drifted pose is too far out for ICP. The search supplies an initial
guess only; forward ICP, reverse ICP, and the round-trip gate are unchanged.

`--match-score S` adds a likelihood gate after ICP: the accepted transform is
scored against a surface grid built from the target. That is a different signal
from "points found neighbours" and catches repetitive geometry that happens to
fit. `S` is a mean likelihood in 0..1; zero disables the gate.

`--submap-graph` switches the optimiser from keyframe nodes to tracking-submap
nodes, with the keyframes rigidly attached. That is a sparse pose
adjustment shape. The graph is smaller and a correction moves whole submaps,
which matches the cached surfaces, but it cannot deform inside one. The keyframe
graph remains the default accuracy path.

`--graph-worker` runs the online submap-graph solve on a worker thread and
applies the result before the next frame's world pose is derived. This is the
local-SLAM/global-SLAM split: capture does not stall behind a
dense solve. `close_loops` flushes any in-flight job before the final pass.

`--loop-worker` moves only the ICP verification to a worker thread. Keyframe
selection stays on the capture thread, results are drained each frame, and
`close_loops` flushes anything still in flight. That keeps a candidate from
stalling a live scan without changing which loops are accepted.

`--online-loops` applies a verified loop as soon as it arrives. Future frames
use the corrected trajectory, and the cached submap surfaces are moved by a
rigid transform. `--incremental-rebuild` makes the same trade at the end of the
scan: it moves the cached surfaces instead of re-integrating every frame. Both
are fast and coherent, but a rigid submap move cannot fix drift *inside* a
submap. The default full rebuild re-fuses every kept frame at the corrected
poses, which is the accuracy path.

The loop index has bounded working memory. The most recent 80 keyframes keep
full-resolution clouds; older signatures keep every second point. When a
candidate points at an old keyframe, its cloud is rebuilt from retained depth for
that verification and then dropped again. Telemetry reports loops, attempts,
dropped loops, points held, rehydrated clouds, online corrections, and pending
worker jobs.

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
The surface stays in the submap's own frame with its world transform beside it,
so a correction can move it without touching the voxels. `--no-submaps` restores
the single model and the skipped frames.

`--tsdf-chunk N` is a separate, cheaper boundary: every N fused frames the
active volume is extracted and cached, and a fresh volume starts in the same
odometry frame. A chunk is not a pose-graph node and adds no boundary guess. It
bounds the active volume and the fusion cost on a long single-submap scan.

## Frozen-map localisation

`scan localize --in capture.k2df --map prior.ply` treats an existing point cloud
as an immutable map. Each frame gets a 4-DOF global search for an initial pose,
point-to-plane ICP refines it, and the same inlier and RMSE gates decide whether
the pose is returned. Nothing is fused and the prior is never modified; the
output is a trajectory, optionally written as a point cloud of positions.

The prior reader accepts ASCII and binary little-endian PLY. Only the vertex
positions are used; colours, normals, and faces are skipped. This is the depth
camera's version of pure localisation, not a general map
editor.

## Colour

Colour is not fused into the TSDF. Accepted views are registered, visibility
tested, scored, and painted onto the finished mesh or packed into a texture
atlas. See [Colour](colour.md).

## Further reading

- [Command line](cli.md)
- [Colour](colour.md)
- [Performance](performance.md)
- [Gaussian-splatting plan](../GAUSSIAN_SPLATTING.md)
