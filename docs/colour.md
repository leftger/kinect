# Colour

The scanner can capture the Kinect's colour stream, register it into the depth
camera's grid, and paint the finished mesh or texture atlas.

## Capture, record, replay

- `live --color` captures colour and paints the finished mesh.
- `record --color` stores the registered colour in the `.k2df` file.
- `replay` paints from a colour recording when present.
- A recording made without `--color` is depth only and has no colour to paint.
- `record --drain-color` reads and discards colour packets without storing them.

Examples:

```sh
scan live --color --frames 300 --out scan.ply
scan live --color --color-mode blend --color-depth-tolerance 0.03 --out scan.ply
scan live --color --loop-closure --out room.obj
scan live --color --out room.glb
scan live --color --out room.gltf

scan record --color --out capture.k2df
scan replay --in capture.k2df --out mesh.ply

scan synth --dataset out/ds-synth --frames 24 --out out/synth.ply
```

## How registration works

Colour is registered as each frame arrives
(`Registration::undistort_depth_and_color`). A view is a 512x424 RGB image plus
the undistorted depth and registration mask from the same frame.

That costs about 1.66 MiB per kept view, or about 1 GiB at 600 views, on top of
the depth frames loop closure retains (about 850 KB each).

Projecting a vertex needs only the depth intrinsics. The view's own depth is the
occlusion test, the same one projective odometry already uses.

A sample is kept only when:

- that depth agrees with the vertex;
- the registration mask says the pixel was copied;
- the sample clears the image margin;
- a known normal faces the camera.

Without the depth test, a vertex on a far wall seen through a doorway would be
painted with whatever is in front of it.

## Sample scoring

Surviving samples are scored:

```text
score = incidence × centrality × proximity × agreement × tracking
```

| Term | Meaning |
| --- | --- |
| `incidence` | How squarely the camera looks at the surface |
| `centrality` | Prefers the middle of the frame |
| `proximity` | Prefers a nearer camera |
| `agreement` | How much of the depth tolerance is left |
| `tracking` | Inlier ratio of that frame |

## Colour modes

`--color-mode` chooses the per-vertex PLY colour. The default is `best`.

| Mode | Behaviour |
| --- | --- |
| `best` | Keeps the single highest score, after per-view exposure correction in linear light |
| `blend` | Mixes the top three scores in linear light, weighted by score, with the same correction |
| `average` | Equal mean of every visible sample in stored byte values; no exposure correction |

The visibility test is the same for all three.

## Texture atlases

`.obj`, `.gltf`, and `.glb` write a texture atlas. A triangle is assigned to the
one source view that passes the visibility test on all three corners and the
triangle centroid, and that has the best score.

`blend` and `average` apply to per-vertex PLY colour; atlas triangles get the
best source view.

Atlas charts are the bounding boxes of those faces, shelf-packed with a
replicated gutter, and never larger than 8192 on a side. If the native charts
would overflow that square, they are downscaled together.

## Exposure correction

The correction is one scalar per view, estimated from vertices that view shares
with the best-observed view. It is not a model of the colour camera's exposure,
gain, or gamma; those values are kept for diagnostics.

A correction needs several consistent overlaps, ignores near-black samples, and
is clamped.

## Depth tolerance

`--color-depth-tolerance` defaults to 2 cm (`0.02`). That is about the depth
noise at typical range and still allows a vertex that sits between voxels.

The earlier painter used 5 cm, which let colour through thin structure. The
value must be greater than zero: too tight and nothing is visible; too loose and
colour bleeds through walls.

## Why colour is not fused

Colour is not stored in the TSDF volume. That costs no extra voxels, and once
the model is final every accepted view that saw a surface can contribute, not
only the frames that arrived while that surface was being integrated.

A frame the tracker rejects is not kept: its pose was never used to build the
model, so its colour is not used to paint it. The end-of-scan line reports how
many colour views were kept against how many were dropped with rejected frames.

## Loop closure and colour

Views are stored at the odometry pose of the frame they came from. When a loop
correction is accepted, the volume is rebuilt and each view is rewritten to the
corrected pose of its frame before anything is painted.

A refused correction leaves both the model and the views where odometry put
them. Painting from the old poses after the mesh had moved would colour the
surface from the wrong camera.

## Output files

| `--out` | Files |
| --- | --- |
| `scan.ply` | Per-vertex `uchar red/green/blue` in that file |
| `room.obj` | `room.obj`, `room.mtl`, `room.png` |
| `room.gltf` | `room.gltf`, `room.bin`, `room.png` |
| `room.glb` | One binary GLB, geometry and PNG inside it |

PLY has no texture coordinates in its core format, so it stays per-vertex. That
is what MeshLab, CloudCompare, and Blender read, and it is the colour
`--color-mode` controls.

An uncompressed PNG of an 8192 atlas would be about 192 MB, most of it the flat
fallback colour around the charts. The PNG is deflate-compressed so that empty
region collapses.

The end-of-scan line for an atlas reports its size, how many triangles were
painted, and the unpainted remainder. A PLY reports how many vertices were
painted and how many samples contributed to each.

The atlas is built in the volume's frame and only then mirrored, so positions,
normals, and winding flip together and the texture coordinates stay on the image
that was just painted.

## Cost

The colour stream delivers at about a third of the depth rate.

On the Linux machine this was first measured on, waiting for a packet took a
frame from roughly 270 ms to 530 ms. On the M4 Pro, a 20-frame colour scan and a
30-frame depth-only scan both took about 3 s of wall time, so colour still
lowers throughput. Capturing colour every Nth frame would pay that less often.

## Alignment check

The alignment this relies on was checked before any of it was written:
`crates/scan/examples/color_check.rs` writes the registered colour and the depth
discontinuities overlaid. The depth edges land on colour edges with a one-pixel
offset.
