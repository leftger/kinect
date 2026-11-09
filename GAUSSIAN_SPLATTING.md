# Gaussian Splatting from the Kinect scanner

A working plan for turning this handheld depth scanner plus a Gaussian-splatting
trainer into something that produces high-resolution, photoreal environments.

This is a plan, not a summary. It records what is verified, what is not, the
things that silently break, and the order to do the rest in. Numbers in it were
measured on this machine unless marked otherwise.

> **Status after the scanner window and Splats panel.** `scan gui` now owns the
> capture, the dataset hand-off, and the trainer job. The window can scan a live
> sensor (optionally writing a `.k2df` as it reconstructs), replay a colour
> recording, or run the synthetic room; it writes the dataset on Export, then
> launches a configured Brush command as a child process, streams its log, and
> cancels it. It still does not render the trained splats. This document marks
> that boundary explicitly: what already lives in the window, and what should
> come next.

---

## 1. The split, and why it is the right one

Two systems, each doing the half it is good at:

| | Produces | Role |
| --- | --- | --- |
| **This scanner** | Dense depth, metric poses, a fused surface | Geometry and pose authority |
| **3DGS trainer** (Brush) | Appearance, view synthesis | Photometry |

The overlap is that a trainer needs *posed images*, and the scanner already holds
exactly that: `ColorView` in `crates/geom/src/coloring.rs` carries a full 6-DoF
`Isometry3` pose, intrinsics, RGB, and a validity mask per fused frame.

**The structural advantage: metric scale.** A COLMAP-driven 3DGS run is
scale-ambiguous — it recovers a scene up to an unknown similarity. Poses from
this scanner are in real metres, so splats come out at true scale with no
normalisation step. That is worth protecting: never introduce a scale-free
alignment into the pipeline.

**The structural risk: pose error.** 3DGS treats poses as ground truth and has
no pose optimisation of its own (checked: nothing in Brush's `brush-train`
touches depth or does bundle adjustment). A pose that is 2 m wrong does not
produce slightly fuzzy splats, it produces a contradiction the optimiser cannot
resolve. Everything about pose quality lands squarely on this repo.

---

## 2. Where things stand

### Built and verified

**`--dataset` exporter** (`crates/scan/src/dataset.rs`). Writes a Nerfstudio
dataset: `transforms.json`, `images/`, `masks/`, `init.ply`. Format requirements
were read out of Brush's source and are enforced by tests — see §3. The same
export is reachable from `scan gui`: the Output panel has an optional dataset
directory, and Export writes the mesh, trajectory and dataset together. The
worker's log reports the dataset path, view count and initial-point count.

**Colour recordings.** `record --color` writes a version-2 `.k2df` with the
registered colour views, and `replay --in FILE --dataset DIR` exports a dataset
from one. The window's Replay source uses the same path. This is no longer a
live-only workflow; the old M2.4 is done.

**Scanner window** (`crates/scan/src/gui/`). `scan gui` and `live --viewer`
open the same window: live/record/replay/synthetic sources, start/pause/stop/
reset/export, a navigable TSDF preview, the dataset field, a dataset result
card after Export, and a Splats panel that launches a configured Brush child
process. It does not render splats. `Scanner::quality` marks unmerged submaps
or heavy frame loss as not trainable before the panel starts a trainer.

**`TsdfVolume::raycast`** (`crates/geom/src/tsdf.rs`). Renders the fused surface
as a depth image, which is what any frame-to-model work needs.

| Measurement | Value |
| --- | --- |
| Render bias vs the frame that built the model | **+0.16 mm mean**, 10 mm worst, over 5800 px |
| Render vs source coverage | 5484 trackable vs the frame's 5477 |
| Cost, 96×80 render, release | **0.06 s** (was 0.40 s before the block cache) |
| Cost, 512×424 render, release | ~2.3 s (measured through `scan replay`) |

**Frame-to-model tracking**, opt-in via `--frame-to-model`. Not yet shown to
reduce drift — see §5, M2.

### Known limitations

- **Colour is 512×424, not 1080p.** Registration copies colour into the depth
  grid, so the dataset is capped at depth resolution before training starts.
  Lifting this needs the depth→colour extrinsics; see §5, M3.
- **The exporter needs colour views.** A depth-only scan or depth-only `.k2df`
  has no dataset to write. `live --color`, `record --color` and the synthetic
  room do. The window checks live and Record at Start and rejects a depth-only
  replay before it starts.
- **The window launches the trainer but does not render splats.** The Splats
  panel runs a child process and reports the exported `.ply`; loading that file
  into the viewport is still M1.6.
- **The paths are still text fields.** The dataset and recording paths are
  preflighted at Start, but there is no native folder picker yet; that is the
  one M0 item left.
- **`cargo test --workspace` does not build on this machine** (`turbojpeg-sys`
  wants NASM, pulled in by `kinect-one`'s default features). Pre-existing and
  unrelated; use `-p geom -p scan`.
- **`out/room.k2df` still defeats the tracker around frame 22.** The default
  frame-to-frame path fuses 77 of 150 frames, and the failures are first views of
  surfaces the last frame never saw. The spatial search (`--relocalize`) recovers
  none of them: there is no overlap to align to (see §5, M2.1). Submaps now keep
  those views (150/150 fused across 19 fragments). A synthetic front-side-return
  test verifies a cross-submap loop edge and a 0.20 m rebuild correction; this
  capture has no verified revisit, so its fragments stay at guessed origins. A
  real closed-loop capture is still needed. No drift comparison means anything
  on a capture that loses tracking.

### The window's role

The window owns the scanner front end and the trainer job, but not the splat
renderer. The boundary is deliberate and matches the split in §1:

```text
scan gui (live | replay | synthetic)
  -> tracking + fusion + colour views
  -> final poses after loop closure
  -> dataset (transforms.json, images/, masks/, init.ply)
  -> trainer child process (Brush, from the Splats panel)
  -> splat .ply
  -> external viewer, or the window after M1.6
```

| Step | Current owner | Where it lives |
| --- | --- | --- |
| Capture, tracking, fusion | this repo | `scan gui` worker |
| Colour views and poses | this repo | `ColorView`; exported by `--dataset` |
| Dataset files | this repo | Output panel / `--dataset DIR` |
| Splat training | external Brush | a terminal, outside the window |
| Splat rendering | external viewer | outside the window |

The workflow in the window today:

1. Choose **Live** with **Colour** ticked, **Record** with **Colour** ticked,
   **Replay** of a colour recording, or **Synthetic** (always coloured).
2. Set **Output → dataset directory**. For **Record**, set the `.k2df` path in
   **Source**; Start preflights both paths.
3. **Start**, then **Stop** or **Export**. A Record session writes its `.k2df`
   while it reconstructs; `finish_scan` writes the mesh and the dataset in one
   pass, after loop closure has moved the colour views to their final poses.
4. The telemetry panel reports the exported mesh and
   `dataset: N views, M initial points in DIR`. That is the hand-off point for
   an external trainer.

The dataset is deliberately written from the *final* pose set. Exporting before
loop closure would hand the trainer the drifted poses the scanner is trying to
remove. A Record session's `.k2df` can be replayed later to re-export without
the sensor.

What the window does **not** do yet, and must not be described as doing:

- start a trainer or monitor its progress,
- load the trainer's `.ply` and render Gaussians.

M1.5 and M1.6 below are the milestones that move those last steps into the
window. Until they land, the window's viewport is a TSDF mesh/point preview, not
a splat viewer.

---

## 3. The format contract

Hard-won, and every item here fails *silently* or with a confusing error if
broken. Do not re-derive these from the Nerfstudio docs alone — Brush reads a
subset and rejects things the docs permit.

| Requirement | Why |
| --- | --- |
| `frames` is a **required** array | `JsonScene.frames` has no serde default |
| `camera_model` must be **absent** or `"PERSPECTIVE"` | `"PINHOLE"` hits `Some(other) => Err`. The obvious spelling is rejected outright |
| `transform_matrix` is **OpenGL camera-to-world** | Brush applies `c2w.y_axis *= -1; c2w.z_axis *= -1` on load. Our pose is world-from-camera, x right / y down / z forward, so the y and z axes are negated |
| Masks live in a directory named `masks` | `find_mask_path` looks for that component and matches the path suffix plus the file stem |
| `init.ply` carries **only** `x,y,z,red,green,blue` | `sh_count` counts `r/g/b/red/green/blue`, so RGB becomes SH DC. Adding `scale_0`/`opacity`/`rot_0`/`f_dc_*` replaces good defaults (1.8 cm Gaussians, identity rotation) with **zeros** |
| Do **not** write `f_dc_*` *and* `red/green/blue` | `sh_count` becomes 6 and `sh_rest_coeffs()[..3]` indexes past the end |

Two more invariants that are ours, not Brush's:

- **Everything in a dataset must share one frame.** Images, poses and `init.ply`
  are all written in the sensor frame. The `--mirror` flag flips the exported
  *mesh* and must never touch a dataset: mirrored poses with unmirrored images
  cannot train.
- **`red/green/blue` is read as `u8 / 254`.** Writing floats would be read by
  `visit_f32` instead; both work, but do not mix them across a file.

---

## 4. How to verify each layer without a sensor

The recurring difficulty is that the interesting failures need hardware. Four
harnesses make most of it testable offline:

1. **Synthetic room.** `render_room` in `crates/scan/src/odometry.rs` ray-traces
   a box interior analytically, so poses are known exactly and drift is
   measurable against truth. Use it for tracking changes.
2. **Recorded captures.** `out/*.k2df` are real 150-frame depth captures.
   `scan replay` needs no sensor, so *tracking and fusion changes can be
   validated against real data today*. This is the strongest available signal.
3. **Format checks.** `cargo test -p scan dataset` asserts the JSON shape, the
   pose convention (by round-tripping through Brush's own conversion, not a copy
   of our arithmetic), mask semantics and the `init.ply` property set.
4. **The window without a display.** The GUI's tests are headless: the session
   worker, a synthetic scan, the dataset export and the software renderer all
   run without opening a window (`cargo test -p geom -p scan --features gui`).
   A synthetic GUI run is the cheapest way to exercise the same `finish_scan`
   dataset path a live scan uses. Start it, then press **Export** or close the
   window:

   ```sh
   target/release/scan gui --synthetic --auto-start --frames 12 --voxel 0.04 \
       --dataset out/ds-synth --out out/synth.ply
   ```

   This proves the hand-off, not the appearance: the synthetic room is a
   fixture, and its poses are known rather than tracked from a real scene.

**Brush parse and a short train (synthetic, done).** The headless form of the
same export is `scan synth`:

```sh
target/release/scan synth --dataset out/ds-synth --frames 24 --out out/synth.ply
~/Projects/brush/target/release/brush-cli out/ds-synth \
    --total-train-iters 100 --export-every 100 \
    --export-path /tmp/brush-synth --export-name export.ply
```

Brush loaded all 24 views, ran 100 iterations and wrote a 119 MB PLY with
504,178 splats and the expected position/scale/rotation/opacity/SH properties.
The 24-frame synthetic run also crossed a submap boundary at frame 23, so the
parse covers a dataset with two fragments. This proves the format and trainer
hand-off; it does not prove that a real capture renders recognisably, because
the synthetic room is a fixture and its poses are known rather than tracked.

What is *not* yet covered: a real capture through Brush. See M1.

**Beware the synthetic case.** The synthetic trajectory reports an inlier ratio
of 0.683 for frame-to-model; real captures report 0.76–0.89. The synthetic scene
moves 6.3 cm per frame at 2 cm voxels, which is far harsher than a real handheld
scan. Do not tune acceptance thresholds against it.

---

## 5. Roadmap

Ordered by what unblocks the most. Each milestone has a definition of done that
is a measurement, not a feeling.

**Three tracks, because they need different things.** M1's real half needs the
Kinect; Brush is built and the synthetic half is proven. **M0 and M2.1b do not
need hardware.** M2.1b is now in: 150/150 frames land in 19 submaps, and a
synthetic front-side-return test proves the loop edge moves a submap and rebuilds
one model; a real capture with a genuine loop is still needed for the real-data
case. The next pose work is that capture, then M2.2 drift. If the sensor is not
on the desk, the M0 folder picker, M1.5's interactive click-through, and M1.6 are
the remaining software work; M1.6 needs a trained result.

### M0 — Finish the dataset side of the window (done, except the folder picker)

Landed:

1. **Record source.** The window writes a version-2 `.k2df` while it
   reconstructs. Tick Colour to store the colour views a dataset needs. Stop,
   Reset and a sensor dropout all close the recording through
   `finish_recording`, so the file is usable after an interrupted scan.
2. **Dataset preflight.** Start checks that the dataset and recording paths
   have a writable nearest ancestor, and rejects a live or Record dataset with
   Colour off before a worker starts. A depth-only replay is rejected when the
   recording is opened.
3. **Dataset result card.** `finish_scan_with_report` returns what was written;
   the telemetry panel shows the dataset directory, view count, initial points,
   skipped views and the exported mesh counts, not only the log.

Remaining:

4. **Folder picker.** Replace the free-text dataset and recording paths with a
   native directory chooser where a dialog backend exists; keep the text fields
   as the fallback.

**Done when:** a colour capture can be recorded, replayed and exported as a
dataset entirely from the window, and the result is visible in the window. This
is met with the text fields; the folder picker is polish.

### M1 — Prove the exporter end to end

*Why first:* a silent format mismatch invalidates everything downstream, and it
is cheap to rule out. The synthetic parse/train in §4 now clears the format;
what remains is a real capture.

1. **Synthetic format proof — done.** `scan synth` wrote a 24-view colour
   dataset, and the installed `brush-cli` loaded it, trained 100 iterations and
   exported a 504,178-splat PLY. The same run crossed a submap boundary, so the
   parse covers a two-fragment dataset too.
2. **Real capture.** Plug in the Kinect and take a short colour scan, either
   from the CLI:
   `scan live --color --dataset out/ds --frames 60 --out out/ds_scan.ply`
   or from the window: **Live** + **Colour** + dataset directory, Start, Export.
3. Point Brush at a real `out/ds` and train.

**Done when:** a real capture loads without error and produces a `.ply` of
splats that renders recognisably. Expect it to look *wrong* — poses drift — but
it must be recognisably the room. The synthetic half is already done.

*Also do:* a 60-frame scan is the right size. Do not spend 300 frames before the
round trip is proven.

### M1.5 — A trainer job in the window

*Why after M1:* the window should not orchestrate a trainer until the dataset
has passed through Brush once. The synthetic Brush run in §4 clears the format;
the panel can now assume a readable dataset.

**Implemented.** The **Splats** panel in `scan gui` has:

- trainer executable (default `brush-cli`), dataset directory, output
  directory, and iteration count;
- Start / Cancel, status, elapsed time, and a scrolling log;
- Brush-shaped arguments (`<dataset> --total-train-iters N --export-every N
  --export-path DIR --export-name export.ply`);
- the dataset directory auto-filled from the last export, with a **use last
  export** button;
- an exported dataset with quality warnings is marked *not trainable* and
  Start requires **Train anyway**.

`gui/trainer.rs` runs the child on its own thread, merges stdout and stderr
into a bounded 240-line log, kills the child on Cancel, and joins it when the
window closes. Three headless tests cover a missing dataset, a successful run
that writes the PLY, and cancellation.

**Still open:** the full click-through with the real `brush-cli` (a display and
a manual start, not a sensor), arbitrary extra trainer arguments, and parsing a
loss value rather than showing raw lines. The worker covers the process-control
half; the synthetic Brush run covers the command half.

**Done when:** a dataset exported from the window trains from the window,
Cancel actually stops the child process, and the output path is reported. The
viewport does not have to show the splats yet.

### M1.6 — Show the trained splats (optional, after M1.5)

The current viewport is a triangle/point z-buffer (`gui/render.rs`). A 3DGS
result is anisotropic Gaussians; drawing it correctly needs sorted, blended
billboards or an alpha-composited GPU pass. Do not call the current preview a
splat viewer before this exists.

Minimum useful version:

- load the trainer's `.ply` (positions, scales, rotations, opacity, colour/SH),
- render screen-aligned billboards with depth sorting and alpha blending,
- reuse the existing orbit/pan/zoom camera,
- toggle between the TSDF mesh and the splat result.

**Done when:** the window shows the trained result beside the mesh it came from,
and the two agree on pose and scale.

### M2 — Make the poses good enough

The measured baselines from `out/` logs, all "distance from origin" at the end of
a capture (only true drift if the operator returned to the start — for a closed
loop it is, otherwise it is mostly path length):

| Run | Frames | Fused | Ends |
| --- | --- | --- | --- |
| `out/room.k2df`, this session | 150 | 53 (97 skipped) | 1.259 m |
| recorded earlier | 150 | 131 | 2.151 m |
| recorded earlier | 25 | 25 | 0.451–0.663 m |

Two separate problems are visible in that table and they need different fixes,
and **the first one dominates**: on this capture both trackers lose the plot
early, so the table's endpoint distances are not measuring drift at all.

**M2.1 — Stop losing tracking. This is the first priority.** The bounded-gap
recovery below is in; `scan replay` on `out/room.k2df` now reports **77 fused /
73 skipped** and ends 2.822 m from origin. The first failure is frame 22 (17.7%
inliers); frame 23 then matches frame 22 at 88.0% but is rejected as an
"implausible rotation" because its absolute pose is still anchored at the last
trusted frame. The rejection map for the capture is:

**What frames 1–22 are not:** they are not a fast pan, and not junk depth. The
depth decoder clips every frame at 4.50 m, and no frame in the capture has a
single reading beyond the working range, so the "junk far past the rated range"
the README warns about is not what breaks it. What does change at frame 22 is the
scene: the median scene depth jumps from 1.97 m to 2.95 m in one frame while the
tracked pose moves 2.5 cm, which is a large view change the tracker is not
keeping up with. From frame 22 on the inlier ratio is bimodal — frames either
match well (75–90%) or not at all (1–45%) — and the rejection map for the whole
capture is:

```text
frames   1- 50:  ......................XXX.XXXXXXXXXXXXXX.XXXXXXXXX
frames  51-100:  XXXXXXX.XXXXXXXX.XXXXXXX.X.XX.XXX.X.XXXXX.XX.XXX..
frames 101-150:  .X..XX.XX....XX.X......XXXXXXXXXX.X.XXXXXXXXXX.XXX
```

**The structural defect, now fixed.** A rejected frame's motion was discarded
*and* the committed pose stood still, which is right — the motion was not
trusted — but nothing could then make up the difference: `pose * transform` adds
a single inter-frame step per *accepted* frame, so falling one frame behind was
permanent. A separate search origin now advances by the motion model even when
the committed pose does not, with the extrapolation capped at two frames and a
recovery bounded to a jump that allowance justifies. That removed the permanent
one-frame offset: the capture now fuses **77 of 150** frames (not 53) and ends
2.822 m from origin. It does not make the failing frames track.

**Still open, and it is the main problem.** 73 of 150 frames are still rejected.
The bounded recovery stops one rejection from becoming permanent; it does not
make the failing frames track. An earlier attempt at an eighth-resolution 0.40 m
pyramid level was *much worse* (19 fused frames): a radius that wide lets the
coarse level match the wrong surface and the finer levels refine a wrong answer.
The ladder in `Level::handheld_default` is at the point where widening it costs
more than it buys, so the next attempt used a separate spatial search instead.

**M2.1 search — measured, and it does not recover this capture.** Implemented as
`scan replay --relocalize` (off by default): on rejection, build a voxel-hashed
cloud of the last accepted frame (4 cm voxels, 12 cm normals), run
`geom::icp::align` with a 25 cm correspondence radius and up to 30 iterations,
and accept only at ≥55% inliers, ≤25 mm RMSE, and absolute translation/rotation
caps. Results on `out/room.k2df`:

| | default | `--relocalize` |
| --- | --- | --- |
| fused / skipped | 77 / 73 | 77 / 73 |
| accepted relocalisations | — | 0 |
| tracking | 136 ms/frame | 651 ms/frame |
| final position | 2.822 m | 2.822 m |

The debug trace explains why. On the transition frames the spatial search finds
48–98% correspondences inside 25 cm, but the RMS point-to-plane residual is
10–13 cm and the solve does not converge. Widening the radius to 50 cm raises the
inlier ratios to about 95% and the residual to 13–19 cm; still no convergence.
The strict gates refuse every candidate, which is the right outcome: the rejected
frame is the first view of a surface the last accepted frame never saw, so there
is no overlapping surface to align to. This capture falsifies the "wider search"
hypothesis: the failures are missing overlap, not a narrow projective window.
`spatial_icp_aligns_a_small_pair_of_room_views` covers the cloud pipeline, so a
failure here is a data problem, not a broken search.

**What the next frame proves — and why it is not enough.** Frame 23 matches frame
22 at 88% / 15 mm, and later pairs do the same. That is a good *relative*
measurement, but it does not anchor the pair in the world: the fit says where the
pair sits relative to itself, not where it sits relative to the trusted
trajectory. A deferred-fusion experiment that fused the rejected frame at
`pose(next) * motion^-1` raised the fused count from 77 to 108, but the pose
inherited the stale search origin, so the geometry would have been placed by a
guess with a good-looking residual. It was removed. For the same reason the
fast-motion acceptance is restricted to `gap == 0`, where the reference pose is
trusted; `--relocalize` is kept as an opt-in diagnostic.

**M2.1b — Submaps on a lost track (done, structurally).** When the projective
tracker rejects a frame and the spatial search does not save it, the scanner now
starts a new submap: the rejected frame becomes the new submap's first frame at
the failed alignment's candidate pose, the odometry restarts at that local
origin, and fusion continues into a fresh TSDF. A finished submap is extracted
to a world-space mesh and its volume is dropped. The boundary edge is omitted
from the pose graph, so a cross-submap loop edge can move the whole submap
instead of being pinned by the guess.

Results on `out/room.k2df`:

| | `--no-submaps` | default |
| --- | --- | --- |
| fused / skipped | 77 / 73 | **150 / 0** |
| submaps | 1 | **19** |
| memory | 463.8 MB | **220.7 MB** |
| tracking / fusion / submap | 152 / 108 / 0 ms per frame | 91 / 205 / 142 ms per frame |
| final position | 2.822 m | 1.778 m (current submap guess) |

Every frame is now accounted for. The `fused` count is no longer a quality
metric: it means "surfaced in some submap", not "tracked into the first one".
The per-frame log prints each boundary with its rejection reason. Finalising a
submap extracts its surface; that is the 142 ms/frame average submap cost,
concentrated in roughly a second at each boundary.

The synthetic test `a_revisit_merges_submaps_at_the_corrected_pose` now
demonstrates the merge: front views, a side view that forces a boundary, and a
returned front view 20 cm to the right form three submaps. Loop closure verifies
the return, the boundary edge is skipped, and the pose graph moves the third
submap by 0.1995 m and rebuilds one model. That is the plumbing working.

The remaining limitation is real data. Loop closure on `out/room.k2df` verifies
no revisits, so its 19 fragments stay at their candidate-pose origins and the
exported mesh is 5.66 M vertices / 10.1 M triangles with an 8.05 x 5.43 x 5.44 m
extent — a set of locally clean fragments, not a single room. A real capture
that returns somewhere it has already been is still needed.

**Still needed: a capture with ground truth.** A deliberate slow walk out and
back is the single most valuable piece of test data this project could have; every
pose number in this document is currently unfalsifiable without it.

**M2.2 — Then drift.** Frame-to-model is implemented but **not yet shown to
help, and on this recording it does worse** — it locks out three frames earlier
than the baseline. That is expected while the render-based target is less
forgiving than a raw frame, and it is why M2.1 comes first. On 20 real frames it
fuses 18/20 and ends 0.393 m from origin against the baseline's 20/20 and
0.544 m — suggestive, not conclusive, because neither is ground truth. Current
state on the synthetic trajectory:

```
frame_to_model=false: fused 41/41, mean inlier 0.937   (synthetic, harsh)
frame_to_model=true:  fused 18/40, mean inlier 0.683
```

The `frame_to_model_drifts_less_than_frame_to_frame` test is `#[ignore]`d with
those numbers. Two things to try, in order:

- **Make the render denser.** It is accurate (0.16 mm) and about as complete as
  the source frame, but "about as complete" still costs correspondences, and the
  gates were tuned for two equally complete frames. Feeding the **TSDF gradient**
  as the render's normals instead of cross products would help twice: gradient
  normals are defined wherever the field is observed (no four-neighbour
  requirement) and they are better conditioned than depth-image normals.
- **Then decide whether the gates should be mode-aware.** A frame whose target
  covers 80% of the image cannot achieve a 100% inlier ratio. Normalising the
  inlier gate by target coverage is principled; lowering a threshold until a test
  passes is not.

**Done when:** on a real recording, frame-to-model beats frame-to-frame on a
metric that is not self-referential — a closed-loop capture, where ending near
the origin *is* truth.

**M2.3 — Make the raycast cheaper, if it is still the bottleneck.** The block
cache already took a 96×80 render from 0.40 s to 0.06 s (6.7×), and it is now
~2.3 s/frame at 512×424. Further options, roughly in order of payoff:

- Bounding the march with a depth hint from the live frame. The ray currently
  walks the full 0.5–4.5 m range in 8 cm strides; the surface is known to be
  within a few centimetres of the live depth at the same pixel.
- A cached model AABB so rays can start at the model's near plane. Must be
  maintained incrementally — walking 150k blocks per frame is not affordable.
- Raycasting the tracking target at half resolution. Halves accuracy of the
  target; try only if the above is not enough.

**Do not** add a GPU raycast yet. The decide-later option is real, but the
current cost is tolerable offline and the complexity is not.

**M2.4 — (done) Store colour in `.k2df`.** The recording format has a version-2
payload with the registered colour, `record --color` writes it, and
`replay --dataset` exports from it. The window's Record source writes it and its
Replay source reads it, so the offline loop is closed.

**M2.5 — Stop a bad dataset before the trainer starts (implemented for the
last export).** A dataset with many rejected frames, or several unmerged
submaps, is not a training failure; it is a scan the operator should redo.
`Scanner::quality()` now returns frames, fused, rejected, submaps, loop edges,
rebuilt/refused, final distance, and a list of warnings. The CLI prints
`quality: trainable` or `NOT trainable` with each warning; the GUI dataset card
shows the same; the Splats panel disables a plain Start on a warned dataset and
requires **Train anyway**.

The remaining gap is an arbitrary dataset path typed into the Splats panel:
quality is known for the just-exported dataset, not for a directory chosen from
an earlier run. Re-inspecting an arbitrary dataset needs a small manifest or a
replay of its `transforms.json`, which is future work.

### M3 — Lift the resolution ceiling

The dataset is 512×424 because registration copies colour into the depth grid.
The colour sensor is 1920×1080. If "high resolution" is the goal, this is the
hard cap, and it is worth checking whether it is the *binding* one before
spending effort: train at 512×424 first (M1) and see whether appearance or pose
is what disappoints.

To lift it:

1. Expose the native colour frame through `Capture` (it is already processed and
   then discarded — see `capture.rs::next_frame`).
2. Get a real camera model for the colour sensor. `ColorParams` has
   `fx, fy, cx, cy`, so the intrinsics are there.
3. **The blocker: extrinsics.** `Registration` holds only Microsoft's
   `depth_to_color` polynomial, not a rigid 4×4. Options: fit one from the
   registration's own correspondences (the map is available for every depth
   pixel), or accept the depth-camera pose for the colour frame and quantify the
   error first.
4. Write HD images with the colour intrinsics, and convert the camera-to-world
   matrix the same way (the axis convention in §3 still applies).

**Done when:** a dataset whose images are 1920×1080 trains and renders sharper
than the 512×424 version on the same capture. If it does not, the extrinsics are
wrong.

### M4 — Tighten the coupling (only if M1–M3 leave something on the table)

Both of these are research-grade and neither should start before the loose
coupling is proven:

- **Depth supervision.** Add a loss on rendered depth so the Kinect's depth
  constrains the splats directly. Real work: Brush's trainer is Burn compute
  kernels in WGSL, and this means forking it.
- **Joint SLAM + splatting.** The literature here is exactly this problem:
  **SplaTAM**, **MonoGS**, **Gaussian-SLAM** are RGB-D 3DGS SLAM systems. Read
  them before building anything.

The honest sequencing: M0 removed the remaining terminal steps from the window
(except the folder picker), M1 tells you whether the plumbing works, M2 tells
you whether the poses are good enough, M3 tells you whether resolution is the
constraint. M1.5 and M1.6 are window conveniences that ride on M1; do not let
them get ahead of it. Only then is it clear whether M4 is needed at all.

---

## 6. Performance budget

Measured on this machine (Intel Broadwell-U Iris 6100, release build).

| Stage | Cost | Notes |
| --- | --- | --- |
| Depth decode | ~50 ms/frame | ~256 ms/frame total with filters on |
| Tracking, frame-to-frame | 69–129 ms/frame | Measured through `scan replay` |
| Tracking, frame-to-model | ~2344 ms/frame | Dominated by the 512×424 raycast |
| Fusion | 63–221 ms/frame | Scales with voxel size and model |
| Raycast render | 0.06 s @ 96×80 | 6.7× faster after the block cache |

**Debug builds are ~17× slower than release.** Every timing above is release.
Do not benchmark anything from `cargo test` without `--release`.

Model sizes are the reason a raycast (O(pixels)) was chosen over anything that
walks the volume: real scans reach **150k blocks / 616 MB**, and allocated blocks
are 4 KB each because they hold `distance` and `weight` per voxel.

The window adds a cost this table does not include: every `--preview-every`
frames it extracts a mesh from the TSDF and rasterises it on the CPU. That is
paid by the operator's frame rate, not by the dataset, and it has not been
benchmarked separately. Do not assume the numbers above describe a `scan gui`
session; if preview extraction is hurting capture, raise `--preview-every`.

---

## 7. Open questions

- **What happens at frame 19–22 of `out/room.k2df`?** Both trackers are healthy
  up to that point and then lose the pose permanently. This is now the highest
  value thing to understand in the project — see M2.1 — and it needs no
  hardware, only `scan replay` and a look at the depth frames.
- **Why does `out/room.k2df` fuse 53/150 here but 131/150 in the recorded log?**
  Partially answered: the 97 rejections are a genuine lock-out, not a reporting
  difference. But the 131/19 run is still unexplained, and if that run exists it
  is a far better baseline to compare against.
- **Is the endpoint distance actually drift?** None of the captures is confirmed
  to be a closed loop, and on this one the tracker is frozen for most of the
  capture, so "ends 1.26 m from origin" is meaningless as a drift figure. A
  deliberate slow walk out and back is the single most valuable piece of test
  data this project could have, and it should be recorded before further tuning.
- **How much of the frame-to-model gap is the metric rather than the tracker?**
  The inlier ratio is computed over *source* points while a render target cannot
  cover all of them. Worth deciding before tuning gates.
- **Would gradient normals close the gap on their own?** Plausible and cheap to
  test; it is the first thing in M2.2 for that reason.
- **Should the window own the trainer or shell out to it?** M1.5 assumes a child
  process with a log and a Cancel button. Embedding Brush would mean a library
  API it does not have. Decide after M1 proves the dataset is readable.
- **What does a useful splat preview cost?** M1.6 needs sorted, alpha-blended
  billboards. The current CPU rasteriser is enough for a mesh; a million
  Gaussians may not be. Measure before promising real time.

---

## 8. Commands

```sh
# Build the CLI, and the window
cargo build --release -p scan
cargo build --release -p scan --features gui

# Tests (the workspace target does not build here; see §2)
cargo test -p geom -p scan
cargo test -p geom -p scan --features gui   # includes the headless window tests
cargo test --release -p geom -p scan

# Live colour scan straight into a dataset, from the CLI
target/release/scan live --color --dataset out/ds --frames 60 --out out/ds_scan.ply

# The same scan from the window:
#   scan gui -> Live + Colour + Output/dataset directory: out/ds
#   Start, then Export. The telemetry panel reports the dataset hand-off.
#
#   Record + Colour writes a replayable .k2df while it reconstructs:
#   Source/Record, recording path out/room_color.k2df, then Start and Export.

# Synthetic dataset and window, no sensor at all (Export or close to write it)
target/release/scan gui --synthetic --auto-start --frames 24 --voxel 0.02 \
    --dataset out/ds-synth --out out/synth.ply

# Colour recording -> dataset, then the same recording through the window
target/release/scan record --color --out out/room_color.k2df --frames 60
target/release/scan replay --in out/room_color.k2df --dataset out/ds --out out/room.ply
target/release/scan gui --in out/room_color.k2df --auto-start \
    --dataset out/ds-gui --out out/room_gui.ply   # Export or close to write it

# Offline: replay a depth recording, no sensor needed
target/release/scan replay --in out/room.k2df --out out/room.ply
target/release/scan replay --in out/room.k2df --out out/room.ply --frame-to-model

# The known-failing comparison, run on demand
cargo test --release -p scan frame_to_model_drifts -- --ignored --nocapture
```

---

## 9. Invariants worth protecting

1. **Never let a dataset be mirrored.** Images, poses and the point cloud share
   one frame or the dataset is worthless.
2. **Never add Gaussian attributes to `init.ply`.** Colour only.
3. **Never normalise scale.** Metric poses are the reason to use this scanner.
4. **Never tune acceptance gates against the synthetic trajectory.** Real
   captures report 0.76–0.89 inlier ratios where the synthetic case reports
   0.68; the synthetic scene is deliberately harsher than reality.
5. **A rejected frame is a hole, not a small error.** 97/150 rejected is a
   broken scan even if the resulting trajectory looks smooth.
6. **The window writes the dataset before the trainer runs.** Training reads the
   final pose set, not the live scanner. A trainer job must not share the
   scanner's worker or its pose state.
7. **Do not call the viewport a splat viewer until it renders splats.** The
   current preview is a TSDF mesh/point rasteriser; M1.6 is the line between the
   two.
8. **A trainer is an external process with its own lifecycle.** Closing the
   window must not silently kill a running job, and Cancel must actually stop
   the child process, not just hide its output.
