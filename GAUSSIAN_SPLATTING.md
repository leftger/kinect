# Gaussian Splatting from the Kinect scanner

A working plan for turning this handheld depth scanner plus a Gaussian-splatting
trainer into something that produces high-resolution, photoreal environments.

This is a plan, not a summary. It records what is verified, what is not, the
things that silently break, and the order to do the rest in. Numbers in it were
measured on this machine unless marked otherwise.

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
were read out of Brush's source and are enforced by tests — see §3.

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
- **The exporter is live-only.** A `.k2df` recording stores depth frames and
  nothing else, so a dataset needs someone holding the sensor through a fresh
  scan. This is the biggest workflow tax in the whole pipeline; see §5, M2.4.
- **`cargo test --workspace` does not build on this machine** (`turbojpeg-sys`
  wants NASM, pulled in by `kinect-one`'s default features). Pre-existing and
  unrelated; use `-p geom -p scan`.
- **`out/room.k2df` defeats both trackers.** Run through `scan replay`, both
  modes lock out within the first 25 frames and never recover: frame-to-model
  first rejects at frame 19, frame-to-frame at frame 22, and the pose then sits
  frozen for the rest of the capture (97 of 150 frames rejected). Whatever
  happens around frame 19–22 is a hard tracking failure, not drift. See §5,
  M2.1 — this is now the first thing to fix, because no drift comparison means
  anything on a capture that loses tracking.

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

The recurring difficulty is that the interesting failures need hardware. Three
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

What is *not* yet covered: an actual Brush parse of our output. See M1.

**Beware the synthetic case.** The synthetic trajectory reports an inlier ratio
of 0.683 for frame-to-model; real captures report 0.76–0.89. The synthetic scene
moves 6.3 cm per frame at 2 cm voxels, which is far harsher than a real handheld
scan. Do not tune acceptance thresholds against it.

---

## 5. Roadmap

Ordered by what unblocks the most. Each milestone has a definition of done that
is a measurement, not a feeling.

**Two tracks, because one needs hardware and one does not.** M1 needs the Kinect
plugged in. **M2.1 does not**, and it is the highest-value work available right
now: the current recording loses tracking at frame ~20, which makes every pose
statement from it unreliable. If the sensor is not on the desk, start at M2.1.

### M1 — Prove the exporter end to end

*Why first:* the dataset format was implemented from reading Brush's source and
has never been through the real parser. A silent mismatch invalidates everything
downstream, and it is cheap to rule out.

1. Plug in the Kinect and take a short colour scan:
   `scan live --color --dataset out/ds --frames 60 --out out/ds_scan.ply`
2. Build Brush (`rustc 1.97` is installed and clears its 1.88 floor; Vulkan
   compute works on this GPU — `cargo run --example vk_check` proves it).
3. Point Brush at `out/ds` and train.

**Done when:** Brush loads the dataset without error and produces a `.ply` of
splats that renders recognisably. Expect it to look *wrong* — poses drift — but
it must be recognisably the room.

*Also do:* a 60-frame scan is the right size. Do not spend 300 frames before the
round trip is proven.

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

**M2.1 — Stop losing tracking. This is the first priority.** `scan replay` on
`out/room.k2df` shows both modes healthy for about eighteen frames (inlier
ratios 75–92%) and then collapsing:

| | Frame-to-frame | Frame-to-model |
| --- | --- | --- |
| First rejection | frame 22 (19.9% inliers) | frame 19 (35.9% inliers) |
| Then | 78.7% but "implausible rotation", then 3.2%, 201 cm of motion | 28.6%, 25.2%, never recovers |
| Total rejected | 97 of 150 | 9 and climbing, then locked out |
| Pose after | frozen at [0.517 0.134 0.104] | frozen at [0.361 0.077 0.136] |

Once a frame is rejected the pose stops moving, so the next frame is measured
from a stale position, which is precisely the lock-out cascade the comments in
`odometry.rs` warn about. Something around frame 19–22 — a fast pan, a
reflective or featureless surface, or the sensor being covered — breaks the
correspondence and neither tracker gets it back.

Work it in this order:

1. **Find out what happens at frame 19.** Replay `--frames 30` and dump the
   trajectory and a few depth frames around it. This is a concrete, bounded
   investigation and it is worth doing before any tuning.
2. **Log which gate trips and when.** The end-of-frame line already reports the
   reason; what is missing is the *sequence*. A rejection histogram per run
   would show lock-out versus genuine failure immediately.
3. **Consider recovery rather than prevention.** If a frame is unusable, the
   honest options are to keep tracking against the model (which does not move)
   or to re-localise against it, which is exactly what frame-to-model is for —
   but the render must be *searchable* over a wider radius, not just a few
   centimetres from the last pose. This is where frame-to-model would earn its
   place, and it is a better use of it than drift reduction.
4. **Check the capture itself.** One `.k2df` that defeats both trackers is thin
   evidence. Recording a slow, deliberate walk with no fast pans would separate
   "the tracker is fragile" from "this capture is hard".

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

**M2.4 — Store colour in `.k2df`.** Today a dataset requires a live scan, so
every pose experiment costs someone holding a sensor. Extending the recording
format (bump `VERSION`, keep depth-only files readable) to optionally carry
colour views makes the whole loop offline: record once, replay and re-export
freely. This is pure workflow leverage and it compounds with M2.2 — you cannot
iterate on poses cheaply while each iteration needs a fresh capture.

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

The honest sequencing: M1 tells you whether the plumbing works, M2 tells you
whether the poses are good enough, M3 tells you whether resolution is the
constraint. Only then is it clear whether M4 is needed at all.

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

---

## 8. Commands

```sh
# Build
cargo build --release -p scan

# Tests (the workspace target does not build here; see §2)
cargo test -p geom -p scan
cargo test --release -p geom -p scan

# Live colour scan straight into a dataset
target/release/scan live --color --dataset out/ds --frames 60 --out out/ds_scan.ply

# Offline: replay a recording, no sensor needed
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
