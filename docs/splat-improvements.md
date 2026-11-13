# Improving splats: using depth, and live training updates

Two questions came up while getting a real room through Brush: can the Kinect's
depth improve the splats, and how do other projects show training as it happens?
This page records what was checked, what was not, and the options.

Each claim below says whether it was **checked** (read in source or run here) or
is **not checked**. Nothing here has been measured on a quality metric.

## Using the Kinect's depth to improve splats

### What already uses depth

- **Poses.** They come from depth tracking plus loop closure, in metres. A splat
  trainer treats poses as ground truth and cannot correct them, so pose error
  sets a ceiling on sharpness. On the 900-frame room capture, loop closure moved
  poses by up to 26.7 cm and 4.7 degrees, so residual error is plausible.
- **The seed cloud.** `init.ply` is decimated from the fused depth surface and
  coloured (205,355 points on that capture), instead of coming from a
  structure-from-motion step.
- **Masks.** Pixels registration did not fill are masked out.

### What does not: depth supervision

Brush 0.3.0 cannot use depth maps. **Checked:** the released tag's trainer,
config and dataset-loader sources (`brush-train/src/train.rs`, `config.rs`,
`brush-dataset/src/scene.rs`) contain no mention of depth. Its losses are
photometric (L1 plus SSIM, optionally LPIPS) with opacity and scale
regularisers. The CLI exposes no depth option either.

So with Brush, depth can only help through poses, the seed cloud and masks.

### Options

| Option | What it is | Status |
| --- | --- | --- |
| Use a trainer with a depth loss | `GaussianSplatting.jl` (a Julia trainer) has depth supervision | Read, not run. See below |
| Add a depth term to Brush | It is open-source Rust | Not checked whether its renderer can output depth, so the effort is unknown |
| Prune floaters against the fused surface | Delete splats far from the depth-built mesh after training. Needs no trainer change | Not built |
| Refine poses | Adjust poses against the photometric and depth error. Brush has no pose optimisation | Not built |
| Use native 1080p colour | Not depth, but our images are capped at 512x424 | Known limit, see [Datasets](datasets.md) |

**GaussianSplatting.jl.** **Checked:** its depth supervision reads depth maps from
a `depths/` directory beside the images (8-bit, 16-bit or float images), and its
three loss modes (`ssi`, `ssi_disparity`, `ssi_depth`) are all scale-and-shift
invariant. That design is for monocular depth networks, whose output has no true
scale, and it is fitted to a structure-from-motion point cloud at startup. Metric
Kinect depth would work with it but would discard its absolute scale.
**Not checked:** whether it loads our Nerfstudio-style `transforms.json`
dataset.

### What the exporter would need

Not built. Each `ColorView` already carries depth "from the same frame and the
same grid" as its colour, because registration copies colour into the depth grid.
Depth would therefore be pixel-aligned with each image and need no extrinsics.
A 16-bit PNG in millimetres is the obvious encoding. It would have to be flipped
left to right with the images when the dataset is un-mirrored, or the depth and
the image would disagree.

### Kinect depth is not clean

These are general properties of the sensor, **not checked** against this
capture: noise grows with range, depth edges produce flying pixels, concave
corners add multipath error, and dark, shiny or glass surfaces drop out. A depth
loss would need a validity mask, a robust loss such as Huber, and probably less
weight at range. For reference, the tracker's alignment error on the room capture
was about 4.5 mm RMSE, which says the geometry is consistent, not that raw depth
is accurate at every pixel.

### How to tell whether a change helped

Not run. Brush can hold out views (`--eval-split-every`, `--eval-every`). Train a
baseline with that first and keep its held-out numbers, so any change has
something to beat. Depth supervision should mostly help geometry (floaters,
flat walls, textureless regions) rather than texture detail, but that is an
expectation, not a result.

## Live training updates

### How GaussianSplatting.jl does it

**Checked** in `src/gui/worker.jl`. Training and viewing share one process and one
model:

- One worker task owns the GPU and runs both training steps and view renders. The
  viewer rasterises the same arrays the trainer is updating, so nothing is
  exported, written or parsed.
- Each loop iteration does one training step. It re-renders if the camera changed,
  or if it trained and more than 0.1 s has passed. The code comment gives this as
  "at most 10 FPS to not starve `step!`".
- Running both on one task keeps densification, which reallocates the model
  arrays, from racing a render.
- The UI publishes a copy of the camera under a lock and wakes the worker. The
  worker renders into a back buffer and swaps it with the front buffer under the
  lock. Step, loss and splat count are atomic values the UI reads directly.
- Rasterisation runs on the GPU (CUDA, Metal and AMD backends).

### Why this repo cannot do the same today

Brush is a separate process, so the only channels are the files it exports and
the lines it logs. **Measured here** on the room capture: an export was 50 to
62 MB at 274,000 splats, the viewer loads one in about 0.3 s, and the CPU splat
renderer takes about 0.5 s per frame in a release build. Exporting every second
would need roughly 50 to 60 MB a second of writes plus about 0.8 s of viewer work
per update (0.3 s to load, 0.5 s to render), which uses most of the second, and
it would slow training. That is an estimate from these figures, not a run.

### Options

- **Brush's own viewer.** `--with-viewer` runs inside Brush's process, which is
  the same design as the Julia project. It is the **Brush window** checkbox in the
  Splats panel. A full 30,000-step run completed with it on. The window itself
  was not seen, so how it looked is **not checked**.
- **Rerun logging.** Brush has `--rerun-enabled` and `--rerun-log-splats-every`;
  its own help calls splat logging heavy. **Not tried.**
- **Train in-process.** Link Brush's Rust training crates into `scan` and read the
  live splats, plus a GPU splat renderer to replace the CPU one. **Not
  evaluated**: whether the crates can be embedded this way, and the effort, are
  unknown.

## Related

- [Datasets and splats](datasets.md): dataset layout, installing Brush, the
  orientation of the exported dataset and results on a real room.
- [Scanner window](gui.md#splats): the Splats panel, progress bar and live
  preview.
- [GAUSSIAN_SPLATTING.md](../GAUSSIAN_SPLATTING.md): the original plan and what
  it verified.
