# Datasets and Gaussian splatting

The scanner exports a Nerfstudio/Brush-style dataset for an external Gaussian
splatting trainer. It does not bundle a trainer.

The long-form plan, milestones, and measurements are in
[GAUSSIAN_SPLATTING.md](../GAUSSIAN_SPLATTING.md).

## What gets written

```text
DATASET_DIR/
  transforms.json
  images/
  masks/
  init.ply
```

- `transforms.json` contains the camera intrinsics and final poses.
- `images/` contains the posed colour views.
- `masks/` marks valid pixels.
- `init.ply` is a coloured seed cloud for the trainer.

The seed cloud is voxel-decimated to at most 250,000 points. The full surface
stays in the exported mesh.

## Requirements

A dataset needs colour views. Use one of:

```sh
scan live --color --out room.ply --dataset out/room-ds
scan record --color --out capture.k2df
scan replay --in capture.k2df --out room.ply --dataset out/room-ds
scan synth --dataset out/ds-synth --frames 24 --out out/synth.ply
```

A depth-only recording cannot produce a colour dataset.

The GUI preflights the dataset path. Tick **Colour** on Live or Record, or replay
a recording made with `record --color`. See [Scanner window](gui.md).

## Quality gate

The scanner checks the finished scan before training:

- no frames;
- no fused frames;
- fewer than half the frames fused;
- at least one third rejected;
- more than one unmerged submap;
- a refused loop correction.

A dataset with warnings is marked **not trainable**. The GUI still allows
training, but requires **Train anyway**. The command line prints the warnings.

## Brush smoke test without a sensor

```sh
scan synth --dataset out/ds-synth --frames 24 --out out/synth.ply

brush-cli out/ds-synth --total-train-iters 100 --export-every 100 \
    --export-path /tmp/brush-synth --export-name export.ply
```

Brush loaded all 24 views and wrote a 119 MB PLY with 504,178 splats and the
expected position, scale, rotation, opacity, and SH properties.

The 24-frame synthetic run also crosses a submap boundary at frame 23, so the
parse covers a dataset with two fragments. This proves the format and trainer
hand-off; it does not prove that a real room renders recognisably.

The synthetic fixture is harsher than a handheld scan, so the quality check
marks this run *not trainable*. That is expected for a format smoke test, not a
verdict on the exporter.

## Real closed-loop capture

The first real closed-loop capture is `out/closed-loop.k2df`.

| Metric | Result |
| --- | --- |
| Frames fused | 120 / 120 |
| Submaps | 1 |
| Colour views | 120 kept |
| Loop revisits | 1 after keyframe thinning |
| Pose-graph cost | 0.020 → 0.000 |
| Largest pose change | 10.9 cm |
| Final distance from origin | 0.542 m |
| Quality | trainable |

Brush loads and trains the dataset.

A 5,000-iteration run used 60 views at 256 px and a 199k-point seed, densified to
519,041 splats. It shows the couch, plant, and table, but is still visibly
smeared. A full ~30k-iteration run is the next step.

The exporter now caps `init.ply` at 250k points; the original 2.24M-vertex seed
made training impractically slow.

## Training from the window

After export, the Splats panel can run a configured Brush command as a child
process. It exposes the executable, dataset, output directory, and iteration
count, with Start/Cancel and a streamed log.

The trained `.ply` can be loaded into the viewport with Mesh/Splats,
Fit-splats, and First-person controls. See [Scanner window](gui.md#splats).

## Current limits

- The exported geometry and poses are usable, but visual splat quality depends
  on a clean capture, enough views, and enough training.
- The preview renders DC colour only; higher SH bands are not evaluated.
- A recognisable real result still needs a cleaner, slower static capture or
  longer, higher-resolution training.
