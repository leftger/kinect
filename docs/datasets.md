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

## Orientation

The Kinect v2 reads out mirrored frames, so the sensor's own frame is a
reflection of the room. `scan` flips the exported mesh back by default, and the
dataset follows it, so a trained splat model lines up with the mesh and with the
real room.

The whole dataset is reflected together: images and masks are reversed left to
right, each pose becomes `M T M` (with `M = diag(-1, 1, 1)`, still a proper
rotation), the principal point moves to `width - cx`, and the seed cloud's x is
negated. Reflecting only some of these trains to nothing, which is why it is
done in one place. `--mirror` writes the raw sensor frame instead, again all at
once.

A dataset written before this change is mirrored relative to the room: re-export
it, or flip the splat model's x when you use it.

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

## Installing Brush

Brush publishes prebuilt binaries at
<https://github.com/ArthurBrussee/brush/releases>. On Apple silicon:

```sh
gh release download v0.3.0 -R ArthurBrussee/brush -p 'brush-app-aarch64-apple-darwin.tar.xz*'
shasum -a 256 -c brush-app-aarch64-apple-darwin.tar.xz.sha256
tar -xJf brush-app-aarch64-apple-darwin.tar.xz
install -m 755 brush-app-aarch64-apple-darwin/brush_app ~/.cargo/bin/brush
brush --version        # brush-cli 0.3.0
```

The archive's binary is `brush_app`; the README calls it `brush`. The window
looks for `brush`, `brush_app` and `brush-cli` on `PATH`. Run Brush from a
directory it can write to: it creates a `target/` kernel cache in its working
directory and panics with a `No such file or directory` otherwise (the window
does this for you).

## Brush smoke test without a sensor

```sh
scan synth --dataset out/ds-synth --frames 24 --out out/synth.ply

brush out/ds-synth --total-steps 100 --export-every 50 \
    --export-path /tmp/brush-synth
```

Brush 0.3.0 names its flags `--total-steps` and `--export-name`, whose default
is `export_{iter}.ply`, so each export keeps its own file. The flag was
`--total-train-iters` in the older build this repo was first checked against.
Brush prints nothing by default; `RUST_LOG=brush_cli=info` makes it report
`Refine iter N, M splats.` and `Done training!`, which is what the window reads.

An earlier Brush loaded all 24 views and wrote a 119 MB PLY with 504,178 splats
and the expected position, scale, rotation, opacity, and SH properties.

**Small images.** Brush 0.3.0 reads exactly 16,387 bytes from each image to learn
its format and fails any smaller file with `early eof`. A real 512×424 frame is
190–435 KB and never hits this, but the synthetic room's 160×120 frames are
about 11 KB and did. The exporter pads every image past that with an ancillary
PNG `tEXt` chunk placed after the image data: pixels are untouched and every
decoder ignores it.

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

After export, the Splats panel finds Brush, runs it as a child process, and
shows each intermediate model in the viewport while training continues, with a
progress bar (steps, splat count, elapsed, rough time left). **Train after
export** starts a run as soon as a dataset is written. See
[Scanner window](gui.md#splats).

## A real room

A 900-frame handheld capture of a living and dining room, replayed with loop
closure and exported as a dataset (900 views of 512×424, a 198,341-point seed):

| | |
| --- | --- |
| Brush | 0.3.0, Apple silicon GPU |
| Steps | 3,000, a preview every 1,000 |
| Time | 42 s, about 75 steps per second |
| Splats | 206,502 at step 201, 274,117 at the end |
| Exports | 50, 56 and 62 MB |

That is a smoke test of the pipeline, not a quality result: 3,000 steps is a
tenth of a normal run, and nobody has yet judged how the splats look.

## Current limits

- The exported geometry and poses are usable, but visual splat quality depends
  on a clean capture, enough views, and enough training.
- The preview renders DC colour only; higher SH bands are not evaluated.
- A recognisable real result still needs a cleaner, slower static capture or
  longer, higher-resolution training.
