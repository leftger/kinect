# kinect

A handheld 3D scanner for the Kinect v2, written in Rust. It captures depth
frames, recovers the sensor's motion from depth alone, fuses the frames into a
TSDF volume, and extracts a triangle mesh.

There is no IMU and no GPS. Motion comes from the depth images, and drift is
addressed with loop closure rather than extra sensors.

## Quick start

### Try it without a Kinect

```sh
cargo build --release -p scan --features gui
./target/release/scan gui --synthetic --auto-start
```

That opens the scanner window and runs the full real-time path against an
analytic room. No sensor or recording is needed.

### Run a live scan

```sh
# One-time Linux USB setup
sudo cp platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# Check the sensor, then open the scanner window
cargo run --release -p probe
cargo build --release -p scan --features gui
./target/release/scan gui --color
```

For the command-line path:

```sh
./target/release/scan live --color --frames 300 --out scan.ply
```

See [Getting started](docs/getting-started.md) for platform requirements, build
features, and the first-scan checklist.

## What it does

- Captures, records, and replays depth and optional colour at 512x424.
- Tracks frame-to-frame motion with projective point-to-plane ICP, and can fall
  back to a local map of recent keyframes for a spatial re-localisation.
- Fuses frames into a voxel-hashed TSDF and extracts a triangle mesh.
- Writes PLY, OBJ, glTF, and GLB with per-vertex colour or a texture atlas.
- Detects revisits with loop closure, verifies each against a window of nearby
  keyframes, optimises a keyframe pose graph, and corrects the model.
- Applies a verified loop during the scan (`--online-loops`) or at the end, and
  can move cached submap surfaces instead of re-fusing every frame
  (`--incremental-rebuild`).
- Keeps unplaceable frames in submaps instead of discarding them.
- Exports a Nerfstudio/Brush dataset for Gaussian splatting.
- Runs a scanner GUI with a live 3D preview, source picker, transport, export,
  telemetry, and an optional trainer/Splats panel.

## Status

**Working**

- Live capture, `.k2df` recording, and sensor-free replay.
- Depth-only odometry, TSDF fusion, mesh extraction, trajectory export.
- Colour capture, registration, per-vertex colour, and texture atlases.
- A real closed-loop colour capture: 120/120 frames, one submap, one verified
  revisit, 10.9 cm maximum correction, and a trainable dataset.
- Keyframe-graph loop closure with multi-keyframe verification targets, optional
  online correction, an optional background ICP worker, and an incremental
  submap correction path.
- Gaussian-splatting dataset export and a tested Brush hand-off on synthetic and
  real data. The window can launch Brush and load its `.ply` into a basic splat
  preview.

**Limits**

- Drift is real on long handheld captures. Loop closure corrects a revisit, but
  the final model still depends on capture quality.
- Online correction moves cached submap surfaces rigidly and does not re-fuse the
  volume until the final pass. Incremental rebuild makes the same trade: it is
  fast, but it cannot remove drift *inside* a submap.
- The GPU decoder frees host CPU time but does not shorten a scan, because
  capture waits on USB.
- Fusion is the largest per-frame cost and grows with model size.
- A recognisable real Brush result still needs a cleaner capture or longer,
  higher-resolution training.

The long-form Gaussian-splatting plan and measurements are in
[GAUSSIAN_SPLATTING.md](GAUSSIAN_SPLATTING.md).

## How it fits together

```text
depth frame from USB
  -> decode            raw packet to metres
  -> odometry          pose relative to the last frame, local keyframe map
  -> TSDF fusion       integrate into a voxel-hashed volume
  -> loop closure      keyframe graph, online or end-of-scan correction
  -> mesh extraction   surface nets, then PLY / OBJ / glTF / GLB
  -> optional colour   register views, paint the mesh or atlas
  -> optional dataset  posed images, masks, and init.ply for a trainer
```

Read [How it works](docs/how-it-works.md) for tracking, fusion, mesh extraction,
loop closure, and submaps.

The loop-closure architecture is adapted from
[LIO-SAM](https://github.com/TixiaoShan/LIO-SAM): keyframe graph nodes, a local
window of keyframes as the verification target, and corrections applied to
cached pieces rather than an unconditional re-fuse. The front end is depth-only;
there is no lidar and no IMU.

## Documentation

| Guide | What is inside |
| --- | --- |
| [Getting started](docs/getting-started.md) | Hardware, USB permissions, builds, sensor check, first scan |
| [Command line](docs/cli.md) | Commands, options, recording/replay, loop closure, submaps |
| [Scanner window](docs/gui.md) | Sources, panels, viewport controls, transport, telemetry, Splats |
| [Colour](docs/colour.md) | Registration, visibility tests, colour modes, output formats |
| [Datasets and splats](docs/datasets.md) | Dataset layout, quality gate, Brush smoke test, real-capture results |
| [How it works](docs/how-it-works.md) | Pipeline, odometry, fusion, mesh extraction, loop closure |
| [Performance](docs/performance.md) | Measured timings, GPU decode results, profiling notes |
| [Development](docs/development.md) | Layout, tests, vendored driver, diagnostics, known gaps |
| [Gaussian-splatting plan](GAUSSIAN_SPLATTING.md) | Milestones, experiments, and open questions for the splat hand-off |

## Licence

No licence has been chosen for this code yet, so it is all rights reserved by
default. The vendored driver keeps its own MIT licence; the translated decode
kernels carry libfreenect2's Apache-2.0 / GPL-2.0 dual licence. See
[Development](docs/development.md#licence) for the exact file paths.
