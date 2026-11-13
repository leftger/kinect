# Command line

The `scan` binary has five main workflows plus a sensor-free synthetic
generator:

| Command | Use it for |
| --- | --- |
| `gui` | Scanner window with a live preview |
| `live` | Capture a new scan from the sensor |
| `record` | Write a `.k2df` recording without reconstructing |
| `replay` | Reconstruct from a `.k2df` recording |
| `localize` | Localise a recording or the live sensor against a frozen PLY map |
| `synth` | Render the analytic room and write a dataset |

Build first:

```sh
cargo build --release -p scan --features gui
```

Omitting `--features gui` leaves `gui` unavailable but keeps the other commands.

## Synopsis

```text
scan gui    [--in FILE | --synthetic] [--auto-start] [--frames N]
            [--out mesh.ply] [--dataset DIR] [--color] [--preview-every N]
            [--playback-fps F] [--loop-closure] [--online-loops]
            [--loop-worker] [--loop-window N] [--incremental-rebuild]
            [--loop-global] [--match-score S] [--submap-graph] [--graph-worker]
            [--submap-frames N] [--submap-metres M]
            [--fusion-filter] [--tsdf-chunk N] [--relocalize]
            [--relocalize-global] [--relocalize-keyframes N] [--no-submaps]

scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--color]
            [--color-mode best|blend|average] [--color-depth-tolerance M]
            [--gpu] [--loop-closure] [--online-loops] [--loop-worker]
            [--loop-window N] [--incremental-rebuild] [--loop-global]
            [--match-score S] [--submap-graph] [--graph-worker]
            [--submap-frames N] [--submap-metres M]
            [--fusion-filter] [--tsdf-chunk N] [--relocalize]
            [--relocalize-global] [--relocalize-keyframes N] [--viewer]
            [--dataset DIR] [--no-submaps]

scan record --out capture.k2df [--frames N] [--no-filter] [--drain-color]
            [--gpu] [--color]

scan replay --in capture.k2df [--out mesh.ply] [--voxel M] [--frames N]
            [--loop-closure] [--online-loops] [--loop-worker]
            [--loop-window N] [--incremental-rebuild] [--loop-global]
            [--match-score S] [--submap-graph] [--graph-worker]
            [--submap-frames N] [--submap-metres M]
            [--fusion-filter] [--tsdf-chunk N] [--relocalize]
            [--relocalize-global] [--relocalize-keyframes N] [--no-submaps]
            [--dataset DIR]

scan localize (--in capture.k2df | --live) --map prior.ply [--frames N]
            [--out trajectory.ply]

scan synth  --dataset DIR [--frames N] [--out mesh.ply] [--voxel M]
```

## Commands

### `gui`

Opens the scanner window. Pick a live sensor, Record, a recording, or the
synthetic room; start, pause, stop, adjust parameters, and export.

- Needs a build with `--features gui` and a display.
- Does not need a sensor: `scan gui --synthetic --auto-start` runs the full
  real-time path.
- Record captures while reconstructing and writes a `.k2df` for later replay.
- The output field can also write a Gaussian-splatting dataset.

See [Scanner window](gui.md) for the panels and controls.

### `live`

Captures and reconstructs as it goes.

```sh
scan live --frames 300 --color --out scan.ply
```

`--dataset DIR` writes a Nerfstudio/Brush dataset alongside the mesh. See
[Datasets and splats](datasets.md).

### `record`

Stores frames in a `.k2df` file for offline work:

```sh
scan record --out capture.k2df --frames 300 --color
```

- With `--color`, the registered colour is stored too.
- Without it, the recording is depth only.
- `--drain-color` reads and discards colour packets without storing them.
- A recording is deterministic, so replaying a prefix is a repeatable test case.

### `replay`

Reconstructs from a recording and needs no sensor:

```sh
scan replay --in capture.k2df --out mesh.ply
```

If the recording carries colour, the mesh is painted from it and
`--dataset DIR` can write a dataset. A depth-only recording cannot be painted or
used for a colour dataset.

### `localize`

Localises a recording or the live sensor against a frozen prior map and writes
a trajectory:

```sh
scan localize --in capture.k2df --map prior.ply --out trajectory.ply
scan localize --live --map prior.ply --frames 60
```

The prior is a PLY point cloud, ASCII or binary little-endian. Only vertex
positions are read; colours, normals, and faces are skipped. Non-finite and
sentinel vertices from mesh PLYs are discarded. The prior is indexed once and
its global-search grid is cached; each frame first runs ICP from the previous
pose, and only a failed ICP falls back to the 4-DOF global search. Nothing is
fused and the prior is never changed. Without `--out`, the poses are printed
only.

A map saved by `scan` is mirrored in X unless it was exported with `--mirror`,
and `localize` compares against the sensor frame. It does not matter which you
have: if no frame has been located yet and one fails, that frame is retried
against an X-flipped copy of the prior. The first orientation that locates is
used for the rest of the run, and a line says which one it was.

`--frames N` limits how many recording frames or live frames are localised; live
defaults to 60. The first frame starts from identity, so the recording should
begin near the prior map's origin or within the global search radius. A room
with repetitive geometry can be ambiguous from one view; the fallback recovers
most later frames, but the first pose is the hardest.

### `synth`

Renders the analytic room, writes the mesh, and writes a colour dataset. It
needs no sensor and no display:

```sh
scan synth --dataset out/ds-synth --frames 24 --out out/synth.ply
```

This is the cheapest way to exercise the Brush hand-off.

## Common options

| Option | Meaning |
| --- | --- |
| `--frames N` | Capture or replay frame limit; default `60` |
| `--voxel M` | TSDF voxel size in metres; default `0.01` |
| `--out FILE` | Mesh path; extension selects the format |
| `--dataset DIR` | Also write a Gaussian-splatting dataset |
| `--loop-closure` | Detect revisits, optimise the keyframe graph, and correct the model |
| `--loop-window N` | Keyframes merged on each side of a loop candidate; default `2` |
| `--online-loops` | Apply a verified loop during the scan; implies `--loop-closure` |
| `--loop-worker` | Run loop ICP on a worker thread; implies `--loop-closure` |
| `--incremental-rebuild` | Move cached submap surfaces instead of re-fusing; implies `--loop-closure` |
| `--relocalize` | Retry a rejected frame with spatial ICP against a local keyframe map |
| `--relocalize-keyframes N` | Recent keyframes in that local map; default `6`; implies `--relocalize` |
| `--relocalize-global` | Add a 4-DOF branch-and-bound fallback when that ICP fails; implies `--relocalize` |
| `--fusion-filter` | Track but do not fuse near-duplicate frames; forces one every 5 frames |
| `--tsdf-chunk N` | Finish the active TSDF into a cached chunk every N fused frames |
| `--loop-global` | Add a 4-DOF branch-and-bound fallback when loop ICP fails; implies `--loop-closure` |
| `--match-score S` | Minimum mean likelihood for a verified loop, `0..1`; zero disables |
| `--submap-graph` | Optimise submap nodes instead of keyframe nodes; implies `--loop-closure` |
| `--submap-frames N` | Finish a tracking submap after N fused frames; odometry stays continuous |
| `--submap-metres M` | Finish a tracking submap after M metres of camera motion |
| `--graph-worker` | Solve the online pose graph (keyframe or submap) off the capture thread; implies `--online-loops` |
| `--level` | Rotate the exported mesh and trajectory so the floor or ceiling is horizontal; not applied to `--dataset` |
| `--map FILE` | `localize`: prior map as ASCII or binary little-endian PLY |
| `--live` | `localize`: use the sensor instead of `--in` |
| `--no-submaps` | Keep one model and skip unplaceable frames instead of starting a submap |
| `--gpu` | GPU depth decode; requires `--features wgpu-decode` |
| `--mirror` | Keep the raw sensor mirror orientation instead of flipping X. A `--dataset` follows the same choice, all of it at once |
| `--no-filter` | Disable decoder bilateral and edge-aware filters |

### Loop closure tuning

The defaults are deliberately conservative. On a clean scan, the useful
experiments are:

```sh
# Correct as the scan runs, and verify off the capture thread.
scan replay --in out/closed-loop.k2df --loop-closure --online-loops --loop-worker

# A wider verification window for a sensor that turns in place.
scan replay --in out/closed-loop.k2df --loop-closure --loop-window 4

# Fast final correction: move the cached submaps, do not re-fuse.
scan replay --in out/closed-loop.k2df --loop-closure --incremental-rebuild

# Recover a lost track with a wide 4-DOF search before accepting the frame.
scan replay --in out/closed-loop.k2df --relocalize --relocalize-global

# Refuse loops that ICP fits but the target surface does not support.
scan replay --in out/closed-loop.k2df --loop-closure --match-score 0.55

# Submap-node graph, solved off the capture thread.
scan replay --in out/closed-loop.k2df --loop-closure --online-loops \
    --submap-graph --graph-worker

# Skip near-duplicate fusion and bound the active volume.
scan replay --in out/room.k2df --fusion-filter --tsdf-chunk 80
```

`--online-loops` and `--incremental-rebuild` trade accuracy inside a submap for
speed: the cached surface moves rigidly. The default full rebuild re-integrates
every kept frame at the corrected poses and is the one to compare against.

The global-search flags are fallbacks, not replacements: they only supply an
initial guess when ICP loses its basin, and the ordinary inlier, residual, and
reciprocal gates still decide whether the result is trusted. They cost a
multi-resolution grid and a bounded branch-and-bound search, so leave them off
for a clean scan.

`--fusion-filter` and `--tsdf-chunk` reduce fusion time without changing the
trajectory: filtered frames are still tracked and retained, and a chunk is a
memory boundary inside one submap, not a new pose boundary.

`--submap-graph` moves whole submaps rigidly. It is cheaper than the keyframe
graph and matches the cached surfaces, but it cannot deform inside a submap;
keep the keyframe graph for the accuracy reference. `--graph-worker` moves the
online keyframe **or** submap solve off the capture thread. It does not force a
submap graph.

`--submap-frames N` and `--submap-metres M` add a planned boundary that becomes
a real graph node. Odometry continues across it, so the edge is trusted; this is
different from a lost-track boundary, which is a guess. The defaults are zero.

### Voxel size and memory

Memory scales with the inverse cube of `--voxel`. A 1 cm room model reaches
several hundred megabytes; `--voxel 0.02` cuts that by roughly eight.

### Filters

`--no-filter` is faster and keeps more junk points. It is useful for profiling
or deliberately raw captures, not for normal scans.

### Output format

`--out` selects by extension:

| Path | Files written |
| --- | --- |
| `scan.ply` | Per-vertex colour in that file |
| `room.obj` | `room.obj`, `room.mtl`, `room.png` |
| `room.gltf` | `room.gltf`, `room.bin`, `room.png` |
| `room.glb` | One binary GLB with geometry and PNG inside |

A trajectory PLY is written alongside the mesh as `<out>.trajectory.ply`.

See [Colour](colour.md) for the painting and atlas details.

## GUI-only options

| Option | Meaning |
| --- | --- |
| `--synthetic` | Select the analytic room instead of the sensor |
| `--in FILE` | Open a `.k2df` recording as the source |
| `--auto-start` | Start the selected source when the window opens |
| `--preview-every N` | Extract a preview mesh every N frames; default `2` |
| `--playback-fps F` | Pace replay/synthetic playback; default `10`, zero means unlimited |

`--viewer` on `live` is a compatibility alias: it opens the same GUI on the live
source and starts immediately. Closing the window stops the scan and exports it.

## Loop closure

`--loop-closure` detects revisits and rebuilds the model from corrected poses.

- Pose proximity proposes the normal candidates.
- On colour scans, if none of those verifies, a compact appearance signature
  (8x8 grayscale plus an RGB histogram) can propose a frame outside the pose
  radius. The same spatial ICP still has to accept it.
- Loop keyframes are thinned to every 5 cm / 0.10 rad of motion, with a
  ten-frame fallback, so a submap reset cannot starve the loop index. Two
  candidates are verified per keyframe.
- An accepted loop must also align in reverse.
- Each edge is weighted by inlier coverage and residual.
- If the pose graph asks for an implausible correction, the weakest loop is
  dropped and the graph is retried.
- The loop index has bounded working memory. The most recent 60 keyframes keep
  full-resolution clouds; older signatures keep every second point. A candidate
  that points at an old keyframe rebuilds its cloud from retained depth for that
  verification, then drops the temporary cloud again.

A live colour scan moves its colour views onto corrected poses before painting.
The rebuild keeps every depth frame, about 850 KB each, so a 200-frame scan holds
around 170 MB of depth. Each colour view is separate: about 1.66 MiB, or about
1 GiB at 600 views.

Loop closure only helps a scan that returns somewhere it has already been.

## Submaps and relocalisation

Submaps are on by default. When the tracker cannot place a frame and the spatial
search does not recover it, that frame starts a new submap at the failed
alignment's candidate pose. Tracking and fusion continue locally.

On `out/room.k2df`:

| Mode | Fused | Skipped | Submaps | Memory |
| --- | --- | --- | --- | --- |
| Default | 150 | 0 | 19 | 220 MB |
| `--no-submaps` | 77 | 73 | 1 | 464 MB |

No revisit was verified on that recording, so its fragments stay at guessed
origins. Finalising each submap extracts its surface, about a second per
boundary.

`--relocalize` is an experimental spatial search for rejected frames. It aligns
the frame's voxel cloud to the last accepted frame's cloud and accepts only a
strict fit: at least 55% inliers and at most 25 mm RMSE. On `out/room.k2df` with
`--no-submaps` it recovered 0 of 73 rejected frames (they were first views of
surfaces the last frame never saw) and cost about 4.8x the tracking time. It is
off by default.

Detailed measurements are in
[GAUSSIAN_SPLATTING.md](../GAUSSIAN_SPLATTING.md), milestones M2.1 and M2.1b.

## GPU decoding

`--gpu` decodes depth on the GPU during `live` and `record`. `replay` reads depth
that was already decoded, so the flag does nothing there.

It needs a build with `--features wgpu-decode`. OpenCL (`--features gpu-decode`)
is the other backend and does not work with Rusticl on the development machine.
See [Performance](performance.md#gpu-decoding).

## Diagnostics

`SCAN_DEBUG_LOOPS=1` prints loop candidates and ICP outcomes.
`SCAN_DEBUG_F2M=1` prints frame-to-model render coverage.
