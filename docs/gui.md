# Scanner window

`scan gui` is the full scanner application: source selection, live preview,
transport, parameters, export, telemetry, and an optional splat trainer panel.

It needs a build with `--features gui` and a display. It does not need a sensor:
the synthetic room runs the whole real-time path on a machine with nothing
plugged in.

## Launch

```sh
# Build once
cargo build --release -p scan --features gui

# Synthetic room, starts immediately
./target/release/scan gui --synthetic --auto-start

# Replay a recording
./target/release/scan gui --in out/capture.k2df --auto-start

# Live sensor, choose the source in the window
./target/release/scan gui --color
```

Without `--auto-start`, the window opens on the source picker and waits for
**Start**.

One-shot equivalent:

```sh
cargo run --release -p scan --features gui -- gui --synthetic --auto-start
```

## Sources

| Source | What it does |
| --- | --- |
| Live sensor | Captures and reconstructs from the Kinect |
| Record | Captures while writing a `.k2df` recording |
| Recording | Replays a selected `.k2df` file |
| Synthetic room | Analytic colour room with no hardware |

The source and parameters are locked once a scan starts. **Reset** returns to
the picker.

## Panels

### Reconstruction and output

Controls frames, voxel size, colour and its mode, colour-depth tolerance,
filters, loop closure, frame-to-model, submaps, GPU decode, mirror, preview
interval, playback rate, the output path, and the optional dataset directory.

The output extension selects the mesh format exactly as it does on the command
line. See [Command line](cli.md#output-format).

### 3D viewport

- Drag to orbit.
- Right-drag or middle-drag to pan.
- Scroll to zoom.
- Double-click or **Fit view** to frame the model.

The surface is extracted from the TSDF every `--preview-every` frames, so it
refines as the scan proceeds. The latest depth frame is drawn as coloured
points, the camera path as a cyan line, and the current sensor as a yellow
frustum.

Mesh colour in the preview comes from only the most recent colour views, which
keeps preview cost bounded as the scan grows. The exported mesh still uses every
view.

### Telemetry

Shows:

- frames, fused, rejected, and submap counts;
- position and TSDF size;
- tracking, fusion, and submap-finalisation time;
- preview rate and the worker log;
- loops, loop ICP attempts, dropped loops, loop points, and rehydrated clouds;
- exported mesh and dataset summary after Export.

### Splats

After an export, the Splats panel can run a configured Brush command as a child
process.

Editable fields:

- executable (`brush-cli` by default);
- dataset directory;
- output directory;
- iteration count.

The panel has **Start**, **Cancel**, and a streamed log. An exported dataset with
unmerged submaps or too many rejected frames is marked *not trainable*; Start
then requires **Train anyway**.

The trained `.ply` can be loaded into the viewport with:

- **Mesh / Splats** toggle;
- **Fit splats**;
- **First-person** view.

The splat preview projects the anisotropic covariance and blends DC colour.
Higher SH bands are not evaluated. Splat-size and minimum-opacity sliders make
under-trained or floating-splat models inspectable.

Clouds over 250k splats render at half raster resolution and are scaled by egui.
First-person splats use the colour frame's x direction.

## Transport

| Control | Effect |
| --- | --- |
| **Start** | Runs the selected source |
| **Pause** | Stops consuming frames without losing the scanner |
| **Resume** | Continues from Pause |
| **Stop** | Ends capture but keeps the scanner for inspection/export |
| **Export** | Writes the model; stops the scan first if needed |
| **Reset** | Throws the scan away |

Closing the window exports an unexported scan rather than losing it, so **Reset**
is the explicit discard.

Record streams the same frames to the `.k2df` path while reconstructing. Stop,
Reset, or a sensor dropout closes the recording rather than losing it.

## Dataset export

The optional dataset directory writes the Gaussian-splatting hand-off:
`transforms.json`, posed `images/`, `masks/`, and a coloured `init.ply`.

It needs colour views, so either:

- tick **Colour** on a Live or Record source; or
- replay a recording made with `record --color`.

Start preflights the path. Export shows the view and initial-point counts in the
telemetry panel. The Splats panel can then run a configured Brush child process
and load its `.ply` into the preview.

See [Datasets and splats](datasets.md) for the layout, quality gate, Brush smoke
test, and real-capture results.

## Implementation note

The preview is rasterised on the CPU into an ordinary egui texture. That keeps
the window independent of the GPU decoder and testable without a display; at a
few frames per second the cost is not the bottleneck.
