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
filters, fusion filtering and TSDF chunking, loop closure and its
online/worker/incremental variants, the global-search fallbacks, the likelihood
gate, submap-graph and graph-worker modes, submap frame/metre budgets,
frame-to-model, submaps, GPU decode, mirror, level, preview interval, playback rate,
the output path, and the optional dataset directory.

The dependent modes imply loop closure, as they do on the command line: ticking
**Online loops**, **Loop worker**, **Incremental**, **Global loops**,
**Submap graph**, or **Graph worker** turns correction on rather than silently
doing nothing. **Graph worker** enables online loops but does not force the
submap graph; it solves whichever graph is selected. **Global relocalize**
implies relocalisation. A non-zero **Match score** enables the loop likelihood
gate.

The output extension selects the mesh format exactly as it does on the command
line. See [Command line](cli.md#output-format).

**Level** is the `--level` flag: on export, the mesh and trajectory are rotated
so the floor or ceiling is horizontal. The scan's frame is the first camera
frame, so a tilted start gives a tilted model. It applies to the exported files
only; the live preview stays as scanned, and a dataset is never levelled. If no
large horizontal surface is found, the model is left unchanged and the status
log says so.

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

- frames, fused, rejected, motion-filtered, and submap counts;
- position, TSDF size, and cached chunk count;
- tracking, fusion, and submap-finalisation time;
- preview rate and the worker log;
- loops, loop ICP attempts, dropped loops, loop points, rehydrated clouds, online
  corrections, background graph fixes, and pending loop/graph jobs;
- exported mesh and dataset summary after Export.

### Splats

After an export, the Splats panel runs [Brush](https://github.com/ArthurBrussee/brush)
as a child process and shows the model improving in the viewport while it trains.
See [Datasets and splats](datasets.md) for installing Brush.

The top of the panel says whether Brush was found on `PATH` (as `brush`,
`brush_app` or `brush-cli`) and which version, with a **Re-detect** button and a
download link when it is missing. Leave **Trainer** blank to use the one found,
or type a path.

Editable fields:

- trainer executable (blank = the Brush found on `PATH`);
- dataset directory (filled from the last export);
- output directory;
- **Steps**, the training length;
- **Preview every**, how often Brush writes a complete model.

Three checkboxes:

- **Live preview** loads each preview model into the viewport as Brush writes it.
  The first one frames the camera and switches to Splats; later ones only swap
  the model, so your view is not reset. Models are read on a background thread,
  so a large one does not freeze the window.
- **Brush window** also opens Brush's own live viewer, a separate window.
- **Train after export** starts training the moment an export writes a dataset.
  It decides once, when the export lands, so ticking it later does not retrain
  an export already on screen. It needs a dataset directory in Output.

While training, a progress bar shows steps, the live splat count, elapsed time
and a rough time left (linear in steps, so it runs optimistic: refinement makes
later steps slower), and a line says which step's preview the viewport shows.
Progress comes from the `export_<step>.ply` files and from Brush's
`Refine iter N, M splats.` log lines. If a Brush version words those lines
differently the bar still follows the exports, and the log is still streamed.

The panel has **Train**, **Cancel**, and a streamed log. An exported dataset with
unmerged submaps or too many rejected frames is marked *not trainable*; Train
then requires **Train anyway**.

The window runs Brush from the output directory, because Brush writes a kernel
cache into its working directory and panics when that is not writable. It asks
the installed Brush which command-line flags it speaks (`--total-steps` in the
current release, `--total-train-iters` in older ones) and adapts. An older Brush
writes only the final model, so there are no intermediate previews.

A trained `.ply` can also be loaded into the viewport by hand, with:

- **Mesh / Splats** toggle;
- **Fit splats**;
- **First-person** view.

The splat preview projects the anisotropic covariance and blends DC colour.
Higher SH bands are not evaluated. Splat-size and minimum-opacity sliders make
under-trained or floating-splat models inspectable.

The preview is drawn on the CPU. A 274,000-splat room model takes about half a
second a frame at 640×480 in a release build (a debug build is ten times slower),
so orbiting a large model is sluggish. Loading and swapping previews is cheap.

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
