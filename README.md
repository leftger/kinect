# kinect

A handheld 3D scanner for the Kinect v2, in Rust. It captures depth frames,
tracks the sensor's motion from the depth data alone, fuses the frames into a
TSDF volume, and extracts a triangle mesh.

There is no IMU and no GPS. Motion is recovered from the depth images themselves,
and drift is addressed with loop closure rather than with extra sensors. An IMU
would help a little; a GPS is useless indoors, and the Kinect v2 has neither.

## Status

This is a working scanner with one large, measured weakness and several honest
gaps. Read this section before the code.

**Works**

- Live capture, recording, and replay of depth frames at 512x424
- Frame-to-frame odometry recovered from depth only, 81 ms per frame
- TSDF fusion and triangle-mesh extraction, written as binary PLY
- Trajectory export, which is the quickest way to see how badly a scan drifted
- Loop closure and pose-graph optimisation, wired and tested
- Optional colour, registered onto the mesh as per-vertex PLY colours

**Found by profiling, and fixed**

- **The capture rate was capped by the colour stream.** The capture loop used to
  read and discard a colour packet for every depth frame, on the reasoning that a
  stream has to keep being drained or its transfers stop being resubmitted. The
  colour stream delivers at about a third of the depth rate, so every frame
  waited on it. Reading colour is now off by default and the `live` pipeline went
  from 633 ms to 271 ms per frame over 100 frames.

  It went unnoticed for a long time because it costs no CPU: it shows up purely
  as waiting, and every earlier measurement had assumed the decode was the
  expensive part. Building a GPU decoder to attack the decode is what disproved
  that, which is the one useful thing that came out of that work.

**Does not work, or has not been shown to**

- **Drift is real.** On a 150-frame handheld capture the reconstructed trajectory
  covers 4.8 m of path and ends 2.15 m from where it started. Loop closure exists
  to correct exactly this, but it has never been exercised on a real loop: the
  test capture contains none.
- **The GPU depth decoder is correct but does not shorten a scan.** It cuts
  host CPU time sharply and leaves the wall clock about where it was, because
  capture waits on USB. See [GPU decoding](#gpu-decoding).
- **Fusion is now the largest remaining per-frame cost**, about 100 ms at 1 cm
  voxels, and it scales with the size of the model. Tracking is about 35 ms.

## Requirements

- A **Kinect v2** (the Xbox One sensor, USB id `045e:02c4`). This is not the
  Kinect v1 and not the Azure Kinect DK; the Azure device is a different sensor
  with a different driver and an IMU.
- A **USB 3.0 port**. The sensor is SuperSpeed-only and needs its own power
  supply; the USB cable carries data, not power.
- Linux or macOS. On Linux, libusb access comes from the udev rule in
  `platform/linux/udev/90-kinect2.rules`, installed as root:

  ```
  sudo cp platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

  Without it every capture fails with an opaque permission error.

  On macOS, install libusb (`brew install libusb`). There is no udev rule.
  The operating system is chosen when the binary is compiled. A Mac build
  drives control, colour, and depth through libusb, because `nusb` opens the
  device exclusively on macOS and has no isochronous transfers there. A Linux
  build keeps the `nusb` path. Windows is not supported. `--gpu` selects the
  wgpu depth decoder, which uses Metal on macOS and Vulkan on Linux.

## Build

Build on the machine that will run the scanner. The USB backend is selected
at compile time, so a Mac binary does not become a Linux binary by being
copied across.

```
cargo build --release -p scan
```

GPU depth decode needs the wgpu feature. On macOS that is Metal; on Linux it
is Vulkan.

```
cargo build --release -p scan --features wgpu-decode
```

The scanner binary is `target/release/scan`. To check that the sensor works
before using the scanner, `cargo run --release -p probe`.

`--frames` defaults to 60. A 300-frame GPU scan on this Mac is:

```
target/release/scan live --gpu --frames 300 --out scan.ply
```

Add `--color` to paint the mesh.

## Use

```
scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--color] [--gpu] [--loop-closure] [--viewer]
scan record  --out capture.k2df [--frames N] [--no-filter] [--drain-color] [--gpu]
scan replay  --in capture.k2df [--out mesh.ply] [--voxel M] [--frames N] [--loop-closure]
```

- `live` captures and reconstructs as it goes.
- `record` stores depth frames to a file for offline work. A recording is
  deterministic, so replaying a prefix of it is a repeatable test case.
- `replay` reconstructs from a recording and needs no sensor.

Useful options:

- `--voxel M` sets the TSDF voxel size in metres, default 0.01. Memory scales
  with the inverse cube of this: a 1 cm model of a room reaches several hundred
  megabytes, and `--voxel 0.02` cuts that by roughly eight.
- `--no-filter` disables the decoder's bilateral and edge-aware filters. It is
  faster and keeps more junk points.
- `--frames N` is how many frames to capture or replay. The default is 60.
- `--loop-closure` detects revisits and rebuilds the model from corrected poses.
  It buffers the frames to do that, about 850 KB each, so a 200-frame scan costs
  around 170 MB. It only helps a scan that returns somewhere it has already been.
- `--color` captures the colour stream during `live` and paints the finished
  mesh. Off by default, because that stream is slower than depth.
- `--drain-color` applies to `record`. It reads and discards colour packets so
  the pipe stays drained, without writing them into the recording.
- `--viewer` opens a live window showing the scan as it builds: the reconstruction
  so far, the frame count, the tracked position and the model size. Needs a build
  with `--features viewer` and a display. The window backend in that feature is
  Wayland and X11. Closing the window stops the scan, and the mesh is written
  as usual.
- `--gpu` decodes depth on the GPU during `live` and `record`. `replay` reads
  depth that was already decoded, so the flag does nothing there. Needs a build
  with `--features wgpu-decode`. OpenCL (`--features gpu-decode`) is the other
  backend and does not work with Rusticl; see below.
- `--mirror` keeps the raw sensor mirror orientation instead of flipping X (un-mirrored by default).

A trajectory PLY is written alongside the mesh as `<out>.trajectory.ply`.

## How it works

```
depth frame from USB
  -> decode            raw packet to metres          (vendored libfreenect2 port)
  -> odometry          pose relative to the last frame
  -> TSDF fusion       integrate into a voxel-hashed volume
  -> mesh extraction   naive surface nets, binary PLY
```

**Odometry** uses projective data association: each point in the current frame is
projected into the previous depth image to find its correspondence, with an
occlusion test on depth disagreement, rather than searching a spatial index. That
change alone took tracking from 928 ms to 81 ms per frame with no loss of
accuracy. Point-to-plane ICP with Huber weighting and a coarse-to-fine pyramid
does the solving. Frames whose solution fails acceptance thresholds are dropped
rather than fused, and the reference frame still advances so that one bad frame
cannot lock out the rest of the scan.

**Fusion** uses a voxel-hashed TSDF, one hash entry per 8x8x8 block. Each block
is enumerated once per frame and the projective update runs over its voxels.

**Mesh extraction** is naive surface nets: one vertex per sign change, then
connectivity is recovered by, for each grid edge with a sign change, joining the
four surrounding cubes. It is far less code than marching cubes and produces
cleaner quads at the cost of some detail.

## Colour

`--color` captures the colour stream and paints it onto the finished mesh, written
as per-vertex `uchar red/green/blue` properties that MeshLab, CloudCompare and
Blender read.

The colour is registered into the *depth* camera's grid as each frame arrives
(`Registration::undistort_depth_and_color`), so a view is a 512x424 RGB image plus
the undistorted depth from the same frame. That registration is what makes the
rest simple: projecting a vertex needs only the depth intrinsics, and the view's
own depth is directly usable as an occlusion test, which is the same test the
projective odometry already does.

A vertex is only painted by a view whose measured depth agrees with how far away
the vertex actually is. Without that, a vertex on a far wall seen through a
doorway from some other pose would be painted with whatever is in front of it.
Every view that passes contributes, so the finished model is smoother than any
single frame.

PLY has no texture coordinates in its core format, so this is per-vertex colour
rather than a texture atlas. Unlike fused colour it costs no extra volume and
lets every frame that saw a surface contribute, not only the frames that arrived
while that surface was being integrated. A real texture atlas would mean
outputting OBJ or glTF instead.

**It costs capture rate.** The colour stream delivers at about a third of the
depth rate. On the Linux machine this was first measured on, waiting for a
packet took a frame from roughly 270 ms to 530 ms. On the M4 Pro, a 20-frame
colour scan and a 30-frame depth-only scan both took about 3 s of wall time, so
colour still lowers throughput. Capturing colour every Nth frame would pay that
less often.

The alignment this relies on was checked before any of it was written:
`crates/scan/examples/color_check.rs` writes the registered colour and the depth
discontinuities overlaid, and the depth edges land on colour edges with a
one-pixel offset. That is the check the GPU port should have had first.

## The vendored driver

`vendor/kinect-one` is the pure-Rust libfreenect2 port at upstream revision
`24c6dc0`, MIT licensed. It is vendored rather than used as a git dependency so
the patches are visible and reviewable.

`vendor/kinect-one/VENDORED.txt` lists the local changes. One of them is a
genuine upstream bug: `undistort_depth` indexes a frame with a sentinel that
upstream libfreenect2 bounds-checks and the port does not, so a border pixel
panics on real frames. The macOS build adds `src/libusb_host.rs` and does not
yet appear in `UPSTREAM.patch`.

`UPSTREAM.patch` is the diff from before the macOS host. Regenerating it is
described in `VENDORED.txt`. Until then, this check does not cover
`libusb_host.rs`:

```
cp -r /tmp/k1/src /tmp/verify/src
(cd /tmp/verify && patch -p1 < UPSTREAM.patch && diff -rq src <crate>/src)
```

## GPU decoding

Two backends exist. wgpu is the one that runs.

- **OpenCL** (`--features gpu-decode`) is unusable on the Linux machine this was
  developed on. Mesa's Rusticl enumerates a device, allocates buffers, and
  transfers data correctly, then silently never executes a kernel.
  `crates/scan/examples/ocl_check.rs` demonstrates this with a trivial kernel.
- **wgpu** (`--features wgpu-decode`) works. The four libfreenect2 decode
  kernels were translated to WGSL. On Linux, wgpu uses Vulkan. On macOS, the
  same binary feature uses Metal. The log line names the adapter and the
  backend, for example `Apple M4 Pro (Metal)`.

The translation is verified against the CPU decoder on real sensor data, feeding
one packet to both:

| stage | agreement |
| --- | --- |
| stage 1, via the IR frame | 212842 of 216240 pixels within 1 count of 65535 |
| stages 1 and 2, filters off | mean difference 1.73 mm, 0.104% of depth |
| stages 1 and 2, filters on | 96.7% agreement on valid pixels, mean 2.02 mm |

That residual is below the sensor's own noise, so it is not visible through it.

What it buys is less wall-clock time than hoped.

On Linux, 100 frames of `live` with the filters off:

| | wall | per frame | user CPU |
| --- | --- | --- | --- |
| CPU decode | 25.6 s | 256 ms | 25.4 s |
| Vulkan decode | 23.9 s | 239 ms | 16.7 s |

About 6% off the wall clock, close to the run-to-run noise, and a third off the
host CPU time.

On an Apple M4 Pro, filters on, the sensor sitting still:

| | wall | user CPU |
| --- | --- | --- |
| record, 40 frames, CPU | 3.52 s | 2.50 s |
| record, 40 frames, Metal | 3.31 s | 0.25 s |
| live, 30 frames, CPU | 2.91 s | 2.02 s |
| live, 30 frames, Metal | 2.90 s | 0.22 s |
| live --color, 20 frames, CPU | 2.97 s | 1.50 s |
| live --color, 20 frames, Metal | 2.99 s | 0.35 s |

Recording got about 6% faster. The live scans did not. Host CPU time dropped by
about an order of magnitude. Tracking and fusion on that small scene were a few
milliseconds per frame, so the frame time is the USB wait.

The premise it was built on, that the decode caps capture rate, was wrong: the
measurement that produced it was taken with the filters on, where the decode cost
about 148 ms per frame. Measured now, decode is perhaps 50 ms of a 256 ms frame,
and tracking plus fusion is 140 ms of it. Killing the decode entirely would buy
about 20%, which is why the port is kept but is not the interesting problem.

There are two driver diagnostics, both runnable without a sensor:

```
cargo run --release -p scan --example ocl_check   --features gpu-decode
cargo run --release -p scan --example vk_check    --features wgpu-decode
```

## Layout

```
crates/geom         driver-independent geometry: voxel hashing, normals,
                    point-to-plane ICP, projective association, TSDF, surface
                    nets, PLY output, pose-graph optimisation
crates/scan         the scanner: capture, recording, odometry, fusion, CLI
crates/probe        bring-up probe: confirms the sensor streams on this machine
vendor/kinect-one   vendored libfreenect2 port, with patches
platform/linux      udev rule
```

## Tests

```
cargo test -p geom -p scan
```

54 tests. The geometry is tested against synthetic scenes and known transforms
rather than against recorded data, so the suite runs without a sensor.

Two tests need hardware and are ignored by default:

```
cargo test --release -p scan --features wgpu-decode -- --ignored --nocapture
```

`live_decoders_agree` feeds one real sensor packet to both depth decoders, and
`diagnose_the_divergence` prints the stage-by-stage comparison.

## Known gaps

1. **Reduce fusion cost**, now the largest single per-frame item at about
   100 ms, ahead of tracking at 35 ms and decode at about 50 ms. It scales with
   the size of the model, so it is also what makes large `--voxel 0.01` scans
   slow rather than merely memory-hungry.
2. **Exercise loop closure on a real loop.** It is implemented and tested
   against synthetic scans, but the property that has been verified is only that
   it does not damage a correct trajectory. A capture that walks out and returns
   to its start would show whether it removes drift.
4. **File the `undistort_depth` bug upstream**, since it affects anyone using the
   port on real hardware.
5. **The GPU decoder frees the CPU and does not shorten the scan.** Either find
   a use for the spare cores, or accept the wall-clock result above.

## Licence

No licence has been chosen for this code yet, which means it is all rights
reserved by default. The vendored driver keeps its own MIT licence in
`vendor/kinect-one/LICENSE`. The decode kernels translated to WGSL in
`crates/scan/shaders/depth_decode.wgsl` derive from libfreenect2 and carry its
Apache-2.0 / GPL-2.0 dual licence, see `vendor/kinect-one/LIBFREENECT2_CONTRIB`.
