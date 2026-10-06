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
- **The GPU depth decoder is correct but buys little.** About 6% off the wall
  clock and a third off host CPU time. See `crates/scan/src/wgpu_depth.rs`.
- **Fusion is now the largest remaining per-frame cost**, about 100 ms at 1 cm
  voxels, and it scales with the size of the model. Tracking is about 35 ms.

## Requirements

- A **Kinect v2** (the Xbox One sensor, USB id `045e:02c4`). This is not the
  Kinect v1 and not the Azure Kinect DK; the Azure device is a different sensor
  with a different driver and an IMU.
- A **USB 3.0 port**. The sensor is SuperSpeed-only and needs its own power
  supply; the USB cable carries data, not power.
- Linux with libusb access to the device. The udev rule in
  `platform/linux/udev/90-kinect2.rules` grants that, and must be installed as
  root:

  ```
  sudo cp platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

  Without it every capture fails with an opaque permission error.

## Build

```
cargo build --release
```

The scanner binary is `target/release/scan`. To check that the sensor works
before using the scanner, `cargo run --release -p probe`.

## Use

```
scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--drain-color] [--gpu] [--loop-closure]
scan record  --out capture.k2df [--frames N] [--no-filter] [--drain-color] [--gpu]
scan replay  --in capture.k2df [--out mesh.ply] [--voxel M] [--frames N] [--gpu] [--loop-closure]
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
- `--loop-closure` detects revisits and rebuilds the model from corrected poses.
  It buffers the frames to do that, about 850 KB each, so a 200-frame scan costs
  around 170 MB. It only helps a scan that returns somewhere it has already been.
- `--drain-color` reads and discards the colour stream. Off by default, because
  the scanner never uses colour and waiting for it costs more than half the frame
  time. Turn it on only if you are extending the scanner to use colour.
- `--viewer` opens a live window showing the scan as it builds: the reconstruction
  so far, the frame count, the tracked position and the model size. Needs a build
  with `--features viewer` and a display. Closing the window stops the scan, and
  the mesh is written as usual.
- `--gpu` decodes on the GPU. Needs a build with `--features wgpu-decode`
  (Vulkan) or `--features gpu-decode` (OpenCL); see below.
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
depth rate, so waiting for a packet takes a frame from roughly 270 ms to 530 ms.
Capturing colour every Nth frame would pay that proportionally less often and is
the obvious next step.

The alignment this relies on was checked before any of it was written:
`crates/scan/examples/color_check.rs` writes the registered colour and the depth
discontinuities overlaid, and the depth edges land on colour edges with a
one-pixel offset. That is the check the GPU port should have had first.

## The vendored driver

`vendor/kinect-one` is the pure-Rust libfreenect2 port at upstream revision
`24c6dc0`, MIT licensed, with four local changes. It is vendored rather than used
as a git dependency so the patches are visible and reviewable.

`vendor/kinect-one/UPSTREAM.patch` holds the complete diff. `VENDORED.txt`
explains each change. One of them is a genuine upstream bug: `undistort_depth`
indexes a frame with a sentinel that upstream libfreenect2 bounds-checks and the
port does not, so a border pixel panics on real frames.

To verify the vendored copy is upstream plus only those changes:

```
cp -r /tmp/k1/src /tmp/verify/src
(cd /tmp/verify && patch -p1 < UPSTREAM.patch && diff -rq src <crate>/src)
```

## GPU decoding

Two backends exist, and on the machine this was developed on only one of them
works.

- **OpenCL** (`--features gpu-decode`) is unusable. Mesa's Rusticl enumerates a
  device, allocates buffers, and transfers data correctly, then silently never
  executes a kernel. `crates/scan/examples/ocl_check.rs` demonstrates this with a
  trivial kernel.
- **Vulkan** (`--features wgpu-decode`) works. The four libfreenect2 decode
  kernels were translated to WGSL and are driven with wgpu. This is the
  interesting half of the work and it did not pay off: see below.

The translation is verified against the CPU decoder on real sensor data, feeding
one packet to both:

| stage | agreement |
| --- | --- |
| stage 1, via the IR frame | 212842 of 216240 pixels within 1 count of 65535 |
| stages 1 and 2, filters off | mean difference 1.73 mm, 0.104% of depth |
| stages 1 and 2, filters on | 96.7% agreement on valid pixels, mean 2.02 mm |

That residual is below the sensor's own noise, so it is not visible through it.

What it buys is less than hoped:

| 100 frames, `live`, filters off | wall | per frame | user CPU |
| --- | --- | --- | --- |
| CPU decode | 25.6 s | 256 ms | 25.4 s |
| Vulkan decode | 23.9 s | 239 ms | 16.7 s |

So roughly 6% off the wall clock, which is close to the run-to-run noise of these
measurements, and a third off the host CPU time, which is not.

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
5. **The GPU decoder does not pay off.** Either find it a use, for example
   running tracking and fusion on the freed cores, or accept it as a documented
   negative result.

## Licence

No licence has been chosen for this code yet, which means it is all rights
reserved by default. The vendored driver keeps its own MIT licence in
`vendor/kinect-one/LICENSE`. The decode kernels translated to WGSL in
`crates/scan/shaders/depth_decode.wgsl` derive from libfreenect2 and carry its
Apache-2.0 / GPL-2.0 dual licence, see `vendor/kinect-one/LIBFREENECT2_CONTRIB`.
