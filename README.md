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

**Does not work, or has not been shown to**

- **Capture rate is limited by something not yet identified.** A frame costs
  roughly 760 ms end to end in the `live` pipeline. The depth decode is a small
  part of that, which was established by writing a GPU decoder and measuring no
  improvement. Profiling where the rest of the time goes is the highest-value
  next step in this project.
- **Drift is real.** On a 150-frame handheld capture the reconstructed trajectory
  covers 4.8 m of path and ends 2.15 m from where it started. Loop closure exists
  to correct exactly this, but it has never been exercised on a real loop: the
  test capture contains none.
- **The GPU depth decoder is correct but buys no speed.** It reduces host CPU
  time by 6x to 13x and does not move the wall clock. See
  `crates/scan/src/wgpu_depth.rs` for the measurements.
- Fusion is expensive and scales with the size of the model, about 168 ms per
  frame at 1 cm voxels.

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
scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--no-color] [--gpu] [--loop-closure]
scan record  --out capture.k2df [--frames N] [--no-filter] [--no-color] [--gpu]
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
- `--gpu` decodes on the GPU. Needs a build with `--features wgpu-decode`
  (Vulkan) or `--features gpu-decode` (OpenCL); see below.

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

What it did **not** do is make anything faster:

| 60 frames | wall | user CPU |
| --- | --- | --- |
| `record`, CPU decode | 18.1 s | 8.4 s |
| `record`, Vulkan decode | 15.7 s | 1.4 s |
| `live`, CPU decode | 45.4 s | 24.3 s |
| `live`, Vulkan decode | 46.3 s | 18.2 s |

The decode gets six to thirteen times cheaper on the host and the wall clock does
not move. The premise it was built on, that the decode caps capture rate, came
from a measurement taken with the filters on, where decode cost about 148 ms per
frame. With them off it is a small part of a 760 ms frame. The port is kept
because it is correct and documented, and because a cheap host is worth
something if the freed cores are put to work on tracking and fusion.

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

1. **Find the actual bottleneck.** Roughly 760 ms per frame goes somewhere that
   is not the depth decode. Nothing else should be optimised before this is
   measured.
2. **Exercise loop closure on a real loop.** It is implemented and tested
   against synthetic scans, but the property that has been verified is only that
   it does not damage a correct trajectory. A capture that walks out and returns
   to its start would show whether it removes drift.
3. **Reduce fusion cost**, which scales with model size and dominates the
   remaining CPU time at 1 cm voxels.
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
