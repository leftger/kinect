# Development

This page covers the repository layout, tests, vendored driver, diagnostics, and
known gaps.

## Layout

```text
crates/geom         driver-independent geometry: voxel hashing, normals,
                    point-to-plane ICP, projective association, TSDF, surface
                    nets, colouring, texture atlases, PLY/OBJ/glTF/GLB output,
                    pose-graph optimisation
crates/scan         the scanner: capture, recording, odometry, fusion, CLI,
                    and the scanner window (source picker, 3D preview,
                    transport, export)
crates/probe        bring-up probe: confirms the sensor streams on this machine
vendor/kinect-one   vendored libfreenect2 port, with patches
platform/linux      udev rule
```

## Build features

| Feature | Purpose |
| --- | --- |
| none | CLI scanner |
| `gui` | Scanner window; alias `viewer` |
| `wgpu-decode` | Vulkan/Metal depth decoder |
| `gpu-decode` | OpenCL decoder; not usable with Rusticl on the development machine |

Build examples:

```sh
cargo build --release -p scan
cargo build --release -p scan --features gui
cargo build --release -p scan --features wgpu-decode
```

## Tests

```sh
cargo test -p geom -p scan --features gui
```

245 tests, 3 ignored. The geometry is tested against synthetic scenes and known
transforms rather than recorded data, so the suite runs without a sensor. The
loop-closure tests cover the keyframe window, the background worker, online
correction, and the incremental submap path. The newer geometry tests cover the
likelihood grid and branch-and-bound search, the PLY point reader, frozen-map
localisation, motion-filtered fusion, TSDF chunks, submap-node optimisation, and
the background graph worker.

The window tests are headless too: they exercise the session worker, a synthetic
scan and its export, the trainer child-process worker, the splat
loader/renderer, and the software renderer without opening a window.

The non-GUI suite is 220 tests.

### Ignored hardware tests

```sh
cargo test --release -p scan --features wgpu-decode -- --ignored --nocapture
```

`live_decoders_agree` feeds one real sensor packet to both depth decoders.
`diagnose_the_divergence` prints the stage-by-stage comparison.

### Manual splat load/render test

```sh
SCAN_SPLAT_PLY=/path/to/export.ply \
  cargo test --release -p scan --features gui gui:: -- --ignored --nocapture
```

## Vendored driver

`vendor/kinect-one` is the pure-Rust libfreenect2 port at upstream revision
`24c6dc0`, MIT licensed. It is vendored rather than used as a git dependency so
the patches are visible and reviewable.

`vendor/kinect-one/VENDORED.txt` lists the local changes. One is a genuine
upstream bug: `undistort_depth` indexes a frame with a sentinel that upstream
libfreenect2 bounds-checks and the port does not, so a border pixel panics on
real frames.

The macOS build adds `src/libusb_host.rs` and does not yet appear in
`UPSTREAM.patch`.

`UPSTREAM.patch` is the diff from before the macOS host. Regenerating it is
described in `VENDORED.txt`. Until then, this check does not cover
`libusb_host.rs`:

```sh
cp -r /tmp/k1/src /tmp/verify/src
(cd /tmp/verify && patch -p1 < UPSTREAM.patch && diff -rq src <crate>/src)
```

## Diagnostics

| Variable / command | Purpose |
| --- | --- |
| `SCAN_DEBUG_LOOPS=1` | Print loop candidates and ICP outcomes |
| `SCAN_DEBUG_F2M=1` | Print frame-to-model render coverage |
| `cargo run --release -p scan --example ocl_check --features gpu-decode` | OpenCL probe |
| `cargo run --release -p scan --example vk_check --features wgpu-decode` | wgpu probe |
| `cargo run --release -p probe` | Sensor stream probe |

## Known gaps

1. **Reduce fusion cost.** It is the largest single per-frame item at about
   100 ms, ahead of tracking at 35 ms and decode at about 50 ms. It scales with
   model size, so it is what makes large `--voxel 0.01` scans slow rather than
   merely memory-hungry.

2. **Exercise loop closure on a real loop.** The synthetic
   `a_revisit_merges_submaps_at_the_corrected_pose` test verifies a cross-submap
   revisit: front, side, and returned-front views form three submaps, the loop
   edge moves the third by 0.20 m, and the rebuild produces one model. A real
   capture that walks out and returns is still needed to show that it removes
   drift and makes a better room.

3. **File the `undistort_depth` bug upstream**, since it affects anyone using the
   port on real hardware.

4. **The GPU decoder frees the CPU and does not shorten the scan.** Either find a
   use for the spare cores, or accept the wall-clock result in
   [Performance](performance.md).

5. **Benchmark the new correction paths on real data.** The tests prove online
   correction, the background worker, incremental submap moves, global search,
   the likelihood gate, submap-node optimisation, the graph worker, motion
   filtering, TSDF chunks, and frozen-map localisation on synthetic scans. They
   do not yet show which wins on `out/closed-loop.k2df`, or how the 4-DOF search
   behaves on a rolling or tilted capture: it assumes the scanner is roughly
   upright and there is no gravity sensor to correct that assumption. Online and
   incremental moves are rigid per submap, so they cannot remove drift *inside*
   one; the default full rebuild stays the accuracy reference. The keyframe graph
   also still uses the dense numerical-Jacobian solve, fine at these node counts
   but the next thing to replace (sparse linear algebra plus analytic SE(3)
   Jacobians) before very long scans.

## Licence

No licence has been chosen for this code yet, which means it is all rights
reserved by default.

The vendored driver keeps its own MIT licence in
`vendor/kinect-one/LICENSE`.

The decode kernels translated to WGSL in
`crates/scan/shaders/depth_decode.wgsl` derive from libfreenect2 and carry its
Apache-2.0 / GPL-2.0 dual licence. See
`vendor/kinect-one/LIBFREENECT2_CONTRIB`.
