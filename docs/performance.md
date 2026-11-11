# Performance

The scanner is limited by USB capture, fusion, and tracking in that order on
most machines. This page records the measurements behind the current defaults.

## Per-frame costs

At 1 cm voxels on the development machine:

| Stage | Approximate cost |
| --- | --- |
| Decode | ~50 ms |
| Tracking | ~35 ms |
| Fusion | ~100 ms |
| Submap finalisation | ~1 s at each boundary |

Fusion is the largest single item and grows with the size of the model. That is
why large `--voxel 0.01` scans become slow, not merely memory-hungry.

## Drift

Drift is real. On a 150-frame handheld capture, the reconstructed trajectory
covers 4.8 m of path and ends 2.15 m from where it started.

Loop closure exists to correct exactly this. A later real closed-loop colour
capture fused 120/120 frames in one submap, found one verified revisit, and
applied a 10.9 cm maximum correction. See
[Datasets and splats](datasets.md#real-closed-loop-capture).

## The colour-stream discovery

The capture rate used to be capped by the colour stream. The capture loop read
and discarded a colour packet for every depth frame, on the reasoning that a
stream has to keep being drained or its transfers stop being resubmitted.

The colour stream delivers at about a third of the depth rate, so every frame
waited on it. Reading colour is now off by default, and the `live` pipeline went
from 633 ms to 271 ms per frame over 100 frames.

It went unnoticed because it costs no CPU: it shows up purely as waiting, and
every earlier measurement had assumed decode was the expensive part.

## GPU decoding

Two backends exist. wgpu is the one that runs.

| Backend | Feature | Status |
| --- | --- | --- |
| OpenCL | `gpu-decode` | Unusable on the Linux development machine: Rusticl enumerates, allocates, and transfers correctly, then silently never executes a kernel |
| wgpu | `wgpu-decode` | Works |

The four libfreenect2 decode kernels were translated to WGSL. On Linux, wgpu
uses Vulkan. On macOS, the same feature uses Metal. The log line names the
adapter and backend, for example `Apple M4 Pro (Metal)`.

### Decoder agreement

The translation was verified against the CPU decoder on real sensor data,
feeding one packet to both:

| Stage | Agreement |
| --- | --- |
| Stage 1, via the IR frame | 212842 of 216240 pixels within 1 count of 65535 |
| Stages 1 and 2, filters off | Mean difference 1.73 mm, 0.104% of depth |
| Stages 1 and 2, filters on | 96.7% agreement on valid pixels, mean 2.02 mm |

The residual is below the sensor's own noise.

### Wall-clock results

On Linux, 100 frames of `live` with filters off:

| Decode | Wall | Per frame | User CPU |
| --- | --- | --- | --- |
| CPU | 25.6 s | 256 ms | 25.4 s |
| Vulkan | 23.9 s | 239 ms | 16.7 s |

About 6% off the wall clock, close to run-to-run noise, and a third off host CPU
time.

On an Apple M4 Pro, filters on, sensor stationary:

| Run | Wall | User CPU |
| --- | --- | --- |
| record, 40 frames, CPU | 3.52 s | 2.50 s |
| record, 40 frames, Metal | 3.31 s | 0.25 s |
| live, 30 frames, CPU | 2.91 s | 2.02 s |
| live, 30 frames, Metal | 2.90 s | 0.22 s |
| live --color, 20 frames, CPU | 2.97 s | 1.50 s |
| live --color, 20 frames, Metal | 2.99 s | 0.35 s |

Recording got about 6% faster. Live scans did not. Host CPU time dropped by
about an order of magnitude. Tracking and fusion on that scene were a few
milliseconds per frame, so the frame time was USB wait.

### What was learned

The premise that decode capped capture rate was wrong. The measurement that
produced it was taken with filters on, where decode cost about 148 ms per frame.

Measured now, decode is perhaps 50 ms of a 256 ms frame, and tracking plus fusion
is 140 ms. Killing decode entirely would buy about 20%, which is why the port is
kept but is not the interesting problem.

## Diagnostics

Both run without a sensor:

```sh
cargo run --release -p scan --example ocl_check --features gpu-decode
cargo run --release -p scan --example vk_check  --features wgpu-decode
```

## Related

- [Command line](cli.md)
- [How it works](how-it-works.md)
- [Development](development.md)
