# Getting started

This guide takes you from a clean machine to a first scan.

## 1. Check the hardware

You need:

- A **Kinect v2** (the Xbox One sensor, USB id `045e:02c4`). This is not the
  Kinect v1 and not the Azure Kinect DK; the Azure device is a different sensor
  with a different driver and an IMU.
- A **USB 3.0 port**. The sensor is SuperSpeed-only and needs its own power
  supply. The USB cable carries data, not power.
- Linux or macOS. Windows is not supported.

### Linux USB permissions

Install the udev rule as root:

```sh
sudo cp platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Without this step, every capture fails with an opaque permission error.

### macOS

Install libusb:

```sh
brew install libusb
```

There is no udev rule on macOS.

The operating system is chosen when the binary is compiled. A Mac build drives
control, colour, and depth through libusb, because `nusb` opens the device
exclusively on macOS and has no isochronous transfers there. A Linux build keeps
the `nusb` path.

## 2. Build

Build on the machine that will run the scanner. The USB backend is selected at
compile time, so a Mac binary does not become a Linux binary by being copied
across.

The workspace uses Rust 2024 edition and resolver 3, so it needs rustc 1.85 or
newer. The manifests keep `rust-version = "1.85"` as the floor.

### Basic scanner

```sh
cargo build --release -p scan
```

The binary is `target/release/scan`.

### GPU depth decode

On Linux this uses Vulkan; on macOS it uses Metal.

```sh
cargo build --release -p scan --features wgpu-decode
```

### Scanner window

The GUI uses the glow backend (Wayland and X11 on Linux), not wgpu, so it can be
combined with `wgpu-decode` without pulling in a second GPU stack.

```sh
cargo build --release -p scan --features gui
```

The old feature name `viewer` is an alias for `gui`, so existing build commands
keep working.

## 3. Check the sensor

Before debugging the scanner, confirm that the sensor streams:

```sh
cargo run --release -p probe
```

You should see depth and colour stream names, intrinsics, frame rate, and a
valid-depth percentage.

## 4. First live scan

```sh
target/release/scan live --frames 60 --out scan.ply
```

`--frames` defaults to 60. To paint colour:

```sh
target/release/scan live --color --frames 300 --out scan.ply
```

`--out scan.ply` keeps per-vertex colour. `.obj`, `.gltf`, and `.glb` write a
texture atlas instead. A trajectory PLY is written beside the mesh as
`<out>.trajectory.ply`.

## 5. First GUI scan

```sh
target/release/scan gui --color
```

Pick the live sensor in the Source panel and press **Start**. To skip the picker:

```sh
target/release/scan gui --color --auto-start
```

The window needs a display, but it does not need a sensor. The synthetic room is
the quickest way to exercise the real-time path:

```sh
target/release/scan gui --synthetic --auto-start
```

## Next steps

- [Command line](cli.md): recording, replay, options, loop closure, submaps.
- [Scanner window](gui.md): controls, viewport, telemetry, datasets, Splats.
- [Colour](colour.md): how colour is registered and painted.
- [Datasets and splats](datasets.md): exporting for Brush and training notes.
