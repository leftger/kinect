//! Bring-up probe for the Kinect v2 (Xbox One Kinect) on Linux.
//!
//! Answers two questions before any scanning code gets written:
//! * does the pure-Rust driver actually pull depth frames off this sensor?
//! * where does the per-frame time actually go?
//!
//! It opens the device, starts streaming, runs the CPU depth packet processor,
//! and dumps registration-corrected depth frames plus back-projected point
//! clouds. No OpenCL, no C++ toolchain, no OpenGL.
//!
//! Usage: `probe [--no-color] [--no-filter]`
//!   --no-color    never poll the colour stream (tests how much USB bandwidth and
//!                 JPEG handling the unused colour stream is costing us)
//!   --no-filter   disable the decoder's bilateral/edge-aware filters, which is
//!                 the single biggest lever on decode time
//!
//! Undistortion uses the vendored driver's `Registration::undistort_depth`. That
//! used to panic on border pixels because the port dropped upstream
//! libfreenect2's out-of-image bounds check; the fix lives in
//! `vendor/kinect-one/src/processor/registration.rs` (see `LOCAL PATCH` there).

use std::error::Error;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use kinect_one::config::Config;
use kinect_one::processor::depth::{CpuDepthProcessor, DepthFrame, DepthProcessorTrait};
use kinect_one::processor::{ProcessTrait, Registration};
use kinect_one::{DeviceEnumerator, DEPTH_HEIGHT, DEPTH_SIZE, DEPTH_WIDTH};

/// Frames discarded while the depth processor and exposure settle.
const WARMUP_FRAMES: usize = 5;
/// Frames to measure and save.
const CAPTURE_FRAMES: usize = 20;
/// Spatial subsampling when building the point cloud (2 = every other pixel).
const CLOUD_STRIDE: usize = 2;
/// Fail loudly if the stream starts but never produces a frame.
const STREAM_TIMEOUT: Duration = Duration::from_secs(15);

const OUT_DIR: &str = "out";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let drain_color = !std::env::args().any(|a| a == "--no-color");
    let use_filters = !std::env::args().any(|a| a == "--no-filter");

    fs::create_dir_all(OUT_DIR)?;

    println!("[probe] opening Kinect v2 ...");
    let mut device = DeviceEnumerator::open_default(true).await.map_err(|e| {
        format!(
            "could not open a Kinect v2: {e}\n\
             \x20   - is it on a USB 3.0 port (it needs SuperSpeed)?\n\
             \x20   - do you have permission for the USB device node (udev rule / root)?"
        )
    })?;

    device
        .start()
        .await
        .map_err(|e| format!("starting streams: {e}"))?;
    println!("[probe] streams started (colour polling: {drain_color})");

    // NOTE: `start()` is what populates the factory calibration, so these have to
    // be read *after* it. Read beforehand you silently get all-zero intrinsics
    // and a P0 table of zeroes, which decodes every depth pixel to nothing.
    let ir_params = *device.get_ir_params();
    let color_params = *device.get_color_params();
    let p0_tables = device.get_p0_tables().clone();

    println!(
        "[probe] IR intrinsics: fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        ir_params.fx, ir_params.fy, ir_params.cx, ir_params.cy
    );

    if p0_tables.p0_table0.iter().all(|&v| v == 0) {
        return Err("P0 calibration table is all zeros; depth decoding cannot work".into());
    }

    let mut depth_processor =
        CpuDepthProcessor::new().map_err(|e| format!("creating CPU depth processor: {e}"))?;
    depth_processor.set_p0_tables(&p0_tables)?;
    depth_processor.set_ir_params(&ir_params)?;
    depth_processor.set_config(&Config {
        enable_bilateral_filter: use_filters,
        enable_edge_aware_filter: use_filters,
        ..Config::default()
    })?;

    let mut registration = Registration::new();
    registration.set_ir_params(&ir_params);
    registration.set_color_params(&color_params);

    println!("[probe] waiting for depth frames ...");

    let started = Instant::now();
    let mut seen = 0usize;
    let mut saved = 0usize;
    let mut timing = Timing::default();

    while saved < CAPTURE_FRAMES {
        if seen == 0 && started.elapsed() > STREAM_TIMEOUT {
            return Err(format!(
                "stream started but no depth frame arrived within {STREAM_TIMEOUT:?}"
            )
            .into());
        }

        // Each poll re-submits that stream's isochronous transfers. Not polling
        // colour at all leaves the colour endpoints idle, which frees USB
        // bandwidth and skips JPEG handling entirely.
        let t_poll = Instant::now();
        if drain_color {
            let _ = device.poll_color_packet().await;
        }
        let Some(packet) = device.poll_depth_packet().await? else {
            continue;
        };
        timing.poll += t_poll.elapsed();

        let t = Instant::now();
        let (_ir_frame, depth_frame) = packet
            .process(&depth_processor)
            .await
            .map_err(|e| format!("processing depth packet: {e}"))?;
        timing.process += t.elapsed();

        seen += 1;
        if seen <= WARMUP_FRAMES {
            continue;
        }

        let t = Instant::now();
        let undistorted = registration.undistort_depth(&depth_frame);
        timing.undistort += t.elapsed();

        let t = Instant::now();
        let cloud = backproject(&registration, &undistorted);
        timing.backproject += t.elapsed();

        let t = Instant::now();
        if saved < 3 {
            let ply_path = format!("{OUT_DIR}/frame_{saved:03}.ply");
            let pgm_path = format!("{OUT_DIR}/depth_{saved:03}.pgm");
            write_ply(Path::new(&ply_path), &cloud)?;
            write_pgm(Path::new(&pgm_path), &undistorted.buffer)?;
        }
        timing.write += t.elapsed();

        timing.frames += 1;
        timing.total += t_poll.elapsed();

        let stats = depth_stats(&depth_frame.buffer);
        if saved < 5 || saved % 5 == 0 {
            println!(
                "[probe] frame {seen} seq={} valid={:.1}% range=[{:.0}..{:.0}]mm points={} hash={:016x}",
                depth_frame.sequence,
                stats.valid_fraction * 100.0,
                stats.min_mm,
                stats.max_mm,
                cloud.len(),
                depth_hash(&depth_frame.buffer),
            );
        }

        saved += 1;
    }

    device.stop().await?;

    println!("\n[probe] --- timing per frame ({} frames) ---", timing.frames);
    println!("{:>12}: {:8.1} ms", "poll (USB)", timing.poll.as_secs_f64() * 1000.0 / timing.frames as f64);
    println!("{:>12}: {:8.1} ms", "depth proc", timing.process.as_secs_f64() * 1000.0 / timing.frames as f64);
    println!("{:>12}: {:8.1} ms", "undistort", timing.undistort.as_secs_f64() * 1000.0 / timing.frames as f64);
    println!("{:>12}: {:8.1} ms", "backproject", timing.backproject.as_secs_f64() * 1000.0 / timing.frames as f64);
    println!("{:>12}: {:8.1} ms", "file write", timing.write.as_secs_f64() * 1000.0 / timing.frames as f64);
    println!(
        "{:>12}: {:8.1} fps",
        "throughput",
        timing.frames as f64 / timing.total.as_secs_f64()
    );

    println!("\n[probe] SUCCESS: driver works end-to-end on this machine");
    Ok(())
}

#[derive(Default)]
struct Timing {
    poll: Duration,
    process: Duration,
    undistort: Duration,
    backproject: Duration,
    write: Duration,
    total: Duration,
    frames: usize,
}

struct DepthStats {
    valid_fraction: f32,
    min_mm: f32,
    max_mm: f32,
}

/// A stable, scheduling-independent hash of a depth buffer, used to prove that a
/// change to the decode path (e.g. parallelising it) produces *identical* pixels
/// rather than merely similar-looking ones.
///
/// FNV-1a over each value's bit pattern, in pixel order. Thread scheduling
/// cannot affect the result, but any changed pixel -- including a NaN that
/// became a 0.0 -- will change it.
fn depth_hash(buffer: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for d in buffer {
        for byte in d.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Depth values are `f32` millimetres; invalid pixels come back as NaN/0.
fn depth_stats(buffer: &[f32]) -> DepthStats {
    let mut valid = 0usize;
    let mut min_mm = f32::INFINITY;
    let mut max_mm: f32 = 0.0;

    for &d in buffer {
        if d.is_finite() && d > 1.0 {
            valid += 1;
            min_mm = min_mm.min(d);
            max_mm = max_mm.max(d);
        }
    }

    DepthStats {
        valid_fraction: valid as f32 / buffer.len() as f32,
        min_mm: if valid > 0 { min_mm } else { 0.0 },
        max_mm,
    }
}

/// Turn an undistorted depth image into 3D points (metres) in the depth camera
/// frame. Because `undistorted` has had lens distortion removed, the plain
/// pinhole model in `Registration::point_to_xyz` applies.
fn backproject(registration: &Registration, undistorted: &DepthFrame) -> Vec<[f32; 3]> {
    let capacity = DEPTH_SIZE / (CLOUD_STRIDE * CLOUD_STRIDE);
    let mut points = Vec::with_capacity(capacity);

    for y in (0..DEPTH_HEIGHT).step_by(CLOUD_STRIDE) {
        for x in (0..DEPTH_WIDTH).step_by(CLOUD_STRIDE) {
            let (px, py, pz) = registration.point_to_xyz(undistorted, x, y);
            if px.is_finite() && py.is_finite() && pz.is_finite() {
                points.push([px, py, pz]);
            }
        }
    }

    points
}

fn write_ply(path: &Path, points: &[[f32; 3]]) -> Result<(), Box<dyn Error>> {
    let mut w = BufWriter::new(fs::File::create(path)?);

    write!(
        w,
        "ply\nformat binary_little_endian 1.0\n\
         element vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         end_header\n",
        points.len()
    )?;

    for p in points {
        for c in p {
            w.write_all(&c.to_le_bytes())?;
        }
    }

    w.flush()?;
    Ok(())
}

/// 8-bit grayscale preview, clipped to the 0.5-4.5 m working range.
fn write_pgm(path: &Path, depth_mm: &[f32]) -> Result<(), Box<dyn Error>> {
    const MIN_MM: f32 = 500.0;
    const MAX_MM: f32 = 4500.0;

    let mut w = BufWriter::new(fs::File::create(path)?);
    write!(w, "P5\n{DEPTH_WIDTH} {DEPTH_HEIGHT}\n255\n")?;

    let mut row = Vec::with_capacity(DEPTH_WIDTH);
    for y in 0..DEPTH_HEIGHT {
        row.clear();
        for x in 0..DEPTH_WIDTH {
            let d = depth_mm[y * DEPTH_WIDTH + x];
            let v = if d.is_finite() && d > MIN_MM {
                (((d - MIN_MM) / (MAX_MM - MIN_MM)).clamp(0.0, 1.0) * 255.0) as u8
            } else {
                0
            };
            row.push(v);
        }
        w.write_all(&row)?;
    }

    w.flush()?;
    Ok(())
}
