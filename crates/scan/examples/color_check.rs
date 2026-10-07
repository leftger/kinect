//! Does the colour stream line up with the depth?
//!
//! Texturing the mesh is only worth building if the IR and colour cameras are
//! actually registered. That is a cheap thing to check and an expensive thing to
//! assume: if the alignment is off, no amount of blending code will fix it. This
//! is the same check that should have preceded the Vulkan port.
//!
//! `Registration::undistort_depth_and_color` is the primitive: it warps the
//! colour image into the *distorted* depth frame's grid, so the result is a
//! 512x424 RGB image that should look like a photograph and should have its
//! edges in the same places as the depth discontinuities.
//!
//! Writes two PPM files next to each other, which is a format Pillow can convert
//! without this crate taking an image dependency:
//!
//!   color_registered.ppm   the colour warped into the depth grid
//!   color_overlay.ppm      the same, with depth discontinuities marked magenta
//!
//! If the two sets of edges coincide, registration works and texturing is
//! straightforward. If the magenta outline walks off the colour edges, it is not
//! and the problem is upstream of anything worth writing.
//!
//! Run with:
//!     cargo run --release -p scan --example color_check

use std::error::Error;

use kinect_one::config::Config;
use kinect_one::processor::color::{ColorSpace, ZuneColorProcessor};
use kinect_one::processor::depth::{CpuDepthProcessor, DepthProcessorTrait};
use kinect_one::processor::{ProcessTrait, ProcessorTrait, Registration};
use kinect_one::{DeviceEnumerator, DEPTH_HEIGHT, DEPTH_SIZE, DEPTH_WIDTH};

/// A depth step larger than this counts as a discontinuity, in millimetres.
///
/// Millimetres, because the decoder's output is millimetres: the scanner divides
/// by 1000 on the way in. Using 0.05 here marked every pixel in the frame as an
/// edge, which is a mistake worth leaving a note about.
const EDGE_MM: f32 = 50.0;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut device = DeviceEnumerator::open_default(true)
        .await
        .map_err(|e| format!("could not open a Kinect v2: {e}"))?;
    device
        .start()
        .await
        .map_err(|e| format!("starting streams: {e}"))?;

    // Calibration is only populated *by* start().
    let ir_params = *device.get_ir_params();
    let color_params = *device.get_color_params();
    let p0_tables = device.get_p0_tables().clone();
    eprintln!(
        "[color] ir fx {:.2} fy {:.2} cx {:.2} cy {:.2}",
        ir_params.fx, ir_params.fy, ir_params.cx, ir_params.cy
    );
    eprintln!(
        "[color] color fx {:.2} fy {:.2} cx {:.2} cy {:.2} shift_m {:.4}",
        color_params.fx, color_params.fy, color_params.cx, color_params.cy, color_params.shift_m
    );

    let color_processor = ZuneColorProcessor::new(ColorSpace::RGB)?;

    let mut depth_processor = CpuDepthProcessor::new()?;
    depth_processor.set_config(&Config::default())?;
    depth_processor.set_p0_tables(&p0_tables)?;
    depth_processor.set_ir_params(&ir_params)?;

    let mut registration = Registration::new();
    registration.set_ir_params(&ir_params);
    registration.set_color_params(&color_params);

    // Colour packets arrive at roughly a third of the depth rate, so the depth
    // frame is taken first and the colour frame that follows it. The scene is
    // static, so the gap does not matter.
    eprintln!("[color] waiting for a depth frame ...");
    let depth_packet = loop {
        if let Some(packet) = device.poll_depth_packet().await? {
            break packet;
        }
    };

    eprintln!("[color] waiting for a colour frame (this is the slow stream) ...");
    let color_packet = loop {
        if let Some(packet) = device.poll_color_packet().await? {
            break packet;
        }
    };

    let (_, depth_frame) = depth_packet.process(&depth_processor).await?;
    let color_frame = color_packet.process(&color_processor).await?;

    eprintln!(
        "[color] colour frame {}x{} {:?}, exposure {:.4} gain {:.4} gamma {:.4}",
        color_frame.width,
        color_frame.height,
        color_frame.color_space,
        color_frame.exposure,
        color_frame.gain,
        color_frame.gamma
    );

    let valid_depth = depth_frame.buffer.iter().filter(|d| **d > 0.0).count();
    eprintln!("[color] depth frame: {valid_depth}/{DEPTH_SIZE} pixels carry depth");

    // The registration primitive. Input is the *raw* depth frame: the function
    // walks `distort_map` from undistorted to distorted coordinates itself.
    let (registered, undistorted) =
        registration.undistort_depth_and_color(&color_frame, &depth_frame, false);

    let bytes_per_pixel = registered.color_space.bytes_per_pixel();
    eprintln!(
        "[color] registered frame {}x{} {:?}, {} bytes/px",
        registered.width, registered.height, registered.color_space, bytes_per_pixel
    );

    if bytes_per_pixel < 3 {
        return Err(format!("unexpected colour space {:?}", registered.color_space).into());
    }

    write_ppm(
        "color_registered.ppm",
        &registered.buffer,
        DEPTH_WIDTH,
        DEPTH_HEIGHT,
    )?;

    // Mark depth discontinuities so the two edge sets can be compared by eye.
    let mut overlay = registered.buffer.clone();
    let mut marked = 0usize;

    for y in 0..DEPTH_HEIGHT - 1 {
        for x in 0..DEPTH_WIDTH - 1 {
            let index = y * DEPTH_WIDTH + x;
            let depth = undistorted.buffer[index];

            // Only where there is a surface on both sides: an edge against the
            // background is not evidence of anything.
            let right = undistorted.buffer[index + 1];
            let below = undistorted.buffer[index + DEPTH_WIDTH];
            if depth <= 0.0 || right <= 0.0 || below <= 0.0 {
                continue;
            }

            let step = (right - depth).abs().max((below - depth).abs());
            if step <= EDGE_MM {
                continue;
            }

            marked += 1;
            let pixel = index * bytes_per_pixel;
            overlay[pixel] = 255;
            overlay[pixel + 1] = 0;
            overlay[pixel + 2] = 255;
        }
    }

    eprintln!("[color] marked {marked} depth-discontinuity pixels");

    write_ppm("color_overlay.ppm", &overlay, DEPTH_WIDTH, DEPTH_HEIGHT)?;

    eprintln!("[color] wrote color_registered.ppm and color_overlay.ppm");
    eprintln!("[color] convert with: python3 -c \"from PIL import Image; ...\"");

    Ok(())
}

/// Binary PPM (P6). Avidemux of formats, but readable by Pillow and requiring no
/// dependency here.
fn write_ppm(path: &str, buffer: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;
    write!(file, "P6\n{width} {height}\n255\n")?;
    file.write_all(&buffer[..width * height * 3])?;

    Ok(())
}
