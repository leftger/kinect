//! Live capture from the Kinect v2.

use std::error::Error;

#[cfg(feature = "wgpu-decode")]
use crate::wgpu_depth::WgpuDepthProcessor;
use geom::Intrinsics;
use kinect_one::config::Config;
use kinect_one::data::P0Tables;
use kinect_one::processor::color::{ColorSpace, ZuneColorProcessor};
#[cfg(feature = "gpu-decode")]
use kinect_one::processor::depth::OpenCLDepthProcessor;
use kinect_one::processor::depth::{
    CpuDepthProcessor, DepthFrame, DepthPacket, DepthProcessorTrait, IrFrame,
};
use kinect_one::processor::{ProcessTrait, ProcessorTrait, Registration};
use kinect_one::{
    Device, DeviceEnumerator, Opened, DEPTH_HEIGHT, DEPTH_SIZE, DEPTH_WIDTH, LUT_SIZE,
};

/// Owns the sensor and turns USB packets into metric depth frames.
pub struct Capture {
    device: Device<Opened>,
    registration: Registration,
    depth_processor: DepthBackend,
    intrinsics: Intrinsics,
    /// Capture colour as well as depth, for texturing. Off by default: the
    /// colour stream delivers at about a third of the depth rate, so waiting for
    /// a packet costs more than half the frame time.
    color: bool,
    color_processor: Option<ZuneColorProcessor>,
    captured: Option<CapturedColor>,
    frame: Vec<f32>,
}

/// Which decoder turns raw depth packets into depth frames.
///
/// The two backends implement the same traits, so the rest of the pipeline does
/// not care which is in use. The enum exists only so the choice can be made at
/// runtime: the OpenCL path needs a driver that may not be installed, and the
/// CPU path is the fallback that always works.
/// Colour for one frame, registered into the depth camera's grid.
///
/// Registering it here rather than at texturing time is deliberate: registration
/// needs the colour frame and the raw depth frame together, and afterwards only
/// the pose is missing. It is also the expensive part, done once per frame rather
/// than once per vertex.
pub struct CapturedColor {
    /// RGB, `DEPTH_WIDTH * DEPTH_HEIGHT * 3`, in the depth grid.
    pub rgb: Vec<u8>,
    /// Undistorted depth in metres, same grid, for the visibility test.
    pub depth: Vec<f32>,
    /// One byte per depth pixel. `1` where `rgb` was copied from the colour
    /// camera, `0` where registration left it untouched. A zero colour sample
    /// is also a real black pixel, so this mask is the only way to tell.
    pub valid: Vec<u8>,
    /// Colour-camera settings for this frame, copied off the colour packet.
    pub exposure: f32,
    pub gain: f32,
    pub gamma: f32,
}

enum DepthBackend {
    Cpu(CpuDepthProcessor),
    #[cfg(feature = "gpu-decode")]
    Gpu(OpenCLDepthProcessor),
    #[cfg(feature = "wgpu-decode")]
    Wgpu(WgpuDepthProcessor),
}

impl DepthProcessorTrait for DepthBackend {
    fn set_config(&mut self, config: &Config) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Cpu(processor) => processor.set_config(config),
            #[cfg(feature = "gpu-decode")]
            Self::Gpu(processor) => processor.set_config(config),
            #[cfg(feature = "wgpu-decode")]
            Self::Wgpu(processor) => processor.set_config(config),
        }
    }

    fn set_p0_tables(&mut self, p0_tables: &P0Tables) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Cpu(processor) => processor.set_p0_tables(p0_tables),
            #[cfg(feature = "gpu-decode")]
            Self::Gpu(processor) => processor.set_p0_tables(p0_tables),
            #[cfg(feature = "wgpu-decode")]
            Self::Wgpu(processor) => processor.set_p0_tables(p0_tables),
        }
    }

    fn set_x_z_tables(
        &mut self,
        x_table: &[f32; DEPTH_SIZE],
        z_table: &[f32; DEPTH_SIZE],
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Cpu(processor) => processor.set_x_z_tables(x_table, z_table),
            #[cfg(feature = "gpu-decode")]
            Self::Gpu(processor) => processor.set_x_z_tables(x_table, z_table),
            #[cfg(feature = "wgpu-decode")]
            Self::Wgpu(processor) => processor.set_x_z_tables(x_table, z_table),
        }
    }

    fn set_lookup_table(&mut self, lut: &[i16; LUT_SIZE]) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Cpu(processor) => processor.set_lookup_table(lut),
            #[cfg(feature = "gpu-decode")]
            Self::Gpu(processor) => processor.set_lookup_table(lut),
            #[cfg(feature = "wgpu-decode")]
            Self::Wgpu(processor) => processor.set_lookup_table(lut),
        }
    }
}

impl ProcessorTrait<DepthPacket, (IrFrame, DepthFrame)> for DepthBackend {
    async fn process(&self, input: DepthPacket) -> Result<(IrFrame, DepthFrame), Box<dyn Error>> {
        match self {
            Self::Cpu(processor) => processor.process(input).await,
            #[cfg(feature = "gpu-decode")]
            Self::Gpu(processor) => processor.process(input).await,
            #[cfg(feature = "wgpu-decode")]
            Self::Wgpu(processor) => processor.process(input).await,
        }
    }
}

/// Pick an OpenCL device, preferring an actual GPU.
///
/// A CPU device (pocl) is accepted as a fallback so the path can at least be
/// validated, but it is reported clearly: it would be a correctness check, not a
/// speed-up.
#[cfg(feature = "gpu-decode")]
fn opencl_device() -> Result<ocl::Device, Box<dyn Error>> {
    use ocl::{Device, DeviceType, Platform};

    let platforms = Platform::list();
    if platforms.is_empty() {
        return Err("no OpenCL platform found. Install a driver, e.g.\n\
                    \x20   sudo apt install mesa-opencl-icd\n\
                    (Mesa's Rusticl supports Intel Gen8 and newer), then retry."
            .into());
    }

    // Ask for GPUs explicitly rather than listing everything and inspecting the
    // type: `ocl`'s device-type query is an `Option<DeviceType>` filter here, and
    // comparing devices for identity is not part of its API.
    let mut fallback = None;
    for platform in &platforms {
        for device in Device::list(platform, Some(DeviceType::GPU))? {
            eprintln!(
                "[scan] OpenCL GPU: {} on {}",
                device.name()?,
                platform.name()?
            );
            return Ok(device);
        }

        // No GPU on this platform. Remember one CPU device so the path can still
        // be validated on a box with only pocl -- it is a correctness check, not
        // a speed-up, and it is reported as such.
        if fallback.is_none() {
            if let Some(device) = Device::list(platform, None)?.into_iter().next() {
                eprintln!(
                    "[scan] OpenCL device: {} on {} (CPU device, so this is a \
                     correctness check rather than a speed-up)",
                    device.name()?,
                    platform.name()?
                );
                fallback = Some(device);
            }
        }
    }

    fallback.ok_or_else(|| "OpenCL platforms are present but expose no devices".into())
}

fn build_backend(use_gpu: bool) -> Result<DepthBackend, Box<dyn Error>> {
    if !use_gpu {
        return Ok(DepthBackend::Cpu(
            CpuDepthProcessor::new().map_err(|e| format!("creating CPU depth processor: {e}"))?,
        ));
    }

    // Preference order is deliberate. Vulkan works on this GPU; Rusticl does
    // not, despite enumerating a device and accepting every buffer -- it simply
    // never executes a kernel. See examples/ocl_check.rs and vk_check.rs.
    #[cfg(feature = "wgpu-decode")]
    {
        let processor = WgpuDepthProcessor::new()
            .map_err(|e| format!("creating the GPU depth processor: {e}"))?;
        return Ok(DepthBackend::Wgpu(processor));
    }

    #[cfg(all(feature = "gpu-decode", not(feature = "wgpu-decode")))]
    {
        let processor = OpenCLDepthProcessor::new(opencl_device()?)
            .map_err(|e| format!("creating OpenCL depth processor: {e}"))?;
        return Ok(DepthBackend::Gpu(processor));
    }

    #[cfg(all(not(feature = "gpu-decode"), not(feature = "wgpu-decode")))]
    {
        Err(
            "this binary has no GPU decoder; rebuild with `--features wgpu-decode` \
             (Vulkan or Metal) or `--features gpu-decode` (OpenCL)"
                .into(),
        )
    }
}

impl Capture {
    /// Open the first Kinect v2 and start streaming.
    ///
    /// `filters` toggles the decoder's bilateral and edge-aware filters. They do
    /// not change the noise level on good surfaces, but they remove flying pixels
    /// and out-of-range junk; turning them off roughly doubles throughput.
    ///
    /// `use_gpu` selects the OpenCL decoder when the binary was built with the
    /// `gpu-decode` feature.
    pub async fn open(filters: bool, color: bool, use_gpu: bool) -> Result<Self, Box<dyn Error>> {
        let mut device = DeviceEnumerator::open_default(true)
            .await
            .map_err(|e| format!("could not open a Kinect v2: {e}"))?;

        device
            .start()
            .await
            .map_err(|e| format!("starting streams: {e}"))?;

        // Calibration is only populated *by* start(); reading it earlier silently
        // yields all-zero intrinsics.
        let ir_params = *device.get_ir_params();
        let color_params = *device.get_color_params();
        let p0_tables = device.get_p0_tables().clone();

        if ir_params.fx == 0.0 {
            return Err("device reported zero focal length; capture is unreliable".into());
        }

        let mut depth_processor = build_backend(use_gpu)?;

        // Order matters, and only for the GPU path: the OpenCL processor rebuilds
        // its program *and allocates fresh buffers* in `set_config`, so any table
        // uploaded before it is silently thrown away. Config first, then tables.
        // The CPU processor does not care, so one order serves both.
        depth_processor.set_config(&Config {
            enable_bilateral_filter: filters,
            enable_edge_aware_filter: filters,
            ..Config::default()
        })?;
        depth_processor.set_p0_tables(&p0_tables)?;
        depth_processor.set_ir_params(&ir_params)?;

        let color_processor = if color {
            Some(ZuneColorProcessor::new(ColorSpace::RGB)?)
        } else {
            None
        };

        let mut registration = Registration::new();
        registration.set_ir_params(&ir_params);
        registration.set_color_params(&color_params);

        Ok(Self {
            device,
            registration,
            depth_processor,
            intrinsics: Intrinsics {
                fx: ir_params.fx,
                fy: ir_params.fy,
                cx: ir_params.cx,
                cy: ir_params.cy,
            },
            color,
            color_processor,
            captured: None,
            frame: Vec::new(),
        })
    }

    pub fn intrinsics(&self) -> Intrinsics {
        self.intrinsics
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (DEPTH_WIDTH, DEPTH_HEIGHT)
    }

    /// Block until the next depth frame arrives, and return it in metres,
    /// undistorted.
    pub async fn next_frame(&mut self) -> Result<&[f32], Box<dyn Error>> {
        loop {
            let Some(packet) = self.device.poll_depth_packet().await? else {
                continue;
            };

            let (_ir_frame, depth_frame) = packet
                .process(&self.depth_processor)
                .await
                .map_err(|e| format!("processing depth packet: {e}"))?;

            // Colour is captured every frame when enabled, and the pair is
            // registered together because that is what registration needs. It
            // costs the wait documented on the `color` field; capturing it less
            // often would cost proportionally less and is the obvious next step.
            if let Some(processor) = self.color_processor.as_ref() {
                let color_packet = loop {
                    if let Some(packet) = self.device.poll_color_packet().await? {
                        break packet;
                    }
                };

                let color_frame = color_packet
                    .process(processor)
                    .await
                    .map_err(|e| format!("processing colour packet: {e}"))?;

                // The occlusion filter drops the farther depth pixel when two
                // of them register onto the same colour sample.
                let (registered, undistorted_depth, valid) = self
                    .registration
                    .undistort_depth_and_color_with_validity(&color_frame, &depth_frame, true);

                self.captured = Some(CapturedColor {
                    rgb: registered.buffer,
                    depth: undistorted_depth
                        .buffer
                        .iter()
                        .map(|millimetres| millimetres / 1000.0)
                        .collect(),
                    valid,
                    exposure: registered.exposure,
                    gain: registered.gain,
                    gamma: registered.gamma,
                });
            }

            let undistorted = self.registration.undistort_depth(&depth_frame);

            self.frame.clear();
            self.frame
                .extend(undistorted.buffer.iter().map(|millimetres| {
                    let metres = millimetres / 1000.0;
                    // The decoder uses 0 for "no measurement"; make that uniform with
                    // the NaN convention the geometry crate expects.
                    if metres.is_finite() && metres > 0.0 {
                        metres
                    } else {
                        f32::NAN
                    }
                }));

            return Ok(&self.frame);
        }
    }

    /// The colour for the frame most recently returned by `next_frame`.
    pub fn color(&self) -> Option<&CapturedColor> {
        self.captured.as_ref()
    }

    pub async fn stop(&mut self) -> Result<(), Box<dyn Error>> {
        self.device.stop().await?;
        Ok(())
    }
}
