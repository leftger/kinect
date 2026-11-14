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
    COLOR_HEIGHT, COLOR_WIDTH, DEPTH_HEIGHT, DEPTH_SIZE, DEPTH_WIDTH, Device, DeviceEnumerator,
    LUT_SIZE, Opened,
};
use nalgebra::{Isometry3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};

/// Owns the sensor and turns USB packets into metric depth frames.
pub struct Capture {
    device: Device<Opened>,
    registration: Registration,
    color_from_depth: Isometry3<f32>,
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
    /// Native compressed `1920x1080` JPEG frame and calibration, if captured.
    pub raw: Option<crate::scanner::RawColorData>,
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
        match WgpuDepthProcessor::new() {
            Ok(processor) => return Ok(DepthBackend::Wgpu(processor)),
            Err(e) => {
                eprintln!("[scan] wgpu GPU depth decode unavailable ({e}); falling back to CPU");
                return Ok(DepthBackend::Cpu(
                    CpuDepthProcessor::new()
                        .map_err(|err| format!("creating CPU depth processor: {err}"))?,
                ));
            }
        }
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
        let depth_intrinsics = Intrinsics {
            fx: ir_params.fx,
            fy: ir_params.fy,
            cx: ir_params.cx,
            cy: ir_params.cy,
        };
        let color_from_depth = fit_color_from_depth(&registration, &depth_intrinsics);

        Ok(Self {
            device,
            registration,
            color_from_depth,
            depth_processor,
            intrinsics: depth_intrinsics,
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

                let jpeg = color_packet.jpeg_buffer.clone();
                let color_cp = *self.registration.color_params();
                let color_frame = color_packet
                    .process(processor)
                    .await
                    .map_err(|e| format!("processing colour packet: {e}"))?;

                // The occlusion filter drops the farther depth pixel when two
                // of them register onto the same colour sample.
                let (registered, undistorted_depth, valid) = self
                    .registration
                    .undistort_depth_and_color_with_validity(&color_frame, &depth_frame, true);

                let raw = (color_cp.fx > 0.0 && color_cp.fy > 0.0 && !jpeg.is_empty()).then(|| {
                    crate::scanner::RawColorData {
                        jpeg,
                        width: COLOR_WIDTH as u32,
                        height: COLOR_HEIGHT as u32,
                        intrinsics: Intrinsics {
                            fx: color_cp.fx,
                            fy: color_cp.fy,
                            cx: color_cp.cx,
                            cy: color_cp.cy,
                        },
                        color_from_depth: self.color_from_depth,
                    }
                });

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
                    raw,
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

/// Fit the rigid transform `color_from_depth` (`Isometry3<f32>`) mapping 3D depth-camera
/// points into the `1920x1080` RGB camera frame by minimizing reprojection residuals
/// against `Registration::xyz_to_point`.
pub(crate) fn fit_color_from_depth(
    reg: &Registration,
    depth_intrinsics: &Intrinsics,
) -> Isometry3<f32> {
    let cp = reg.color_params();
    if cp.fx <= 0.0 || cp.fy <= 0.0 {
        return Isometry3::translation(-0.052, 0.0, 0.0);
    }

    let mut obs: Vec<(Vector3<f64>, [f64; 2])> = Vec::new();
    for dy in (48..DEPTH_HEIGHT.saturating_sub(48)).step_by(32) {
        for dx in (64..DEPTH_WIDTH.saturating_sub(64)).step_by(32) {
            for &z_m in &[0.6_f32, 1.0, 1.8, 3.0] {
                let (uc, vc) = reg.xyz_to_point(dx, dy, z_m * 1000.0);
                if uc.is_finite()
                    && vc.is_finite()
                    && uc >= 0.0
                    && uc < COLOR_WIDTH as f32
                    && vc >= 0.0
                    && vc < COLOR_HEIGHT as f32
                {
                    let p = depth_intrinsics.back_project(dx as f32, dy as f32, z_m);
                    obs.push((
                        Vector3::new(p.x as f64, p.y as f64, p.z as f64),
                        [uc as f64, vc as f64],
                    ));
                }
            }
        }
    }

    if obs.len() < 12 {
        return Isometry3::translation(-0.052, 0.0, 0.0);
    }

    let fx = cp.fx as f64;
    let fy = cp.fy as f64;
    let cx = cp.cx as f64;
    let cy = cp.cy as f64;

    let mut iso: Isometry3<f64> = Isometry3::translation(-0.052, 0.0, 0.0);
    for _ in 0..12 {
        let mut h = Matrix6::<f64>::identity() * 1e-6;
        let mut b = Vector6::<f64>::zeros();
        for &(p_d, target) in &obs {
            let p_c = iso.transform_point(&nalgebra::Point3::from(p_d));
            if p_c.z <= 0.05 {
                continue;
            }
            let x = p_c.x;
            let y = p_c.y;
            let z = p_c.z;
            let inv_z = 1.0 / z;
            let inv_z2 = inv_z * inv_z;

            let r = [
                fx * (x * inv_z) + cx - target[0],
                fy * (y * inv_z) + cy - target[1],
            ];
            let j0 = [
                fx * inv_z,
                0.0,
                -fx * x * inv_z2,
                -fx * x * y * inv_z2,
                fx * (z * z + x * x) * inv_z2,
                -fx * y * inv_z,
            ];
            let j1 = [
                0.0,
                fy * inv_z,
                -fy * y * inv_z2,
                -fy * (z * z + y * y) * inv_z2,
                fy * x * y * inv_z2,
                fy * x * inv_z,
            ];
            for row_i in 0..6 {
                b[row_i] -= j0[row_i] * r[0] + j1[row_i] * r[1];
                for col_j in 0..6 {
                    h[(row_i, col_j)] += j0[row_i] * j0[col_j] + j1[row_i] * j1[col_j];
                }
            }
        }
        let Some(delta) = h.lu().solve(&b) else {
            break;
        };
        let dt = Vector3::new(delta[0], delta[1], delta[2]);
        let dw = Vector3::new(delta[3], delta[4], delta[5]);
        iso = Isometry3::from_parts(Translation3::from(dt), UnitQuaternion::from_scaled_axis(dw))
            * iso;
        if delta.norm() < 1e-8 {
            break;
        }
    }

    let t = iso.translation.vector;
    let q = iso.rotation.quaternion();
    Isometry3::from_parts(
        Translation3::new(t.x as f32, t.y as f32, t.z as f32),
        UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            q.w as f32, q.i as f32, q.j as f32, q.k as f32,
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_color_from_depth_extrinsics_from_registration_polynomial() {
        // Synthetic factory-like Kinect v2 calibration: ~52mm baseline, zero poly distortion
        // except exact shift_m = fx_c * baseline * 1000? Notice Registration defines:
        // u_c = (depth_to_color_map_x + shift_m / z_mm) * fx_c + cx_c
        // For pure x translation t_x = -0.052m -> X_c/Z_c = X_d/Z_d - 0.052/Z_m = X_d/Z_d - 52/Z_mm.
        let mut reg = Registration::new();
        let ir = kinect_one::data::IrParams {
            fx: 365.0,
            fy: 365.0,
            cx: 256.0,
            cy: 212.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let fx_c = 1081.37_f32;
        let cx_c = 959.5_f32;
        let cy_c = 539.5_f32;
        let color_q = 0.002199_f32;
        let depth_q = 0.01_f32;
        // wx / (fx_c * color_q) = (mx) * depth_q / ... let's express X_d/Z_d linearly:
        // mx_depth = (x - ir.cx) * depth_q -> X_d/Z_d = mx_depth / (ir.fx * depth_q)
        // We want depth_to_color_map_x = X_d/Z_d (with shift_d huge -> shift_m/shift_d ~ 0)
        // and shift_m = -52.0 (in mm! Since -52.0 / z_mm = -0.052 / z_m).
        // And depth_to_color_map_y = wy / color_q + cy_c = fy_c * (y - ir.cy)/ir.fy + cy_c.
        let cp = kinect_one::data::ColorParams {
            fx: fx_c,
            fy: fx_c,
            cx: cx_c,
            cy: cy_c,
            shift_d: 1e9,
            shift_m: -52.0,
            mx_x1y0: (fx_c * color_q) / (ir.fx * depth_q),
            my_x0y1: (fx_c * color_q) / (ir.fy * depth_q),
            ..Default::default()
        };
        reg.set_ir_params(&ir);
        reg.set_color_params(&cp);

        let depth_intrinsics = Intrinsics {
            fx: ir.fx,
            fy: ir.fy,
            cx: ir.cx,
            cy: ir.cy,
        };
        let fit = fit_color_from_depth(&reg, &depth_intrinsics);
        let t = fit.translation.vector;
        assert!(
            (t.x + 0.052).abs() < 1e-4 && t.y.abs() < 1e-4 && t.z.abs() < 1e-4,
            "recovered translation {t:?} far from (-0.052, 0, 0)"
        );
    }
}
