//! Depth decoding on the GPU, via the WGSL port of libfreenect2's OpenCL kernels.
//!
//! Same four stages as the CPU decoder, same traits, so it is a drop-in
//! alternative to `CpuDepthProcessor`:
//!
//! ```text
//! processPixelStage1 -> filterPixelStage1 -> processPixelStage2 -> filterPixelStage2
//! ```
//!
//! The shader source lives in `shaders/depth_decode.wgsl`. Its compile-time
//! constants come from `DepthProcessorParams` and `Config`, exactly as the
//! OpenCL processor passes them as `-D` defines to its kernel build, and are
//! substituted into a marker line here.
//!
//! Why this exists at all: the decode caps capture rate. It is 5.3 fps on the
//! CPU, and the sensor streams at 30, so faster decode means more frames at
//! smaller motion steps, which is what the odometry wants.
//!
//! Why not OpenCL: Rusticl on this GPU enumerates a device, accepts buffers and
//! transfers them correctly, and then silently never executes a kernel. Vulkan
//! compute works. `examples/ocl_check.rs` and `examples/vk_check.rs` are the two
//! measurements behind that.
//!
//! # What it actually bought
//!
//! Not capture rate, which is what it was built for.
//!
//! `record`, 60 frames, filters off, on this machine:
//!
//! ```text
//!              wall     user CPU
//!   CPU        18.1 s      8.4 s
//!   Vulkan     15.7 s      1.4 s
//! ```
//!
//! and the whole `live` pipeline -- decode, odometry, fusion, mesh -- 60 frames:
//!
//! ```text
//!              wall     user CPU
//!   CPU        45.4 s     24.3 s
//!   Vulkan     46.3 s     18.2 s
//! ```
//!
//! So the decode gets six to thirteen times cheaper on the host and the wall
//! clock does not move. The premise this port was built on -- that the decode is
//! what caps capture rate -- came from a measurement taken with the filters *on*,
//! where the decode cost about 148 ms a frame. With them off it is a small part
//! of a frame that costs roughly 760 ms end to end, and the rest is spent
//! somewhere this port does not touch.
//!
//! The next person to optimise this should measure where that 760 ms goes before
//! assuming; the decode is no longer it.

use std::error::Error;

use kinect_one::config::Config;
use kinect_one::processor::depth::{DepthFrame, DepthPacket, DepthProcessorTrait, IrFrame};
use kinect_one::processor::ProcessorTrait;
use kinect_one::{DepthProcessorParams, DEPTH_SIZE, LUT_SIZE};
use wgpu::util::DeviceExt;

const WORKGROUP: u32 = 64;

/// One storage buffer per binding in the shader.
const BINDINGS: u32 = 15;

/// `p0_table` is uploaded as `vec4<f32>`: WGSL gives `vec3<f32>` an array stride
/// of 16, so a packed `[f32; 3n]` buffer would be read at the wrong offsets.
const P0_STRIDE: usize = 4;

struct Buffers {
    lut: wgpu::Buffer,
    z_table: wgpu::Buffer,
    x_table: wgpu::Buffer,
    p0_table: wgpu::Buffer,
    packet: wgpu::Buffer,
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    n: wgpu::Buffer,
    ir: wgpu::Buffer,
    a_filtered: wgpu::Buffer,
    b_filtered: wgpu::Buffer,
    edge_test: wgpu::Buffer,
    depth: wgpu::Buffer,
    ir_sum: wgpu::Buffer,
    filtered: wgpu::Buffer,
    /// Read-back staging for depth and IR.
    depth_staging: wgpu::Buffer,
    ir_staging: wgpu::Buffer,
}

/// The stages to dispatch, in order, for a given configuration.
///
/// The OpenCL original compiles the filters out entirely when they are disabled
/// rather than guarding inside them, and stage 2 reads whichever amplitude
/// buffers stage 1 or the bilateral filter produced.
fn pass_list(config: &Config) -> Vec<&'static str> {
    let mut passes = vec!["process_pixel_stage1"];

    if config.enable_bilateral_filter {
        passes.push("filter_pixel_stage1");
    }
    passes.push("process_pixel_stage2");
    if config.enable_edge_aware_filter {
        passes.push("filter_pixel_stage2");
    }

    passes
}

/// Render the `const` block the shader is built with.
fn constants(params: &DepthProcessorParams, config: &Config) -> String {
    // Derived exactly as the OpenCL build options do.
    let joint_bilateral_threshold = (params.joint_bilateral_ab_threshold
        * params.joint_bilateral_ab_threshold)
        / (params.ab_multiplier * params.ab_multiplier);

    let gaussian = |i: usize| params.gaussian_kernel[i];

    format!(
        r#"
const BFI_BITMASK: u32 = {bfi}u;
const AB_MULTIPLIER: f32 = {ab_multiplier};
const AB_PER_FRQ: vec3<f32> = vec3<f32>({ab0}, {ab1}, {ab2});
const AB_OUTPUT_MULTIPLIER: f32 = {ab_output_multiplier};
const PHASE: vec3<f32> = vec3<f32>({ph0}, {ph1}, {ph2});
const M_PI_F: f32 = {pi};
const PHASE_OFFSET: f32 = {phase_offset};
const UNAMBIGUOUS_DIST: f32 = {unambiguous_dist};
const INDIVIDUAL_AB_THRESHOLD: f32 = {individual_ab_threshold};
const AB_THRESHOLD: f32 = {ab_threshold};
const AB_CONFIDENCE_SLOPE: f32 = {ab_confidence_slope};
const AB_CONFIDENCE_OFFSET: f32 = {ab_confidence_offset};
const MIN_DEALIAS_CONFIDENCE: f32 = {min_dealias_confidence};
const MAX_DEALIAS_CONFIDENCE: f32 = {max_dealias_confidence};
const GAUSSIAN_0: f32 = {g0};
const GAUSSIAN_1: f32 = {g1};
const GAUSSIAN_2: f32 = {g2};
const GAUSSIAN_3: f32 = {g3};
const GAUSSIAN_4: f32 = {g4};
const GAUSSIAN_5: f32 = {g5};
const GAUSSIAN_6: f32 = {g6};
const GAUSSIAN_7: f32 = {g7};
const GAUSSIAN_8: f32 = {g8};
const JOINT_BILATERAL_EXP: f32 = {joint_bilateral_exp};
const JOINT_BILATERAL_THRESHOLD: f32 = {joint_bilateral_threshold};
const JOINT_BILATERAL_MAX_EDGE: f32 = {joint_bilateral_max_edge};
const USE_BILATERAL: bool = {use_bilateral};
const EDGE_AB_AVG_MIN_VALUE: f32 = {edge_ab_avg_min_value};
const EDGE_AB_STD_DEV_THRESHOLD: f32 = {edge_ab_std_dev_threshold};
const EDGE_CLOSE_DELTA_THRESHOLD: f32 = {edge_close_delta_threshold};
const EDGE_FAR_DELTA_THRESHOLD: f32 = {edge_far_delta_threshold};
const EDGE_MAX_DELTA_THRESHOLD: f32 = {edge_max_delta_threshold};
const EDGE_AVG_DELTA_THRESHOLD: f32 = {edge_avg_delta_threshold};
const MAX_EDGE_COUNT: f32 = {max_edge_count};
const MIN_DEPTH: f32 = {min_depth};
const MAX_DEPTH: f32 = {max_depth};
"#,
        bfi = 0x180,
        ab_multiplier = params.ab_multiplier,
        ab0 = params.ab_multiplier_per_frq[0],
        ab1 = params.ab_multiplier_per_frq[1],
        ab2 = params.ab_multiplier_per_frq[2],
        ab_output_multiplier = params.ab_output_multiplier,
        ph0 = params.phase_in_rad[0],
        ph1 = params.phase_in_rad[1],
        ph2 = params.phase_in_rad[2],
        pi = std::f32::consts::PI,
        phase_offset = params.phase_offset,
        unambiguous_dist = params.unambiguous_dist,
        individual_ab_threshold = params.individual_ab_threshold,
        ab_threshold = params.ab_threshold,
        ab_confidence_slope = params.ab_confidence_slope,
        ab_confidence_offset = params.ab_confidence_offset,
        min_dealias_confidence = params.min_dealias_confidence,
        max_dealias_confidence = params.max_dealias_confidence,
        g0 = gaussian(0),
        g1 = gaussian(1),
        g2 = gaussian(2),
        g3 = gaussian(3),
        g4 = gaussian(4),
        g5 = gaussian(5),
        g6 = gaussian(6),
        g7 = gaussian(7),
        g8 = gaussian(8),
        joint_bilateral_exp = params.joint_bilateral_exp,
        joint_bilateral_threshold = joint_bilateral_threshold,
        joint_bilateral_max_edge = params.joint_bilateral_max_edge,
        use_bilateral = config.enable_bilateral_filter,
        edge_ab_avg_min_value = params.edge_ab_avg_min_value,
        edge_ab_std_dev_threshold = params.edge_ab_std_dev_threshold,
        edge_close_delta_threshold = params.edge_close_delta_threshold,
        edge_far_delta_threshold = params.edge_far_delta_threshold,
        edge_max_delta_threshold = params.edge_max_delta_threshold,
        edge_avg_delta_threshold = params.edge_avg_delta_threshold,
        // The kernel compares this against a float accumulator, so it has to
        // arrive as one.
        max_edge_count = params.max_edge_count,
        min_depth = config.min_depth * 1000.0,
        max_depth = config.max_depth * 1000.0,
    )
}

pub struct WgpuDepthProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_group: wgpu::BindGroup,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    buffers: Buffers,
    config: Config,
    params: DepthProcessorParams,
    /// In dispatch order; built in `set_config` because the filters are
    /// compile-time in the original.
    passes: Vec<wgpu::ComputePipeline>,
}

impl WgpuDepthProcessor {
    /// Pick a GPU adapter and build the processor.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| format!("no GPU adapter: {error}"))?;

        let info = adapter.get_info();
        eprintln!("[scan] GPU device: {} ({:?})", info.name, info.backend);

        // The decode needs 15 storage buffers in one stage. wgpu's default limit
        // is 8, and asking for more fails as a validation error rather than as
        // wrong output. Request whatever the adapter actually offers.
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < BINDINGS {
            return Err(format!(
                "this GPU allows {} storage buffers per stage, the decoder needs {BINDINGS}",
                limits.max_storage_buffers_per_shader_stage
            )
            .into());
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu-depth"),
            required_limits: limits,
            ..Default::default()
        }))
            .map_err(|error| format!("creating the GPU device: {error}"))?;

        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let vertex3 = DEPTH_SIZE as u64 * P0_STRIDE as u64 * 4;
        let scalar = DEPTH_SIZE as u64 * 4;

        let buffers = Buffers {
            lut: allocated(&device, "lut", LUT_SIZE as u64 * 4, storage),
            z_table: allocated(&device, "z_table", scalar, storage),
            x_table: allocated(&device, "x_table", scalar, storage),
            p0_table: allocated(&device, "p0_table", vertex3, storage),
            // 10 sub-images of 424 rows of 352 16-bit samples, with slack for the
            // one-word lookahead the unpack does.
            packet: allocated(&device, "packet", (424 * 10 * 352 + 16) as u64 * 4, storage),
            a: allocated(&device, "a", DEPTH_SIZE as u64 * 16, storage),
            b: allocated(&device, "b", DEPTH_SIZE as u64 * 16, storage),
            n: allocated(&device, "n", DEPTH_SIZE as u64 * 16, storage),
            ir: allocated(&device, "ir", scalar, storage),
            a_filtered: allocated(&device, "a_filtered", DEPTH_SIZE as u64 * 16, storage),
            b_filtered: allocated(&device, "b_filtered", DEPTH_SIZE as u64 * 16, storage),
            edge_test: allocated(&device, "edge_test", scalar, storage),
            depth: allocated(&device, "depth", scalar, storage),
            ir_sum: allocated(&device, "ir_sum", scalar, storage),
            filtered: allocated(&device, "filtered", scalar, storage),
            depth_staging: allocated(
                &device,
                "depth_staging",
                scalar,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            ir_staging: allocated(
                &device,
                "ir_staging",
                scalar,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        };

        // One explicit layout for every pipeline. Auto layouts are per entry
        // point, so stage 1 would get a layout containing only its own bindings
        // and a single shared bind group could not satisfy all four.
        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..BINDINGS)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        // Bindings 0..4 are the read-only inputs.
                        read_only: binding < 5,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decode"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decode"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decode"),
            layout: &layout,
            entries: &[
                binding(0, &buffers.lut),
                binding(1, &buffers.z_table),
                binding(2, &buffers.p0_table),
                binding(3, &buffers.packet),
                binding(4, &buffers.x_table),
                binding(5, &buffers.a),
                binding(6, &buffers.b),
                binding(7, &buffers.n),
                binding(8, &buffers.ir),
                binding(9, &buffers.a_filtered),
                binding(10, &buffers.b_filtered),
                binding(11, &buffers.edge_test),
                binding(12, &buffers.depth),
                binding(13, &buffers.ir_sum),
                binding(14, &buffers.filtered),
            ],
        });

        let mut processor = Self {
            device,
            queue,
            bind_group,
            layout,
            pipeline_layout,
            buffers,
            config: Config::default(),
            params: DepthProcessorParams::default(),
            passes: Vec::new(),
        };
        processor.build_pipelines()?;

        Ok(processor)
    }

    /// Compile the shader and the pipelines for the current config and params.
    fn build_pipelines(&mut self) -> Result<(), Box<dyn Error>> {
        let source = include_str!("../shaders/depth_decode.wgsl")
            // Only the first occurrence: the marker is also named in the header
            // comment, and replacing that too would inject the constants into a
            // comment and produce an unparseable shader.
            .replacen("//@@CONSTANTS@@", &constants(&self.params, &self.config), 1);

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("depth_decode"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let mut passes = Vec::new();
        for entry_point in pass_list(&self.config) {
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&self.pipeline_layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
            passes.push(pipeline);
        }

        self.passes = passes;
        Ok(())
    }

    /// Read one f32 buffer back into `out`.
    fn read_back(
        &self,
        staging: &wgpu::Buffer,
        source: &wgpu::Buffer,
        out: &mut [f32],
    ) -> Result<(), Box<dyn Error>> {
        let bytes = (out.len() * 4) as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read"),
            });
        encoder.copy_buffer_to_buffer(source, 0, staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|error| format!("waiting for the GPU: {error}"))?;
        receiver
            .recv()
            .map_err(|error| format!("map callback dropped: {error}"))?
            .map_err(|error| format!("mapping the staging buffer: {error}"))?;

        {
            let mapped = slice.get_mapped_range();
            for (index, chunk) in mapped.chunks_exact(4).enumerate() {
                out[index] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        staging.unmap();

        Ok(())
    }
}

impl DepthProcessorTrait for WgpuDepthProcessor {
    fn set_config(&mut self, config: &Config) -> Result<(), Box<dyn Error>> {
        // Which filters run is compiled into the shader, exactly as in the
        // OpenCL build options, so the pipelines have to be rebuilt.
        self.config = config.clone();
        self.build_pipelines()
    }

    fn set_p0_tables(
        &mut self,
        p0_tables: &kinect_one::data::P0Tables,
    ) -> Result<(), Box<dyn Error>> {
        // Same scaling the OpenCL processor applies before uploading.
        let scale = 0.000031 * std::f32::consts::PI;

        let mut padded = vec![0.0f32; DEPTH_SIZE * P0_STRIDE];
        for index in 0..DEPTH_SIZE {
            padded[index * P0_STRIDE] = -(p0_tables.p0_table0[index] as f32) * scale;
            padded[index * P0_STRIDE + 1] = -(p0_tables.p0_table1[index] as f32) * scale;
            padded[index * P0_STRIDE + 2] = -(p0_tables.p0_table2[index] as f32) * scale;
        }

        self.queue
            .write_buffer(&self.buffers.p0_table, 0, &as_bytes(&padded));

        Ok(())
    }

    fn set_x_z_tables(
        &mut self,
        x_table: &[f32; DEPTH_SIZE],
        z_table: &[f32; DEPTH_SIZE],
    ) -> Result<(), Box<dyn Error>> {
        self.queue
            .write_buffer(&self.buffers.x_table, 0, &as_bytes(x_table));
        self.queue
            .write_buffer(&self.buffers.z_table, 0, &as_bytes(z_table));

        Ok(())
    }

    fn set_lookup_table(&mut self, lut: &[i16; LUT_SIZE]) -> Result<(), Box<dyn Error>> {
        // Widened to i32: WGSL has no 16-bit integers. OpenCL promotes these to
        // int for the shifts the unpack performs anyway, so this is exact.
        let widened: Vec<i32> = lut.iter().map(|value| *value as i32).collect();

        self.queue
            .write_buffer(&self.buffers.lut, 0, &as_bytes(&widened));

        Ok(())
    }
}

impl ProcessorTrait<DepthPacket, (IrFrame, DepthFrame)> for WgpuDepthProcessor {
    async fn process(&self, input: DepthPacket) -> Result<(IrFrame, DepthFrame), Box<dyn Error>> {
        // The kernel reads the packet as 16-bit samples; widen to u32 for the
        // same reason as the lookup table above.
        let words: Vec<u32> = input
            .buffer
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as u32)
            .collect();
        self.queue
            .write_buffer(&self.buffers.packet, 0, &as_bytes(&words));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decode"),
            });
        for pipeline in &self.passes {
            // A pass boundary is an implicit barrier, so no stage reads a buffer
            // another is still writing.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((DEPTH_SIZE as u32).div_ceil(WORKGROUP), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // The edge-aware filter writes its own buffer; without it the raw depth
        // is the decoder's output. Mirrors the OpenCL processor's choice.
        let source = if self.config.enable_edge_aware_filter {
            &self.buffers.filtered
        } else {
            &self.buffers.depth
        };

        let mut depth = vec![0.0f32; DEPTH_SIZE];
        let mut ir = vec![0.0f32; DEPTH_SIZE];
        self.read_back(&self.buffers.depth_staging, source, &mut depth)?;
        self.read_back(&self.buffers.ir_staging, &self.buffers.ir, &mut ir)?;

        Ok((
            IrFrame::from_packet(ir, &input),
            DepthFrame::from_packet(depth, &input),
        ))
    }
}

fn allocated(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Reinterpret a plain-data slice as bytes for upload.
fn as_bytes<T: bytemuck_free::Pod>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

/// Marker trait for the plain-data types uploaded here (`f32`, `i32`, `u32`).
///
/// Declared locally rather than pulling in `bytemuck` for three primitive types.
mod bytemuck_free {
    /// # Safety
    /// Implementors must be plain data: no padding, no pointers, every bit
    /// pattern valid.
    pub unsafe trait Pod: Copy {}

    unsafe impl Pod for f32 {}
    unsafe impl Pod for i32 {}
    unsafe impl Pod for u32 {}
}

#[cfg(all(test, feature = "wgpu-decode"))]
mod tests {
    use super::*;
    use kinect_one::data::{IrParams, P0Tables};
    use kinect_one::processor::depth::CpuDepthProcessor;
    use kinect_one::processor::ProcessTrait;

    /// Plausible sensor intrinsics. Exact values do not matter for comparing two
    /// decoders, only that both receive the same ones.
    fn ir_params() -> IrParams {
        IrParams {
            fx: 367.13,
            fy: 367.13,
            cx: 261.08,
            cy: 211.21,
            k1: 0.093,
            k2: -0.028,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// The decoder's own size for a raw packet: ten sub-images of 298496 bytes.
    fn packet_bytes() -> Vec<u8> {
        let mut state: u64 = 0x2545F4914F6CDD1D;
        (0..10 * 298_496)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // A 16-bit spread rather than a byte spread: the unpack reads
                // 16-bit samples, so single-byte noise would exercise less.
                (state >> 33) as u8
            })
            .collect()
    }

    /// The gate: the same packet through both decoders must give the same depth.
    ///
    /// This is deliberately an offline test with a synthetic packet. Comparing two
    /// live captures would confound decoder differences with sensor noise and with
    /// the scene changing between runs; feeding one packet to both removes all of
    /// that. It is the same input-in, same output-out check that caught the OpenCL
    /// path silently producing zeros.
    ///
    /// It is not a substitute for a real capture -- a synthetic packet exercises
    /// the arithmetic but not the values a real sensor produces -- but it is the
    /// strongest check available without one.
    fn filters(bilateral: bool, edge: bool) -> Config {
        Config {
            min_depth: 0.5,
            max_depth: 4.5,
            enable_bilateral_filter: bilateral,
            enable_edge_aware_filter: edge,
        }
    }

    /// Stage 1, compared directly.
    ///
    /// The IR frame is stage 1's output and both decoders return it, so this
    /// compares the trickiest kernel with nothing downstream able to mask a
    /// fault: the packed-11-bit unpack, the sincos table lookups, the amplitude
    /// sums and the saturation handling. This is the part of the port that is
    /// verified rather than merely exercised.
    #[test]
    fn stage_one_agrees_with_the_cpu_decoder() {
        let run = decode_both(filters(false, false), &packet_bytes());

        let mut both = 0usize;
        let mut close = 0usize;
        let mut worst = 0.0f32;

        for (a, b) in run.cpu_ir.iter().zip(&run.gpu_ir) {
            if *a > 0.0 && *b > 0.0 {
                both += 1;
                let diff = (a - b).abs();
                worst = worst.max(diff);
                if diff <= 1.0 {
                    close += 1;
                }
            }
        }

        assert!(both > 200_000, "only {both} pixels carried any IR");
        assert!(
            close as f64 / both as f64 > 0.999,
            "stage 1 disagrees on {:.2}% of pixels (worst {worst:.2} of 65535)",
            100.0 * (1.0 - close as f64 / both as f64)
        );
    }

    /// Stage 1 and stage 2 with both filters off: the only configuration where
    /// the CPU and the OpenCL kernel are the *same* algorithm, and therefore the
    /// only place a close comparison means anything.
    #[test]
    fn the_unfiltered_decode_agrees_with_the_cpu_decoder() {
        let run = decode_both(filters(false, false), &packet_bytes());

        let mut both = 0usize;
        let mut cpu_only = 0usize;
        let mut gpu_only = 0usize;
        let mut sum = 0.0f64;
        let mut sum_cpu = 0.0f64;

        for (a, b) in run.cpu_depth.iter().zip(&run.gpu_depth) {
            match (*a > 0.0, *b > 0.0) {
                (true, true) => {
                    both += 1;
                    sum += (a - b).abs() as f64;
                    sum_cpu += *a as f64;
                }
                (true, false) => cpu_only += 1,
                (false, true) => gpu_only += 1,
                (false, false) => {}
            }
        }

        let total = both + cpu_only + gpu_only;
        let agreement = both as f64 / total as f64;
        assert!(
            agreement > 0.95,
            "the decoders agree on only {:.1}% of valid pixels",
            100.0 * agreement
        );

        // Relative rather than absolute, because the residual scales with depth:
        // this is precision in the phase-to-depth chain, not an offset. At the
        // 0.09% measured, a 4 m measurement is out by about 4 mm, which is below
        // the sensor's own 5.2 mm RMS planar noise.
        let relative = sum / sum_cpu;
        assert!(
            relative < 0.005,
            "mean difference is {:.3}% of the depth, too large to be \
             single-precision noise at the end of the chain",
            100.0 * relative
        );
    }

    /// The edge-aware filter differs between the two upstream implementations,
    /// and this records that rather than failing on it.
    ///
    /// `filterPixelStage2` in the OpenCL kernel zeroes a pixel whenever the
    /// bilateral edge test failed. The CPU processor's equivalent zeroes only
    /// when `cond0` holds. They are different algorithms, not two renderings of
    /// one: on smooth real surfaces most 3x3 neighbourhoods pass the edge test
    /// and the two behave similarly, but on the synthetic noise used here almost
    /// none pass, so the OpenCL path -- and this faithful port of it -- rejects
    /// nearly everything. The bilateral filter differs too: the CPU skips the
    /// centre tap and applies `gaussian[0..7]` to the eight neighbours, while the
    /// kernel taps all nine with `gaussian[4]` on the centre.
    ///
    /// How much this matters in practice is measured by `live_decoders_agree`:
    /// on a real packet the two agree on 96.7% of valid pixels with a 2.0 mm mean
    /// difference, because a real scene is smooth and most neighbourhoods pass
    /// the edge test. The divergence below is what noise exposes, not what a
    /// normal scene sees.
    ///
    /// The consequence for this port is that the CPU decoder is not a valid
    /// reference for the filtered path. Verifying it needs libfreenect2's own
    /// OpenCL output, and that cannot be produced on this machine: Rusticl
    /// enumerates a device and then never executes a kernel -- see
    /// examples/ocl_check.rs.
    #[test]
    fn the_edge_aware_filter_is_a_different_algorithm_upstream() {
        let run = decode_both(filters(true, true), &packet_bytes());

        let cpu_valid = run.cpu_depth.iter().filter(|v| **v > 0.0).count();
        let gpu_valid = run.gpu_depth.iter().filter(|v| **v > 0.0).count();

        assert!(
            cpu_valid > 1000,
            "the CPU decoder produced almost nothing ({cpu_valid}), so this is \
             not exercising anything"
        );
        assert!(
            gpu_valid < cpu_valid / 10,
            "the relationship changed ({cpu_valid} cpu vs {gpu_valid} gpu). \
             That could be a fix or a new fault -- read both filter \
             implementations before assuming either."
        );
    }

    /// One decode, with every buffer both decoders expose.
    struct Run {
        cpu_ir: Vec<f32>,
        gpu_ir: Vec<f32>,
        cpu_depth: Vec<f32>,
        gpu_depth: Vec<f32>,
    }

    fn decode_both(config: Config, bytes: &[u8]) -> Run {
        let ir = ir_params();
        let p0 = P0Tables {
            p0_table0: Box::new([0u16; DEPTH_SIZE]),
            p0_table1: Box::new([0u16; DEPTH_SIZE]),
            p0_table2: Box::new([0u16; DEPTH_SIZE]),
        };

        let mut cpu = CpuDepthProcessor::new().expect("CPU processor");
        cpu.set_config(&config).expect("cpu config");
        cpu.set_p0_tables(&p0).expect("cpu p0");
        cpu.set_ir_params(&ir).expect("cpu ir");

        let mut gpu = WgpuDepthProcessor::new().expect("Vulkan processor");
        gpu.set_config(&config).expect("gpu config");
        gpu.set_p0_tables(&p0).expect("gpu p0");
        gpu.set_ir_params(&ir).expect("gpu ir");

        let runtime = tokio::runtime::Runtime::new().expect("runtime");
        let packet = |buffer: Vec<u8>| DepthPacket {
            sequence: 0,
            timestamp: 0,
            buffer,
        };

        let (cpu_ir, cpu_depth) = runtime
            .block_on(packet(bytes.to_vec()).process(&cpu))
            .expect("cpu decode");
        let (gpu_ir, gpu_depth) = runtime
            .block_on(packet(bytes.to_vec()).process(&gpu))
            .expect("gpu decode");

        Run {
            cpu_ir: cpu_ir.buffer,
            gpu_ir: gpu_ir.buffer,
            cpu_depth: cpu_depth.buffer,
            gpu_depth: gpu_depth.buffer,
        }
    }

    fn compare(label: &str, cpu: &[f32], gpu: &[f32], tolerance: f32) {
        let mut both = 0usize;
        let mut cpu_only = 0usize;
        let mut gpu_only = 0usize;
        let mut close = 0usize;
        let mut sum = 0.0f64;
        let mut sum_cpu = 0.0f64;
        let mut worst = 0.0f32;

        for (a, b) in cpu.iter().zip(gpu) {
            match (*a > 0.0, *b > 0.0) {
                (true, true) => {
                    both += 1;
                    let diff = (a - b).abs();
                    sum += diff as f64;
                    sum_cpu += *a as f64;
                    worst = worst.max(diff);
                    if diff <= tolerance {
                        close += 1;
                    }
                }
                (true, false) => cpu_only += 1,
                (false, true) => gpu_only += 1,
                (false, false) => {}
            }
        }

        let mean = if both > 0 {
            sum / both as f64
        } else {
            f64::NAN
        };
        let mean_cpu = if both > 0 {
            sum_cpu / both as f64
        } else {
            f64::NAN
        };
        // A bias is the tell for a scale error; scatter alone would be noise.
        let bias = if mean_cpu > 0.0 {
            100.0 * mean / mean_cpu
        } else {
            f64::NAN
        };

        println!(
            "  {label:<7} cpu-only {cpu_only:>5}  gpu-only {gpu_only:>5}  both {both:>6}  \
             within {tolerance} {close:>6}  mean {mean:>9.4}  worst {worst:>9.3}  \
             mean is {bias:.3}% of cpu"
        );
    }

    /// Localising aid, asserts nothing.
    ///
    /// The IR frame is stage 1's output and both decoders return it, so it splits
    /// the pipeline in half: if the two IR frames agree then stage 1 is correct
    /// and the fault is in one of the three stages after it. Without that split
    /// the only observable is the final depth, which every stage can ruin.
    #[test]
    #[ignore = "diagnostic: prints stage-by-stage agreement, asserts nothing"]
    fn diagnose_the_divergence() {
        let bytes = packet_bytes();

        for (label, config) in [
            (
                "filters off",
                Config {
                    min_depth: 0.5,
                    max_depth: 4.5,
                    enable_bilateral_filter: false,
                    enable_edge_aware_filter: false,
                },
            ),
            (
                "bilateral only",
                Config {
                    min_depth: 0.5,
                    max_depth: 4.5,
                    enable_bilateral_filter: true,
                    enable_edge_aware_filter: false,
                },
            ),
            (
                "edge only",
                Config {
                    min_depth: 0.5,
                    max_depth: 4.5,
                    enable_bilateral_filter: false,
                    enable_edge_aware_filter: true,
                },
            ),
            (
                "filters on",
                Config {
                    min_depth: 0.5,
                    max_depth: 4.5,
                    enable_bilateral_filter: true,
                    enable_edge_aware_filter: true,
                },
            ),
        ] {
            let run = decode_both(config, &bytes);
            println!("{label}:");
            // Stage 1's output. Loose tolerance: these are 0..65535 counts.
            compare("ir", &run.cpu_ir, &run.gpu_ir, 1.0);
            // Final depth, in millimetres.
            compare("depth", &run.cpu_depth, &run.gpu_depth, 1.0);
        }
    }

    /// The check that actually settles it: one packet straight off the sensor,
    /// decoded by both.
    ///
    /// The synthetic tests above can only exercise the arithmetic. A real packet
    /// carries the values the sensor really produces -- a smooth scene, valid
    /// measurements, no saturation storms -- and that is what decides whether the
    /// port is usable. Feeding *one* packet to both decoders removes the two
    /// things that would otherwise confound the comparison: sensor noise between
    /// captures, and the scene changing in between.
    ///
    /// Needs hardware, so it is ignored by default:
    ///
    ///     cargo test --release -p scan --features wgpu-decode -- --ignored \
    ///         live_decoders_agree --nocapture
    #[test]
    #[ignore = "needs a Kinect v2 attached"]
    fn live_decoders_agree() {
        use kinect_one::DeviceEnumerator;

        let runtime = tokio::runtime::Runtime::new().expect("runtime");

        let (packet, p0, ir) = runtime.block_on(async {
            let mut device = DeviceEnumerator::open_default(true)
                .await
                .expect("opening the Kinect v2");
            device.start().await.expect("starting the streams");

            // Calibration is only populated *by* start(); reading it earlier
            // silently yields all-zero intrinsics.
            let ir = *device.get_ir_params();
            let p0 = device.get_p0_tables().clone();

            // The first packets arrive before the streams have settled.
            for _ in 0..500 {
                if let Some(packet) = device.poll_depth_packet().await.expect("polling") {
                    return (packet, p0, ir);
                }
            }
            panic!("no depth packet arrived");
        });

        let clone = |packet: &DepthPacket| DepthPacket {
            sequence: packet.sequence,
            timestamp: packet.timestamp,
            buffer: packet.buffer.clone(),
        };

        let mut unfiltered = None;

        for (label, config) in [
            ("filters off", filters(false, false)),
            ("filters on", filters(true, true)),
        ] {
            let mut cpu = CpuDepthProcessor::new().expect("CPU processor");
            cpu.set_config(&config).expect("cpu config");
            cpu.set_p0_tables(&p0).expect("cpu p0");
            cpu.set_ir_params(&ir).expect("cpu ir");

            let mut gpu = WgpuDepthProcessor::new().expect("Vulkan processor");
            gpu.set_config(&config).expect("gpu config");
            gpu.set_p0_tables(&p0).expect("gpu p0");
            gpu.set_ir_params(&ir).expect("gpu ir");

            let (cpu_ir, cpu_depth) = runtime
                .block_on(clone(&packet).process(&cpu))
                .expect("cpu decode");
            let (gpu_ir, gpu_depth) = runtime
                .block_on(clone(&packet).process(&gpu))
                .expect("gpu decode");

            println!("{label}:");
            compare("ir", &cpu_ir.buffer, &gpu_ir.buffer, 1.0);
            compare("depth", &cpu_depth.buffer, &gpu_depth.buffer, 1.0);

            if unfiltered.is_none() {
                unfiltered = Some((cpu_depth.buffer, gpu_depth.buffer));
            }
        }

        // The unfiltered path is the one that is verified, so it is the one worth
        // asserting on. The filters are left to the printed output: the CPU and
        // the kernel are different algorithms there, so a threshold would be
        // asserting my reading of both rather than the port.
        let (cpu_depth, gpu_depth) = unfiltered.expect("unfiltered run");
        let mut both = 0usize;
        let mut total = 0usize;

        for (a, b) in cpu_depth.iter().zip(&gpu_depth) {
            if *a > 0.0 || *b > 0.0 {
                total += 1;
                if *a > 0.0 && *b > 0.0 {
                    both += 1;
                }
            }
        }

        assert!(
            total > 1000,
            "only {total} pixels carried depth, so the scene was not visible"
        );
        let agreement = both as f64 / total as f64;
        assert!(
            agreement > 0.9,
            "on a real packet the decoders agree on only {:.1}% of valid pixels",
            100.0 * agreement
        );
    }
}
