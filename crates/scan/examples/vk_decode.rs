//! Vulkan compute port of the depth decoder: does the whole chain run?
//!
//! `vk_check` established that this GPU executes compute under Vulkan, even
//! though Rusticl silently refuses to. `vk_decode` runs all four translated
//! kernels -- stage 1, both filters, stage 2 -- end to end on synthetic data, to
//! prove the translation compiles under naga, binds correctly, and produces a
//! depth image rather than a zero buffer.
//!
//! It is *not* yet a numerical check against the CPU decoder. The constants below
//! are chosen to let data flow through unmodified rather than to match
//! `DepthProcessorParams`, and the input is a pseudo-random packet. What this
//! catches is the class of failure that killed the OpenCL path: kernels that run
//! and write nothing. Agreement with the CPU decoder needs a real packet and the
//! real parameters, which is the next step.
//!
//! Run with:
//!     cargo run --release -p scan --example vk_decode

use std::sync::mpsc;

const WIDTH: u32 = 512;
const HEIGHT: u32 = 424;
const PIXELS: u32 = WIDTH * HEIGHT;
const WORKGROUP: u32 = 64;

/// The values the OpenCL build passed as `-D` defines, derived from
/// `DepthProcessorParams` and `Config`.
///
/// These stand-ins deliberately *pass data through*: the amplitude thresholds
/// are zero and the edge thresholds are enormous, so every pixel reaches the
/// final buffer instead of being filtered out. A degenerate-but-filtered result
/// would be indistinguishable from a broken kernel.
fn constants() -> String {
    format!(
        r#"
const BFI_BITMASK: u32 = {bfi}u;
const AB_MULTIPLIER: f32 = 0.5;
const AB_PER_FRQ: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const AB_OUTPUT_MULTIPLIER: f32 = 1.0;
const PHASE: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const M_PI_F: f32 = {pi};
const PHASE_OFFSET: f32 = 0.0;
const UNAMBIGUOUS_DIST: f32 = 2083.3333;
const INDIVIDUAL_AB_THRESHOLD: f32 = 0.0;
const AB_THRESHOLD: f32 = 0.0;
const AB_CONFIDENCE_SLOPE: f32 = 0.5;
const AB_CONFIDENCE_OFFSET: f32 = 0.0;
const MIN_DEALIAS_CONFIDENCE: f32 = 0.0;
const MAX_DEALIAS_CONFIDENCE: f32 = 1.0;
const GAUSSIAN_0: f32 = 0.05;
const GAUSSIAN_1: f32 = 0.1;
const GAUSSIAN_2: f32 = 0.05;
const GAUSSIAN_3: f32 = 0.1;
const GAUSSIAN_4: f32 = 0.4;
const GAUSSIAN_5: f32 = 0.1;
const GAUSSIAN_6: f32 = 0.05;
const GAUSSIAN_7: f32 = 0.1;
const GAUSSIAN_8: f32 = 0.05;
const JOINT_BILATERAL_EXP: f32 = 0.0;
const JOINT_BILATERAL_THRESHOLD: f32 = 1.0e9;
const JOINT_BILATERAL_MAX_EDGE: f32 = 1.0e9;
const USE_BILATERAL: bool = true;
const EDGE_AB_AVG_MIN_VALUE: f32 = 1.0;
const EDGE_AB_STD_DEV_THRESHOLD: f32 = 1.0e9;
const EDGE_CLOSE_DELTA_THRESHOLD: f32 = 1.0e9;
const EDGE_FAR_DELTA_THRESHOLD: f32 = 1.0e9;
const EDGE_MAX_DELTA_THRESHOLD: f32 = 1.0e9;
const EDGE_AVG_DELTA_THRESHOLD: f32 = 1.0e9;
const MAX_EDGE_COUNT: f32 = 0.0;
const MIN_DEPTH: f32 = 0.0;
const MAX_DEPTH: f32 = 100000.0;
"#,
        bfi = 0x180,
        pi = std::f32::consts::PI,
    )
}

fn shader_source() -> String {
    include_str!("../shaders/depth_decode.wgsl").replacen("//@@CONSTANTS@@", &constants(), 1)
}

fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        println!("no Vulkan adapter");
        return;
    };
    let info = adapter.get_info();
    println!("adapter : {} ({:?})", info.name, info.backend);

    // The full pipeline needs 15 storage buffers in one stage. wgpu's default
    // limit is 8, so the adapter's own limits have to be requested -- this is the
    // kind of thing that fails as a validation error rather than wrong numbers.
    let limits = adapter.limits();
    println!(
        "limits  : max_storage_buffers_per_shader_stage = {}",
        limits.max_storage_buffers_per_shader_stage
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vk_decode"),
        required_limits: limits,
        ..Default::default()
    }))
    .expect("device");

    // ---- inputs -------------------------------------------------------------
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let packet_words = 424 * 10 * 352 + 16;
    let packet: Vec<u32> = (0..packet_words).map(|_| (next() & 0xFFFF) as u32).collect();
    let lut: Vec<i32> = (0..2048).map(|i| (i as i32 % 2048) - 1024).collect();
    // z_table must be > 0 or stage 1 marks the pixel invalid, and its magnitude
    // matters: it is millimetres of depth per unit of unwrapped phase, so a
    // realistic value is in the hundreds to thousands. Using ~1.0 here yields a
    // depth image of a few millimetres, which looks plausible in a summary and
    // would hide a scale error.
    let z_table: Vec<f32> = (0..PIXELS).map(|i| 900.0 + (i % 7) as f32 * 20.0).collect();
    let x_table: Vec<f32> = (0..PIXELS).map(|i| 0.001 + (i % 5) as f32 * 0.0001).collect();
    let mut p0: Vec<f32> = Vec::with_capacity(PIXELS as usize * 4);
    for i in 0..PIXELS as usize {
        p0.push((i % 97) as f32 * 0.0001);
        p0.push((i % 89) as f32 * 0.0001);
        p0.push((i % 83) as f32 * 0.0001);
        p0.push(0.0);
    }

    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let lut_buf = from_bytes(&device, "lut", &bytes_i32(&lut), storage);
    let z_buf = from_bytes(&device, "z_table", &bytes_f32(&z_table), storage);
    let x_buf = from_bytes(&device, "x_table", &bytes_f32(&x_table), storage);
    let p0_buf = from_bytes(&device, "p0_table", &bytes_f32(&p0), storage);
    let packet_buf = from_bytes(&device, "packet", &bytes_u32(&packet), storage);
    let a_buf = blank(&device, "a", PIXELS as u64 * 16, storage);
    let b_buf = blank(&device, "b", PIXELS as u64 * 16, storage);
    let n_buf = blank(&device, "n", PIXELS as u64 * 16, storage);
    let ir_buf = blank(&device, "ir", PIXELS as u64 * 4, storage);
    let a_f_buf = blank(&device, "a_filtered", PIXELS as u64 * 16, storage);
    let b_f_buf = blank(&device, "b_filtered", PIXELS as u64 * 16, storage);
    let edge_buf = blank(&device, "edge_test", PIXELS as u64 * 4, storage);
    let depth_buf = blank(&device, "depth", PIXELS as u64 * 4, storage);
    let irsum_buf = blank(&device, "ir_sum", PIXELS as u64 * 4, storage);
    let filtered_buf = blank(&device, "filtered", PIXELS as u64 * 4, storage);

    // ---- one explicit layout shared by all four pipelines --------------------
    // Auto layouts are per-entry-point, so stage 1 would get a layout with only
    // the bindings it uses and a single shared bind group would not validate.
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..15)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
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
            bind(0, &lut_buf),
            bind(1, &z_buf),
            bind(2, &p0_buf),
            bind(3, &packet_buf),
            bind(4, &x_buf),
            bind(5, &a_buf),
            bind(6, &b_buf),
            bind(7, &n_buf),
            bind(8, &ir_buf),
            bind(9, &a_f_buf),
            bind(10, &b_f_buf),
            bind(11, &edge_buf),
            bind(12, &depth_buf),
            bind(13, &irsum_buf),
            bind(14, &filtered_buf),
        ],
    });

    // ---- four pipelines, four passes ----------------------------------------
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("depth_decode"),
        source: wgpu::ShaderSource::Wgsl(shader_source().into()),
    });

    let passes = [
        "process_pixel_stage1",
        "filter_pixel_stage1",
        "process_pixel_stage2",
        "filter_pixel_stage2",
    ];

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("decode") });

    for entry_point in passes {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        // One pass per kernel: a pass boundary is an implicit barrier, so no
        // stage can read a buffer another is still writing.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(PIXELS.div_ceil(WORKGROUP), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    // ---- results ------------------------------------------------------------
    let depth = read_f32(&device, &queue, &depth_buf, PIXELS as u64 * 4);
    let filtered = read_f32(&device, &queue, &filtered_buf, PIXELS as u64 * 4);
    let ir = read_f32(&device, &queue, &ir_buf, PIXELS as u64 * 4);
    let edges = read_u32(&device, &queue, &edge_buf, PIXELS as u64 * 4);

    let stats = |name: &str, values: &[f32]| {
        let finite = values.iter().filter(|v| v.is_finite()).count();
        let nonzero = values.iter().filter(|v| **v != 0.0).count();
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = values.iter().filter(|v| **v > 0.0).cloned().fold(f32::INFINITY, f32::min);
        println!(
            "  {name:<9}: {nonzero:>6}/{PIXELS} non-zero, {finite:>6} finite, range {min:.2}..{max:.2}"
        );
    };

    println!("\nall four kernels over {PIXELS} pixels:");
    stats("ir", &ir);
    stats("depth", &depth);
    stats("filtered", &filtered);
    println!(
        "  edge_test: {}/{} set",
        edges.iter().filter(|v| **v != 0).count(),
        PIXELS
    );

    let good = filtered.iter().filter(|v| v.is_finite() && **v > 0.0).count();
    println!();
    if good == 0 {
        println!("VERDICT: the chain runs but yields nothing usable. Either a stage is");
        println!("         writing zeros or the constants filter everything out.");
    } else {
        println!("VERDICT: all four translated kernels compile under naga, dispatch in");
        println!("         sequence, and produce {good} usable depth pixels. The port is");
        println!("         structurally sound end to end; numerical agreement with the CPU");
        println!("         decoder still needs a real packet and the real parameters.");
    }
}

fn bind<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn from_bytes(
    device: &wgpu::Device,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
    })
}

fn blank(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait).expect("poll");
    receiver.recv().expect("callback").expect("map");

    let mapped = slice.get_mapped_range();
    let out = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    out
}

fn read_f32(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer, size: u64) -> Vec<f32> {
    read_bytes(device, queue, source, size)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer, size: u64) -> Vec<u32> {
    read_bytes(device, queue, source, size)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
