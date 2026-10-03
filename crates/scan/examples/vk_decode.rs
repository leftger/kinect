//! Vulkan compute port of the depth decoder — stage 1, and does it run?
//!
//! `vk_check` proved this GPU executes compute under Vulkan (ANV) even though
//! Rusticl silently refuses to. This is the first piece of the actual port: the
//! trickiest of the four kernels, `processPixelStage1`, translated from the
//! OpenCL C in `opencl_depth_packet_processor.cl` to WGSL.
//!
//! Stage 1 is done first deliberately. It is where the translation risk lives --
//! bit manipulation on packed 11-bit phase data, `sincos` over `float3`s, and
//! `select`/`isless`/`any` on vector masks. The other three kernels are mostly
//! scalar arithmetic and neighbourhood reads, and are much less likely to hit a
//! WGSL surprise.
//!
//! # Translation notes
//!
//! * **16-bit integers are widened.** OpenCL's `short`/`ushort` have no WGSL
//!   equivalent, so `lut11to16` and the packet are uploaded as `i32`/`u32`. This
//!   is not a lossy shortcut: OpenCL promotes them to `int` for the shifts and
//!   masks this kernel performs anyway, so the arithmetic is identical.
//! * **`float3` buffers are `array<vec4<f32>>`.** WGSL gives `vec3<f32>` an
//!   alignment of 16 and an array stride of 16, which would silently disagree
//!   with a tightly packed `[f32; 3n]` host buffer. Padding to `vec4` makes the
//!   layout unambiguous, at the cost of a quarter more memory.
//! * **`select(a, b, c)` maps directly.** Both languages mean "b where c, else
//!   a", so the argument order is unchanged.
//! * **`sincos(x, &out)` becomes `sin`/`cos`.** The OpenCL helper returns the
//!   sine and writes the cosine, and the caller negates the sine.
//! * **Constants are templated in.** The OpenCL build passes ~40 `-D` defines
//!   from `DepthProcessorParams`; here they are substituted into the source
//!   before compilation, which is the same idea with less tooling.
//!
//! Run with:
//!     cargo run --release -p scan --example vk_decode

use std::sync::mpsc;

/// Depth image dimensions, matching `kinect_one::{DEPTH_WIDTH, DEPTH_HEIGHT}`.
const WIDTH: u32 = 512;
const HEIGHT: u32 = 424;
const PIXELS: u32 = WIDTH * HEIGHT;
const WORKGROUP: u32 = 64;

/// Shader source with the compile-time constants filled in.
///
/// The values are stand-ins for `DepthProcessorParams`; what is being tested here
/// is that the *translated shader compiles and produces sane output*, not yet that
/// it matches the CPU decoder numerically. Wiring the real parameters in is a
/// one-line change once the harness exists.
fn shader_source() -> String {
    const BFI_BITMASK: u32 = 0x180;
    const AB_MULTIPLIER: f32 = 1.0;
    const AB_MULTIPLIER_PER_FRQ: [f32; 3] = [0.5, 0.5, 0.5];
    const AB_OUTPUT_MULTIPLIER: f32 = 1.0;
    const PHASE_IN_RAD: [f32; 3] = [0.0, 0.0, 0.0];
    const PI: f32 = std::f32::consts::PI;

    format!(
        r#"
const BFI_BITMASK: u32 = {BFI_BITMASK}u;
const AB_MULTIPLIER: f32 = {AB_MULTIPLIER};
const AB_OUTPUT_MULTIPLIER: f32 = {AB_OUTPUT_MULTIPLIER};
const PHASE: vec3<f32> = vec3<f32>({p0}, {p1}, {p2});
const AB_PER_FRQ: vec3<f32> = vec3<f32>({a0}, {a1}, {a2});

@group(0) @binding(0) var<storage, read> lut11to16: array<i32>;
@group(0) @binding(1) var<storage, read> z_table: array<f32>;
@group(0) @binding(2) var<storage, read> p0_table: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> packet: array<u32>;
@group(0) @binding(4) var<storage, read_write> a_out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> b_out: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> n_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> ir_out: array<f32>;

fn decode_pixel_measurement(sub: u32, x: u32, y: u32) -> f32 {{
    let row_idx = (424u * sub + y) * 352u;
    let idx = (((x >> 2u) + ((x << 7u) & BFI_BITMASK)) * 11u);
    let col_idx = idx >> 4u;
    let upper_bytes = idx & 15u;
    let lower_bytes = 16u - upper_bytes;
    let data_idx0 = row_idx + col_idx;
    let data_idx1 = row_idx + col_idx + 1u;

    // The OpenCL original reads lut11to16[0] for these pixels rather than
    // skipping them, which is why they end up invalid later.
    if (x < 1u || 510u < x || col_idx > 352u) {{
        return f32(lut11to16[0]);
    }}

    let packed = ((packet[data_idx0] >> upper_bytes) | (packet[data_idx1] << lower_bytes)) & 2047u;
    return f32(lut11to16[packed]);
}}

@compute @workgroup_size({WORKGROUP})
fn process_pixel_stage1(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= {PIXELS}u) {{
        return;
    }}

    let x = i % 512u;
    let y = i / 512u;

    let y_tmp = 423u - y;
    let y_in = select(423u - y_tmp, y_tmp + 212u, y_tmp < 212u);

    // `(int)(0.0f >= z_table[i])` -- a bool splatted into an int vector.
    let invalid_flag = select(0, 1, 0.0 >= z_table[i]);
    let invalid = vec3<i32>(invalid_flag);

    let p0 = p0_table[i].xyz;

    let p0x_sin = -sin(PHASE + p0.x);
    let p0x_cos = cos(PHASE + p0.x);
    let p0y_sin = -sin(PHASE + p0.y);
    let p0y_cos = cos(PHASE + p0.y);
    let p0z_sin = -sin(PHASE + p0.z);
    let p0z_cos = cos(PHASE + p0.z);

    let v0 = vec3<f32>(
        decode_pixel_measurement(0u, x, y_in),
        decode_pixel_measurement(1u, x, y_in),
        decode_pixel_measurement(2u, x, y_in),
    );
    let v1 = vec3<f32>(
        decode_pixel_measurement(3u, x, y_in),
        decode_pixel_measurement(4u, x, y_in),
        decode_pixel_measurement(5u, x, y_in),
    );
    let v2 = vec3<f32>(
        decode_pixel_measurement(6u, x, y_in),
        decode_pixel_measurement(7u, x, y_in),
        decode_pixel_measurement(8u, x, y_in),
    );

    var a = vec3<f32>(dot(v0, p0x_cos), dot(v1, p0y_cos), dot(v2, p0z_cos)) * AB_PER_FRQ;
    var b = vec3<f32>(dot(v0, p0x_sin), dot(v1, p0y_sin), dot(v2, p0z_sin)) * AB_PER_FRQ;

    a = select(a, vec3<f32>(0.0), invalid != vec3<i32>(0));
    b = select(b, vec3<f32>(0.0), invalid != vec3<i32>(0));
    let n = sqrt(a * a + b * b);

    // Saturated pixels are the sentinel 32767 in any component.
    let saturated = vec3<i32>(
        select(0, 1, any(v0 == vec3<f32>(32767.0))),
        select(0, 1, any(v1 == vec3<f32>(32767.0))),
        select(0, 1, any(v2 == vec3<f32>(32767.0))),
    );
    let saturated_mask = saturated != vec3<i32>(0);

    a_out[i] = vec4<f32>(select(a, vec3<f32>(0.0), saturated_mask), 0.0);
    b_out[i] = vec4<f32>(select(b, vec3<f32>(0.0), saturated_mask), 0.0);
    n_out[i] = vec4<f32>(n, 0.0);

    let ir = dot(
        select(n, vec3<f32>(65535.0), saturated_mask),
        vec3<f32>(0.333333333 * AB_MULTIPLIER * AB_OUTPUT_MULTIPLIER),
    );
    ir_out[i] = min(ir, 65535.0);
}}
"#,
        p0 = PHASE_IN_RAD[0],
        p1 = PHASE_IN_RAD[1],
        p2 = PHASE_IN_RAD[2],
        a0 = AB_MULTIPLIER_PER_FRQ[0],
        a1 = AB_MULTIPLIER_PER_FRQ[1],
        a2 = AB_MULTIPLIER_PER_FRQ[2],
    )
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

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vk_decode"),
        ..Default::default()
    }))
    .expect("device");

    // ---- inputs -------------------------------------------------------------
    // A deterministic, varied packet so a silent no-op cannot masquerade as
    // success. A fixed xorshift keeps this reproducible without a dependency.
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    // The packet is sized by the unpacked index arithmetic: 10 sub-images of
    // 424 rows of 352 ushorts, plus slack for the +1 lookahead.
    const PACKET_WORDS: usize = 424 * 10 * 352 + 16;
    let packet: Vec<u32> = (0..PACKET_WORDS).map(|_| (next() & 0xFFFF) as u32).collect();
    let lut: Vec<i32> = (0..2048).map(|i| (i as i32 % 2048) - 1024).collect();
    let z_table: Vec<f32> = vec![1.5; PIXELS as usize];
    // p0 padded to vec4: [x, y, z, _].
    let mut p0: Vec<f32> = Vec::with_capacity(PIXELS as usize * 4);
    for i in 0..PIXELS as usize {
        p0.push((i % 97) as f32 * 0.001);
        p0.push((i % 89) as f32 * 0.001);
        p0.push((i % 83) as f32 * 0.001);
        p0.push(0.0);
    }

    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;

    let lut_buf = make(&device, "lut", &bytemuck_i32(&lut), storage);
    let z_buf = make(&device, "z", &bytemuck_f32(&z_table), storage);
    let p0_buf = make(&device, "p0", &bytemuck_f32(&p0), storage);
    let packet_buf = make(&device, "packet", &bytemuck_u32(&packet), storage);
    let a_buf = blank(&device, "a", PIXELS as u64 * 16, storage);
    let b_buf = blank(&device, "b", PIXELS as u64 * 16, storage);
    let n_buf = blank(&device, "n", PIXELS as u64 * 16, storage);
    let ir_buf = blank(&device, "ir", PIXELS as u64 * 4, storage);

    // ---- pipeline -----------------------------------------------------------
    let source = shader_source();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("stage1"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stage1"),
        layout: None,
        module: &module,
        entry_point: Some("process_pixel_stage1"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stage1"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            entry(0, &lut_buf),
            entry(1, &z_buf),
            entry(2, &p0_buf),
            entry(3, &packet_buf),
            entry(4, &a_buf),
            entry(5, &b_buf),
            entry(6, &n_buf),
            entry(7, &ir_buf),
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("stage1") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(PIXELS.div_ceil(WORKGROUP), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    // ---- read back ----------------------------------------------------------
    let ir = read_f32(&device, &queue, &ir_buf, PIXELS as u64 * 4);
    let n = read_f32_padded(&device, &queue, &n_buf, PIXELS as u64 * 16);

    let ir_nonzero = ir.iter().filter(|v| **v != 0.0).count();
    let ir_finite = ir.iter().filter(|v| v.is_finite()).count();
    let n_nonzero = n.iter().filter(|v| **v != 0.0).count();
    let max_ir = ir.iter().cloned().fold(0.0f32, f32::max);

    println!("\nstage 1 over {PIXELS} pixels:");
    println!("  ir   : {ir_nonzero}/{PIXELS} non-zero, {ir_finite} finite, max {max_ir:.1}");
    println!("  norm  : {n_nonzero}/{PIXELS} non-zero");
    println!("  ir[0..6] = {:?}", &ir[0..6]);

    println!();
    if ir_finite < ir.len() || ir_nonzero == 0 {
        println!("VERDICT: the translated stage 1 compiles but produces degenerate output.");
    } else {
        println!("VERDICT: the translated stage 1 compiles under naga, dispatches on this");
        println!("         GPU, and produces varied finite output. The translation is");
        println!("         structurally sound; numerical agreement with the CPU decoder");
        println!("         still has to be checked against a real packet and real tables.");
    }
}

fn entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn make(device: &wgpu::Device, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
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

fn stage(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read_bytes(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let staging = stage(device, size);
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

/// Read a `vec4<f32>` buffer, keeping only `.x` of each element.
fn read_f32_padded(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
) -> Vec<f32> {
    read_bytes(device, queue, source, size)
        .chunks_exact(16)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// Small byte encoders, so the example needs no extra dependency.
fn bytemuck_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytemuck_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytemuck_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
