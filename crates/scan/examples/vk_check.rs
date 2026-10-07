//! Does Vulkan compute actually execute on this GPU?
//!
//! The OpenCL equivalent (`ocl_check`) found that Rusticl on this Broadwell GPU
//! enumerates a device, allocates buffers, transfers data correctly -- and then
//! silently never runs a kernel. Vulkan is a completely different driver path
//! (`intel_icd.json`, the ANV driver) and its runtime is already installed, so it
//! is worth asking the same question before writing any porting code.
//!
//! Same shape of test, so the two verdicts are directly comparable:
//!   1. a trivial dispatch that writes `i * 2 + 1`, checked value by value
//!   2. a host -> device -> host round trip
//!
//! Run with:
//!     cargo run --release -p scan --example vk_check

use std::sync::mpsc;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&data)) {
        data[i] = f32(i) * 2.0 + 1.0;
    }
}
"#;

const N: usize = 4096;
const WORKGROUP: u32 = 64;

fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter =
        match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        {
            Ok(adapter) => adapter,
            Err(error) => {
                println!("no Vulkan adapter: {error}");
                println!("VERDICT: no usable adapter, so compute cannot run at all.");
                return;
            }
        };

    let info = adapter.get_info();
    println!(
        "adapter : {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    println!("driver  : {} {}", info.driver, info.driver_info);

    let (device, queue) =
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vk_check"),
            ..Default::default()
        })) {
            Ok(pair) => pair,
            Err(error) => {
                println!("device creation failed: {error}");
                println!("VERDICT: adapter present but no device; compute cannot run.");
                return;
            }
        };

    let bytes = (N * std::mem::size_of::<f32>()) as u64;

    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // --- test 1: a trivial dispatch ------------------------------------------
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fill"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fill"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fill"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fill"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((N as u32).div_ceil(WORKGROUP), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let out = read_back(&device, &queue, &storage, &staging, bytes);

    let nonzero = out.iter().filter(|value| **value != 0.0).count();
    let correct = (0..N).all(|i| (out[i] - (i as f32 * 2.0 + 1.0)).abs() < 1e-3);
    println!("\ntest 1  dispatch      : {nonzero}/{N} non-zero, values correct: {correct}");
    println!("        first 4       : {:?}", &out[0..4]);

    // --- host round trip ------------------------------------------------------
    // Separates "kernels do not execute" from "the whole stack is inert".
    let host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    queue.write_buffer(&storage, 0, as_bytes(&host));
    let back = read_back(&device, &queue, &storage, &staging, bytes);
    println!(
        "        transfer      : host->device->host {}",
        if back == host { "OK" } else { "FAILED" }
    );

    println!();
    if nonzero == 0 {
        println!("VERDICT: the device does not execute compute. Vulkan is no better");
        println!("         than OpenCL here, and a decode port would be wasted effort.");
    } else if !correct {
        println!("VERDICT: compute runs but miscomputes. Suspect the shader or the driver.");
    } else {
        println!("VERDICT: Vulkan compute works correctly on this GPU. Unlike Rusticl, a");
        println!("         decode port has a chance of paying off here.");
    }
}

/// Submit, copy to a mappable buffer, wait, and decode.
fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    bytes: u64,
) -> Vec<f32> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("copy"),
    });
    encoder.copy_buffer_to_buffer(source, 0, staging, 0, bytes);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::PollType::Wait).expect("poll");
    receiver.recv().expect("map callback").expect("map");

    let mapped = slice.get_mapped_range();
    let out = mapped
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    drop(mapped);
    staging.unmap();

    out
}

fn as_bytes(values: &[f32]) -> &[u8] {
    // Safety: `f32` is plain data with no padding or invalid bit patterns that
    // matter here, and the slice lifetime is tied to the input.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}
