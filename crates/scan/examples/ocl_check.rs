//! Minimal OpenCL health check: does this device actually execute compute?
//!
//! Written because the vendored OpenCL depth decoder returns all zeros on this
//! machine, and the cause is ambiguous. Either the port's plumbing never uploads
//! its tables, or Rusticl on this Gen8 GPU cannot run kernels reliably. This
//! isolates the two by running kernels whose expected output is known exactly.
//!
//! Run with:
//!     RUSTICL_ENABLE=iris cargo run --release -p scan --features gpu-decode \
//!         --example ocl_check
//!
//! `RUSTICL_ENABLE=iris` is required: Mesa ships Rusticl but exposes no devices
//! until a driver is named, so without it you get a platform with zero devices.

#[cfg(not(feature = "gpu-decode"))]
fn main() {
    eprintln!("build with `--features gpu-decode` to run this check");
}

#[cfg(feature = "gpu-decode")]
fn main() {
    use ocl::{Buffer, Device, DeviceType, Platform, ProQue};

    let platforms = Platform::list();
    println!("platforms: {}", platforms.len());
    if platforms.is_empty() {
        eprintln!("no OpenCL platform; is an ICD installed?");
        return;
    }

    let mut chosen = None;
    for platform in &platforms {
        println!(
            "platform: {} / {}",
            platform.name().unwrap_or_default(),
            platform.vendor().unwrap_or_default()
        );

        let gpus = Device::list(platform, Some(DeviceType::GPU)).unwrap_or_default();
        let all = Device::list(platform, None).unwrap_or_default();
        for device in &all {
            println!("  device: {}", device.name().unwrap_or_default());
        }

        if chosen.is_none() {
            chosen = gpus.into_iter().next().or_else(|| all.into_iter().next());
        }
    }

    let Some(device) = chosen else {
        eprintln!("no devices on any platform");
        return;
    };
    println!("using: {}\n", device.name().unwrap_or_default());

    const N: usize = 4096;

    // --- test 1: the simplest possible kernel ---------------------------------
    // If this fails, nothing about the port matters: the device cannot run
    // compute at all.
    let pro_que = ProQue::builder()
        .device(device.clone())
        .dims(N)
        .src(
            r#"
            __kernel void fill(__global float *out) {
                const uint i = get_global_id(0);
                out[i] = (float)i * 2.0f + 1.0f;
            }
            "#,
        )
        .build()
        .expect("building the trivial program");

    let out_buffer = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(N)
        .build()
        .expect("allocating the output buffer");

    let fill = pro_que
        .kernel_builder("fill")
        .arg(&out_buffer)
        .build()
        .expect("building the fill kernel");

    unsafe {
        fill.enq().expect("enqueueing the fill kernel");
    }

    let mut out = vec![0.0f32; N];
    out_buffer.read(&mut out).enq().expect("reading back");

    let nonzero = out.iter().filter(|v| **v != 0.0).count();
    let correct = (0..N).all(|i| (out[i] - (i as f32 * 2.0 + 1.0)).abs() < 1e-3);
    println!("test 1  fill          : {nonzero}/{N} non-zero, values correct: {correct}");
    println!("        first 4       : {:?}", &out[0..4]);

    // --- transfer check -------------------------------------------------------
    // Separates "kernels do not execute" from "the whole stack is inert". Those
    // are different bugs with different owners, so it is worth knowing which.
    let host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let probe = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(N)
        .copy_host_slice(&host)
        .build()
        .expect("allocating the probe buffer");
    let mut back = vec![0.0f32; N];
    probe.read(&mut back).enq().expect("reading the probe back");
    println!(
        "        transfer      : host->device->host {}",
        if back == host { "OK" } else { "FAILED" }
    );

    // --- test 2: the operations stage 1 actually uses -------------------------
    // short/ushort buffers, integer shifts, and a table indexed by a computed
    // value. This is a close proxy for `processPixelStage1`, minus the float3s.
    let program = ProQue::builder()
        .device(device.clone())
        .dims(N)
        .src(
            r#"
            __kernel void lookup(__global const short *lut,
                                 __global const ushort *data,
                                 __global float *out) {
                const uint i = get_global_id(0);
                const ushort raw = data[i];
                const uint combined = ((raw >> 4) | (raw << 2)) & 2047u;
                out[i] = (float)lut[combined] + (float)combined;
            }
            "#,
        )
        .build()
        .expect("building the lookup program");

    // A table and input that make a wrong answer obvious rather than plausible.
    let lut: Vec<i16> = (0..2048).map(|i| i as i16 - 1024).collect();
    let data: Vec<u16> = (0..N).map(|i| (i as u16).wrapping_mul(7) + 3).collect();

    let lut_buffer = Buffer::<i16>::builder()
        .queue(program.queue().clone())
        .len(lut.len())
        .copy_host_slice(&lut)
        .build()
        .expect("allocating the lut");
    let data_buffer = Buffer::<u16>::builder()
        .queue(program.queue().clone())
        .len(N)
        .copy_host_slice(&data)
        .build()
        .expect("allocating the data");
    let result_buffer = Buffer::<f32>::builder()
        .queue(program.queue().clone())
        .len(N)
        .build()
        .expect("allocating the result");

    let lookup = program
        .kernel_builder("lookup")
        .arg(&lut_buffer)
        .arg(&data_buffer)
        .arg(&result_buffer)
        .build()
        .expect("building the lookup kernel");

    unsafe {
        lookup.enq().expect("enqueueing the lookup kernel");
    }

    let mut result = vec![0.0f32; N];
    result_buffer.read(&mut result).enq().expect("reading back");

    // Mirror OpenCL's integer promotion exactly: `ushort << 2` widens to int, so
    // it does not truncate. Shifting a u16 in Rust would.
    let expect = |raw: u16| -> f32 {
        let combined = (((raw as u32) >> 4) | ((raw as u32) << 2)) & 2047;
        lut[combined as usize] as f32 + combined as f32
    };

    let wrong = (0..N)
        .filter(|i| (result[*i] - expect(data[*i])).abs() > 1e-3)
        .count();
    let zeros = result.iter().filter(|v| **v == 0.0).count();
    println!("test 2  table+bitops  : {wrong}/{N} wrong, {zeros}/{N} zero");
    println!("        result[0..4]  : {:?}", &result[0..4]);
    println!(
        "        expected      : {:?}",
        (0..4).map(|i| expect(data[i])).collect::<Vec<_>>()
    );

    println!();
    if nonzero == 0 {
        println!("VERDICT: the device does not execute kernels at all (test 1 dead).");
    } else if wrong > 0 {
        println!("VERDICT: kernels run but miscompute -- suspect the device or the kernel.");
    } else {
        println!("VERDICT: the device runs kernels correctly. The decoder's zero output");
        println!("         is the port's plumbing, not the GPU. Check the table uploads.");
    }
}
