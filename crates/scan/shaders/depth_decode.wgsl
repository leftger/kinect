// WGSL port of libfreenect2's OpenCL depth decode.
//
// Translated from
//   vendor/kinect-one/src/processor/depth/opencl/opencl_depth_packet_processor.cl
// which is Apache-2.0 / GPL-2.0 (see the vendored crate's LICENSE and
// LIBFREENECT2_CONTRIB). Four kernels run in sequence per depth packet:
//
//   processPixelStage1 -> filterPixelStage1 -> processPixelStage2 -> filterPixelStage2
//
// The placeholder line further down is replaced at load time with the values
// the OpenCL build passed as -D defines, derived from `DepthProcessorParams`
// and the decoder `Config`.
//
// Translation notes, all of which would otherwise be silent bugs:
//
//  * Index arithmetic is done in i32 and converted to u32 only at the array
//    access. The code subtracts 1 from coordinates and the guards in the OpenCL
//    original rely on wrap-free signed arithmetic.
//  * 16-bit integers are widened. WGSL has no i16/u16, so `lut11to16` and the
//    packet arrive as i32/u32. OpenCL promotes them to int for these shifts and
//    masks anyway, so the arithmetic is identical, not approximated.
//  * `float3` buffers are `array<vec4<f32>>`. WGSL gives vec3 an array stride of
//    16, which would quietly disagree with a packed [f32; 3n] host buffer.
//  * `select(a, b, c)` means the same in both languages: b where c, else a.
//  * `sincos(x, &o)` returns the sine and writes the cosine, so it becomes
//    separate `sin`/`cos` calls with the sine negated as the caller did.
//  * Integer `1`/`0` written to what OpenCL typed as `uchar` becomes `u32`.

//@@CONSTANTS@@

@group(0) @binding(0) var<storage, read> lut11to16: array<i32>;
@group(0) @binding(1) var<storage, read> z_table: array<f32>;
@group(0) @binding(2) var<storage, read> p0_table: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> packet: array<u32>;
@group(0) @binding(4) var<storage, read> x_table: array<f32>;
@group(0) @binding(5) var<storage, read_write> a_buffer: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> b_buffer: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> n_buffer: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> ir_buffer: array<f32>;
@group(0) @binding(9) var<storage, read_write> a_filtered: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> b_filtered: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> edge_test: array<u32>;
@group(0) @binding(12) var<storage, read_write> depth_out: array<f32>;
@group(0) @binding(13) var<storage, read_write> ir_sum_out: array<f32>;
@group(0) @binding(14) var<storage, read_write> filtered: array<f32>;

const PIXELS: u32 = 512u * 424u;

// ---------------------------------------------------------------------------
// Stage 1: raw packet -> per-frequency sine/cosine amplitudes
// ---------------------------------------------------------------------------

fn decode_pixel_measurement(sub: u32, x: u32, y: u32) -> f32 {
    let row_idx = (424u * sub + y) * 352u;
    let idx = (((x >> 2u) + ((x << 7u) & BFI_BITMASK)) * 11u);

    let col_idx = idx >> 4u;
    let upper_bytes = idx & 15u;
    let lower_bytes = 16u - upper_bytes;

    let data_idx0 = row_idx + col_idx;
    let data_idx1 = row_idx + col_idx + 1u;

    // The OpenCL original reads lut11to16[0] for these pixels rather than
    // skipping them, which is why they end up invalid further down.
    if (x < 1u || 510u < x || col_idx > 352u) {
        return f32(lut11to16[0]);
    }

    let packed = ((packet[data_idx0] >> upper_bytes) | (packet[data_idx1] << lower_bytes)) & 2047u;
    return f32(lut11to16[packed]);
}

@compute @workgroup_size(64)
fn process_pixel_stage1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= PIXELS) {
        return;
    }

    let x = i % 512u;
    let y = i / 512u;

    let y_tmp = 423u - y;
    let y_in = select(423u - y_tmp, y_tmp + 212u, y_tmp < 212u);

    // `(int)(0.0f >= z_table[i])`, a bool splatted into an int3.
    let invalid = select(0, 1, 0.0 >= z_table[i]);

    let p0 = p0_table[i].xyz;

    // sincos(x, &cos) returned the sine; the caller negated it.
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

    let invalid_mask = invalid == 1;

    var a = vec3<f32>(dot(v0, p0x_cos), dot(v1, p0y_cos), dot(v2, p0z_cos)) * AB_PER_FRQ;
    var b = vec3<f32>(dot(v0, p0x_sin), dot(v1, p0y_sin), dot(v2, p0z_sin)) * AB_PER_FRQ;

    a = select(a, vec3<f32>(0.0), invalid_mask);
    b = select(b, vec3<f32>(0.0), invalid_mask);
    let n = sqrt(a * a + b * b);

    // Saturated pixels show the 32767 sentinel in any component.
    let saturated = vec3<i32>(
        select(0, 1, any(v0 == vec3<f32>(32767.0))),
        select(0, 1, any(v1 == vec3<f32>(32767.0))),
        select(0, 1, any(v2 == vec3<f32>(32767.0))),
    );
    let saturated_mask = saturated == vec3<i32>(1);

    a_buffer[i] = vec4<f32>(select(a, vec3<f32>(0.0), saturated_mask), 0.0);
    b_buffer[i] = vec4<f32>(select(b, vec3<f32>(0.0), saturated_mask), 0.0);
    n_buffer[i] = vec4<f32>(n, 0.0);

    let ir = dot(
        select(n, vec3<f32>(65535.0), saturated_mask),
        vec3<f32>(0.333333333 * AB_MULTIPLIER * AB_OUTPUT_MULTIPLIER),
    );
    ir_buffer[i] = min(ir, 65535.0);
}

// ---------------------------------------------------------------------------
// Filter stage 1: joint bilateral filter over the amplitude vectors
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn filter_pixel_stage1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= PIXELS) {
        return;
    }

    let xi = i32(i);
    let x = xi % 512;
    let y = xi / 512;

    let self_a = a_buffer[i].xyz;
    let self_b = b_buffer[i].xyz;

    var gaussian = array<f32, 9>(
        GAUSSIAN_0, GAUSSIAN_1, GAUSSIAN_2,
        GAUSSIAN_3, GAUSSIAN_4, GAUSSIAN_5,
        GAUSSIAN_6, GAUSSIAN_7, GAUSSIAN_8,
    );

    if (x < 1 || y < 1 || x > 510 || y > 422) {
        a_filtered[i] = vec4<f32>(self_a, 0.0);
        b_filtered[i] = vec4<f32>(self_b, 0.0);
        edge_test[i] = 1u;
        return;
    }

    var threshold = vec3<f32>(JOINT_BILATERAL_THRESHOLD);
    var bilateral_exp = vec3<f32>(JOINT_BILATERAL_EXP);

    let self_norm = n_buffer[i].xyz;
    let self_normalized_a = self_a / self_norm;
    let self_normalized_b = self_b / self_norm;

    var weight_acc = vec3<f32>(0.0);
    var weighted_a_acc = vec3<f32>(0.0);
    var weighted_b_acc = vec3<f32>(0.0);
    var dist_acc = vec3<f32>(0.0);

    let c0 = self_norm * self_norm < threshold;
    threshold = select(threshold, vec3<f32>(0.0), c0);
    bilateral_exp = select(bilateral_exp, vec3<f32>(0.0), c0);

    var j = 0;
    for (var yi = -1; yi < 2; yi = yi + 1) {
        var i_other = (y + yi) * 512 + x - 1;

        for (var xj = -1; xj < 2; xj = xj + 1) {
            let other = u32(i_other);
            let other_a = a_buffer[other].xyz;
            let other_b = b_buffer[other].xyz;
            let other_norm = n_buffer[other].xyz;
            let other_normalized_a = other_a / other_norm;
            let other_normalized_b = other_b / other_norm;

            let c1 = other_norm * other_norm < threshold;

            let dist = 0.5 * (vec3<f32>(1.0)
                - (self_normalized_a * other_normalized_a + self_normalized_b * other_normalized_b));
            let weight = select(
                vec3<f32>(gaussian[j]) * exp(-1.442695 * bilateral_exp * dist),
                vec3<f32>(0.0),
                c1,
            );

            weighted_a_acc = weighted_a_acc + weight * other_a;
            weighted_b_acc = weighted_b_acc + weight * other_b;
            weight_acc = weight_acc + weight;
            dist_acc = dist_acc + select(dist, vec3<f32>(0.0), c1);

            j = j + 1;
            i_other = i_other + 1;
        }
    }

    let c2 = vec3<f32>(0.0) < weight_acc;
    a_filtered[i] = vec4<f32>(select(vec3<f32>(0.0), weighted_a_acc / weight_acc, c2), 0.0);
    b_filtered[i] = vec4<f32>(select(vec3<f32>(0.0), weighted_b_acc / weight_acc, c2), 0.0);

    edge_test[i] = select(0u, 1u, all(dist_acc < vec3<f32>(JOINT_BILATERAL_MAX_EDGE)));
}

// ---------------------------------------------------------------------------
// Stage 2: amplitudes -> phase -> depth
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn process_pixel_stage2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= PIXELS) {
        return;
    }

    let a = select(a_buffer[i].xyz, a_filtered[i].xyz, USE_BILATERAL);
    let b = select(b_buffer[i].xyz, b_filtered[i].xyz, USE_BILATERAL);

    var phase = atan2(b, a);
    phase = select(phase, phase + vec3<f32>(2.0 * M_PI_F), phase < vec3<f32>(0.0));
    // `isNan` is not available under this naga version, and the IEEE spelling
    // is exact: a value is NaN exactly when it does not equal itself.
    phase = select(phase, vec3<f32>(0.0), phase != phase);

    let ir = sqrt(a * a + b * b) * AB_MULTIPLIER;
    let ir_sum = ir.x + ir.y + ir.z;
    let ir_min = min(ir.x, min(ir.y, ir.z));
    let ir_max = max(ir.x, max(ir.y, ir.z));

    var phase_final = 0.0;

    if (ir_min >= INDIVIDUAL_AB_THRESHOLD && ir_sum >= AB_THRESHOLD) {
        let t = phase / (2.0 * M_PI_F) * vec3<f32>(3.0, 15.0, 2.0);

        let t0 = t.x;
        let t1 = t.y;
        let t2 = t.z;

        let t5 = floor((t1 - t0) * 0.333333 + 0.5) * 3.0 + t0;
        var t3 = -t2 + t5;
        let t4 = t3 * 2.0;

        let c1 = t4 >= -t4;
        let f1 = select(-2.0, 2.0, c1);
        let f2 = select(-0.5, 0.5, c1);
        t3 = t3 * f2;
        t3 = (t3 - floor(t3)) * f1;

        let c2 = 0.5 < abs(t3) && abs(t3) < 1.5;

        var t6 = select(t5, t5 + 15.0, c2);
        var t7 = select(t1, t1 + 15.0, c2);

        var t8 = (floor((-t2 + t6) * 0.5 + 0.5) * 2.0 + t2) * 0.5;

        t6 = t6 * 0.333333;
        t7 = t7 * 0.066667;

        let t9 = t8 + t6 + t7;
        var t10 = t9 * 0.333333;

        t6 = t6 * (2.0 * M_PI_F);
        t7 = t7 * (2.0 * M_PI_F);
        t8 = t8 * (2.0 * M_PI_F);

        let t8_new = t7 * 0.826977 - t8 * 0.110264;
        let t6_new = t8 * 0.551318 - t6 * 0.826977;
        let t7_new = t6 * 0.110264 - t7 * 0.551318;

        t8 = t8_new;
        t6 = t6_new;
        t7 = t7_new;

        let norm = t8 * t8 + t6 * t6 + t7 * t7;
        let mask = select(0.0, 1.0, t9 >= 0.0);
        t10 = t10 * mask;

        let slope_positive = 0.0 < AB_CONFIDENCE_SLOPE;
        var ir_x = select(ir_max, ir_min, slope_positive);

        ir_x = log(ir_x);
        ir_x = (ir_x * AB_CONFIDENCE_SLOPE * 0.301030 + AB_CONFIDENCE_OFFSET) * 3.321928;
        ir_x = exp(ir_x);
        ir_x = clamp(ir_x, MIN_DEALIAS_CONFIDENCE, MAX_DEALIAS_CONFIDENCE);
        ir_x = ir_x * ir_x;

        let mask2 = select(0.0, 1.0, ir_x >= norm);
        let t11 = t10 * mask2;

        let mask3 = select(0.0, 1.0, MAX_DEALIAS_CONFIDENCE * MAX_DEALIAS_CONFIDENCE >= norm);
        t10 = t10 * mask3;

        // The OpenCL original has `true ? t11 : t10` here: the modeMask branch
        // was compiled out upstream and always takes t11.
        phase_final = t11;
    }

    let zmultiplier = z_table[i];
    var xmultiplier = x_table[i];

    phase_final = select(phase_final, phase_final + PHASE_OFFSET, 0.0 < phase_final);

    let depth_linear = zmultiplier * phase_final;
    let max_depth = phase_final * UNAMBIGUOUS_DIST * 2.0;

    let cond1 = 0.0 < depth_linear && 0.0 < max_depth;

    xmultiplier = (xmultiplier * 90.0) / (max_depth * max_depth * 8192.0);

    var depth_fit = depth_linear / (-depth_linear * xmultiplier + 1.0);
    depth_fit = select(0.0, depth_fit, depth_fit >= 0.0);

    depth_out[i] = select(depth_linear, depth_fit, cond1);
    ir_sum_out[i] = ir_sum;
}

// ---------------------------------------------------------------------------
// Filter stage 2: remove flying pixels across depth discontinuities
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn filter_pixel_stage2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= PIXELS) {
        return;
    }

    let xi = i32(i);
    let x = xi % 512;
    let y = xi / 512;

    let raw_depth = depth_out[i];
    let ir_sum = ir_sum_out[i];
    let edge = edge_test[i];

    if (!(raw_depth >= MIN_DEPTH && raw_depth <= MAX_DEPTH)) {
        filtered[i] = 0.0;
        return;
    }

    if (x < 1 || y < 1 || x > 510 || y > 422) {
        filtered[i] = raw_depth;
        return;
    }

    var ir_sum_acc = ir_sum;
    var squared_ir_sum_acc = ir_sum * ir_sum;
    var min_depth = raw_depth;
    var max_depth = raw_depth;

    for (var yi = -1; yi < 2; yi = yi + 1) {
        var i_other = (y + yi) * 512 + x - 1;

        for (var xj = -1; xj < 2; xj = xj + 1) {
            if (i_other != xi) {
                let other = u32(i_other);
                let raw_depth_other = depth_out[other];
                let ir_sum_other = ir_sum_out[other];

                ir_sum_acc = ir_sum_acc + ir_sum_other;
                squared_ir_sum_acc = squared_ir_sum_acc + ir_sum_other * ir_sum_other;

                if (0.0 < raw_depth_other) {
                    min_depth = min(min_depth, raw_depth_other);
                    max_depth = max(max_depth, raw_depth_other);
                }
            }

            i_other = i_other + 1;
        }
    }

    var tmp0 = sqrt(squared_ir_sum_acc * 9.0 - ir_sum_acc * ir_sum_acc) / 9.0;
    let edge_avg = max(ir_sum_acc / 9.0, EDGE_AB_AVG_MIN_VALUE);
    tmp0 = tmp0 / edge_avg;

    let abs_min_diff = abs(raw_depth - min_depth);
    let abs_max_diff = abs(raw_depth - max_depth);

    let avg_diff = (abs_min_diff + abs_max_diff) * 0.5;
    let max_abs_diff = max(abs_min_diff, abs_max_diff);

    let cond0 = 0.0 < raw_depth
        && tmp0 >= EDGE_AB_STD_DEV_THRESHOLD
        && EDGE_CLOSE_DELTA_THRESHOLD < abs_min_diff
        && EDGE_FAR_DELTA_THRESHOLD < abs_max_diff
        && EDGE_MAX_DELTA_THRESHOLD < max_abs_diff
        && EDGE_AVG_DELTA_THRESHOLD < avg_diff;

    if (cond0) {
        filtered[i] = 0.0;
        return;
    }

    if (edge == 0u) {
        filtered[i] = 0.0;
        return;
    }

    // `tmp1` is computed by the original and then never used: `edge_count` is
    // initialised to 0 and the only thing read is the comparison below, so this
    // reduces to "keep raw_depth unless MAX_EDGE_COUNT is negative".
    let edge_count = 0.0;
    filtered[i] = select(raw_depth, 0.0, edge_count > MAX_EDGE_COUNT);
}
