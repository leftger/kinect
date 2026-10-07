use std::f32::{INFINITY, NAN};

use crate::{
    data::{ColorParams, IrParams},
    COLOR_HEIGHT, COLOR_SIZE, COLOR_WIDTH, DEPTH_HEIGHT, DEPTH_SIZE, DEPTH_WIDTH,
};

use super::{color::ColorFrame, depth::DepthFrame};

const FILTER_WIDTH_HALF: isize = 2;
const FILTER_HEIGHT_HALF: isize = 1;
const FILTER_TOLERANCE: f32 = 0.01;

// these seem to be hardcoded in the original SDK
const DEPTH_Q: f32 = 0.01;
const COLOR_Q: f32 = 0.002199;

/// LOCAL PATCH: sentinel stored in `distort_map` for an output pixel whose
/// source falls outside the raw image. Upstream libfreenect2 uses `-1` for this
/// and writes depth 0 in `apply()`. Upstream's Rust port dropped the check and
/// indexed the frame directly, which panics once the distortion pushes a border
/// pixel's rounded coordinate one row/column past the edge.
const DISTORT_MAP_OUTSIDE: usize = usize::MAX;

/// Registration will only work contiguous color space
pub struct Registration {
    /// Depth camera parameters.
    ir_params: IrParams,
    /// Color camera parameters.
    color_params: ColorParams,
    distort_map: Box<[usize; DEPTH_SIZE]>,
    depth_to_color_map_x: Box<[f32; DEPTH_SIZE]>,
    depth_to_color_map_y: Box<[f32; DEPTH_SIZE]>,
}

/// Heap allocation of a depth-sized array.
///
/// `Box::new([0; DEPTH_SIZE])` builds the array on the stack first. A depth map
/// of `usize` is about 1.7 MB, and four of them overflow the 2 MB stack a test
/// thread gets. `Vec` allocates directly.
fn depth_sized<T: Copy>(value: T) -> Box<[T; DEPTH_SIZE]> {
    let boxed: Box<[T]> = vec![value; DEPTH_SIZE].into_boxed_slice();
    boxed.try_into().ok().expect("vec length is DEPTH_SIZE")
}

impl Registration {
    pub fn new() -> Self {
        Self {
            ir_params: Default::default(),
            color_params: Default::default(),
            distort_map: depth_sized(0),
            depth_to_color_map_x: depth_sized(0.0),
            depth_to_color_map_y: depth_sized(0.0),
        }
    }

    fn fill_depth_to_color_map(&mut self) {
        for y in 0..DEPTH_HEIGHT {
            for x in 0..DEPTH_WIDTH {
                let offset = x + y * DEPTH_WIDTH;

                // compute the dirstored coordinate for current pixel
                let (mx, my) = self.distort(x, y);
                // LOCAL PATCH: round towards zero, as upstream does with
                // `(int)(v + 0.5f)`, and record out-of-image sources as a
                // sentinel instead of forming an out-of-bounds index.
                let ix = (mx + 0.5).trunc();
                let iy = (my + 0.5).trunc();

                let inside =
                    ix >= 0.0 && iy >= 0.0 && ix < DEPTH_WIDTH as f32 && iy < DEPTH_HEIGHT as f32;

                // computing the index from the coordianted for faster access to the data
                self.distort_map[offset] = if inside {
                    iy as usize * DEPTH_WIDTH + ix as usize
                } else {
                    DISTORT_MAP_OUTSIDE
                };

                // compute the depth to color mapping entries for the current pixel
                let (rx, ry) = self.depth_to_color(x as f32, y as f32);

                self.depth_to_color_map_x[offset] = rx;
                self.depth_to_color_map_y[offset] = ry;
            }
        }
    }

    pub fn set_ir_params(&mut self, ir_params: &IrParams) {
        self.ir_params = *ir_params;
        self.fill_depth_to_color_map();
    }

    pub fn set_color_params(&mut self, color_params: &ColorParams) {
        self.color_params = *color_params;
        self.fill_depth_to_color_map();
    }

    pub fn undistort_depth_and_color(
        &self,
        color_frame: &ColorFrame,
        depth_frame: &DepthFrame,
        enable_filter: bool,
    ) -> (ColorFrame, DepthFrame) {
        let (registered, undistorted, _valid) =
            self.undistort_depth_and_color_with_validity(color_frame, depth_frame, enable_filter);
        (registered, undistorted)
    }

    /// As [`undistort_depth_and_color`], plus which depth-grid pixels received colour.
    ///
    /// `valid[i]` is `1` when pixel `i` was copied from the colour camera and `0`
    /// when it was not: no depth, a source outside the image, or a sample the
    /// filter rejected. Those pixels stay `0` in the colour buffer, and `0` is
    /// also a real black sample, so the buffer alone cannot say which is which.
    pub fn undistort_depth_and_color_with_validity(
        &self,
        color_frame: &ColorFrame,
        depth_frame: &DepthFrame,
        enable_filter: bool,
    ) -> (ColorFrame, DepthFrame, Vec<u8>) {
        let bytes_per_pixel = color_frame.color_space.bytes_per_pixel();
        let mut registered_frame = ColorFrame {
            color_space: color_frame.color_space,
            width: DEPTH_WIDTH,
            height: DEPTH_HEIGHT,
            buffer: vec![0; DEPTH_SIZE * bytes_per_pixel],
            sequence: color_frame.sequence,
            timestamp: color_frame.timestamp,
            exposure: color_frame.exposure,
            gain: color_frame.gain,
            gamma: color_frame.gamma,
        };
        let mut undistorted_frame = DepthFrame {
            width: DEPTH_WIDTH,
            height: DEPTH_HEIGHT,
            buffer: Vec::with_capacity(DEPTH_SIZE),
            sequence: depth_frame.sequence,
            timestamp: depth_frame.timestamp,
        };
        // Parallel to the registered colour. Stays 0 until a sample is copied.
        let mut valid = vec![0u8; DEPTH_SIZE];

        // map for storing the min z values used for each color pixel
        // initializing the depth_map with values outside of the Kinect2 range if filter is enabled
        //
        // Heap, not a local array: COLOR_SIZE is 1920x1080 floats, about 8 MB, which
        // overflows the main thread stack on macOS.
        let mut filter_map = vec![INFINITY; COLOR_SIZE];

        // map for storing the color offset for each depth pixel
        let mut depth_to_c_off = Vec::with_capacity(DEPTH_SIZE);

        /* Fix depth distortion, and compute pixel to use from 'color' based on depth measurement,
         * stored as x/y offset in the color data.
         */

        // iterating over all pixels from undistorted depth and registered color image
        // the four maps have the same structure as the images, so their pointers are increased each iteration as well
        for i in 0..DEPTH_SIZE {
            // getting index of distorted depth pixel
            let index = self.distort_map[i];

            // LOCAL PATCH: an out-of-image source pixel carries no depth.
            // Upstream writes 0 for these, and depth 0 is treated as invalid.
            if index == DISTORT_MAP_OUTSIDE {
                undistorted_frame.buffer.push(0.0);
                depth_to_c_off.push(None);
                continue;
            }

            // getting depth value for current pixel
            let z = depth_frame.buffer[index];

            undistorted_frame.buffer.push(z);

            // checking for invalid depth value
            if z <= 0.0 {
                depth_to_c_off.push(None);
                continue;
            }

            // Colour-image coordinate. Checked before the `usize` cast: a
            // value past the width still forms an in-range linear index
            // (`x + y * COLOR_WIDTH`) that addresses the next row, and a
            // non-finite or negative value saturates to 0 or `usize::MAX`.
            let x = (self.depth_to_color_map_x[i] + (self.color_params.shift_m / z))
                * self.color_params.fx
                + self.color_params.cx.round();
            let y = self.depth_to_color_map_y[i] + 0.5;
            let Some((cx, cy)) = color_pixel(x, y) else {
                depth_to_c_off.push(None);
                continue;
            };
            let c_off = cx + cy * COLOR_WIDTH;

            if c_off >= COLOR_SIZE {
                depth_to_c_off.push(None);
                continue;
            }

            // saving the offset for later
            depth_to_c_off.push(Some(c_off));

            if enable_filter {
                // setting a window around the filter map pixel corresponding to the color pixel with the current z value
                for y_off in -FILTER_HEIGHT_HALF..FILTER_HEIGHT_HALF {
                    for x_off in -FILTER_WIDTH_HALF..FILTER_WIDTH_HALF {
                        if let (Some(cx), Some(cy)) =
                            (cx.checked_add_signed(x_off), cy.checked_add_signed(y_off))
                        {
                            let offset = cx + cy * COLOR_WIDTH;

                            // only set if the current z is smaller
                            if offset < COLOR_SIZE && z < filter_map[offset] {
                                filter_map[offset] = z;
                            }
                        }
                    }
                }
            }
        }

        /* Construct 'registered' image. */

        // run through all registered color pixels and set them based on filter results if enabled
        for i in 0..DEPTH_SIZE {
            let Some(c_off) = depth_to_c_off[i] else {
                // if offset is out of image
                continue;
            };

            /* Filter drops duplicate pixels due to aspect of two cameras. */
            if enable_filter {
                let min_z = filter_map[c_off];
                let z = undistorted_frame.buffer[i];

                // check for allowed depth noise
                if (z - min_z) / z > FILTER_TOLERANCE {
                    continue;
                }
            }

            let c_off = c_off * bytes_per_pixel;
            let r_off = i * bytes_per_pixel;

            registered_frame.buffer[r_off..r_off + bytes_per_pixel]
                .copy_from_slice(&color_frame.buffer[c_off..c_off + bytes_per_pixel]);
            valid[i] = 1;
        }

        (registered_frame, undistorted_frame, valid)
    }

    pub fn undistort_depth(&self, depth_frame: &DepthFrame) -> DepthFrame {
        let mut undistorted_frame = DepthFrame {
            width: DEPTH_WIDTH,
            height: DEPTH_HEIGHT,
            buffer: Vec::with_capacity(DEPTH_SIZE),
            sequence: depth_frame.sequence,
            timestamp: depth_frame.timestamp,
        };

        /* Fix depth distortion, and compute pixel to use from 'color' based on depth measurement,
         * stored as x/y offset in the color data.
         */

        // iterating over all pixels from undistorted depth and registered color image
        // the four maps have the same structure as the images, so their pointers are increased each iteration as well
        for i in 0..DEPTH_SIZE {
            // LOCAL PATCH: a source pixel outside the raw image carries no depth.
            let source = self.distort_map[i];
            undistorted_frame.buffer.push(
                if source == DISTORT_MAP_OUTSIDE || source >= depth_frame.buffer.len() {
                    0.0
                } else {
                    depth_frame.buffer[source]
                },
            );
        }

        undistorted_frame
    }

    pub fn xyz_to_point(&self, dx: usize, dy: usize, dz: f32) -> (f32, f32) {
        let index = dx + dy * DEPTH_WIDTH;

        (
            (self.depth_to_color_map_x[index] + (self.color_params.shift_m / dz))
                * self.color_params.fx
                + self.color_params.cx,
            self.depth_to_color_map_y[index],
        )
    }

    pub fn point_to_xyz_pixel(
        &self,
        undistorted_frame: &DepthFrame,
        registered_frame: &ColorFrame,
        x: usize,
        y: usize,
    ) -> (f32, f32, f32, Vec<u8>) {
        let bytes_per_pixel = registered_frame.color_space.bytes_per_pixel();
        let (x, y, z) = self.point_to_xyz(undistorted_frame, x, y);
        let c_off = DEPTH_WIDTH * y as usize + x as usize;
        let pixel = if z.is_nan() {
            vec![0; bytes_per_pixel]
        } else {
            registered_frame.buffer[c_off..c_off + bytes_per_pixel].to_vec()
        };

        (x, y, z, pixel)
    }

    pub fn point_to_xyz(
        &self,
        undistorted_frame: &DepthFrame,
        x: usize,
        y: usize,
    ) -> (f32, f32, f32) {
        let depth_val = undistorted_frame.buffer[DEPTH_WIDTH * y + x] / 1000.0; // scaling factor, so that value of 1 is one meter.

        if depth_val.is_nan() || depth_val <= 0.001 {
            // depth value is not valid
            (NAN, NAN, NAN)
        } else {
            (
                (x as f32 + 0.5 - self.ir_params.cx) * (1.0 / self.ir_params.fx) * depth_val,
                (y as f32 + 0.5 - self.ir_params.cy) * (1.0 / self.ir_params.fy) * depth_val,
                depth_val,
            )
        }
    }

    pub fn distort(&self, mx: usize, my: usize) -> (f32, f32) {
        // see http://en.wikipedia.org/wiki/Distortion_(optics) for description
        let dx = (mx as f32 - self.ir_params.cx) / self.ir_params.fx;
        let dy = (my as f32 - self.ir_params.cy) / self.ir_params.fy;
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let r2 = dx2 + dy2;
        let dxdy2 = 2.0 * dx * dy;
        let kr = 1.0 + ((self.ir_params.k3 * r2 + self.ir_params.k2) * r2 + self.ir_params.k1) * r2;

        (
            self.ir_params.fx
                * (dx * kr + self.ir_params.p2 * (r2 + 2.0 * dx2) + self.ir_params.p1 * dxdy2)
                + self.ir_params.cx,
            self.ir_params.fy
                * (dy * kr + self.ir_params.p1 * (r2 + 2.0 * dy2) + self.ir_params.p2 * dxdy2)
                + self.ir_params.cy,
        )
    }

    pub fn depth_to_color(&self, mx: f32, my: f32) -> (f32, f32) {
        let mx = (mx - self.ir_params.cx) * DEPTH_Q;
        let my = (my - self.ir_params.cy) * DEPTH_Q;
        let mxy = mx * my;
        let mx2 = mx * mx;
        let my2 = my * my;
        let mx3 = mx * mx2;
        let my3 = my * my2;
        let mx2y = mx2 * my;
        let mxy2 = mx * my2;

        let wx = (mx3 * self.color_params.mx_x3y0)
            + (my3 * self.color_params.mx_x0y3)
            + (mx2y * self.color_params.mx_x2y1)
            + (mxy2 * self.color_params.mx_x1y2)
            + (mx2 * self.color_params.mx_x2y0)
            + (my2 * self.color_params.mx_x0y2)
            + (mxy * self.color_params.mx_x1y1)
            + (mx * self.color_params.mx_x1y0)
            + (my * self.color_params.mx_x0y1)
            + (self.color_params.mx_x0y0);

        let wy = (mx3 * self.color_params.my_x3y0)
            + (my3 * self.color_params.my_x0y3)
            + (mx2y * self.color_params.my_x2y1)
            + (mxy2 * self.color_params.my_x1y2)
            + (mx2 * self.color_params.my_x2y0)
            + (my2 * self.color_params.my_x0y2)
            + (mxy * self.color_params.my_x1y1)
            + (mx * self.color_params.my_x1y0)
            + (my * self.color_params.my_x0y1)
            + (self.color_params.my_x0y0);

        (
            (wx / (self.color_params.fx * COLOR_Q))
                - (self.color_params.shift_m / self.color_params.shift_d),
            (wy / COLOR_Q) + self.color_params.cy,
        )
    }
}

/// Colour pixel for a mapped coordinate, or `None` when it must not be cast.
///
/// `x` and `y` are the floats that registration would otherwise convert with
/// `as usize`. Negatives and NaNs become 0, and a huge value saturates to
/// `usize::MAX`. An x at or past [`COLOR_WIDTH`] is the row-alias case: added
/// to `y * COLOR_WIDTH` it can still land inside [`COLOR_SIZE`], one row down.
fn color_pixel(x: f32, y: f32) -> Option<(usize, usize)> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= COLOR_WIDTH as f32
        || y >= COLOR_HEIGHT as f32
    {
        return None;
    }
    Some((x as usize, y as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColorParams, IrParams};
    use crate::processor::color::ColorSpace;
    use crate::processor::depth::DepthFrame;

    /// No lens distortion, and a colour mapping that lands inside the colour
    /// image for every depth pixel. The polynomial is all zeros, so every depth
    /// pixel reads the same colour pixel; the tests only care which depth pixels
    /// are marked registered.
    fn flat_registration() -> Registration {
        let mut registration = Registration::new();
        registration.set_ir_params(&IrParams {
            fx: 365.0,
            fy: 365.0,
            cx: (DEPTH_WIDTH / 2) as f32,
            cy: (DEPTH_HEIGHT / 2) as f32,
            ..IrParams::default()
        });
        registration.set_color_params(&ColorParams {
            fx: 1.0,
            fy: 1.0,
            cx: 10.0,
            cy: 20.0,
            shift_d: 1.0,
            shift_m: 0.0,
            ..ColorParams::default()
        });
        registration
    }

    fn color_frame(rgb: [u8; 3], exposure: f32, gain: f32, gamma: f32) -> ColorFrame {
        let mut buffer = Vec::with_capacity(COLOR_SIZE * 3);
        for _ in 0..COLOR_SIZE {
            buffer.extend_from_slice(&rgb);
        }
        ColorFrame {
            color_space: ColorSpace::RGB,
            width: COLOR_WIDTH,
            height: crate::COLOR_HEIGHT,
            buffer,
            sequence: 7,
            timestamp: 9,
            exposure,
            gain,
            gamma,
        }
    }

    fn depth_frame(buffer: Vec<f32>) -> DepthFrame {
        DepthFrame {
            width: DEPTH_WIDTH,
            height: DEPTH_HEIGHT,
            buffer,
            sequence: 3,
            timestamp: 4,
        }
    }

    #[test]
    fn the_existing_call_matches_the_masked_one() {
        let registration = flat_registration();
        let color = color_frame([11, 22, 33], 0.25, 1.5, 2.2);
        let depth = depth_frame(vec![1500.0; DEPTH_SIZE]);

        let (plain_color, plain_depth) =
            registration.undistort_depth_and_color(&color, &depth, false);
        let (masked_color, masked_depth, valid) =
            registration.undistort_depth_and_color_with_validity(&color, &depth, false);

        assert_eq!(plain_color.buffer, masked_color.buffer);
        assert_eq!(plain_depth.buffer, masked_depth.buffer);
        assert_eq!(plain_color.exposure, 0.25);
        assert_eq!(masked_color.gain, 1.5);
        assert_eq!(masked_color.gamma, 2.2);
        assert!(valid.iter().all(|flag| *flag == 1));
    }

    #[test]
    fn a_black_sample_is_still_a_valid_registration() {
        // The registered buffer is zeros either way. The mask is the only record
        // that these pixels were copied rather than left untouched.
        let registration = flat_registration();
        let color = color_frame([0, 0, 0], 1.0, 1.0, 1.0);
        let depth = depth_frame(vec![1000.0; DEPTH_SIZE]);

        let (registered, _, valid) =
            registration.undistort_depth_and_color_with_validity(&color, &depth, false);

        assert!(registered.buffer.iter().all(|byte| *byte == 0));
        assert_eq!(valid.iter().filter(|flag| **flag == 1).count(), DEPTH_SIZE);
    }

    #[test]
    fn missing_depth_and_an_unmapped_colour_pixel_are_invalid() {
        let registration = flat_registration();
        let color = color_frame([9, 8, 7], 0.1, 1.0, 1.0);
        let mut depth = vec![0.0; DEPTH_SIZE];
        depth[10] = 800.0;
        depth[20] = 800.0;

        let (registered, undistorted, valid) = registration
            .undistort_depth_and_color_with_validity(&color, &depth_frame(depth), false);

        assert_eq!(valid.iter().filter(|flag| **flag == 1).count(), 2);
        assert_eq!(valid[10], 1);
        assert_eq!(valid[20], 1);
        assert_eq!(valid[0], 0);
        assert_eq!(&registered.buffer[10 * 3..10 * 3 + 3], &[9, 8, 7]);
        assert_eq!(&registered.buffer[0..3], &[0, 0, 0]);
        assert_eq!(undistorted.buffer[10], 800.0);
        assert_eq!(undistorted.buffer[0], 0.0);

        // Push the whole mapping past the colour image. Depth is still there;
        // colour is not, and the mask has to say so.
        let mut registration = flat_registration();
        registration.set_color_params(&ColorParams {
            fx: 1.0,
            fy: 1.0,
            cx: 10.0,
            cy: 5_000.0,
            shift_d: 1.0,
            shift_m: 0.0,
            ..ColorParams::default()
        });
        let (registered, undistorted, valid) = registration
            .undistort_depth_and_color_with_validity(
                &color,
                &depth_frame(vec![800.0; DEPTH_SIZE]),
                false,
            );
        assert!(valid.iter().all(|flag| *flag == 0));
        assert!(registered.buffer.iter().all(|byte| *byte == 0));
        assert!(undistorted.buffer.iter().all(|z| *z == 800.0));
    }

    #[test]
    fn the_filter_drops_a_farther_pixel_that_shares_a_colour_sample() {
        let registration = flat_registration();
        let color = color_frame([4, 5, 6], 0.2, 1.2, 0.9);
        let mut depth = vec![0.0; DEPTH_SIZE];
        // Both pixels map to the same colour location under the flat mapping.
        // The filter keeps the nearer one and rejects the one more than 1% farther.
        depth[100] = 1_000.0;
        depth[200] = 2_000.0;

        let (_, _, valid) = registration.undistort_depth_and_color_with_validity(
            &color,
            &depth_frame(depth.clone()),
            true,
        );

        assert_eq!(valid[100], 1, "the nearer sample should be kept");
        assert_eq!(valid[200], 0, "the farther sample should be filtered out");
        assert_eq!(valid.iter().filter(|flag| **flag == 1).count(), 1);

        let (_, _, unfiltered) = registration.undistort_depth_and_color_with_validity(
            &color,
            &depth_frame(depth),
            false,
        );
        assert_eq!(unfiltered[100], 1);
        assert_eq!(unfiltered[200], 1);
    }

    fn registration_at(cx: f32, cy: f32) -> Registration {
        let mut registration = flat_registration();
        registration.set_color_params(&ColorParams {
            fx: 1.0,
            fy: 1.0,
            cx,
            cy,
            shift_d: 1.0,
            shift_m: 0.0,
            ..ColorParams::default()
        });
        registration
    }

    /// Paint one colour pixel and register a full depth frame through `params`.
    fn register_marked(cx: f32, cy: f32, marked: usize) -> (Vec<u8>, Vec<u8>) {
        let registration = registration_at(cx, cy);
        let mut color = color_frame([1, 1, 1], 1.0, 1.0, 1.0);
        let offset = marked * 3;
        color.buffer[offset..offset + 3].copy_from_slice(&[9, 8, 7]);
        let (registered, _, valid) = registration.undistort_depth_and_color_with_validity(
            &color,
            &depth_frame(vec![1_000.0; DEPTH_SIZE]),
            true,
        );
        (registered.buffer, valid)
    }

    #[test]
    fn coordinates_outside_the_colour_image_stay_invalid_and_do_not_alias() {
        // x = 1920 is the first column of the next row if it is added into
        // `y * COLOR_WIDTH` without a width check. y is stored as `(cy + 0.5)`.
        let aliased = 1920 + 20 * COLOR_WIDTH;
        let (registered, valid) = register_marked(1920.0, 20.0, aliased);
        assert!(valid.iter().all(|flag| *flag == 0));
        assert!(
            registered.iter().all(|byte| *byte == 0),
            "an out-of-width x must not copy the next row"
        );

        // Negative x used to saturate to column 0 of that row.
        let column_zero = 20 * COLOR_WIDTH;
        let (registered, valid) = register_marked(-4.0, 20.0, column_zero);
        assert!(valid.iter().all(|flag| *flag == 0));
        assert!(registered.iter().all(|byte| *byte == 0));

        // Negative y used to saturate to row 0.
        let (registered, valid) = register_marked(10.0, -8.0, 10);
        assert!(valid.iter().all(|flag| *flag == 0));
        assert!(registered.iter().all(|byte| *byte == 0));

        // NaN x used to become column 0. Infinity y used to saturate and then
        // overflow the row stride.
        let (registered, valid) = register_marked(f32::NAN, 20.0, column_zero);
        assert!(valid.iter().all(|flag| *flag == 0));
        assert!(registered.iter().all(|byte| *byte == 0));

        let (registered, valid) = register_marked(10.0, f32::INFINITY, 10);
        assert!(valid.iter().all(|flag| *flag == 0));
        assert!(registered.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn the_occlusion_filter_leaves_the_rejected_pixel_uncoloured() {
        let registration = flat_registration();
        let color = color_frame([4, 5, 6], 0.2, 1.2, 0.9);
        let mut depth = vec![0.0; DEPTH_SIZE];
        depth[100] = 1_000.0;
        depth[200] = 2_000.0;

        let (registered, _, valid) =
            registration.undistort_depth_and_color_with_validity(&color, &depth_frame(depth), true);

        assert_eq!(valid[100], 1);
        assert_eq!(valid[200], 0);
        assert_eq!(&registered.buffer[100 * 3..100 * 3 + 3], &[4, 5, 6]);
        assert_eq!(&registered.buffer[200 * 3..200 * 3 + 3], &[0, 0, 0]);
        assert_eq!(valid.iter().filter(|flag| **flag == 1).count(), 1);
    }
}
