//! RGB8 PNG encoder.
//!
//! Textured exports need a real PNG: the sibling of an OBJ or glTF file, and
//! the image embedded in a GLB. The atlas is capped at 8192×8192. Writing that
//! square with stored (uncompressed) deflate blocks keeps every byte:
//!
//! ```text
//! 8192 × 8192 × 3 = 201_326_592 pixel bytes
//! + one filter byte per row, plus a few kilobytes of block headers
//! ≈ 192 MiB
//! ```
//!
//! Most of an atlas is the flat fallback colour around the charts. Deflate
//! collapses that run; a stored block does not, and the file would sit beside
//! the mesh or inside the GLB at full size. This module therefore compresses
//! the IDAT with the `png` crate. Decoding inflates that stream and checks the
//! pixels. It does not depend on which deflate strategy produced them.

use std::io::{self, ErrorKind};

/// Encode `rgb` as an 8-bit truecolor PNG, row-major, three bytes per pixel.
pub fn encode_rgb8(width: u32, height: u32, rgb: &[u8]) -> io::Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(invalid("png dimensions must be non-zero"));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| invalid("png dimensions overflow"))?;
    if rgb.len() != expected {
        return Err(invalid(format!(
            "png pixel buffer is {} bytes, expected {expected} for a {width}x{height} rgb image",
            rgb.len()
        )));
    }

    let mut png = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        // Fast still crushes the flat fallback region, which is the bulk of a
        // large atlas, and it does not stall the end of a scan the way a
        // maximum-effort deflate of a 192 MiB buffer would.
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().map_err(png_error)?;
        writer.write_image_data(rgb).map_err(png_error)?;
        writer.finish().map_err(png_error)?;
    }
    Ok(png)
}

/// Encode `gray` as a 16-bit grayscale PNG, row-major, one `u16` per pixel.
pub fn encode_gray16(width: u32, height: u32, gray: &[u16]) -> io::Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(invalid("png dimensions must be non-zero"));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| invalid("png dimensions overflow"))?;
    if gray.len() != expected {
        return Err(invalid(format!(
            "png pixel buffer is {} pixels, expected {expected} for a {width}x{height} gray16 image",
            gray.len()
        )));
    }

    // PNG 16-bit samples are big-endian.
    let mut bytes = Vec::with_capacity(expected * 2);
    for &sample in gray {
        bytes.extend_from_slice(&sample.to_be_bytes());
    }

    let mut png = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png, width, height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Sixteen);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().map_err(png_error)?;
        writer.write_image_data(&bytes).map_err(png_error)?;
        writer.finish().map_err(png_error)?;
    }
    Ok(png)
}

/// Grow `png` to at least `min_len` bytes by adding an ancillary `tEXt` chunk
/// between the last image-data chunk and `IEND`.
///
/// Some readers sniff an image's format and size from a fixed-size first read
/// and treat a shorter file as truncated. Brush 0.3.0 reads exactly 16,387
/// bytes with `read_exact`, so any image file below that fails to load with
/// "early eof" -- which is every frame of a small synthetic capture. Padding
/// keeps such files loadable without touching a single pixel.
///
/// The chunk goes *after* the image data on purpose: a decoder only needs the
/// header and the first `IDAT` to learn the dimensions, so those stay at the
/// front of the file whatever the padding length. `tEXt` after `IDAT` is legal
/// and every decoder ignores it. A file already long enough is returned as is,
/// and so is anything that does not end in a well-formed `IEND`.
pub fn pad_to_min_size(mut png: Vec<u8>, min_len: usize) -> Vec<u8> {
    const IEND: [u8; 12] = [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82];
    const KEYWORD: &[u8] = b"Comment\0";
    // length (4) + type (4) + keyword + crc (4)
    const OVERHEAD: usize = 12 + KEYWORD.len();

    if png.len() >= min_len || png.len() < 8 + IEND.len() || !png.ends_with(&IEND) {
        return png;
    }

    let body_len = (min_len - png.len()).max(OVERHEAD) - 12;
    let mut body = Vec::with_capacity(body_len);
    body.extend_from_slice(KEYWORD);
    body.resize(body_len, b' ');

    let mut chunk = Vec::with_capacity(12 + body_len);
    chunk.extend_from_slice(&(body_len as u32).to_be_bytes());
    chunk.extend_from_slice(b"tEXt");
    chunk.extend_from_slice(&body);
    let crc = crc32(&chunk[4..]);
    chunk.extend_from_slice(&crc.to_be_bytes());

    let iend_at = png.len() - IEND.len();
    png.splice(iend_at..iend_at, chunk);
    png
}

/// CRC-32 (IEEE, reflected) as PNG chunks use it. Bitwise: chunks here are
/// small and this runs once per padded image.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Inflate a PNG this module produced and return `(width, height, rgb)`.
///
/// Public so a test in another crate can read back what an exporter wrote and
/// compare pixels, rather than inferring them from file sizes.
pub fn decode_rgb8(bytes: &[u8]) -> io::Result<(u32, u32, Vec<u8>)> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().map_err(png_error)?;
    let (width, height, truecolor) = {
        let info = reader.info();
        (
            info.width,
            info.height,
            info.bit_depth == png::BitDepth::Eight
                && info.color_type == png::ColorType::Rgb
                && !info.interlaced,
        )
    };
    if !truecolor {
        return Err(invalid("png is not 8-bit non-interlaced truecolor rgb"));
    }
    let mut buf = vec![0; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).map_err(png_error)?;
    let rgb = buf[..frame.buffer_size()].to_vec();
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| invalid("png dimensions overflow"))?;
    if rgb.len() != expected {
        return Err(invalid("inflated png size does not match IHDR"));
    }
    Ok((width, height, rgb))
}

fn png_error(error: impl ToString) -> io::Error {
    invalid(error.to_string())
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIGNATURE: &[u8] = &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

    /// Bytes a stored-block PNG of this image would occupy, filter byte and
    /// chunk framing included. The encoder must beat this on flat colour:
    /// an 8192 atlas of stored blocks is about 192 MiB.
    fn stored_png_bytes(width: u32, height: u32) -> usize {
        let filtered = width as usize * height as usize * 3 + height as usize;
        let blocks = filtered.div_ceil(65_535);
        // signature + IHDR chunk + IDAT chunk around a zlib stored stream + IEND
        8 + 25 + (12 + 2 + blocks * 5 + filtered + 4) + 12
    }

    #[test]
    fn a_one_pixel_png_round_trips_and_describes_itself() {
        let png = encode_rgb8(1, 1, &[255, 0, 0]).expect("encode");
        assert_eq!(&png[..8], SIGNATURE);

        let (width, height, rgb) = decode_rgb8(&png).expect("decode");
        assert_eq!((width, height), (1, 1));
        assert_eq!(rgb, vec![255, 0, 0]);

        let length = u32::from_be_bytes(png[8..12].try_into().unwrap());
        assert_eq!(length, 13);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..20], &1u32.to_be_bytes());
        assert_eq!(&png[20..24], &1u32.to_be_bytes());
        assert_eq!(&png[24..29], &[8, 2, 0, 0, 0]);
        assert!(png.windows(4).any(|window| window == b"IEND"));
    }

    #[test]
    fn rows_round_trip() {
        let rgb = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 22, 33, 44, 55, 66, 77, 88, 99,
        ];
        let png = encode_rgb8(3, 2, &rgb).expect("encode");
        let (width, height, decoded) = decode_rgb8(&png).expect("decode");
        assert_eq!((width, height), (3, 2));
        assert_eq!(decoded, rgb);
    }

    #[test]
    fn a_tampered_checksum_is_rejected() {
        let mut png = encode_rgb8(1, 1, &[1, 2, 3]).expect("encode");
        // IHDR's CRC sits immediately after its 13-byte payload. The image
        // chunk is later in the file; flipping the trailing IEND checksum is
        // not consulted once the pixels have already been inflated.
        let ihdr_crc = 8 + 4 + 4 + 13;
        png[ihdr_crc] ^= 0xff;
        let error = decode_rgb8(&png).expect_err("crc");
        let message = error.to_string().to_ascii_lowercase();
        assert!(message.contains("crc"), "unexpected error: {error}");
    }

    #[test]
    fn the_pixel_buffer_must_match_the_header() {
        let error = encode_rgb8(2, 2, &[0, 0, 0]).expect_err("short");
        assert!(error.to_string().contains("expected"));
        assert!(encode_rgb8(0, 1, &[]).is_err());
    }

    #[test]
    fn flat_colour_compresses_far_below_a_stored_block() {
        let png = encode_rgb8(64, 64, &vec![128; 64 * 64 * 3]).expect("encode");
        let stored = stored_png_bytes(64, 64);
        assert!(
            png.len() * 4 < stored,
            "compressed {} bytes, stored encoding would be {stored}",
            png.len()
        );
    }

    #[test]
    fn a_16bit_grayscale_png_round_trips() {
        let samples = vec![0u16, 1, 500, 1500, 4500, 65535];
        let encoded = encode_gray16(3, 2, &samples).expect("encode gray16");
        assert_eq!(&encoded[..8], SIGNATURE);

        let mut decoder = png::Decoder::new(std::io::Cursor::new(&encoded));
        decoder.set_transformations(png::Transformations::IDENTITY);
        let mut reader = decoder.read_info().expect("read info");
        let info = reader.info();
        assert_eq!((info.width, info.height), (3, 2));
        assert_eq!(info.color_type, png::ColorType::Grayscale);
        assert_eq!(info.bit_depth, png::BitDepth::Sixteen);

        let mut buf = vec![0u8; reader.output_buffer_size()];
        let frame = reader.next_frame(&mut buf).expect("frame");
        let decoded: Vec<u16> = buf[..frame.buffer_size()]
            .chunks_exact(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn an_8192_atlas_is_too_large_to_store_uncompressed() {
        let bytes = stored_png_bytes(8192, 8192);
        assert!(
            bytes > 190_000_000,
            "stored 8192 atlas would be {bytes} bytes"
        );
    }

    #[test]
    fn crc32_matches_the_png_specification_check_value() {
        // The standard CRC-32 check value for the ASCII digits "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        // IEND's well-known CRC, which `pad_to_min_size` relies on.
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }

    #[test]
    fn padding_reaches_the_minimum_without_changing_a_pixel() {
        let rgb: Vec<u8> = (0..12 * 9 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let png = encode_rgb8(12, 9, &rgb).expect("encode");
        assert!(png.len() < 16_400, "the fixture must start small");

        let padded = pad_to_min_size(png.clone(), 16_400);
        assert!(padded.len() >= 16_400, "padded to {}", padded.len());
        assert!(padded.len() < 16_400 + 64, "overshot to {}", padded.len());

        // Still a valid PNG with the same image: the decoder verifies chunk CRCs.
        let (width, height, decoded) = decode_rgb8(&padded).expect("decode padded");
        assert_eq!((width, height), (12, 9));
        assert_eq!(decoded, rgb);

        // The header and the first IDAT stay at the front, where a size sniff
        // looks; only the tail grew.
        assert_eq!(&padded[..png.len() - 12], &png[..png.len() - 12]);
        assert!(
            padded.ends_with(&png[png.len() - 12..]),
            "IEND must stay last"
        );
    }

    #[test]
    fn padding_leaves_a_long_enough_or_malformed_file_alone() {
        let png = encode_rgb8(4, 4, &[9; 48]).expect("encode");
        assert_eq!(pad_to_min_size(png.clone(), png.len()), png);
        assert_eq!(pad_to_min_size(png.clone(), 10), png);

        let truncated = png[..png.len() - 3].to_vec();
        assert_eq!(pad_to_min_size(truncated.clone(), 20_000), truncated);
        assert_eq!(pad_to_min_size(Vec::new(), 100), Vec::<u8>::new());
    }

    #[test]
    fn a_tiny_deficit_still_produces_a_valid_chunk() {
        let png = encode_rgb8(4, 4, &[9; 48]).expect("encode");
        // One byte short: the chunk has a fixed minimum size, so it overshoots.
        let padded = pad_to_min_size(png.clone(), png.len() + 1);
        assert!(padded.len() > png.len() + 1);
        assert!(decode_rgb8(&padded).is_ok());
    }
}
