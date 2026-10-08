//! Depth-frame recording and replay.
//!
//! Replay exists because tuning odometry against a live sensor is miserable: the
//! scene changes between runs, so two parameter sets never see the same input,
//! and this laptop throttles hard enough to swamp timing. Recording the
//! *processed* depth frames (not the raw USB packets) gives byte-identical input
//! and is ~3x smaller than the raw packets while still exercising all the
//! geometry.
//!
//! Colour is not in the file. `--color` is live-only; replaying a `.k2df`
//! reconstructs geometry and has no views to paint.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use geom::Intrinsics;

const MAGIC: &[u8; 4] = b"K2DF";
pub const VERSION_DEPTH_ONLY: u32 = 1;
pub const VERSION_WITH_COLOR: u32 = 2;

use crate::scanner::FrameColor;

/// Appends depth frames (and optionally registered colour) to a recording.
pub struct FrameWriter {
    writer: BufWriter<File>,
    pub width: usize,
    pub height: usize,
    pub count: usize,
    has_color: bool,
}

impl FrameWriter {
    pub fn create(
        path: &Path,
        intrinsics: Intrinsics,
        width: usize,
        height: usize,
    ) -> io::Result<Self> {
        Self::create_with_color_flag(path, intrinsics, width, height, false)
    }

    pub fn create_with_color(
        path: &Path,
        intrinsics: Intrinsics,
        width: usize,
        height: usize,
        has_color: bool,
    ) -> io::Result<Self> {
        Self::create_with_color_flag(path, intrinsics, width, height, has_color)
    }

    fn create_with_color_flag(
        path: &Path,
        intrinsics: Intrinsics,
        width: usize,
        height: usize,
        has_color: bool,
    ) -> io::Result<Self> {
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(MAGIC)?;
        let version = if has_color {
            VERSION_WITH_COLOR
        } else {
            VERSION_DEPTH_ONLY
        };
        writer.write_all(&version.to_le_bytes())?;
        writer.write_all(&(width as u32).to_le_bytes())?;
        writer.write_all(&(height as u32).to_le_bytes())?;
        // Intrinsics are stored so that replay needs no sensor and no hardcoded
        // calibration: they are per-device factory values.
        for value in [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy] {
            writer.write_all(&value.to_le_bytes())?;
        }

        Ok(Self {
            writer,
            width,
            height,
            count: 0,
            has_color,
        })
    }

    /// `depth` is in metres, row-major, `width * height` long.
    pub fn write_frame(&mut self, timestamp: f32, depth: &[f32]) -> io::Result<()> {
        self.write_frame_with_color(timestamp, depth, None)
    }

    /// Write depth along with optional registered colour.
    pub fn write_frame_with_color(
        &mut self,
        timestamp: f32,
        depth: &[f32],
        color: Option<FrameColor<'_>>,
    ) -> io::Result<()> {
        assert_eq!(
            depth.len(),
            self.width * self.height,
            "frame does not match the recording's dimensions"
        );

        self.writer.write_all(&timestamp.to_le_bytes())?;
        for value in depth {
            self.writer.write_all(&value.to_le_bytes())?;
        }

        if self.has_color {
            match color {
                Some(c) => {
                    assert_eq!(
                        c.rgb.len(),
                        self.width * self.height * 3,
                        "colour buffer does not match recording dimensions"
                    );
                    assert_eq!(
                        c.depth.len(),
                        self.width * self.height,
                        "colour depth buffer does not match recording dimensions"
                    );
                    assert_eq!(
                        c.valid.len(),
                        self.width * self.height,
                        "validity mask does not match recording dimensions"
                    );

                    self.writer.write_all(&[1u8])?;
                    self.writer.write_all(c.rgb)?;
                    for value in c.depth {
                        self.writer.write_all(&value.to_le_bytes())?;
                    }
                    self.writer.write_all(c.valid)?;
                    self.writer.write_all(&c.exposure.to_le_bytes())?;
                    self.writer.write_all(&c.gain.to_le_bytes())?;
                    self.writer.write_all(&c.gamma.to_le_bytes())?;
                }
                None => {
                    self.writer.write_all(&[0u8])?;
                }
            }
        }

        self.count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<usize> {
        self.writer.flush()?;
        Ok(self.count)
    }
}

/// Colour payload stored for one frame in a version 2 recording.
#[derive(Clone, Debug)]
pub struct RecordedColor {
    pub rgb: Vec<u8>,
    pub depth: Vec<f32>,
    pub valid: Vec<u8>,
    pub exposure: f32,
    pub gain: f32,
    pub gamma: f32,
}

#[derive(Debug)]
pub struct RecordedFrame {
    pub timestamp: f32,
    pub depth: Vec<f32>,
    pub color: Option<RecordedColor>,
}

#[derive(Debug)]
pub struct Recording {
    pub width: usize,
    pub height: usize,
    /// Camera intrinsics captured with the frames, so replay needs no sensor.
    pub intrinsics: Intrinsics,
    pub has_color: bool,
    pub frames: Vec<RecordedFrame>,
}

/// Read a whole recording into memory. Depth frames are under a megabyte each,
/// so a few hundred frames is unproblematic.
pub fn read(path: &Path) -> io::Result<Recording> {
    let mut reader = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a Kinect depth recording (bad magic)",
        ));
    }

    let version = read_u32(&mut reader)?;
    if version != VERSION_DEPTH_ONLY && version != VERSION_WITH_COLOR {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported recording version {version}"),
        ));
    }
    let has_color = version == VERSION_WITH_COLOR;

    let width = read_u32(&mut reader)? as usize;
    let height = read_u32(&mut reader)? as usize;
    let intrinsics = Intrinsics {
        fx: read_f32(&mut reader)?,
        fy: read_f32(&mut reader)?,
        cx: read_f32(&mut reader)?,
        cy: read_f32(&mut reader)?,
    };

    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "implausible dimensions"))?;

    let mut frames = Vec::new();
    let mut depth_buffer = vec![0u8; pixels * 4];

    loop {
        let mut timestamp_bytes = [0u8; 4];
        match reader.read_exact(&mut timestamp_bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }

        reader.read_exact(&mut depth_buffer)?;

        let depth = depth_buffer
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let color = if has_color {
            let mut flag = [0u8; 1];
            reader.read_exact(&mut flag)?;
            match flag[0] {
                0 => None,
                1 => {
                    let mut rgb = vec![0u8; pixels * 3];
                    reader.read_exact(&mut rgb)?;

                    let mut color_depth_buffer = vec![0u8; pixels * 4];
                    reader.read_exact(&mut color_depth_buffer)?;
                    let color_depth = color_depth_buffer
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();

                    let mut valid = vec![0u8; pixels];
                    reader.read_exact(&mut valid)?;

                    let exposure = read_f32(&mut reader)?;
                    let gain = read_f32(&mut reader)?;
                    let gamma = read_f32(&mut reader)?;

                    Some(RecordedColor {
                        rgb,
                        depth: color_depth,
                        valid,
                        exposure,
                        gain,
                        gamma,
                    })
                }
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("corrupt colour flag {other} in frame"),
                    ));
                }
            }
        } else {
            None
        };

        frames.push(RecordedFrame {
            timestamp: f32::from_le_bytes(timestamp_bytes),
            depth,
            color,
        });
    }

    Ok(Recording {
        width,
        height,
        intrinsics,
        has_color,
        frames,
    })
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_intrinsics() -> Intrinsics {
        Intrinsics {
            fx: 367.13,
            fy: 367.13,
            cx: 261.08,
            cy: 211.21,
        }
    }

    #[test]
    fn round_trips_frames_exactly() {
        let dir = std::env::temp_dir().join("k2df-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.k2df");

        let frames: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, f32::NAN, 0.0],
            vec![0.5, 0.25, 3.5, 4.5],
            vec![f32::INFINITY, -1.0, 7.0, 8.0],
        ];

        let mut writer = FrameWriter::create(&path, test_intrinsics(), 2, 2).expect("create");
        for (i, frame) in frames.iter().enumerate() {
            writer.write_frame(i as f32 * 0.5, frame).expect("write");
        }
        assert_eq!(writer.finish().expect("finish"), 3);

        let recording = read(&path).expect("read");
        assert_eq!(recording.width, 2);
        assert_eq!(recording.height, 2);
        assert_eq!(recording.frames.len(), 3);
        assert_eq!(recording.intrinsics, test_intrinsics());

        for (recorded, expected) in recording.frames.iter().zip(&frames) {
            // Bit-exact, including the NaN pattern, which is what makes replay a
            // usable baseline for odometry parameters.
            for (a, b) in recorded.depth.iter().zip(expected) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }

        assert_eq!(recording.frames[2].timestamp, 1.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_a_foreign_file() {
        let dir = std::env::temp_dir().join("k2df-badmagic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.k2df");
        std::fs::write(&path, b"NOPEnot a recording").unwrap();

        let error = read(&path).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_recording_has_no_frames() {
        let dir = std::env::temp_dir().join("k2df-empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.k2df");

        FrameWriter::create(&path, test_intrinsics(), 8, 8)
            .unwrap()
            .finish()
            .unwrap();

        let recording = read(&path).expect("read");
        assert_eq!(recording.width, 8);
        assert!(recording.frames.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn round_trips_frames_with_color_exactly() {
        let dir = std::env::temp_dir().join("k2df-color-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("color.k2df");

        let depth_frame = vec![1.0, 2.0, 3.0, 4.0];
        let rgb = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let color_depth = vec![1.05, 2.05, 3.05, 4.05];
        let valid = vec![1, 1, 0, 1];

        let color = FrameColor {
            rgb: &rgb,
            depth: &color_depth,
            valid: &valid,
            exposure: 0.033,
            gain: 1.5,
            gamma: 2.2,
        };

        let mut writer = FrameWriter::create_with_color(&path, test_intrinsics(), 2, 2, true).unwrap();
        // Frame 1 with colour
        writer.write_frame_with_color(0.0, &depth_frame, Some(color)).unwrap();
        // Frame 2 without colour (e.g. colour packet not arrived yet)
        writer.write_frame_with_color(0.033, &depth_frame, None).unwrap();
        assert_eq!(writer.finish().unwrap(), 2);

        let recording = read(&path).unwrap();
        assert!(recording.has_color);
        assert_eq!(recording.frames.len(), 2);

        let f0 = &recording.frames[0];
        assert_eq!(f0.timestamp, 0.0);
        let c0 = f0.color.as_ref().expect("frame 0 has color");
        assert_eq!(c0.rgb, rgb);
        assert_eq!(c0.valid, valid);
        assert_eq!(c0.exposure, 0.033);
        assert_eq!(c0.gain, 1.5);
        assert_eq!(c0.gamma, 2.2);
        for (a, b) in c0.depth.iter().zip(&color_depth) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        let f1 = &recording.frames[1];
        assert_eq!(f1.timestamp, 0.033);
        assert!(f1.color.is_none());

        std::fs::remove_file(&path).ok();
    }
}
