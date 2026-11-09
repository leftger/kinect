//! Load a trained 3D Gaussian-splatting PLY.
//!
//! Brush exports the usual 3DGS vertex layout: position, log scales, opacity,
//! quaternion, the DC term of the colour SH, then the higher-order SH bands.
//! The viewer only needs the DC colour and a circular screen-space footprint,
//! so the higher bands are skipped rather than decoded. This is deliberately a
//! basic viewer: it proves the trained file can be displayed beside the mesh,
//! not that every anisotropic Gaussian is reproduced exactly.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use nalgebra::Vector3;

/// `1 / sqrt(4 pi)`, the SH DC scale used by 3DGS.
const SH_C0: f32 = 0.282_094_8;

/// One splat after the PLY's log-scale and logit-opacity fields are decoded.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Splat {
    pub position: Vector3<f32>,
    /// Largest linear axis, used as an isotropic screen-space radius.
    pub scale: f32,
    pub opacity: f32,
    pub color: [f32; 3],
}

/// A loaded splat cloud and its world-space bounds.
#[derive(Debug)]
pub(crate) struct SplatCloud {
    pub splats: Vec<Splat>,
    pub bounds: (Vector3<f32>, Vector3<f32>),
}

impl SplatCloud {
    pub(crate) fn len(&self) -> usize {
        self.splats.len()
    }

    /// Parse a binary little-endian Brush/3DGS PLY.
    ///
    /// Only `property float` fields are accepted. That is what Brush writes;
    /// a different trainer that mixes integer or double fields should fail
    /// loudly rather than silently reading shifted bytes.
    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let file =
            File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let mut reader = BufReader::new(file);

        let mut line = String::new();
        let mut vertex_count = None;
        let mut properties = Vec::new();
        let mut in_vertex = false;

        loop {
            line.clear();
            let read = reader
                .read_line(&mut line)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if read == 0 {
                return Err("PLY header ended before `end_header`".to_string());
            }

            let trimmed = line.trim();
            if trimmed == "end_header" {
                break;
            }
            if let Some(format) = trimmed.strip_prefix("format ") {
                if !format.starts_with("binary_little_endian") {
                    return Err(format!("unsupported PLY format `{format}`"));
                }
            } else if let Some(rest) = trimmed.strip_prefix("element ") {
                in_vertex = rest.starts_with("vertex ");
                if let Some(count) = rest.strip_prefix("vertex ") {
                    vertex_count = Some(
                        count
                            .parse::<usize>()
                            .map_err(|error| format!("bad vertex count `{count}`: {error}"))?,
                    );
                }
            } else if let Some(name) = trimmed.strip_prefix("property float ") {
                if !in_vertex {
                    return Err("only vertex properties are supported".to_string());
                }
                properties.push(name.to_string());
            } else if trimmed.starts_with("property ") && in_vertex {
                return Err(format!(
                    "only float vertex properties are supported, found `{trimmed}`"
                ));
            }
        }

        let vertex_count = vertex_count.ok_or_else(|| "PLY has no vertex element".to_string())?;
        if properties.is_empty() {
            return Err("PLY vertex has no properties".to_string());
        }

        let offset = |name: &str| {
            properties
                .iter()
                .position(|property| property == name)
                .ok_or_else(|| format!("PLY vertex has no `{name}` property"))
        };
        let x = offset("x")?;
        let y = offset("y")?;
        let z = offset("z")?;
        let scale = [offset("scale_0")?, offset("scale_1")?, offset("scale_2")?];
        let opacity = offset("opacity")?;
        let color = [offset("f_dc_0")?, offset("f_dc_1")?, offset("f_dc_2")?];

        let stride = properties.len() * 4;
        let expected = vertex_count
            .checked_mul(stride)
            .ok_or_else(|| "PLY vertex data is too large".to_string())?;
        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .map_err(|error| format!("cannot read vertex data: {error}"))?;
        if data.len() < expected {
            return Err(format!(
                "PLY vertex data is {} bytes, expected {expected}",
                data.len()
            ));
        }

        let float = |base: usize, index: usize| -> f32 {
            let start = base + index * 4;
            f32::from_le_bytes([
                data[start],
                data[start + 1],
                data[start + 2],
                data[start + 3],
            ])
        };

        let mut splats = Vec::with_capacity(vertex_count);
        let mut min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for vertex in 0..vertex_count {
            let base = vertex * stride;
            let position = Vector3::new(float(base, x), float(base, y), float(base, z));
            let log_scales = [
                float(base, scale[0]),
                float(base, scale[1]),
                float(base, scale[2]),
            ];
            let linear_scale = log_scales
                .into_iter()
                .map(f32::exp)
                .filter(|value| value.is_finite() && *value > 0.0)
                .fold(0.0f32, f32::max);
            let logit = float(base, opacity);
            let opacity = 1.0 / (1.0 + (-logit).exp());
            let rgb = [
                (0.5 + SH_C0 * float(base, color[0])).clamp(0.0, 1.0),
                (0.5 + SH_C0 * float(base, color[1])).clamp(0.0, 1.0),
                (0.5 + SH_C0 * float(base, color[2])).clamp(0.0, 1.0),
            ];

            if !position.iter().all(|value| value.is_finite())
                || !opacity.is_finite()
                || linear_scale <= 0.0
                || !linear_scale.is_finite()
            {
                continue;
            }

            min = min.inf(&position);
            max = max.sup(&position);
            splats.push(Splat {
                position,
                scale: linear_scale,
                opacity: opacity.clamp(0.0, 1.0),
                color: rgb,
            });
        }

        if splats.is_empty() {
            return Err("PLY contains no usable splats".to_string());
        }

        Ok(Self {
            splats,
            bounds: (min, max),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_file(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("scan-splat-{label}-{}", std::process::id()))
    }

    fn write_ply(path: &Path, fields: &[f32]) {
        let header = "\
ply
format binary_little_endian 1.0
element vertex 1
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
";
        let mut file = fs::File::create(path).unwrap();
        file.write_all(header.as_bytes()).unwrap();
        for value in fields {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn loads_the_dc_colour_scale_and_opacity() {
        let path = temp_file("load");
        write_ply(&path, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let cloud = SplatCloud::load(&path).expect("load");

        assert_eq!(cloud.len(), 1);
        let splat = cloud.splats[0];
        assert!((splat.position - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-6);
        assert!((splat.scale - 1.0).abs() < 1e-6);
        assert!((splat.opacity - 0.5).abs() < 1e-6);
        assert!((splat.color[0] - 0.5).abs() < 1e-6);
        assert!((cloud.bounds.0 - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-6);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn rejects_a_missing_property() {
        let path = temp_file("missing");
        let header = "\
ply
format binary_little_endian 1.0
element vertex 1
property float x
end_header
";
        fs::write(&path, header).unwrap();
        let error = SplatCloud::load(&path).expect_err("missing fields");
        assert!(error.contains("property"), "{error}");
        let _ = fs::remove_file(path);
    }

    #[test]
    #[ignore = "set SCAN_SPLAT_PLY to a trained Brush export"]
    fn loads_a_real_brush_export() {
        use std::time::Instant;

        let path = std::env::var("SCAN_SPLAT_PLY").expect("SCAN_SPLAT_PLY");
        let started = Instant::now();
        let cloud = SplatCloud::load(Path::new(&path)).expect("load Brush PLY");
        println!(
            "{} splats in {:?}; bounds {:?} .. {:?}",
            cloud.len(),
            started.elapsed(),
            cloud.bounds.0,
            cloud.bounds.1
        );
        assert!(cloud.len() > 1_000);
    }
}
