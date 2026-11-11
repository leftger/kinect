//! Triangle meshes and PLY input/output.

use nalgebra::Vector3;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// An indexed triangle mesh.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vector3<f32>>,
    pub triangles: Vec<[u32; 3]>,
    /// Per-vertex RGB, one entry per vertex, or `None` for an uncoloured mesh.
    ///
    /// Not fused into the TSDF: colouring the finished mesh lets every frame that
    /// saw a surface contribute, not just the ones that arrived while it was
    /// being integrated. See `crate::coloring`.
    pub colors: Option<Vec<[u8; 3]>>,
}

impl Mesh {
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Axis-aligned bounds, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<(Vector3<f32>, Vector3<f32>)> {
        let mut iter = self.vertices.iter();
        let first = *iter.next()?;

        let mut min = first;
        let mut max = first;
        for v in iter {
            min = min.inf(v);
            max = max.sup(v);
        }

        Some((min, max))
    }

    pub fn triangle_normal(&self, triangle: &[u32; 3]) -> Vector3<f32> {
        let a = self.vertices[triangle[0] as usize];
        let b = self.vertices[triangle[1] as usize];
        let c = self.vertices[triangle[2] as usize];
        (b - a).cross(&(c - a))
    }

    /// Area-weighted vertex normals, one per vertex.
    ///
    /// Each triangle contributes its unnormalized cross product. That vector's
    /// length is twice the triangle's area, so a large face pulls a shared
    /// vertex harder than a small one. The sum is then normalized. A vertex
    /// with no finite incident area — unused, or lying only on a degenerate
    /// face — stays at zero. Colouring treats that zero as "no orientation"
    /// rather than as a direction.
    pub fn vertex_normals(&self) -> Vec<Vector3<f32>> {
        let mut normals = vec![Vector3::zeros(); self.vertices.len()];

        for triangle in &self.triangles {
            let face = self.triangle_normal(triangle);
            if !face.x.is_finite() || !face.y.is_finite() || !face.z.is_finite() {
                continue;
            }
            for index in triangle {
                normals[*index as usize] += face;
            }
        }

        for normal in &mut normals {
            let length = normal.norm();
            if length > 1e-12 && length.is_finite() {
                *normal /= length;
            } else {
                *normal = Vector3::zeros();
            }
        }

        normals
    }

    pub fn centroid(&self, triangle: &[u32; 3]) -> Vector3<f32> {
        let a = self.vertices[triangle[0] as usize];
        let b = self.vertices[triangle[1] as usize];
        let c = self.vertices[triangle[2] as usize];
        (a + b + c) / 3.0
    }

    /// Invert the horizontal axis and reverse face winding.
    ///
    /// The Kinect v2 sensor reads out mirrored frames. Negating the X coordinate
    /// un-mirrors the reconstruction, while reversing triangle vertex winding
    /// ([a, b, c] -> [a, c, b]) keeps surface normals pointing outward.
    pub fn flip_x(&mut self) {
        for v in &mut self.vertices {
            v.x = -v.x;
        }
        for tri in &mut self.triangles {
            tri.swap(1, 2);
        }
    }

    pub fn save_ply(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_ply(&mut writer)?;
        writer.flush()
    }

    /// Binary little-endian PLY with vertices and triangle faces.
    ///
    /// Per-vertex colour is written as `uchar red/green/blue` when the mesh has
    /// one entry per vertex. Those are the conventional property names, which is
    /// what MeshLab, CloudCompare and Blender read. PLY has no texture
    /// coordinates in its core format, so an atlas goes out through
    /// [`crate::export`] as OBJ or glTF instead of an extension here.
    pub fn write_ply<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let colored = self
            .colors
            .as_ref()
            .is_some_and(|colors| colors.len() == self.vertices.len());

        write!(
            writer,
            "ply\nformat binary_little_endian 1.0\n\
             element vertex {}\n\
             property float x\nproperty float y\nproperty float z\n",
            self.vertices.len()
        )?;

        if colored {
            write!(
                writer,
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            )?;
        }

        write!(
            writer,
            "element face {}\n\
             property list uchar int vertex_indices\n\
             end_header\n",
            self.triangles.len()
        )?;

        for (index, v) in self.vertices.iter().enumerate() {
            for c in [v.x, v.y, v.z] {
                writer.write_all(&c.to_le_bytes())?;
            }
            if colored {
                if let Some(colors) = &self.colors {
                    writer.write_all(&colors[index])?;
                }
            }
        }

        for triangle in &self.triangles {
            writer.write_all(&[3u8])?;
            for index in triangle {
                writer.write_all(&(*index as i32).to_le_bytes())?;
            }
        }

        Ok(())
    }
}

/// Binary little-endian PLY containing just points.
pub fn write_points_ply<W: Write>(writer: &mut W, points: &[Vector3<f32>]) -> io::Result<()> {
    write!(
        writer,
        "ply\nformat binary_little_endian 1.0\n\
         element vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         end_header\n",
        points.len()
    )?;

    for p in points {
        for c in [p.x, p.y, p.z] {
            writer.write_all(&c.to_le_bytes())?;
        }
    }

    Ok(())
}

pub fn save_points_ply(path: &Path, points: &[Vector3<f32>]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_points_ply(&mut writer, points)?;
    writer.flush()
}

/// Binary little-endian PLY of points, optionally with per-point colour.
///
/// Deliberately *not* the same shape as [`Mesh::write_ply`]: there is no `face`
/// element, because a consumer that wants a point cloud rather than a surface
/// keys off the absence of one. Colour goes out as `uchar red/green/blue`, the
/// conventional names.
///
/// The property set matters to Gaussian-splatting loaders. They read `x/y/z`,
/// take `red/green/blue` as the colour, and leave anything they do not find at
/// their own defaults. Writing Gaussian attributes here — `scale_0`, `opacity`,
/// `rot_0`, `f_dc_*` — would replace those defaults with zeros, which is worse
/// than saying nothing.
pub fn write_colored_points_ply<W: Write>(
    writer: &mut W,
    points: &[Vector3<f32>],
    colors: Option<&[[u8; 3]]>,
) -> io::Result<()> {
    let colored = colors.is_some_and(|colors| colors.len() == points.len());

    write!(
        writer,
        "ply\nformat binary_little_endian 1.0\n\
         element vertex {}\n\
         property float x\nproperty float y\nproperty float z\n",
        points.len()
    )?;

    if colored {
        write!(
            writer,
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )?;
    }

    write!(writer, "end_header\n")?;

    for (index, p) in points.iter().enumerate() {
        for c in [p.x, p.y, p.z] {
            writer.write_all(&c.to_le_bytes())?;
        }
        if colored {
            if let Some(colors) = colors {
                writer.write_all(&colors[index])?;
            }
        }
    }

    Ok(())
}

pub fn save_colored_points_ply(
    path: &Path,
    points: &[Vector3<f32>],
    colors: Option<&[[u8; 3]]>,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_colored_points_ply(&mut writer, points, colors)?;
    writer.flush()
}

/// Read vertex positions from an ASCII or binary little-endian PLY.
///
/// Only the `vertex` element is used; faces, colours, normals and any other
/// properties are skipped. This is the input side of `scan localize`, not a
/// general-purpose PLY importer.
pub fn read_points_ply(path: &Path) -> io::Result<Vec<Vector3<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    let mut format: Option<String> = None;
    let mut vertex_count = 0usize;
    let mut current_element = String::new();
    let mut properties: Vec<(String, String)> = Vec::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(invalid("PLY header has no end_header"));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("comment") || trimmed.starts_with("obj_info") {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts.as_slice() {
            ["ply"] => {}
            ["format", kind, _] => format = Some((*kind).to_string()),
            ["element", element, count] => {
                current_element = (*element).to_string();
                if *element == "vertex" {
                    vertex_count = count
                        .parse()
                        .map_err(|_| invalid("bad vertex count in PLY header"))?;
                }
            }
            ["property", "list", ..] => {
                if current_element == "vertex" {
                    return Err(invalid("list properties are not supported on vertices"));
                }
            }
            ["property", ty, name] => {
                if current_element == "vertex" {
                    properties.push(((*ty).to_string(), (*name).to_string()));
                }
            }
            ["end_header"] => break,
            _ => {}
        }
    }

    let format = format.ok_or_else(|| invalid("PLY has no format line"))?;
    if vertex_count == 0 {
        return Ok(Vec::new());
    }
    let x = property_index(&properties, "x")?;
    let y = property_index(&properties, "y")?;
    let z = property_index(&properties, "z")?;

    match format.as_str() {
        "ascii" => {
            let mut points = Vec::with_capacity(vertex_count);
            let mut index = 0;
            while index < vertex_count {
                line.clear();
                if reader.read_line(&mut line)? == 0 {
                    return Err(invalid("PLY ended before all vertices"));
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let values: Vec<&str> = trimmed.split_whitespace().collect();
                let parse = |column: usize| -> io::Result<f32> {
                    values
                        .get(column)
                        .and_then(|value| value.parse::<f32>().ok())
                        .ok_or_else(|| invalid("bad ASCII vertex in PLY"))
                };
                points.push(Vector3::new(parse(x)?, parse(y)?, parse(z)?));
                index += 1;
            }
            Ok(points)
        }
        "binary_little_endian" => {
            let mut offsets = Vec::with_capacity(properties.len());
            let mut stride = 0usize;
            for (ty, _) in &properties {
                offsets.push(stride);
                stride += property_size(ty)?;
            }
            let bytes = stride
                .checked_mul(vertex_count)
                .ok_or_else(|| invalid("PLY vertex block is too large"))?;
            let mut block = vec![0u8; bytes];
            reader.read_exact(&mut block)?;

            let mut points = Vec::with_capacity(vertex_count);
            for index in 0..vertex_count {
                let base = index * stride;
                points.push(Vector3::new(
                    read_binary_f32(&block, base + offsets[x], &properties[x].0)?,
                    read_binary_f32(&block, base + offsets[y], &properties[y].0)?,
                    read_binary_f32(&block, base + offsets[z], &properties[z].0)?,
                ));
            }
            Ok(points)
        }
        "binary_big_endian" => Err(invalid("big-endian PLY is not supported")),
        other => Err(invalid(&format!("unsupported PLY format {other}"))),
    }
}

fn property_index(properties: &[(String, String)], name: &str) -> io::Result<usize> {
    properties
        .iter()
        .position(|(_, property)| property == name)
        .ok_or_else(|| invalid(&format!("PLY vertex has no {name} property")))
}

fn property_size(ty: &str) -> io::Result<usize> {
    match ty {
        "char" | "uchar" | "int8" | "uint8" => Ok(1),
        "short" | "ushort" | "int16" | "uint16" => Ok(2),
        "int" | "uint" | "int32" | "uint32" | "float" | "float32" => Ok(4),
        "double" | "float64" => Ok(8),
        other => Err(invalid(&format!("unsupported PLY property type {other}"))),
    }
}

fn read_binary_f32(block: &[u8], offset: usize, ty: &str) -> io::Result<f32> {
    let bytes = block
        .get(offset..)
        .ok_or_else(|| invalid("PLY vertex is truncated"))?;
    match ty {
        "float" | "float32" => {
            let bytes: [u8; 4] = bytes
                .get(..4)
                .and_then(|bytes| bytes.try_into().ok())
                .ok_or_else(|| invalid("truncated PLY float"))?;
            Ok(f32::from_le_bytes(bytes))
        }
        "double" | "float64" => {
            let bytes: [u8; 8] = bytes
                .get(..8)
                .and_then(|bytes| bytes.try_into().ok())
                .ok_or_else(|| invalid("truncated PLY double"))?;
            Ok(f64::from_le_bytes(bytes) as f32)
        }
        other => Err(invalid(&format!("{other} is not a coordinate type"))),
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetrahedron() -> Mesh {
        Mesh {
            colors: None,
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            triangles: vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        }
    }

    #[test]
    fn bounds_cover_all_vertices() {
        let (min, max) = tetrahedron().bounds().expect("non-empty");
        assert_eq!(min, Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(max, Vector3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn empty_mesh_has_no_bounds() {
        assert!(Mesh::default().bounds().is_none());
        assert!(Mesh::default().is_empty());
    }

    #[test]
    fn writes_a_readable_ply_header() {
        let mut buffer = Vec::new();
        tetrahedron().write_ply(&mut buffer).expect("write");

        let header_end = buffer
            .windows(11)
            .position(|w| w == b"end_header\n")
            .expect("header")
            + 11;

        let header = std::str::from_utf8(&buffer[..header_end]).expect("utf8 header");
        assert!(header.contains("element vertex 4"));
        assert!(header.contains("element face 4"));
        assert!(header.contains("binary_little_endian"));

        // 4 vertices * 12 bytes + 4 faces * (1 + 12) bytes
        assert_eq!(buffer.len() - header_end, 4 * 12 + 4 * 13);
    }

    #[test]
    fn point_ply_records_every_point() {
        let points = vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(-1.0, -2.0, -3.0)];
        let mut buffer = Vec::new();
        write_points_ply(&mut buffer, &points).expect("write");

        // Only the header is text; the vertices after it are binary.
        let header_end = buffer
            .windows(11)
            .position(|w| w == b"end_header\n")
            .expect("header")
            + 11;
        let header = std::str::from_utf8(&buffer[..header_end]).expect("utf8 header");

        assert!(header.contains("element vertex 2"));
        assert_eq!(buffer.len() - header_end, 2 * 12);
    }

    #[test]
    fn colored_point_ply_carries_rgb_and_no_face_element() {
        let points = vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(-4.0, 5.0, -6.0)];
        let colors = vec![[9u8, 8, 7], [6, 5, 4]];

        let mut buffer = Vec::new();
        write_colored_points_ply(&mut buffer, &points, Some(&colors)).expect("write");

        let header_end = buffer
            .windows(11)
            .position(|w| w == b"end_header\n")
            .expect("header")
            + 11;
        let header = std::str::from_utf8(&buffer[..header_end]).expect("utf8 header");

        assert!(header.contains("element vertex 2"));
        assert!(header.contains("property uchar red"));
        assert!(
            !header.contains("element face"),
            "a point cloud must not advertise faces: {header}"
        );

        // Two points, 12 bytes of position and 3 of colour each.
        assert_eq!(buffer.len() - header_end, 2 * 15);
        assert_eq!(&buffer[header_end + 12..header_end + 15], &[9, 8, 7]);
        assert_eq!(&buffer[header_end + 27..header_end + 30], &[6, 5, 4]);
    }

    #[test]
    fn colored_point_ply_drops_a_mismatched_colour_buffer() {
        // A colour list that does not line up with the points would silently
        // shift every sample by one. Fall back to uncoloured instead.
        let points = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0)];
        let colors = vec![[1u8, 2, 3]];

        let mut buffer = Vec::new();
        write_colored_points_ply(&mut buffer, &points, Some(&colors)).expect("write");

        // Only the header is text; the vertices after it are binary.
        let header_end = buffer
            .windows(11)
            .position(|w| w == b"end_header\n")
            .expect("header")
            + 11;
        let header = std::str::from_utf8(&buffer[..header_end]).expect("utf8 header");

        assert!(header.contains("element vertex 2"));
        assert!(!header.contains("property uchar red"));
        // Uncoloured points are 12 bytes each, with no colour bytes appended.
        assert_eq!(buffer.len() - header_end, 2 * 12);
    }

    #[test]
    fn vertex_normals_are_unit_and_follow_the_face() {
        let mesh = Mesh {
            colors: None,
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(2.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
            triangles: vec![[0, 1, 2]],
        };

        let normals = mesh.vertex_normals();
        for normal in &normals {
            assert!((normal.norm() - 1.0).abs() < 1e-5, "{normal:?}");
            assert!(normal.z > 0.99, "winding should face +z, got {normal:?}");
        }
    }

    #[test]
    fn a_larger_face_wins_the_shared_vertex() {
        // v0 is shared. The +z triangle has cross-product length 16; the
        // opposing one has length 1. Equal weighting would nearly cancel, and
        // area weighting must keep the sign of the large face.
        let mesh = Mesh {
            colors: None,
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(4.0, 0.0, 0.0),
                Vector3::new(0.0, 4.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
            triangles: vec![[0, 1, 2], [0, 4, 3]],
        };

        let normals = mesh.vertex_normals();
        assert!(normals[0].z > 0.99, "shared vertex {:?}", normals[0]);
        assert!(normals[1].z > 0.99 && normals[2].z > 0.99);
        assert!(normals[3].z < -0.99 && normals[4].z < -0.99);
    }

    #[test]
    fn equal_opposing_areas_cancel() {
        let mesh = Mesh {
            colors: None,
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
            triangles: vec![[0, 1, 2], [0, 2, 1]],
        };

        let normals = mesh.vertex_normals();
        assert!(
            normals.iter().all(|normal| normal.norm() < 1e-5),
            "{normals:?}"
        );
    }

    #[test]
    fn unused_and_degenerate_vertices_have_no_normal() {
        let mesh = Mesh {
            colors: None,
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(2.0, 0.0, 0.0),
                Vector3::new(5.0, 5.0, 5.0),
            ],
            triangles: vec![[0, 1, 2]],
        };

        let normals = mesh.vertex_normals();
        assert_eq!(normals[0], Vector3::zeros());
        assert_eq!(normals[3], Vector3::zeros());
    }

    #[test]
    fn flip_x_negates_x_and_preserves_outward_normals() {
        let mut mesh = tetrahedron();
        let normal_before = mesh.triangle_normal(&mesh.triangles[0]);
        // Triangle 0 has normal pointing with some x, y, z component.
        mesh.flip_x();

        // X coordinate of vertices should be negated
        assert_eq!(mesh.vertices[1].x, -1.0);

        // Triangle 0's normal should have negated x, but same y and z
        let normal_after = mesh.triangle_normal(&mesh.triangles[0]);
        assert!((normal_after.x - (-normal_before.x)).abs() < 1e-6);
        assert!((normal_after.y - normal_before.y).abs() < 1e-6);
        assert!((normal_after.z - normal_before.z).abs() < 1e-6);
    }

    #[test]
    fn points_ply_round_trips_through_the_reader() {
        let path = std::env::temp_dir().join("scan-points-reader.ply");
        let points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, -2.0, 3.5),
            Vector3::new(-0.25, 0.5, 4.0),
        ];
        save_points_ply(&path, &points).expect("write");
        let read = read_points_ply(&path).expect("read");
        assert_eq!(read.len(), points.len());
        for (actual, expected) in read.iter().zip(&points) {
            assert!((actual - expected).norm() < 1e-6);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn reads_ascii_points_and_ignores_extra_properties() {
        let path = std::env::temp_dir().join("scan-ascii-points.ply");
        let text = "ply\nformat ascii 1.0\n\
                    element vertex 2\n\
                    property float x\nproperty float y\nproperty float z\n\
                    property uchar red\nproperty uchar green\nproperty uchar blue\n\
                    end_header\n\
                    0.0 1.0 2.0 255 0 0\n\
                    -1.0 0.5 3.0 0 255 0\n";
        std::fs::write(&path, text).expect("write");
        let read = read_points_ply(&path).expect("read");
        assert_eq!(read.len(), 2);
        assert!((read[0] - Vector3::new(0.0, 1.0, 2.0)).norm() < 1e-6);
        assert!((read[1] - Vector3::new(-1.0, 0.5, 3.0)).norm() < 1e-6);
        std::fs::remove_file(&path).ok();
    }
}
