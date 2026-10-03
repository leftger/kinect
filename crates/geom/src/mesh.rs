//! Triangle meshes and binary PLY output.

use nalgebra::Vector3;
use std::fs::File;
use std::io::{self, BufWriter, Write};
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

    pub fn centroid(&self, triangle: &[u32; 3]) -> Vector3<f32> {
        let a = self.vertices[triangle[0] as usize];
        let b = self.vertices[triangle[1] as usize];
        let c = self.vertices[triangle[2] as usize];
        (a + b + c) / 3.0
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
    /// coordinates in its core format, so a proper texture atlas would need OBJ
    /// or glTF rather than an extension here.
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
}
