//! Mesh files selected by extension: `.ply`, `.obj`, `.gltf`, and `.glb`.
//!
//! PLY stays the binary vertex-colour file [`crate::mesh::Mesh::write_ply`]
//! already writes. The other three can carry a texture atlas. OBJ and glTF
//! keep the image beside the mesh; GLB puts the geometry and the PNG in the
//! one binary container. A mesh with no atlas is still a valid file of whichever
//! of those three was asked for: positions, normals and indices, and nothing
//! that points at a missing texture.
//!
//! Coordinates are the mesh's own. PLY, OBJ and glTF therefore describe the
//! same scan. glTF's Y-up convention is not applied on the way out, or the
//! formats would disagree about which way is up.
//!
//! Texture coordinates are the atlas's: `(0, 0)` is the top-left pixel and V
//! grows downward, which is also glTF's rule, so those files store the UVs
//! unchanged. OBJ readers treat V as growing upward, so that writer flips V
//! and only V. The PNG is the same image either way.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use nalgebra::Vector3;

use crate::mesh::Mesh;
use crate::png::encode_rgb8;
use crate::texturing::TexturedMesh;

const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const TRIANGLES: u32 = 4;
const LINEAR: u32 = 9729;
const CLAMP_TO_EDGE: u32 = 33071;

/// Which container [`save_mesh`] or [`save_textured`] will write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshFormat {
    Ply,
    Obj,
    Gltf,
    Glb,
}

impl MeshFormat {
    /// The format named by `path`'s extension, ignoring ASCII case.
    pub fn from_path(path: &Path) -> Result<Self, ExportError> {
        let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
            return Err(ExportError::new(
                "output path has no extension; expected .ply, .obj, .gltf, or .glb",
            ));
        };
        if extension.is_empty() {
            return Err(ExportError::new(
                "output path has no extension; expected .ply, .obj, .gltf, or .glb",
            ));
        }
        match extension.to_ascii_lowercase().as_str() {
            "ply" => Ok(Self::Ply),
            "obj" => Ok(Self::Obj),
            "gltf" => Ok(Self::Gltf),
            "glb" => Ok(Self::Glb),
            _ => Err(ExportError::new(format!(
                "unsupported mesh extension \".{extension}\"; expected .ply, .obj, .gltf, or .glb"
            ))),
        }
    }

    /// OBJ, glTF and GLB can reference an atlas. PLY cannot, and keeps the
    /// per-vertex colour it already writes.
    pub fn uses_texture_atlas(self) -> bool {
        matches!(self, Self::Obj | Self::Gltf | Self::Glb)
    }
}

/// Why a mesh file could not be written.
#[derive(Debug)]
pub struct ExportError {
    message: String,
}

impl ExportError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ExportError {}

/// Write `mesh` as the format named by `path`'s extension.
///
/// Per-vertex colour is written only for PLY. OBJ, glTF and GLB from this
/// function are geometry: positions, normals and indices. An atlas is
/// [`save_textured`].
pub fn save_mesh(mesh: &Mesh, path: &Path) -> Result<(), ExportError> {
    let format = MeshFormat::from_path(path)?;
    if format == MeshFormat::Ply {
        return mesh
            .save_ply(path)
            .map_err(|err| ExportError::new(format!("failed to write {}: {err}", path.display())));
    }

    let mut indices = Vec::with_capacity(mesh.triangles.len() * 3);
    for triangle in &mesh.triangles {
        indices.extend(triangle);
    }
    // Normals walk every corner. An index past the vertex buffer has to be
    // reported here, before that walk, rather than as a panic.
    reject_bad_indices(mesh.vertices.len(), &indices)?;
    let normals = mesh.vertex_normals();
    write_geometry(format, path, &mesh.vertices, &normals, None, &indices, None)
}

/// Write a textured mesh as `.obj`, `.gltf` or `.glb`.
///
/// `.obj` also writes a sibling `.mtl` and `.png`. `.gltf` writes a sibling
/// `.bin` and `.png`. `.glb` keeps both the geometry and the PNG inside the
/// one file. `.ply` is refused: that format has no place for this atlas.
pub fn save_textured(mesh: &TexturedMesh, path: &Path) -> Result<(), ExportError> {
    let format = MeshFormat::from_path(path)?;
    if format == MeshFormat::Ply {
        return Err(ExportError::new(
            "PLY cannot store a texture atlas; use .obj, .gltf, or .glb",
        ));
    }
    let (width, height) = atlas_size(mesh)?;
    let png = encode_rgb8(width, height, &mesh.atlas)
        .map_err(|err| ExportError::new(format!("failed to encode the texture atlas: {err}")))?;
    write_geometry(
        format,
        path,
        &mesh.positions,
        &mesh.normals,
        Some(&mesh.uvs),
        &mesh.indices,
        Some(&png),
    )
}

fn write_geometry(
    format: MeshFormat,
    path: &Path,
    positions: &[Vector3<f32>],
    normals: &[Vector3<f32>],
    uvs: Option<&[[f32; 2]]>,
    indices: &[u32],
    png: Option<&[u8]>,
) -> Result<(), ExportError> {
    validate_indexed(
        positions.len(),
        normals.len(),
        uvs.map(|uv| uv.len()),
        indices,
    )?;
    if uvs.is_some() != png.is_some() {
        return Err(ExportError::new(
            "a textured mesh needs both texture coordinates and an atlas image",
        ));
    }

    match format {
        MeshFormat::Ply => Err(ExportError::new(
            "PLY cannot store a texture atlas; use .obj, .gltf, or .glb",
        )),
        MeshFormat::Obj => write_obj(path, positions, normals, uvs, indices, png),
        MeshFormat::Gltf => write_gltf_folder(path, positions, normals, uvs, indices, png),
        MeshFormat::Glb => {
            let packed = pack_geometry(positions, normals, uvs, indices, png);
            let json = gltf_json(&packed, None, None)?;
            let bytes = wrap_glb(&json, &packed.bin)?;
            write_bytes(path, &bytes)
        }
    }
}

fn validate_indexed(
    vertex_count: usize,
    normal_count: usize,
    uv_count: Option<usize>,
    indices: &[u32],
) -> Result<(), ExportError> {
    if normal_count != vertex_count {
        return Err(ExportError::new(format!(
            "mesh has {normal_count} normals for {vertex_count} vertices"
        )));
    }
    if let Some(uv_count) = uv_count {
        if uv_count != vertex_count {
            return Err(ExportError::new(format!(
                "mesh has {uv_count} texture coordinates for {vertex_count} vertices"
            )));
        }
    }
    reject_bad_indices(vertex_count, indices)
}

fn reject_bad_indices(vertex_count: usize, indices: &[u32]) -> Result<(), ExportError> {
    if indices.len() % 3 != 0 {
        return Err(ExportError::new(
            "triangle indices are not a multiple of three",
        ));
    }
    if let Some(index) = indices
        .iter()
        .copied()
        .find(|index| *index as usize >= vertex_count)
    {
        return Err(ExportError::new(format!(
            "triangle index {index} is past the {vertex_count} vertices"
        )));
    }
    Ok(())
}

fn atlas_size(mesh: &TexturedMesh) -> Result<(u32, u32), ExportError> {
    let width = u32::try_from(mesh.atlas_width)
        .map_err(|_| ExportError::new("texture atlas width does not fit in a png header"))?;
    let height = u32::try_from(mesh.atlas_height)
        .map_err(|_| ExportError::new("texture atlas height does not fit in a png header"))?;
    if width == 0 || height == 0 {
        return Err(ExportError::new("texture atlas is empty"));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| ExportError::new("texture atlas dimensions overflow"))?;
    if mesh.atlas.len() != expected {
        return Err(ExportError::new(format!(
            "texture atlas is {width}x{height} but has {} bytes, expected {expected}",
            mesh.atlas.len()
        )));
    }
    Ok((width, height))
}

/// File name of a sibling, derived from the output's own last component.
///
/// The result is a single path component. Callers join it with
/// [`Path::with_file_name`], which cannot be pointed at another directory by
/// anything in that component. The same string is what OBJ, MTL and glTF
/// store, so the reference and the file that was written cannot drift apart.
fn sidecar_file_name(output: &Path, extension: &str) -> Result<String, ExportError> {
    let Some(file_name) = output.file_name().and_then(|name| name.to_str()) else {
        return Err(ExportError::new(
            "output path has no UTF-8 file name to derive a sidecar from",
        ));
    };
    let Some(stem) = output.file_stem().and_then(|stem| stem.to_str()) else {
        return Err(ExportError::new(
            "output path has no UTF-8 file name to derive a sidecar from",
        ));
    };
    if !stem_is_safe(stem) {
        return Err(ExportError::new(format!(
            "output basename \"{stem}\" cannot be used as a sidecar file name; use letters, digits, '.', '_' or '-'"
        )));
    }
    let own_extension = output
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    if file_name != format!("{stem}.{own_extension}") {
        return Err(ExportError::new(format!(
            "output file name \"{file_name}\" cannot be used as a sidecar file name"
        )));
    }
    if extension.is_empty() || !extension.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
        return Err(ExportError::new(
            "internal sidecar extension is not a file suffix",
        ));
    }
    Ok(format!("{stem}.{extension}"))
}

fn stem_is_safe(stem: &str) -> bool {
    if stem.is_empty() || stem == "." || stem == ".." {
        return false;
    }
    stem.bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'.' || byte == b'_' || byte == b'-')
}

fn sibling_path(output: &Path, extension: &str) -> Result<PathBuf, ExportError> {
    Ok(output.with_file_name(sidecar_file_name(output, extension)?))
}

fn write_obj(
    path: &Path,
    positions: &[Vector3<f32>],
    normals: &[Vector3<f32>],
    uvs: Option<&[[f32; 2]]>,
    indices: &[u32],
    png: Option<&[u8]>,
) -> Result<(), ExportError> {
    let mtl_name = if png.is_some() {
        Some(sidecar_file_name(path, "mtl")?)
    } else {
        None
    };
    let png_name = if png.is_some() {
        Some(sidecar_file_name(path, "png")?)
    } else {
        None
    };

    let mut obj = String::new();
    if let Some(mtl_name) = &mtl_name {
        obj.push_str("mtllib ");
        obj.push_str(mtl_name);
        obj.push('\n');
    }
    for position in positions {
        push_obj_vertex(&mut obj, "v", &[position.x, position.y, position.z]);
    }
    if let Some(uvs) = uvs {
        for uv in uvs {
            // OBJ's V axis runs bottom-up. The atlas and glTF share a top-left
            // origin, so this is the only place the coordinate is flipped.
            push_obj_vertex(&mut obj, "vt", &[uv[0], 1.0 - finite(uv[1])]);
        }
    }
    for normal in normals {
        push_obj_vertex(&mut obj, "vn", &[normal.x, normal.y, normal.z]);
    }
    if mtl_name.is_some() {
        obj.push_str("usemtl atlas\n");
    }
    for triangle in indices.chunks_exact(3) {
        obj.push('f');
        for index in triangle {
            let slot = u64::from(*index) + 1;
            obj.push(' ');
            if uvs.is_some() {
                obj.push_str(&format!("{slot}/{slot}/{slot}"));
            } else {
                obj.push_str(&format!("{slot}//{slot}"));
            }
        }
        obj.push('\n');
    }

    if let (Some(png), Some(png_name)) = (png, png_name.as_deref()) {
        let mtl_path = sibling_path(path, "mtl")?;
        let png_path = sibling_path(path, "png")?;
        write_bytes(&png_path, png)?;
        write_bytes(mtl_path.as_path(), write_mtl(png_name).as_bytes())?;
    }
    write_bytes(path, obj.as_bytes())
}

fn push_obj_vertex(out: &mut String, kind: &str, components: &[f32]) {
    out.push_str(kind);
    for component in components {
        out.push(' ');
        out.push_str(&format!("{:.6}", finite(*component)));
    }
    out.push('\n');
}

fn write_mtl(png_name: &str) -> String {
    format!(
        "newmtl atlas\n\
         Ka 1.000000 1.000000 1.000000\n\
         Kd 1.000000 1.000000 1.000000\n\
         Ks 0.000000 0.000000 0.000000\n\
         illum 2\n\
         map_Kd {png_name}\n"
    )
}

fn write_gltf_folder(
    path: &Path,
    positions: &[Vector3<f32>],
    normals: &[Vector3<f32>],
    uvs: Option<&[[f32; 2]]>,
    indices: &[u32],
    png: Option<&[u8]>,
) -> Result<(), ExportError> {
    let bin_name = sidecar_file_name(path, "bin")?;
    let png_name = if png.is_some() {
        Some(sidecar_file_name(path, "png")?)
    } else {
        None
    };
    // The PNG stays a sibling, so the geometry buffer does not embed it.
    let packed = pack_geometry(positions, normals, uvs, indices, None);
    let json = gltf_json(&packed, Some(&bin_name), png_name.as_deref())?;
    if let Some(png) = png {
        write_bytes(sibling_path(path, "png")?.as_path(), png)?;
    }
    write_bytes(sibling_path(path, "bin")?.as_path(), &packed.bin)?;
    write_bytes(path, json.as_bytes())
}

struct Section {
    offset: usize,
    length: usize,
    /// `None` for the embedded PNG. Vertex attributes and indices set the
    /// glTF buffer-view target; an image must not.
    target: Option<u32>,
}

struct Packed {
    bin: Vec<u8>,
    views: Vec<Section>,
    position: usize,
    normal: usize,
    texcoord: Option<usize>,
    indices: usize,
    image: Option<usize>,
    vertex_count: usize,
    index_count: usize,
    bounds: Option<([f32; 3], [f32; 3])>,
}

fn pack_geometry(
    positions: &[Vector3<f32>],
    normals: &[Vector3<f32>],
    uvs: Option<&[[f32; 2]]>,
    indices: &[u32],
    png: Option<&[u8]>,
) -> Packed {
    let mut bin = Vec::new();
    let mut views = Vec::new();

    let position = push_view(&mut bin, &mut views, ARRAY_BUFFER, |bin| {
        for position in positions {
            push_f32(bin, position.x);
            push_f32(bin, position.y);
            push_f32(bin, position.z);
        }
    });
    let normal = push_view(&mut bin, &mut views, ARRAY_BUFFER, |bin| {
        for normal in normals {
            push_f32(bin, normal.x);
            push_f32(bin, normal.y);
            push_f32(bin, normal.z);
        }
    });
    let texcoord = uvs.map(|uvs| {
        push_view(&mut bin, &mut views, ARRAY_BUFFER, |bin| {
            for uv in uvs {
                push_f32(bin, uv[0]);
                push_f32(bin, uv[1]);
            }
        })
    });
    let indices_index = push_view(&mut bin, &mut views, ELEMENT_ARRAY_BUFFER, |bin| {
        for index in indices {
            bin.extend_from_slice(&index.to_le_bytes());
        }
    });
    let image = png.map(|png| {
        let index = views.len();
        let offset = bin.len();
        bin.extend_from_slice(png);
        views.push(Section {
            offset,
            length: png.len(),
            target: None,
        });
        pad4(&mut bin, 0);
        index
    });
    pad4(&mut bin, 0);

    Packed {
        bin,
        views,
        position,
        normal,
        texcoord,
        indices: indices_index,
        image,
        vertex_count: positions.len(),
        index_count: indices.len(),
        bounds: position_bounds(positions),
    }
}

fn push_view(
    bin: &mut Vec<u8>,
    views: &mut Vec<Section>,
    target: u32,
    write: impl FnOnce(&mut Vec<u8>),
) -> usize {
    let index = views.len();
    let offset = bin.len();
    let start = bin.len();
    write(bin);
    views.push(Section {
        offset,
        length: bin.len() - start,
        target: Some(target),
    });
    pad4(bin, 0);
    index
}

fn push_f32(bin: &mut Vec<u8>, value: f32) {
    bin.extend_from_slice(&finite(value).to_le_bytes());
}

fn pad4(buf: &mut Vec<u8>, fill: u8) {
    while buf.len() % 4 != 0 {
        buf.push(fill);
    }
}

fn position_bounds(positions: &[Vector3<f32>]) -> Option<([f32; 3], [f32; 3])> {
    let mut iter = positions.iter();
    let first = *iter.next()?;
    let mut min = [finite(first.x), finite(first.y), finite(first.z)];
    let mut max = min;
    for position in iter {
        let value = [finite(position.x), finite(position.y), finite(position.z)];
        for axis in 0..3 {
            min[axis] = min[axis].min(value[axis]);
            max[axis] = max[axis].max(value[axis]);
        }
    }
    Some((min, max))
}

fn finite(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn gltf_json(
    packed: &Packed,
    bin_uri: Option<&str>,
    image_uri: Option<&str>,
) -> Result<String, ExportError> {
    let mut json = JsonBuf::new();
    json.begin_object();
    json.key("asset");
    json.begin_object();
    json.key("version");
    json.string("2.0");
    json.key("generator");
    json.string("kinect");
    json.end_object();

    json.key("scene");
    json.u32(0);
    json.key("scenes");
    json.begin_array();
    json.begin_object();
    json.key("nodes");
    json.begin_array();
    json.u32(0);
    json.end_array();
    json.end_object();
    json.end_array();

    json.key("nodes");
    json.begin_array();
    json.begin_object();
    json.key("mesh");
    json.u32(0);
    json.end_object();
    json.end_array();

    json.key("meshes");
    json.begin_array();
    json.begin_object();
    json.key("primitives");
    json.begin_array();
    json.begin_object();
    json.key("attributes");
    json.begin_object();
    // Accessors are emitted in the same order as these buffer views, so the
    // accessor index and the buffer-view index are the same number.
    json.key("POSITION");
    json.usize(packed.position);
    json.key("NORMAL");
    json.usize(packed.normal);
    if let Some(texcoord) = packed.texcoord {
        json.key("TEXCOORD_0");
        json.usize(texcoord);
    }
    json.end_object();
    json.key("indices");
    json.usize(packed.indices);
    if packed.texcoord.is_some() {
        json.key("material");
        json.u32(0);
    }
    json.key("mode");
    json.u32(TRIANGLES);
    json.end_object();
    json.end_array();
    json.end_object();
    json.end_array();

    json.key("accessors");
    json.begin_array();
    write_accessor(
        &mut json,
        packed.position,
        FLOAT,
        packed.vertex_count,
        "VEC3",
        packed.bounds,
    );
    write_accessor(
        &mut json,
        packed.normal,
        FLOAT,
        packed.vertex_count,
        "VEC3",
        None,
    );
    if let Some(texcoord) = packed.texcoord {
        write_accessor(
            &mut json,
            texcoord,
            FLOAT,
            packed.vertex_count,
            "VEC2",
            None,
        );
    }
    write_accessor(
        &mut json,
        packed.indices,
        UNSIGNED_INT,
        packed.index_count,
        "SCALAR",
        None,
    );
    json.end_array();

    json.key("bufferViews");
    json.begin_array();
    for view in &packed.views {
        json.begin_object();
        json.key("buffer");
        json.u32(0);
        json.key("byteOffset");
        json.usize(view.offset);
        json.key("byteLength");
        json.usize(view.length);
        if let Some(target) = view.target {
            json.key("target");
            json.u32(target);
        }
        json.end_object();
    }
    json.end_array();

    json.key("buffers");
    json.begin_array();
    json.begin_object();
    if let Some(bin_uri) = bin_uri {
        json.key("uri");
        json.string(&uri_encode(bin_uri));
    }
    json.key("byteLength");
    json.usize(packed.bin.len());
    json.end_object();
    json.end_array();

    if packed.texcoord.is_some() {
        json.key("materials");
        json.begin_array();
        json.begin_object();
        json.key("name");
        json.string("atlas");
        json.key("pbrMetallicRoughness");
        json.begin_object();
        json.key("baseColorTexture");
        json.begin_object();
        json.key("index");
        json.u32(0);
        json.end_object();
        json.key("baseColorFactor");
        json.begin_array();
        for _ in 0..4 {
            json.f32(1.0);
        }
        json.end_array();
        // The default metallic factor is 1, which would throw the atlas away
        // and shade the mesh as bare metal. Zero keeps the photographed colour.
        json.key("metallicFactor");
        json.f32(0.0);
        json.key("roughnessFactor");
        json.f32(1.0);
        json.end_object();
        json.end_object();
        json.end_array();

        json.key("textures");
        json.begin_array();
        json.begin_object();
        json.key("sampler");
        json.u32(0);
        json.key("source");
        json.u32(0);
        json.end_object();
        json.end_array();

        json.key("images");
        json.begin_array();
        json.begin_object();
        if let Some(image_uri) = image_uri {
            json.key("uri");
            json.string(&uri_encode(image_uri));
        } else if let Some(image) = packed.image {
            json.key("bufferView");
            json.usize(image);
            json.key("mimeType");
            json.string("image/png");
        } else {
            return Err(ExportError::new(
                "textured glTF has no image to attach to its material",
            ));
        }
        json.end_object();
        json.end_array();

        json.key("samplers");
        json.begin_array();
        json.begin_object();
        json.key("magFilter");
        json.u32(LINEAR);
        json.key("minFilter");
        json.u32(LINEAR);
        // Clamp, rather than repeat, so a sample on a chart border stays in
        // the gutter instead of reading the chart packed beside it.
        json.key("wrapS");
        json.u32(CLAMP_TO_EDGE);
        json.key("wrapT");
        json.u32(CLAMP_TO_EDGE);
        json.end_object();
        json.end_array();
    }

    json.end_object();
    Ok(json.finish()?)
}

fn write_accessor(
    json: &mut JsonBuf,
    buffer_view: usize,
    component_type: u32,
    count: usize,
    kind: &str,
    bounds: Option<([f32; 3], [f32; 3])>,
) {
    json.begin_object();
    json.key("bufferView");
    json.usize(buffer_view);
    json.key("byteOffset");
    json.u32(0);
    json.key("componentType");
    json.u32(component_type);
    json.key("count");
    json.usize(count);
    json.key("type");
    json.string(kind);
    if let Some((min, max)) = bounds {
        json.key("min");
        write_vec3(json, min);
        json.key("max");
        write_vec3(json, max);
    }
    json.end_object();
}

fn write_vec3(json: &mut JsonBuf, value: [f32; 3]) {
    json.begin_array();
    for component in value {
        json.f32(component);
    }
    json.end_array();
}

fn uri_encode(name: &str) -> String {
    let mut encoded = String::with_capacity(name.len());
    for &byte in name.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn wrap_glb(json: &str, bin: &[u8]) -> Result<Vec<u8>, ExportError> {
    if bin.len() % 4 != 0 {
        return Err(ExportError::new(
            "glTF binary chunk is not aligned to 4 bytes",
        ));
    }
    let mut json_bytes = json.as_bytes().to_vec();
    pad4(&mut json_bytes, b' ');
    let json_len = chunk_len(json_bytes.len(), "glTF JSON")?;
    let bin_len = chunk_len(bin.len(), "glTF binary")?;
    let total = 12usize
        .checked_add(8 + json_bytes.len())
        .and_then(|n| n.checked_add(8 + bin.len()))
        .ok_or_else(|| ExportError::new("GLB is larger than its header can describe"))?;
    let total_len = chunk_len(total, "GLB")?;

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&total_len.to_le_bytes());
    out.extend_from_slice(&json_len.to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&bin_len.to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(bin);
    Ok(out)
}

fn chunk_len(len: usize, what: &str) -> Result<u32, ExportError> {
    u32::try_from(len)
        .map_err(|_| ExportError::new(format!("{what} is larger than a glTF chunk can hold")))
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), ExportError> {
    let file = File::create(path)
        .map_err(|err| ExportError::new(format!("failed to create {}: {err}", path.display())))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .map_err(|err| ExportError::new(format!("failed to write {}: {err}", path.display())))?;
    writer
        .flush()
        .map_err(|err| ExportError::new(format!("failed to write {}: {err}", path.display())))?;
    Ok(())
}

struct JsonBuf {
    buf: String,
    /// Whether the container at this depth already holds one value.
    started: Vec<bool>,
    expect_value: bool,
}

impl JsonBuf {
    fn new() -> Self {
        Self {
            buf: String::new(),
            started: Vec::new(),
            expect_value: false,
        }
    }

    fn finish(self) -> Result<String, ExportError> {
        if !self.started.is_empty() || self.expect_value {
            return Err(ExportError::new("glTF JSON was left unclosed"));
        }
        Ok(self.buf)
    }

    fn begin_object(&mut self) {
        self.begin('{');
    }

    fn end_object(&mut self) {
        self.end('}');
    }

    fn begin_array(&mut self) {
        self.begin('[');
    }

    fn end_array(&mut self) {
        self.end(']');
    }

    fn begin(&mut self, open: char) {
        self.write_prefix();
        self.buf.push(open);
        self.started.push(false);
    }

    fn end(&mut self, close: char) {
        self.started.pop();
        self.buf.push(close);
    }

    fn key(&mut self, key: &str) {
        self.write_prefix();
        push_escaped(&mut self.buf, key);
        self.buf.push(':');
        self.expect_value = true;
    }

    fn string(&mut self, value: &str) {
        self.write_prefix();
        push_escaped(&mut self.buf, value);
    }

    fn u32(&mut self, value: u32) {
        self.usize(value as usize);
    }

    fn usize(&mut self, value: usize) {
        self.write_prefix();
        self.buf.push_str(&value.to_string());
    }

    fn f32(&mut self, value: f32) {
        self.write_prefix();
        self.buf.push_str(&json_f32(value));
    }

    fn write_prefix(&mut self) {
        if self.expect_value {
            self.expect_value = false;
            return;
        }
        if let Some(started) = self.started.last_mut() {
            if *started {
                self.buf.push(',');
            } else {
                *started = true;
            }
        }
    }
}

fn push_escaped(buf: &mut String, value: &str) {
    buf.push('"');
    for ch in value.chars() {
        match ch {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            other if other.is_control() => {
                buf.push_str(&format!("\\u{:04x}", other as u32));
            }
            other => buf.push(other),
        }
    }
    buf.push('"');
}

fn json_f32(value: f32) -> String {
    let value = finite(value);
    let rendered = format!("{value:.8}");
    let trimmed = rendered.trim_end_matches('0').trim_end_matches('.');
    if trimmed.contains('.') || trimmed.contains('e') || trimmed.contains('E') {
        trimmed.to_string()
    } else {
        format!("{trimmed}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::png::decode_rgb8;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn triangle() -> Mesh {
        Mesh {
            colors: None,
            vertices: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 2.0, 0.0),
            ],
            triangles: vec![[0, 2, 1]],
        }
    }

    fn colored_triangle() -> Mesh {
        let mut mesh = triangle();
        mesh.colors = Some(vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        mesh
    }

    fn textured_triangle() -> TexturedMesh {
        TexturedMesh {
            positions: triangle().vertices,
            normals: vec![Vector3::new(0.0, 0.0, 1.0); 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.25, 0.75]],
            indices: vec![0, 2, 1],
            atlas: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
            atlas_width: 2,
            atlas_height: 2,
        }
    }

    fn scratch(label: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "kinect-export-{label}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("temp dir");
        path
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            Self(scratch(label))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn ply_keeps_vertex_colours_and_rejects_an_unknown_extension() {
        let dir = TempDir::new("ply");
        let path = dir.path().join("scan.PLY");
        let mesh = colored_triangle();
        save_mesh(&mesh, &path).expect("ply");

        let mut expected = Vec::new();
        mesh.write_ply(&mut expected).expect("write_ply");
        let written = fs::read(&path).expect("read");
        assert_eq!(written, expected);
        let header_end = written
            .windows(11)
            .position(|window| window == b"end_header\n")
            .expect("header")
            + 11;
        let header = std::str::from_utf8(&written[..header_end]).expect("header utf8");
        assert!(header.contains("property uchar red"));
        assert!(header.contains("property uchar green"));
        assert!(header.contains("property uchar blue"));

        let error = save_mesh(&mesh, &dir.path().join("scan.stl")).expect_err("stl");
        assert!(
            error
                .to_string()
                .contains("unsupported mesh extension \".stl\""),
            "{error}"
        );
        assert!(error.to_string().contains(".ply"));
        assert!(!dir.path().join("scan.stl").exists());

        let missing = save_mesh(&mesh, &dir.path().join("scan")).expect_err("no ext");
        assert!(missing.to_string().contains("no extension"), "{missing}");
    }

    #[test]
    fn obj_sidecars_reference_only_the_output_basename() {
        let dir = TempDir::new("obj");
        let nested = dir.path().join("nested");
        fs::create_dir_all(&nested).expect("nested");
        // The `..` stays in the path we hand the writer. The sidecar has to
        // replace the file name, not follow that component into the reference.
        let out = nested.join("../my.scan.obj");
        save_textured(&textured_triangle(), &out).expect("obj");

        let obj_path = dir.path().join("my.scan.obj");
        let mtl_path = dir.path().join("my.scan.mtl");
        let png_path = dir.path().join("my.scan.png");
        let obj = fs::read_to_string(&obj_path).expect("obj text");
        let mtl = fs::read_to_string(&mtl_path).expect("mtl text");

        assert!(obj.starts_with("mtllib my.scan.mtl\n"), "{obj}");
        assert!(!obj.contains("nested"), "{obj}");
        assert!(!obj.contains(".."), "{obj}");
        assert!(obj.contains("usemtl atlas\n"));
        assert!(obj.contains("f 1/1/1 3/3/3 2/2/2\n"), "{obj}");
        assert!(obj.contains("vt 0.000000 1.000000\n"), "{obj}");
        assert!(obj.contains("vt 1.000000 1.000000\n"), "{obj}");
        assert!(obj.contains("vt 0.250000 0.250000\n"), "{obj}");
        assert!(obj.contains("vn 0.000000 0.000000 1.000000\n"));

        assert!(mtl.contains("newmtl atlas\n"), "{mtl}");
        assert!(mtl.contains("map_Kd my.scan.png\n"), "{mtl}");
        assert!(!mtl.contains('/'), "{mtl}");

        let (width, height, rgb) = decode_rgb8(&fs::read(&png_path).expect("png")).expect("decode");
        assert_eq!((width, height), (2, 2));
        assert_eq!(rgb, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
    }

    #[test]
    fn an_unsafe_basename_does_not_write_a_sidecar() {
        let dir = TempDir::new("unsafe");
        let path = dir.path().join("my scan.obj");
        let error = save_textured(&textured_triangle(), &path).expect_err("space");
        assert!(error.to_string().contains("my scan"), "{error}");
        assert!(error.to_string().contains("sidecar"), "{error}");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);

        let hashed = save_mesh(&triangle(), &dir.path().join("bad#name.gltf")).expect_err("hash");
        assert!(hashed.to_string().contains("bad#name"), "{hashed}");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn geometry_only_obj_has_no_texture_and_no_sidecars() {
        let dir = TempDir::new("obj-plain");
        let path = dir.path().join("room.obj");
        save_mesh(&triangle(), &path).expect("obj");
        let obj = fs::read_to_string(&path).expect("text");
        assert!(!obj.contains("mtllib"), "{obj}");
        assert!(!obj.contains("vt "), "{obj}");
        assert!(!obj.contains("usemtl"), "{obj}");
        assert!(obj.contains("f 1//1 3//3 2//2\n"), "{obj}");
        // Winding [0, 2, 1] faces -Z. The index order above is what proves the
        // face was not rewritten as 0, 1, 2.
        assert!(obj.contains("vn 0.000000 0.000000 -1.000000\n"));
        assert!(!dir.path().join("room.mtl").exists());
        assert!(!dir.path().join("room.png").exists());
    }

    #[test]
    fn gltf_json_describes_accessors_and_external_sidecars() {
        let dir = TempDir::new("gltf");
        let path = dir.path().join("my.scan.gltf");
        save_textured(&textured_triangle(), &path).expect("gltf");

        let json = fs::read_to_string(&path).expect("json");
        let bin = fs::read(dir.path().join("my.scan.bin")).expect("bin");
        let png = fs::read(dir.path().join("my.scan.png")).expect("png");
        let value = parse_json(&json);
        assert_eq!(value.string(&["asset", "version"]), "2.0");

        let primitive = value.index(&["meshes"], 0).index(&["primitives"], 0);
        assert_eq!(primitive.number(&["attributes", "POSITION"]), 0.0);
        assert_eq!(primitive.number(&["attributes", "NORMAL"]), 1.0);
        assert_eq!(primitive.number(&["attributes", "TEXCOORD_0"]), 2.0);
        assert_eq!(primitive.number(&["indices"]), 3.0);
        assert_eq!(primitive.number(&["material"]), 0.0);
        assert_eq!(primitive.number(&["mode"]), 4.0);

        let accessors = value.array(&["accessors"]);
        assert_eq!(accessors.len(), 4);
        assert_eq!(accessors[0].number(&["componentType"]), f64::from(FLOAT));
        assert_eq!(accessors[0].string(&["type"]), "VEC3");
        assert_eq!(accessors[0].number(&["count"]), 3.0);
        assert_eq!(accessors[0].numbers(&["min"]), vec![0.0, 0.0, 0.0]);
        assert_eq!(accessors[0].numbers(&["max"]), vec![1.0, 2.0, 0.0]);
        assert_eq!(accessors[1].string(&["type"]), "VEC3");
        assert_eq!(accessors[1].number(&["componentType"]), f64::from(FLOAT));
        assert_eq!(accessors[2].string(&["type"]), "VEC2");
        assert_eq!(accessors[2].number(&["count"]), 3.0);
        assert_eq!(accessors[3].string(&["type"]), "SCALAR");
        assert_eq!(
            accessors[3].number(&["componentType"]),
            f64::from(UNSIGNED_INT)
        );
        assert_eq!(accessors[3].number(&["count"]), 3.0);

        let views = value.array(&["bufferViews"]);
        assert_eq!(views.len(), 4);
        assert_aligned(&views, &bin);
        assert_eq!(views[0].number(&["target"]), f64::from(ARRAY_BUFFER));
        assert_eq!(
            views[3].number(&["target"]),
            f64::from(ELEMENT_ARRAY_BUFFER)
        );
        assert_eq!(
            value.number(&["buffers", "0", "byteLength"]),
            bin.len() as f64
        );
        assert_eq!(value.string(&["buffers", "0", "uri"]), "my.scan.bin");
        assert!(!value.string(&["buffers", "0", "uri"]).contains('/'));
        assert_eq!(value.string(&["images", "0", "uri"]), "my.scan.png");
        assert_eq!(
            value.number(&["materials", "0", "pbrMetallicRoughness", "metallicFactor"]),
            0.0
        );
        assert_eq!(
            value.number(&[
                "materials",
                "0",
                "pbrMetallicRoughness",
                "baseColorTexture",
                "index"
            ]),
            0.0
        );

        let positions = read_f32s(&bin, &views[0]);
        assert_eq!(positions[0], 0.0);
        assert_eq!(positions[1], 0.0);
        assert_eq!(positions[2], 0.0);
        assert_eq!(positions[3], 1.0);
        assert_eq!(positions[4], 0.0);
        assert_eq!(positions[5], 0.0);
        assert_eq!(positions[6], 0.0);
        assert_eq!(positions[7], 2.0);
        assert_eq!(positions[8], 0.0);
        let uvs = read_f32s(&bin, &views[2]);
        assert_eq!(uvs, vec![0.0, 0.0, 1.0, 0.0, 0.25, 0.75]);
        let decoded_indices = read_u32s(&bin, &views[3]);
        assert_eq!(decoded_indices, vec![0, 2, 1]);

        let (width, height, rgb) = decode_rgb8(&png).expect("png");
        assert_eq!((width, height), (2, 2));
        assert_eq!(rgb[0], 255);
    }

    #[test]
    fn geometry_only_gltf_omits_the_material_and_the_png() {
        let dir = TempDir::new("gltf-plain");
        let path = dir.path().join("room.gltf");
        save_mesh(&triangle(), &path).expect("gltf");
        let json = fs::read_to_string(&path).expect("json");
        let value = parse_json(&json);
        let attributes = value
            .index(&["meshes"], 0)
            .index(&["primitives"], 0)
            .object(&["attributes"]);
        assert!(attributes.contains_key("POSITION"));
        assert!(attributes.contains_key("NORMAL"));
        assert!(!attributes.contains_key("TEXCOORD_0"));
        assert!(value.pointer(&["materials"]).is_none());
        assert!(value.pointer(&["images"]).is_none());
        assert!(!dir.path().join("room.png").exists());
        assert!(dir.path().join("room.bin").is_file());
        assert_eq!(value.string(&["asset", "version"]), "2.0");
        assert_eq!(value.array(&["accessors"]).len(), 3);
    }

    #[test]
    fn glb_chunks_are_aligned_and_embed_the_png() {
        let dir = TempDir::new("glb");
        let path = dir.path().join("room.glb");
        save_textured(&textured_triangle(), &path).expect("glb");
        let bytes = fs::read(&path).expect("glb");
        assert!(!dir.path().join("room.bin").exists());
        assert!(!dir.path().join("room.png").exists());

        let (json, bin) = split_glb(&bytes);
        let value = parse_json(&json);
        assert_eq!(value.string(&["asset", "version"]), "2.0");
        assert!(value.pointer(&["buffers", "0", "uri"]).is_none());
        assert_eq!(
            value.number(&["buffers", "0", "byteLength"]),
            bin.len() as f64
        );
        assert_eq!(value.string(&["images", "0", "mimeType"]), "image/png");
        let image_view = value.number(&["images", "0", "bufferView"]) as usize;
        let views = value.array(&["bufferViews"]);
        assert_eq!(views.len(), 5);
        assert_aligned(&views, &bin);
        assert!(views[image_view].pointer(&["target"]).is_none());

        let offset = views[image_view].number(&["byteOffset"]) as usize;
        let length = views[image_view].number(&["byteLength"]) as usize;
        assert_eq!(offset % 4, 0);
        let png = &bin[offset..offset + length];
        assert_eq!(
            &png[..8],
            &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']
        );
        let (width, height, rgb) = decode_rgb8(png).expect("embedded png");
        assert_eq!((width, height), (2, 2));
        assert_eq!(rgb[0], 255);
        assert!(bin[offset + length..].iter().all(|byte| *byte == 0));

        let primitive = value.index(&["meshes"], 0).index(&["primitives"], 0);
        assert_eq!(primitive.number(&["attributes", "POSITION"]), 0.0);
        assert_eq!(primitive.number(&["attributes", "NORMAL"]), 1.0);
        assert_eq!(primitive.number(&["attributes", "TEXCOORD_0"]), 2.0);
        assert_eq!(primitive.number(&["material"]), 0.0);
    }

    #[test]
    fn geometry_only_glb_is_one_aligned_file() {
        let dir = TempDir::new("glb-plain");
        let path = dir.path().join("room.GLB");
        save_mesh(&triangle(), &path).expect("glb");
        let bytes = fs::read(&path).expect("read");
        let (json, bin) = split_glb(&bytes);
        let value = parse_json(&json);
        assert!(value.pointer(&["images"]).is_none());
        assert!(value.pointer(&["materials"]).is_none());
        assert_eq!(value.array(&["bufferViews"]).len(), 3);
        assert_eq!(bin.len() % 4, 0);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_broken_triangle_is_refused_before_any_file_is_created() {
        let dir = TempDir::new("bad-index");
        let mesh = Mesh {
            colors: None,
            vertices: triangle().vertices,
            triangles: vec![[0, 1, 9]],
        };
        let error = save_mesh(&mesh, &dir.path().join("bad.obj")).expect_err("index");
        assert!(error.to_string().contains("index 9"), "{error}");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    fn assert_aligned(views: &[Json], bin: &[u8]) {
        assert_eq!(bin.len() % 4, 0);
        let mut covered = 0usize;
        for view in views {
            let offset = view.number(&["byteOffset"]) as usize;
            let length = view.number(&["byteLength"]) as usize;
            assert_eq!(offset % 4, 0, "bufferView offset {offset}");
            assert!(offset + length <= bin.len());
            covered = covered.max(offset + length);
        }
        assert!(bin.len() - covered < 4);
        assert!(bin[covered..].iter().all(|byte| *byte == 0));
    }

    fn read_f32s(bin: &[u8], view: &Json) -> Vec<f32> {
        let offset = view.number(&["byteOffset"]) as usize;
        let length = view.number(&["byteLength"]) as usize;
        bin[offset..offset + length]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn read_u32s(bin: &[u8], view: &Json) -> Vec<u32> {
        let offset = view.number(&["byteOffset"]) as usize;
        let length = view.number(&["byteLength"]) as usize;
        bin[offset..offset + length]
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn split_glb(bytes: &[u8]) -> (String, Vec<u8>) {
        assert!(bytes.len() >= 20, "glb shorter than a header");
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        let total = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(total, bytes.len());
        assert_eq!(total % 4, 0);

        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(&bytes[16..20], b"JSON");
        assert_eq!(json_len % 4, 0);
        let json_end = 20 + json_len;
        let json_chunk = &bytes[20..json_end];
        let body_end = json_chunk
            .iter()
            .rposition(|byte| *byte != b' ')
            .expect("json")
            + 1;
        assert!(json_chunk[body_end..].iter().all(|byte| *byte == b' '));
        let json = std::str::from_utf8(&json_chunk[..body_end]).expect("json utf8");
        assert!(json.ends_with('}'));

        let bin_len =
            u32::from_le_bytes(bytes[json_end..json_end + 4].try_into().unwrap()) as usize;
        assert_eq!(&bytes[json_end + 4..json_end + 8], b"BIN\0");
        assert_eq!(bin_len % 4, 0);
        let bin_end = json_end + 8 + bin_len;
        assert_eq!(bin_end, bytes.len());
        assert_eq!((json_end) % 4, 0, "BIN chunk is not 4-byte aligned");
        (json.to_string(), bytes[json_end + 8..bin_end].to_vec())
    }

    #[derive(Clone, Debug)]
    enum Json {
        Null,
        Bool,
        Number(f64),
        String(String),
        Array(Vec<Json>),
        Object(Vec<(String, Json)>),
    }

    impl Json {
        fn pointer(&self, path: &[&str]) -> Option<&Json> {
            let mut cursor = self;
            for key in path {
                cursor = match cursor {
                    Json::Object(entries) => entries
                        .iter()
                        .find(|(name, _)| name == key)
                        .map(|(_, value)| value)?,
                    Json::Array(items) => {
                        let index: usize = key.parse().ok()?;
                        items.get(index)?
                    }
                    _ => return None,
                };
            }
            Some(cursor)
        }

        fn object(&self, path: &[&str]) -> std::collections::HashMap<String, Json> {
            match self.pointer(path) {
                Some(Json::Object(entries)) => entries.iter().cloned().collect(),
                other => panic!("expected object at {path:?}, got {other:?}"),
            }
        }

        fn array(&self, path: &[&str]) -> Vec<Json> {
            match self.pointer(path) {
                Some(Json::Array(items)) => items.clone(),
                other => panic!("expected array at {path:?}, got {other:?}"),
            }
        }

        fn index(&self, path: &[&str], index: usize) -> &Json {
            match self.pointer(path) {
                Some(Json::Array(items)) => &items[index],
                other => panic!("expected array at {path:?}, got {other:?}"),
            }
        }

        fn string(&self, path: &[&str]) -> String {
            match self.pointer(path) {
                Some(Json::String(value)) => value.clone(),
                other => panic!("expected string at {path:?}, got {other:?}"),
            }
        }

        fn number(&self, path: &[&str]) -> f64 {
            match self.pointer(path) {
                Some(Json::Number(value)) => *value,
                other => panic!("expected number at {path:?}, got {other:?}"),
            }
        }

        fn numbers(&self, path: &[&str]) -> Vec<f64> {
            match self.pointer(path) {
                Some(Json::Array(items)) => items
                    .iter()
                    .map(|item| match item {
                        Json::Number(value) => *value,
                        other => panic!("expected number, got {other:?}"),
                    })
                    .collect(),
                other => panic!("expected array at {path:?}, got {other:?}"),
            }
        }
    }

    fn parse_json(text: &str) -> Json {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            index: 0,
        };
        let value = parser.parse_value();
        parser.skip();
        assert_eq!(parser.index, parser.bytes.len(), "trailing json");
        value
    }

    struct Parser<'a> {
        bytes: &'a [u8],
        index: usize,
    }

    impl Parser<'_> {
        fn parse_value(&mut self) -> Json {
            self.skip();
            match self.peek() {
                b'{' => self.parse_object(),
                b'[' => self.parse_array(),
                b'"' => Json::String(self.parse_string()),
                b't' => {
                    self.consume(b"true");
                    Json::Bool
                }
                b'f' => {
                    self.consume(b"false");
                    Json::Bool
                }
                b'n' => {
                    self.consume(b"null");
                    Json::Null
                }
                b'-' | b'0'..=b'9' => self.parse_number(),
                other => panic!("unexpected json byte {other}"),
            }
        }

        fn parse_object(&mut self) -> Json {
            self.bump(b'{');
            let mut entries = Vec::new();
            self.skip();
            if self.peek() == b'}' {
                self.index += 1;
                return Json::Object(entries);
            }
            loop {
                self.skip();
                let key = self.parse_string();
                self.skip();
                self.bump(b':');
                let value = self.parse_value();
                entries.push((key, value));
                self.skip();
                match self.peek() {
                    b',' => self.index += 1,
                    b'}' => {
                        self.index += 1;
                        break;
                    }
                    other => panic!("expected comma or end of object, got {other}"),
                }
            }
            Json::Object(entries)
        }

        fn parse_array(&mut self) -> Json {
            self.bump(b'[');
            let mut items = Vec::new();
            self.skip();
            if self.peek() == b']' {
                self.index += 1;
                return Json::Array(items);
            }
            loop {
                items.push(self.parse_value());
                self.skip();
                match self.peek() {
                    b',' => self.index += 1,
                    b']' => {
                        self.index += 1;
                        break;
                    }
                    other => panic!("expected comma or end of array, got {other}"),
                }
            }
            Json::Array(items)
        }

        fn parse_string(&mut self) -> String {
            self.bump(b'"');
            let mut out = String::new();
            while self.index < self.bytes.len() {
                let byte = self.bytes[self.index];
                self.index += 1;
                match byte {
                    b'"' => return out,
                    b'\\' => {
                        let escaped = self.bytes[self.index];
                        self.index += 1;
                        match escaped {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'u' => {
                                let hex =
                                    std::str::from_utf8(&self.bytes[self.index..self.index + 4])
                                        .expect("hex");
                                self.index += 4;
                                out.push(
                                    char::from_u32(u32::from_str_radix(hex, 16).expect("code"))
                                        .expect("char"),
                                );
                            }
                            other => panic!("bad escape {other}"),
                        }
                    }
                    other => out.push(other as char),
                }
            }
            panic!("unterminated string");
        }

        fn parse_number(&mut self) -> Json {
            let start = self.index;
            if self.peek() == b'-' {
                self.index += 1;
            }
            while self.index < self.bytes.len()
                && matches!(
                    self.bytes[self.index],
                    b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-'
                )
            {
                self.index += 1;
            }
            let text = std::str::from_utf8(&self.bytes[start..self.index]).expect("number");
            Json::Number(text.parse().unwrap_or_else(|_| panic!("bad number {text}")))
        }

        fn consume(&mut self, expected: &[u8]) {
            assert_eq!(
                &self.bytes[self.index..self.index + expected.len()],
                expected
            );
            self.index += expected.len();
        }

        fn bump(&mut self, expected: u8) {
            assert_eq!(self.peek(), expected);
            self.index += 1;
        }

        fn peek(&self) -> u8 {
            self.bytes[self.index]
        }

        fn skip(&mut self) {
            while self.index < self.bytes.len() && self.bytes[self.index].is_ascii_whitespace() {
                self.index += 1;
            }
        }
    }
}
