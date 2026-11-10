//! Sampling a sparse attribute volume at points (FlexGEMM's `grid_sample_3d`, trilinear) and writing a mesh with
//! per-vertex PBR attributes as a binary glTF: `COLOR_0` (base color and alpha), and metallic and roughness as the
//! custom attributes `_METALLIC` and `_ROUGHNESS`, the material's factors set to their means.

use std::collections::HashMap;

use rayon::prelude::*;
use serde_json::json;

use super::mesh::Mesh;

const GLB_MAGIC: u32 = 0x4654_6C67;
const GLB_VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;
// glTF component and target enums
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const TRIANGLES: u32 = 4;
const LINEAR: u32 = 9729;
// the sRGB transfer function (IEC 61966-2-1)
const SRGB_LINEAR_KNEE: f32 = 0.04045;
const SRGB_LINEAR_SLOPE: f32 = 12.92;
const SRGB_OFFSET: f32 = 0.055;
const SRGB_GAMMA: f32 = 2.4;
const LINEAR_MIPMAP_LINEAR: u32 = 9987;
const REPEAT: u32 = 10497;
const PAD: u8 = 0x20;
// a weight sum this small means no voxel around the point
const MIN_WEIGHT: f32 = 1e-12;

/// PBR attributes in the texture decoder's layout.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pbr {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub alpha: f32,
}

impl Pbr {
    fn from_channels(c: &[f32]) -> Self {
        Self {
            base_color: [c[0], c[1], c[2]],
            metallic: c[3],
            roughness: c[4],
            alpha: c[5],
        }
    }
}

/// The attributes `values` (`[cells][6]`) of the cells `coords`, interpolated trilinearly at `points` given in cell
/// units (cell `c` centered at `c + 0.5`): the 8 cells around a point weighted by `prod(1 - |cell + 0.5 - point|)`,
/// absent cells left out and the weights renormalized.
pub fn sample_attributes(
    coords: &[[i32; 3]],
    values: &[[f32; 6]],
    points: &[[f32; 3]],
) -> Vec<Pbr> {
    let index: HashMap<[i32; 3], usize> = coords.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    points
        .par_iter()
        .map(|p| {
            let (mut acc, mut total) = ([0f32; 6], 0f32);
            for corner in 0..8 {
                let offset = |bit: usize| if (corner >> bit) & 1 == 1 { 0.5 } else { -0.5 };
                // torch's `.int()`: toward zero
                let cell = [0, 1, 2].map(|a| (p[a] + offset(2 - a)) as i32);
                let Some(&i) = index.get(&cell) else { continue };
                let w: f32 = (0..3)
                    .map(|a| 1.0 - (cell[a] as f32 + 0.5 - p[a]).abs())
                    .product();
                for (a, v) in acc.iter_mut().zip(&values[i]) {
                    *a += w * v;
                }
                total += w;
            }
            let scale = 1.0 / total.max(MIN_WEIGHT);
            Pbr::from_channels(&acc.map(|a| a * scale))
        })
        .collect()
}

// a binary glTF of the JSON and the binary chunk
fn assemble(gltf: &serde_json::Value, mut bin: Vec<u8>) -> Vec<u8> {
    let mut json = serde_json::to_vec(gltf).expect("a JSON value serializes");
    while !json.len().is_multiple_of(4) {
        json.push(PAD);
    }
    pad(&mut bin);
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    for word in [
        GLB_MAGIC,
        GLB_VERSION,
        total as u32,
        json.len() as u32,
        CHUNK_JSON,
    ] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&json);
    for word in [bin.len() as u32, CHUNK_BIN] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&bin);
    out
}

fn pad(bin: &mut Vec<u8>) {
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
}

/// TRELLIS.2's z-up frame to glTF's y-up one, as `to_glb` turns it: (x, y, z) -> (x, z, -y)
pub fn y_up(v: [f32; 3]) -> [f32; 3] {
    [v[0], v[2], -v[1]]
}

fn bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in points {
        for a in 0..3 {
            min[a] = min[a].min(v[a]);
            max[a] = max[a].max(v[a]);
        }
    }
    (min, max)
}

// the volume's base color is sRGB, as the baked texture stores it; glTF's COLOR_0 is linear
fn srgb_to_linear(c: f32) -> f32 {
    if c <= SRGB_LINEAR_KNEE {
        c / SRGB_LINEAR_SLOPE
    } else {
        ((c + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)).powf(SRGB_GAMMA)
    }
}

fn push_f32s(bin: &mut Vec<u8>, values: impl Iterator<Item = f32>) -> (usize, usize) {
    let start = bin.len();
    for v in values {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    (start, bin.len() - start)
}

/// The mesh as a binary glTF (y up) with each vertex's PBR attributes (clamped to [0, 1]; the base color made linear).
pub fn to_glb(mesh: &Mesh, attrs: &[Pbr]) -> Vec<u8> {
    let clamp = |v: f32| v.clamp(0.0, 1.0);
    let mut bin = Vec::new();
    let vertices: Vec<[f32; 3]> = mesh.vertices.iter().map(|&v| y_up(v)).collect();
    let (min, max) = bounds(&vertices);
    let positions = push_f32s(&mut bin, vertices.iter().flatten().copied());
    let colors = push_f32s(
        &mut bin,
        attrs.iter().flat_map(|p| {
            let [r, g, b] = p.base_color.map(|c| srgb_to_linear(clamp(c)));
            [r, g, b, clamp(p.alpha)]
        }),
    );
    let metallic = push_f32s(&mut bin, attrs.iter().map(|p| clamp(p.metallic)));
    let roughness = push_f32s(&mut bin, attrs.iter().map(|p| clamp(p.roughness)));
    let start = bin.len();
    for f in &mesh.faces {
        for i in f {
            bin.extend_from_slice(&i.to_le_bytes());
        }
    }
    let indices = (start, bin.len() - start);
    let n = attrs.len().max(1) as f32;
    let mean = |f: fn(&Pbr) -> f32| attrs.iter().map(|p| clamp(f(p))).sum::<f32>() / n;
    let view = |(offset, length): (usize, usize), target: u32| json!({"buffer": 0, "byteOffset": offset, "byteLength": length, "target": target});
    let gltf = json!({
        "asset": {"version": "2.0", "generator": "inference.rs"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "COLOR_0": 1, "_METALLIC": 2, "_ROUGHNESS": 3},
            "indices": 4,
            "material": 0,
            "mode": TRIANGLES,
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": mean(|p| p.metallic),
                "roughnessFactor": mean(|p| p.roughness),
            },
            "alphaMode": "OPAQUE",
        }],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            view(positions, ARRAY_BUFFER),
            view(colors, ARRAY_BUFFER),
            view(metallic, ARRAY_BUFFER),
            view(roughness, ARRAY_BUFFER),
            view(indices, ELEMENT_ARRAY_BUFFER),
        ],
        "accessors": [
            {"bufferView": 0, "componentType": FLOAT, "count": mesh.vertices.len(), "type": "VEC3",
             "min": min, "max": max},
            {"bufferView": 1, "componentType": FLOAT, "count": attrs.len(), "type": "VEC4"},
            {"bufferView": 2, "componentType": FLOAT, "count": attrs.len(), "type": "SCALAR"},
            {"bufferView": 3, "componentType": FLOAT, "count": attrs.len(), "type": "SCALAR"},
            {"bufferView": 4, "componentType": UNSIGNED_INT, "count": mesh.faces.len() * 3, "type": "SCALAR"},
        ],
    });
    assemble(&gltf, bin)
}

// PNG bytes of a `size` x `size` image with `channels` 3 or 4
fn png(pixels: &[u8], size: usize, channels: usize) -> Vec<u8> {
    let color = if channels == 4 {
        image::ExtendedColorType::Rgba8
    } else {
        image::ExtendedColorType::Rgb8
    };
    let mut out = Vec::new();
    image::ImageEncoder::write_image(
        image::codecs::png::PngEncoder::new(&mut out),
        pixels,
        size as u32,
        size as u32,
        color,
    )
    .expect("a whole image encodes");
    out
}

/// The baked mesh as a binary glTF (y up): positions, normals and uvs, a base color + alpha texture and a
/// metallic-roughness texture, double-sided unless remeshed, as `to_glb` makes it.
pub fn to_textured_glb(baked: &super::bake::Baked) -> Vec<u8> {
    let u = &baked.unwrapped;
    let mut bin = Vec::new();
    let vertices: Vec<[f32; 3]> = u.mesh.vertices.iter().map(|&v| y_up(v)).collect();
    let (min, max) = bounds(&vertices);
    let positions = push_f32s(&mut bin, vertices.iter().flatten().copied());
    let normals = push_f32s(&mut bin, u.normals.iter().flat_map(|&n| y_up(n)));
    let uvs = push_f32s(&mut bin, u.uvs.iter().flatten().copied());
    let start = bin.len();
    for i in u.mesh.faces.iter().flatten() {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    let indices = (start, bin.len() - start);
    let mut image_view = |pixels: &[u8], channels: usize| {
        let start = bin.len();
        bin.extend_from_slice(&png(pixels, baked.size, channels));
        let view = (start, bin.len() - start);
        pad(&mut bin);
        view
    };
    let base = image_view(&baked.base_color, 4);
    let material = image_view(&baked.metallic_roughness, 3);
    let view = |(offset, length): (usize, usize), target: Option<u32>| {
        let mut v = json!({"buffer": 0, "byteOffset": offset, "byteLength": length});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        v
    };
    let n = u.mesh.vertices.len();
    let gltf = json!({
        "asset": {"version": "2.0", "generator": "inference.rs"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3,
            "material": 0,
            "mode": TRIANGLES,
        }]}],
        "materials": [{
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "baseColorTexture": {"index": 0},
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0,
                "metallicRoughnessTexture": {"index": 1},
            },
            "alphaMode": "OPAQUE",
            "doubleSided": baked.double_sided,
        }],
        "samplers": [{"magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR, "wrapS": REPEAT, "wrapT": REPEAT}],
        "textures": [{"sampler": 0, "source": 0}, {"sampler": 0, "source": 1}],
        "images": [{"bufferView": 4, "mimeType": "image/png"}, {"bufferView": 5, "mimeType": "image/png"}],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            view(positions, Some(ARRAY_BUFFER)),
            view(normals, Some(ARRAY_BUFFER)),
            view(uvs, Some(ARRAY_BUFFER)),
            view(indices, Some(ELEMENT_ARRAY_BUFFER)),
            view(base, None),
            view(material, None),
        ],
        "accessors": [
            {"bufferView": 0, "componentType": FLOAT, "count": n, "type": "VEC3", "min": min, "max": max},
            {"bufferView": 1, "componentType": FLOAT, "count": n, "type": "VEC3"},
            {"bufferView": 2, "componentType": FLOAT, "count": n, "type": "VEC2"},
            {"bufferView": 3, "componentType": UNSIGNED_INT, "count": u.mesh.faces.len() * 3, "type": "SCALAR"},
        ],
    });
    assemble(&gltf, bin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_between_two_cells_blends_them_and_ignores_absent_ones() {
        let coords = [[0, 0, 0], [1, 0, 0]];
        let values = [[0.0; 6], [1.0; 6]];
        // halfway between the two centers: x = 1.0, and on the y/z centers
        let out = sample_attributes(&coords, &values, &[[1.0, 0.5, 0.5], [0.5, 0.5, 0.5]]);
        assert!((out[0].metallic - 0.5).abs() < 1e-6);
        assert_eq!(out[1].metallic, 0.0);
    }

    #[test]
    fn srgb_colors_become_linear() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
        // the standard's mid-gray: 0.5 sRGB is about 0.214 linear
        assert!((srgb_to_linear(0.5) - 0.214_041).abs() < 1e-5);
    }

    #[test]
    fn the_glb_has_its_header_and_aligned_chunks() {
        let mesh = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
        };
        let glb = to_glb(&mesh, &[Pbr::default(); 3]);
        let word = |i: usize| u32::from_le_bytes(glb[i..i + 4].try_into().expect("4 bytes"));
        assert_eq!(word(0), GLB_MAGIC);
        assert_eq!(word(8) as usize, glb.len());
        assert_eq!(word(12) % 4, 0);
        let json: serde_json::Value =
            serde_json::from_slice(&glb[20..20 + word(12) as usize]).expect("the JSON chunk");
        assert_eq!(json["accessors"][4]["count"], 3);
    }
}
