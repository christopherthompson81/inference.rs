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

fn push_f32s(bin: &mut Vec<u8>, values: impl Iterator<Item = f32>) -> (usize, usize) {
    let start = bin.len();
    for v in values {
        bin.extend_from_slice(&v.to_le_bytes());
    }
    (start, bin.len() - start)
}

/// The mesh as a binary glTF with each vertex's PBR attributes (clamped to [0, 1]).
pub fn to_glb(mesh: &Mesh, attrs: &[Pbr]) -> Vec<u8> {
    let clamp = |v: f32| v.clamp(0.0, 1.0);
    let mut bin = Vec::new();
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for a in 0..3 {
            min[a] = min[a].min(v[a]);
            max[a] = max[a].max(v[a]);
        }
    }
    let positions = push_f32s(&mut bin, mesh.vertices.iter().flatten().copied());
    let colors = push_f32s(
        &mut bin,
        attrs
            .iter()
            .flat_map(|p| [p.base_color[0], p.base_color[1], p.base_color[2], p.alpha].map(clamp)),
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
    let mut json = serde_json::to_vec(&gltf).expect("a JSON value serializes");
    while !json.len().is_multiple_of(4) {
        json.push(PAD);
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
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
