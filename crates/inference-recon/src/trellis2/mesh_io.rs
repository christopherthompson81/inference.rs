//! Meshes read from files for the texturing pipeline: binary glTF (`.glb`, every triangle primitive of the default
//! scene placed by its nodes' transforms), Wavefront `.obj` and `.ply` (ASCII or binary). Vertices are kept as the file
//! splits them, so seams stay seams. Texture coordinates come back with row 0 at v = 0 (glTF's convention; OBJ's and
//! PLY's are flipped), and only when one primitive carries them all, since several would share one atlas.

use std::collections::HashMap;
use std::path::Path;

use super::mesh::Mesh;

const GLB_MAGIC: u32 = 0x4654_6c67;
const CHUNK_JSON: u32 = 0x4e4f_534a;
const CHUNK_BIN: u32 = 0x004e_4942;
const MODE_TRIANGLES: u64 = 4;
const COMPONENT_U8: u64 = 5121;
const COMPONENT_U16: u64 = 5123;
const COMPONENT_U32: u64 = 5125;
const COMPONENT_F32: u64 = 5126;

/// A mesh as read, with its texture coordinates when it has one usable set.
pub struct ReadMesh {
    pub mesh: Mesh,
    pub uvs: Option<Vec<[f32; 2]>>,
}

type IoResult<T> = std::result::Result<T, String>;

/// The mesh in `path`, by its extension.
pub fn read_mesh(path: &Path) -> IoResult<ReadMesh> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let read = match ext.as_str() {
        "glb" => read_glb(&bytes)?,
        "obj" => read_obj(&String::from_utf8_lossy(&bytes))?,
        "ply" => read_ply(&bytes)?,
        other => {
            return Err(format!(
                "meshes are read from .glb, .obj or .ply, not `.{other}`"
            ));
        }
    };
    if read.mesh.faces.is_empty() {
        return Err(format!("{} holds no triangles", path.display()));
    }
    Ok(read)
}

// row 0 at v = 0 from OpenGL's v up
fn flip(uv: [f32; 2]) -> [f32; 2] {
    [uv[0], 1.0 - uv[1]]
}

// fan triangulation of a polygon's corners
fn fan(corners: &[u32], faces: &mut Vec<[u32; 3]>) {
    for k in 1..corners.len().saturating_sub(1) {
        faces.push([corners[0], corners[k], corners[k + 1]]);
    }
}

fn read_obj(text: &str) -> IoResult<ReadMesh> {
    let (mut positions, mut texcoords) = (Vec::<[f32; 3]>::new(), Vec::<[f32; 2]>::new());
    let mut corner_index: HashMap<(usize, Option<usize>), u32> = HashMap::new();
    let (mut vertices, mut uvs, mut faces) = (Vec::new(), Vec::new(), Vec::new());
    let mut all_textured = true;
    let resolve = |i: i64, len: usize| -> IoResult<usize> {
        let k = if i < 0 { len as i64 + i } else { i - 1 };
        usize::try_from(k)
            .ok()
            .filter(|&k| k < len)
            .ok_or_else(|| format!("obj index {i} out of range"))
    };
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let numbers = |parts: std::str::SplitWhitespace| -> Vec<f32> {
            parts.filter_map(|p| p.parse().ok()).collect()
        };
        match parts.next() {
            Some("v") => {
                let n = numbers(parts);
                if n.len() < 3 {
                    return Err(format!("obj vertex `{line}`"));
                }
                positions.push([n[0], n[1], n[2]]);
            }
            Some("vt") => {
                let n = numbers(parts);
                texcoords.push([
                    n.first().copied().unwrap_or(0.0),
                    n.get(1).copied().unwrap_or(0.0),
                ]);
            }
            Some("f") => {
                let mut corners = Vec::new();
                for p in parts {
                    let mut ids = p.split('/');
                    let v: i64 = ids
                        .next()
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| format!("obj face `{line}`"))?;
                    let v = resolve(v, positions.len())?;
                    let t = match ids.next().filter(|s| !s.is_empty()) {
                        Some(s) => Some(resolve(
                            s.parse().map_err(|_| format!("obj face `{line}`"))?,
                            texcoords.len(),
                        )?),
                        None => None,
                    };
                    all_textured &= t.is_some();
                    let next = vertices.len() as u32;
                    let id = *corner_index.entry((v, t)).or_insert_with(|| {
                        vertices.push(positions[v]);
                        uvs.push(t.map_or([0.0; 2], |t| flip(texcoords[t])));
                        next
                    });
                    corners.push(id);
                }
                fan(&corners, &mut faces);
            }
            _ => {}
        }
    }
    Ok(ReadMesh {
        mesh: Mesh { vertices, faces },
        uvs: (all_textured && !texcoords.is_empty()).then_some(uvs),
    })
}

#[derive(Clone, Copy, PartialEq)]
enum PlyFormat {
    Ascii,
    Little,
    Big,
}

struct PlyProperty {
    name: String,
    kind: String,
    // a list's count type
    list: Option<String>,
}

struct PlyElement {
    name: String,
    count: usize,
    properties: Vec<PlyProperty>,
}

fn ply_size(kind: &str) -> IoResult<usize> {
    Ok(match kind {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "int" | "uint" | "float" | "int32" | "uint32" | "float32" => 4,
        "double" | "float64" => 8,
        other => return Err(format!("ply type `{other}`")),
    })
}

struct PlyReader<'a> {
    format: PlyFormat,
    bytes: &'a [u8],
    at: usize,
    tokens: std::str::SplitAsciiWhitespace<'a>,
}

impl PlyReader<'_> {
    fn value(&mut self, kind: &str) -> IoResult<f64> {
        if self.format == PlyFormat::Ascii {
            return self
                .tokens
                .next()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| "ply data ends early".to_string());
        }
        let n = ply_size(kind)?;
        let raw = self
            .bytes
            .get(self.at..self.at + n)
            .ok_or_else(|| "ply data ends early".to_string())?;
        self.at += n;
        let mut b = [0u8; 8];
        b[..n].copy_from_slice(raw);
        if self.format == PlyFormat::Big {
            b[..n].reverse();
        }
        Ok(match kind {
            "char" | "int8" => f64::from(b[0] as i8),
            "uchar" | "uint8" => f64::from(b[0]),
            "short" | "int16" => f64::from(i16::from_le_bytes([b[0], b[1]])),
            "ushort" | "uint16" => f64::from(u16::from_le_bytes([b[0], b[1]])),
            "int" | "int32" => f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            "uint" | "uint32" => f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            "float" | "float32" => f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            _ => f64::from_le_bytes(b),
        })
    }
}

fn read_ply(bytes: &[u8]) -> IoResult<ReadMesh> {
    let end = bytes
        .windows(10)
        .position(|w| w == b"end_header")
        .ok_or("ply without end_header")?;
    let body = end
        + 10
        + bytes[end + 10..]
            .iter()
            .position(|&b| b == b'\n')
            .ok_or("ply header")?
        + 1;
    let header = String::from_utf8_lossy(&bytes[..end]);
    let mut format = None;
    let mut elements: Vec<PlyElement> = Vec::new();
    for line in header.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.as_slice() {
            ["format", f, ..] => {
                format = Some(match *f {
                    "ascii" => PlyFormat::Ascii,
                    "binary_little_endian" => PlyFormat::Little,
                    "binary_big_endian" => PlyFormat::Big,
                    other => return Err(format!("ply format `{other}`")),
                })
            }
            ["element", name, count] => elements.push(PlyElement {
                name: name.to_string(),
                count: count.parse().map_err(|_| format!("ply `{line}`"))?,
                properties: Vec::new(),
            }),
            ["property", "list", count, kind, name] => elements
                .last_mut()
                .ok_or("ply property before an element")?
                .properties
                .push(PlyProperty {
                    name: name.to_string(),
                    kind: kind.to_string(),
                    list: Some(count.to_string()),
                }),
            ["property", kind, name] => elements
                .last_mut()
                .ok_or("ply property before an element")?
                .properties
                .push(PlyProperty {
                    name: name.to_string(),
                    kind: kind.to_string(),
                    list: None,
                }),
            _ => {}
        }
    }
    let format = format.ok_or("ply without a format")?;
    let ascii = String::from_utf8_lossy(if format == PlyFormat::Ascii {
        &bytes[body..]
    } else {
        &[]
    })
    .into_owned();
    let mut reader = PlyReader {
        format,
        bytes,
        at: body,
        tokens: ascii.split_ascii_whitespace(),
    };
    let (mut vertices, mut uvs, mut faces) = (Vec::new(), Vec::new(), Vec::new());
    let mut textured = false;
    for element in &elements {
        let find = |names: &[&str]| {
            element
                .properties
                .iter()
                .position(|p| names.contains(&p.name.as_str()))
        };
        let xyz = [find(&["x"]), find(&["y"]), find(&["z"])];
        let st = [
            find(&["s", "u", "texture_u", "texture_s"]),
            find(&["t", "v", "texture_v", "texture_t"]),
        ];
        textured |= element.name == "vertex" && st.iter().all(Option::is_some);
        for _ in 0..element.count {
            let mut values = vec![0f64; element.properties.len()];
            let mut list = Vec::new();
            for (k, p) in element.properties.iter().enumerate() {
                match &p.list {
                    Some(count) => {
                        let n = reader.value(count)? as usize;
                        let items = (0..n)
                            .map(|_| reader.value(&p.kind))
                            .collect::<IoResult<Vec<_>>>()?;
                        if matches!(p.name.as_str(), "vertex_indices" | "vertex_index") {
                            list = items;
                        }
                    }
                    None => values[k] = reader.value(&p.kind)?,
                }
            }
            match element.name.as_str() {
                "vertex" => {
                    let get = |i: Option<usize>| i.map_or(0.0, |i| values[i] as f32);
                    vertices.push(xyz.map(get));
                    uvs.push(flip(st.map(get)));
                }
                "face" => {
                    let corners: Vec<u32> = list.iter().map(|&i| i as u32).collect();
                    fan(&corners, &mut faces);
                }
                _ => {}
            }
        }
    }
    if faces
        .iter()
        .flatten()
        .any(|&i| i as usize >= vertices.len())
    {
        return Err("ply face index out of range".into());
    }
    Ok(ReadMesh {
        mesh: Mesh { vertices, faces },
        uvs: textured.then_some(uvs),
    })
}

type Mat4 = [[f32; 4]; 4];

const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as f32)
                .collect()
        })
        .unwrap_or_default()
}

// a node's local transform (row-major) from its column-major `matrix` or its translation, rotation and scale
fn local(node: &serde_json::Value) -> Mat4 {
    let m = floats(&node["matrix"]);
    if m.len() == 16 {
        return std::array::from_fn(|i| std::array::from_fn(|j| m[j * 4 + i]));
    }
    let t = floats(&node["translation"]);
    let r = floats(&node["rotation"]);
    let s = floats(&node["scale"]);
    let [x, y, z, w] = if r.len() == 4 {
        [r[0], r[1], r[2], r[3]]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    let rot = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let scale = if s.len() == 3 {
        [s[0], s[1], s[2]]
    } else {
        [1.0; 3]
    };
    let mut out = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = rot[i][j] * scale[j];
        }
        out[i][3] = t.get(i).copied().unwrap_or(0.0);
    }
    out
}

struct Glb<'a> {
    json: serde_json::Value,
    bin: &'a [u8],
}

impl Glb<'_> {
    // an accessor's elements as f32 rows of `width` (normalized integers scaled to [0, 1])
    fn floats(&self, index: u64, width: usize) -> IoResult<Vec<Vec<f32>>> {
        let a = &self.json["accessors"][index as usize];
        if a.get("sparse").is_some() {
            return Err("sparse glTF accessors are not read".into());
        }
        let count = a["count"].as_u64().ok_or("accessor count")? as usize;
        let component = a["componentType"]
            .as_u64()
            .ok_or("accessor componentType")?;
        let normalized = a["normalized"].as_bool().unwrap_or(false);
        let size = match component {
            COMPONENT_F32 | COMPONENT_U32 => 4,
            COMPONENT_U16 => 2,
            COMPONENT_U8 => 1,
            other => return Err(format!("glTF component type {other}")),
        };
        let view = &self.json["bufferViews"][a["bufferView"]
            .as_u64()
            .ok_or("an accessor without a buffer view")?
            as usize];
        if view["buffer"].as_u64().unwrap_or(0) != 0 {
            return Err("glTF data outside the GLB's buffer".into());
        }
        let start = view["byteOffset"].as_u64().unwrap_or(0) as usize
            + a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = view["byteStride"]
            .as_u64()
            .map_or(width * size, |s| s as usize);
        (0..count)
            .map(|i| {
                (0..width)
                    .map(|k| {
                        let at = start + i * stride + k * size;
                        let b = self
                            .bin
                            .get(at..at + size)
                            .ok_or("glTF accessor past its buffer")?;
                        Ok(match component {
                            COMPONENT_F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                            COMPONENT_U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
                            COMPONENT_U16 => {
                                let v = f32::from(u16::from_le_bytes([b[0], b[1]]));
                                if normalized {
                                    v / f32::from(u16::MAX)
                                } else {
                                    v
                                }
                            }
                            _ => {
                                let v = f32::from(b[0]);
                                if normalized {
                                    v / f32::from(u8::MAX)
                                } else {
                                    v
                                }
                            }
                        })
                    })
                    .collect()
            })
            .collect()
    }

    fn indices(&self, index: u64) -> IoResult<Vec<u32>> {
        let a = &self.json["accessors"][index as usize];
        let component = a["componentType"].as_u64().unwrap_or(0);
        if ![COMPONENT_U8, COMPONENT_U16, COMPONENT_U32].contains(&component) {
            return Err(format!("glTF index component type {component}"));
        }
        Ok(self
            .floats(index, 1)?
            .into_iter()
            .map(|r| r[0] as u32)
            .collect())
    }
}

fn read_glb(bytes: &[u8]) -> IoResult<ReadMesh> {
    let word = |at: usize| -> IoResult<u32> {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| "a truncated GLB".to_string())
    };
    if word(0)? != GLB_MAGIC {
        return Err("not a binary glTF".into());
    }
    let (mut json, mut bin) = (None, &[][..]);
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let (len, kind) = (word(at)? as usize, word(at + 4)?);
        let chunk = bytes
            .get(at + 8..at + 8 + len)
            .ok_or("a truncated GLB chunk")?;
        match kind {
            CHUNK_JSON => {
                json = Some(
                    serde_json::from_slice::<serde_json::Value>(chunk)
                        .map_err(|e| format!("GLB JSON: {e}"))?,
                )
            }
            CHUNK_BIN => bin = chunk,
            _ => {}
        }
        at += 8 + len;
    }
    let glb = Glb {
        json: json.ok_or("a GLB without JSON")?,
        bin,
    };
    if let Some(ext) = glb.json["extensionsRequired"]
        .as_array()
        .filter(|a| !a.is_empty())
    {
        return Err(format!(
            "glTF extensions required: {ext:?} (compressed meshes are not read)"
        ));
    }
    let nodes = glb.json["nodes"].as_array().cloned().unwrap_or_default();
    let scene = glb.json["scene"].as_u64().unwrap_or(0) as usize;
    let roots: Vec<u64> = match glb.json["scenes"][scene]["nodes"].as_array() {
        Some(r) => r.iter().filter_map(|v| v.as_u64()).collect(),
        None => {
            // no scene: the nodes nothing else lists as a child
            let children: std::collections::HashSet<u64> = nodes
                .iter()
                .flat_map(|n| n["children"].as_array().cloned().unwrap_or_default())
                .filter_map(|v| v.as_u64())
                .collect();
            (0..nodes.len() as u64)
                .filter(|i| !children.contains(i))
                .collect()
        }
    };
    let (mut vertices, mut faces, mut uvs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut primitives, mut textured) = (0usize, 0usize);
    let mut stack: Vec<(u64, Mat4)> = roots.into_iter().rev().map(|r| (r, IDENTITY)).collect();
    while let Some((n, parent)) = stack.pop() {
        let node = nodes.get(n as usize).ok_or("a glTF node out of range")?;
        let world = mul(&parent, &local(node));
        if let Some(m) = node["mesh"].as_u64() {
            for prim in glb.json["meshes"][m as usize]["primitives"]
                .as_array()
                .cloned()
                .unwrap_or_default()
            {
                if prim["mode"].as_u64().unwrap_or(MODE_TRIANGLES) != MODE_TRIANGLES {
                    continue;
                }
                let attrs = &prim["attributes"];
                let Some(pos) = attrs["POSITION"].as_u64() else {
                    continue;
                };
                let base = vertices.len() as u32;
                let points = glb.floats(pos, 3)?;
                vertices.extend(points.iter().map(|p| {
                    std::array::from_fn(|i| {
                        (0..3).map(|k| world[i][k] * p[k]).sum::<f32>() + world[i][3]
                    })
                }));
                match attrs["TEXCOORD_0"].as_u64() {
                    Some(t) => {
                        textured += 1;
                        uvs.extend(glb.floats(t, 2)?.into_iter().map(|r| [r[0], r[1]]));
                    }
                    None => uvs.extend(std::iter::repeat_n([0.0; 2], points.len())),
                }
                let index: Vec<u32> = match prim["indices"].as_u64() {
                    Some(i) => glb.indices(i)?,
                    None => (0..points.len() as u32).collect(),
                };
                if index.iter().any(|&i| i as usize >= points.len()) {
                    return Err("a glTF index out of range".into());
                }
                // a mirroring transform turns the faces inside out; glTF flips their winding back
                let det = world[0][0] * (world[1][1] * world[2][2] - world[1][2] * world[2][1])
                    - world[0][1] * (world[1][0] * world[2][2] - world[1][2] * world[2][0])
                    + world[0][2] * (world[1][0] * world[2][1] - world[1][1] * world[2][0]);
                faces.extend(index.chunks_exact(3).map(|t| {
                    if det < 0.0 {
                        [base + t[0], base + t[2], base + t[1]]
                    } else {
                        [base + t[0], base + t[1], base + t[2]]
                    }
                }));
                primitives += 1;
            }
        }
        for child in node["children"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .rev()
        {
            if let Some(c) = child.as_u64() {
                stack.push((c, world));
            }
        }
    }
    Ok(ReadMesh {
        mesh: Mesh { vertices, faces },
        uvs: (primitives == 1 && textured == 1).then_some(uvs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_obj_splits_vertices_at_texture_seams_and_flips_v() {
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.5\n\
                   f 1/1 2/2 3/3 4/4\nf 1/5 3/3 2/2\n";
        let read = read_obj(obj).expect("an obj");
        // the quad fans into two triangles; vertex 1 with a second texture coordinate is its own vertex
        assert_eq!(read.mesh.faces, vec![[0, 1, 2], [0, 2, 3], [4, 2, 1]]);
        assert_eq!(read.mesh.vertices.len(), 5);
        assert_eq!(read.uvs.expect("uvs")[3], [0.0, 0.0]);
    }

    #[test]
    fn an_ascii_and_a_binary_ply_read_the_same() {
        let ascii = b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n\
                      element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";
        let mut binary = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\n\
                           property float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
            .to_vec();
        for v in [[0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            v.iter().for_each(|c| binary.extend(c.to_le_bytes()));
        }
        binary.push(3);
        [0i32, 1, 2]
            .iter()
            .for_each(|i| binary.extend(i.to_le_bytes()));
        let (a, b) = (
            read_ply(ascii).expect("ascii"),
            read_ply(&binary).expect("binary"),
        );
        assert_eq!(a.mesh.vertices, b.mesh.vertices);
        assert_eq!(a.mesh.faces, vec![[0, 1, 2]]);
        assert_eq!(b.mesh.faces, a.mesh.faces);
        assert!(a.uvs.is_none());
    }

    #[test]
    fn a_glb_places_its_primitives_by_their_nodes() {
        // one triangle, its node translated by 2 along x and mirrored in y
        let mut bin = Vec::new();
        for v in [[0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            v.iter().for_each(|c| bin.extend(c.to_le_bytes()));
        }
        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "translation": [2.0, 0.0, 0.0], "scale": [1.0, -1.0, 1.0]}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "accessors": [{"bufferView": 0, "componentType": COMPONENT_F32, "count": 3, "type": "VEC3"}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": bin.len()}],
            "buffers": [{"byteLength": bin.len()}],
        });
        let mut text = serde_json::to_vec(&json).expect("json");
        while text.len() % 4 != 0 {
            text.push(b' ');
        }
        let mut glb = Vec::new();
        glb.extend(GLB_MAGIC.to_le_bytes());
        glb.extend(2u32.to_le_bytes());
        glb.extend(((12 + 8 + text.len() + 8 + bin.len()) as u32).to_le_bytes());
        glb.extend((text.len() as u32).to_le_bytes());
        glb.extend(CHUNK_JSON.to_le_bytes());
        glb.extend(&text);
        glb.extend((bin.len() as u32).to_le_bytes());
        glb.extend(CHUNK_BIN.to_le_bytes());
        glb.extend(&bin);
        let read = read_glb(&glb).expect("a glb");
        assert_eq!(
            read.mesh.vertices,
            vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, -1.0, 0.0]]
        );
        // the mirror flips the winding back
        assert_eq!(read.mesh.faces, vec![[0, 2, 1]]);
        assert!(read.uvs.is_none());
    }
}
