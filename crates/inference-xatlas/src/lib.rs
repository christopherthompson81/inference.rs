//! xatlas (jpcy/xatlas, MIT; see `third_party/README.md`) through its C API: each mesh added is segmented into charts,
//! each chart parameterized, and every chart packed into one atlas. TRELLIS.2's export (CuMesh's `uv_unwrap`) adds each
//! of its normal-cone clusters as a mesh and calls it with xatlas's default options.

use std::os::raw::c_void;

#[repr(C)]
struct XChart {
    face_array: *mut u32,
    atlas_index: u32,
    face_count: u32,
    kind: u32,
    material: u32,
}

#[repr(C)]
struct XVertex {
    atlas_index: i32,
    chart_index: i32,
    uv: [f32; 2],
    xref: u32,
}

#[repr(C)]
struct XMesh {
    chart_array: *mut XChart,
    index_array: *mut u32,
    vertex_array: *mut XVertex,
    chart_count: u32,
    index_count: u32,
    vertex_count: u32,
}

#[repr(C)]
struct XAtlas {
    image: *mut u32,
    meshes: *mut XMesh,
    utilization: *mut f32,
    width: u32,
    height: u32,
    atlas_count: u32,
    chart_count: u32,
    mesh_count: u32,
    texels_per_unit: f32,
}

#[repr(C)]
struct XMeshDecl {
    vertex_position_data: *const c_void,
    vertex_normal_data: *const c_void,
    vertex_uv_data: *const c_void,
    index_data: *const c_void,
    face_ignore_data: *const bool,
    face_material_data: *const u32,
    face_vertex_count: *const u8,
    vertex_count: u32,
    vertex_position_stride: u32,
    vertex_normal_stride: u32,
    vertex_uv_stride: u32,
    index_count: u32,
    index_offset: i32,
    face_count: u32,
    index_format: u32,
    epsilon: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XChartOptions {
    param_func: *const c_void,
    max_chart_area: f32,
    max_boundary_length: f32,
    normal_deviation_weight: f32,
    roundness_weight: f32,
    straightness_weight: f32,
    normal_seam_weight: f32,
    texture_seam_weight: f32,
    max_cost: f32,
    max_iterations: u32,
    use_input_mesh_uvs: bool,
    fix_winding: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XPackOptions {
    max_chart_size: u32,
    padding: u32,
    texels_per_unit: f32,
    resolution: u32,
    bilinear: bool,
    block_align: bool,
    brute_force: bool,
    create_image: bool,
    rotate_charts_to_axis: bool,
    rotate_charts: bool,
}

// xatlasIndexFormat
const INDEX_FORMAT_UINT32: u32 = 1;
// xatlasAddMeshError
const ADD_MESH_SUCCESS: u32 = 0;
// xatlasChartType
const CHART_TYPE_INVALID: u32 = 4;

unsafe extern "C" {
    fn xatlasCreate() -> *mut XAtlas;
    fn xatlasDestroy(atlas: *mut XAtlas);
    fn xatlasAddMesh(atlas: *mut XAtlas, decl: *const XMeshDecl, mesh_count_hint: u32) -> u32;
    fn xatlasComputeCharts(atlas: *mut XAtlas, options: *const XChartOptions);
    fn xatlasPackCharts(atlas: *mut XAtlas, options: *const XPackOptions);
    fn xatlasMeshDeclInit(decl: *mut XMeshDecl);
    fn xatlasChartOptionsInit(options: *mut XChartOptions);
    fn xatlasPackOptionsInit(options: *mut XPackOptions);
}

/// A mesh to add: positions `[x, y, z]` and triangles.
pub struct MeshInput<'a> {
    pub positions: &'a [[f32; 3]],
    pub triangles: &'a [[u32; 3]],
}

/// One added mesh's result: each output vertex's source vertex (`xref`) and uv (normalized by the atlas size), and
/// the triangles over the output vertices (vertices are split along chart seams).
#[derive(Clone, Debug, Default)]
pub struct MeshOutput {
    pub xref: Vec<u32>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    /// Charts xatlas could not parameterize validly
    pub invalid_charts: usize,
}

/// The whole atlas: its size in texels and each mesh's result, in the order added.
#[derive(Clone, Debug, Default)]
pub struct AtlasOutput {
    pub width: u32,
    pub height: u32,
    pub meshes: Vec<MeshOutput>,
}

#[derive(Debug)]
pub struct Error(String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

struct Atlas(*mut XAtlas);

impl Drop for Atlas {
    fn drop(&mut self) {
        // SAFETY: created by xatlasCreate and destroyed once
        unsafe { xatlasDestroy(self.0) }
    }
}

/// Charts, parameterizes and packs `meshes` with xatlas's default chart and pack options, as CuMesh does.
pub fn unwrap(meshes: &[MeshInput]) -> Result<AtlasOutput, Error> {
    // SAFETY: the atlas is created here and freed by `Atlas`'s drop; every pointer handed to xatlas outlives the call
    // that reads it (xatlasAddMesh copies the mesh), and the output arrays are read before the atlas is destroyed
    unsafe {
        let atlas = Atlas(xatlasCreate());
        for (i, mesh) in meshes.iter().enumerate() {
            let mut decl = std::mem::zeroed::<XMeshDecl>();
            xatlasMeshDeclInit(&mut decl);
            decl.vertex_position_data = mesh.positions.as_ptr().cast();
            decl.vertex_count = mesh.positions.len() as u32;
            decl.vertex_position_stride = std::mem::size_of::<[f32; 3]>() as u32;
            decl.index_data = mesh.triangles.as_ptr().cast();
            decl.index_count = (mesh.triangles.len() * 3) as u32;
            decl.index_format = INDEX_FORMAT_UINT32;
            let err = xatlasAddMesh(atlas.0, &decl, meshes.len() as u32);
            if err != ADD_MESH_SUCCESS {
                return Err(Error(format!("xatlas could not add mesh {i}: error {err}")));
            }
        }
        let mut chart = std::mem::zeroed::<XChartOptions>();
        xatlasChartOptionsInit(&mut chart);
        xatlasComputeCharts(atlas.0, &chart);
        let mut pack = std::mem::zeroed::<XPackOptions>();
        xatlasPackOptionsInit(&mut pack);
        xatlasPackCharts(atlas.0, &pack);
        let a = &*atlas.0;
        let (w, h) = (a.width as f32, a.height as f32);
        let out_meshes = (0..a.mesh_count as usize)
            .map(|m| {
                let mesh = &*a.meshes.add(m);
                let vertices =
                    std::slice::from_raw_parts(mesh.vertex_array, mesh.vertex_count as usize);
                let indices =
                    std::slice::from_raw_parts(mesh.index_array, mesh.index_count as usize);
                let charts =
                    std::slice::from_raw_parts(mesh.chart_array, mesh.chart_count as usize);
                MeshOutput {
                    xref: vertices.iter().map(|v| v.xref).collect(),
                    uvs: vertices
                        .iter()
                        .map(|v| {
                            if w > 0.0 && h > 0.0 {
                                [v.uv[0] / w, v.uv[1] / h]
                            } else {
                                [0.0, 0.0]
                            }
                        })
                        .collect(),
                    triangles: indices
                        .chunks_exact(3)
                        .map(|t| [t[0], t[1], t[2]])
                        .collect(),
                    invalid_charts: charts
                        .iter()
                        .filter(|c| c.kind == CHART_TYPE_INVALID)
                        .count(),
                }
            })
            .collect();
        Ok(AtlasOutput {
            width: a.width,
            height: a.height,
            meshes: out_meshes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cube_unwraps_into_one_atlas_with_every_face_mapped() {
        let positions: Vec<[f32; 3]> = (0..8)
            .map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32])
            .collect();
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        let triangles: Vec<[u32; 3]> = quads
            .iter()
            .flat_map(|q: &[u32; 4]| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
            .collect();
        let out = unwrap(&[MeshInput {
            positions: &positions,
            triangles: &triangles,
        }])
        .expect("a cube unwraps");
        assert!(out.width > 0 && out.height > 0);
        let mesh = &out.meshes[0];
        assert_eq!(mesh.triangles.len(), 12);
        assert_eq!(mesh.invalid_charts, 0);
        assert!(mesh.xref.iter().all(|&x| x < 8));
        assert!(mesh.uvs.iter().flatten().all(|&c| (0.0..=1.0).contains(&c)));
    }
}
