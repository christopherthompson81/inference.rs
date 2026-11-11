//! Image to 3D with TRELLIS.2 end to end: the object cut out (BiRefNet unless it already has transparency), its
//! sparse structure, shape and texture latents, and the mesh they decode to with its PBR attributes. Writes a GLB (the
//! mesh cleaned, unwrapped and its PBR volume baked into textures, or per-vertex color and material), the extracted
//! mesh as a PLY, the prepared image and a `meta.json`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use inference_recon::trellis2::bake;
use inference_recon::trellis2::glb::{to_glb, to_textured_glb};
use inference_recon::trellis2::mesh::Mesh;
use inference_recon::trellis2::pipeline::{self, Trellis2Paths};
use inference_tensor::Device;
use serde::Serialize;

pub use inference_recon::progress::{NoProgress, Progress};
pub use inference_recon::trellis2::bake::{BakeOptions, DECIMATION_TARGET};
pub use inference_recon::trellis2::pipeline::{
    MAX_SAMPLES, MAX_STEPS, SamplerChoice, Settings, Variant,
};

pub const GLB_FILE: &str = "model.glb";
pub const MESH_FILE: &str = "mesh.ply";
pub const PREPARED_FILE: &str = "prepared.png";
pub const META_FILE: &str = "meta.json";
const WEIGHTS_FILE: &str = "model.safetensors";
/// The face targets an export accepts
pub const FACES_RANGE: std::ops::RangeInclusive<usize> = 1_000..=10_000_000;

/// The checkpoints an image-to-3D run reads.
#[derive(Clone, Debug)]
pub struct ImageTo3dModels {
    pub paths: Trellis2Paths,
    /// BiRefNet's folder, for images without a transparent background
    pub birefnet: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
pub struct ImageTo3dOptions {
    pub settings: Settings,
    /// Remove the background even when the image has transparency of its own
    pub remove_background: bool,
    /// Faces the exported mesh is simplified to
    pub faces: usize,
    /// Bake textures (`None`: per-vertex color and material instead); its decimation target is `faces`
    pub bake: Option<BakeOptions>,
}

impl Default for ImageTo3dOptions {
    fn default() -> Self {
        Self {
            settings: Settings::new(Variant::Cascade1024, 0),
            remove_background: false,
            faces: DECIMATION_TARGET,
            bake: Some(BakeOptions::default()),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ImageTo3dReport {
    pub variant: String,
    pub resolution: usize,
    pub seed: u64,
    pub background_removed: bool,
    pub structure_cells: usize,
    pub latent_cells: usize,
    /// The mesh as extracted, before any cleanup
    pub raw_faces: usize,
    /// The GLB's mesh
    pub vertices: usize,
    pub faces: usize,
    /// The GLB's textures' size, when baked
    pub texture_size: Option<usize>,
    /// The export after generation; a bake's cleanup and unwrap run beside the texture stage and are not counted
    pub export_seconds: f64,
    pub total_seconds: f64,
}

// a repository's `files` in the hub cache, downloading what is missing; the folder they share
pub(crate) fn fetch_files(
    repo: &str,
    files: &[&str],
    progress: &dyn Progress,
) -> anyhow::Result<PathBuf> {
    let api = crate::hub_api()?.model(repo.to_string());
    progress.stage(crate::DOWNLOAD_STAGE, files.len());
    let mut root = None;
    for (i, file) in files.iter().enumerate() {
        inference_recon::progress::check(progress)?;
        let path = api
            .get(file)
            .map_err(|e| anyhow::anyhow!("fetching `{repo}`: {e}"))?;
        root = root.or_else(|| {
            path.ancestors()
                .nth(Path::new(file).components().count())
                .map(Path::to_path_buf)
        });
        progress.advance(i + 1);
    }
    root.ok_or_else(|| anyhow::anyhow!("`{repo}` gave no files"))
}

/// The checkpoints given, the others fetched from their repositories (TRELLIS.2-4B, TRELLIS-image-large's structure
/// decoder, the gated DINOv3 ViT-L/16, BiRefNet) into the hub cache. `birefnet` is `None` when not wanted, `Some(None)`
/// to fetch it.
pub fn fetch_models(
    trellis2: Option<PathBuf>,
    structure_decoder: Option<PathBuf>,
    dinov3: Option<PathBuf>,
    birefnet: Option<Option<PathBuf>>,
    progress: &dyn Progress,
) -> anyhow::Result<ImageTo3dModels> {
    let trellis2 = match trellis2 {
        Some(p) => p,
        None => fetch_files(crate::TRELLIS2_REPO, &pipeline::FILES, progress)?,
    };
    let structure_decoder = match structure_decoder {
        Some(p) => p,
        None => fetch_files(
            crate::TRELLIS_LARGE_REPO,
            &[pipeline::STRUCTURE_DECODER_FILE],
            progress,
        )?,
    };
    let dinov3 = match dinov3 {
        Some(p) => p,
        None => fetch_files(crate::DINOV3_REPO, &[WEIGHTS_FILE], progress)?,
    };
    let birefnet = match birefnet {
        Some(Some(p)) => Some(p),
        Some(None) => Some(fetch_files(
            crate::BIREFNET_REPO,
            &[WEIGHTS_FILE],
            progress,
        )?),
        None => None,
    };
    Ok(ImageTo3dModels {
        paths: Trellis2Paths {
            trellis2,
            structure_decoder,
            dinov3,
        },
        birefnet,
    })
}

/// The folder sample `index` of `samples` writes into: `out_dir` itself for a run of one, `out_dir/<index + 1>` otherwise.
pub fn sample_dir(out_dir: &Path, index: usize, samples: usize) -> PathBuf {
    if samples == 1 {
        out_dir.to_path_buf()
    } else {
        out_dir.join((index + 1).to_string())
    }
}

/// The 3D object in `image` into `out_dir`: `prepared.png` (the cut-out object), and for each sample (see
/// [`sample_dir`]) `model.glb` (baked: base color and metallic-roughness textures; otherwise per-vertex base color and
/// alpha as `COLOR_0`, metallic and roughness as custom attributes), `mesh.ply` and `meta.json`. One report per sample.
pub fn generate_3d_on(
    models: &ImageTo3dModels,
    image: &Path,
    out_dir: &Path,
    options: &ImageTo3dOptions,
    progress: &dyn Progress,
    device: &Device,
) -> anyhow::Result<Vec<ImageTo3dReport>> {
    let start = Instant::now();
    std::fs::create_dir_all(out_dir)?;
    let (input, removed) = crate::voxels::object_image(
        image,
        models.birefnet.as_deref(),
        options.remove_background,
        progress,
        device,
    )?;
    let prepared_path = out_dir.join(PREPARED_FILE);
    image::RgbaImage::from_raw(
        input.width as u32,
        input.height as u32,
        input.pixels.clone(),
    )
    .expect("a whole image")
    .save(&prepared_path)?;
    progress.preview("the object", &prepared_path);
    // the export's shape half (the bake's preparation, or the vertex-colored mesh's cleanup) runs on a worker thread
    // while the textures are made, one sample after another; the scope joins it on every path
    enum ShapeHalf {
        Baked(Box<bake::Prepared>),
        Colored(Mesh),
    }
    let samples = options.settings.samples;
    let bake_options = options.bake.map(|b| BakeOptions {
        decimation_target: options.faces,
        ..b
    });
    let faces = options.faces;
    let (outs, export, halves) = std::thread::scope(|scope| -> anyhow::Result<_> {
        let (shapes, inbox) = std::sync::mpsc::channel::<(usize, Mesh, usize)>();
        let worker = scope.spawn(move || {
            inbox
                .into_iter()
                .map(|(i, mesh, resolution)| {
                    let half = match &bake_options {
                        Some(bake_options) => {
                            bake::prepare(&mesh, resolution, bake_options, device, &|| {
                                progress.cancelled()
                            })
                            .map(|p| ShapeHalf::Baked(Box::new(p)))
                        }
                        None => pipeline::vertex_mesh(&mesh, faces, device).map(ShapeHalf::Colored),
                    };
                    (i, half)
                })
                .collect::<Vec<_>>()
        });
        let outs = pipeline::generate_with(
            &models.paths,
            &input,
            &options.settings,
            progress,
            device,
            &mut |i, mesh, resolution| {
                // the worker only stops receiving when it panicked, which the join reports
                let _ = shapes.send((i, mesh.clone(), resolution));
            },
        );
        drop(shapes);
        let halves = worker
            .join()
            .map_err(|_| anyhow::anyhow!("the export's shape half panicked"))?;
        let outs = outs?;
        Ok((outs, Instant::now(), halves))
    })?;
    progress.stage("export", 0);
    let mut halves: Vec<Option<ShapeHalf>> = {
        let mut slots: Vec<Option<ShapeHalf>> = (0..samples).map(|_| None).collect();
        for (i, half) in halves {
            slots[i] = Some(half?);
        }
        slots
    };
    let mut reports = Vec::with_capacity(samples);
    for (i, out) in outs.into_iter().enumerate() {
        let half = halves[i]
            .take()
            .ok_or_else(|| anyhow::anyhow!("generation never handed on sample {i}'s shape"))?;
        let (glb, mesh) = match half {
            ShapeHalf::Baked(prepared) => {
                let baked = bake::finish(*prepared, &out.voxels, &out.voxel_attrs)?;
                (to_textured_glb(&baked), baked.unwrapped.mesh)
            }
            ShapeHalf::Colored(mesh) => {
                let (mesh, attrs) = out.colored(mesh);
                (to_glb(&mesh, &attrs), mesh)
            }
        };
        let dir = sample_dir(out_dir, i, samples);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join(GLB_FILE), glb)?;
        std::fs::write(dir.join(MESH_FILE), mesh.to_ply())?;
        let report = ImageTo3dReport {
            variant: options.settings.variant.name().to_string(),
            resolution: out.resolution,
            seed: options.settings.seed.wrapping_add(i as u64),
            background_removed: removed,
            structure_cells: out.structure_cells,
            latent_cells: out.latent_cells,
            raw_faces: out.mesh.faces.len(),
            vertices: mesh.vertices.len(),
            faces: mesh.faces.len(),
            texture_size: options.bake.map(|b| b.texture_size),
            export_seconds: export.elapsed().as_secs_f64(),
            total_seconds: start.elapsed().as_secs_f64(),
        };
        std::fs::write(dir.join(META_FILE), serde_json::to_vec_pretty(&report)?)?;
        reports.push(report);
    }
    Ok(reports)
}
