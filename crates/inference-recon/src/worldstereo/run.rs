//! A panorama to a generated world: the reference's stages run as a schedule of models, each loaded for its stage and
//! dropped after it, so the 24 GB card only ever holds one stage's models: MoGe-2 (scene depth, then the memory
//! selector's features), umT5 (the caption, once), the clip loop (VAE, CLIP image encoder, the streamed WorldStereo
//! transformer), then WorldMirror on the memory bank with its cameras as priors.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub use inference_models_diffusion::wan::AttentionKind;
use inference_models_diffusion::wan::clip::WanImageEncoder;
use inference_models_diffusion::wan::pipeline::{ClipRequest, MemoryFrames, WorldStereoPipeline};
use inference_models_diffusion::wan::text::WanTextEncoder;
use inference_models_diffusion::wan::transformer::{
    WanConfig, WorldStereoConfig, WorldStereoTransformer,
};
use inference_models_diffusion::wan::vae::{VaeConfig, WanVae};
use inference_quant::ShardedSafeTensors;
pub use inference_tensor::quantized::GgmlDType;
use inference_tensor::{DType, Device, Result, Tensor};
use serde::Serialize;

use super::align::{AlignFrame, Alignment, align};
use super::cameras;
use super::gs_data::{GsFrame, write_gs_data};
use super::memory::{MemoryBank, MemoryFrame};
use super::panorama::{Rgb8, pano_depth};
use super::render::render;
use super::scene::{Scene, View};
use super::trajectory::{self, Mat4, MeshBvh, Obstacles, PointGrid};
use crate::geometry::invert4;
use crate::moge::MoGe;
use crate::worldmirror::priors::PriorInputs;
use crate::worldmirror::run::{
    Axes, OPENCV_AXES, ReconstructOptions, depth_with_priors, reconstruct_with_priors,
};
use crate::worldmirror::{WorldMirror, preprocess::RgbImage};

// the reference generates each view's aerial move first, the two turns then lean on it
const TRAJECTORY_ORDER: [usize; 3] = [2, 0, 1];
// WorldMirror runs on the whole memory bank (~99 frames); the reference's 832 is ~200K tokens, past a 24 GB card
pub const DEFAULT_RECONSTRUCT_SIZE: usize = 518;
// the reference aligns depth inferred at 832 with WorldMirror's depth head alone
const ALIGN_DEPTH_SIZE: usize = 832;
const DEFAULT_SEED: u64 = 1024;
const QUANTIZATION: &str = "WorldStereo's main-block linears as settings.block_weights states (quantized at load, streamed \
                            from host memory), its other weights and umT5 in bf16; Wan VAE, CLIP image encoder and MoGe-2 \
                            in f32; WorldMirror as its own meta.json states. Attention: settings.attention";
const DIFFERENCES_FROM_REFERENCE: [&str; 6] = [
    "sky from MoGe-2's validity mask instead of GroundingDINO and ZIM",
    "one caption for every clip instead of a vision-language model's caption per start view",
    "rendered conditions used as rendered, not round-tripped through MP4",
    "memory retrieval on MoGe-2's DINOv2-L features instead of a separate DINOv2-base",
    "reconstruction at 518 on the long edge by default instead of 832, which does not fit 24 GB at ~99 frames",
    "only the regular trajectories; no VLM-planned ones",
];

/// Where each model's weights live.
#[derive(Clone, Debug)]
pub struct WorldGenModels {
    /// the `worldstereo-memory-dmd` folder (config.json, model.safetensors)
    pub worldstereo: PathBuf,
    /// the Wan2.1-I2V-14B-480P diffusers folder (vae, image_encoder, text_encoder, tokenizer, transformer/config.json)
    pub wan: PathBuf,
    /// MoGe-2 ViT-L normal, converted (config.json, model.safetensors)
    pub moge: PathBuf,
    /// WorldMirror 2.0; the final reconstruction is skipped without it
    pub worldmirror: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct WorldGenOptions {
    /// the caption every clip is generated with (the reference writes one per trajectory with a VLM)
    pub prompt: String,
    /// outdoor scenes get far-field compression; None decides by whether any sky was found
    pub outdoor: Option<bool>,
    pub seed: u64,
    /// WorldMirror's longest edge, floored to a multiple of 14
    pub reconstruct_size: usize,
    /// how WorldStereo's joint self-attentions run
    pub attention: AttentionKind,
    /// the GGUF type WorldStereo's main-block linears are quantized to at load; None keeps them bf16
    pub block_weights: Option<GgmlDType>,
}

impl Default for WorldGenOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            outdoor: None,
            seed: DEFAULT_SEED,
            reconstruct_size: DEFAULT_RECONSTRUCT_SIZE,
            attention: AttentionKind::Int8Qk,
            block_weights: Some(GgmlDType::Q8_0),
        }
    }
}

/// One stage's wall time and the device memory in use at its end, for the report and `meta.json`.
#[derive(Clone, Debug, Serialize)]
pub struct StageReport {
    pub stage: String,
    pub seconds: f64,
    pub device_mib_at_end: u64,
    /// the peak of live allocations during the stage (weights and activations), without the CUDA context or pool slack
    pub allocated_peak_mib: u64,
}

/// How the reconstruction's frame (the first bank camera's, OpenCV) maps into the scene's (the panorama's centre at the
/// origin, +z up, MoGe-2's metric scale): `scene = scale * rotation * recon + translation`, fitted on the cameras.
#[derive(Debug, Serialize)]
pub struct SceneTransform {
    pub scale: f64,
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    /// how well the predicted cameras land on the given ones after the transform (scene units, degrees)
    pub position_residual_median: f64,
    pub rotation_residual_median_deg: f64,
}

/// The weights a run read and how it ran them.
#[derive(Debug, Serialize)]
pub struct WorldGenCheckpoints {
    pub worldstereo: String,
    pub wan: String,
    pub moge: String,
    pub worldmirror: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WorldGenSettings {
    pub prompt: String,
    pub seed: u64,
    pub scene: &'static str,
    /// "--scene" when the caller chose, else "sky share" (outdoor when sky covers 0.5% of the panorama or more)
    pub scene_decided_by: &'static str,
    pub sampling_steps: usize,
    /// "bf16", or "int8 QK" (int8 Q.K^T, fp16 P.V)
    pub attention: &'static str,
    /// the main blocks' linear weights: "bf16" or a GGUF type such as "Q8_0"
    pub block_weights: String,
    pub frames_per_clip: usize,
    pub frame_width: usize,
    pub frame_height: usize,
    pub reconstruct_target_size: usize,
}

/// `meta.json` of a run: conventions, models and settings, then what each stage did.
#[derive(Debug, Serialize)]
pub struct WorldGenReport {
    pub model: &'static str,
    pub engine: String,
    pub checkpoints: WorldGenCheckpoints,
    pub quantization: &'static str,
    pub camera_convention: &'static str,
    pub camera_axes: Axes,
    pub extrinsics: &'static str,
    pub world_frame: &'static str,
    pub world_up: &'static str,
    pub handedness: &'static str,
    pub scale: &'static str,
    pub intrinsics: &'static str,
    pub panorama_width: usize,
    pub panorama_height: usize,
    pub settings: WorldGenSettings,
    pub differences_from_reference: [&'static str; 6],
    pub stages: Vec<StageReport>,
    pub clips: Vec<String>,
    pub frames: usize,
    pub reconstructed_frames: usize,
    pub reconstruction_to_scene: Option<SceneTransform>,
    /// the per-frame depth alignment (aligned/): frames fitted, borrowing a clip neighbour's fit, or left out
    pub alignment: Option<AlignedReport>,
}

#[derive(Debug, Serialize)]
pub struct AlignedReport {
    pub frames: usize,
    pub fitted: usize,
    pub borrowed: usize,
    pub abandoned: usize,
    pub points: usize,
}

// the transform from WorldMirror's frame to the scene's, from its predicted cameras and the given ones
fn scene_transform(camera_params: &Path, given_c2w: &[Mat4]) -> Result<SceneTransform> {
    let json: serde_json::Value = read_json(camera_params)?;
    let predicted = json["extrinsics"]
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|e| serde_json::from_value::<Vec<Vec<f64>>>(e["matrix"].clone()).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if predicted.len() != given_c2w.len() {
        inference_tensor::bail!(
            "{} holds {} cameras, {} were given",
            camera_params.display(),
            predicted.len(),
            given_c2w.len()
        );
    }
    let p = predicted
        .iter()
        .map(|m| [m[0][3], m[1][3], m[2][3]])
        .collect::<Vec<_>>();
    let q = given_c2w
        .iter()
        .map(|m| [m[0][3], m[1][3], m[2][3]])
        .collect::<Vec<_>>();
    let (scale, rotation, translation) = crate::geometry::similarity_fit(&p, &q);
    let median = |mut v: Vec<f64>| {
        v.sort_unstable_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let positions = p
        .iter()
        .zip(&q)
        .map(|(a, b)| {
            let m = [0, 1, 2].map(|i| {
                scale * (0..3).map(|j| rotation[i][j] * a[j]).sum::<f64>() + translation[i]
            });
            (0..3).map(|d| (m[d] - b[d]).powi(2)).sum::<f64>().sqrt()
        })
        .collect::<Vec<_>>();
    let angles = predicted
        .iter()
        .zip(given_c2w)
        .map(|(a, b)| {
            // the angle of (R * R_pred)^T R_given
            let ra = [0, 1, 2]
                .map(|i| [0, 1, 2].map(|j| (0..3).map(|k| rotation[i][k] * a[k][j]).sum::<f64>()));
            let trace = (0..3)
                .map(|i| (0..3).map(|k| ra[k][i] * b[k][i]).sum::<f64>())
                .sum::<f64>();
            ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
        })
        .collect::<Vec<_>>();
    Ok(SceneTransform {
        scale,
        rotation,
        translation,
        position_residual_median: median(positions),
        rotation_residual_median_deg: median(angles),
    })
}

#[derive(Serialize)]
struct ClipMeta {
    name: String,
    camera_to_world: Vec<Mat4>,
    intrinsics: Vec<[[f64; 3]; 3]>,
    width: usize,
    height: usize,
    memory_frames: Vec<String>,
    memory_index: Vec<usize>,
}

fn device_mib(device: &Device) -> u64 {
    inference_nn::utils::memory_usage::MemoryUsage
        .query(device)
        .map(|m| ((m.total() - m.available()) >> 20) as u64)
        .unwrap_or_default()
}

fn pool_peak_mib(device: &Device) -> u64 {
    #[cfg(feature = "cuda")]
    if let Ok(Some(snapshot)) =
        inference_nn::utils::memory_usage::MemoryUsage.query_cuda_allocator(device)
        && let Some(pool) = snapshot.async_pool
    {
        return (pool.used_high >> 20) as u64;
    }
    let _ = device;
    0
}

fn reset_pool_peak(device: &Device) {
    #[cfg(feature = "cuda")]
    let _ = inference_nn::utils::memory_usage::MemoryUsage.reset_cuda_memory_pool_high(device);
    let _ = device;
}

// Returns the memory pool's cached blocks to the driver once a stage's models are dropped: blocks cached for one stage's
// sizes (umT5's weights) otherwise fragment the next stage's room
fn release(device: &Device) -> Result<()> {
    device.synchronize()?;
    #[cfg(feature = "cuda")]
    inference_nn::utils::memory_usage::MemoryUsage.trim_cuda_memory_pool(device, 0)?;
    Ok(())
}

// A clip already written by an earlier run, loaded instead of generated again (the reference's skip_exist)
fn load_clip(dir: &Path, frames: usize) -> Option<Vec<Rgb8>> {
    (0..frames)
        .map(|f| {
            let img = image::open(dir.join(format!("frame_{f:02}.png")))
                .ok()?
                .to_rgb8();
            Some(Rgb8 {
                width: img.width() as usize,
                height: img.height() as usize,
                pixels: img.into_raw(),
            })
        })
        .collect()
}

fn sharded(
    files: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<inference_quant::ShardedVarBuilder> {
    // SAFETY: the weights files are not modified while the models are loaded
    unsafe { ShardedSafeTensors::sharded(files, dtype, device, None, Arc::new(|_| true)) }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_str(&std::fs::read_to_string(path)?)
        .map_err(|e| inference_tensor::Error::msg(format!("{}: {e}", path.display())))
}

// an 8-bit image as `[3, H, W]` in [0, 1]
pub(crate) fn image_unit_tensor(img: &Rgb8, device: &Device) -> Result<Tensor> {
    let (w, h) = (img.width, img.height);
    let mut chw = vec![0f32; 3 * w * h];
    for (p, rgb) in img.pixels.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            chw[c * w * h + p] = f32::from(rgb[c]) / 255.0;
        }
    }
    Tensor::from_vec(chw, (3, h, w), device)
}

/// `aligned/`: each frame's metric depth (`depth_NNNN.npy`, 0 where left out), `frames.json` (name, planned
/// world-to-camera, intrinsics, fit and its source) and the merged cloud `points.ply`.
fn write_alignment(dir: &Path, alignment: &Alignment) -> Result<()> {
    #[derive(Serialize)]
    struct FrameMeta<'a> {
        index: usize,
        name: &'a str,
        w2c: Mat4,
        intrinsics: [[f64; 3]; 3],
        inverse_depth_fit: Option<(f64, f64)>,
        fit_source: String,
        width: usize,
        height: usize,
    }
    std::fs::create_dir_all(dir)?;
    let mut metas = Vec::new();
    for (i, f) in alignment.frames.iter().enumerate() {
        let (w, h) = (
            (f.intrinsics[0][2] * 2.0).round() as usize,
            (f.intrinsics[1][2] * 2.0).round() as usize,
        );
        write_npy_f32(&dir.join(format!("depth_{i:04}.npy")), &f.depth, (h, w))?;
        metas.push(FrameMeta {
            index: i,
            name: &f.name,
            w2c: f.w2c,
            intrinsics: f.intrinsics,
            inverse_depth_fit: f.fit,
            fit_source: format!("{:?}", f.source),
            width: w,
            height: h,
        });
    }
    std::fs::write(
        dir.join("frames.json"),
        serde_json::to_string_pretty(&metas).map_err(inference_tensor::Error::wrap)?,
    )?;
    crate::worldmirror::output::write_points_ply(
        &dir.join("points.ply"),
        &alignment.points,
        &alignment.colors,
    )
}

/// `scene/`: the panorama's cleaned distance (`distance.npy`), `sky_mask.png` (255 off the sky), the point cloud
/// (`points.ply`) and each start view (`start_frame_N.png`), the inputs the Gaussian training data is cut from.
fn write_scene(dir: &Path, scene: &Scene, views: &[View]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let (w, h) = (scene.width, scene.height);
    write_npy_f32(&dir.join("distance.npy"), &scene.distance, (h, w))?;
    let ground = scene.sky.iter().map(|&s| if s { 0 } else { 255 }).collect();
    image::GrayImage::from_raw(w as u32, h as u32, ground)
        .expect("one byte a pixel")
        .save(dir.join("sky_mask.png"))
        .map_err(inference_tensor::Error::wrap)?;
    crate::worldmirror::output::write_points_ply(
        &dir.join("points.ply"),
        &scene.points,
        &scene.colors,
    )?;
    for (i, v) in views.iter().enumerate() {
        save_png(&v.image, &dir.join(format!("start_frame_{i}.png")))?;
    }
    Ok(())
}

// a float32 array (rows, cols) as a .npy file
fn write_npy_f32(path: &Path, values: &[f32], (rows, cols): (usize, usize)) -> Result<()> {
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // magic, version 1.0, header length; the header padded with spaces to a 64-byte boundary, ending in a newline
    let unpadded = 10 + header.len() + 1;
    let padded = unpadded.div_ceil(64) * 64;
    let mut out = Vec::with_capacity(padded + values.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&((padded - 10) as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(b' ', padded - unpadded));
    out.push(b'\n');
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out)?;
    Ok(())
}

// an 8-bit image as `[3, H, W]` in [-1, 1]
fn image_tensor(img: &Rgb8, device: &Device) -> Result<Tensor> {
    let (w, h) = (img.width, img.height);
    let mut chw = vec![0f32; 3 * w * h];
    for (p, rgb) in img.pixels.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            chw[c * w * h + p] = f32::from(rgb[c]) / 127.5 - 1.0;
        }
    }
    Tensor::from_vec(chw, (3, h, w), device)
}

fn save_png(img: &Rgb8, path: &Path) -> Result<()> {
    image::RgbImage::from_raw(img.width as u32, img.height as u32, img.pixels.clone())
        .expect("three bytes a pixel")
        .save(path)
        .map_err(|e| inference_tensor::Error::msg(format!("{}: {e}", path.display())))
}

struct Planned {
    view: usize,
    traj: usize,
    c2ws: Vec<Mat4>,
}

/// Generates a world from an equirectangular panorama into `out_dir`.
pub fn generate_world(
    models: &WorldGenModels,
    pano: &Rgb8,
    out_dir: &Path,
    options: &WorldGenOptions,
    device: &Device,
) -> Result<WorldGenReport> {
    if pano.width != 2 * pano.height {
        inference_tensor::bail!(
            "the panorama is {}x{}; an equirectangular panorama is twice as wide as it is tall",
            pano.width,
            pano.height
        );
    }
    std::fs::create_dir_all(out_dir.join("clips"))?;
    let mut stages = Vec::new();
    let mut stage = |name: &str, start: Instant, device: &Device| {
        stages.push(StageReport {
            stage: name.into(),
            seconds: start.elapsed().as_secs_f64(),
            device_mib_at_end: device_mib(device),
            allocated_peak_mib: pool_peak_mib(device),
        });
        reset_pool_peak(device);
    };
    reset_pool_peak(device);

    // 1. scene: panorama depth from MoGe-2, cleanup, point cloud, mesh, start views, memory bank crops
    let t = Instant::now();
    let moge = MoGe::load(&models.moge, DType::F32, device)?;
    let depth = pano_depth(&moge, pano, device)?;
    release(device)?;
    let scene = Scene::new(pano, &depth, options.outdoor);
    let (vw, vh) = trajectory::view_size();
    let k = trajectory::intrinsics(
        vw,
        vh,
        trajectory::VIEW_FOV_X_DEG,
        trajectory::VIEW_FOV_Y_DEG,
    );
    let views = trajectory::start_views()
        .into_iter()
        .map(|c2w| scene.view(pano, c2w))
        .collect::<Vec<View>>();
    write_scene(&out_dir.join("scene"), &scene, &views)?;
    let ground_median = scene.ground_median();
    let bank = scene
        .memory_bank(pano)
        .into_iter()
        .enumerate()
        .map(|(i, v)| MemoryFrame {
            w2c: invert4(&v.c2w),
            intrinsics: v.intrinsics,
            image: v.image,
            name: format!("render_results/pano_bank/{i:04}"),
        })
        .collect::<Vec<_>>();
    let mut memory = MemoryBank::new(bank, (vw, vh), ground_median);
    stage("scene (MoGe-2 panorama depth)", t, device);

    // 2. trajectories and their point-cloud renders (CPU)
    let t = Instant::now();
    let grid = PointGrid::new(&scene.points);
    let mesh = MeshBvh::new(&scene.mesh_vertices, &scene.mesh_triangles);
    let obstacles = Obstacles {
        mesh: &mesh,
        points: &grid,
    };
    let mut planned = Vec::new();
    for (v, view) in views.iter().enumerate() {
        let Some(median) = scene.view_median_depth(view) else {
            continue;
        };
        let moves = trajectory::regular_moves();
        for &traj in &TRAJECTORY_ORDER {
            if let Some(c2ws) = trajectory::plan(&view.c2w, moves[traj].0, median, &obstacles) {
                planned.push(Planned {
                    view: v,
                    traj,
                    c2ws,
                });
            }
        }
    }
    stage("trajectories", t, device);

    // 3. the caption through umT5, then the encoder is dropped
    let t = Instant::now();
    let text_states = {
        let te = models.wan.join("text_encoder");
        let cfg = read_json(&te.join("config.json"))?;
        let mut shards = std::fs::read_dir(&te)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect::<Vec<_>>();
        shards.sort();
        let tokenizer =
            tokenizers::Tokenizer::from_file(models.wan.join("tokenizer/tokenizer.json"))
                .map_err(inference_tensor::Error::msg)?;
        let mut encoder =
            WanTextEncoder::new(sharded(&shards, DType::BF16, device)?, &cfg, tokenizer)?;
        encoder.encode(&options.prompt)?
    };
    release(device)?;
    stage("caption (umT5-XXL)", t, device);

    // 4. the clip loop
    let t = Instant::now();
    let vae_cfg: VaeConfig = read_json(&models.wan.join("vae/config.json"))?;
    let clip_cfg = read_json(&models.wan.join("image_encoder/config.json"))?;
    let wan_cfg: WanConfig = read_json(&models.wan.join("transformer/config.json"))?;
    let ws_cfg: WorldStereoConfig = read_json(&models.worldstereo.join("config.json"))?;
    let pipeline = WorldStereoPipeline {
        vae: WanVae::new(
            sharded(
                &[models.wan.join("vae/diffusion_pytorch_model.safetensors")],
                DType::F32,
                device,
            )?,
            &vae_cfg,
            DType::F32,
        )?,
        image_encoder: WanImageEncoder::new(
            sharded(
                &[models.wan.join("image_encoder/model.safetensors")],
                DType::F32,
                device,
            )?,
            &clip_cfg,
            DType::F32,
        )?,
        transformer: {
            let mut t = WorldStereoTransformer::new_streamed(
                &wan_cfg,
                &ws_cfg,
                sharded(
                    &[models.worldstereo.join("model.safetensors")],
                    DType::BF16,
                    device,
                )?,
            )?;
            t.set_attention(options.attention);
            if let Some(dtype) = options.block_weights {
                t.quantize_blocks(dtype)?;
            }
            t
        },
        schedule: Default::default(),
        device: device.clone(),
    };
    let schedule_steps = pipeline.schedule.steps;
    let mut clips = Vec::new();
    let mut total_frames = 0;
    for (i, plan) in planned.iter().enumerate() {
        let (view_id, traj_id) = (format!("view{}", plan.view), format!("traj{}", plan.traj));
        let name = format!("{view_id}_{traj_id}");
        let w2cs = plan.c2ws.iter().map(invert4).collect::<Vec<_>>();
        let ks = vec![k; w2cs.len()];
        let dir = out_dir.join("clips").join(&name);
        if let Some(done) = load_clip(&dir, w2cs.len()) {
            memory.update(&view_id, &traj_id, &done, &w2cs, &ks);
            total_frames += done.len();
            clips.push(name);
            continue;
        }
        let mut features = |imgs: &[&Rgb8]| -> Vec<Vec<f32>> {
            imgs.iter()
                .map(|img| {
                    moge.image_features(&img.pixels, (img.width, img.height))
                        .unwrap_or_default()
                })
                .collect()
        };
        let retrieval = memory.retrieve(&view_id, &traj_id, &w2cs, &ks, &mut features);
        let start = &views[plan.view].image;
        let mut renders = vec![image_tensor(start, device)?];
        let mut masks = vec![Tensor::zeros((1, vh, vw), DType::F32, device)?];
        for w2c in &w2cs[1..] {
            let r = render(&scene.points, &scene.colors, w2c, &k, (vw, vh));
            renders.push(Tensor::from_vec(r.rgb, (3, vh, vw), device)?);
            masks.push(Tensor::from_vec(r.empty, (1, vh, vw), device)?);
        }
        let renders = Tensor::stack(&renders, 0)?;
        let render_mask = Tensor::stack(&masks, 0)?;
        let rays = Tensor::from_vec(
            cameras::ray_embedding(&w2cs, &ks, vh, vw),
            (6, w2cs.len(), vh, vw),
            device,
        )?;
        let ref_w2cs = retrieval
            .frames
            .iter()
            .map(|&f| memory.frames[f].w2c)
            .collect::<Vec<_>>();
        let (qt, qt_ref) = cameras::camera_qt(&w2cs, &ref_w2cs);
        let qt = Tensor::from_vec(qt.concat(), (w2cs.len(), 7), device)?;
        let qt_ref = Tensor::from_vec(qt_ref.concat(), (ref_w2cs.len(), 7), device)?;
        let memory_frames = Tensor::stack(
            &retrieval
                .frames
                .iter()
                .map(|&f| image_tensor(&memory.frames[f].image, device))
                .collect::<Result<Vec<_>>>()?,
            0,
        )?;
        let start_t = image_tensor(start, device)?;
        let req = ClipRequest {
            start: &start_t,
            start_rgb: &start.pixels,
            renders: &renders,
            render_mask: &render_mask,
            ray_embedding: &rays,
            camera_qt: &qt,
            memory: Some(MemoryFrames {
                frames: &memory_frames,
                camera_qt: &qt_ref,
                index: &retrieval.index,
            }),
            text_states: &text_states,
            noise: None,
            seed: options.seed + i as u64,
        };
        let frames = pipeline.generate(&req)?;
        // [F, 3, H, W] in [-1, 1] to 8-bit frames
        let bytes = ((frames.to_dtype(DType::F32)? + 1.0)? * 127.5)?
            .round()?
            .clamp(0.0, 255.0)?
            .permute((0, 2, 3, 1))?
            .to_dtype(DType::U8)?
            .to_device(&Device::Cpu)?
            .flatten_from(1)?
            .to_vec2::<u8>()?;
        let generated = bytes
            .into_iter()
            .map(|pixels| Rgb8 {
                width: vw,
                height: vh,
                pixels,
            })
            .collect::<Vec<_>>();
        std::fs::create_dir_all(&dir)?;
        for (f, img) in generated.iter().enumerate() {
            save_png(img, &dir.join(format!("frame_{f:02}.png")))?;
        }
        let meta = ClipMeta {
            name: name.clone(),
            camera_to_world: plan.c2ws.clone(),
            intrinsics: ks.clone(),
            width: vw,
            height: vh,
            memory_frames: retrieval
                .frames
                .iter()
                .map(|&f| memory.frames[f].name.clone())
                .collect(),
            memory_index: retrieval.index.clone(),
        };
        std::fs::write(
            dir.join("cameras.json"),
            serde_json::to_string_pretty(&meta).map_err(inference_tensor::Error::wrap)?,
        )?;
        memory.update(&view_id, &traj_id, &generated, &w2cs, &ks);
        total_frames += generated.len();
        clips.push(name);
        release(device)?;
    }
    drop(pipeline);
    drop(moge);
    release(device)?;
    stage("clips (WorldStereo-2)", t, device);

    // 5. alignment: each memory frame's WorldMirror depth fitted to the panorama cloud, then the preview reconstruction
    let mut reconstructed = 0;
    let mut transform = None;
    let mut aligned_report = None;
    if let Some(wm_dir) = &models.worldmirror {
        let t = Instant::now();
        let model = WorldMirror::load(wm_dir, DType::BF16, device)?;
        let images = memory
            .frames
            .iter()
            .map(|f| RgbImage {
                width: f.image.width,
                height: f.image.height,
                pixels: f.image.pixels.clone(),
            })
            .collect::<Vec<_>>();
        let bank_priors = PriorInputs {
            camera_to_world: Some(
                memory
                    .frames
                    .iter()
                    .map(|f| invert4(&f.w2c).map(|r| r.map(|v| v as f32)))
                    .collect(),
            ),
            intrinsics: Some(
                memory
                    .frames
                    .iter()
                    .map(|f| f.intrinsics.map(|r| r.map(|v| v as f32)))
                    .collect(),
            ),
            depths: None,
        };
        let depths = depth_with_priors(&model, &images, &bank_priors, ALIGN_DEPTH_SIZE)?;
        release(device)?;
        stage("alignment depth (WorldMirror 2.0)", t, device);

        let t = Instant::now();
        let moge = MoGe::load(&models.moge, DType::F32, device)?;
        let mut align_frames = Vec::with_capacity(memory.frames.len());
        for (f, depth) in memory.frames.iter().zip(depths) {
            let out = moge.infer(
                &image_unit_tensor(&f.image, device)?,
                trajectory::VIEW_FOV_X_DEG,
            )?;
            let mask = out
                .mask
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|m| m > 0.5)
                .collect();
            let normal = out
                .normal
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .as_chunks::<3>()
                .0
                .to_vec();
            align_frames.push(AlignFrame {
                name: &f.name,
                image: &f.image,
                w2c: f.w2c,
                intrinsics: f.intrinsics,
                depth,
                mask,
                normal,
            });
        }
        drop(moge);
        release(device)?;
        let alignment = align(&align_frames, &scene);
        write_alignment(&out_dir.join("aligned"), &alignment)?;
        aligned_report = Some(AlignedReport {
            frames: alignment.frames.len(),
            fitted: alignment.fitted,
            borrowed: alignment.borrowed,
            abandoned: alignment.abandoned,
            points: alignment.points.len(),
        });
        drop(align_frames);
        stage("alignment (MoGe-2 masks, fit to the panorama)", t, device);

        let t = Instant::now();
        let moge = MoGe::load(&models.moge, DType::F32, device)?;
        let by_name = alignment
            .frames
            .iter()
            .map(|f| (f.name.as_str(), f))
            .collect::<std::collections::HashMap<_, _>>();
        let mut clip_frames = Vec::new();
        for plan in &planned {
            let (view_id, traj_id) = (format!("view{}", plan.view), format!("traj{}", plan.traj));
            let dir = out_dir.join("clips").join(format!("{view_id}_{traj_id}"));
            if let Some(mut frames) = load_clip(&dir, plan.c2ws.len()) {
                // the clip's first frame is its start view (the reference swaps in start_frame.png)
                frames[0] = views[plan.view].image.clone();
                clip_frames.push((view_id, traj_id, plan, frames));
            }
        }
        let gs_frames = clip_frames
            .iter()
            .flat_map(|(view_id, traj_id, plan, frames)| {
                let by_name = &by_name;
                frames.iter().enumerate().map(move |(i, image)| GsFrame {
                    name: format!("{view_id}-{traj_id}_{i:06}"),
                    image,
                    c2w: plan.c2ws[i],
                    intrinsics: k,
                    depth: by_name
                        .get(format!("{view_id}/{traj_id}/{i:04}").as_str())
                        .map(|f| f.depth.as_slice())
                        .filter(|d| d.iter().any(|&v| v > 0.0)),
                })
            })
            .collect::<Vec<_>>();
        write_gs_data(
            &out_dir.join("gs_data"),
            &gs_frames,
            pano,
            &scene,
            &alignment,
            &moge,
            device,
        )?;
        drop(moge);
        release(device)?;
        stage(
            "gaussian splatting data (gs_data, MoGe-2 normals)",
            t,
            device,
        );

        let t = Instant::now();
        let frames = memory
            .frames
            .iter()
            .map(|f| {
                let img = RgbImage {
                    width: f.image.width,
                    height: f.image.height,
                    pixels: f.image.pixels.clone(),
                };
                (f.name.replace('/', "-"), img)
            })
            .collect::<Vec<_>>();
        let priors = PriorInputs {
            camera_to_world: Some(
                memory
                    .frames
                    .iter()
                    .map(|f| invert4(&f.w2c).map(|r| r.map(|v| v as f32)))
                    .collect(),
            ),
            intrinsics: Some(
                memory
                    .frames
                    .iter()
                    .map(|f| f.intrinsics.map(|r| r.map(|v| v as f32)))
                    .collect(),
            ),
            depths: None,
        };
        reconstructed = frames.len();
        let options = ReconstructOptions {
            target_size: options.reconstruct_size,
            ..Default::default()
        };
        let recon_dir = out_dir.join("reconstruction");
        reconstruct_with_priors(&model, wm_dir, frames, None, priors, &recon_dir, &options)?;
        let given = memory
            .frames
            .iter()
            .map(|f| invert4(&f.w2c))
            .collect::<Vec<_>>();
        transform = Some(scene_transform(
            &recon_dir.join("camera_params.json"),
            &given,
        )?);
        stage("reconstruction (WorldMirror 2.0)", t, device);
    }
    let path = |p: &Path| p.display().to_string();
    let report = WorldGenReport {
        model: "Tencent HY-World 2.0 WorldStereo-2 (worldstereo-memory-dmd), WorldMirror 2.0 on its keyframes",
        engine: format!("inference.rs {}", env!("CARGO_PKG_VERSION")),
        checkpoints: WorldGenCheckpoints {
            worldstereo: path(&models.worldstereo),
            wan: path(&models.wan),
            moge: path(&models.moge),
            worldmirror: models.worldmirror.as_deref().map(path),
        },
        quantization: QUANTIZATION,
        camera_convention: "opencv",
        camera_axes: OPENCV_AXES,
        extrinsics: "camera_to_world",
        world_frame: "the scene frame: the panorama's centre at the origin; the reconstruction has its own frame, \
                      mapped into this one by reconstruction_to_scene",
        world_up: "+z (gravity aligned as far as the panorama is level)",
        handedness: "right",
        scale: "approximately metric for the clip cameras (MoGe-2's estimate, metres; on the tested panoramas it ran 2 \
                to 3.5x small, the camera ~0.45 m above the ground), so fit to known geometry; the reconstruction is \
                arbitrary until reconstruction_to_scene is applied",
        intrinsics: "pixels of the clip frames (width x height in each cameras.json)",
        panorama_width: pano.width,
        panorama_height: pano.height,
        settings: WorldGenSettings {
            prompt: options.prompt.clone(),
            seed: options.seed,
            scene: if scene.outdoor { "outdoor" } else { "indoor" },
            scene_decided_by: if options.outdoor.is_some() {
                "--scene"
            } else {
                "sky share"
            },
            sampling_steps: schedule_steps,
            attention: match options.attention {
                AttentionKind::Bf16 => "bf16",
                AttentionKind::Int8Qk => "int8 QK",
            },
            block_weights: options
                .block_weights
                .map_or_else(|| "bf16".to_string(), |d| format!("{d:?}")),
            frames_per_clip: trajectory::TRAJECTORY_FRAMES,
            frame_width: vw,
            frame_height: vh,
            reconstruct_target_size: options.reconstruct_size,
        },
        differences_from_reference: DIFFERENCES_FROM_REFERENCE,
        stages,
        clips,
        frames: total_frames,
        reconstructed_frames: reconstructed,
        reconstruction_to_scene: transform,
        alignment: aligned_report,
    };
    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&report).map_err(inference_tensor::Error::wrap)?,
    )?;
    Ok(report)
}
