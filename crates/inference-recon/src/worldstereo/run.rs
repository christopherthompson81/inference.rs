//! A panorama to a generated world: the reference's stages run as a schedule of models, each loaded for its stage and
//! dropped after it, so the 24 GB card only ever holds one stage's models: MoGe-2 (scene depth, then the memory
//! selector's features), umT5 (the caption, once), the clip loop (VAE, CLIP image encoder, the streamed WorldStereo
//! transformer), then WorldMirror on the memory bank with its cameras as priors.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use inference_models_diffusion::wan::clip::WanImageEncoder;
use inference_models_diffusion::wan::pipeline::{ClipRequest, MemoryFrames, WorldStereoPipeline};
use inference_models_diffusion::wan::text::WanTextEncoder;
use inference_models_diffusion::wan::transformer::{
    WanConfig, WorldStereoConfig, WorldStereoTransformer,
};
use inference_models_diffusion::wan::vae::{VaeConfig, WanVae};
use inference_quant::ShardedSafeTensors;
use inference_tensor::{DType, Device, Result, Tensor};
use serde::Serialize;

use super::cameras;
use super::memory::{MemoryBank, MemoryFrame};
use super::panorama::{Rgb8, pano_depth};
use super::render::render;
use super::scene::{Scene, View};
use super::trajectory::{self, Mat4, Obstacles, PointGrid};
use crate::geometry::invert4;
use crate::moge::MoGe;
use crate::worldmirror::priors::PriorInputs;
use crate::worldmirror::run::{ReconstructOptions, reconstruct_with_priors};
use crate::worldmirror::{WorldMirror, preprocess::RgbImage};

// the reference generates each view's aerial move first, the two turns then lean on it
const TRAJECTORY_ORDER: [usize; 3] = [2, 0, 1];
// WorldMirror runs on the memory bank as the reference's alignment stage calls it
const RECONSTRUCT_TARGET_SIZE: usize = 832;
const DEFAULT_SEED: u64 = 1024;

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
}

impl Default for WorldGenOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            outdoor: None,
            seed: DEFAULT_SEED,
        }
    }
}

/// One stage's wall time and the device memory in use at its end, for the report and `meta.json`.
#[derive(Clone, Debug, Serialize)]
pub struct StageReport {
    pub stage: String,
    pub seconds: f64,
    pub device_mib_at_end: u64,
}

#[derive(Debug, Serialize)]
pub struct WorldGenReport {
    pub stages: Vec<StageReport>,
    pub clips: Vec<String>,
    pub frames: usize,
    pub reconstructed_frames: usize,
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
        });
    };

    // 1. scene: panorama depth from MoGe-2, cleanup, point cloud, mesh, start views, memory bank crops
    let t = Instant::now();
    let moge = MoGe::load(&models.moge, DType::F32, device)?;
    let depth = pano_depth(&moge, pano, device)?;
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
    let obstacles = Obstacles {
        vertices: &scene.mesh_vertices,
        triangles: &scene.mesh_triangles,
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
        transformer: WorldStereoTransformer::new_streamed(
            &wan_cfg,
            &ws_cfg,
            sharded(
                &[models.worldstereo.join("model.safetensors")],
                DType::BF16,
                device,
            )?,
        )?,
        schedule: Default::default(),
        device: device.clone(),
    };
    let mut clips = Vec::new();
    let mut total_frames = 0;
    for (i, plan) in planned.iter().enumerate() {
        let (view_id, traj_id) = (format!("view{}", plan.view), format!("traj{}", plan.traj));
        let name = format!("{view_id}_{traj_id}");
        let w2cs = plan.c2ws.iter().map(invert4).collect::<Vec<_>>();
        let ks = vec![k; w2cs.len()];
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
        let dir = out_dir.join("clips").join(&name);
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
    }
    drop(pipeline);
    drop(moge);
    stage("clips (WorldStereo-2)", t, device);

    // 5. WorldMirror on the memory bank, the bank's cameras as priors
    let mut reconstructed = 0;
    if let Some(wm_dir) = &models.worldmirror {
        let t = Instant::now();
        let model = WorldMirror::load(wm_dir, DType::BF16, device)?;
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
            target_size: RECONSTRUCT_TARGET_SIZE,
            ..Default::default()
        };
        reconstruct_with_priors(
            &model,
            wm_dir,
            frames,
            None,
            priors,
            &out_dir.join("reconstruction"),
            &options,
        )?;
        stage("reconstruction (WorldMirror 2.0)", t, device);
    }
    let report = WorldGenReport {
        stages,
        clips,
        frames: total_frames,
        reconstructed_frames: reconstructed,
    };
    std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&report).map_err(inference_tensor::Error::wrap)?,
    )?;
    Ok(report)
}
