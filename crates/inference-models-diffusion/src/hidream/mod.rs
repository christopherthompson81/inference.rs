//! HiDream-O1-Image: one Qwen3-VL transformer denoising 32 px RGB patches directly, from text and reference images.

pub mod gguf_writer;
pub mod model;
pub mod references;
pub mod sampling;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use indicatif::MultiProgress;
use inference_quant::ShardedSafeTensors;
use inference_tensor::{DType, Device, IndexOp, Result, Tensor};
use rand::Rng;
use tokenizers::Tokenizer;

use crate::DiffusionGenerationParams;
use crate::noise::SeededNoise;
use inference_nn::device_map::DummyDeviceMapper;
use inference_nn::model::MAX_REFERENCE_IMAGES;
use inference_nn::model::{DiffusionModel, NormalLoadingMetadata};
use model::{Conditioning, HiDreamO1, PATCH_DIM, PATCH_SIZE};
use references::References;
use sampling::{DEV_TIMESTEPS, NUM_TRAIN_TIMESTEPS, Schedule, UniPc};

// the timestep embedding replaces this token, the prompt's last
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>";
const PROMPT_PREFIX: &str = "<|im_start|>user\n";
// Qwen3-VL's chat template for one image in a user turn; its single pad grows to the image's ViT tokens
const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
pub const UNCONDITIONAL_PROMPT: &str = " ";
const T_EPS: f32 = 0.001;
/// The sizes the model was trained at, about 4.2 MP each; a request snaps to the closest aspect.
pub const RESOLUTIONS: [(usize, usize); 11] = [
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
];
const FULL_STEPS: usize = 50;
const FULL_GUIDANCE: f64 = 5.0;
const FULL_SHIFT: f64 = 3.0;
const FULL_NOISE_SCALE: f32 = 8.0;
const DEV_NOISE_SCALE: f32 = 7.5;
const DEV_NOISE_CLIP_STD: f32 = 2.5;
// Dev's edit runs plain Euler at the pipeline's default noise scale, not its flash sampler's
const DEV_EDIT_NOISE_SCALE: f32 = 8.0;

/// The undistilled model samples with UniPC and guidance; Dev with its fixed re-noising schedule and none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Full,
    Dev,
}

/// The step rule: UniPC's predictor-corrector, Dev's re-noising flash Euler, or the plain Euler Dev edits with.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    UniPc,
    Flash { clip_std: f32 },
    Euler,
}

/// How one image is sampled.
#[derive(Clone, Debug)]
pub struct Settings {
    pub schedule: Schedule,
    pub guidance_scale: f64,
    pub sampler: Sampler,
    /// The initial noise's scale, and Dev's re-noising scale.
    pub noise_scale: f32,
}

impl Variant {
    /// The reference pipeline's settings for `references` images; `steps` and `guidance_scale` override Full's.
    pub fn settings(
        self,
        steps: Option<usize>,
        guidance_scale: Option<f64>,
        references: usize,
    ) -> Result<Settings> {
        Ok(match self {
            Self::Full => Settings {
                schedule: Schedule::unipc(steps.unwrap_or(FULL_STEPS), FULL_SHIFT),
                guidance_scale: guidance_scale.unwrap_or(FULL_GUIDANCE),
                sampler: Sampler::UniPc,
                noise_scale: FULL_NOISE_SCALE,
            },
            Self::Dev => {
                // a scale of 1 or less is no guidance, as the reference runs Dev with 0
                if steps.is_some_and(|s| s != DEV_TIMESTEPS.len())
                    || guidance_scale.is_some_and(|g| g > 1.0)
                {
                    inference_tensor::bail!(
                        "HiDream-O1 Dev is distilled to its fixed {}-step schedule without guidance",
                        DEV_TIMESTEPS.len()
                    );
                }
                // inference.py's Dev default: editing (exactly one reference) steps plain Euler
                let (sampler, noise_scale) = if references == 1 {
                    (Sampler::Euler, DEV_EDIT_NOISE_SCALE)
                } else {
                    let clip_std = DEV_NOISE_CLIP_STD;
                    (Sampler::Flash { clip_std }, DEV_NOISE_SCALE)
                };
                Settings {
                    schedule: Schedule::fixed(&DEV_TIMESTEPS),
                    guidance_scale: 0.0,
                    sampler,
                    noise_scale,
                }
            }
        })
    }
}

/// Standard normal noise in patch layout, (1, patches, 3072) in F32.
pub trait Noise {
    fn sample(&mut self, patches: usize, device: &Device) -> Result<Tensor>;
}

impl Noise for SeededNoise {
    fn sample(&mut self, patches: usize, device: &Device) -> Result<Tensor> {
        self.normal((1, patches, PATCH_DIM), device)
    }
}

pub const CONFIG_FILE: &str = "config.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
// ComfyUI-GGUF's marker for a diffusion model, which these files carry with `general.architecture = qwen`
pub(crate) const GGUF_TYPE_KEY: &str = "general.type";
const GGUF_DIFFUSION: &str = "diffusion";
/// HiDream's patch embedder, which no Qwen3-VL checkpoint has.
pub const PATCH_EMBED_TENSOR: &str = "model.x_embedder.proj1.weight";

/// Whether `path` is a HiDream-O1 GGUF: ComfyUI-GGUF's diffusion marker over HiDream's patch embedder.
pub fn is_gguf(path: &Path) -> bool {
    let header = std::fs::File::open(path)
        .map_err(inference_tensor::Error::wrap)
        .and_then(|file| {
            inference_tensor::quantized::gguf_file::Header::read(&mut std::io::BufReader::new(file))
        });
    let Ok(header) = header else {
        return false;
    };
    let diffusion = header
        .metadata
        .get(GGUF_TYPE_KEY)
        .and_then(|v| v.to_string().ok())
        .is_some_and(|t| t == GGUF_DIFFUSION);
    diffusion && header.tensor_infos.contains_key(PATCH_EMBED_TENSOR)
}

/// The HiDream-O1 GGUF `model_id` names: the file itself, or a directory's only GGUF when it holds no checkpoint.
pub fn local_gguf(model_id: &Path) -> Option<PathBuf> {
    let is_gguf_file = |p: &Path| {
        p.extension()
            .is_some_and(|e| e == crate::gguf::GGUF_EXTENSION)
    };
    if model_id.is_file() {
        return (is_gguf_file(model_id) && is_gguf(model_id)).then(|| model_id.to_path_buf());
    }
    let names: Vec<String> = std::fs::read_dir(model_id)
        .ok()?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    // a checkpoint's own weights win over a GGUF written beside them
    if names.iter().any(|n| n == CONFIG_FILE) && !weight_shards(names.clone()).is_empty() {
        return None;
    }
    // several GGUFs could be anything, and probing each costs a header parse: the caller names the file
    let mut ggufs = names
        .iter()
        .map(|n| model_id.join(n))
        .filter(|p| is_gguf_file(p));
    match (ggufs.next(), ggufs.next()) {
        (Some(one), None) if is_gguf(&one) => Some(one),
        _ => None,
    }
}

/// The checkpoint's weight files among `names`, in shard order.
pub fn weight_shards(names: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut shards: Vec<String> = names.into_iter().filter(|n| is_weight_shard(n)).collect();
    shards.sort();
    shards
}

/// Whether `name` is one of the checkpoint's weight files: `model.safetensors` or `model-0000N-of-0000M.safetensors`.
fn is_weight_shard(name: &str) -> bool {
    let Some(stem) = name
        .strip_prefix("model")
        .and_then(|s| s.strip_suffix(".safetensors"))
    else {
        return false;
    };
    stem.is_empty()
        || stem
            .strip_prefix('-')
            .and_then(|s| s.split_once("-of-"))
            .is_some_and(|(a, b)| {
                [a, b]
                    .iter()
                    .all(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()))
            })
}

/// The resolution among [`RESOLUTIONS`] whose aspect is closest to `width` x `height`, first on ties.
pub fn closest_resolution(width: usize, height: usize) -> (usize, usize) {
    let ratio = width as f64 / height as f64;
    let mut best = RESOLUTIONS[0];
    let mut best_diff = f64::INFINITY;
    for (w, h) in RESOLUTIONS {
        let diff = (w as f64 / h as f64 - ratio).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (w, h);
        }
    }
    best
}

pub struct HiDreamO1Pipeline {
    model: HiDreamO1,
    tokenizer: Tokenizer,
    variant: Variant,
}

impl HiDreamO1Pipeline {
    pub fn new(model: HiDreamO1, tokenizer: Tokenizer, variant: Variant) -> Self {
        Self {
            model,
            tokenizer,
            variant,
        }
    }

    /// The model from a checkpoint directory (`config.json`, `tokenizer.json`, `model*.safetensors`), outside the engine.
    pub fn load(dir: &Path, variant: Variant, device: &Device, dtype: DType) -> Result<Self> {
        let names = std::fs::read_dir(dir)
            .map_err(|e| path_error(dir, &e))?
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok());
        let shards: Vec<PathBuf> = weight_shards(names).iter().map(|n| dir.join(n)).collect();
        // SAFETY: the weights are read-only for the model's life; a file changed underneath it is undefined
        let vb = unsafe {
            ShardedSafeTensors::sharded(&shards, dtype, device, None, Arc::new(|_| true))?
        };
        Self::from_parts(dir, vb, variant, device)
    }

    /// As [`HiDreamO1Pipeline::load`], with the weights from a GGUF and the config and tokenizer from `dir`.
    pub fn load_gguf(
        gguf: &Path,
        dir: &Path,
        variant: Variant,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let (vb, _) =
            crate::gguf::var_builder(gguf, crate::gguf::hidream_native_name, dtype, device)?;
        Self::from_parts(dir, vb, variant, device)
    }

    fn from_parts(
        dir: &Path,
        vb: inference_quant::ShardedVarBuilder,
        variant: Variant,
        device: &Device,
    ) -> Result<Self> {
        let config =
            std::fs::read_to_string(dir.join(CONFIG_FILE)).map_err(|e| path_error(dir, &e))?;
        let config = model::Config::from_json(&config).map_err(|e| path_error(dir, &e))?;
        let tokenizer =
            Tokenizer::from_file(dir.join(TOKENIZER_FILE)).map_err(|e| path_error(dir, &*e))?;
        let metadata = NormalLoadingMetadata {
            mapper: Box::new(DummyDeviceMapper {
                nm_device: device.clone(),
            }),
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(MultiProgress::new()),
            matformer_slicing_config: None,
            rope_pairing: None,
        };
        Ok(Self::new(
            HiDreamO1::new(&config, vb, metadata)?,
            tokenizer,
            variant,
        ))
    }

    pub fn model(&self) -> &HiDreamO1 {
        &self.model
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// The chat-templated prompt: the references' image pads, the caption, then the image and timestep tokens.
    pub fn prompt_tokens(&self, prompt: &str, references: Option<&References>) -> Result<Vec<u32>> {
        let pads = references.map(References::vit_tokens).unwrap_or_default();
        let images = IMAGE_PLACEHOLDER.repeat(pads.len());
        let text = format!("{PROMPT_PREFIX}{images}{prompt}{PROMPT_SUFFIX}");
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(inference_tensor::Error::msg)?
            .get_ids()
            .to_vec();
        let pad = self.model.image_token_id();
        let mut pads = pads.into_iter();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            match (id == pad).then(|| pads.next()).flatten() {
                Some(n) => out.extend(std::iter::repeat_n(pad, n)),
                None => out.push(id),
            }
        }
        Ok(out)
    }

    /// The conditional sequence and, under guidance, the unconditional one: both see the same references.
    pub fn sequences(
        &self,
        prompt: &str,
        (rows, cols): (usize, usize),
        guided: bool,
        references: Option<&References>,
    ) -> Result<(model::Sequence, Option<model::Sequence>)> {
        let layout = |prompt: &str| {
            self.model.sequence(
                &self.prompt_tokens(prompt, references)?,
                rows,
                cols,
                references,
            )
        };
        let uncond = guided.then(|| layout(UNCONDITIONAL_PROMPT)).transpose()?;
        Ok((layout(prompt)?, uncond))
    }

    /// The final patches (1, patches, 3072) of a `width` x `height` image (multiples of 32), seeing any `references`.
    pub fn generate(
        &self,
        prompt: &str,
        (width, height): (usize, usize),
        settings: &Settings,
        noise: &mut dyn Noise,
        references: Option<&References>,
        observe: &mut dyn FnMut(usize, &Tensor),
    ) -> Result<Tensor> {
        if width % PATCH_SIZE != 0 || height % PATCH_SIZE != 0 {
            inference_tensor::bail!("{width}x{height} is not a multiple of {PATCH_SIZE}");
        }
        let (rows, cols) = (height / PATCH_SIZE, width / PATCH_SIZE);
        let device = self.model.device();
        let dtype = self.model.dtype();
        let guided = settings.guidance_scale > 1.0;
        let (cond, uncond) = self.sequences(prompt, (rows, cols), guided, references)?;
        // the ViT and the reference patches' embedding are the same for both samples and every step
        let conditioning: Option<Conditioning> =
            references.map(|r| self.model.condition(r)).transpose()?;
        let conditioning = conditioning.as_ref();
        let mut z = noise
            .sample(rows * cols, device)?
            .affine(f64::from(settings.noise_scale), 0.0)?
            .to_dtype(dtype)?;
        let schedule = &settings.schedule;
        let flow_times: Vec<f32> = schedule
            .timesteps
            .iter()
            .map(|step_t| 1.0 - step_t / NUM_TRAIN_TIMESTEPS as f32)
            .collect();
        let t_embs = self.model.timestep_embeddings(&flow_times)?;
        let mut unipc = UniPc::default();
        let mut ahead = None;
        for (i, &step_t) in schedule.timesteps.iter().enumerate() {
            let t_emb = t_embs.narrow(0, i, 1)?;
            let sigma = (step_t / NUM_TRAIN_TIMESTEPS as f32).max(T_EPS);
            let z32 = z.to_dtype(DType::F32)?;
            let velocity = |x_pred: Tensor| -> Result<Tensor> {
                x_pred
                    .to_dtype(DType::F32)?
                    .sub(&z32)?
                    .affine(1.0 / f64::from(sigma), 0.0)
            };
            let x_pred = self.model.forward(&cond, &z, &t_emb, conditioning)?;
            observe(i, &x_pred);
            let mut v = velocity(x_pred)?;
            if let Some(uncond) = &uncond {
                let v_uncond = velocity(self.model.forward(uncond, &z, &t_emb, conditioning)?)?;
                v = (&v_uncond + v.sub(&v_uncond)?.affine(settings.guidance_scale, 0.0)?)?;
            }
            let model_output = v.neg()?;
            let next = match settings.sampler {
                Sampler::UniPc => unipc.step(&model_output, &z32, i, schedule)?,
                Sampler::Flash { clip_std } => sampling::flash_step(
                    &model_output,
                    &z32,
                    &noise.sample(rows * cols, device)?,
                    schedule.sigmas[i],
                    schedule.sigmas[i + 1],
                    settings.noise_scale,
                    clip_std,
                )?,
                Sampler::Euler => sampling::euler_step(
                    &model_output,
                    &z32,
                    schedule.sigmas[i],
                    schedule.sigmas[i + 1],
                )?,
            };
            z = next.to_dtype(dtype)?;
            // a step is seconds of GPU work: queued far ahead, the thread spins a core on the full launch queue, so it
            // stays one step ahead and sleeps on the one before
            if let Some(previous) = ahead.replace(device.fence()?) {
                previous.wait()?;
            }
        }
        Ok(z)
    }
}

fn path_error(dir: &Path, e: &dyn std::fmt::Display) -> inference_tensor::Error {
    inference_tensor::Error::msg(format!("{}: {e}", dir.display()))
}

/// Patches (1, rows * cols, 3072) in [-1, 1] as an image (1, 3, height, width) in [0, 255].
pub fn patches_to_image(z: &Tensor, width: usize, height: usize) -> Result<Tensor> {
    let (rows, cols) = (height / PATCH_SIZE, width / PATCH_SIZE);
    // the reference adds 1 in the model's dtype, then halves in F32
    z.affine(1.0, 1.0)?
        .to_dtype(DType::F32)?
        .affine(0.5, 0.0)?
        .reshape((1, rows, cols, model::CHANNELS, PATCH_SIZE, PATCH_SIZE))?
        .permute((0, 3, 1, 4, 2, 5))?
        .reshape((1, model::CHANNELS, height, width))?
        .clamp(0.0, 1.0)?
        .affine(255.0, 0.0)?
        .round()
}

impl DiffusionModel for HiDreamO1Pipeline {
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> Result<Tensor> {
        let images_in = &params.references.0;
        let settings =
            self.variant
                .settings(params.steps, params.guidance_scale, images_in.len())?;
        let snapped = closest_resolution(params.width, params.height);
        let (references, (width, height)) = if images_in.is_empty() {
            (None, snapped)
        } else {
            let size = (!params.match_reference).then_some(snapped);
            let (references, size) = References::prepare(images_in, size, self.model.device())?;
            (Some(references), size)
        };
        let mut images = Vec::with_capacity(prompts.len());
        for prompt in &prompts {
            let seed = params.seed.unwrap_or_else(|| rand::rng().random());
            let mut noise = SeededNoise::new(seed);
            let z = self.generate(
                prompt,
                (width, height),
                &settings,
                &mut noise,
                references.as_ref(),
                &mut |_, _| {},
            )?;
            images.push(patches_to_image(&z, width, height)?.i(0)?);
        }
        Tensor::stack(&images, 0)
    }

    fn device(&self) -> &Device {
        self.model.device()
    }

    fn validate(&self, params: &DiffusionGenerationParams) -> std::result::Result<(), String> {
        let images = &params.references.0;
        let references = images.len();
        if references > 0 && !self.model.sees_images() {
            return Err(model::NO_VISION_TOWER.to_string());
        }
        if references > MAX_REFERENCE_IMAGES {
            return Err(format!(
                "HiDream-O1 takes at most {MAX_REFERENCE_IMAGES} reference images"
            ));
        }
        if params.match_reference && references != 1 {
            return Err("only a single reference image can set the output size".to_string());
        }
        if references > 0 {
            let dims: Vec<(usize, usize)> = images
                .iter()
                .map(|i| (i.width() as usize, i.height() as usize))
                .collect();
            let size =
                (!params.match_reference).then(|| closest_resolution(params.width, params.height));
            References::output_size(&dims, size).map_err(|err| err.to_string())?;
        }
        self.variant
            .settings(params.steps, params.guidance_scale, references)
            .map(|_| ())
            .map_err(|err| err.to_string())
    }

    fn max_seq_len(&self) -> usize {
        usize::MAX
    }
}
