//! HiDream-O1-Image text-to-image: one Qwen3-VL transformer that denoises 32 px RGB patches directly, no VAE.

pub mod model;
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
use inference_nn::model::{DiffusionModel, NormalLoadingMetadata};
use model::{HiDreamO1, PATCH_DIM, PATCH_SIZE};
use sampling::{DEV_TIMESTEPS, NUM_TRAIN_TIMESTEPS, Schedule, UniPc};

// the timestep embedding replaces this token, the prompt's last
const PROMPT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<|boi_token|><|tms_token|>";
const PROMPT_PREFIX: &str = "<|im_start|>user\n";
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

/// The undistilled model samples with UniPC and guidance; Dev with its fixed re-noising schedule and none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Full,
    Dev,
}

/// The step rule: UniPC's multistep predictor-corrector, or Dev's flash Euler that re-noises each step.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    UniPc,
    Flash { clip_std: f32 },
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
    /// The reference pipeline's settings; `steps` and `guidance_scale` override the full model's.
    pub fn settings(self, steps: Option<usize>, guidance_scale: Option<f64>) -> Result<Settings> {
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
                Settings {
                    schedule: Schedule::fixed(&DEV_TIMESTEPS),
                    guidance_scale: 0.0,
                    sampler: Sampler::Flash {
                        clip_std: DEV_NOISE_CLIP_STD,
                    },
                    noise_scale: DEV_NOISE_SCALE,
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
        let msg = |e: &dyn std::fmt::Display| {
            inference_tensor::Error::msg(format!("{}: {e}", dir.display()))
        };
        let config = std::fs::read_to_string(dir.join(CONFIG_FILE)).map_err(|e| msg(&e))?;
        let config = model::Config::from_json(&config).map_err(|e| msg(&e))?;
        let tokenizer = Tokenizer::from_file(dir.join(TOKENIZER_FILE)).map_err(|e| msg(&e))?;
        let names = std::fs::read_dir(dir)
            .map_err(|e| msg(&e))?
            .filter_map(|entry| entry.ok()?.file_name().into_string().ok());
        let shards: Vec<PathBuf> = weight_shards(names).iter().map(|n| dir.join(n)).collect();
        // SAFETY: the weights are read-only for the model's life; a file changed underneath it is undefined
        let vb = unsafe {
            ShardedSafeTensors::sharded(&shards, dtype, device, None, Arc::new(|_| true))?
        };
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

    /// The chat-templated prompt, ending in the image and timestep tokens.
    pub fn prompt_tokens(&self, prompt: &str) -> Result<Vec<u32>> {
        let text = format!("{PROMPT_PREFIX}{prompt}{PROMPT_SUFFIX}");
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(inference_tensor::Error::msg)?
            .get_ids()
            .to_vec())
    }

    /// The final patches (1, patches, 3072) of a `width` x `height` image, both multiples of 32.
    pub fn generate(
        &self,
        prompt: &str,
        (width, height): (usize, usize),
        settings: &Settings,
        noise: &mut dyn Noise,
        observe: &mut dyn FnMut(usize, &Tensor),
    ) -> Result<Tensor> {
        if width % PATCH_SIZE != 0 || height % PATCH_SIZE != 0 {
            inference_tensor::bail!("{width}x{height} is not a multiple of {PATCH_SIZE}");
        }
        let (rows, cols) = (height / PATCH_SIZE, width / PATCH_SIZE);
        let device = self.model.device();
        let dtype = self.model.dtype();
        let cond = self
            .model
            .sequence(&self.prompt_tokens(prompt)?, rows, cols)?;
        let uncond = (settings.guidance_scale > 1.0)
            .then(|| {
                self.model
                    .sequence(&self.prompt_tokens(UNCONDITIONAL_PROMPT)?, rows, cols)
            })
            .transpose()?;
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
            let x_pred = self.model.forward(&cond, &z, &t_emb)?;
            observe(i, &x_pred);
            let mut v = velocity(x_pred)?;
            if let Some(uncond) = &uncond {
                let v_uncond = velocity(self.model.forward(uncond, &z, &t_emb)?)?;
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
            };
            z = next.to_dtype(dtype)?;
        }
        Ok(z)
    }
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
        let (width, height) = closest_resolution(params.width, params.height);
        let settings = self.variant.settings(params.steps, params.guidance_scale)?;
        let mut images = Vec::with_capacity(prompts.len());
        for prompt in &prompts {
            let seed = params.seed.unwrap_or_else(|| rand::rng().random());
            let mut noise = SeededNoise::new(seed);
            let z = self.generate(
                prompt,
                (width, height),
                &settings,
                &mut noise,
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
        self.variant
            .settings(params.steps, params.guidance_scale)
            .map(|_| ())
            .map_err(|err| err.to_string())
    }

    fn max_seq_len(&self) -> usize {
        usize::MAX
    }
}
