use std::{
    cmp::Ordering,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::Module;
use inference_tensor::{D, DType, Device, Result, Tensor};
use tokenizers::Tokenizer;
use tracing::info;

use inference_nn::model::DiffusionModel;
use inference_nn::utils::tokenizer::tokenizer_from_file;

use crate::{
    DiffusionGenerationParams,
    clip::text::{ClipConfig, ClipTextTransformer},
    flux,
    t5::{self, T5EncoderModel},
    utils::varbuilder_utils::{DeviceForLoadTensor, from_mmaped_safetensors},
};

use super::{autoencoder::AutoEncoder, model::Flux};

const HUB_REVISION: &str = "main";
const T5_TOKENIZER_REPO: &str = "EricB/t5_tokenizer";
const T5_XXL_REPO: &str = "EricB/t5-v1_1-xxl-enc-only";
const CLIP_REPO: &str = "openai/clip-vit-large-patch14";

/// Resolves `(repo_id, revision, file)` to a local path; the loader owns hub access and offline mode.
pub type RepoFileFetcher = Box<dyn Fn(&str, &str, &str) -> Result<PathBuf> + Send + Sync>;

const T5_XXL_SAFETENSOR_FILES: &[&str] =
    &["t5_xxl-shard-0.safetensors", "t5_xxl-shard-1.safetensors"];

#[derive(Clone, Copy, Debug)]
pub struct FluxStepperShift {
    pub base_shift: f64,
    pub max_shift: f64,
    pub guidance_scale: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FluxStepperConfig {
    pub num_steps: usize,
    pub guidance_config: Option<FluxStepperShift>,
    pub is_guidance: bool,
}

impl FluxStepperConfig {
    pub fn default_for_guidance(has_guidance: bool) -> Self {
        if has_guidance {
            Self {
                num_steps: 50,
                guidance_config: Some(FluxStepperShift {
                    base_shift: 0.5,
                    max_shift: 1.15,
                    guidance_scale: 4.0,
                }),
                is_guidance: true,
            }
        } else {
            Self {
                num_steps: 4,
                guidance_config: None,
                is_guidance: false,
            }
        }
    }
}

/// Local text-encoder weights to use instead of the hub's; configs and tokenizers still come from the hub.
#[derive(Debug, Clone, Default)]
pub struct LocalTextEncoders {
    /// T5-XXL encoder weights, a GGUF file in llama.cpp `t5encoder` naming.
    pub t5: Option<PathBuf>,
    /// CLIP-L text encoder weights, a safetensors file in HF naming.
    pub clip: Option<PathBuf>,
}

pub struct FluxStepperLoad<'a> {
    pub dtype: DType,
    pub device: &'a Device,
    pub silent: bool,
    pub offloaded: bool,
    pub fetch: RepoFileFetcher,
    pub text_encoders: LocalTextEncoders,
}

pub struct FluxStepper {
    cfg: FluxStepperConfig,
    t5_tok: Tokenizer,
    clip_tok: Tokenizer,
    clip_text: ClipTextTransformer,
    flux_model: Flux,
    flux_vae: AutoEncoder,
    is_guidance: bool,
    device: Device,
    dtype: DType,
    fetch: RepoFileFetcher,
    t5_weights: Option<PathBuf>,
    silent: bool,
    offloaded: bool,
}

fn get_t5_tokenizer(fetch: &RepoFileFetcher) -> anyhow::Result<Tokenizer> {
    let tokenizer_filename = fetch(
        T5_TOKENIZER_REPO,
        HUB_REVISION,
        "t5-v1_1-xxl.tokenizer.json",
    )?;
    tokenizer_from_file(tokenizer_filename.as_ref())
}

fn get_t5_model(
    fetch: &RepoFileFetcher,
    local_weights: Option<&Path>,
    dtype: DType,
    device: &Device,
    silent: bool,
    offloaded: bool,
) -> inference_tensor::Result<T5EncoderModel> {
    let repo_id = T5_XXL_REPO;

    let vb = match local_weights {
        Some(path) => crate::gguf::var_builder(path, crate::gguf::t5_native_name, dtype, device)?.0,
        None => from_mmaped_safetensors(
            T5_XXL_SAFETENSOR_FILES
                .iter()
                .map(|f| fetch(repo_id, HUB_REVISION, f))
                .collect::<inference_tensor::Result<Vec<_>>>()?,
            Some(dtype),
            device,
            vec![None],
            silent,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?,
    };
    let config_filename = fetch(repo_id, HUB_REVISION, "config.json")?;
    let config = std::fs::read_to_string(config_filename)?;
    let config: t5::Config = serde_json::from_str(&config).map_err(inference_tensor::Error::msg)?;

    t5::T5EncoderModel::load(vb, &config, device, offloaded)
}

fn get_clip_model_and_tokenizer(
    fetch: &RepoFileFetcher,
    local_weights: Option<&Path>,
    device: &Device,
    silent: bool,
) -> anyhow::Result<(ClipTextTransformer, Tokenizer)> {
    let repo_id = CLIP_REPO;

    let model_file = match local_weights {
        Some(path) => path.to_path_buf(),
        None => fetch(repo_id, HUB_REVISION, "model.safetensors")?,
    };
    let vb = from_mmaped_safetensors(
        vec![model_file],
        None,
        device,
        vec![None],
        silent,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?;
    let config_file = fetch(repo_id, HUB_REVISION, "config.json")?;
    let config: ClipConfig = serde_json::from_reader(File::open(config_file)?)?;
    let config = config.text_config;
    let model = ClipTextTransformer::new(vb.pp("text_model"), &config)?;

    let tokenizer_filename = fetch(repo_id, HUB_REVISION, "tokenizer.json")?;
    let tokenizer = tokenizer_from_file(tokenizer_filename.as_ref())?;

    Ok((model, tokenizer))
}

fn get_tokenization(tok: &Tokenizer, prompts: Vec<String>, device: &Device) -> Result<Tensor> {
    Tensor::new(
        tok.encode_batch(prompts, true)
            .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?
            .into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect::<Vec<_>>(),
        device,
    )
}

impl FluxStepper {
    pub fn new(
        cfg: FluxStepperConfig,
        (flux_vb, flux_cfg): (ShardedVarBuilder, &flux::model::Config),
        (flux_ae_vb, flux_ae_cfg): (ShardedVarBuilder, &flux::autoencoder::Config),
        FluxStepperLoad {
            dtype,
            device,
            silent,
            offloaded,
            fetch,
            text_encoders,
        }: FluxStepperLoad<'_>,
    ) -> anyhow::Result<Self> {
        info!("Loading T5 XXL tokenizer.");
        let t5_tokenizer = get_t5_tokenizer(&fetch)?;
        info!("Loading CLIP model and tokenizer.");
        let (clip_encoder, clip_tokenizer) =
            get_clip_model_and_tokenizer(&fetch, text_encoders.clip.as_deref(), device, silent)?;

        Ok(Self {
            cfg,
            t5_tok: t5_tokenizer,
            clip_tok: clip_tokenizer,
            clip_text: clip_encoder,
            flux_model: Flux::new(flux_cfg, flux_vb, device.clone(), offloaded)?,
            flux_vae: AutoEncoder::new(flux_ae_cfg, flux_ae_vb)?,
            is_guidance: cfg.is_guidance,
            device: device.clone(),
            dtype,
            fetch,
            t5_weights: text_encoders.t5,
            silent,
            offloaded,
        })
    }
}

impl DiffusionModel for FluxStepper {
    fn forward(
        &mut self,
        prompts: Vec<String>,
        params: DiffusionGenerationParams,
    ) -> Result<Tensor> {
        let mut t5_input_ids = get_tokenization(&self.t5_tok, prompts.clone(), &self.device)?;
        if !self.is_guidance {
            match t5_input_ids.dim(1)?.cmp(&256) {
                Ordering::Greater => {
                    inference_tensor::bail!(
                        "T5 embedding length greater than 256, please shrink the prompt or use the -dev (with guidance distillation) version."
                    )
                }
                Ordering::Less | Ordering::Equal => {
                    t5_input_ids =
                        t5_input_ids.pad_with_zeros(D::Minus1, 0, 256 - t5_input_ids.dim(1)?)?;
                }
            }
        }

        let t5_embed = {
            info!("Hotloading T5 XXL model.");
            let mut t5_encoder = get_t5_model(
                &self.fetch,
                self.t5_weights.as_deref(),
                self.dtype,
                &self.device,
                self.silent,
                self.offloaded,
            )?;
            t5_encoder.forward(&t5_input_ids)?
        };

        let clip_input_ids = get_tokenization(&self.clip_tok, prompts, &self.device)?;
        let clip_embed = self
            .clip_text
            .forward(&clip_input_ids)?
            .to_dtype(self.dtype)?;

        let img = flux::sampling::get_noise(
            t5_embed.dim(0)?,
            params.height,
            params.width,
            self.device(),
        )?
        .to_dtype(self.dtype)?;

        let state = flux::sampling::State::new(&t5_embed, &clip_embed, &img)?;
        let timesteps = flux::sampling::get_schedule(
            self.cfg.num_steps,
            self.cfg
                .guidance_config
                .map(|s| (state.img.dims()[1], s.base_shift, s.max_shift)),
        );

        let img = if let Some(guidance_cfg) = &self.cfg.guidance_config {
            flux::sampling::denoise(
                &mut self.flux_model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
                guidance_cfg.guidance_scale,
            )?
        } else {
            flux::sampling::denoise_no_guidance(
                &mut self.flux_model,
                &state.img,
                &state.img_ids,
                &state.txt,
                &state.txt_ids,
                &state.vec,
                &timesteps,
            )?
        };

        let latent_img = flux::sampling::unpack(&img, params.height, params.width)?;

        let img = self.flux_vae.decode(&latent_img)?;

        let normalized_img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;

        Ok(normalized_img)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_seq_len(&self) -> usize {
        if self.is_guidance { usize::MAX } else { 256 }
    }
}
