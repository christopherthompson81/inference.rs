use super::{
    AdapterPaths, AnyMoePipelineMixin, Cache, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, InputProcessorOutput, InputsProcessor, InputsProcessorType, IsqPipelineMixin,
    Loader, MessagesAction, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, Processor, TokenSource,
};
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{WorkerTransferData, use_ring};
use crate::paged_attention::PagedAttentionMeta;
use crate::pipeline::LoadOptions;
use crate::pipeline::tokens::get_token;
use crate::pipeline::{ChatTemplate, EmbeddingModulePaths, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::varbuilder_utils::from_mmaped_safetensors;
use crate::{DeviceMapSetting, MessageContent, Pipeline, SpeechGenerationConfig, distributed};
use anyhow::Result;
use futures::future::BoxFuture;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use indexmap::IndexMap;
use inference_models_speech::kokoro::{GGUF_EXTENSION, KokoroConfig, KokoroTts, is_kokoro_gguf};
use inference_models_speech::{DiaConfig, DiaPipeline, SpeechGenerationOutput, SpeechOptions};
use inference_quant::IsqType;
use inference_tensor::nn::VarBuilder;
use inference_tensor::{Device, Tensor};
use rand_isaac::Isaac64Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::env;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

const DIA_DAC_MODEL: &str = "EricB/dac_44khz";
const KOKORO_VOICES_DIR: &str = "voices";
const KOKORO_SAFETENSORS: &str = "model.safetensors";
const KOKORO_CONFIG: &str = "config.json";

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, strum::EnumIter)]
pub enum SpeechLoaderType {
    #[serde(rename = "dia")]
    Dia,
    #[serde(rename = "kokoro")]
    Kokoro,
}

impl FromStr for SpeechLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dia" => Ok(Self::Dia),
            "kokoro" => Ok(Self::Kokoro),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `dia`, `kokoro`."
            )),
        }
    }
}

impl SpeechLoaderType {
    /// Auto-detect speech loader type from a config.json string.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        if serde_json::from_str::<KokoroConfig>(config).is_ok() {
            return Some(Self::Kokoro);
        }
        if DiaConfig::from_json(config).is_ok() {
            return Some(Self::Dia);
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct SpeechModelPaths {
    weights: Vec<PathBuf>,
    config: PathBuf,
    voices: Vec<PathBuf>,
}

impl ModelPaths for SpeechModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.weights
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        unreachable!("Use `std::any::Any`.")
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        unreachable!("Use `std::any::Any`.")
    }
}

pub struct SpeechProcessor;

impl Processor for SpeechProcessor {
    fn process(
        &self,
        _pipeline: &dyn Pipeline,
        _messages: Vec<IndexMap<String, MessageContent>>,
        _add_generation_prompt: bool,
        _add_special_tokens: bool,
        _enable_thinking: Option<bool>,
        _reasoning_effort: Option<crate::request::ReasoningEffort>,
        _tools: Vec<crate::Tool>,
    ) -> Result<(Vec<u32>, String)> {
        anyhow::bail!(
            "SpeechProcessor::process should not be used. It does not expect chat messages."
        )
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(SpeechInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        // Just a default
        MessagesAction::FlattenOnlyText
    }
}

pub struct SpeechInputsProcessor;

#[derive(Clone)]
pub struct ModelInputs {
    pub(crate) prompts: Vec<String>,
    pub(crate) options: Vec<SpeechOptions>,
}

impl InputsProcessor for SpeechInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Text
    }

    fn process_inputs(
        &self,
        _tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _is_prompt: bool,
        _device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _sliding_window: Option<usize>,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        _mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let inputs = ModelInputs {
            prompts: input_seqs
                .iter()
                .map(|seq| seq.get_initial_prompt().to_string())
                .collect(),
            options: input_seqs
                .iter()
                .map(|seq| seq.speech_options().cloned().unwrap_or_default())
                .collect(),
        };
        Ok(InputProcessorOutput {
            inputs: Box::new(inputs),
            seq_indices: (0..input_seqs.len()).collect::<Vec<_>>(),
        })
    }
}

enum SpeechModel {
    Dia(Box<DiaPipeline>),
    Kokoro(Box<KokoroTts>),
}

pub struct SpeechPipeline {
    model_id: String,
    model: SpeechModel,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: EitherCache,
    cfg: SpeechGenerationConfig,
}

pub struct SpeechLoader {
    pub model_id: String,
    pub dac_model_id: Option<String>,
    /// Unset reads the architecture from the model's `config.json`.
    pub arch: Option<SpeechLoaderType>,
    pub cfg: Option<SpeechGenerationConfig>,
}

/// A local Kokoro GGUF: the file itself, or the one `.gguf` in a directory without the release's `config.json`.
pub(crate) fn local_kokoro_gguf(model_id: &str) -> Option<PathBuf> {
    let path = std::path::Path::new(model_id);
    let file = if path.is_dir() {
        if path.join(KOKORO_CONFIG).exists() {
            return None;
        }
        let mut ggufs = std::fs::read_dir(path)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == GGUF_EXTENSION));
        match (ggufs.next(), ggufs.next()) {
            (Some(only), None) => only,
            _ => return None,
        }
    } else {
        path.to_path_buf()
    };
    is_kokoro_gguf(&file).then_some(file)
}

fn detect_arch(config: &std::path::Path) -> Result<SpeechLoaderType> {
    if is_kokoro_gguf(config) {
        return Ok(SpeechLoaderType::Kokoro);
    }
    SpeechLoaderType::auto_detect_from_config(&std::fs::read_to_string(config)?).ok_or_else(|| {
        anyhow::anyhow!(
            "`{}` is not a Dia or Kokoro config; pass the architecture explicitly",
            config.display()
        )
    })
}

impl Loader for SpeechLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let silent = options.silent;
        let _progress_guard = ProgressScopeGuard::new(silent);
        if let Some(gguf) = local_kokoro_gguf(&self.model_id) {
            if let Some(arch) = self.arch.filter(|arch| *arch != SpeechLoaderType::Kokoro) {
                anyhow::bail!("`{}` is a Kokoro GGUF, not {arch:?}", gguf.display())
            }
            let paths = SpeechModelPaths {
                weights: vec![gguf.clone()],
                config: gguf,
                voices: Vec::new(),
            };
            return self.load_model_from_path(&paths, options);
        }
        let arch = match self.arch {
            Some(arch) => arch,
            None => {
                let api = ApiBuilder::new()
                    .with_progress(!silent)
                    .with_token(get_token(&token_source)?)
                    .build()?;
                let rev = revision.clone().unwrap_or_else(|| "main".to_string());
                let api = api.repo(Repo::with_revision(
                    self.model_id.clone(),
                    RepoType::Model,
                    rev.clone(),
                ));
                let id = std::path::Path::new(&self.model_id);
                detect_arch(&crate::pipeline::hf::get_file(
                    &api,
                    id,
                    "config.json",
                    &rev,
                )?)?
            }
        };
        if arch == SpeechLoaderType::Kokoro {
            let paths = kokoro_paths(&self.model_id, revision, &token_source, silent)?;
            return self.load_model_from_path(&paths, options);
        }
        let paths: anyhow::Result<Box<dyn ModelPaths>> = {
            // Main weights first, DAC is the final one.
            let mut weights = Vec::new();

            // Main model
            let config = {
                let api = ApiBuilder::new()
                    .with_progress(!silent)
                    .with_token(get_token(&token_source)?)
                    .build()?;
                let revision = revision.clone().unwrap_or("main".to_string());
                let api = api.repo(Repo::with_revision(
                    self.model_id.to_string(),
                    RepoType::Model,
                    revision.clone(),
                ));
                let model_id = std::path::Path::new(&self.model_id);

                let weight =
                    crate::pipeline::hf::get_file(&api, model_id, "model.safetensors", &revision)?;
                let config =
                    crate::pipeline::hf::get_file(&api, model_id, "config.json", &revision)?;
                weights.push(weight);
                config
            };

            // DAC model
            {
                let api = ApiBuilder::new()
                    .with_progress(!silent)
                    .with_token(get_token(&token_source)?)
                    .build()?;
                let revision = revision.unwrap_or("main".to_string());

                // Apply default here
                let dac_model = self
                    .dac_model_id
                    .clone()
                    .unwrap_or_else(|| DIA_DAC_MODEL.to_string());

                let api = api.repo(Repo::with_revision(
                    dac_model.clone(),
                    RepoType::Model,
                    revision.clone(),
                ));
                let model_id = std::path::Path::new(&dac_model);

                let weight =
                    crate::pipeline::hf::get_file(&api, model_id, "model.safetensors", &revision)?;
                weights.push(weight);
            }

            Ok(Box::new(SpeechModelPaths {
                weights,
                config,
                voices: Vec::new(),
            }))
        };
        self.load_model_from_path(paths?.as_ref(), options)
    }

    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let LoadOptions {
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config: _,
        } = options;
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths = paths
            .as_any()
            .downcast_ref::<SpeechModelPaths>()
            .expect("Path downcast failed.");

        if matches!(mapper, DeviceMapSetting::Map(_)) {
            anyhow::bail!("Device mapping is not supported for speech models.")
        }

        let arch = match self.arch {
            Some(arch) => arch,
            None => detect_arch(&paths.config)?,
        };
        if arch == SpeechLoaderType::Kokoro {
            if in_situ_quant.is_some() {
                anyhow::bail!("Kokoro does not support in-situ quantization.")
            }
            let [weights] = paths.weights.as_slice() else {
                anyhow::bail!(
                    "Kokoro loads from one weight file, got {}",
                    paths.weights.len()
                )
            };
            let model = KokoroTts::load(&paths.config, weights, &paths.voices, device)?;
            return Ok(Arc::new(Mutex::new(SpeechPipeline::new(
                self.model_id.clone(),
                SpeechModel::Kokoro(Box::new(model)),
                inference_tensor::DType::F32,
                match self.cfg {
                    None => SpeechGenerationConfig::kokoro_default(),
                    Some(cfg @ SpeechGenerationConfig::Kokoro { .. }) => cfg,
                    Some(cfg) => anyhow::bail!("a Kokoro model was given {cfg:?}"),
                },
            ))));
        }

        inference_quant::set_immediate_isq(
            in_situ_quant,
            vec![Regex::new(".*")?],
            inference_quant::IsqCaptureMode::Immediate,
        );

        let cfg = DiaConfig::from_json(&std::fs::read_to_string(&paths.config)?)?;

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }
        let use_nccl = inference_quant::distributed::use_nccl();
        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { worker_rank, .. } = payload;
            vec![inference_tensor::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl || use_ring() {
            vec![inference_tensor::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };

        let mapper =
            DeviceMapSetting::dummy().into_mapper(usize::MAX, device, None, &available_devices)?;
        let dtype = mapper.get_min_dtype(dtype)?;

        // Last weight is the dac.
        let model_weights = paths.weights[..paths.weights.len() - 1].to_vec();
        let vb = from_mmaped_safetensors(
            model_weights,
            Some(dtype),
            device,
            vec![None],
            silent,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;

        let dac_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[paths.weights.last().unwrap()], dtype, device)?
        };

        let model = DiaPipeline::new(&cfg, vb, dac_vb)?;
        Ok(Arc::new(Mutex::new(SpeechPipeline::new(
            self.model_id.clone(),
            SpeechModel::Dia(Box::new(model)),
            dtype,
            match self.cfg {
                None => SpeechGenerationConfig::dia_default(),
                Some(cfg @ SpeechGenerationConfig::Dia { .. }) => cfg,
                Some(cfg) => anyhow::bail!("a Dia model was given {cfg:?}"),
            },
        ))))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

impl SpeechPipeline {
    fn new(
        model_id: String,
        model: SpeechModel,
        activation_dtype: inference_tensor::DType,
        cfg: SpeechGenerationConfig,
    ) -> Self {
        Self {
            model_id,
            model,
            metadata: Arc::new(GeneralMetadata {
                max_seq_len: 1024,
                llg_factory: None,
                no_prefix_cache: false,
                num_hidden_layers: 1, // read only to size caches
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true, // NOTE(EricLBuehler): no cache for these.
                activation_dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Audio],
                },
                loaded_for_uqff_write: false,
            }),
            dummy_cache: EitherCache::Full(Cache::new(0)),
            cfg,
        }
    }
}

/// Kokoro's files: `config.json`, the `.pth` release (or `model.safetensors`), and every voice pack under `voices/`.
fn kokoro_paths(
    model_id: &str,
    revision: Option<String>,
    token_source: &TokenSource,
    silent: bool,
) -> Result<SpeechModelPaths> {
    let revision = revision.unwrap_or_else(|| "main".to_string());
    let api = ApiBuilder::new()
        .with_progress(!silent)
        .with_token(get_token(token_source)?)
        .build()?
        .repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.clone(),
        ));
    let id = std::path::Path::new(model_id);
    let files: Vec<String> = if id.is_dir() {
        let mut files = Vec::new();
        // a missing voices directory falls through to the loader's "needs a voice pack" error
        let voices = std::fs::read_dir(id.join(KOKORO_VOICES_DIR))
            .into_iter()
            .flatten();
        for entry in std::fs::read_dir(id)?.chain(voices) {
            let path = entry?.path();
            if let Ok(rel) = path.strip_prefix(id) {
                files.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        files
    } else {
        crate::pipeline::hf::list_repo_files(&api, id, true, &revision)?
    };
    let weight = files
        .iter()
        .find(|f| f.as_str() == KOKORO_SAFETENSORS)
        .or_else(|| {
            files
                .iter()
                .find(|f| !f.contains('/') && f.ends_with(".pth"))
        })
        .ok_or_else(|| anyhow::anyhow!("{model_id} has no `.pth` or `{KOKORO_SAFETENSORS}`"))?;
    let mut voices = files
        .iter()
        .filter(|f| {
            f.strip_prefix(KOKORO_VOICES_DIR)
                .and_then(|f| f.strip_prefix('/'))
                .is_some_and(|f| f.ends_with(".pt") || f.ends_with(".bin"))
        })
        .map(|f| crate::pipeline::hf::get_file(&api, id, f, &revision))
        .collect::<Result<Vec<_>>>()?;
    voices.sort();
    Ok(SpeechModelPaths {
        weights: vec![crate::pipeline::hf::get_file(&api, id, weight, &revision)?],
        config: crate::pipeline::hf::get_file(&api, id, "config.json", &revision)?,
        voices,
    })
}

impl PreProcessingMixin for SpeechPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(SpeechProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for SpeechPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("Speech models do not support ISQ for now.")
    }
}

impl CacheManagerMixin for SpeechPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) -> inference_tensor::Result<()> {
        Ok(())
    }
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) -> inference_tensor::Result<()> {
        Ok(())
    }
    fn cache(&self) -> &EitherCache {
        &self.dummy_cache
    }
}

impl MetadataMixin for SpeechPipeline {
    fn device(&self) -> Device {
        match &self.model {
            SpeechModel::Dia(model) => model.device().clone(),
            SpeechModel::Kokoro(model) => model.device().clone(),
        }
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        None
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

impl Pipeline for SpeechPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> inference_tensor::Result<ForwardInputsResult> {
        assert!(!return_raw_logits);

        let ModelInputs { prompts, options } = *inputs.downcast().expect("Downcast failed.");
        let mut pcms = Vec::new();
        let mut rates = Vec::new();
        let mut channels_all = Vec::new();
        for (prompt, options) in prompts.iter().zip(&options) {
            let SpeechGenerationOutput {
                pcm,
                rate,
                channels,
            } = match (&self.model, self.cfg) {
                (SpeechModel::Dia(model), cfg) => model.generate(prompt, &cfg)?,
                (SpeechModel::Kokoro(model), SpeechGenerationConfig::Kokoro { speed }) => {
                    model.generate(options, speed, options.seed.unwrap_or_else(rand::random))?
                }
                (SpeechModel::Kokoro(_), _) => {
                    inference_tensor::bail!("Kokoro was given another model's speech config")
                }
            };
            pcms.push(pcm);
            rates.push(rate);
            channels_all.push(channels);
        }

        Ok(ForwardInputsResult::Speech {
            pcms,
            rates,
            channels: channels_all,
        })
    }

    fn sample_causal_gen<'a>(
        &'a self,
        _seqs: &'a mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &'a mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _srng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<(), inference_tensor::Error>> {
        Box::pin(std::future::ready(Err(inference_tensor::Error::Msg(
            "`sample_causal_gen` is incompatible with `SpeechPipeline`".to_string(),
        )
        .bt())))
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Speech
    }

    fn validate_speech_options(&self, options: &SpeechOptions) -> std::result::Result<(), String> {
        match &self.model {
            // OpenAI clients always send a voice, so only a field no client sends by default is refused
            SpeechModel::Dia(_) if options.phonemes.is_some() => {
                Err("Dia speaks `input` text; it does not take `phonemes`".to_string())
            }
            SpeechModel::Dia(_) => Ok(()),
            SpeechModel::Kokoro(model) => model.validate(options).map_err(|err| err.to_string()),
        }
    }
}

impl AnyMoePipelineMixin for SpeechPipeline {}
