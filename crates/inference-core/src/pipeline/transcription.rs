use super::{
    AdapterPaths, AnyMoePipelineMixin, Cache, CacheManagerMixin, EitherCache, ForwardInputsResult,
    GeneralMetadata, InputProcessorOutput, InputsProcessor, InputsProcessorType, IsqPipelineMixin,
    Loader, MessagesAction, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, Processor, TokenSource,
};
use crate::device_map::DeviceMapper;
use crate::paged_attention::PagedAttentionMeta;
use crate::pipeline::LoadOptions;
use crate::pipeline::tokens::get_token;
use crate::pipeline::{ChatTemplate, EmbeddingModulePaths, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::progress::ProgressScopeGuard;
use crate::{MessageContent, Pipeline};
use anyhow::Result;
use futures::future::BoxFuture;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use indexmap::IndexMap;
use inference_audio::AudioInput;
use inference_models_speech::parakeet::{MODEL_TYPES, Parakeet, ParakeetFiles};
use inference_quant::IsqType;
use inference_tensor::{Device, Tensor};
use rand_isaac::Isaac64Rng;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

const CONFIG: &str = "config.json";
const PROCESSOR_CONFIG: &str = "processor_config.json";
const TOKENIZER: &str = "tokenizer.json";
const WEIGHTS: &str = "model.safetensors";
// the engine's sequence bookkeeping wants a length; one-shot audio requests never reach it
const METADATA_MAX_SEQ_LEN: usize = 1024;

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, strum::EnumIter)]
pub enum TranscriptionLoaderType {
    #[serde(rename = "parakeet")]
    Parakeet,
}

impl FromStr for TranscriptionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "parakeet" => Ok(Self::Parakeet),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `parakeet`."
            )),
        }
    }
}

impl TranscriptionLoaderType {
    /// Detects the architecture from a `config.json`'s `model_type`.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        let value: serde_json::Value = serde_json::from_str(config).ok()?;
        let model_type = value.get("model_type")?.as_str()?;
        MODEL_TYPES
            .iter()
            .any(|(name, _)| *name == model_type)
            .then_some(Self::Parakeet)
    }
}

#[derive(Clone, Debug)]
pub struct TranscriptionModelPaths {
    files: ParakeetFiles,
}

impl ModelPaths for TranscriptionModelPaths {
    fn get_config_filename(&self) -> &PathBuf {
        &self.files.config
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        &self.files.tokenizer
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.files.weights
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

pub struct TranscriptionProcessor;

impl Processor for TranscriptionProcessor {
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
        anyhow::bail!("a transcription model takes audio, not chat messages")
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(TranscriptionInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

pub struct TranscriptionInputsProcessor;

struct ModelInputs {
    audios: Vec<AudioInput>,
}

impl InputsProcessor for TranscriptionInputsProcessor {
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
        let audios = input_seqs
            .iter()
            .map(|seq| {
                seq.audios()
                    .and_then(|a| a.first().cloned())
                    .ok_or_else(|| anyhow::anyhow!("a transcription request carries no audio"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs { audios }),
            seq_indices: (0..input_seqs.len()).collect::<Vec<_>>(),
        })
    }
}

pub struct TranscriptionPipeline {
    model_id: String,
    model: Parakeet,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: EitherCache,
}

pub struct TranscriptionLoader {
    pub model_id: String,
    /// Unset reads the architecture from the model's `config.json`.
    pub arch: Option<TranscriptionLoaderType>,
}

impl Loader for TranscriptionLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(options.silent);
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let api = ApiBuilder::new()
            .with_progress(!options.silent)
            .with_token(get_token(&token_source)?)
            .build()?
            .repo(Repo::with_revision(
                self.model_id.clone(),
                RepoType::Model,
                revision.clone(),
            ));
        let id = std::path::Path::new(&self.model_id);
        let get = |file: &str| crate::pipeline::hf::get_file(&api, id, file, &revision);
        let paths = TranscriptionModelPaths {
            files: ParakeetFiles {
                config: get(CONFIG)?,
                processor_config: get(PROCESSOR_CONFIG)?,
                tokenizer: get(TOKENIZER)?,
                weights: vec![get(WEIGHTS)?],
            },
        };
        self.load_model_from_path(&paths, options)
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
            in_situ_quant,
            ..
        } = options;
        let _progress_guard = ProgressScopeGuard::new(silent);
        if in_situ_quant.is_some() {
            anyhow::bail!("Transcription models do not support in-situ quantization.")
        }
        let paths = paths
            .as_any()
            .downcast_ref::<TranscriptionModelPaths>()
            .expect("Path downcast failed.");
        let config = std::fs::read_to_string(&paths.files.config)?;
        let arch = match self.arch {
            Some(arch) => arch,
            None => TranscriptionLoaderType::auto_detect_from_config(&config).ok_or_else(|| {
                anyhow::anyhow!(
                    "`{}` is not a Parakeet config; pass the architecture explicitly",
                    paths.files.config.display()
                )
            })?,
        };
        let dtype = dtype.try_into_dtype(&[device])?;
        let model = match arch {
            TranscriptionLoaderType::Parakeet => Parakeet::load(&paths.files, device, dtype)?,
        };
        Ok(Arc::new(Mutex::new(TranscriptionPipeline::new(
            self.model_id.clone(),
            model,
            dtype,
        ))))
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        ModelKind::Normal
    }
}

impl TranscriptionPipeline {
    fn new(model_id: String, model: Parakeet, activation_dtype: inference_tensor::DType) -> Self {
        Self {
            model_id,
            model,
            metadata: Arc::new(GeneralMetadata {
                max_seq_len: METADATA_MAX_SEQ_LEN,
                llg_factory: None,
                no_prefix_cache: true,
                num_hidden_layers: 1,
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true,
                activation_dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Audio],
                    output: vec![SupportedModality::Text],
                },
                loaded_for_uqff_write: false,
            }),
            dummy_cache: EitherCache::Full(Cache::new(0)),
        }
    }
}

impl PreProcessingMixin for TranscriptionPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(TranscriptionProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for TranscriptionPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("Transcription models do not support ISQ.")
    }
}

impl CacheManagerMixin for TranscriptionPipeline {
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

impl MetadataMixin for TranscriptionPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
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

impl Pipeline for TranscriptionPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> inference_tensor::Result<ForwardInputsResult> {
        assert!(!return_raw_logits);
        let ModelInputs { audios } = *inputs.downcast().expect("Downcast failed.");
        let transcripts = audios
            .iter()
            .map(|audio| {
                self.model
                    .transcribe(&audio.to_mono(), audio.sample_rate)
                    .map_err(|e| e.to_string())
            })
            .collect();
        Ok(ForwardInputsResult::Transcription { transcripts })
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
            "`sample_causal_gen` is incompatible with `TranscriptionPipeline`".to_string(),
        )
        .bt())))
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::Transcription
    }
}

impl AnyMoePipelineMixin for TranscriptionPipeline {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parakeet_model_types_are_detected() {
        for (model_type, _) in MODEL_TYPES {
            let config = format!(r#"{{"model_type": "{model_type}"}}"#);
            assert_eq!(
                TranscriptionLoaderType::auto_detect_from_config(&config),
                Some(TranscriptionLoaderType::Parakeet)
            );
        }
        assert_eq!(
            TranscriptionLoaderType::auto_detect_from_config(r#"{"model_type": "llama"}"#),
            None
        );
    }
}
