use super::llg::build_llg_factory;
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, IsqPipelineMixin,
    MetadataMixin, ModelCategory, PreProcessingMixin,
};
use super::{
    GeneralMetadata, Loader, ModelKind, ModelPaths, QuantizationKind, TokenSource,
    text_models_inputs_processor::ModelInputs,
};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::DeviceMapper;
#[cfg(feature = "models-llama")]
use crate::models::quantized_llama::ModelWeights as QLlama;
use crate::pipeline::ChatTemplate;
use crate::pipeline::LoadOptions;
use crate::pipeline::chat_template::{GenerationConfig, calculate_eos_tokens};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::tokenizer::get_tokenizer;
use crate::pipeline::{Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::debug::DEBUG;
use crate::utils::debug::DeviceRepr;
use crate::utils::progress::ProgressScopeGuard;
use crate::{DeviceMapSetting, Pipeline, Topology};
use anyhow::Result;
use futures::future::BoxFuture;
use inference_nn::gguf::QuantizedModel;
use inference_quant::IsqType;
use inference_tensor::quantized::ggml_file;
use inference_tensor::{DType, Device, Tensor};
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::fs;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};

pub struct GGMLPipeline {
    model: Box<dyn QuantizedModel>,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    generation_defaults: Option<crate::ModelGenerationDefaults>,
}

/// A loader for a GGML model.
pub struct GGMLLoader {
    model_id: String,
    config: GGMLSpecificConfig,
    quantized_model_id: Option<String>,
    quantized_filename: Option<String>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    kind: ModelKind,
    jinja_explicit: Option<String>,
}

#[derive(Clone, Default)]
/// Config for a GGML loader.
pub struct GGMLSpecificConfig {
    pub gqa: usize,
    pub topology: Option<Topology>,
}

#[derive(Default)]
/// A builder for a GGML loader.
pub struct GGMLLoaderBuilder {
    model_id: Option<String>,
    config: GGMLSpecificConfig,
    quantized_model_id: String,
    quantized_filename: String,
    kind: ModelKind,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    jinja_explicit: Option<String>,
}

impl GGMLLoaderBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: GGMLSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Ggml,
        };

        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind,
            quantized_filename,
            quantized_model_id,
            no_kv_cache,
            jinja_explicit,
        }
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGMLLoader {
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            quantized_filename: Some(self.quantized_filename),
            quantized_model_id: Some(self.quantized_model_id),
            jinja_explicit: self.jinja_explicit,
        })
    }
}

impl Loader for GGMLLoader {
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
            mut paged_attn_config,
        } = options;
        let _progress_guard = ProgressScopeGuard::new(silent);
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGML model. This will not do anything."
            );
        }

        if matches!(mapper, DeviceMapSetting::Map(_)) {
            anyhow::bail!("Device mapping is not supported for diffusion models.")
        }

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for GGML models, disabling it.");

            paged_attn_config = None;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        info!(
            "Loading model `{}` on {}.",
            self.get_id(),
            device.device_pretty_repr()
        );

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }

        let mut file = std::fs::File::open(paths.get_weight_filenames().first().unwrap())?;
        let model = ggml_file::Content::read(&mut file, device)
            .map_err(|e| e.with_path(paths.get_weight_filenames().first().unwrap()))?;

        trace!("Model config: {:?}", model.hparams);

        if DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            let mut tensors = Vec::new();
            for (name, t) in &model.tensors {
                tensors.push(format!(
                    "name = `{name}`, shape = {:?}, dtype = {:?}",
                    t.shape().clone(),
                    t.dtype(),
                ));
            }
            fs::write(
                "inference_ggml_tensors.txt",
                serde_json::to_string_pretty(&tensors).expect("Serialization failed."),
            )?;

            info!(
                "Debug is enabled, wrote the names and information about each tensor to `inference_ggml_tensors.txt`."
            );
        }

        let _ = if paged_attn_config.is_none() {
            warn!("GGML does not currently support PagedAttention, running without");
            None
        } else {
            paged_attn_config
        };

        let internal_dtype = dtype.try_into_dtype(&[device]).unwrap();
        let model = ggml_model(model, self.config.gqa, internal_dtype)?;

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;
        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let chat_template = super::loading::load_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            self.chat_template.as_ref(),
            None,
        );

        let max_seq_len = model.max_seq_len();
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let num_hidden_layers = model.num_hidden_layers();
        let generation_defaults = gen_conf
            .as_ref()
            .and_then(GenerationConfig::generation_defaults);
        let eos = calculate_eos_tokens(&chat_template, gen_conf.as_ref(), &tokenizer);
        Ok(Arc::new(Mutex::new(GGMLPipeline {
            model,
            tokenizer: tokenizer.into(),
            chat_template: Arc::new(chat_template),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                no_kv_cache: self.no_kv_cache,
                no_prefix_cache: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                activation_dtype: internal_dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Text],
                },
                loaded_for_uqff_write: false,
            }),
            generation_defaults,
        })))
    }

    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let silent = options.silent;
        let _progress_guard = ProgressScopeGuard::new(silent);
        let quantized_filenames = vec![self.quantized_filename.as_ref().unwrap().clone()];
        let paths = super::paths::get_paths(
            super::paths::PathsRequest {
                model_id: &self.model_id,
                tokenizer_json: self.tokenizer_json.as_deref(),
                chat_template: self.chat_template.as_deref(),
                token_source: &token_source,
                revision,
                quantized_model_id: self.quantized_model_id.as_ref(),
                quantized_filenames: Some(&quantized_filenames),
                silent,
                loading_uqff: false,
            },
            None,
        );
        self.load_model_from_path(&paths?, options)
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for GGMLPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for GGMLPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

impl CacheManagerMixin for GGMLPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> inference_tensor::Result<()> {
        super::cache_manager::clone_in_cache_by_kind(self, seqs)
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        super::cache_manager::clone_out_cache_by_kind(self, seqs)
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> inference_tensor::Result<()> {
        super::cache_manager::set_none_cache_by_kind(
            self,
            seqs,
            modify_draft_cache,
            load_preallocated_cache,
        )
    }
    fn cache(&self) -> &EitherCache {
        self.model.cache()
    }
}

impl MetadataMixin for GGMLPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

impl Pipeline for GGMLPipeline {
    fn requires_uniform_completion_batch(&self) -> bool {
        false
    }

    fn supports_batched_cuda_sampling(&self) -> bool {
        true
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, inference_tensor::Error> {
        let ModelInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids: _,    // NOTE(EricLBuehler): ignore, it is for phi3
            paged_attn_meta: _, // NOTE(EricLBuehler): ignore it for ggml
            flash_meta: _,
            recurrent_batch_kind: _,
            adapter_leases: _adapter_leases,
        } = *inputs.downcast().expect("Downcast failed.");
        let logits = self
            .model
            .forward_step(&input_ids, &seqlen_offsets, context_lens)?;
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }
    fn sample_causal_gen<'a>(
        &'a self,
        seqs: &'a mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &'a mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> BoxFuture<'a, Result<(), inference_tensor::Error>> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng)
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
}

impl AnyMoePipelineMixin for GGMLPipeline {}

// GGML files carry no architecture; they are all Llama models.
#[cfg(feature = "models-llama")]
fn ggml_model(ct: ggml_file::Content, gqa: usize, dtype: DType) -> Result<Box<dyn QuantizedModel>> {
    use inference_nn::gguf::FromGGML;
    Ok(Box::new(QLlama::from_ggml(ct, gqa, dtype)?))
}

#[cfg(not(feature = "models-llama"))]
fn ggml_model(
    _ct: ggml_file::Content,
    _gqa: usize,
    _dtype: DType,
) -> Result<Box<dyn QuantizedModel>> {
    anyhow::bail!("GGML models are Llama models, which this build leaves out (`models-llama`)")
}
