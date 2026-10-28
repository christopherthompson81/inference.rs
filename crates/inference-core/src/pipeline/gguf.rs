use super::{Loader, ModelKind, ModelPaths, QuantizationKind, TokenSource};
use crate::decision::{KEV_OPTION_END, LabelDecision, LabelFamily};
use crate::gguf::embedding_gguf::{
    DecoderEmbeddingGguf, EmbeddingGgufArchitecture, EmbeddingGgufPooling, add_encoder_specials,
    build_embedding_bindings, embedding_pooling, synthesize_embedding_config, task_adapter_prefix,
};
use crate::gguf::{
    GgufTokenizerConversion,
    base_model::infer_hf_base_model_id,
    convert_gguf_metadata_to_hf_tokenizer, get_gguf_chat_template_from_metadata,
    multimodal_bindings::build_gemma4_bindings,
    multimodal_vision_registry::resolve_native_multimodal_gguf,
    muse_glimmer_bindings::normalize_muse_glimmer_config,
    normal_bindings::build_normal_bindings,
    normal_config::{
        normal_loader_hint_from_external_config, normalize_external_normal_config,
        synthesize_normal_config, validate_normal_config_tensor_inventory,
    },
    normal_registry::{GgufDescriptor, RopePairing, resolve_native_adapter},
    qwen_multimodal_bindings::{
        build_qwen_multimodal_bindings, normalize_qwen_multimodal_config,
        qwen_multimodal_loader_type,
    },
    validate_external_gguf_tokenizer,
};
#[cfg(feature = "models-gemma")]
use crate::gguf::{
    gemma3_bindings::build_gemma3_text_bindings,
    gemma3_config::{
        ensure_gemma3_image_token, ensure_gemma3_vision_config,
        gemma3_text_uses_language_model_prefix, prepare_gemma3_text_config,
    },
};
use crate::pipeline::LoadOptions;
use crate::pipeline::chat_template::GenerationConfig;
use crate::pipeline::hf::{build_api, get_file, list_repo_files};
use crate::pipeline::loaders::stamp_qk_rope_layout;
use crate::pipeline::loading::PreparedSource;
use crate::pipeline::multimodal::{MultimodalLoaderBuilder, MultimodalSpecificConfig};
use crate::pipeline::normal::{NormalLoaderBuilder, NormalSpecificConfig};
use crate::pipeline::tokenizer::get_tokenizer;
use crate::pipeline::{EmbeddingLoaderBuilder, EmbeddingLoaderType, EmbeddingSpecificConfig};
use crate::utils::progress::ProgressScopeGuard;
use crate::{
    LocalModelPaths, LoraAdapterSpec, LoraRuntimeConfig, MultimodalLoaderType, Pipeline, Topology,
    UqffWriteConfig,
};
use anyhow::{Context, Result, bail};
use hf_hub::{Repo, RepoType};
use inference_tensor::Device;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

// the architecture whose full-attention layers can read a prompt bidirectionally
const BIDIRECTIONAL_ARCHITECTURE: &str = "qwen35";
// clef's GGUF is Qwen3.5 under its own name, with a joint decision head after the trunk
const CLEF_ARCHITECTURE: &str = "clef";
const CLEF_TRUNK: &str = "qwen35";
const JOINT_WIDTH_TENSOR: &str = "decision.proj_memory.weight";
const JOINT_FEEDFORWARD_TENSOR: &str = "dec.blk.0.ffn_up.weight";
// vllm-decision2's GGUF is Qwen3 under its own name, with an option head after the trunk
const DECISION2_ARCHITECTURE: &str = "vllm-decision2";
const DECISION2_TRUNK: &str = "qwen3";
const OPTION_HEAD_TENSOR: &str = "decision.key.weight";
// d1-omni's GGUF is `lfm2` read bidirectionally with laya's head; it runs as an embedder over d1-omni's prompts
const D1_OMNI_ARCHITECTURE: &str = "lfm2";
const D1_OMNI_TYPE: &str = "lfm2-d1-omni";
// the prompt length d1-omni was trained for, which llama.cpp's server cuts its prompts to
const D1_OMNI_MAX_TOKENS: usize = 16384;
const MASK_TOKEN_KEY: &str = "tokenizer.ggml.mask_token_id";
const DECISION_TEMPLATE_KEY: &str = "tokenizer.chat_template.systemone";
// llama.cpp's bounds on a score's level count
const SCORE_BIAS_LEVELS: std::ops::RangeInclusive<usize> = 2..=255;
const PROJECTOR_REQUIRED_ARCHITECTURES: &[&str] = &[
    "gemma3n",
    "gemma4",
    "llama4",
    "muse-glimmer",
    "qwen2vl",
    "qwen3vl",
    "qwen3vlmoe",
];

fn preferred_hf_config(files: &[String]) -> Option<&'static str> {
    if files.iter().any(|file| file == "config.json") {
        Some("config.json")
    } else if files.iter().any(|file| file == "params.json") {
        Some("params.json")
    } else {
        None
    }
}

fn requires_multimodal_projector(architecture: &str) -> bool {
    PROJECTOR_REQUIRED_ARCHITECTURES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(architecture))
}

/// Loader for a GGUF model.
pub struct GGUFLoader {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    mmproj_filenames: Option<Vec<String>>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    dynamic_lora: Option<DynamicLoraConfig>,
    kind: ModelKind,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
    encoder_cache_memory_bytes: Option<usize>,
}

#[derive(Clone, Default)]
/// Config for a GGUF loader.
pub struct GGUFSpecificConfig {
    pub topology: Option<Topology>,
    pub organization: crate::pipeline::IsqOrganization,
    pub write_uqff: Option<UqffWriteConfig>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub max_edge: Option<u32>,
    pub max_model_len: Option<usize>,
    pub hf_cache_path: Option<PathBuf>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
    /// Load the checkpoint's built-in MTP head (`nextn` layers) for speculative decoding.
    pub mtp: bool,
}

impl GGUFSpecificConfig {
    fn multimodal_config(&self) -> MultimodalSpecificConfig {
        MultimodalSpecificConfig {
            topology: self.topology.clone(),
            organization: self.organization,
            write_uqff: self.write_uqff.clone(),
            imatrix: self.imatrix.clone(),
            calibration_file: self.calibration_file.clone(),
            max_edge: self.max_edge,
            max_model_len: self.max_model_len,
            hf_cache_path: self.hf_cache_path.clone(),
            matformer_config_path: self.matformer_config_path.clone(),
            matformer_slice_name: self.matformer_slice_name.clone(),
            ..Default::default()
        }
    }
}

struct NativeMultimodalLoadArgs<'a> {
    paths: &'a dyn ModelPaths,
    mmproj_paths: &'a [PathBuf],
    options: LoadOptions<'a>,
}

fn prepare_native_multimodal_config(
    loader_type: &MultimodalLoaderType,
    config: &str,
) -> Result<String> {
    let config = super::isq::sanitize_quantized_weight_source_config(config)?;
    let config = normalize_qwen_multimodal_config(loader_type, &config)?;
    normalize_muse_glimmer_config(loader_type, &config)
}

struct NativeNormalLoadArgs<'a> {
    paths: &'a dyn ModelPaths,
    options: LoadOptions<'a>,
}

#[derive(Clone)]
struct DynamicLoraConfig {
    adapters: Vec<LoraAdapterSpec>,
    runtime: LoraRuntimeConfig,
}

struct ResolvedGgufTokenizer {
    conversion: GgufTokenizerConversion,
    generation_config_compatible: bool,
}

#[derive(Clone, Copy)]
enum TokenizerFallback {
    Strict,
    Automatic,
    ModelAssets,
}

fn resolve_tokenizer_candidate(
    path: &Path,
    fallback: TokenizerFallback,
    external: Result<GgufTokenizerConversion>,
    embedded: impl FnOnce() -> Result<GgufTokenizerConversion>,
) -> Result<ResolvedGgufTokenizer> {
    match external {
        Ok(conversion) => Ok(ResolvedGgufTokenizer {
            conversion,
            generation_config_compatible: true,
        }),
        Err(error) if !matches!(fallback, TokenizerFallback::Strict) => {
            match fallback {
                TokenizerFallback::Automatic => warn!(
                    "Ignoring automatically discovered tokenizer `{}` because it does not match \
                     the GGUF vocabulary: {error}",
                    path.display()
                ),
                TokenizerFallback::ModelAssets => warn!(
                    "Tokenizer from model assets `{}` does not match the GGUF vocabulary; using \
                     the tokenizer embedded in the GGUF instead: {error}",
                    path.display()
                ),
                TokenizerFallback::Strict => unreachable!(),
            }
            Ok(ResolvedGgufTokenizer {
                conversion: embedded().with_context(|| {
                    format!(
                        "Tokenizer `{}` was rejected ({error}); GGUF tokenizer conversion also \
                         failed",
                        path.display()
                    )
                })?,
                generation_config_compatible: false,
            })
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "Tokenizer `{}` is incompatible with the GGUF model vocabulary",
                path.display()
            )
        }),
    }
}

#[derive(Default)]
/// A builder for a GGUF loader.
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    mmproj_filenames: Option<Vec<String>>,
    kind: ModelKind,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    dynamic_lora: Option<DynamicLoraConfig>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
    encoder_cache_memory_bytes: Option<usize>,
}

impl GGUFLoaderBuilder {
    /// Create a loader builder for a GGUF model. `tok_model_id` optionally overrides embedded
    /// configuration and tokenizer assets. If `chat_template` is specified, it is treated as a path
    /// and used over remote files, removing all remote accesses.
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFSpecificConfig,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Gguf,
        };

        Self {
            chat_template,
            model_id: tok_model_id,
            kind,
            quantized_filenames,
            quantized_model_id,
            config,
            jinja_explicit,
            no_kv_cache,
            ..Default::default()
        }
    }

    pub fn with_mmproj_files(mut self, mmproj_filenames: Vec<String>) -> Self {
        self.mmproj_filenames = Some(mmproj_filenames);
        self
    }

    pub fn with_tokenizer_json(mut self, tokenizer_json: String) -> Self {
        self.tokenizer_json = Some(tokenizer_json);
        self
    }

    pub fn with_encoder_cache_memory_bytes(mut self, max_bytes: Option<usize>) -> Self {
        if let Some(max_bytes) = max_bytes {
            assert!(max_bytes > 0, "encoder cache memory must be nonzero");
        }
        self.encoder_cache_memory_bytes = max_bytes;
        self
    }

    pub fn with_dynamic_lora(
        mut self,
        adapters: Vec<LoraAdapterSpec>,
        runtime: LoraRuntimeConfig,
    ) -> Self {
        self.kind = ModelKind::GgufLora {
            quant: QuantizationKind::Gguf,
        };
        self.dynamic_lora = Some(DynamicLoraConfig { adapters, runtime });
        self
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFLoader {
            model_id: self.model_id,
            kind: self.kind,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            dynamic_lora: self.dynamic_lora,
            quantized_filenames: self.quantized_filenames,
            mmproj_filenames: self.mmproj_filenames,
            quantized_model_id: self.quantized_model_id,
            config: self.config,
            jinja_explicit: self.jinja_explicit,
            encoder_cache_memory_bytes: self.encoder_cache_memory_bytes,
        })
    }
}

const JINA_BERT_V3: &str = "jina-bert-v3";
const TASK_ADAPTER_FILE_PREFIX: &str = "lora-";
const GGUF_EXTENSION: &str = ".gguf";
const LORA_TASK_NAME: &str = "adapter.lora.task_name";

fn is_architecture(archive: &inference_quant::GgufArchive, architecture: &str) -> bool {
    matches!(
        archive.metadata_value("general.architecture"),
        Some(inference_tensor::quantized::gguf_file::Value::String(value)) if value == architecture
    )
}

fn is_task_adapter_file(name: &str) -> bool {
    name.starts_with(TASK_ADAPTER_FILE_PREFIX) && name.ends_with(GGUF_EXTENSION)
}

/// Merges the `lora-<task>-*.gguf` adapters beside `weights`, one per task, so the encoder runs each task.
fn merge_task_adapters(
    archive: &mut inference_quant::GgufArchive,
    weights: &[PathBuf],
) -> Result<()> {
    let Some(dir) = weights.first().and_then(|weight| weight.parent()) else {
        return Ok(());
    };
    let mut files = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_task_adapter_file)
        })
        .collect::<Vec<_>>();
    files.sort();
    let mut tasks = std::collections::HashSet::new();
    for file in files {
        let adapter = inference_quant::GgufArchive::open_file(&file)?;
        let Some(inference_tensor::quantized::gguf_file::Value::String(task)) =
            adapter.metadata_value(LORA_TASK_NAME).cloned()
        else {
            continue;
        };
        // a repository may carry one task in several precisions
        if tasks.insert(task.clone()) {
            info!(
                "Loading the `{task}` task adapter from `{}`",
                file.display()
            );
            archive.merge_adapter(adapter, &task_adapter_prefix(tasks.len() - 1))?;
        }
    }
    Ok(())
}

/// `config` as `class` reads it: the decoder translation's fields under an embedding model's class.
fn with_architecture(config: &str, class: &str) -> Result<String> {
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    config["architectures"] = serde_json::json!([class]);
    Ok(serde_json::to_string(&config)?)
}

/// The config with its text model's full-attention layers reading prompts bidirectionally.
fn bidirectional_config(config: &str) -> Result<String> {
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    let text = match config.get_mut("text_config") {
        Some(text) => text,
        None => &mut config,
    };
    text["is_causal"] = serde_json::Value::Bool(false);
    Ok(serde_json::to_string(&config)?)
}

/// A decision model's GGUF filed under its trunk's architecture (clef is Qwen3.5, vllm-decision2 is Qwen3), so the
/// trunk loads as that model does.
fn alias_decision_architecture(archive: &mut inference_quant::GgufArchive) -> Result<()> {
    for (architecture, trunk) in [
        (CLEF_ARCHITECTURE, CLEF_TRUNK),
        (DECISION2_ARCHITECTURE, DECISION2_TRUNK),
    ] {
        if is_architecture(archive, architecture) {
            archive.rename_architecture(trunk)?;
        }
    }
    Ok(())
}

/// The config with clef's joint head: its block and head counts and norm epsilon from the metadata, its widths from
/// the tensors (the converter writes no keys for them).
fn joint_head_config(config: &str, archive: &inference_quant::GgufArchive) -> Result<String> {
    let key = |name: &str| {
        let key = format!("{CLEF_TRUNK}.{name}");
        archive
            .metadata_value(&key)
            .with_context(|| format!("decision model has no `{key}`"))
    };
    let rows = |name: &str| -> Result<usize> {
        Ok(*archive
            .tensor_info(name)?
            .shape()
            .first()
            .with_context(|| format!("`{name}` has no shape"))?)
    };
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    let text = match config.get_mut("text_config") {
        Some(text) => text,
        None => &mut config,
    };
    text["joint_head"] = serde_json::json!({
        "width": rows(JOINT_WIDTH_TENSOR)?,
        "feedforward": rows(JOINT_FEEDFORWARD_TENSOR)?,
        "heads": key("decision.head_count")?.to_u32()?,
        "routing_layers": key("decision.routing_block_count")?.to_u32()?,
        "layers": key("decision.block_count")?.to_u32()?,
        "norm_eps": key("attention.layer_norm_epsilon")?.to_f32()?,
    });
    Ok(serde_json::to_string(&config)?)
}

/// The config with vllm-decision2's option head: its width from the key projection, its norm epsilon from the metadata.
fn option_head_config(config: &str, archive: &inference_quant::GgufArchive) -> Result<String> {
    let eps_key = format!("{DECISION2_TRUNK}.attention.layer_norm_epsilon");
    let norm_eps = archive
        .metadata_value(&eps_key)
        .with_context(|| format!("decision model has no `{eps_key}`"))?
        .to_f32()?;
    let head_dim = *archive
        .tensor_info(OPTION_HEAD_TENSOR)?
        .shape()
        .first()
        .with_context(|| format!("`{OPTION_HEAD_TENSOR}` has no shape"))?;
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    config["option_head"] = serde_json::json!({"head_dim": head_dim, "norm_eps": norm_eps});
    Ok(serde_json::to_string(&config)?)
}

/// d1-omni's config from its `lfm2.*` keys: per-block KV heads (0 for a short convolution) and feed-forward widths,
/// the trunk followed by the head.
fn d1_omni_config(archive: &inference_quant::GgufArchive) -> Result<String> {
    let key = |name: &str| {
        let key = format!("{D1_OMNI_ARCHITECTURE}.{name}");
        archive
            .metadata_value(&key)
            .with_context(|| format!("d1-omni GGUF has no `{key}`"))
    };
    // the converter writes per-block counts as signed or unsigned integers
    let count = |value: &inference_tensor::quantized::gguf_file::Value| -> Result<u64> {
        match value.to_i32() {
            Ok(v) => u64::try_from(v).context("a negative per-block count"),
            Err(_) => Ok(value.to_u64()?),
        }
    };
    let per_block =
        |name: &str| -> Result<Vec<u64>> { key(name)?.to_vec()?.iter().map(count).collect() };
    Ok(serde_json::to_string(&serde_json::json!({
        "architectures": [crate::pipeline::loaders::LFM2_DECISION_CLASS],
        "vocab_size": key("vocab_size")?.to_u32()?,
        "hidden_size": key("embedding_length")?.to_u32()?,
        "num_attention_heads": key("attention.head_count")?.to_u32()?,
        "num_key_value_heads": per_block("attention.head_count_kv")?,
        "intermediate_sizes": per_block("feed_forward_length")?,
        "head_layers": key("decision.block_count")?.to_u32()?,
        "rope_theta": key("rope.freq_base")?.to_f32()?,
        "max_position_embeddings": D1_OMNI_MAX_TOKENS,
        "rms_norm_eps": key("attention.layer_norm_rms_epsilon")?.to_f32()?,
        "layer_norm_eps": key("attention.layer_norm_epsilon")?.to_f32()?,
    }))?)
}

/// The label-coded decision profile a GGUF carries (`<arch>.decision.*` and its `systemone` template), if any.
fn label_decision(
    metadata: &HashMap<String, inference_tensor::quantized::gguf_file::Value>,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Option<LabelDecision>> {
    let Some(architecture) = metadata
        .get("general.architecture")
        .and_then(|value| value.to_string().ok())
    else {
        return Ok(None);
    };
    let prefix = format!("{architecture}.decision.");
    let Some(kind) = metadata
        .get(&format!("{prefix}type"))
        .and_then(|value| value.to_string().ok())
    else {
        return Ok(None);
    };
    let Some(family) = LabelFamily::from_name(kind) else {
        warn!("GGUF decision type `{kind}` is not supported; the model loads without decisions");
        return Ok(None);
    };
    let causal = metadata
        .get(&format!("{architecture}.attention.causal"))
        .and_then(|value| value.to_bool().ok());
    // d1-omni's encoder reads its prompt both ways by construction
    if (causal == Some(false) || family.non_causal())
        && architecture != BIDIRECTIONAL_ARCHITECTURE
        && !family.scores_markers()
    {
        warn!(
            "GGUF decision type `{kind}` needs non-causal attention, which `{architecture}` does not run; it loads without decisions"
        );
        return Ok(None);
    }
    let template = metadata
        .get(DECISION_TEMPLATE_KEY)
        .and_then(|value| value.to_string().ok())
        .with_context(|| format!("decision model has no `{DECISION_TEMPLATE_KEY}`"))?
        .clone();
    let mut temperatures = HashMap::new();
    let mut score_bias: HashMap<usize, Vec<f32>> = HashMap::new();
    let temperature_prefix = format!("{prefix}temperature.");
    let bias_prefix = format!("{prefix}score_bias.");
    for (key, value) in metadata {
        if let Some(name) = key.strip_prefix(&temperature_prefix) {
            let temperature = value.to_f32()?;
            anyhow::ensure!(temperature > 0.0, "invalid decision temperature `{key}`");
            temperatures.insert(name.to_string(), temperature);
        } else if let Some(name) = key.strip_prefix(&bias_prefix) {
            let parsed = name
                .split_once('.')
                .and_then(|(n, level)| {
                    Some((n.parse::<usize>().ok()?, level.parse::<usize>().ok()?))
                })
                .filter(|&(n, level)| SCORE_BIAS_LEVELS.contains(&n) && level < n);
            let Some((n, level)) = parsed else {
                bail!("invalid decision score bias `{key}`");
            };
            score_bias.entry(n).or_insert_with(|| vec![0.0; n])[level] = value.to_f32()?;
        }
    }
    let encode = |text: &str| {
        tokenizer
            .encode_fast(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
    };
    let mut decision = LabelDecision::new(family, template, temperatures, score_bias, encode)
        .map_err(anyhow::Error::msg)?;
    if family.labels_from_metadata() {
        let key = format!("{prefix}labels");
        let texts = metadata
            .get(&key)
            .with_context(|| format!("decision model has no `{key}`"))?
            .to_vec()?
            .iter()
            .map(|value| value.to_string().cloned())
            .collect::<inference_tensor::Result<Vec<_>>>()?;
        decision
            .set_labels(texts, encode)
            .map_err(anyhow::Error::msg)?;
    }
    if family.labels_per_question() {
        for form in crate::decision::d1_label_forms() {
            if let Ok([token]) = encode(&form).as_deref() {
                decision.single_tokens.insert(form, *token);
            }
        }
    }
    if family.prepares_texts() {
        let mut spellings = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .map(|token| token.content)
            .filter(|content| !content.is_empty())
            .collect::<Vec<_>>();
        // longest first, so a spelling that starts another is tried after it
        spellings.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        decision.quoted_spellings = spellings;
    }
    if family.scores_markers() {
        decision.marker = Some(
            metadata
                .get(MASK_TOKEN_KEY)
                .with_context(|| format!("decision model has no `{MASK_TOKEN_KEY}`"))?
                .to_u32()?,
        );
    }
    if family.reads_pointer() {
        decision.option_end = Some(
            tokenizer
                .token_to_id(KEV_OPTION_END)
                .with_context(|| format!("decision model has no `{KEV_OPTION_END}` token"))?,
        );
    }
    info!(
        "Decision model type `{kind}`, {} labels",
        decision.labels.len()
    );
    Ok(Some(decision))
}

impl GGUFLoader {
    fn dynamic_lora_adapters(&self) -> Option<&[LoraAdapterSpec]> {
        self.dynamic_lora
            .as_ref()
            .map(|config| config.adapters.as_slice())
    }

    /// The GGUF weights, tokenizer, generation config and chat template, with no processor configs.
    fn prepare_source(
        &self,
        archive: &Arc<inference_quant::GgufArchive>,
        bindings: &dyn inference_quant::GgufBindingResolver,
        paths: &dyn ModelPaths,
        options: &LoadOptions<'_>,
        config: String,
        rope_pairing: RopePairing,
    ) -> Result<PreparedSource> {
        let dtype = options.dtype.try_into_dtype(&[options.device])?;
        let source = Arc::new(
            inference_quant::GgufWeightSource::new(archive.clone(), bindings, dtype)?
                .with_multi_row_decode(options.multi_row_decode),
        );
        let tokenizer = self.resolve_tokenizer(paths, archive.metadata())?;
        let generation_config = self
            .resolve_generation_config(paths, &tokenizer)
            .or_else(|| GenerationConfig::from_gguf_metadata(archive.metadata()));
        let label_decision =
            label_decision(archive.metadata(), &tokenizer.conversion.tokenizer)?.map(Arc::new);
        // a converter may leave out the causal flag of a family trained with bidirectional prompts
        let config = match &label_decision {
            Some(decision) if decision.family.non_causal() => bidirectional_config(&config)?,
            Some(decision) if decision.family.joint_per_question() => {
                option_head_config(&config, archive)?
            }
            Some(decision) if decision.family.joint() => joint_head_config(&config, archive)?,
            _ => config,
        };
        let chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template_from_metadata(archive.metadata())?
            } else {
                None
            };
        Ok(PreparedSource {
            config,
            weights: source.sharded_var_builder(Device::Cpu),
            tokenizer: tokenizer.conversion.tokenizer,
            generation_config,
            chat_template,
            bos_token: tokenizer.conversion.bos,
            eos_token: tokenizer.conversion.eos,
            unk_token: tokenizer.conversion.unk,
            processor_config: None,
            preprocessor_config: None,
            source_weight_files: paths.get_weight_filenames().to_vec(),
            rope_pairing,
            label_decision,
        })
    }

    fn resolve_tokenizer(
        &self,
        paths: &dyn ModelPaths,
        metadata: &HashMap<String, inference_tensor::quantized::gguf_file::Value>,
    ) -> Result<ResolvedGgufTokenizer> {
        let (path, fallback) = if let Some(tokenizer_json) = self.tokenizer_json.as_ref() {
            (PathBuf::from(tokenizer_json), TokenizerFallback::Strict)
        } else if paths.get_tokenizer_filename().as_os_str().is_empty() {
            return Ok(ResolvedGgufTokenizer {
                conversion: convert_gguf_metadata_to_hf_tokenizer(metadata)?,
                generation_config_compatible: true,
            });
        } else {
            (
                paths.get_tokenizer_filename().clone(),
                if self.model_id.is_none() {
                    TokenizerFallback::Automatic
                } else {
                    TokenizerFallback::ModelAssets
                },
            )
        };

        let external = match get_tokenizer(&path, None) {
            Ok(tokenizer) => validate_external_gguf_tokenizer(tokenizer, metadata),
            Err(error) if matches!(fallback, TokenizerFallback::ModelAssets) => {
                return Err(error).with_context(|| {
                    format!(
                        "Failed to load tokenizer from model assets `{}`",
                        path.display()
                    )
                });
            }
            Err(error) => Err(error),
        };
        resolve_tokenizer_candidate(&path, fallback, external, || {
            convert_gguf_metadata_to_hf_tokenizer(metadata)
        })
    }

    fn resolve_generation_config(
        &self,
        paths: &dyn ModelPaths,
        tokenizer: &ResolvedGgufTokenizer,
    ) -> Option<GenerationConfig> {
        let filename = paths.get_gen_conf_filename()?;
        if !tokenizer.generation_config_compatible {
            warn!(
                "Ignoring generation config `{}` because the external tokenizer was incompatible \
                 with the GGUF model",
                filename.display()
            );
            return None;
        }
        let config = match fs::read_to_string(filename)
            .with_context(|| format!("Failed to read `{}`", filename.display()))
            .and_then(|raw| {
                serde_json::from_str::<GenerationConfig>(&raw)
                    .context("Failed to parse generation_config.json")
            }) {
            Ok(config) => config,
            Err(error) => {
                warn!("Ignoring generation config: {error:#}");
                return None;
            }
        };
        if let Err(error) =
            config.validate_token_ids(tokenizer.conversion.tokenizer.get_vocab_size(true))
        {
            warn!("Ignoring generation config: {error}");
            return None;
        }
        Some(config)
    }

    fn load_native_normal(
        &self,
        args: NativeNormalLoadArgs<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let NativeNormalLoadArgs { paths, options } = args;
        let mut archive = inference_quant::GgufArchive::open(paths.get_weight_filenames())?;
        // read through its decoder's translation, then loaded as an embedder
        let decoder_embedding = DecoderEmbeddingGguf::from_metadata(archive.metadata());
        if let Some(decoder) = decoder_embedding {
            archive.rename_architecture(decoder.decoder_architecture())?;
        }
        alias_decision_architecture(&mut archive)?;
        if is_architecture(&archive, JINA_BERT_V3) {
            merge_task_adapters(&mut archive, paths.get_weight_filenames())?;
        }
        if is_architecture(&archive, D1_OMNI_ARCHITECTURE)
            && archive
                .metadata_value(&format!("{D1_OMNI_ARCHITECTURE}.decision.type"))
                .is_some_and(|value| value.to_string().is_ok_and(|kind| kind == D1_OMNI_TYPE))
        {
            return self.load_d1_omni(Arc::new(archive), paths, options);
        }
        let archive = Arc::new(archive);
        let architecture = match archive.metadata_value("general.architecture") {
            Some(inference_tensor::quantized::gguf_file::Value::String(value)) => value.as_str(),
            Some(value) => {
                bail!("GGUF `general.architecture` must be a string, got {value:?}")
            }
            None => bail!("GGUF metadata is missing `general.architecture`"),
        };
        if let Some(encoder) = EmbeddingGgufArchitecture::from_architecture(architecture) {
            return self.load_native_embedding(archive, encoder, paths, options);
        }
        #[cfg(not(feature = "models-gemma"))]
        if architecture.eq_ignore_ascii_case("gemma3") {
            bail!("GGUF Gemma 3 models are not built in; enable the `models-gemma` feature");
        }
        #[cfg(feature = "models-gemma")]
        if architecture.eq_ignore_ascii_case("gemma3") {
            return self.load_native_gemma3_text(archive, NativeNormalLoadArgs { paths, options });
        }
        let metadata_keys = archive
            .metadata()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let tensor_names = archive
            .tensors()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let metadata_string = |key| match archive.metadata_value(key) {
            Some(inference_tensor::quantized::gguf_file::Value::String(value)) => {
                Some(value.as_str())
            }
            _ => None,
        };
        if requires_multimodal_projector(architecture) {
            bail!(
                "GGUF architecture `{architecture}` is supported only as a multimodal model. Pass \
                 its companion projector with `--mmproj <file>`, or load from a GGUF repository \
                 that publishes one; text-only `{architecture}` checkpoints are not supported"
            );
        }
        let descriptor = GgufDescriptor::new(architecture, &metadata_keys, &tensor_names)?
            .with_model_identity(
                metadata_string("general.name"),
                metadata_string("general.basename"),
            );

        // a repository's safetensors config describes the encoder, not the decoder translation the GGUF takes
        let external_config =
            if paths.get_config_filename().as_os_str().is_empty() || decoder_embedding.is_some() {
                None
            } else {
                Some(fs::read_to_string(paths.get_config_filename())?)
            };
        let explicit_loader = external_config
            .as_deref()
            .map(normal_loader_hint_from_external_config)
            .transpose()?;
        let explicit_loader = explicit_loader.flatten();
        let resolved = resolve_native_adapter(&descriptor, explicit_loader)?;
        let rope_pairing = decoder_embedding
            .and_then(DecoderEmbeddingGguf::rope_pairing)
            .unwrap_or(
                crate::gguf::normal_registry::schema_for(descriptor.architecture).rope_pairing,
            );
        debug!(
            "Loading GGUF architecture `{}` through native {:?} ({:?}, layouts {:?})",
            descriptor.architecture,
            resolved.adapter.loader,
            resolved.reason,
            resolved.adapter.layouts
        );
        let loader_type = resolved.adapter.loader.clone();
        let tensor_names = archive.tensors().keys().cloned().collect::<Vec<_>>();
        let config = match external_config {
            Some(config) => {
                let config = normalize_external_normal_config(
                    &loader_type,
                    descriptor.architecture,
                    &config,
                )?;
                validate_normal_config_tensor_inventory(&config, &tensor_names)?;
                config
            }
            None => synthesize_normal_config(&loader_type, archive.metadata(), &tensor_names)?,
        };
        let config = stamp_qk_rope_layout(&config, rope_pairing)?;
        let bindings = build_normal_bindings(&archive, &loader_type, descriptor.architecture)?;
        if let Some(decoder) = decoder_embedding {
            let config = with_architecture(&config, decoder.class())?;
            let mut source =
                self.prepare_source(&archive, &bindings, paths, &options, config, rope_pairing)?;
            add_encoder_specials(&mut source.tokenizer, archive.metadata())?;
            let loader_type = match decoder {
                DecoderEmbeddingGguf::EuroBert | DecoderEmbeddingGguf::LlamaEmbed => {
                    EmbeddingLoaderType::LlamaEmbedding
                }
                DecoderEmbeddingGguf::Qwen3 => EmbeddingLoaderType::Qwen3Embedding,
            };
            return self.load_prepared_embedding(
                loader_type,
                source,
                decoder.pooling(&archive)?,
                paths,
                options,
            );
        }
        let source =
            self.prepare_source(&archive, &bindings, paths, &options, config, rope_pairing)?;
        let mut loader = NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: self.config.topology.clone(),
                organization: self.config.organization,
                write_uqff: self.config.write_uqff.clone(),
                imatrix: self.config.imatrix.clone(),
                calibration_file: self.config.calibration_file.clone(),
                hf_cache_path: self.config.hf_cache_path.clone(),
                matformer_config_path: self.config.matformer_config_path.clone(),
                matformer_slice_name: self.config.matformer_slice_name.clone(),
                max_model_len: self.config.max_model_len,
                ..Default::default()
            },
            None,
            None,
            Some(self.quantized_model_id.clone()),
            self.no_kv_cache,
            self.jinja_explicit.clone(),
        )
        .with_mtp(self.config.mtp);
        if let Some(dynamic_lora) = self.dynamic_lora.as_ref() {
            loader = loader.with_lora(dynamic_lora.adapters.clone(), dynamic_lora.runtime);
        }
        let loader = loader.build_with_source(loader_type, source, self.kind.clone())?;
        loader.load_model_from_path(paths, options)
    }

    /// An encoder GGUF (BERT, Nomic BERT) as an embedding model: its synthesized config, its tensors by their
    /// native names, and the pooling its metadata names.
    fn load_native_embedding(
        &self,
        archive: Arc<inference_quant::GgufArchive>,
        encoder: EmbeddingGgufArchitecture,
        paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let config = synthesize_embedding_config(&archive, encoder)?;
        let bindings = build_embedding_bindings(&archive, encoder)?;
        let pooling = embedding_pooling(&archive, encoder)?;
        // encoders take no rope layout stamp; the pairing is unused past the source
        let mut source = self.prepare_source(
            &archive,
            &bindings,
            paths,
            &options,
            config,
            RopePairing::HalfSplit,
        )?;
        add_encoder_specials(&mut source.tokenizer, archive.metadata())?;
        let loader_type = match encoder {
            EmbeddingGgufArchitecture::Bert
            | EmbeddingGgufArchitecture::JinaBertV2
            | EmbeddingGgufArchitecture::JinaBertV3 => EmbeddingLoaderType::Bert,
            EmbeddingGgufArchitecture::NomicBert | EmbeddingGgufArchitecture::NomicBertMoe => {
                EmbeddingLoaderType::NomicBert
            }
            EmbeddingGgufArchitecture::ModernBert => EmbeddingLoaderType::ModernBert,
            EmbeddingGgufArchitecture::NeoBert => EmbeddingLoaderType::NeoBert,
        };
        self.load_prepared_embedding(loader_type, source, pooling, paths, options)
    }

    /// d1-omni: its trunk and head under their GGUF names, with the decision profile that renders its prompts.
    fn load_d1_omni(
        &self,
        archive: Arc<inference_quant::GgufArchive>,
        paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let config = d1_omni_config(&archive)?;
        let mut bindings = inference_quant::GgufBindingMap::new();
        for name in archive.tensors().keys() {
            bindings.insert(
                name.clone(),
                inference_quant::GgufTensorBinding::tensor(name.clone()),
            );
        }
        let source = self.prepare_source(
            &archive,
            &bindings,
            paths,
            &options,
            config,
            RopePairing::HalfSplit,
        )?;
        self.load_prepared_embedding(
            EmbeddingLoaderType::Lfm2Decision,
            source,
            EmbeddingGgufPooling::Rank,
            paths,
            options,
        )
    }

    /// A GGUF translated to `loader_type`'s config and tensor names, run as an embedder pooled as it names.
    fn load_prepared_embedding(
        &self,
        loader_type: EmbeddingLoaderType,
        source: PreparedSource,
        pooling: EmbeddingGgufPooling,
        paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let loader = EmbeddingLoaderBuilder::new(
            EmbeddingSpecificConfig {
                topology: self.config.topology.clone(),
                write_uqff: self.config.write_uqff.clone(),
                imatrix: self.config.imatrix.clone(),
                calibration_file: self.config.calibration_file.clone(),
                hf_cache_path: self.config.hf_cache_path.clone(),
                ..Default::default()
            },
            None,
            Some(self.quantized_model_id.clone()),
        )
        .build_with_source(loader_type, source, pooling)?;
        loader.load_model_from_path(paths, options)
    }

    #[cfg(feature = "models-gemma")]
    fn load_native_gemma3_text(
        &self,
        archive: Arc<inference_quant::GgufArchive>,
        args: NativeNormalLoadArgs<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let NativeNormalLoadArgs { paths, options } = args;
        let external_config = if paths.get_config_filename().as_os_str().is_empty() {
            None
        } else {
            Some(fs::read_to_string(paths.get_config_filename())?)
        };
        let tensor_names = archive.tensors().keys().cloned().collect::<Vec<_>>();
        let config = prepare_gemma3_text_config(
            external_config.as_deref(),
            archive.metadata(),
            &tensor_names,
        )?;
        let config = stamp_qk_rope_layout(&config, RopePairing::HalfSplit)?;
        let use_language_model_prefix = gemma3_text_uses_language_model_prefix(&config)?;
        let bindings = build_gemma3_text_bindings(&archive, use_language_model_prefix)?;
        let source = self.prepare_source(
            &archive,
            &bindings,
            paths,
            &options,
            config,
            RopePairing::HalfSplit,
        )?;
        let mut loader = MultimodalLoaderBuilder::new(
            self.config.multimodal_config(),
            None,
            None,
            Some(self.quantized_model_id.clone()),
            self.jinja_explicit.clone(),
        )
        .with_encoder_cache_memory_bytes(self.encoder_cache_memory_bytes);
        if let Some(dynamic_lora) = self.dynamic_lora.as_ref() {
            loader = loader.with_lora(dynamic_lora.adapters.clone(), dynamic_lora.runtime);
        }
        let loader =
            loader.build_with_source(MultimodalLoaderType::Gemma3, source, self.kind.clone())?;
        loader.load_model_from_path(paths, options)
    }

    fn load_native_multimodal(
        &self,
        args: NativeMultimodalLoadArgs<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let NativeMultimodalLoadArgs {
            paths,
            mmproj_paths,
            options,
        } = args;
        if !matches!(self.kind, ModelKind::GgufQuantized { .. }) && self.dynamic_lora.is_none() {
            bail!("multimodal GGUF does not support legacy GGUF adapters");
        }
        let mut archive = inference_quant::GgufArchive::open(paths.get_weight_filenames())?;
        alias_decision_architecture(&mut archive)?;
        let architecture = match archive.metadata_value("general.architecture") {
            Some(inference_tensor::quantized::gguf_file::Value::String(value)) => value.clone(),
            Some(value) => {
                bail!("GGUF `general.architecture` must be a string, got {value:?}")
            }
            None => bail!("GGUF metadata is missing `general.architecture`"),
        };
        let mmproj = inference_quant::GgufArchive::open_components(mmproj_paths)?;
        archive.merge_components(mmproj)?;
        let archive = Arc::new(archive);

        let (loader_type, bindings, rope_pairing) = match architecture.as_str() {
            "gemma4" => (
                MultimodalLoaderType::Gemma4,
                build_gemma4_bindings(&archive)?,
                RopePairing::HalfSplit,
            ),
            "qwen2vl" | "qwen3vl" | "qwen3vlmoe" | "qwen35" | "qwen35moe" | "qwen4exp" => (
                qwen_multimodal_loader_type(&archive)?,
                build_qwen_multimodal_bindings(&archive)?,
                RopePairing::HalfSplit,
            ),
            architecture => {
                let Some(resolved) = resolve_native_multimodal_gguf(&archive)? else {
                    bail!(
                        "multimodal GGUF architecture `{architecture}` is not supported by the native multimodal loader"
                    );
                };
                (
                    resolved.loader_type,
                    resolved.bindings,
                    resolved.rope_pairing,
                )
            }
        };
        if paths.get_config_filename().as_os_str().is_empty() {
            bail!(
                "multimodal GGUF architecture `{architecture}` requires its original `config.json`; pass `--tok-model-id <original-model-id>`"
            );
        }
        let config = prepare_native_multimodal_config(
            &loader_type,
            &fs::read_to_string(paths.get_config_filename())?,
        )?;
        let config = stamp_qk_rope_layout(&config, rope_pairing)?;
        #[cfg(feature = "models-gemma")]
        if architecture == "gemma3" {
            ensure_gemma3_vision_config(&config)?;
        }
        let source =
            self.prepare_source(&archive, &bindings, paths, &options, config, rope_pairing)?;
        #[cfg(feature = "models-gemma")]
        let source = {
            let mut source = source;
            if architecture == "gemma3" {
                ensure_gemma3_image_token(&mut source.tokenizer, &source.config)?;
            }
            source
        };
        let processor_config = paths
            .get_processor_config()
            .as_ref()
            .map(fs::read_to_string)
            .transpose()?;
        let preprocessor_config = paths
            .get_preprocessor_config()
            .as_ref()
            .map(fs::read_to_string)
            .transpose()?;
        let mut source_weight_files = paths.get_weight_filenames().to_vec();
        source_weight_files.extend_from_slice(mmproj_paths);
        let source = PreparedSource {
            processor_config,
            preprocessor_config,
            source_weight_files,
            ..source
        };
        let mut loader = MultimodalLoaderBuilder::new(
            self.config.multimodal_config(),
            None,
            None,
            Some(self.quantized_model_id.clone()),
            self.jinja_explicit.clone(),
        )
        .with_encoder_cache_memory_bytes(self.encoder_cache_memory_bytes)
        .with_mtp(self.config.mtp);
        if let Some(dynamic_lora) = self.dynamic_lora.as_ref() {
            loader = loader.with_lora(dynamic_lora.adapters.clone(), dynamic_lora.runtime);
        }
        let loader = loader.build_with_source(loader_type, source, self.kind.clone())?;
        loader.load_model_from_path(paths, options)
    }

    /// The `lora-*.gguf` files the GGUF repository lists beside its weights.
    fn repository_task_adapters(
        &self,
        token_source: &TokenSource,
        revision: &str,
        silent: bool,
    ) -> Result<Vec<String>> {
        let api = build_api(token_source, !silent)?.repo(Repo::with_revision(
            self.quantized_model_id.clone(),
            RepoType::Model,
            revision.to_string(),
        ));
        let files = list_repo_files(&api, Path::new(&self.quantized_model_id), true, revision)?;
        Ok(files
            .into_iter()
            .filter(|file| is_task_adapter_file(file))
            .collect())
    }

    fn infer_multimodal_asset_paths(
        &self,
        paths: &dyn ModelPaths,
        mmproj_paths: &[PathBuf],
        token_source: &TokenSource,
        silent: bool,
    ) -> Result<Option<LocalModelPaths<PathBuf>>> {
        if self.model_id.is_some() {
            return Ok(None);
        }

        if !paths.get_config_filename().as_os_str().is_empty()
            && paths.get_preprocessor_config().is_some()
            && paths.get_processor_config().is_some()
            && !paths.get_tokenizer_filename().as_os_str().is_empty()
        {
            return Ok(None);
        }

        let config_missing = paths.get_config_filename().as_os_str().is_empty();
        let mut model_archive = inference_quant::GgufArchive::open(paths.get_weight_filenames())?;
        alias_decision_architecture(&mut model_archive)?;
        let projector_archives = inference_quant::GgufArchive::open_components(mmproj_paths)?;
        let projector_labels = projector_archives
            .iter()
            .enumerate()
            .map(|(index, archive)| {
                format!(
                    "projector {} (`{}`)",
                    index + 1,
                    archive.shards()[0].path().display()
                )
            })
            .collect::<Vec<_>>();
        let inferred_model_id = infer_hf_base_model_id(
            std::iter::once(("model", model_archive.metadata())).chain(
                projector_labels
                    .iter()
                    .zip(&projector_archives)
                    .map(|(label, archive)| (label.as_str(), archive.metadata())),
            ),
        );
        let inferred_model_id = match inferred_model_id {
            Ok(inferred_model_id) => inferred_model_id,
            Err(error) if config_missing => {
                return Err(error).context(
                    "Cannot infer original model assets from GGUF base-model metadata; pass \
                     `--tok-model-id <original-model-id>` to override it",
                );
            }
            Err(error) => {
                warn!(
                    "Could not infer optional GGUF processor assets from base-model metadata: \
                     {error}"
                );
                return Ok(None);
            }
        };
        let Some(inferred_model_id) = inferred_model_id else {
            if config_missing {
                bail!(
                    "multimodal GGUF requires its original `config.json`, but the GGUF files do \
                     not identify one unambiguous Hugging Face base model; pass \
                     `--tok-model-id <original-model-id>`"
                );
            }
            return Ok(None);
        };

        let revision = "main";
        let api = match build_api(token_source, !silent) {
            Ok(api) => api,
            Err(error) if config_missing => return Err(error),
            Err(error) => {
                warn!("Could not prepare optional GGUF processor asset discovery: {error}");
                return Ok(None);
            }
        };
        let api = api.repo(Repo::with_revision(
            inferred_model_id.clone(),
            RepoType::Model,
            revision.to_string(),
        ));
        let model_id = Path::new(&inferred_model_id);
        let files = match list_repo_files(&api, model_id, true, revision) {
            Ok(files) => files,
            Err(error) if config_missing => return Err(error),
            Err(error) => {
                warn!(
                    "Could not discover optional GGUF processor assets from \
                     `{inferred_model_id}`: {error}"
                );
                return Ok(None);
            }
        };
        let get_optional = |filename: &str| -> Option<PathBuf> {
            if files.iter().any(|file| file == filename) {
                match get_file(&api, model_id, filename, revision) {
                    Ok(path) => Some(path),
                    Err(error) => {
                        warn!(
                            "Could not load optional GGUF processor asset `{filename}` from \
                             `{inferred_model_id}`: {error}"
                        );
                        None
                    }
                }
            } else {
                None
            }
        };

        let config_filename = if config_missing {
            let Some(config_name) = preferred_hf_config(&files) else {
                bail!(
                    "GGUF base model `{inferred_model_id}` has no `config.json` or `params.json`; \
                     pass `--tok-model-id <original-model-id>`"
                );
            };
            get_file(&api, model_id, config_name, revision)?
        } else {
            paths.get_config_filename().clone()
        };
        let preprocessor_config = match paths.get_preprocessor_config() {
            Some(filename) => Some(filename.clone()),
            None => get_optional("preprocessor_config.json"),
        };
        let video_preprocessor_config = paths
            .get_video_preprocessor_config()
            .cloned()
            .or_else(|| get_optional("video_preprocessor_config.json"));
        let processor_config = match paths.get_processor_config() {
            Some(filename) => Some(filename.clone()),
            None => get_optional("processor_config.json"),
        };
        let tokenizer_filename = if paths.get_tokenizer_filename().as_os_str().is_empty() {
            get_optional("tokenizer.json").unwrap_or_default()
        } else {
            paths.get_tokenizer_filename().clone()
        };
        let gen_conf = paths
            .get_gen_conf_filename()
            .cloned()
            .or_else(|| get_optional("generation_config.json"));

        info!("Using GGUF base-model assets from `{inferred_model_id}`.");
        Ok(Some(LocalModelPaths {
            tokenizer_filename,
            config_filename,
            template_filename: paths.get_template_filename().clone(),
            filenames: paths.get_weight_filenames().to_vec(),
            adapter_paths: paths.get_adapter_paths().clone(),
            gen_conf,
            preprocessor_config,
            video_preprocessor_config,
            processor_config,
            chat_template_json_filename: paths.get_chat_template_explicit().clone(),
        }))
    }
}

impl Loader for GGUFLoader {
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let silent = options.silent;
        let _progress_guard = ProgressScopeGuard::new(silent);
        super::loading::install_hf_cache(self.config.hf_cache_path.clone());
        let revision = revision.unwrap_or_else(|| "main".to_string());
        if self.mmproj_filenames.as_ref().is_some_and(|filenames| {
            filenames.is_empty() || filenames.iter().any(|filename| filename.trim().is_empty())
        }) {
            bail!("multimodal GGUF requires at least one nonempty projector filename");
        }
        let request = |quantized_filenames| super::paths::GgufPathsRequest {
            model_id: self.model_id.as_deref(),
            quantized_model_id: &self.quantized_model_id,
            quantized_filenames,
            chat_template: self.chat_template.as_deref(),
            token_source: &token_source,
            revision: Some(revision.clone()),
            silent,
        };
        let paths = super::paths::get_paths_gguf(
            request(&self.quantized_filenames),
            self.dynamic_lora_adapters(),
        )?;
        // a jina-bert-v3 GGUF's task adapters, fetched beside it for the load to find
        if is_architecture(
            &inference_quant::GgufArchive::open(paths.get_weight_filenames())?,
            JINA_BERT_V3,
        ) {
            let adapters = self.repository_task_adapters(&token_source, &revision, silent)?;
            if !adapters.is_empty() {
                super::paths::get_paths_gguf(
                    super::paths::GgufPathsRequest {
                        quantized_filenames: &adapters,
                        ..request(&self.quantized_filenames)
                    },
                    None,
                )?;
            }
        }
        if let Some(mmproj_filenames) = self.mmproj_filenames.as_ref() {
            let mmproj_paths = super::paths::get_paths_gguf(request(mmproj_filenames), None)?;
            let inferred_paths = self.infer_multimodal_asset_paths(
                &paths,
                mmproj_paths.get_weight_filenames(),
                &token_source,
                silent,
            )?;
            let paths = inferred_paths
                .as_ref()
                .map(|paths| paths as &dyn ModelPaths)
                .unwrap_or(&paths);
            return self.load_native_multimodal(NativeMultimodalLoadArgs {
                paths,
                mmproj_paths: mmproj_paths.get_weight_filenames(),
                options,
            });
        }

        self.load_model_from_path(&paths, options)
    }

    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        options: LoadOptions<'_>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(options.silent);
        self.load_native_normal(NativeNormalLoadArgs { paths, options })
    }

    fn get_id(&self) -> String {
        self.model_id
            .as_ref()
            .unwrap_or(&self.quantized_model_id)
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GGUFSpecificConfig, GgufTokenizerConversion, TokenizerFallback, preferred_hf_config,
        prepare_native_multimodal_config, requires_multimodal_projector,
        resolve_tokenizer_candidate,
    };
    use crate::{MultimodalLoaderType, gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY};
    use std::path::{Path, PathBuf};
    use tokenizers::{Tokenizer, models::bpe::BPE};

    fn tokenizer_conversion(marker: &str) -> GgufTokenizerConversion {
        GgufTokenizerConversion {
            tokenizer: Tokenizer::new(BPE::default()),
            bos: None,
            eos: Some(marker.to_string()),
            unk: None,
        }
    }

    #[test]
    fn scopes_main_only_multimodal_architectures() {
        assert!(!requires_multimodal_projector("qwen35"));
        assert!(!requires_multimodal_projector("QWEN35"));
        assert!(!requires_multimodal_projector("gemma3"));
        assert!(requires_multimodal_projector("gemma3n"));
        assert!(requires_multimodal_projector("gemma4"));
        assert!(!requires_multimodal_projector("qwen35moe"));
        assert!(!requires_multimodal_projector("mistral3"));
        assert!(requires_multimodal_projector("muse-glimmer"));
    }

    #[test]
    fn native_multimodal_configs_ignore_checkpoint_quantization_metadata() {
        let raw = r#"{
            "architectures":["ExampleForConditionalGeneration"],
            "quantization_config":{"quant_method":"awq"},
            "text_config":{
                "hidden_size":128,
                "quantization_config":{"quant_method":"fp8"},
                "rope_parameters":{"quantization_config":{"quant_method":"gptq"}}
            },
            "vision_config":{"hidden_size":64}
        }"#;
        for loader_type in [
            MultimodalLoaderType::Gemma3,
            MultimodalLoaderType::Gemma3n,
            MultimodalLoaderType::Gemma4,
            MultimodalLoaderType::Idefics3,
            MultimodalLoaderType::Llama4,
            MultimodalLoaderType::Lfm2Vl,
            MultimodalLoaderType::Mistral3,
            MultimodalLoaderType::MuseGlimmer,
            MultimodalLoaderType::Qwen2VL,
            MultimodalLoaderType::Qwen2_5VL,
            MultimodalLoaderType::Qwen3VL,
            MultimodalLoaderType::Qwen3VLMoE,
            MultimodalLoaderType::Qwen3_5,
            MultimodalLoaderType::Qwen3_5Moe,
            MultimodalLoaderType::Qwen4Exp,
        ] {
            let config = prepare_native_multimodal_config(&loader_type, raw).unwrap();
            let config: serde_json::Value = serde_json::from_str(&config).unwrap();
            assert!(config["quantization_config"].is_null(), "{loader_type:?}");
            assert!(
                config["text_config"]["quantization_config"].is_null(),
                "{loader_type:?}"
            );
            assert!(
                config["text_config"]["rope_parameters"]
                    .get("quantization_config")
                    .is_none(),
                "{loader_type:?}"
            );
            assert_eq!(
                config["architectures"][0],
                "ExampleForConditionalGeneration"
            );
            assert_eq!(config["vision_config"]["hidden_size"], 64);
            if matches!(
                loader_type,
                MultimodalLoaderType::Qwen3_5
                    | MultimodalLoaderType::Qwen3_5Moe
                    | MultimodalLoaderType::Qwen4Exp
            ) {
                assert_eq!(config["text_config"][GDN_V_HEAD_LAYOUT_CONFIG_KEY], "tiled");
            }
            if matches!(loader_type, MultimodalLoaderType::MuseGlimmer) {
                assert_eq!(
                    config["_inference_muse_glimmer_gguf_collapsed_temporal"],
                    true
                );
            }
        }
    }

    #[test]
    fn automatic_tokenizer_mismatch_falls_back_to_embedded_metadata() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("automatic-tokenizer.json"),
            TokenizerFallback::Automatic,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("embedded")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("embedded"));
        assert!(!resolved.generation_config_compatible);
    }

    #[test]
    fn model_asset_tokenizer_mismatch_falls_back_to_embedded_metadata() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("model-assets/tokenizer.json"),
            TokenizerFallback::ModelAssets,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("embedded")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("embedded"));
        assert!(!resolved.generation_config_compatible);
    }

    #[test]
    fn compatible_model_asset_tokenizer_is_preserved() {
        let resolved = resolve_tokenizer_candidate(
            Path::new("model-assets/tokenizer.json"),
            TokenizerFallback::ModelAssets,
            Ok(tokenizer_conversion("external")),
            || Ok(tokenizer_conversion("unused")),
        )
        .unwrap();

        assert_eq!(resolved.conversion.eos.as_deref(), Some("external"));
        assert!(resolved.generation_config_compatible);
    }

    #[test]
    fn explicit_tokenizer_file_mismatch_is_an_error() {
        let error = resolve_tokenizer_candidate(
            Path::new("explicit-tokenizer.json"),
            TokenizerFallback::Strict,
            Err(anyhow::anyhow!("vocabulary mismatch")),
            || Ok(tokenizer_conversion("unused")),
        )
        .err()
        .expect("explicit mismatch must fail");

        assert!(error.to_string().contains("incompatible"));
    }

    #[test]
    fn prefers_hugging_face_config_over_native_params() {
        let files = vec!["params.json".to_string(), "config.json".to_string()];
        assert_eq!(preferred_hf_config(&files), Some("config.json"));
        assert_eq!(
            preferred_hf_config(&["params.json".to_string()]),
            Some("params.json")
        );
        assert_eq!(preferred_hf_config(&[]), None);
    }

    #[test]
    fn prepared_multimodal_config_preserves_runtime_context_cap() {
        let config = GGUFSpecificConfig {
            organization: crate::pipeline::IsqOrganization::MoeExpertsOnly,
            imatrix: Some(PathBuf::from("model.imatrix")),
            max_edge: Some(1024),
            max_model_len: Some(8192),
            hf_cache_path: Some(PathBuf::from("hf-cache")),
            matformer_config_path: Some(PathBuf::from("matformer.csv")),
            matformer_slice_name: Some("small".to_string()),
            ..Default::default()
        }
        .multimodal_config();

        assert_eq!(config.max_edge, Some(1024));
        assert_eq!(config.max_model_len, Some(8192));
        assert!(matches!(
            config.organization,
            crate::pipeline::IsqOrganization::MoeExpertsOnly
        ));
        assert_eq!(config.imatrix, Some(PathBuf::from("model.imatrix")));
        assert_eq!(config.hf_cache_path, Some(PathBuf::from("hf-cache")));
        assert_eq!(
            config.matformer_config_path,
            Some(PathBuf::from("matformer.csv"))
        );
        assert_eq!(config.matformer_slice_name.as_deref(), Some("small"));
    }
}
