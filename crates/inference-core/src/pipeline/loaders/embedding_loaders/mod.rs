pub use crate::model::EmbeddingModel;
use std::{
    fmt::{self, Debug, Display},
    path::PathBuf,
    str::FromStr,
};

use crate::{matformer::MatformerSliceConfig, pipeline::NormalLoadingMetadata};

use crate::{
    paged_attention::{AttentionImplementation, ModelConfigLike},
    pipeline::isq::IsqModelLoader,
};
use anyhow::Result;
use inference_quant::log::once_log_debug;
use inference_tensor::DType;

use inference_quant::ShardedVarBuilder;

use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize, de::Visitor};

use super::{AutoDeviceMapParams, DeviceMappedModelLoader};

pub use inference_nn::loaders::EmbeddingModelLoader;

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
/// The architecture to load the embedding model as.
pub enum EmbeddingLoaderType {
    #[serde(rename = "embeddinggemma")]
    EmbeddingGemma,
    #[serde(rename = "qwen3embedding")]
    Qwen3Embedding,
    #[serde(rename = "bert")]
    Bert,
    #[serde(rename = "nomicbert")]
    NomicBert,
    #[serde(rename = "modernbert")]
    ModernBert,
    #[serde(rename = "neobert")]
    NeoBert,
    #[serde(rename = "llamaembedding")]
    LlamaEmbedding,
}

const NOMIC_BERT_CLASSES: &[&str] = &["NomicBertModel"];
const MODERN_BERT_CLASSES: &[&str] = &["ModernBertModel", "ModernBertForMaskedLM"];
const NEO_BERT_CLASSES: &[&str] = &["NeoBERT", "NeoBERTLMHead"];
// Llama layers run without a causal mask: EuroBERT, and Llama bidirectional embedders
const LLAMA_EMBEDDING_CLASSES: &[&str] = &[
    "EuroBertModel",
    "EuroBertForMaskedLM",
    "LlamaBidirectionalModel",
];

// Encoder-only classes (BERT, RoBERTa, XLM-R, CamemBERT): checkpoints that can only be embedders
const BERT_CLASSES: &[&str] = &[
    "BertModel",
    "BertForMaskedLM",
    "RobertaModel",
    "RobertaForMaskedLM",
    "XLMRobertaModel",
    "XLMRobertaForMaskedLM",
    "CamembertModel",
    "CamembertForMaskedLM",
    "BertForSequenceClassification",
    "RobertaForSequenceClassification",
    "XLMRobertaForSequenceClassification",
    "CamembertForSequenceClassification",
    "JinaBertModel",
    "JinaBertForMaskedLM",
    "JinaBertForSequenceClassification",
];

const JINA_V5_CLASS: &str = "JinaEmbeddingsV5Model";

// https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
impl EmbeddingLoaderType {
    pub fn from_causal_lm_name(name: &str) -> Result<Self> {
        match name {
            "Gemma3TextModel" => Ok(Self::EmbeddingGemma),
            "Qwen3ForCausalLM" | "Qwen3Model" => Ok(Self::Qwen3Embedding),
            other => Self::from_encoder_name(other).ok_or_else(|| {
                anyhow::anyhow!(
                    "Unsupported Hugging Face Transformers model class `{other}`. Please raise an issue."
                )
            }),
        }
    }

    /// As [`Self::from_causal_lm_name`], with the classes whose base the config tells apart.
    pub fn from_config(name: &str, config: &str) -> Result<Self> {
        if name != JINA_V5_CLASS {
            return Self::from_causal_lm_name(name);
        }
        // Jina v5 wraps Qwen3 (small) or EuroBERT (nano), which attends both ways
        let config: serde_json::Value = serde_json::from_str(config)?;
        Ok(match config["is_decoder"].as_bool() {
            Some(false) => Self::LlamaEmbedding,
            _ => Self::Qwen3Embedding,
        })
    }

    /// The architecture of a class that only ever runs as an embedder, so needs no sentence-transformers marker.
    pub fn from_encoder_name(name: &str) -> Option<Self> {
        [
            (BERT_CLASSES, Self::Bert),
            (NOMIC_BERT_CLASSES, Self::NomicBert),
            (MODERN_BERT_CLASSES, Self::ModernBert),
            (NEO_BERT_CLASSES, Self::NeoBert),
            (LLAMA_EMBEDDING_CLASSES, Self::LlamaEmbedding),
        ]
        .into_iter()
        .find_map(|(classes, tp)| classes.contains(&name).then_some(tp))
    }

    /// The HF `config.json` `architectures` string, the inverse of `from_causal_lm_name`.
    pub fn config_arch(&self) -> &'static str {
        match self {
            Self::EmbeddingGemma => "Gemma3TextModel",
            Self::Qwen3Embedding => "Qwen3ForCausalLM",
            Self::Bert => "BertModel",
            Self::NomicBert => "NomicBertModel",
            Self::ModernBert => "ModernBertModel",
            Self::NeoBert => "NeoBERT",
            Self::LlamaEmbedding => "LlamaBidirectionalModel",
        }
    }
}

impl EmbeddingLoaderType {
    pub(crate) fn loader(&self) -> Result<Box<dyn EmbeddingModelLoader>> {
        match self {
            #[cfg(feature = "models-gemma")]
            Self::EmbeddingGemma => Ok(EmbeddingGemmaLoader::boxed()),
            #[cfg(not(feature = "models-gemma"))]
            Self::EmbeddingGemma => {
                anyhow::bail!(
                    "architecture `{self}` is not built in; enable the `models-gemma` feature"
                )
            }
            #[cfg(feature = "models-qwen")]
            Self::Qwen3Embedding => Ok(Qwen3EmbeddingLoader::boxed()),
            #[cfg(not(feature = "models-qwen"))]
            Self::Qwen3Embedding => {
                anyhow::bail!(
                    "architecture `{self}` is not built in; enable the `models-qwen` feature"
                )
            }
            #[cfg(feature = "models-llama")]
            Self::LlamaEmbedding => Ok(LlamaEmbeddingLoader::boxed()),
            #[cfg(not(feature = "models-llama"))]
            Self::LlamaEmbedding => {
                anyhow::bail!(
                    "architecture `{self}` is not built in; enable the `models-llama` feature"
                )
            }
            #[cfg(feature = "models-other")]
            Self::Bert => Ok(BertLoader::boxed()),
            #[cfg(feature = "models-other")]
            Self::NomicBert => Ok(NomicBertLoader::boxed()),
            #[cfg(feature = "models-other")]
            Self::ModernBert => Ok(ModernBertLoader::boxed()),
            #[cfg(feature = "models-other")]
            Self::NeoBert => Ok(NeoBertLoader::boxed()),
            #[cfg(not(feature = "models-other"))]
            Self::Bert | Self::NomicBert | Self::ModernBert | Self::NeoBert => {
                anyhow::bail!(
                    "architecture `{self}` is not built in; enable the `models-other` feature"
                )
            }
        }
    }
}

impl FromStr for EmbeddingLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "embeddinggemma" => Ok(Self::EmbeddingGemma),
            "qwen3embedding" => Ok(Self::Qwen3Embedding),
            "bert" => Ok(Self::Bert),
            "nomicbert" => Ok(Self::NomicBert),
            "modernbert" => Ok(Self::ModernBert),
            "neobert" => Ok(Self::NeoBert),
            "llamaembedding" => Ok(Self::LlamaEmbedding),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `embeddinggemma`, `qwen3embedding`, `bert`, `nomicbert`, `modernbert`, `neobert`, `llamaembedding`."
            )),
        }
    }
}

impl Display for EmbeddingLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmbeddingGemma => write!(f, "embeddinggemma"),
            Self::Qwen3Embedding => write!(f, "qwen3embedding"),
            Self::Bert => write!(f, "bert"),
            Self::NomicBert => write!(f, "nomicbert"),
            Self::ModernBert => write!(f, "modernbert"),
            Self::NeoBert => write!(f, "neobert"),
            Self::LlamaEmbedding => write!(f, "llamaembedding"),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum EmbeddingModulePaths {
    Transformer {
        path: String,
    },
    Pooling {
        path: String,
        config: PathBuf,
    },
    Dense {
        path: String,
        config: PathBuf,
        model: PathBuf,
    },
    Normalize {
        path: String,
    },
}

impl EmbeddingModulePaths {
    pub fn serialize_modules(modules: &[EmbeddingModulePaths]) -> String {
        #[derive(Serialize)]
        struct OutputModule {
            idx: usize,
            name: String,
            path: String,
            #[serde(rename = "type")]
            ty: String,
        }

        let mapped: Vec<OutputModule> = modules
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let (path, ty) = match m {
                    EmbeddingModulePaths::Transformer { path } => (
                        path.clone(),
                        "sentence_transformers.models.Transformer".to_string(),
                    ),
                    EmbeddingModulePaths::Pooling { path, .. } => (
                        path.clone(),
                        "sentence_transformers.models.Pooling".to_string(),
                    ),
                    EmbeddingModulePaths::Dense { path, .. } => (
                        path.clone(),
                        "sentence_transformers.models.Dense".to_string(),
                    ),
                    EmbeddingModulePaths::Normalize { path } => (
                        path.clone(),
                        "sentence_transformers.models.Normalize".to_string(),
                    ),
                };

                OutputModule {
                    idx: i,
                    name: i.to_string(),
                    path,
                    ty,
                }
            })
            .collect();

        serde_json::to_string_pretty(&mapped).unwrap()
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingModule {
    pub path: String,
    #[serde(rename = "type", deserialize_with = "deserialize_module_type")]
    pub ty: EmbeddingModuleType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModuleType {
    Transformer,
    Pooling,
    Dense,
    Normalize,
}

fn deserialize_module_type<'de, D>(deserializer: D) -> Result<EmbeddingModuleType, D::Error>
where
    D: Deserializer<'de>,
{
    struct ModuleTypeVisitor;

    impl<'de> Visitor<'de> for ModuleTypeVisitor {
        type Value = EmbeddingModuleType;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a sentence-transformers module type string")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            // Accept fully-qualified ("sentence_transformers.models.X") or just "X".
            let last = v.rsplit('.').next().unwrap_or(v).to_ascii_lowercase();
            match last.as_str() {
                "transformer" => Ok(EmbeddingModuleType::Transformer),
                "pooling" => Ok(EmbeddingModuleType::Pooling),
                "dense" => Ok(EmbeddingModuleType::Dense),
                "normalize" => Ok(EmbeddingModuleType::Normalize),
                _ => Err(E::invalid_value(
                    serde::de::Unexpected::Str(v),
                    &"Transformer/Pooling/Dense/Normalize",
                )),
            }
        }
    }

    deserializer.deserialize_str(ModuleTypeVisitor)
}

#[cfg(feature = "models-gemma")]
pub use inference_models_gemma::loaders::EmbeddingGemmaLoader;
#[cfg(feature = "models-llama")]
pub use inference_models_llama::loaders::LlamaEmbeddingLoader;
#[cfg(feature = "models-other")]
pub use inference_models_other::loaders::{
    BertLoader, ModernBertLoader, NeoBertLoader, NomicBertLoader,
};
#[cfg(feature = "models-qwen")]
pub use inference_models_qwen::loaders::Qwen3EmbeddingLoader;

/// Load a model based on the Hugging Face Transformers -CausalLM model class
pub struct AutoEmbeddingLoader;

#[derive(Deserialize)]
struct AutoEmbeddingLoaderConfig {
    architectures: Vec<String>,
}

impl AutoEmbeddingLoader {
    fn get_loader(config: &str) -> Result<Box<dyn EmbeddingModelLoader>> {
        let auto_cfg: AutoEmbeddingLoaderConfig = serde_json::from_str(config)?;
        if auto_cfg.architectures.len() != 1 {
            anyhow::bail!("Expected to have one name for `architectures` config field.")
        }

        let tp = EmbeddingLoaderType::from_config(&auto_cfg.architectures[0], config)?;

        once_log_debug(format!("Automatic loader type determined to be `{tp}`"));

        tp.loader()
    }
}

impl EmbeddingModelLoader for AutoEmbeddingLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        Self::get_loader(config)?.load(config, vb, normal_loading_metadata, attention_mechanism)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Self::get_loader(config)?.get_config_repr(config)
    }
    fn has_causal_attention(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.has_causal_attention(config)
    }
    fn is_classifier(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.is_classifier(config)
    }
    fn embedding_tasks(
        &self,
        config: &str,
    ) -> Result<Option<inference_nn::embedding::EmbeddingTasks>> {
        Self::get_loader(config)?.embedding_tasks(config)
    }
    fn pools_last_token(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.pools_last_token(config)
    }
    fn is_gptx(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.is_gptx(config)
    }
}

impl IsqModelLoader for AutoEmbeddingLoader {
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.promoted_isq_predicates(config)
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates_moqe(config)
    }
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for AutoEmbeddingLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        Self::get_loader(config)?.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            quantization,
            _matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Self::get_loader(config)?.num_layers(config)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Self::get_loader(config)?.layer_sizes_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            _matformer_config,
        )
    }
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &super::AutoDeviceMapParams,
    ) -> Result<usize> {
        Self::get_loader(config)?.mapped_max_act_size_elems(config, params)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Self::get_loader(config)?.model_config(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_promotes_only_embedding_weight(predicates: &[Regex]) {
        assert_eq!(predicates.len(), 1);
        assert!(
            predicates
                .iter()
                .any(|predicate| predicate.is_match("embed_tokens.weight"))
        );

        for name in [
            "lm_head.weight",
            "lm_head.bias",
            "model.embed_tokens.weight",
            "embed_tokens.bias",
            "embed_tokens.weight.extra",
            "other_embed_tokens.weight",
        ] {
            assert!(
                predicates.iter().all(|predicate| !predicate.is_match(name)),
                "unexpected promoted match for {name}"
            );
        }
    }

    #[cfg(feature = "models-gemma")]
    #[test]
    fn embedding_gemma_promotes_only_exact_embedding_weight() {
        let predicates = EmbeddingGemmaLoader.promoted_isq_predicates("{}").unwrap();
        assert_promotes_only_embedding_weight(&predicates);
    }

    #[cfg(feature = "models-qwen")]
    #[test]
    fn qwen3_embedding_promotes_only_exact_embedding_weight() {
        let predicates = Qwen3EmbeddingLoader.promoted_isq_predicates("{}").unwrap();
        assert_promotes_only_embedding_weight(&predicates);
    }

    #[cfg(all(feature = "models-gemma", feature = "models-qwen"))]
    #[test]
    fn auto_embedding_delegates_promoted_predicates() {
        for (config, expected) in [
            (
                r#"{"architectures":["Gemma3TextModel"]}"#,
                EmbeddingGemmaLoader.promoted_isq_predicates("{}").unwrap(),
            ),
            (
                r#"{"architectures":["Qwen3ForCausalLM"]}"#,
                Qwen3EmbeddingLoader.promoted_isq_predicates("{}").unwrap(),
            ),
        ] {
            let actual = AutoEmbeddingLoader.promoted_isq_predicates(config).unwrap();
            assert_eq!(
                actual.iter().map(Regex::as_str).collect::<Vec<_>>(),
                expected.iter().map(Regex::as_str).collect::<Vec<_>>()
            );
            assert_promotes_only_embedding_weight(&actual);
        }
    }
}
