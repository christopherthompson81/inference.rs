//! The loader-type enums. One table row per architecture names it for the CLI, config detection and dispatch.

use std::{fmt::Display, str::FromStr};

use anyhow::Result;
use serde::Deserialize;

macro_rules! normal_loader_types {
    ($($variant:ident {
        cli: $cli:tt,
        hf: $hf:literal,
        model_type: $model_type:literal,
        loader: $loader:ident
        $(, feature: $feature:literal)? $(,)?
    }),* $(,)?) => {
        #[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
        #[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
        /// The architecture to load the normal model as.
        pub enum NormalLoaderType {
            $(#[serde(rename = $cli)] $variant,)*
        }

        // https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
        impl NormalLoaderType {
            const CLI_NAMES: &'static [&'static str] = &[$($cli),*];

            pub fn causal_lm_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $hf,)*
                }
            }

            pub fn model_type_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $model_type,)*
                }
            }

            pub fn from_causal_lm_name(name: &str) -> Result<Self> {
                match name {
                    $($hf => Ok(Self::$variant),)*
                    other => anyhow::bail!(
                        "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
                    ),
                }
            }
        }

        impl FromStr for NormalLoaderType {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($cli => Ok(Self::$variant),)*
                    a => Err(format!(
                        "Unknown architecture `{a}`. Possible architectures: {}.",
                        Self::CLI_NAMES.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
                    )),
                }
            }
        }

        impl Display for NormalLoaderType {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$variant => f.write_str($cli),)*
                }
            }
        }
    };
}

macro_rules! multimodal_loader_types {
    ($($variant:ident {
        cli: $cli:tt $(| $cli_alias:tt)*,
        hf: $hf:literal $(| $hf_alias:literal)*,
        loader: $loader:ident
        $(, feature: $feature:literal)? $(,)?
    }),* $(,)?) => {
        #[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
        #[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
        /// The architecture to load the multimodal model as.
        pub enum MultimodalLoaderType {
            $(#[serde(rename = $cli)] $variant,)*
        }

        // https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
        impl MultimodalLoaderType {
            const CLI_NAMES: &'static [&'static str] = &[$($cli),*];

            pub fn causal_lm_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $hf,)*
                }
            }

            pub fn from_causal_lm_name(name: &str) -> Result<Self> {
                match name {
                    $($hf $(| $hf_alias)* => Ok(Self::$variant),)*
                    other => anyhow::bail!(
                        "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
                    ),
                }
            }
        }

        impl FromStr for MultimodalLoaderType {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($cli $(| $cli_alias)* => Ok(Self::$variant),)*
                    a => Err(format!(
                        "Unknown architecture `{a}`. Possible architectures: {}.",
                        Self::CLI_NAMES.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
                    )),
                }
            }
        }

        impl std::fmt::Display for MultimodalLoaderType {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$variant => f.write_str($cli),)*
                }
            }
        }
    };
}

/// Hands the text architecture table to `$callback`; core expands its loader dispatch from the same rows.
#[macro_export]
macro_rules! normal_loader_table {
    ($callback:ident) => {
        $callback! {
            Mistral { cli: "mistral", hf: "MistralForCausalLM", model_type: "mistral", loader: MistralLoader, feature: "models-llama" },
            Gemma { cli: "gemma", hf: "GemmaForCausalLM", model_type: "gemma", loader: GemmaLoader, feature: "models-gemma" },
            Mixtral { cli: "mixtral", hf: "MixtralForCausalLM", model_type: "mixtral", loader: MixtralLoader, feature: "models-llama" },
            Llama { cli: "llama", hf: "LlamaForCausalLM", model_type: "llama", loader: LlamaLoader, feature: "models-llama" },
            Phi2 { cli: "phi2", hf: "PhiForCausalLM", model_type: "phi", loader: Phi2Loader, feature: "models-phi" },
            Phi3 { cli: "phi3", hf: "Phi3ForCausalLM", model_type: "phi3", loader: Phi3Loader, feature: "models-phi" },
            Qwen2 { cli: "qwen2", hf: "Qwen2ForCausalLM", model_type: "qwen2", loader: Qwen2Loader, feature: "models-qwen" },
            Gemma2 { cli: "gemma2", hf: "Gemma2ForCausalLM", model_type: "gemma2", loader: Gemma2Loader, feature: "models-gemma" },
            Starcoder2 { cli: "starcoder2", hf: "Starcoder2ForCausalLM", model_type: "starcoder2", loader: Starcoder2Loader, feature: "models-other" },
            Phi3_5MoE { cli: "phi3.5moe", hf: "PhiMoEForCausalLM", model_type: "phimoe", loader: Phi3_5MoELoader, feature: "models-phi" },
            DeepSeekV2 {
                cli: "deepseekv2",
                hf: "DeepseekV2ForCausalLM",
                model_type: "deepseek_v2",
                loader: DeepSeekV2Loader, feature: "models-other",
            },
            DeepSeekV3 {
                cli: "deepseekv3",
                hf: "DeepseekV3ForCausalLM",
                model_type: "deepseek_v3",
                loader: DeepSeekV3Loader, feature: "models-other",
            },
            Qwen3 { cli: "qwen3", hf: "Qwen3ForCausalLM", model_type: "qwen3", loader: Qwen3Loader, feature: "models-qwen" },
            GLM4 { cli: "glm4", hf: "Glm4ForCausalLM", model_type: "glm4", loader: GLM4Loader, feature: "models-other" },
            GLM4MoeLite {
                cli: "glm4moelite",
                hf: "Glm4MoeLiteForCausalLM",
                model_type: "glm4_moe_lite",
                loader: GLM4MoeLiteLoader, feature: "models-other",
            },
            GLM4Moe { cli: "glm4moe", hf: "Glm4MoeForCausalLM", model_type: "glm4_moe", loader: GLM4MoeLoader, feature: "models-other" },
            Qwen3Moe { cli: "qwen3moe", hf: "Qwen3MoeForCausalLM", model_type: "qwen3_moe", loader: Qwen3MoELoader, feature: "models-qwen" },
            SmolLm3 { cli: "smollm3", hf: "SmolLM3ForCausalLM", model_type: "smollm3", loader: SmolLm3Loader, feature: "models-llama" },
            GraniteMoeHybrid {
                cli: "granitemoehybrid",
                hf: "GraniteMoeHybridForCausalLM",
                model_type: "granitemoehybrid",
                loader: GraniteMoeHybridLoader, feature: "models-other",
            },
            GptOss { cli: "gpt_oss", hf: "GptOssForCausalLM", model_type: "gpt_oss", loader: GptOssLoader, feature: "models-other" },
            HunYuanDenseV1 {
                cli: "hunyuanv1dense",
                hf: "HunYuanDenseV1ForCausalLM",
                model_type: "hunyuan_v1_dense",
                loader: HunYuanDenseV1Loader, feature: "models-other",
            },
            HunYuanMoEV1 {
                cli: "hunyuanv1moe",
                hf: "HunYuanMoEV1ForCausalLM",
                model_type: "hunyuan_v1_moe",
                loader: HunYuanMoEV1Loader, feature: "models-other",
            },
            Qwen3Next { cli: "qwen3next", hf: "Qwen3NextForCausalLM", model_type: "qwen3_next", loader: Qwen3NextLoader, feature: "models-qwen" },
            Qwen3_5 { cli: "qwen3_5", hf: "Qwen3_5ForCausalLM", model_type: "qwen3_5_text", loader: Qwen3_5TextLoader, feature: "models-qwen" },
            Qwen3_5Moe { cli: "qwen3_5moe", hf: "Qwen3_5MoeForCausalLM", model_type: "qwen3_5_moe_text", loader: Qwen3_5MoeTextLoader, feature: "models-qwen" },
            Lfm2 { cli: "lfm2", hf: "Lfm2ForCausalLM", model_type: "lfm2", loader: Lfm2Loader, feature: "models-other" },
            Lfm2Moe { cli: "lfm2_moe", hf: "Lfm2MoeForCausalLM", model_type: "lfm2_moe", loader: Lfm2Loader, feature: "models-other" },
        }
    };
}

/// Hands the multimodal architecture table to `$callback`; the first `cli` name is canonical, the rest aliases.
#[macro_export]
macro_rules! multimodal_loader_table {
    ($callback:ident) => {
        $callback! {
            Phi3V { cli: "phi3v", hf: "Phi3VForCausalLM", loader: Phi3VLoader, feature: "models-phi" },
            Idefics2 { cli: "idefics2", hf: "Idefics2ForConditionalGeneration", loader: Idefics2Loader, feature: "models-llama" },
            LLaVANext { cli: "llava_next", hf: "LlavaNextForConditionalGeneration", loader: LLaVANextLoader, feature: "models-llama" },
            LLaVA { cli: "llava", hf: "LlavaForConditionalGeneration", loader: LLaVALoader, feature: "models-llama" },
            Lfm2Vl { cli: "lfm2vl" | "lfm2_vl", hf: "Lfm2VlForConditionalGeneration", loader: Lfm2VlLoader, feature: "models-other" },
            VLlama { cli: "vllama", hf: "MllamaForConditionalGeneration", loader: VLlamaLoader, feature: "models-llama" },
            Qwen2VL { cli: "qwen2vl", hf: "Qwen2VLForConditionalGeneration", loader: Qwen2VLLoader, feature: "models-qwen" },
            Idefics3 { cli: "idefics3", hf: "Idefics3ForConditionalGeneration", loader: Idefics3Loader, feature: "models-llama" },
            MiniCpmO { cli: "minicpmo", hf: "MiniCPMO", loader: MiniCpmOLoader, feature: "models-qwen" },
            Phi4MM { cli: "phi4mm", hf: "Phi4MMForCausalLM", loader: Phi4MMLoader, feature: "models-phi" },
            Qwen2_5VL { cli: "qwen2_5vl", hf: "Qwen2_5_VLForConditionalGeneration", loader: Qwen2_5VLLoader, feature: "models-qwen" },
            Gemma3 { cli: "gemma3", hf: "Gemma3ForConditionalGeneration" | "Gemma3ForCausalLM", loader: Gemma3Loader, feature: "models-gemma" },
            Mistral3 { cli: "mistral3", hf: "Mistral3ForConditionalGeneration", loader: Mistral3Loader, feature: "models-llama" },
            Llama4 { cli: "llama4", hf: "Llama4ForConditionalGeneration", loader: VLlama4Loader, feature: "models-llama" },
            Gemma3n { cli: "gemma3n", hf: "Gemma3nForConditionalGeneration", loader: Gemma3nLoader, feature: "models-gemma" },
            Qwen3VL { cli: "qwen3vl", hf: "Qwen3VLForConditionalGeneration", loader: Qwen3VLLoader, feature: "models-qwen" },
            Qwen3VLMoE { cli: "qwen3vlmoe", hf: "Qwen3VLMoeForConditionalGeneration", loader: Qwen3VLMoELoader, feature: "models-qwen" },
            Qwen3_5 { cli: "qwen3_5", hf: "Qwen3_5ForConditionalGeneration", loader: Qwen3_5Loader, feature: "models-qwen" },
            Qwen3_5Moe { cli: "qwen3_5moe", hf: "Qwen3_5MoeForConditionalGeneration", loader: Qwen3_5MoeLoader, feature: "models-qwen" },
            Voxtral { cli: "voxtral", hf: "VoxtralRealtimeForConditionalGeneration", loader: VoxtralLoader, feature: "models-llama" },
            Gemma4 {
                cli: "gemma4",
                hf: "Gemma4ForConditionalGeneration"
                    | "Gemma4ForCausalLM"
                    | "Gemma4UnifiedForConditionalGeneration"
                    | "Gemma4UnifiedForCausalLM",
                loader: Gemma4Loader, feature: "models-gemma",
            },
            MuseGlimmer {
                cli: "muse_glimmer" | "museglimmer",
                hf: "MuseGlimmerForConditionalGeneration",
                loader: MuseGlimmerLoader,
                feature: "models-qwen",
            },
            DiffusionGemma { cli: "diffusiongemma", hf: "DiffusionGemmaForBlockDiffusion", loader: DiffusionGemmaLoader, feature: "models-gemma" },
            PaddleOcrVl { cli: "paddleocr_vl", hf: "PaddleOCRVLForConditionalGeneration", loader: PaddleOcrVlLoader, feature: "models-other" },
        }
    };
}

normal_loader_table!(normal_loader_types);
multimodal_loader_table!(multimodal_loader_types);
