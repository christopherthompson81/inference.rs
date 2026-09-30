pub use crate::model::{NormalLoadingMetadata, NormalModel};
use std::{borrow::Cow, collections::HashMap, fmt::Debug};

use crate::matformer::MatformerSliceConfig;

use crate::{
    lora::{LoraConfig, Ordering},
    paged_attention::{AttentionImplementation, ModelConfigLike},
    pipeline::isq::IsqModelLoader,
};
use anyhow::Result;
use candle_core::DType;
use inference_quant::log::once_log_debug;

use inference_quant::ShardedVarBuilder;

use regex::Regex;
use serde::Deserialize;

use crate::xlora_models::XLoraConfig;

use super::{AutoDeviceMapParams, AutoDeviceMapQuantization, DeviceMappedModelLoader};

pub use inference_nn::loaders::NormalModelLoader;

pub use inference_nn::loaders::NormalLoaderType;

// Expands `loader()` from the architecture table in `inference_nn::loaders`.
macro_rules! normal_loader_dispatch {
    ($($variant:ident {
        cli: $cli:tt,
        hf: $hf:literal,
        model_type: $model_type:literal,
        loader: $loader:ident
        $(, feature: $feature:literal)? $(,)?
    }),* $(,)?) => {
        pub(crate) trait NormalLoaderTypeExt {
            fn loader(&self) -> Result<Box<dyn NormalModelLoader>>;
        }

        impl NormalLoaderTypeExt for NormalLoaderType {
            fn loader(&self) -> Result<Box<dyn NormalModelLoader>> {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        Self::$variant => Ok($loader::boxed()),
                        $(
                            #[cfg(not(feature = $feature))]
                            Self::$variant => anyhow::bail!(
                                "architecture `{}` is not built in; enable the `{}` feature",
                                $cli,
                                $feature
                            ),
                        )?
                    )*
                }
            }
        }
    };
}

inference_nn::normal_loader_table!(normal_loader_dispatch);

mod auto;
pub use auto::*;
#[cfg(feature = "models-gemma")]
pub use inference_models_gemma::loaders::{Gemma2Loader, GemmaLoader};
#[cfg(feature = "models-llama")]
pub use inference_models_llama::loaders::{
    LlamaLoader, MistralLoader, MixtralLoader, SmolLm3Loader,
};
#[cfg(feature = "models-other")]
pub use inference_models_other::loaders::{
    DeepSeekV2Loader, DeepSeekV3Loader, GLM4Loader, GLM4MoeLiteLoader, GLM4MoeLoader, GptOssLoader,
    GraniteMoeHybridLoader, HunYuanDenseV1Loader, HunYuanMoEV1Loader, Lfm2Loader, Starcoder2Loader,
};
#[cfg(feature = "models-phi")]
pub use inference_models_phi::loaders::{Phi2Loader, Phi3_5MoELoader, Phi3Loader};
#[cfg(feature = "models-qwen")]
pub use inference_models_qwen::loaders::{
    Qwen2Loader, Qwen3_5TextLoader, Qwen3Loader, Qwen3MoELoader, Qwen3NextLoader,
};

#[cfg(all(
    test,
    feature = "models-gemma",
    feature = "models-llama",
    feature = "models-other",
    feature = "models-phi",
    feature = "models-qwen"
))]
mod tests;
