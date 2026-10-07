pub use crate::model::MultimodalModel;
use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use inference_quant::ShardedVarBuilder;
use inference_quant::log::once_log_debug;
use inference_tensor::DType;

use regex::Regex;
use serde::Deserialize;

use super::{
    AutoDeviceMapQuantization, DeviceMappedModelLoader, NonMappedSubModel, NormalLoadingMetadata,
};

use crate::device_map::DeviceMapper;
use crate::matformer::MatformerSliceConfig;
use crate::paged_attention::{AttentionImplementation, ModelConfigLike};
use crate::pipeline::isq::IsqModelLoader;
use crate::pipeline::loaders::AutoDeviceMapParams;
use crate::pipeline::{Modalities, MultimodalPromptPrefixer, Processor};
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;

pub use inference_nn::loaders::MultimodalModelLoader;

/// The chat-template half of a multimodal loader, which stays with the engine's `Processor`.
pub(crate) trait MultimodalProcessorFactory {
    fn get_processor(
        &self,
        model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync>;
}

pub use inference_nn::loaders::MultimodalLoaderType;

// Expands `loader()` and `get_processor()` from the architecture table in `inference_nn::loaders`.
macro_rules! multimodal_loader_dispatch {
    ($($variant:ident {
        cli: $cli:tt $(| $cli_alias:tt)*,
        hf: $hf:literal $(| $hf_alias:literal)*,
        loader: $loader:ident
        $(, feature: $feature:literal)? $(,)?
    }),* $(,)?) => {
        pub(crate) trait MultimodalLoaderTypeExt {
            fn loader(&self) -> Result<Box<dyn MultimodalModelLoader>>;

            fn get_processor(
                &self,
                model_config: &str,
                processor_config: Option<ProcessorConfig>,
                preprocessor_config: PreProcessorConfig,
                max_edge: Option<u32>,
            ) -> Result<Arc<dyn Processor + Send + Sync>>;
        }

        impl MultimodalLoaderTypeExt for MultimodalLoaderType {
            fn loader(&self) -> Result<Box<dyn MultimodalModelLoader>> {
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

            #[cfg_attr(
                not(any(
                    feature = "models-gemma",
                    feature = "models-llama",
                    feature = "models-other",
                    feature = "models-phi",
                    feature = "models-qwen"
                )),
                allow(unused_variables)
            )]
            fn get_processor(
                &self,
                model_config: &str,
                processor_config: Option<ProcessorConfig>,
                preprocessor_config: PreProcessorConfig,
                max_edge: Option<u32>,
            ) -> Result<Arc<dyn Processor + Send + Sync>> {
                match self {
                    $(
                        $(#[cfg(feature = $feature)])?
                        Self::$variant => Ok($loader.get_processor(
                            model_config,
                            processor_config,
                            preprocessor_config,
                            max_edge,
                        )),
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

inference_nn::multimodal_loader_table!(multimodal_loader_dispatch);

mod auto;
pub use auto::*;
#[cfg(feature = "models-gemma")]
pub use inference_models_gemma::loaders::{
    DiffusionGemmaLoader, Gemma3Loader, Gemma3nLoader, Gemma4Loader,
};
#[cfg(feature = "models-llama")]
pub use inference_models_llama::loaders::{
    Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, Mistral3Loader, VLlama4Loader,
    VLlamaLoader, VoxtralLoader,
};
#[cfg(feature = "models-other")]
pub use inference_models_other::loaders::{Lfm2VlLoader, PaddleOcrVlLoader};
#[cfg(feature = "models-phi")]
pub use inference_models_phi::loaders::{Phi3VLoader, Phi4MMLoader};
#[cfg(feature = "models-qwen")]
pub use inference_models_qwen::loaders::{
    MiniCpmOLoader, MuseGlimmerLoader, Qwen2_5VLLoader, Qwen2VLLoader, Qwen3_5Loader,
    Qwen3_5MoeLoader, Qwen3VLLoader, Qwen3VLMoELoader,
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
