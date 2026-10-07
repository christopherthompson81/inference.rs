use crate::utils::varbuilder_utils::{
    DeviceForLoadTensor, from_mmaped_safetensors, load_preload_adapters,
};
use anyhow::Result;
use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, quantized::ggml_file};
use std::{collections::HashMap, path::PathBuf, sync::Arc};

use crate::{
    device_map::DeviceMapper,
    gguf::Content,
    lora::{LoraConfig, Ordering},
    pipeline::{AdapterPaths, ModelPaths},
    xlora_models::XLoraConfig,
};

#[derive(derive_more::From)]
pub struct FileGGML {
    pub ct: ggml_file::Content,
    pub gqa: usize,
    pub dtype: DType,
}

#[derive(derive_more::From)]
pub struct Device<'a> {
    device: &'a inference_tensor::Device,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
}

pub struct Adapter<'a> {
    pub xlora_config: Option<XLoraConfig>,
    pub lora_config: &'a [((String, String), LoraConfig)],
    pub vb: ShardedVarBuilder,
    pub ordering: &'a Ordering,
    pub preload_adapters: Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
}

impl<'a> Adapter<'a> {
    // NOTE: It is not possible to store references for values returned by: load_preload_adapters() + from_mmaped_safetensors(),
    // As referenced value would drop after this method, Adapter takes ownership of vb + preload_adapters
    // and then passes by reference to the `from_gguf()` / `from_ggml()` methods when proxying to params.
    // NOTE: Due to reference usage persisting in returned struct, additional lifetime annotations were required.
    pub fn try_new<'b: 'a>(
        paths: &'b dyn ModelPaths,
        device: &'b inference_tensor::Device,
        silent: bool,
        is_xlora: bool,
    ) -> Result<Self> {
        let AdapterPaths::XLora {
            adapter_configs,
            adapter_safetensors,
            classifier_path,
            xlora_order,
            xlora_config,
            lora_preload_adapter_info,
        } = paths.get_adapter_paths()
        else {
            todo!()
        };

        let lora_config = adapter_configs.as_ref().unwrap();
        let ordering = xlora_order.as_ref().unwrap();
        let preload_adapters = load_preload_adapters(
            lora_preload_adapter_info,
            inference_tensor::DType::F32,
            device,
            silent,
        )?;

        // X-LoRA support:
        let mut xlora_paths: Vec<PathBuf> = vec![];
        if is_xlora {
            xlora_paths = vec![classifier_path.as_ref().unwrap().to_path_buf()];
        }

        // Create VarBuilder:
        // TODO: `from_mmaped_safetensors` has `xlora_paths` as the 2nd param (_valid but params need to be named better_)
        let vb = from_mmaped_safetensors(
            xlora_paths,
            adapter_safetensors
                .as_ref()
                .unwrap()
                .iter()
                .map(|(_, x)| (*x).to_owned())
                .collect::<Vec<_>>(),
            Some(inference_tensor::DType::F32),
            device,
            vec![None],
            silent,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;

        Ok(Self {
            lora_config,
            xlora_config: xlora_config.clone(),
            vb,
            ordering,
            preload_adapters,
        })
    }
}

// New type wrappers that segment the distinct parameter sets used by `from_ggml()` + `from_gguf()` methods:
pub struct ParamsGGML(pub FileGGML);
pub struct ParamsGGUF<'a, R: std::io::Seek + std::io::Read>(
    pub Content<'a, R>,
    pub Device<'a>,
    pub DType,
);

// A `None` type vs the `Some` type (`Adapter<'a>`)
pub struct NoAdapter {}

// Marker traits to restrict type input:
// (required workaround to support impl on subtypes, otherwise would use an enum)
pub trait QuantParams {}
impl QuantParams for ParamsGGML {}
impl<R: std::io::Seek + std::io::Read> QuantParams for ParamsGGUF<'_, R> {}

// Emulates `Option<Adapter>` but is compatible as a type bound in `impl<T>` for Some vs None
pub trait MaybeAdapter {}
impl MaybeAdapter for Adapter<'_> {}
impl MaybeAdapter for NoAdapter {}

// `derive_more::From` provides a terser construction for enum variants of `ModelParams`.
#[derive(derive_more::From)]
pub struct Config<Q: QuantParams, A: MaybeAdapter> {
    pub quant: Q,
    pub adapter: A,
}

#[allow(clippy::large_enum_variant)]
pub enum ModelParams<'a, Q: QuantParams> {
    Quantized(Config<Q, NoAdapter>),
    Adapted(Config<Q, Adapter<'a>>),
}

// A `builder()` method is derived from the `new()` method and its params (derived builder struct fields).
// NOTE: Intended to be built via fluent API in a single line, cannot conditionally append params.
// `.adapter(Adapter<' >)` or for conditional usage `.and_adapter(Option<Adapter<' >)` can be used.
// Otherwise omitting an `.adapter()` call prior to calling `build()` is ok, defaults to `None`.
impl<'a, Q: QuantParams> ModelParams<'a, Q> {
    pub fn new<'b: 'a>(quant: Q, adapter: Option<Adapter<'b>>) -> Self {
        match adapter {
            None => Self::Quantized((quant, NoAdapter {}).into()),
            Some(a) => Self::Adapted((quant, a).into()),
        }
    }

    fn expect_quantized(self, msg: &str) -> Config<Q, NoAdapter> {
        match self {
            Self::Quantized(config) => config,
            Self::Adapted(_) => panic!("{msg}"),
        }
    }

    fn expect_adapted(self, msg: &str) -> Config<Q, Adapter<'a>> {
        match self {
            Self::Adapted(config) => config,
            Self::Quantized(_) => panic!("{msg}"),
        }
    }
}

pub use inference_nn::gguf::{FromAdapterGGML, FromAdapterGGUF, FromGGML};

// NOTE: Below is a workaround to proxy params to the existing API methods `get_gguf()` / `get_gmml()` traits covered above.
impl Config<ParamsGGML, NoAdapter> {
    pub fn try_into_model<T: FromGGML>(self) -> Result<T, inference_tensor::Error> {
        // Destructure props:
        let ParamsGGML(FileGGML { ct, gqa, dtype }) = self.quant;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_ggml(ct, gqa, dtype)
    }
}

impl Config<ParamsGGML, Adapter<'_>> {
    pub fn try_into_model<T: FromAdapterGGML>(self) -> Result<T, inference_tensor::Error> {
        // Destructure props:
        let ParamsGGML(FileGGML { ct, gqa, dtype }) = self.quant;

        let Adapter {
            xlora_config,
            lora_config,
            vb,
            ordering,
            preload_adapters,
        } = self.adapter;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_ggml(
            ct,
            gqa,
            lora_config,
            &vb,
            ordering,
            xlora_config,
            &preload_adapters,
            dtype,
        )
    }
}

impl<R: std::io::Seek + std::io::Read> Config<ParamsGGUF<'_, R>, Adapter<'_>> {
    pub fn try_into_model<T: FromAdapterGGUF>(self) -> Result<T, inference_tensor::Error> {
        // Destructure props:
        let ParamsGGUF(ct, Device { device, mapper }, dtype) = self.quant;

        let Adapter {
            xlora_config,
            lora_config,
            vb,
            ordering,
            preload_adapters,
        } = self.adapter;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_gguf(
            ct,
            device,
            lora_config,
            &vb,
            ordering,
            xlora_config,
            mapper,
            &preload_adapters,
            dtype,
        )
    }
}

#[cfg(feature = "models-phi")]
use crate::xlora_models::XLoraQPhi3;
#[cfg(feature = "models-llama")]
use crate::{models::quantized_llama::ModelWeights as QLlama, xlora_models::XLoraQLlama};

#[cfg(feature = "models-llama")]
impl TryFrom<ModelParams<'_, ParamsGGML>> for QLlama {
    type Error = inference_tensor::Error;

    fn try_from(params: ModelParams<'_, ParamsGGML>) -> Result<Self, Self::Error> {
        let config = params.expect_quantized("`Config` should be GGML Quantized");
        config.try_into_model()
    }
}

#[cfg(feature = "models-llama")]
impl TryFrom<ModelParams<'_, ParamsGGML>> for XLoraQLlama {
    type Error = inference_tensor::Error;

    fn try_from(params: ModelParams<'_, ParamsGGML>) -> Result<Self, Self::Error> {
        let config = params.expect_adapted("`Config` should be GGML Quantized with an Adapter");
        config.try_into_model()
    }
}

macro_rules! adapted_gguf_model {
    ($model:ty) => {
        impl<R: std::io::Seek + std::io::Read> TryFrom<ModelParams<'_, ParamsGGUF<'_, R>>>
            for $model
        {
            type Error = inference_tensor::Error;

            fn try_from(params: ModelParams<'_, ParamsGGUF<'_, R>>) -> Result<Self, Self::Error> {
                let config =
                    params.expect_adapted("`Config` should be GGUF Quantized with an Adapter");
                config.try_into_model()
            }
        }
    };
}
#[cfg(feature = "models-llama")]
adapted_gguf_model!(XLoraQLlama);
#[cfg(feature = "models-phi")]
adapted_gguf_model!(XLoraQPhi3);
