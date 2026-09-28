use std::collections::HashMap;

use candle_core::{quantized::ggml_file, DType, Device, Tensor};
use inference_quant::ShardedVarBuilder;

use super::Content;
use crate::{
    attention::FlashParams,
    device_map::DeviceMapper,
    kv_cache::EitherCache,
    lora::{LoraConfig, Ordering},
    xlora::{NonGranularState, XLoraConfig},
};

/// A quantized model built from a GGML file.
pub trait FromGGML {
    fn from_ggml(
        ct: ggml_file::Content,
        gqa: usize,
        dtype: DType,
    ) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

/// A quantized model with LoRA or X-LoRA adapters, built from a GGML file.
pub trait FromAdapterGGML {
    #[allow(clippy::too_many_arguments)]
    fn from_ggml(
        ct: ggml_file::Content,
        gqa: usize,
        lora_config: &[((String, String), LoraConfig)],
        vb: &ShardedVarBuilder,
        ordering: &Ordering,
        xlora_config: Option<XLoraConfig>,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
        dtype: DType,
    ) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

/// A quantized model with LoRA or X-LoRA adapters, built from GGUF content.
pub trait FromAdapterGGUF {
    #[allow(clippy::too_many_arguments)]
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: Content<'_, R>,
        device: &candle_core::Device,
        lora_config: &[((String, String), LoraConfig)],
        vb: &ShardedVarBuilder,
        ordering: &Ordering,
        xlora_config: Option<XLoraConfig>,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
        dtype: DType,
    ) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

/// What a GGML or GGUF adapter pipeline hands its model each step.
pub struct QuantizedForwardInputs<'a> {
    pub input_ids: &'a Tensor,
    pub input_ids_full: &'a Tensor,
    pub seqlen_offsets: &'a [usize],
    pub seqlen_offsets_full: &'a [usize],
    pub no_kv_cache: bool,
    pub non_granular_state: &'a Option<NonGranularState>,
    pub context_lens: Vec<(usize, usize)>,
    pub flash_params: &'a FlashParams,
    pub flash_params_full: &'a FlashParams,
}

/// A quantized model loaded straight from a GGML or GGUF file, as its pipeline drives it.
pub trait QuantizedModel: Send + Sync {
    fn forward_step(&self, inputs: QuantizedForwardInputs<'_>) -> candle_core::Result<Tensor>;
    fn cache(&self) -> &EitherCache;
    fn device(&self) -> &Device;
    fn max_seq_len(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
}
