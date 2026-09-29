use crate::attention::FlashParams;
use crate::paged_attention::PagedAttentionInputMetadata;
use std::{any::Any, sync::Arc};

use candle_core::Tensor;

#[cfg(feature = "models-gemma")]
pub(crate) mod diffusion_gemma;
#[cfg(feature = "models-gemma")]
pub(crate) mod gemma3;
#[cfg(feature = "models-gemma")]
pub(crate) mod gemma3n;
#[cfg(feature = "models-gemma")]
pub(crate) mod gemma4;
#[cfg(feature = "models-llama")]
pub(crate) mod idefics2;
#[cfg(feature = "models-llama")]
pub(crate) mod idefics3;
#[cfg(feature = "models-other")]
pub(crate) mod lfm2_vl;
#[cfg(feature = "models-llama")]
pub(crate) mod llama4;
#[cfg(feature = "models-llama")]
pub(crate) mod llava;
#[cfg(feature = "models-qwen")]
pub(crate) mod minicpmo;
#[cfg(feature = "models-llama")]
pub(crate) mod mistral3;
#[cfg(feature = "models-llama")]
pub(crate) mod mllama;
#[cfg(feature = "models-qwen")]
pub(crate) mod muse_glimmer;
#[cfg(feature = "models-other")]
pub(crate) mod paddleocr_vl;
#[cfg(feature = "models-phi")]
pub(crate) mod phi3;
#[cfg(feature = "models-phi")]
pub(crate) mod phi4;
#[cfg(feature = "models-qwen")]
pub(crate) mod qwen2vl;
#[cfg(feature = "models-qwen")]
pub(crate) mod qwen3_vl;
#[cfg(feature = "models-llama")]
pub(crate) mod voxtral;

pub(crate) mod media_host;
pub(crate) use inference_nn::media_inputs::{preprocessor_config, processor_config};

use crate::gdn::RecurrentBatchKind;

pub struct ModelInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Option<Tensor>,
    pub model_specific_args: Box<dyn Any>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
    pub flash_meta: FlashParams,
    pub recurrent_batch_kind: RecurrentBatchKind,
    pub adapter_leases: Arc<[Option<crate::AdapterLease>]>,
}

pub(crate) fn adapter_leases(
    input_seqs: &[&mut crate::sequence::Sequence],
    seq_indices: &[usize],
) -> Arc<[Option<crate::AdapterLease>]> {
    seq_indices
        .iter()
        .map(|&index| input_seqs[index].adapter_lease().cloned())
        .collect::<Vec<_>>()
        .into()
}
