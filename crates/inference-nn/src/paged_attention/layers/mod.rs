/// One `PagedAttention::attend_selected` call: `(b, heads, s, head_dim)` queries, the `(b * s, n_kv)` additive f16
/// mask (causality included), and each sequence's cached rows, `(b,)` u32.
pub struct SelectedAttention<'a> {
    pub query: &'a inference_tensor::Tensor,
    pub mask: &'a inference_tensor::Tensor,
    pub key_cache: &'a inference_tensor::Tensor,
    pub value_cache: &'a inference_tensor::Tensor,
    pub input_metadata: &'a crate::paged_attention::PagedAttentionInputMetadata,
    pub seq_lens: &'a inference_tensor::Tensor,
    pub sdpa_params: &'a crate::attention::SdpaParams,
}

/// One `PagedAttention::attend_gathered` call: `(rows, heads, 1, head_dim)` queries, each over the leading `kept[row]`
/// of its row of `cells` (`(rows, width)` u32 positions, read through `block_table`'s row), all `(rows,)` u32 for
/// `kept`; `compact_table` is the `(rows, pages)` u32 identity table of the compact cache the cells go into.
pub struct GatheredAttention<'a> {
    pub query: &'a inference_tensor::Tensor,
    pub cells: &'a inference_tensor::Tensor,
    pub kept: &'a inference_tensor::Tensor,
    pub block_table: &'a inference_tensor::Tensor,
    pub compact_table: &'a inference_tensor::Tensor,
    pub key_cache: &'a inference_tensor::Tensor,
    pub value_cache: &'a inference_tensor::Tensor,
    pub sdpa_params: &'a crate::attention::SdpaParams,
}

#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub mod paged_attention;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub use paged_attention::PagedAttention;

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub mod paged_attention {
    use inference_tensor::{Device, Result, Tensor};

    use crate::attention::{AttentionMask, FlashParams, SdpaParams};
    use crate::paged_attention::Fp8AttentionScales;
    use crate::paged_attention::PagedAttentionInputMetadata;

    pub struct PagedAttention {
        fp8_attention_scales: Fp8AttentionScales,
        fp8_attention_scales_calibrated: bool,
    }

    impl PagedAttention {
        pub fn new(
            _head_dim: usize,
            _device: &Device,
            _alibi_slopes: Option<Vec<f32>>,
        ) -> Result<Self> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        pub fn new_with_fp8_attention_scales(
            _head_dim: usize,
            _device: &Device,
            _alibi_slopes: Option<Vec<f32>>,
            _fp8_attention_scales: Option<Fp8AttentionScales>,
        ) -> Result<Self> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        pub fn fp8_attention_scales(&self) -> Fp8AttentionScales {
            self.fp8_attention_scales
        }

        pub fn has_calibrated_fp8_attention_scales(&self) -> bool {
            self.fp8_attention_scales_calibrated
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &self,
            _query: &Tensor,
            _key: &Tensor,
            _value: &Tensor,
            _attention_mask: &AttentionMask,
            _key_cache: Option<Tensor>,
            _value_cache: Option<Tensor>,
            _input_metadata: &PagedAttentionInputMetadata,
            _sdpa_params: &SdpaParams,
            _flash_params: Option<&FlashParams>,
        ) -> Result<Tensor> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        #[allow(clippy::too_many_arguments)]
        pub fn gather_canvas_kv(
            &self,
            _key_cache: &Tensor,
            _value_cache: &Tensor,
            _input_metadata: &PagedAttentionInputMetadata,
            _seq_idx: usize,
            _kv_len: usize,
            _head_dim: usize,
            _dtype: inference_tensor::DType,
        ) -> Result<(Tensor, Tensor)> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        pub fn write_rows(
            &self,
            _key: &Tensor,
            _value: &Tensor,
            _key_cache: &Tensor,
            _value_cache: &Tensor,
            _slot_mapping: &Tensor,
        ) -> Result<()> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        pub fn attend_selected(&self, _call: super::SelectedAttention<'_>) -> Result<Tensor> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        pub fn attend_gathered(&self, _call: super::GatheredAttention<'_>) -> Result<Tensor> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        pub fn selected_kv_len(
            _input_metadata: &PagedAttentionInputMetadata,
            _device: &Device,
        ) -> Result<usize> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward_donor_cache(
            &self,
            _query: &Tensor,
            _key_cache: &Tensor,
            _value_cache: &Tensor,
            _attention_mask: &AttentionMask,
            _input_metadata: &PagedAttentionInputMetadata,
            _sdpa_params: &SdpaParams,
            _flash_params: Option<&FlashParams>,
        ) -> Result<Tensor> {
            inference_tensor::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }
    }
}

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub use paged_attention::PagedAttention;
