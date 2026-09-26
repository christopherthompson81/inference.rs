use crate::attention::FlashParams;
use crate::paged_attention::PagedAttentionInputMetadata;
use candle_core::{Result, Tensor};

use crate::{
    attention::{AttentionMask, Sdpa, SdpaParams},
    kv_cache::KvCache,
    paged_attention::PagedAttention,
};

/// Per-layer attention routing: paged attention when the model has it, else SDPA over the layer's KV cache.
pub struct AttentionDispatch<'a> {
    pub paged_attn: Option<&'a PagedAttention>,
    pub paged_layer: Option<((Tensor, Tensor), &'a PagedAttentionInputMetadata)>,
    pub kv_cache: &'a mut KvCache,
    pub sdpa_params: &'a SdpaParams,
    pub flash_params: &'a FlashParams,
}

impl AttentionDispatch<'_> {
    pub fn run(self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let Some(paged_attn) = self.paged_attn else {
            let (k, v) = self.kv_cache.append(k, v)?;
            return Sdpa.run_attention(q, &k, &v, mask, Some(self.flash_params), self.sdpa_params);
        };
        match self.paged_layer {
            Some(((key_cache, value_cache), metadata)) => paged_attn.forward(
                q,
                k,
                v,
                mask,
                Some(key_cache),
                Some(value_cache),
                metadata,
                self.sdpa_params,
                Some(self.flash_params),
            ),
            None => {
                // no cache blocks means a prompt-only pass (e.g. imatrix collection); a dummy plan skips cache writes
                if matches!(mask, AttentionMask::None) {
                    candle_core::bail!(
                        "paged attention without cache metadata needs a prompt attention mask"
                    );
                }
                let metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                paged_attn.forward(
                    q,
                    k,
                    v,
                    mask,
                    None,
                    None,
                    &metadata,
                    self.sdpa_params,
                    Some(self.flash_params),
                )
            }
        }
    }
}
