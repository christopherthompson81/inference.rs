//! EmbeddingGemma: Gemma 3's layers run as a bidirectional encoder, with no KV cache or `lm_head`.

use inference_quant::{ShardedVarBuilder, softcap};
use inference_tensor::{Device, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams},
    decoder::{DecoderStack, LayerMasks},
    device_map::DeviceMappedMask,
    layers::masker::BidirectionalMasker,
    model::{EmbeddingModel, IsqModel, ModelForwardContext, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    utils::unvarbuilder::UnVarBuilder,
};

/// EmbeddingGemma's config.json is Gemma 3's text config.
pub type EmbeddingGemmaConfig = crate::gemma3::config::Gemma3TextConfig;

pub struct EmbeddingGemma {
    stack: DecoderStack,
    sliding_window: usize,
    final_logit_softcap: Option<f32>,
}

impl EmbeddingGemma {
    pub fn new(
        cfg: &EmbeddingGemmaConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            inference_tensor::bail!("Expected AttentionImplementation::Eager");
        }
        let spec = cfg.decoder_spec();
        Ok(Self {
            stack: DecoderStack::new(
                &spec,
                vb,
                None,
                is_gptx,
                normal_loading_metadata,
                &attention_mechanism,
            )?,
            sliding_window: cfg.sliding_window,
            final_logit_softcap: spec.final_logit_softcap,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.stack.embed(input_ids)
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        xs: Tensor,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let offsets = vec![0; batch];
        let context_lens = vec![(0, seq_len); batch];
        let position_ids = vec![seq_len; batch];
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, flash_params);
        let mask =
            |mask: Tensor| DeviceMappedMask::new(AttentionMask::Custom(mask), &*self.stack.mapper);
        let masks = LayerMasks::new(
            Some(mask(BidirectionalMasker.make_mask(input_ids, xs.dtype())?)?),
            Some(mask(BidirectionalMasker.make_sliding_mask(
                input_ids,
                xs.dtype(),
                self.sliding_window,
            )?)?),
            None,
        );
        let xs = self.stack.forward(xs, &masks, None, &mut ctx)?;
        match self.final_logit_softcap {
            Some(cap) => softcap(&xs, cap)?.to_dtype(xs.dtype()),
            None => Ok(xs),
        }
    }
}

impl IsqModel for EmbeddingGemma {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.stack.residual_uvb(&uvb, false);
        uvb.to_safetensors()
    }
}

impl EmbeddingModel for EmbeddingGemma {
    fn forward(&self, input_ids: &Tensor, flash_params: &FlashParams) -> Result<Tensor> {
        self.forward_embeds(input_ids, self.embed_tokens(input_ids)?, flash_params)
    }
    fn device(&self) -> &Device {
        &self.stack.device
    }
}

impl AnyMoeBaseModelMixin for EmbeddingGemma {}
