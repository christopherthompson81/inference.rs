//! The Qwen3 embedder: the Qwen3 decoder stack without its `model.` prefix, KV cache or `lm_head`.

use inference_quant::ShardedVarBuilder;
use inference_tensor::{Device, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::FlashParams,
    decoder::DecoderStack,
    layers::masker::NotACache,
    model::{EmbeddingModel, IsqModel, ModelForwardContext, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    utils::unvarbuilder::UnVarBuilder,
};

pub use crate::qwen3::Config;

const EMBED_TOKENS: &str = "embed_tokens.weight";
// a GGUF's decoder translation keeps Qwen3ForCausalLM's `model.`
const BASE_PREFIX: &str = "model";

pub struct Model {
    stack: DecoderStack,
    prefix: Option<&'static str>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            inference_tensor::bail!("Expected AttentionImplementation::Eager");
        }
        let prefix = (!vb.contains_tensor(EMBED_TOKENS)).then_some(BASE_PREFIX);
        let vb = match prefix {
            Some(prefix) => vb.pp(prefix),
            None => vb,
        };
        Ok(Self {
            prefix,
            stack: DecoderStack::new(
                &cfg.decoder_spec(),
                vb,
                None,
                is_gptx,
                normal_loading_metadata,
                &attention_mechanism,
            )?,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, flash_params: &FlashParams) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()?;
        let offsets = vec![0; batch];
        let context_lens = vec![(0, seq_len); batch];
        let position_ids = vec![seq_len; batch];
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, flash_params);
        let xs = self.stack.embed(input_ids)?;
        let masks = self.stack.masks(input_ids, xs.dtype(), &NotACache, true)?;
        self.stack.forward(xs, &masks, None, &mut ctx)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let root = UnVarBuilder::new();
        let prefixed = self.prefix.map(|prefix| root.pp(prefix));
        self.stack
            .residual_uvb(prefixed.as_ref().unwrap_or(&root), false);
        root.to_safetensors()
    }
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        _extras: &crate::embedding::EmbeddingExtras,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(input_ids, flash_params)
    }
    fn device(&self) -> &Device {
        &self.stack.device
    }
}

impl AnyMoeBaseModelMixin for Model {}
