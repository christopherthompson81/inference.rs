//! Qwen2-VL's text stack: the shared decoder with M-RoPE, F32 RMS norms and F32 eager attention.

use inference_quant::ShardedVarBuilder;
use inference_tensor::Result;

use crate::{
    decoder::{CausalLm, DecoderSpec, NormKind, RopeKind},
    layers,
    model::NormalLoadingMetadata,
    paged_attention::AttentionImplementation,
};

use super::config::QwenVlConfig;

// MLX checkpoints nest the text stack one level deeper than Hugging Face's `model.*`
const MLX_EMBED_TOKENS: &str = "language_model.model.embed_tokens.weight";

impl<V> QwenVlConfig<V> {
    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::MRope {
                theta: self.rope_theta as f32,
                sections: self.rope_scaling.mrope_section.clone(),
                interleaved: false,
            },
            max_position_embeddings: self.max_position_embeddings,
            qkv_bias: true,
            layer_windows: self.layer_sliding_windows()?,
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            norm: NormKind::F32Rms,
            eager_attention_f32: true,
            ..Default::default()
        })
    }
}

/// The text model, under Hugging Face's `model.` or MLX's `language_model.model.`.
pub fn text_model<V>(
    cfg: &QwenVlConfig<V>,
    vb: ShardedVarBuilder,
    is_gptx: bool,
    normal_loading_metadata: NormalLoadingMetadata,
    attention_mechanism: AttentionImplementation,
) -> Result<CausalLm> {
    let vb_m = if layers::contains_tensor_or_uqff(&vb, MLX_EMBED_TOKENS) {
        vb.pp("language_model").pp("model")
    } else {
        vb.pp("model")
    };
    CausalLm::new_inner(
        &cfg.decoder_spec()?,
        vb_m,
        vb.pp("lm_head"),
        is_gptx,
        normal_loading_metadata,
        attention_mechanism,
    )
}
