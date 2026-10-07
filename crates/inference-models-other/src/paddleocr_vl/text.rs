//! ERNIE-4.5-0.3B (transformers `PaddleOCR*` text classes): the shared decoder with chunked M-RoPE and F32 RMS norms.

use super::config::TextConfig;
use crate::{
    decoder::{DecoderSpec, NormKind, RopeKind},
    layers::Activation,
};

impl TextConfig {
    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self, max_position_embeddings: usize) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            hidden_act: Activation::Silu,
            rms_norm_eps: self.rms_norm_eps,
            // `PaddleOCRRotaryEmbedding` takes f32 frequencies and Qwen2.5-VL's chunked section order
            rope: RopeKind::MRope {
                theta: self.rope_theta as f32,
                sections: self.mrope_section.to_vec(),
                interleaved: false,
            },
            max_position_embeddings,
            layer_windows: vec![None; self.num_hidden_layers],
            quantization_config: self.quantization_config.clone(),
            // `PaddleOCRRMSNorm` normalises in F32 and applies the weight after casting back
            norm: NormKind::F32Rms,
            ..Default::default()
        }
    }
}
