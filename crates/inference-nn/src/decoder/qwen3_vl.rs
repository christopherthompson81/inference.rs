//! Qwen3-VL's text config and its shared-decoder spec, here so HiDream-O1's denoiser reuses the text stack.

use inference_quant::QuantizedConfig;

use super::{DecoderSpec, MLP, MoeRouting, MoeSpec, NormKind, QkNorm, RopeKind};
use crate::layers::Activation;
use crate::moe::ExpertProjNames;
use crate::serde_default_fn;

serde_default_fn!(usize, default_decoder_sparse_step, 1);
serde_default_fn!(bool, default_norm_topk_prob, true);

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MRopeScaling {
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub rope_scaling: MRopeScaling,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    // MoE fields; Qwen3-VL-MoE sets them, and with no experts every layer is a dense MLP.
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
}

impl TextConfig {
    /// Errors unless the experts fields fit the loader: Qwen3-VL-MoE needs experts, Qwen3-VL must have none.
    pub fn check_experts(&self, moe: bool) -> anyhow::Result<()> {
        if !moe && self.num_experts > 0 {
            anyhow::bail!(
                "text_config names {} experts: load it as qwen3vlmoe",
                self.num_experts
            );
        }
        if moe
            && (self.num_experts == 0
                || self.num_experts_per_tok == 0
                || self.moe_intermediate_size == 0
                || self.decoder_sparse_step == 0)
        {
            anyhow::bail!(
                "Qwen3-VL-MoE needs nonzero num_experts, num_experts_per_tok, moe_intermediate_size, decoder_sparse_step"
            );
        }
        Ok(())
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self, tie_word_embeddings: bool) -> DecoderSpec {
        let moe = (self.num_experts > 0).then(|| MoeSpec {
            num_experts: self.num_experts,
            intermediate_size: self.moe_intermediate_size,
            routing: MoeRouting::TopK {
                k: self.num_experts_per_tok,
                renormalize: self.norm_topk_prob,
            },
            quantized_router: false,
            expert_names: ExpertProjNames::DEFAULT,
            name: MLP,
            dense_layers: (0..self.num_hidden_layers)
                .filter(|&layer_idx| !self.is_moe_layer(layer_idx))
                .collect(),
            cuda_decode_graphs: true,
        });
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::MRope {
                theta: self.rope_theta as f32,
                sections: self.rope_scaling.mrope_section.clone(),
                interleaved: true,
            },
            max_position_embeddings: self.max_position_embeddings,
            qk_norm: Some(QkNorm::BEFORE_ROPE),
            // every text layer is full attention, whatever window the config names
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            // transformers normalises in F32 and scales after casting back, dense and MoE alike
            norm: NormKind::F32Rms,
            qk_norm_kind: Some(NormKind::Rms),
            moe,
            ..Default::default()
        }
    }
}
