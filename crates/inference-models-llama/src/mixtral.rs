//! Mixtral: the shared decoder with a sparse-MoE feed-forward under `block_sparse_moe`.

use inference_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{
    decoder::{BLOCK_SPARSE_MOE, DecoderSpec, MoeRouting, MoeSpec, RopeKind},
    layers::Activation,
    moe::ExpertProjNames,
    serde_default_fn,
};

serde_default_fn!(bool, word_emb_default, false);

/// <https://github.com/huggingface/transformers/blob/1a585c1222a56bcaecc070966d558d4a9d862e83/src/transformers/models/mixtral/configuration_mixtral.py#L113>
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
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
    pub num_experts_per_tok: usize,
    pub num_local_experts: usize,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    pub tie_word_embeddings: bool,
}

impl Config {
    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self) -> DecoderSpec {
        DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            hidden_act: self.hidden_act,
            rms_norm_eps: self.rms_norm_eps,
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            layer_windows: vec![self.sliding_window; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            moe: Some(MoeSpec {
                num_experts: self.num_local_experts,
                intermediate_size: self.intermediate_size,
                routing: MoeRouting::TopK {
                    k: self.num_experts_per_tok,
                    renormalize: true,
                },
                quantized_router: true,
                expert_names: ExpertProjNames::MIXTRAL,
                name: BLOCK_SPARSE_MOE,
                dense_layers: Vec::new(),
                cuda_decode_graphs: false,
            }),
            ..Default::default()
        }
    }
}

pub type Model = crate::decoder::CausalLm;
