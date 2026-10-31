//! ERNIE 4.5 text models: dense (`Ernie4_5ForCausalLM`) and MoE (`Ernie4_5_MoeForCausalLM`), the shared decoder with
//! 1D RoPE over adjacent pairs and F32 RMS norms. The MoE routes on f32 softmax plus a selection bias, beside one
//! ungated MLP that fuses the shared experts.

use inference_quant::QuantizedConfig;
use inference_tensor::Result;

use crate::{
    decoder::{
        DecoderSpec, MLP, MoeRouting, MoeSpec, MtpSpec, NormKind, RopeKind, SharedExpertSpec,
    },
    layers::Activation,
    moe::ExpertProjNames,
    serde_default_fn,
};

serde_default_fn!(bool, tie_word_embeddings_default, true);
serde_default_fn!(f64, rope_theta_default, 500_000.0);
serde_default_fn!(f64, norm_min_default, 1e-12);
serde_default_fn!(usize, moe_layer_start_default, 1);
serde_default_fn!(i64, moe_layer_end_default, -1);
serde_default_fn!(usize, moe_layer_interval_default, 1);

const SHARED_EXPERTS: &str = "shared_experts";
// vLLM's `ErnieMultiTokenPredictorLayer`: the block's output feeds `lm_head` with no norm of its own
const MTP: MtpSpec = MtpSpec {
    embedding_norm: "mtp_emb_norm.0",
    hidden_norm: "mtp_hidden_norm.0",
    projection: "mtp_linear_proj.0",
    block: "mtp_block.0",
    final_norm: None,
    dense_block: true,
    mask_first_position: true,
};

#[derive(Debug, Clone, Default, serde::Serialize, inference_json::FromJson)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "rope_theta_default")]
    pub rope_theta: f64,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default = "tie_word_embeddings_default")]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    /// The loader's flag that `--mtp` asked for the checkpoint's MTP head.
    #[serde(default, rename = "_inference_mtp")]
    pub mtp: bool,
}

impl Config {
    /// The MTP layers this load keeps: the checkpoint's one, when `--mtp` asked for it.
    pub fn mtp_layers(&self) -> usize {
        if self.mtp {
            self.num_nextn_predict_layers
        } else {
            0
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        if self.use_bias {
            inference_tensor::bail!(
                "ERNIE 4.5 with biased projections (`use_bias: true`) is not supported"
            );
        }
        if self.mtp_layers() > 1 {
            inference_tensor::bail!(
                "ERNIE 4.5 MTP supports one layer, the config has {}",
                self.num_nextn_predict_layers
            );
        }
        Ok(DecoderSpec {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
            hidden_act: Activation::Silu,
            rms_norm_eps: self.rms_norm_eps,
            // the loader pairs adjacent features (`is_gptx` false), as transformers' interleaved rotate_half does
            rope: RopeKind::Default {
                theta: self.rope_theta as f32,
            },
            max_position_embeddings: self.max_position_embeddings,
            layer_windows: vec![None; self.num_hidden_layers],
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: self.quantization_config.clone(),
            // `Ernie4_5RMSNorm` normalises in F32 and applies the weight after casting back
            norm: NormKind::F32Rms,
            mtp: (self.mtp_layers() == 1).then_some(MTP),
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, inference_json::FromJson)]
pub struct MoeConfig {
    #[serde(flatten)]
    pub base: Config,
    pub moe_intermediate_size: usize,
    pub moe_num_experts: usize,
    pub moe_k: usize,
    #[serde(default)]
    pub moe_num_shared_experts: usize,
    #[serde(default = "moe_layer_start_default")]
    pub moe_layer_start_index: usize,
    /// The last MoE layer; -1 is the last layer.
    #[serde(default = "moe_layer_end_default")]
    pub moe_layer_end_index: i64,
    #[serde(default = "moe_layer_interval_default")]
    pub moe_layer_interval: usize,
    #[serde(default = "norm_min_default")]
    pub moe_norm_min: f64,
}

impl MoeConfig {
    #[allow(clippy::cast_possible_wrap)]
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        let end = match self.moe_layer_end_index {
            -1 => self.base.num_hidden_layers as i64 - 1,
            end => end,
        };
        (layer_idx + 1).is_multiple_of(self.moe_layer_interval)
            && layer_idx >= self.moe_layer_start_index
            && layer_idx as i64 <= end
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        let shared_expert = (self.moe_num_shared_experts > 0).then_some(SharedExpertSpec {
            intermediate_size: self.moe_intermediate_size * self.moe_num_shared_experts,
            name: SHARED_EXPERTS,
            gated: false,
        });
        Ok(DecoderSpec {
            moe: Some(MoeSpec {
                num_experts: self.moe_num_experts,
                intermediate_size: self.moe_intermediate_size,
                routing: MoeRouting::BiasedTopK {
                    k: self.moe_k,
                    norm_min: self.moe_norm_min as f32,
                },
                quantized_router: false,
                expert_names: ExpertProjNames::DEFAULT,
                name: MLP,
                dense_layers: (0..self.base.num_hidden_layers)
                    .filter(|&layer_idx| !self.is_moe_layer(layer_idx))
                    .collect(),
                cuda_decode_graphs: true,
                shared_expert,
            }),
            ..self.base.decoder_spec()?
        })
    }
}

pub type Model = crate::decoder::CausalLm;
