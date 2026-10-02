//! ISQ patterns, weight sizing and paged metadata shared by the DeepSeek-V2/V3 and GLM4-MoE(-Lite) loaders.

use super::*;

pub(super) const LM_HEAD: &str = r"lm_head\.(weight|bias)$";
pub(super) const STACKED_EXPERTS: &str =
    r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$";
pub(super) const O_PROJ: &str = r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$";
pub(super) const MLA_ATTENTION: [&str; 3] = [
    r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.(kv_b|k_b|v_b)_proj\.(weight|bias)$",
    O_PROJ,
];
pub(super) const Q_LORA: [&str; 2] = [
    r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$",
];
pub(super) const Q_PROJ: &str = r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$";
const PROMOTED: [&str; 2] = [
    r"^model\.embed_tokens\.weight$",
    r"^lm_head\.(weight|bias)$",
];
const MOQE: [&str; 2] = [
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
];

pub(super) enum AttentionSizing {
    Mla {
        q_lora_rank: Option<usize>,
        q_head_dim: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        attention_bias: bool,
        // DeepSeek sizes the q projections unpacked, GLM4-MoE-Lite packs them
        packed_q: bool,
    },
    Gqa {
        num_kv_heads: usize,
        head_dim: usize,
        attention_bias: bool,
        qk_norm: bool,
    },
}

pub(super) struct MoeSizing {
    pub n_routed_experts: usize,
    pub moe_intermediate_size: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: Option<usize>,
    pub shared_intermediate: Option<usize>,
    pub correction_bias: bool,
}

/// What the shared loader helpers read from one family config.
pub(super) struct FamilyLoaderSpec {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub num_kv_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub attention: AttentionSizing,
    pub moe: Option<MoeSizing>,
    // in order, ahead of the per-layer MLP patterns
    pub isq_head: Vec<&'static str>,
    // DeepSeek's dense up_proj pattern leaves the layer dots unescaped
    pub loose_dense_up: bool,
}

impl FamilyLoaderSpec {
    // `%` rather than is_multiple_of: a zero moe_layer_freq panics here as it always has
    #[allow(clippy::manual_is_multiple_of)]
    fn moe_layer(&self, layer_idx: usize) -> Option<&MoeSizing> {
        self.moe.as_ref().filter(|moe| {
            layer_idx >= moe.first_k_dense_replace
                && moe.moe_layer_freq.is_none_or(|freq| layer_idx % freq == 0)
        })
    }

    pub fn promoted_isq_predicates() -> Result<Vec<Regex>> {
        isq_regexes(&PROMOTED)
    }

    pub fn isq_layer_regexes_moqe() -> Result<Vec<Regex>> {
        isq_regexes(&MOQE)
    }

    pub fn isq_layer_regexes(&self) -> Result<Vec<Regex>> {
        let mut data = isq_regexes(&self.isq_head)?;
        for layer_idx in 0..self.num_hidden_layers {
            if let Some(moe) = self.moe_layer(layer_idx) {
                for i in 0..moe.n_routed_experts {
                    data.extend(isq_regexes(&[
                        format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ),
                        format!(r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"),
                        format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ),
                    ])?);
                }
                if moe.shared_intermediate.is_some() {
                    data.extend(isq_regexes(&[
                        format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ),
                        format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ),
                        format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ),
                    ])?);
                }
            } else {
                let up = if self.loose_dense_up {
                    format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$")
                } else {
                    format!(r"layers\.{layer_idx}\.mlp\.up_proj\.(weight|bias)$")
                };
                data.extend(isq_regexes(&[
                    format!(r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"),
                    up,
                    format!(r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"),
                ])?);
            }
        }
        Ok(data)
    }

    pub fn non_mapped_size_in_bytes(
        &self,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
    ) -> Result<usize> {
        standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: self.hidden_size,
                vocab_size: self.vocab_size,
                tie_word_embeddings: self.tie_word_embeddings,
            },
            quantization,
            dtype,
            weight_pack_factor,
        )
    }

    fn attention_elems(&self, weight_pack_factor: usize) -> usize {
        let h = self.hidden_size;
        let heads = self.num_attention_heads;
        match self.attention {
            AttentionSizing::Mla {
                q_lora_rank,
                q_head_dim,
                kv_lora_rank,
                qk_rope_head_dim,
                v_head_dim,
                attention_bias,
                packed_q,
            } => {
                let q_pack = if packed_q { weight_pack_factor } else { 1 };
                let q_proj = match q_lora_rank {
                    Some(lora_rank) => {
                        let a = h * lora_rank / q_pack;
                        let norm = lora_rank;
                        let b = (heads * q_head_dim) * lora_rank / q_pack;
                        a + norm + b
                    }
                    None => (heads * q_head_dim) * h / q_pack,
                };
                let kv_a_proj_with_mqa = h * (kv_lora_rank + qk_rope_head_dim) / weight_pack_factor
                    + bias_if!(attention_bias, kv_lora_rank + qk_rope_head_dim);
                let kv_a_layernorm = kv_lora_rank;
                let kv_b_proj = kv_lora_rank * heads * (q_head_dim - qk_rope_head_dim + v_head_dim)
                    / weight_pack_factor;
                let o_proj =
                    heads * v_head_dim * h / weight_pack_factor + bias_if!(attention_bias, h);
                q_proj + kv_a_layernorm + kv_a_proj_with_mqa + kv_b_proj + o_proj
            }
            AttentionSizing::Gqa {
                num_kv_heads,
                head_dim,
                attention_bias,
                qk_norm,
            } => {
                let q_proj = h * heads * head_dim / weight_pack_factor
                    + bias_if!(attention_bias, heads * head_dim);
                let k_proj = h * num_kv_heads * head_dim / weight_pack_factor
                    + bias_if!(attention_bias, num_kv_heads * head_dim);
                let v_proj = h * num_kv_heads * head_dim / weight_pack_factor
                    + bias_if!(attention_bias, num_kv_heads * head_dim);
                let o_proj = heads * head_dim * h / weight_pack_factor;
                let qk_norm = if qk_norm { head_dim * 2 } else { 0 };
                q_proj + k_proj + v_proj + o_proj + qk_norm
            }
        }
    }

    fn mlp_elems(&self, layer_idx: usize, weight_pack_factor: usize) -> usize {
        let h = self.hidden_size;
        match self.moe_layer(layer_idx) {
            Some(moe) => {
                let n = moe.n_routed_experts;
                let i = moe.moe_intermediate_size;
                let gate_proj = h * i / weight_pack_factor * n;
                let up_proj = h * i / weight_pack_factor * n;
                let down_proj = i * h / weight_pack_factor * n;
                let shared_experts = moe.shared_intermediate.map_or(0, |s| {
                    h * s / weight_pack_factor
                        + h * s / weight_pack_factor
                        + s * h / weight_pack_factor
                });
                let gate_weight = n * h;
                let e_score_correction_bias = if moe.correction_bias { n } else { 0 };
                gate_proj
                    + up_proj
                    + down_proj
                    + shared_experts
                    + gate_weight
                    + e_score_correction_bias
            }
            None => {
                let i = self.intermediate_size;
                h * i / weight_pack_factor + h * i / weight_pack_factor + i * h / weight_pack_factor
            }
        }
    }

    pub fn layer_sizes_in_bytes(&self, dtype: DType, weight_pack_factor: usize) -> Vec<usize> {
        let layernorms = 2 * self.hidden_size;
        let attention = self.attention_elems(weight_pack_factor);
        (0..self.num_hidden_layers)
            .map(|layer_idx| {
                (layernorms + attention + self.mlp_elems(layer_idx, weight_pack_factor))
                    * dtype.size_in_bytes()
            })
            .collect()
    }

    pub fn model_config(&self) -> Box<dyn ModelConfigLike> {
        Box::new(ModelConfigMetadata {
            max_seq_len: self.max_position_embeddings,
            num_layers: self.num_hidden_layers,
            hidden_size: self.hidden_size,
            num_kv_heads: self.num_kv_heads,
            num_attn_heads: self.num_attention_heads,
            sliding_window: None,
            k_head_dim: self.k_head_dim,
            v_head_dim: self.v_head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        })
    }
}

/// Implements `IsqModelLoader` and `DeviceMappedModelLoader` for a loader with a `spec(config)` function.
macro_rules! family_loader {
    ($loader:ty) => {
        impl IsqModelLoader for $loader {
            fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
                FamilyLoaderSpec::promoted_isq_predicates()
            }
            fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
                Self::spec(config)?.isq_layer_regexes()
            }
            fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes(config)
            }
            fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
                FamilyLoaderSpec::isq_layer_regexes_moqe()
            }
            fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes_moqe(config)
            }
        }

        impl DeviceMappedModelLoader for $loader {
            fn non_mapped_size_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<usize> {
                Self::spec(config)?.non_mapped_size_in_bytes(
                    dtype,
                    weight_pack_factor,
                    quantization,
                )
            }
            fn layer_sizes_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<Vec<usize>> {
                Ok(Self::spec(config)?.layer_sizes_in_bytes(dtype, weight_pack_factor))
            }
            fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
                Ok(Self::spec(config)?.model_config())
            }
        }
    };
}
pub(super) use family_loader;

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    // shared experts are moe_intermediate_size * n_shared_experts wide, as in HF
    const DS2_PLAIN_Q: [usize; 5] = [26432, 48960, 48960, 48960, 48960];
    const DS3_LORA_Q: [usize; 5] = [24448, 46976, 46976, 46976, 46976];
    const LITE: [usize; 5] = [21376, 43936, 43936, 43936, 43936];
    const GLM: [usize; 5] = [15936, 38496, 38496, 38496, 38496];
    const PACK_FACTOR: usize = 2;

    fn deepseek(q_lora_rank: Option<usize>) -> Value {
        json!({
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48, "moe_intermediate_size": 16,
            "num_hidden_layers": 5, "num_attention_heads": 4, "n_shared_experts": 2,
            "n_routed_experts": 8, "routed_scaling_factor": 2.5, "num_experts_per_tok": 2,
            "moe_layer_freq": 1, "first_k_dense_replace": 1, "hidden_act": "silu",
            "max_position_embeddings": 64, "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
            "rope_theta": 10000.0, "rope_scaling": null, "attention_bias": false,
            "q_lora_rank": q_lora_rank, "qk_rope_head_dim": 8, "kv_lora_rank": 16, "v_head_dim": 16,
            "qk_nope_head_dim": 8, "n_group": 4, "topk_group": 2,
        })
    }

    fn glm() -> Value {
        json!({
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48, "moe_intermediate_size": 16,
            "num_hidden_layers": 5, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "partial_rotary_factor": 0.5, "use_qk_norm": true, "attention_bias": true,
            "q_lora_rank": 16, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 8,
            "v_head_dim": 16, "n_routed_experts": 8, "n_shared_experts": 2, "num_experts_per_tok": 2,
            "first_k_dense_replace": 1, "routed_scaling_factor": 2.5, "n_group": 4, "topk_group": 2,
            "moe_layer_freq": 1, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
            "max_position_embeddings": 64, "hidden_act": "silu", "tie_word_embeddings": false,
        })
    }

    fn sizes(loader: &dyn DeviceMappedModelLoader, config: &Value) -> Result<Vec<usize>> {
        loader.layer_sizes_in_bytes(&config.to_string(), DType::F32, PACK_FACTOR, None)
    }

    #[test]
    fn layer_sizes_are_pinned() -> Result<()> {
        assert_eq!(sizes(&DeepSeekV2Loader, &deepseek(None))?, DS2_PLAIN_Q);
        assert_eq!(sizes(&DeepSeekV3Loader, &deepseek(Some(16)))?, DS3_LORA_Q);
        assert_eq!(sizes(&GLM4MoeLiteLoader, &glm())?, LITE);
        assert_eq!(sizes(&GLM4MoeLoader, &glm())?, GLM);
        Ok(())
    }
}
