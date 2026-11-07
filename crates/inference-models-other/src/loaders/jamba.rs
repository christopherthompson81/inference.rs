use super::*;

use crate::jamba::Config;

/// `NormalLoader` for Jamba (`JambaForCausalLM`): its Mamba 1 and attention layers as Granite's hybrid blocks.
pub struct JambaLoader;

impl NormalModelLoader for JambaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = Config::from_json(config)?;
        Ok(Box::new(crate::granite::GraniteMoeHybrid::new_with_layout(
            &cfg.granite()?,
            cfg.layout(),
            vb.pp("model"),
            vb.pp("lm_head"),
            true,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for JambaLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mamba\.(in_proj|x_proj|out_proj)\.weight$",
            r"layers\.(\d+)\.feed_forward\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for JambaLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Config::from_json(config)?;
        standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                tie_word_embeddings: cfg.tie_word_embeddings,
            },
            quantization,
            dtype,
            weight_pack_factor,
        )
    }
    // the mixer (attention or Mamba 1), the MLP or the experts and their router, and the two norms
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let h = cfg.hidden_size;
        let (inner, n, rank) = (cfg.inner_size(), cfg.mamba_d_state, cfg.dt_rank()?);
        let q = cfg.num_attention_heads * cfg.head_dim();
        let kv = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) * cfg.head_dim();
        let attention = (h * q * 2 + h * kv * 2) / weight_pack_factor;
        let mamba = (2 * inner * h + inner * (rank + 2 * n) + rank * inner + inner * h)
            / weight_pack_factor
            + inner * (cfg.mamba_d_conv + 1)
            + inner * n
            + 2 * inner
            + rank
            + 2 * n;
        let mlp = 3 * h * cfg.intermediate_size / weight_pack_factor;
        Ok((0..cfg.num_hidden_layers)
            .map(|layer| {
                let mixer = if cfg.is_attention(layer) {
                    attention
                } else {
                    mamba
                };
                let ffn = if cfg.is_moe(layer) {
                    cfg.num_experts * mlp + cfg.num_experts * h
                } else {
                    mlp
                };
                (mixer + ffn + 2 * h) * dtype.size_in_bytes()
            })
            .collect())
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Config::from_json(config)?.num_hidden_layers)
    }
    // attention layers page; each Mamba layer's state is a sequence's recurrent bytes
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Config::from_json(config)?;
        let granite = cfg.granite()?;
        let layout = cfg.layout();
        let base = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: granite.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: inference_nn::paged_attention::KvCacheLayout::Standard,
        };
        let mamba_layers = (0..cfg.num_hidden_layers)
            .filter(|&layer| !cfg.is_attention(layer))
            .count();
        let state = crate::granite::recurrent_state_bytes(&granite, mamba_layers, DType::F32);
        Ok(Box::new(crate::granite::hybrid_paged_kv_config(
            base, &layout, false, state,
        )))
    }
}
