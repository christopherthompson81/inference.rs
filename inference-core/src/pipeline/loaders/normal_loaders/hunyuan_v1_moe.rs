use super::*;

/// [`NormalLoader`] for a HunYuanMoEV1 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct HunYuanMoEV1Loader;

impl NormalModelLoader for HunYuanMoEV1Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::hunyuan_v1_moe::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for HunYuanMoEV1Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Dense MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // MoE experts
            Regex::new(r"layers\.(\d+)\.mlp\.shared_mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.shared_mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.shared_mlp\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for HunYuanMoEV1Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;
        let head_dim = cfg.head_dim();

        let mut layer_sizes = Vec::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = head_dim * cfg.num_attention_heads;
            let size_kv = head_dim * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let expert_size = {
                let expert_gate = h_size * cfg.intermediate_size / weight_pack_factor;
                let expert_up = h_size * cfg.intermediate_size / weight_pack_factor;
                let expert_down = cfg.intermediate_size * h_size / weight_pack_factor;
                expert_gate + expert_up + expert_down
            };
            let (router_size, mlp_size) = if cfg.uses_moe() {
                let shared_expert_size = if cfg.use_mixed_mlp_moe {
                    expert_size * cfg.num_shared_expert.get(layer_idx)
                } else {
                    0
                };
                (
                    h_size * cfg.num_experts,
                    shared_expert_size + expert_size * cfg.num_experts,
                )
            } else {
                (0, expert_size)
            };
            let qk_norm = if cfg.use_qk_norm { head_dim * 2 } else { 0 };

            let non_router_elems = input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + mlp_size
                + qk_norm;

            layer_sizes.push(
                non_router_elems * dtype.size_in_bytes() + router_size * DType::F32.size_in_bytes(),
            );
        }

        Ok(layer_sizes)
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
