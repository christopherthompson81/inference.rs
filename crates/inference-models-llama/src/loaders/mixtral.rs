use super::*;

pub struct MixtralLoader;

impl NormalModelLoader for MixtralLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::mixtral::Config::from_json(config)?;

        Ok(Box::new(crate::mixtral::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::mixtral::Config::from_json(config)?;

        Ok(Box::new(crate::xlora::mixtral::XLoraModel::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::mixtral::Config::from_json(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for MixtralLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^model\.embed_tokens\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            // Attention
            r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // Experts
            r"layers\.(\d+)\.block_sparse_moe\.gate\.(weight|bias)$",
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$",
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$",
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$",
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$",
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$",
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$",
            r"layers\.(\d+)\.block_sparse_moe\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for MixtralLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::mixtral::Config::from_json(config)?;
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
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = crate::mixtral::Config::from_json(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let moe_block = {
                let gate = cfg.hidden_size * cfg.num_local_experts;
                // Assume quantizing weight pack factor
                let w1 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w2 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w3 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                gate + cfg.num_local_experts * w1
                    + cfg.num_local_experts * w2
                    + cfg.num_local_experts * w3
            };

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + moe_block
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::mixtral::Config::from_json(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
