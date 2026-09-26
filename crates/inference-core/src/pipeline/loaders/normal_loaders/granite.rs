use super::*;

/// [`NormalLoader`] for a GraniteMoeHybrid model (IBM Granite 4.0).
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GraniteMoeHybridLoader;

impl NormalModelLoader for GraniteMoeHybridLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::granite::GraniteMoeHybrid::new(
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
        anyhow::bail!("GraniteMoeHybrid does not support X-LoRA")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for GraniteMoeHybridLoader {
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
            Regex::new(r"layers\.(\d+)\.mamba\.(in_proj|out_proj)\.(weight|bias)$")?,
            // MLP (GraniteMLP uses shared_mlp.input_linear and shared_mlp.output_linear)
            Regex::new(r"layers\.(\d+)\.shared_mlp\.input_linear\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.shared_mlp\.output_linear\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.(input_linear|output_linear)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![Regex::new(
            r"layers\.(\d+)\.block_sparse_moe\.(input_linear|output_linear)\.weight$",
        )?])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for GraniteMoeHybridLoader {
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

        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        let attention_elems = {
            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;
            q_proj + k_proj + v_proj + o_proj
        };

        let mamba_elems = {
            let intermediate_size = cfg.mamba_intermediate_size();
            let conv_dim = cfg.mamba_conv_dim();
            let num_heads = cfg.mamba_n_heads();
            let projection_size = intermediate_size + conv_dim + num_heads;
            let in_proj =
                projection_size * cfg.hidden_size + bias_if!(cfg.mamba_proj_bias, projection_size);
            let conv1d = conv_dim * cfg.mamba_d_conv + bias_if!(cfg.mamba_conv_bias, conv_dim);
            let state = num_heads * 3;
            let norm = intermediate_size;
            let out_proj = cfg.hidden_size * intermediate_size
                + bias_if!(cfg.mamba_proj_bias, cfg.hidden_size);
            in_proj + conv1d + state + norm + out_proj
        };

        let shared_mlp_elems = {
            let shared_intermediate_size = if cfg.num_local_experts == 0 {
                cfg.shared_intermediate_size()
            } else {
                cfg.shared_intermediate_size.unwrap_or(0)
            };
            cfg.hidden_size * shared_intermediate_size * 2 / weight_pack_factor
                + shared_intermediate_size * cfg.hidden_size / weight_pack_factor
        };
        let routed_moe_elems = if cfg.num_local_experts > 0 {
            let router = cfg.num_local_experts * cfg.hidden_size;
            let input_linear = cfg.num_local_experts * cfg.intermediate_size * 2 * cfg.hidden_size;
            let output_linear = cfg.num_local_experts * cfg.hidden_size * cfg.intermediate_size;
            router + input_linear + output_linear
        } else {
            0
        };
        let common_elems = cfg.hidden_size * 2 + shared_mlp_elems + routed_moe_elems;

        Ok(cfg
            .layer_types()
            .into_iter()
            .map(|layer_type| {
                let operator_elems = match layer_type {
                    crate::models::granite::GraniteLayerType::Attention => attention_elems,
                    crate::models::granite::GraniteLayerType::Mamba => mamba_elems,
                };
                (common_elems + operator_elems) * dtype.size_in_bytes()
            })
            .collect())
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
