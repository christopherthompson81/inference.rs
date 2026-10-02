use super::deepseek_family::*;
use super::*;

/// `NormalLoader` for a DeepSeekV2 model.
pub struct DeepSeekV2Loader;

impl NormalModelLoader for DeepSeekV2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::deepseek2::DeepSeekV2Config::from_json(config)?;

        Ok(Box::new(crate::deepseek2::DeepSeekV2::new(
            &cfg.family(),
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = crate::deepseek2::DeepSeekV2Config::from_json(config)?;
        Ok(Box::new(cfg))
    }
}

impl DeepSeekV2Loader {
    fn spec(config: &str) -> Result<FamilyLoaderSpec> {
        let cfg = crate::deepseek2::DeepSeekV2Config::from_json(config)?;
        let mut isq_head = vec![LM_HEAD];
        isq_head.extend(MLA_ATTENTION);
        isq_head.push(STACKED_EXPERTS);
        if cfg.q_lora_rank.is_some() {
            isq_head.extend(Q_LORA);
        } else {
            isq_head.push(Q_PROJ);
        }
        Ok(FamilyLoaderSpec {
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            max_position_embeddings: cfg.max_position_embeddings,
            tie_word_embeddings: cfg.tie_word_embeddings,
            num_kv_heads: cfg.num_attention_heads,
            k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
            attention: AttentionSizing::Mla {
                q_lora_rank: cfg.q_lora_rank,
                q_head_dim: cfg.q_head_dim(),
                kv_lora_rank: cfg.kv_lora_rank,
                qk_rope_head_dim: cfg.qk_rope_head_dim,
                v_head_dim: cfg.v_head_dim,
                attention_bias: cfg.attention_bias,
                packed_q: false,
            },
            moe: cfg.n_routed_experts.map(|n_routed_experts| MoeSizing {
                n_routed_experts,
                moe_intermediate_size: cfg.moe_intermediate_size,
                first_k_dense_replace: cfg.first_k_dense_replace,
                moe_layer_freq: Some(cfg.moe_layer_freq),
                shared_intermediate: cfg.n_shared_experts.map(|n| cfg.moe_intermediate_size * n),
                correction_bias: false,
            }),
            isq_head,
            loose_dense_up: true,
        })
    }
}

family_loader!(DeepSeekV2Loader);
