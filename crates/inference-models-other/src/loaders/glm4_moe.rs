use super::deepseek_family::*;
use super::*;

const K_PROJ: &str = r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$";
const V_PROJ: &str = r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$";

/// `NormalLoader` for a GLM 4 MoE model (GLM-4.5).
pub struct GLM4MoeLoader;

impl NormalModelLoader for GLM4MoeLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;
        Ok(Box::new(crate::glm4_moe::Glm4Moe::new(
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
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;
        Ok(Box::new(cfg))
    }
}

impl GLM4MoeLoader {
    fn spec(config: &str) -> Result<FamilyLoaderSpec> {
        let cfg = crate::glm4_moe::Glm4MoeConfig::from_json(config)?;
        let head_dim = cfg.head_dim();
        Ok(FamilyLoaderSpec {
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            max_position_embeddings: cfg.max_position_embeddings,
            tie_word_embeddings: cfg.tie_word_embeddings,
            num_kv_heads: cfg.num_key_value_heads,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            attention: AttentionSizing::Gqa {
                num_kv_heads: cfg.num_key_value_heads,
                head_dim,
                attention_bias: cfg.attention_bias,
                qk_norm: cfg.use_qk_norm,
            },
            moe: Some(MoeSizing {
                n_routed_experts: cfg.n_routed_experts,
                moe_intermediate_size: cfg.moe_intermediate_size,
                first_k_dense_replace: cfg.first_k_dense_replace,
                moe_layer_freq: None,
                shared_intermediate: (cfg.n_shared_experts > 0)
                    .then_some(cfg.moe_intermediate_size * cfg.n_shared_experts),
                correction_bias: true,
            }),
            isq_head: vec![
                LM_HEAD,
                Q_PROJ,
                K_PROJ,
                V_PROJ,
                MLA_ATTENTION[2],
                STACKED_EXPERTS,
            ],
            loose_dense_up: false,
        })
    }
}

family_loader!(GLM4MoeLoader);
