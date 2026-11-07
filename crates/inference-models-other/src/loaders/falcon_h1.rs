use super::*;

use crate::falcon_h1::Config;

/// `NormalLoader` for Falcon-H1 (`FalconH1ForCausalLM`): its layers as Granite's parallel hybrid blocks.
pub struct FalconH1Loader;

impl NormalModelLoader for FalconH1Loader {
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

impl IsqModelLoader for FalconH1Loader {
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
            r"layers\.(\d+)\.mamba\.(in_proj|out_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.feed_forward\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for FalconH1Loader {
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
    // attention, the Mamba mixer, the SwiGLU MLP and the two norms
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let h = cfg.hidden_size;
        let q = cfg.num_attention_heads * cfg.head_dim();
        let kv = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) * cfg.head_dim();
        let attention = (h * q * 2 + h * kv * 2) / weight_pack_factor;
        let inner = cfg.mamba_d_ssm;
        let conv_dim = inner + 2 * cfg.mamba_n_groups * cfg.mamba_d_state;
        let in_proj = (inner + conv_dim + cfg.mamba_n_heads) * h;
        let mamba = (in_proj + inner * h) / weight_pack_factor
            + conv_dim * (cfg.mamba_d_conv + 1)
            + cfg.mamba_n_heads * 3
            + inner;
        let mlp = 3 * h * cfg.intermediate_size / weight_pack_factor;
        let layer = (attention + mamba + mlp + 2 * h) * dtype.size_in_bytes();
        Ok(vec![layer; cfg.num_hidden_layers])
    }
    // the stack's layers map to devices; the recurrent slots after them are the cache's, not the mapper's
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Config::from_json(config)?.num_hidden_layers)
    }
    // as the built model reports it: each layer's attention pages, its recurrent state a slot after the stack's
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Config::from_json(config)?;
        let granite = cfg.granite()?;
        let layout = cfg.layout();
        let base = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: 2 * cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: granite.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: inference_nn::paged_attention::KvCacheLayout::Standard,
        };
        let state =
            crate::granite::recurrent_state_bytes(&granite, cfg.num_hidden_layers, DType::F32);
        Ok(Box::new(crate::granite::hybrid_paged_kv_config(
            base, &layout, false, state,
        )))
    }
}
