use super::*;

use crate::mamba2::{Config, MAX_POSITIONS};

/// `NormalLoader` for Mamba-2 (`Mamba2ForCausalLM`): its layers as Granite's hybrid blocks, every one a Mamba block.
pub struct Mamba2Loader;

impl NormalModelLoader for Mamba2Loader {
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
            vb.pp("backbone"),
            vb.pp("lm_head"),
            true,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
    }
}

impl IsqModelLoader for Mamba2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^backbone\.embeddings?\.weight$",
            r"^lm_head\.(weight|bias)$",
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"layers\.(\d+)\.mixer\.(in_proj|out_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Mamba2Loader {
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
    // the Mamba mixer and its norm
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let h = cfg.hidden_size;
        let inner = cfg.inner_size();
        let conv_dim = inner + 2 * cfg.n_groups * cfg.state_size;
        let in_proj = (inner + conv_dim + cfg.num_heads) * h;
        let mamba = (in_proj + inner * h) / weight_pack_factor
            + conv_dim * (cfg.conv_kernel + 1)
            + cfg.num_heads * 3
            + inner;
        Ok(vec![
            (mamba + h) * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Config::from_json(config)?.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Config::from_json(config)?;
        let granite = cfg.granite()?;
        let base = ModelConfigMetadata {
            max_seq_len: MAX_POSITIONS,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: 1,
            num_attn_heads: 1,
            sliding_window: None,
            k_head_dim: cfg.hidden_size,
            v_head_dim: cfg.hidden_size,
            kv_cache_layout: inference_nn::paged_attention::KvCacheLayout::Standard,
        };
        let state =
            crate::granite::recurrent_state_bytes(&granite, cfg.num_hidden_layers, DType::F32);
        Ok(Box::new(crate::granite::hybrid_paged_kv_config(
            base,
            &cfg.layout(),
            false,
            state,
        )))
    }
}
