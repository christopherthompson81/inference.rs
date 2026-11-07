use super::*;

use crate::rwkv::MAX_POSITIONS;
use crate::rwkv7::Config;

/// `NormalLoader` for RWKV-7 (`RWKV7ForCausalLM`, flash-linear-attention's layout).
pub struct Rwkv7Loader;

impl NormalModelLoader for Rwkv7Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(crate::rwkv7::model(
            &Config::from_json(config)?,
            vb,
            normal_loading_metadata,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
    }
}

impl IsqModelLoader for Rwkv7Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"^model\.embeddings\.weight$", r"^lm_head\.(weight|bias)$"])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"layers\.(\d+)\.attn\.(r_proj|k_proj|v_proj|o_proj)\.weight$",
            r"layers\.(\d+)\.ffn\.(key|value)\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Rwkv7Loader {
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
    // the four projections, the channel-mix, the LoRAs and the vectors and norms
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let h = cfg.hidden_size;
        let projections = (4 * h * h + 2 * h * cfg.ffn_size()) / weight_pack_factor;
        let ranks = cfg.decay_low_rank_dim
            + cfg.a_low_rank_dim
            + cfg.v_low_rank_dim
            + cfg.gate_low_rank_dim;
        let small = 2 * h * ranks + 20 * h;
        Ok(vec![
            (projections + small) * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Config::from_json(config)?.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Config::from_json(config)?;
        let base = ModelConfigMetadata {
            max_seq_len: MAX_POSITIONS,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: 1,
            num_attn_heads: 1,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: inference_nn::paged_attention::KvCacheLayout::Standard,
        };
        Ok(Box::new(
            inference_nn::paged_attention::HybridPagedKvCacheConfig::new(
                base,
                vec![false; cfg.num_hidden_layers],
            )
            .with_recurrent_state_bytes(cfg.shape().recurrent_state_bytes(DType::F32)),
        ))
    }
}
