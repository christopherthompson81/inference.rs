use super::*;

use crate::rwkv::MAX_POSITIONS;
use crate::rwkv6::Config;

/// `NormalLoader` for RWKV-6 (`Rwkv6ForCausalLM`, RWKV's HF layout).
pub struct Rwkv6Loader;

impl NormalModelLoader for Rwkv6Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(crate::rwkv6::model(
            &Config::from_json(config)?,
            vb,
            normal_loading_metadata,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
    }
}

impl IsqModelLoader for Rwkv6Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"^rwkv\.embeddings\.weight$", r"^head\.(weight|bias)$"])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^head\.(weight|bias)$",
            r"blocks\.(\d+)\.attention\.(receptance|key|value|gate|output)\.weight$",
            r"blocks\.(\d+)\.feed_forward\.(key|receptance|value)\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Rwkv6Loader {
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
    // the five time-mix projections, the channel-mix, the two LoRAs and the vectors and norms
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let h = cfg.hidden_size;
        let projections = (6 * h * h + 2 * h * cfg.ffn_size()) / weight_pack_factor;
        // llama.cpp's LoRA widths: the mixes' 32 (64 at 4096), the decay's 64 (128)
        let (mix_rank, decay_rank) = if h >= 4096 { (64, 128) } else { (32, 64) };
        let small = 10 * h * mix_rank + 2 * h * decay_rank + 16 * h;
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
            k_head_dim: cfg.head_size,
            v_head_dim: cfg.head_size,
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
