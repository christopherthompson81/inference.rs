use super::*;

use crate::rwkv::{MAX_POSITIONS, RwkvShape};
use crate::rwkv_hybrid::{ArwkvConfig, Qrwkv6Config};

// RWKV's state in the hybrid cache: no attention layer, so nothing to page
fn recurrent_model_config(shape: &RwkvShape<'_>) -> Box<dyn ModelConfigLike> {
    let base = ModelConfigMetadata {
        max_seq_len: MAX_POSITIONS,
        num_layers: shape.layers,
        hidden_size: shape.hidden,
        num_kv_heads: 1,
        num_attn_heads: 1,
        sliding_window: None,
        k_head_dim: shape.head_dim,
        v_head_dim: shape.head_dim,
        kv_cache_layout: inference_nn::paged_attention::KvCacheLayout::Standard,
    };
    Box::new(
        inference_nn::paged_attention::HybridPagedKvCacheConfig::new(
            base,
            vec![false; shape.layers],
        )
        .with_recurrent_state_bytes(shape.recurrent_state_bytes(DType::F32)),
    )
}

fn qwen2_ends(
    shape: &RwkvShape<'_>,
    quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
    dtype: DType,
    weight_pack_factor: usize,
) -> Result<usize> {
    standard_non_mapped_size_in_bytes(
        LanguageModelEnds {
            hidden_size: shape.hidden,
            vocab_size: shape.vocab,
            tie_word_embeddings: shape.tie_word_embeddings,
        },
        quantization,
        dtype,
        weight_pack_factor,
    )
}

// the hybrids' layer names: Qwen2's MLP beside each time-mix's projections
const HYBRID_PROMOTED: [&str; 2] = [
    r"^model\.embed_tokens\.weight$",
    r"^lm_head\.(weight|bias)$",
];
const ARWKV_ISQ: [&str; 3] = [
    r"lm_head\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.time_mixer\.(receptance|key|value|output)\.weight$",
    r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
];
const QRWKV6_ISQ: [&str; 3] = [
    r"lm_head\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj|gate)\.weight$",
    r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$",
];

/// `NormalLoader` for ARWKV (`RwkvHybridForCausalLM`: RWKV-7 time-mixes in Qwen2's decoder).
pub struct ArwkvLoader;

impl NormalModelLoader for ArwkvLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(crate::rwkv_hybrid::arwkv_model(
            &ArwkvConfig::from_json(config)?,
            vb,
            normal_loading_metadata,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(ArwkvConfig::from_json(config)?))
    }
}

impl IsqModelLoader for ArwkvLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&HYBRID_PROMOTED)
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&ARWKV_ISQ)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for ArwkvLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = ArwkvConfig::from_json(config)?;
        qwen2_ends(&cfg.shape(), quantization, dtype, weight_pack_factor)
    }
    // the four projections, the MLP, the LoRAs (llama.cpp's ranks: 64 decay and rate, 32 value, 128 gate) and vectors
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = ArwkvConfig::from_json(config)?;
        let h = cfg.hidden_size;
        let projections = (4 * h * h + 3 * h * cfg.intermediate_size) / weight_pack_factor;
        let ranks = 64 + 64 + 32 + if cfg.wkv_has_gate { 128 } else { 0 };
        let small = 2 * h * ranks + 16 * h;
        Ok(vec![
            (projections + small) * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(ArwkvConfig::from_json(config)?.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Ok(recurrent_model_config(
            &ArwkvConfig::from_json(config)?.shape(),
        ))
    }
}

/// `NormalLoader` for QRWKV6 (`RWKV6Qwen2ForCausalLM`: RWKV-6 time-mixes over gated linear attention in Qwen2's
/// decoder).
pub struct Qrwkv6Loader;

impl NormalModelLoader for Qrwkv6Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(crate::rwkv_hybrid::qrwkv6_model(
            &Qrwkv6Config::from_json(config)?,
            vb,
            normal_loading_metadata,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Qrwkv6Config::from_json(config)?))
    }
}

impl IsqModelLoader for Qrwkv6Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&HYBRID_PROMOTED)
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&QRWKV6_ISQ)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qrwkv6Loader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Qrwkv6Config::from_json(config)?;
        qwen2_ends(&cfg.shape(), quantization, dtype, weight_pack_factor)
    }
    // r, gate and o full width, k and v per key/value head, the MLP, the LoRAs at llama.cpp's widest ranks and vectors
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Qrwkv6Config::from_json(config)?;
        let h = cfg.hidden_size;
        let kv = cfg.kv_heads() * cfg.head_dim();
        let projections =
            (3 * h * h + 2 * h * kv + 3 * h * cfg.intermediate_size) / weight_pack_factor;
        let (mix_rank, decay_rank) = (128, 128);
        let small = 10 * h * mix_rank + 2 * h * decay_rank + 16 * h;
        Ok(vec![
            (projections + small) * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(Qrwkv6Config::from_json(config)?.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Ok(recurrent_model_config(
            &Qrwkv6Config::from_json(config)?.shape(),
        ))
    }
}
