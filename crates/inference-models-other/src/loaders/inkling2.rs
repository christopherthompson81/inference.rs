use super::*;

// a verify row's logits as the head, the softcap and the cut to the vocabulary each copy them
const VERIFY_LOGIT_COPIES: usize = 3;
// the drafter's taps: three layer visits and the normed output
const VERIFY_TAPS: usize = 4;

/// `NormalLoader` for Inkling 2's looped text model.
pub struct Inkling2Loader;

impl NormalModelLoader for Inkling2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = parse(config)?;
        Ok(Box::new(crate::inkling2::Model::new(
            &cfg,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(parse(config)?))
    }

    // the paged cache is read by the fused CUDA attention only
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(cfg!(feature = "cuda"))
    }
}

impl IsqModelLoader for Inkling2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"^model\.embed_tokens\.weight$", r"^lm_head\.weight$"])
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.(q|k|v|o|r|gate)_proj\.weight$",
            r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Inkling2Loader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = parse(config)?;
        standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                tie_word_embeddings: false,
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
        let cfg = parse(config)?;
        let hidden = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let attention = (hidden * (3 * q_dim + 2 * kv_dim) + q_dim * hidden) / weight_pack_factor;
        let experts =
            3 * cfg.n_routed_experts * hidden * cfg.intermediate_size / weight_pack_factor;
        let router = hidden * cfg.n_routed_experts;
        Ok(vec![
            (attention + experts + router) * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    // the weight layers; the cache's slots (each pass of the looped core its own) are `model_config`'s
    fn num_layers(&self, config: &str) -> Result<usize> {
        Ok(parse(config)?.num_hidden_layers)
    }

    // a verify row's logits (f32, the softcap's copies), the four taps and each slot's conv stash (its state and
    // inputs, f32)
    fn speculative_row_bytes(&self, config: &str) -> Result<Option<usize>> {
        let cfg = parse(config)?;
        let f32 = DType::F32.size_in_bytes();
        let logits = VERIFY_LOGIT_COPIES * cfg.vocab_size * f32;
        let taps = VERIFY_TAPS * cfg.hidden_size * f32;
        let stash = 2 * cfg.cache_slots() * cfg.conv_state_channels() * f32;
        Ok(Some(logits + taps + stash))
    }

    // the routed experts load as stored (NVFP4 bytes, MXFP8 values), MXFP8's E8M0 scales widened to f32
    fn checkpoint_tensor_bytes(
        &self,
        _config: &str,
        tensor_name: &str,
        dtype: &str,
        stored_bytes: usize,
    ) -> Option<usize> {
        if !tensor_name.contains(".mlp.experts.") {
            return None;
        }
        let widened = dtype == "U8" && tensor_name.ends_with("_weight_scale");
        Some(if widened {
            stored_bytes * DType::F32.size_in_bytes()
        } else {
            stored_bytes
        })
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = parse(config)?;
        Ok(Box::new(cfg.paged_cache_config(base_metadata(&cfg), None)))
    }

    // on CUDA the sliding slots keep window rings, which the planner counts per sequence rather than per token
    fn model_config_for(
        &self,
        config: &str,
        device: &inference_tensor::Device,
        paged: Option<crate::paged_attention::PagedKvLayout>,
    ) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = parse(config)?;
        let ring = cfg.window_ring(paged, device, DType::BF16);
        Ok(Box::new(
            cfg.paged_cache_config(base_metadata(&cfg), ring.as_ref()),
        ))
    }
}

// the text model's config: the checkpoint's own `text_config` where it nests one beside the vision tower's, whose
// experts' precisions (`hf_quant_config.json`) the checkpoint binding reads in place of its ModelOpt declaration
fn parse(config: &str) -> Result<crate::inkling2::Config> {
    let value: serde_json::Value = serde_json::from_str(config)?;
    match value.get("text_config") {
        Some(text) => {
            let mut cfg = crate::inkling2::Config::from_json(&text.to_string())?;
            cfg.quantization_config = None;
            Ok(cfg)
        }
        None => Ok(crate::inkling2::Config::from_json(config)?),
    }
}

// the slots as the paged cache sees them: one layer each, every visit's heads and head size
fn base_metadata(cfg: &crate::inkling2::Config) -> ModelConfigMetadata {
    let slots = cfg.cache_slots();
    ModelConfigMetadata {
        max_seq_len: cfg.max_position_embeddings,
        num_layers: slots,
        hidden_size: cfg.hidden_size,
        num_kv_heads: cfg.num_key_value_heads,
        num_attn_heads: cfg.num_attention_heads,
        sliding_window: None,
        k_head_dim: cfg.head_dim,
        v_head_dim: cfg.head_dim,
        kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
    }
}
