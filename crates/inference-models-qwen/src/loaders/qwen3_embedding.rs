use super::*;
use crate::qwen3_embedding::{Config as Qwen3EmbeddingConfig, Model as Qwen3EmbeddingModel};

/// `EmbeddingModelLoader` for a Qwen 3 model.
pub struct Qwen3EmbeddingLoader;

impl EmbeddingModelLoader for Qwen3EmbeddingLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let cfg = Qwen3EmbeddingConfig::from_json(config)?;

        Ok(Box::new(Qwen3EmbeddingModel::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn has_causal_attention(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = Qwen3EmbeddingConfig::from_json(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Qwen3EmbeddingLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"^embed_tokens\.weight$"])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            // Attention
            r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            // MLP
            r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen3EmbeddingLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Qwen3EmbeddingConfig::from_json(config)?;
        let elems = {
            let embed_tokens_pack_factor = super::promoted_tensor_pack_factor(
                _quantization,
                "embed_tokens.weight",
                dtype,
                weight_pack_factor,
            )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let norm = cfg.hidden_size;
            embed_tokens + norm
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
        let cfg = Qwen3EmbeddingConfig::from_json(config)?;
        Ok(decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Qwen3EmbeddingConfig::from_json(config)?;
        Ok(Box::new(decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            cfg.decoder_spec().sliding_window(),
        )))
    }
}

fn decoder_shape(cfg: &Qwen3EmbeddingConfig) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        qkv_bias: false,
        o_bias: false,
        qk_norm: true,
        norms: 2,
        mlp: MlpShape::Gated {
            intermediate_size: cfg.intermediate_size,
        },
    }
}
