use super::*;
use crate::modern_bert::{Config as ModernBertConfig, Model as ModernBertModel};
use inference_nn::model::EmbeddingModel;

/// `EmbeddingModelLoader` for ModernBERT encoders.
pub struct ModernBertLoader;

impl EmbeddingModelLoader for ModernBertLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let cfg = ModernBertConfig::from_json(config)?;
        Ok(Box::new(ModernBertModel::new(
            &cfg,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn has_causal_attention(&self, _: &str) -> Result<bool> {
        Ok(false)
    }
    fn is_classifier(&self, config: &str) -> Result<bool> {
        Ok(ModernBertConfig::from_json(config)?.is_classifier())
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(ModernBertConfig::from_json(config)?))
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        layer_indexed_device(
            ROOT_OR_NESTED_LAYER_INDEX_PATTERN,
            ModernBertConfig::from_json(config)?.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for ModernBertLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"layers\.(\d+)\.attn\.(Wqkv|Wo)\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.(Wi|Wo)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for ModernBertLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = ModernBertConfig::from_json(config)?;
        // token rows, the embedding norm and the final norm
        let mut elems = (cfg.vocab_size + 2) * cfg.hidden_size;
        if cfg.is_classifier() {
            elems += crate::classifier::head_elems(cfg.hidden_size, cfg.num_labels());
        }
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = ModernBertConfig::from_json(config)?;
        Ok(
            layer_shape(&cfg).layer_sizes_in_bytes(
                cfg.num_hidden_layers,
                dtype,
                weight_pack_factor,
            ),
        )
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = ModernBertConfig::from_json(config)?;
        Ok(Box::new(layer_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            None,
        )))
    }
}

fn layer_shape(cfg: &ModernBertConfig) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_attention_heads,
        head_dim: cfg.hidden_size / cfg.num_attention_heads,
        qkv_bias: cfg.attention_bias,
        o_bias: cfg.attention_bias,
        qk_norm: false,
        // two bias-free LayerNorms
        norms: 2,
        mlp: MlpShape::Gated {
            intermediate_size: cfg.intermediate_size,
        },
    }
}
