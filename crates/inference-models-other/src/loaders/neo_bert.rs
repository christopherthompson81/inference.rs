use super::*;
use crate::neo_bert::{Config as NeoBertConfig, Model as NeoBertModel};
use inference_nn::model::EmbeddingModel;

// NeoBERT numbers its layers `transformer_encoder.N`
const NEO_BERT_LAYER_INDEX_PATTERN: &str = r"(?:^|\.)transformer_encoder\.(\d+)\.";

/// `EmbeddingModelLoader` for NeoBERT encoders.
pub struct NeoBertLoader;

impl EmbeddingModelLoader for NeoBertLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let cfg = NeoBertConfig::from_json(config)?;
        Ok(Box::new(NeoBertModel::new(
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
        Ok(NeoBertConfig::from_json(config)?.is_classifier())
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(NeoBertConfig::from_json(config)?))
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        layer_indexed_device(
            NEO_BERT_LAYER_INDEX_PATTERN,
            NeoBertConfig::from_json(config)?.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for NeoBertLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"transformer_encoder\.(\d+)\.(qkv|wo)\.(weight|bias)$",
            r"transformer_encoder\.(\d+)\.ffn\.(w12|w3)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for NeoBertLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = NeoBertConfig::from_json(config)?;
        // the token rows and the final norm
        let mut elems = (cfg.vocab_size + 1) * cfg.hidden_size;
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
        let cfg = NeoBertConfig::from_json(config)?;
        Ok(
            layer_shape(&cfg).layer_sizes_in_bytes(
                cfg.num_hidden_layers,
                dtype,
                weight_pack_factor,
            ),
        )
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = NeoBertConfig::from_json(config)?;
        Ok(Box::new(layer_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_length,
            None,
        )))
    }
}

fn layer_shape(cfg: &NeoBertConfig) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_attention_heads,
        head_dim: cfg.hidden_size / cfg.num_attention_heads,
        qkv_bias: false,
        o_bias: false,
        qk_norm: false,
        norms: 2,
        mlp: MlpShape::Gated {
            intermediate_size: cfg.ffn_size(),
        },
    }
}
