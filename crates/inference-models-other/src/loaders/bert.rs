use super::*;
use crate::bert::{Config as BertConfig, Model as BertModel, ORIGINAL_FFN};
use inference_nn::model::EmbeddingModel;

// BERT numbers its layers `encoder.layer.N`, not `layers.N`
const BERT_LAYER_INDEX_PATTERN: &str = r"\.layer\.(\d+)\.";

/// `EmbeddingModelLoader` for BERT-family encoders (BERT, RoBERTa, XLM-R, CamemBERT).
pub struct BertLoader;

impl EmbeddingModelLoader for BertLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let cfg = BertConfig::from_json(config)?;
        Ok(Box::new(BertModel::new(
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
        Ok(BertConfig::from_json(config)?.is_classifier())
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(BertConfig::from_json(config)?))
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        layer_indexed_device(
            BERT_LAYER_INDEX_PATTERN,
            BertConfig::from_json(config)?.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for BertLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"layer\.(\d+)\.attention\.self\.(query|key|value)\.(weight|bias)$",
            r"layer\.(\d+)\.attention\.output\.dense\.(weight|bias)$",
            r"layer\.(\d+)\.intermediate\.dense\.(weight|bias)$",
            r"layer\.(\d+)\.output\.dense\.(weight|bias)$",
            r"layer\.(\d+)\.mlp\.(gated_layers|wo)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for BertLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = BertConfig::from_json(config)?;
        let (hidden, labels) = (cfg.hidden_size, cfg.num_labels());
        let rows = cfg.vocab_size + cfg.max_position_embeddings + cfg.type_vocab_size;
        // the embedding LayerNorm's weight and bias
        let mut elems = rows * hidden + 2 * hidden;
        if cfg.is_classifier() {
            // a hidden-to-hidden dense layer and the label projection, both biased
            elems += hidden * hidden + hidden + hidden * labels + labels;
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
        let cfg = BertConfig::from_json(config)?;
        Ok(
            layer_shape(&cfg).layer_sizes_in_bytes(
                cfg.num_hidden_layers,
                dtype,
                weight_pack_factor,
            ),
        )
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = BertConfig::from_json(config)?;
        Ok(Box::new(layer_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_seq_len(),
            None,
        )))
    }
}

fn layer_shape(cfg: &BertConfig) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_attention_heads,
        head_dim: cfg.head_dim(),
        qkv_bias: true,
        o_bias: true,
        qk_norm: false,
        // two LayerNorms, each with a bias
        norms: 4,
        // a GLU FFN's fused gate-and-up projection holds two intermediate-wide halves
        mlp: if cfg.feed_forward_type == ORIGINAL_FFN {
            MlpShape::Plain {
                intermediate_size: cfg.intermediate_size,
                bias: true,
            }
        } else {
            MlpShape::Gated {
                intermediate_size: cfg.intermediate_size,
            }
        },
    }
}
