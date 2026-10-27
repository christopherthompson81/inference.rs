use super::*;
use crate::nomic_bert::{Config as NomicBertConfig, Model as NomicBertModel};
use inference_nn::model::EmbeddingModel;

/// `EmbeddingModelLoader` for Nomic BERT encoders (nomic-embed-text).
pub struct NomicBertLoader;

impl EmbeddingModelLoader for NomicBertLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let cfg = NomicBertConfig::from_json(config)?;
        Ok(Box::new(NomicBertModel::new(
            &cfg,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn has_causal_attention(&self, _: &str) -> Result<bool> {
        Ok(false)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(NomicBertConfig::from_json(config)?))
    }
}

impl IsqModelLoader for NomicBertLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"layers\.(\d+)\.attn\.(Wqkv|out_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.(fc11|fc12|fc2)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for NomicBertLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = NomicBertConfig::from_json(config)?;
        // word and token type rows, and the embedding LayerNorm's weight and bias
        let elems = (cfg.vocab_size + cfg.type_vocab_size + 2) * cfg.hidden_size();
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = NomicBertConfig::from_json(config)?;
        Ok(layer_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers(),
            dtype,
            weight_pack_factor,
        ))
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = NomicBertConfig::from_json(config)?;
        Ok(Box::new(layer_shape(&cfg).model_config(
            cfg.num_hidden_layers(),
            cfg.max_positions,
            None,
        )))
    }
}

fn layer_shape(cfg: &NomicBertConfig) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size(),
        num_attention_heads: cfg.num_attention_heads(),
        num_key_value_heads: cfg.num_attention_heads(),
        head_dim: cfg.hidden_size() / cfg.num_attention_heads(),
        qkv_bias: cfg.qkv_proj_bias,
        o_bias: cfg.qkv_proj_bias,
        qk_norm: false,
        // two LayerNorms, each with a bias
        norms: 4,
        mlp: MlpShape::Gated {
            intermediate_size: cfg.intermediate_size(),
        },
    }
}
