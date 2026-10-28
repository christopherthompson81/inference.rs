use super::*;
use crate::modern_bert::Config as ModernBertConfig;
use crate::modern_bert_decision::Model as DecisionModel;
use inference_nn::decision::DecisionSpec;
use inference_nn::model::EmbeddingModel;

// the encoder's layers only: the decision head's `head.layers.N` stay on the non-mapped device
const ENCODER_LAYER_INDEX_PATTERN: &str = r"^encoder\.layers\.(\d+)\.";

fn decision_spec(config: &str) -> Result<DecisionSpec> {
    DecisionSpec::from_config(config)?
        .ok_or_else(|| anyhow::anyhow!("a decision model's config has no `decision` section"))
}

/// `EmbeddingModelLoader` for ModernBERT decision models (laya, Julia-1).
pub struct ModernBertDecisionLoader;

impl EmbeddingModelLoader for ModernBertDecisionLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        Ok(Box::new(DecisionModel::new(
            &ModernBertConfig::from_json(config)?,
            &decision_spec(config)?,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn has_causal_attention(&self, _: &str) -> Result<bool> {
        Ok(false)
    }
    // scores per option, with no sentence-transformers pooling after it
    fn is_classifier(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn decision_spec(&self, config: &str) -> Result<Option<DecisionSpec>> {
        decision_spec(config).map(Some)
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
            ENCODER_LAYER_INDEX_PATTERN,
            ModernBertConfig::from_json(config)?.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for ModernBertDecisionLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^encoder\.layers\.(\d+)\.attn\.(Wqkv|Wo)\.(weight|bias)$",
            r"^encoder\.layers\.(\d+)\.mlp\.(Wi|Wo)\.(weight|bias)$",
            r"^head\.layers\.(\d+)\.(linear1|linear2|self_attn\.out_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for ModernBertDecisionLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&AutoDeviceMapQuantization<'_>>,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let width = ModernBertConfig::from_json(config)?.hidden_size;
        // each head block: in_proj, out_proj and the 4x MLP with their biases, and two LayerNorms
        let block = 3 * width * width + width * width + 8 * width * width + 13 * width;
        // the type embedding, the scorer's LayerNorm and its two linears
        let rest = 3 * width + 2 * width + width * width + width + width + 1;
        let head = decision_spec(config)?.head_layers * block + rest;
        Ok(ModernBertLoader.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            quantization,
            matformer_config,
        )? + head * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        ModernBertLoader.layer_sizes_in_bytes(config, dtype, weight_pack_factor, matformer_config)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        ModernBertLoader.model_config(config)
    }
}
