use super::*;
use crate::llama_embedding::{Config as LlamaEmbeddingConfig, Model as LlamaEmbeddingModel};
use inference_nn::device_map::DeviceMapper;
use inference_nn::model::EmbeddingModel;
use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;

/// `EmbeddingModelLoader` for bidirectional Llama encoders (EuroBERT, Llama bidirectional embedders).
pub struct LlamaEmbeddingLoader;

impl EmbeddingModelLoader for LlamaEmbeddingLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let cfg = LlamaEmbeddingConfig::from_json(config)?;
        Ok(Box::new(LlamaEmbeddingModel::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn has_causal_attention(&self, _: &str) -> Result<bool> {
        Ok(false)
    }
    fn embedding_tasks(
        &self,
        config: &str,
    ) -> Result<Option<inference_nn::embedding::EmbeddingTasks>> {
        Ok(inference_nn::embedding::jina_v5_tasks(config)?)
    }
    fn pools_last_token(&self, config: &str) -> Result<bool> {
        Ok(inference_nn::embedding::jina_v5_tasks(config)?.is_some())
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(LlamaEmbeddingConfig::from_json(config)?))
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        layer_indexed_device(
            ROOT_OR_NESTED_LAYER_INDEX_PATTERN,
            LlamaEmbeddingConfig::from_json(config)?.num_hidden_layers,
            loading_isq,
        )
    }
}

impl IsqModelLoader for LlamaEmbeddingLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for LlamaEmbeddingLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = LlamaEmbeddingConfig::from_json(config)?;
        // the token rows and the final norm; an `lm_head` in the checkpoint is not loaded
        let elems = (cfg.vocab_size + 1) * cfg.hidden_size;
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = LlamaEmbeddingConfig::from_json(config)?;
        Ok(super::llama::decoder_shape(&cfg).layer_sizes_in_bytes(
            cfg.num_hidden_layers,
            dtype,
            weight_pack_factor,
        ))
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = LlamaEmbeddingConfig::from_json(config)?;
        Ok(Box::new(super::llama::decoder_shape(&cfg).model_config(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
            None,
        )))
    }
}
