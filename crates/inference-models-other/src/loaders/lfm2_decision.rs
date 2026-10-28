use super::*;
use crate::lfm2_decision::{Config as D1OmniConfig, Model as D1OmniModel};
use inference_nn::model::EmbeddingModel;

// the trunk's blocks only: the decision head's blocks stay on the non-mapped device
const TRUNK_LAYER_INDEX_PATTERN: &str = r"^blk\.(\d+)\.";

fn config(config: &str) -> Result<D1OmniConfig> {
    Ok(serde_json::from_str(config)?)
}

/// `EmbeddingModelLoader` for d1-omni (an LFM2 trunk read bidirectionally under laya's decision head).
pub struct Lfm2DecisionLoader;

impl EmbeddingModelLoader for Lfm2DecisionLoader {
    fn load(
        &self,
        config_json: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        Ok(Box::new(D1OmniModel::new(
            &config(config_json)?,
            vb,
            normal_loading_metadata,
        )?))
    }
    fn has_causal_attention(&self, _: &str) -> Result<bool> {
        Ok(false)
    }
    // scores per option, with no sentence-transformers pooling after it
    fn is_classifier(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config_json: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(config(config_json)?))
    }
    fn get_device_for_tensor(
        &self,
        config_json: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        layer_indexed_device(
            TRUNK_LAYER_INDEX_PATTERN,
            config(config_json)?.trunk_layers(),
            loading_isq,
        )
    }
}

impl IsqModelLoader for Lfm2DecisionLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(Vec::new())
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|attn_qkv|ffn_gate|ffn_up|ffn_down)\.(weight|bias)$",
            r"^blk\.(\d+)\.shortconv\.(in_proj|out_proj)\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

/// One block's elements: its mixer (short convolution or attention), its two norms and its gated MLP.
fn trunk_layer_elems(cfg: &D1OmniConfig, layer: usize, weight_pack_factor: usize) -> usize {
    let width = cfg.hidden_size;
    let head_dim = width / cfg.num_attention_heads;
    let mixer = match cfg.num_key_value_heads[layer] {
        0 => (4 * width * width) / weight_pack_factor + 3 * width,
        kv => (2 * width * width + 2 * width * kv * head_dim) / weight_pack_factor + 2 * head_dim,
    };
    mixer + 3 * width * cfg.intermediate_sizes[layer] / weight_pack_factor + 2 * width
}

impl DeviceMappedModelLoader for Lfm2DecisionLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config_json: &str,
        dtype: DType,
        _weight_pack_factor: usize,
        _quantization: Option<&AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = config(config_json)?;
        let width = cfg.hidden_size;
        // each head block: fused qkv, output and the MLP with their biases, and two LayerNorms
        let head = (cfg.trunk_layers()..cfg.num_key_value_heads.len())
            .map(|i| 4 * width * width + 2 * width * cfg.intermediate_sizes[i] + 9 * width)
            .sum::<usize>();
        // token rows, the final norm, the type embedding and the scorer
        let rest = (cfg.vocab_size + 1 + 3) * width + 3 * width + width * width + width + 1;
        Ok((head + rest) * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config_json: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = config(config_json)?;
        Ok((0..cfg.trunk_layers())
            .map(|i| trunk_layer_elems(&cfg, i, weight_pack_factor) * dtype.size_in_bytes())
            .collect())
    }
    fn model_config(&self, config_json: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = config(config_json)?;
        let kv_heads = cfg.num_key_value_heads.iter().copied().max().unwrap_or(1);
        let shape = DecoderLayerShape {
            hidden_size: cfg.hidden_size,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: kv_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            qkv_bias: false,
            o_bias: false,
            qk_norm: true,
            norms: 2,
            mlp: MlpShape::Gated {
                intermediate_size: cfg.intermediate_sizes.first().copied().unwrap_or_default(),
            },
        };
        Ok(Box::new(shape.model_config(
            cfg.trunk_layers(),
            cfg.max_position_embeddings,
            None,
        )))
    }
}
