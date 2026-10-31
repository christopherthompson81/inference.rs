use std::borrow::Cow;

use super::*;

/// `NormalLoader` for a dense ERNIE 4.5 model.
pub struct Ernie4_5Loader;

/// `NormalLoader` for an ERNIE 4.5 MoE model.
pub struct Ernie4_5MoeLoader;

const MAX_POSITION_EMBEDDINGS: &str = "max_position_embeddings";
const MOE_EXPERT_REGEXES: [&str; 4] = [
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)(\.weight)?$",
];
const LAYER_REGEXES: [&str; 8] = [
    r"lm_head\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.(shared_experts\.)?gate_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.(shared_experts\.)?up_proj\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.(shared_experts\.)?down_proj\.(weight|bias)$",
];
const PROMOTED_REGEXES: [&str; 2] = [
    r"^model\.embed_tokens\.weight$",
    r"^lm_head\.(weight|bias)$",
];

fn decoder_shape(cfg: &crate::ernie4_5::Config, intermediate_size: usize) -> DecoderLayerShape {
    DecoderLayerShape {
        hidden_size: cfg.hidden_size,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        qkv_bias: false,
        o_bias: false,
        qk_norm: false,
        norms: 2,
        mlp: MlpShape::Gated { intermediate_size },
    }
}

fn non_mapped_size(
    cfg: &crate::ernie4_5::Config,
    dtype: DType,
    weight_pack_factor: usize,
    quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
) -> Result<usize> {
    standard_non_mapped_size_in_bytes(
        LanguageModelEnds {
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            tie_word_embeddings: cfg.tie_word_embeddings,
        },
        quantization,
        dtype,
        weight_pack_factor,
    )
}

impl NormalModelLoader for Ernie4_5Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::ernie4_5::Config::from_json(config)?;
        Ok(Box::new(crate::ernie4_5::Model::new(
            &cfg.decoder_spec()?,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        cap_context_length(config, MAX_POSITION_EMBEDDINGS, max_model_len)
    }
    fn is_gptx(&self, _config: &str) -> Result<bool> {
        Ok(false)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(crate::ernie4_5::Config::from_json(config)?))
    }
}

impl IsqModelLoader for Ernie4_5Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&PROMOTED_REGEXES)
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&LAYER_REGEXES)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Ernie4_5Loader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::ernie4_5::Config::from_json(config)?;
        non_mapped_size(&cfg, dtype, weight_pack_factor, quantization)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = crate::ernie4_5::Config::from_json(config)?;
        Ok(
            decoder_shape(&cfg, cfg.intermediate_size).layer_sizes_in_bytes(
                cfg.num_hidden_layers,
                dtype,
                weight_pack_factor,
            ),
        )
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::ernie4_5::Config::from_json(config)?;
        Ok(Box::new(
            decoder_shape(&cfg, cfg.intermediate_size).model_config(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
                None,
            ),
        ))
    }
}

impl NormalModelLoader for Ernie4_5MoeLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = crate::ernie4_5::MoeConfig::from_json(config)?;
        Ok(Box::new(crate::ernie4_5::Model::new(
            &cfg.decoder_spec()?,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        cap_context_length(config, MAX_POSITION_EMBEDDINGS, max_model_len)
    }
    fn is_gptx(&self, _config: &str) -> Result<bool> {
        Ok(false)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(crate::ernie4_5::MoeConfig::from_json(config)?))
    }
}

impl IsqModelLoader for Ernie4_5MoeLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&PROMOTED_REGEXES)
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        let mut regexes = LAYER_REGEXES.to_vec();
        regexes.extend(MOE_EXPERT_REGEXES);
        isq_regexes(&regexes)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&MOE_EXPERT_REGEXES)
    }
}

impl DeviceMappedModelLoader for Ernie4_5MoeLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = crate::ernie4_5::MoeConfig::from_json(config)?;
        non_mapped_size(&cfg.base, dtype, weight_pack_factor, quantization)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = crate::ernie4_5::MoeConfig::from_json(config)?;
        let h = cfg.base.hidden_size;
        let gated = |intermediate_size: usize| 3 * h * intermediate_size / weight_pack_factor;
        let attention =
            decoder_shape(&cfg.base, 0).elems(weight_pack_factor) * dtype.size_in_bytes();
        Ok((0..cfg.base.num_hidden_layers)
            .map(|layer_idx| {
                if !cfg.is_moe_layer(layer_idx) {
                    return attention + gated(cfg.base.intermediate_size) * dtype.size_in_bytes();
                }
                let experts = cfg.moe_num_experts * gated(cfg.moe_intermediate_size)
                    + gated(cfg.moe_intermediate_size * cfg.moe_num_shared_experts);
                // the router and its selection bias stay in f32
                let router = (cfg.moe_num_experts * (h + 1)) * DType::F32.size_in_bytes();
                attention + experts * dtype.size_in_bytes() + router
            })
            .collect())
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = crate::ernie4_5::MoeConfig::from_json(config)?;
        Ok(Box::new(decoder_shape(&cfg.base, 0).model_config(
            cfg.base.num_hidden_layers,
            cfg.base.max_position_embeddings,
            None,
        )))
    }
}
