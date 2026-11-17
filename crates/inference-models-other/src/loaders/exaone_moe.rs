use super::*;

use crate::exaone_moe::Config;

/// `NormalLoader` for a K-EXAONE model.
pub struct ExaoneMoeLoader;

const EXPERT_REGEXES: [&str; 2] = [
    r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)(\.weight)?$",
];
const LAYER_REGEXES: [&str; 5] = [
    r"lm_head\.(weight|bias)$",
    r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
    r"layers\.(\d+)\.mlp\.(shared_experts\.)?(gate_proj|up_proj|down_proj)\.(weight|bias)$",
    r"^mtp\.layers\.(\d+)\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))\.(weight|bias)$",
    r"^mtp\.fc\.weight$",
];
const PROMOTED_REGEXES: [&str; 2] = [
    r"^model\.embed_tokens\.weight$",
    r"^lm_head\.(weight|bias)$",
];

fn shape(cfg: &Config, intermediate_size: usize) -> DecoderLayerShape {
    let text = &cfg.base;
    DecoderLayerShape {
        hidden_size: text.hidden_size,
        num_attention_heads: text.num_attention_heads,
        num_key_value_heads: text.num_key_value_heads,
        head_dim: text.head_dim(),
        qkv_bias: false,
        o_bias: false,
        qk_norm: true,
        norms: 2,
        mlp: MlpShape::Gated { intermediate_size },
    }
}

fn mtp_layers(cfg: &Config) -> usize {
    usize::from(cfg.mtp)
}

impl NormalModelLoader for ExaoneMoeLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = Config::from_json(config)?;
        Ok(Box::new(crate::exaone_moe::Model::new(
            &cfg.decoder_spec()?,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
    }
}

impl IsqModelLoader for ExaoneMoeLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&PROMOTED_REGEXES)
    }
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        let mut regexes = LAYER_REGEXES.to_vec();
        regexes.extend(EXPERT_REGEXES);
        isq_regexes(&regexes)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&EXPERT_REGEXES)
    }
}

impl DeviceMappedModelLoader for ExaoneMoeLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Config::from_json(config)?;
        let text = &cfg.base;
        let ends = standard_non_mapped_size_in_bytes(
            LanguageModelEnds {
                hidden_size: text.hidden_size,
                vocab_size: text.vocab_size,
                tie_word_embeddings: text.tie_word_embeddings,
            },
            quantization,
            dtype,
            weight_pack_factor,
        )?;
        // the MTP head beside the head: its fc over both halves, three norms and a dense layer
        let h = text.hidden_size;
        let mtp = shape(&cfg, text.intermediate_size).elems(weight_pack_factor)
            + 2 * h * h / weight_pack_factor
            + 3 * h;
        Ok(ends + mtp_layers(&cfg) * mtp * dtype.size_in_bytes())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let spec = cfg.decoder_spec()?;
        let dense_layers = spec
            .moe
            .as_ref()
            .map(|moe| moe.dense_layers.clone())
            .unwrap_or_default();
        let h = cfg.base.hidden_size;
        let gated = |intermediate_size: usize| 3 * h * intermediate_size / weight_pack_factor;
        let attention = shape(&cfg, 0).elems(weight_pack_factor) * dtype.size_in_bytes();
        Ok((0..cfg.base.num_hidden_layers)
            .map(|layer| {
                if dense_layers.contains(&layer) {
                    return attention + gated(cfg.base.intermediate_size) * dtype.size_in_bytes();
                }
                let experts = cfg.num_experts * gated(cfg.moe_intermediate_size)
                    + gated(cfg.moe_intermediate_size * cfg.num_shared_experts);
                // the router and its selection bias stay in f32
                let router = cfg.num_experts * (h + 1) * DType::F32.size_in_bytes();
                attention + experts * dtype.size_in_bytes() + router
            })
            .collect())
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Config::from_json(config)?;
        // the MTP head's attention is the paged layer after the stack's
        Ok(Box::new(shape(&cfg, 0).model_config(
            cfg.base.num_hidden_layers + mtp_layers(&cfg),
            cfg.base.max_position_embeddings,
            cfg.base.sliding_window,
        )))
    }
}
