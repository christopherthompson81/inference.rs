use super::*;

use inference_nn::utils::varbuilder_utils::DeviceForLoadTensor;

use crate::nemotron_h::{Config, LayerKind};

const EMBEDDINGS_WEIGHT: &str = "embeddings.weight";
// NeMo's speech models keep the decoder's ends under Llama's names
const NEMO_EMBEDDINGS_WEIGHT: &str = "embed_tokens.weight";
pub const NEMO_DECODER_ENDS: (&str, &str) = ("embed_tokens", "norm");
// (decoder, lm_head): Hub checkpoints keep NVIDIA's `backbone.*`, which transformers renames to `model.*`; NeMo's
// speech models nest the language model under `llm`
const ROOTS: [(&str, &str); 3] = [
    ("backbone", "lm_head"),
    ("model", "lm_head"),
    ("llm.model", "llm.lm_head"),
];

/// `NormalLoader` for Nemotron-H (`NemotronHForCausalLM`): its layers as Granite's hybrid blocks.
pub struct NemotronHLoader;

/// The prefixes the checkpoint's decoder and its `lm_head` sit under.
pub fn nemotron_h_prefixes(vb: &ShardedVarBuilder) -> (&'static str, &'static str) {
    ROOTS
        .into_iter()
        .find(|(root, _)| {
            [EMBEDDINGS_WEIGHT, NEMO_EMBEDDINGS_WEIGHT]
                .iter()
                .any(|weight| {
                    inference_nn::layers::contains_tensor_or_weight_source(&vb.pp(root), weight)
                })
        })
        .unwrap_or(ROOTS[1])
}

impl NormalModelLoader for NemotronHLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = Config::from_json(config)?;
        let (root, lm_head) = nemotron_h_prefixes(&vb);
        let mut layout = cfg.layout()?;
        if inference_nn::layers::contains_tensor_or_weight_source(
            &vb.pp(root),
            NEMO_EMBEDDINGS_WEIGHT,
        ) {
            (layout.embeddings, layout.final_norm) = NEMO_DECODER_ENDS;
        }
        let granite = cfg.granite()?;
        let loading_isq = normal_loading_metadata.loading_isq;
        let mut model = crate::granite::GraniteMoeHybrid::new_with_layout(
            &granite,
            layout,
            vb.pp(root),
            vb.pp(lm_head),
            true,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        if let Some(mtp) = cfg.mtp_layout()? {
            model = model.with_mtp(
                &granite,
                mtp,
                vb.pp(crate::granite::MTP_PREFIX),
                loading_isq,
                &attention_mechanism,
            )?;
        }
        Ok(Box::new(model))
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Config::from_json(config)?))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
    // a checkpoint layer goes where the block holding it does
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        if loading_isq {
            return Ok(Arc::new(|_| DeviceForLoadTensor::Base));
        }
        let blocks = Config::from_json(config)?.layout()?.block_of_layer();
        let re = Regex::new(LAYER_INDEX_PATTERN)?;
        Ok(Arc::new(move |name: String| {
            re.captures(&name)
                .and_then(|captures| captures[1].parse::<usize>().ok())
                .and_then(|layer| blocks.get(&layer).copied())
                .map(DeviceForLoadTensor::Idx)
                .unwrap_or(DeviceForLoadTensor::Base)
        }))
    }
}

impl IsqModelLoader for NemotronHLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"\.embeddings\.weight$", r"^lm_head\.(weight|bias)$"])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[
            r"lm_head\.(weight|bias)$",
            r"layers\.(\d+)\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mixer\.(in_proj|out_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mixer\.(up_proj|down_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mixer\.shared_experts\.(up_proj|down_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mixer\.(fc1_latent_proj|fc2_latent_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(weight|bias)$",
            r"layers\.(\d+)\.mixer\.experts\.(gate_and_up_projs|down_projs)\.weight$",
            r"mtp\.layers\.0\.eh_proj\.weight$",
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        isq_regexes(&[r"layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.(weight|bias)$"])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for NemotronHLoader {
    fn checkpoint_layer_groups(&self, config: &str) -> Result<Option<Vec<usize>>> {
        let blocks = Config::from_json(config)?.layout()?.block_of_layer();
        let layers = blocks.keys().max().map_or(0, |last| last + 1);
        Ok(Some(
            (0..layers)
                .map(|l| blocks.get(&l).copied().unwrap_or(0))
                .collect(),
        ))
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = Config::from_json(config)?;
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
    // the widest MoE layer's (a Puzzle checkpoint prunes them unevenly); a dense model's experts never leave the GPU
    fn prompt_activation_bytes_per_token(&self, config: &str) -> Result<Option<usize>> {
        let cfg = Config::from_json(config)?;
        Ok(cfg
            .layer_kinds()?
            .into_iter()
            .enumerate()
            .filter(|(_, kind)| *kind == LayerKind::Moe)
            .map(|(layer, _)| {
                let spec = cfg.moe_spec(layer);
                inference_nn::loaders::moe_prompt_activation_bytes_per_token(
                    spec.top_k,
                    spec.latent_size.unwrap_or(cfg.hidden_size),
                    spec.intermediate_size,
                    cfg.hidden_size,
                )
            })
            .max())
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = Config::from_json(config)?;
        let h = cfg.hidden_size;
        let layer_elems = |layer: usize, kind: LayerKind| match kind {
            LayerKind::Attention => {
                let q = cfg.num_attention_heads * cfg.head_dim;
                let kv = cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads) * cfg.head_dim;
                (h * q * 2 + h * kv * 2) / weight_pack_factor
            }
            LayerKind::Mamba => {
                let inner = cfg.mamba_num_heads * cfg.mamba_head_dim;
                let conv_dim = inner + 2 * cfg.n_groups * cfg.ssm_state_size;
                let in_proj = (inner + conv_dim + cfg.mamba_num_heads) * h;
                (in_proj + inner * h) / weight_pack_factor
                    + conv_dim * (cfg.conv_kernel + 1)
                    + cfg.mamba_num_heads * 3
                    + inner
            }
            LayerKind::Moe => {
                let spec = cfg.moe_spec(layer);
                let expert_in = spec.latent_size.unwrap_or(h);
                let experts = spec.num_experts * spec.intermediate_size * expert_in * 2;
                let latent = spec.latent_size.map_or(0, |latent| latent * h * 2);
                let shared = spec.shared_intermediate_size * h * 2;
                (experts + latent + shared) / weight_pack_factor + spec.num_experts * (h + 1)
            }
            LayerKind::Mlp => cfg.intermediate_size * h * 2 / weight_pack_factor,
        } + h;
        // a block holds its mixer layer and the feed-forward layer after it
        let mut sizes: Vec<usize> = Vec::new();
        for (layer, kind) in cfg.layer_kinds()?.into_iter().enumerate() {
            let bytes = layer_elems(layer, kind) * dtype.size_in_bytes();
            match kind {
                LayerKind::Mamba | LayerKind::Attention => sizes.push(bytes),
                LayerKind::Moe | LayerKind::Mlp => match sizes.last_mut() {
                    Some(last) => *last += bytes,
                    None => anyhow::bail!("Nemotron-H starts with a feed-forward layer"),
                },
            }
        }
        Ok(sizes)
    }
    // as the built model reports it: only attention layers page, and the Mamba layers' state sizes the recurrent pool
    // (the conv window counted in f32, which the model's dtype is not known to undercut)
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Config::from_json(config)?;
        let layout = cfg.layout()?;
        let mtp = cfg.mtp_layout()?.is_some();
        let base = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: layout.blocks.len() + usize::from(mtp),
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: inference_nn::paged_attention::KvCacheLayout::Standard,
        };
        let mamba_layers = layout
            .blocks
            .iter()
            .filter(|block| matches!(block.mixer, crate::granite::GraniteLayerType::Mamba))
            .count();
        let state =
            crate::granite::recurrent_state_bytes(&cfg.granite()?, mamba_layers, DType::F32);
        Ok(Box::new(crate::granite::hybrid_paged_kv_config(
            base, &layout, mtp, state,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mamba, attention+MoE, Mamba+MLP: three blocks of which only the attention one pages, and two Mamba layers whose
    // f32 state the recurrent pool holds per sequence
    #[test]
    fn the_planned_config_pages_attention_and_holds_mamba_state() -> Result<()> {
        let config = serde_json::json!({
            "vocab_size": 64,
            "hidden_size": 32,
            "layers_block_type": ["mamba", "attention", "moe", "mamba", "mlp"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 48,
            "mlp_hidden_act": "relu2",
            "ssm_state_size": 8,
            "mamba_num_heads": 4,
            "mamba_head_dim": 8,
            "n_groups": 2,
            "conv_kernel": 4,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "moe_shared_expert_intermediate_size": 40,
            "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 4096,
            "mamba_ssm_cache_dtype": "float32",
        })
        .to_string();
        let model = NemotronHLoader.model_config(&config)?;
        let paged = (0..model.num_layers())
            .map(|layer| model.layer_kv_cache_elements_per_token(layer).is_some())
            .collect::<Vec<_>>();
        assert_eq!(paged, [false, true, false]);
        // per Mamba layer: (4 * 8 + 2 * 2 * 8) conv channels by 4 taps, and 4 heads of 8 x 8 state, all f32
        assert_eq!(
            model.recurrent_state_bytes_per_sequence(),
            2 * ((32 + 32) * 4 + 4 * 8 * 8) * 4
        );
        Ok(())
    }
}
