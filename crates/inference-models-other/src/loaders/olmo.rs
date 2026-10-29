use super::*;

use crate::olmo::{Common, Olmo2Config, OlmoConfig, OlmoeConfig};

/// `NormalLoader` for OLMo 1.
pub struct OlmoLoader;
/// `NormalLoader` for OLMo 2 and Olmo 3.
pub struct Olmo2Loader;
/// `NormalLoader` for OLMoE.
pub struct OlmoeLoader;

/// What the three OLMo loaders read from a config.
struct Olmo {
    common: Common,
    spec: inference_nn::decoder::DecoderSpec,
    /// (experts, per-expert intermediate size)
    experts: Option<(usize, usize)>,
}

fn olmo(config: &str) -> Result<Olmo> {
    let cfg = OlmoConfig::from_json(config)?;
    Ok(Olmo {
        spec: cfg.decoder_spec(),
        common: cfg.common,
        experts: None,
    })
}

fn olmo2(config: &str) -> Result<Olmo> {
    let cfg = Olmo2Config::from_json(config)?;
    Ok(Olmo {
        spec: cfg.decoder_spec()?,
        common: cfg.common,
        experts: None,
    })
}

fn olmoe(config: &str) -> Result<Olmo> {
    let cfg = OlmoeConfig::from_json(config)?;
    Ok(Olmo {
        spec: cfg.decoder_spec(),
        experts: Some((cfg.num_experts, cfg.common.intermediate_size)),
        common: cfg.common,
    })
}

impl Olmo {
    fn shape(&self) -> DecoderLayerShape {
        let c = &self.common;
        DecoderLayerShape {
            hidden_size: c.hidden_size,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: self.spec.num_kv_heads,
            head_dim: self.spec.head_dim,
            qkv_bias: c.attention_bias,
            o_bias: c.attention_bias,
            qk_norm: self.spec.qk_norm.is_some(),
            norms: 2,
            mlp: MlpShape::Gated {
                intermediate_size: if self.experts.is_some() {
                    0
                } else {
                    c.intermediate_size
                },
            },
        }
    }

    fn layer_sizes(&self, dtype: DType, weight_pack_factor: usize) -> Vec<usize> {
        let c = &self.common;
        let experts = self.experts.map_or(0, |(n, intermediate)| {
            n * c.hidden_size + n * 3 * c.hidden_size * intermediate / weight_pack_factor
        });
        let per_layer = self.shape().elems(weight_pack_factor) + experts;
        vec![per_layer * dtype.size_in_bytes(); c.num_hidden_layers]
    }

    fn sliding_window(&self) -> Option<usize> {
        self.spec.layer_windows.iter().flatten().max().copied()
    }
}

macro_rules! olmo_loader {
    ($loader:ident, $read:ident, $moe_regexes:expr) => {
        impl NormalModelLoader for $loader {
            fn load(
                &self,
                config: &str,
                vb: ShardedVarBuilder,
                normal_loading_metadata: NormalLoadingMetadata,
                attention_mechanism: AttentionImplementation,
            ) -> Result<Box<dyn NormalModel + Send + Sync>> {
                let olmo = $read(config)?;
                Ok(Box::new(crate::olmo::Model::new(
                    &olmo.spec,
                    vb,
                    self.is_gptx_for(config, &normal_loading_metadata)?,
                    normal_loading_metadata,
                    attention_mechanism,
                )?))
            }
            fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
                Ok(Box::new($read(config)?.common))
            }
        }

        impl IsqModelLoader for $loader {
            fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
                isq_regexes(&[
                    r"^model\.embed_tokens\.weight$",
                    r"^lm_head\.(weight|bias)$",
                ])
            }
            fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
                let mut regexes = vec![
                    r"lm_head\.(weight|bias)$",
                    r"layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
                    r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
                ];
                regexes.extend($moe_regexes);
                isq_regexes(&regexes)
            }
            fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes(config)
            }
            fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
                isq_regexes(&$moe_regexes)
            }
            fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes_moqe(config)
            }
        }

        impl DeviceMappedModelLoader for $loader {
            fn non_mapped_size_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<usize> {
                let c = $read(config)?.common;
                standard_non_mapped_size_in_bytes(
                    LanguageModelEnds {
                        hidden_size: c.hidden_size,
                        vocab_size: c.vocab_size,
                        tie_word_embeddings: c.tie_word_embeddings,
                    },
                    quantization,
                    dtype,
                    weight_pack_factor,
                )
            }
            fn layer_sizes_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<Vec<usize>> {
                Ok($read(config)?.layer_sizes(dtype, weight_pack_factor))
            }
            fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
                let olmo = $read(config)?;
                Ok(Box::new(olmo.shape().model_config(
                    olmo.common.num_hidden_layers,
                    olmo.common.max_position_embeddings,
                    olmo.sliding_window(),
                )))
            }
        }
    };
}

const NO_EXPERTS: [&str; 0] = [];
const OLMOE_EXPERTS: [&str; 1] =
    [r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$"];

olmo_loader!(OlmoLoader, olmo, NO_EXPERTS);
olmo_loader!(Olmo2Loader, olmo2, NO_EXPERTS);
olmo_loader!(OlmoeLoader, olmoe, OLMOE_EXPERTS);
