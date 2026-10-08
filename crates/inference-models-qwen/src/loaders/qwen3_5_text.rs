use super::*;

/// `NormalLoader` for the text backbone of a dense Qwen3.5 model.
pub struct Qwen3_5TextLoader;

/// `NormalLoader` for the text backbone of a Qwen3.5 MoE model.
pub struct Qwen3_5MoeTextLoader;

// The loader injects the MTP flag at the top level, which for this loader is the text config itself.
#[derive(serde::Deserialize)]
struct MtpFlag {
    #[serde(default, rename = "_inference_mtp")]
    mtp: bool,
}

fn mtp_requested(config: &str) -> Result<bool> {
    Ok(serde_json::from_str::<MtpFlag>(config)?.mtp)
}

fn parse_qwen35_text_config(config: &str, moe: bool) -> Result<crate::qwen3_5::TextConfig> {
    let cfg = crate::qwen3_5::TextConfig::from_json(config)?;
    cfg.check_experts(moe)?;
    cfg.validate()?;
    Ok(cfg)
}

macro_rules! qwen3_5_text_loader {
    ($loader:ident, $moe:expr) => {
        impl NormalModelLoader for $loader {
            fn runtime_config<'a>(
                &self,
                config: &'a str,
                max_model_len: Option<usize>,
            ) -> Result<Cow<'a, str>> {
                match max_model_len {
                    Some(max_model_len) => Ok(Cow::Owned(
                        crate::qwen3_5::config::apply_max_model_len(config, max_model_len)?,
                    )),
                    None => Ok(Cow::Borrowed(config)),
                }
            }

            fn load(
                &self,
                config: &str,
                vb: ShardedVarBuilder,
                normal_loading_metadata: NormalLoadingMetadata,
                attention_mechanism: AttentionImplementation,
            ) -> Result<Box<dyn NormalModel + Send + Sync>> {
                let cfg = parse_qwen35_text_config(config, $moe)?;
                Ok(Box::new(crate::qwen3_5::Qwen3_5TextModel::new(
                    &cfg,
                    vb,
                    cfg.tie_word_embeddings,
                    mtp_requested(config)?,
                    normal_loading_metadata,
                    attention_mechanism,
                )?))
            }
            fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
                let cfg = parse_qwen35_text_config(config, $moe)?;
                Ok(Box::new(cfg))
            }
            fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
                Ok(true)
            }
        }

        impl IsqModelLoader for $loader {
            fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
                isq_regexes(&[
                    r"^(model\.language_model|language_model\.model|model)\.embed_tokens\.weight$",
                    r"^lm_head\.(weight|bias)$",
                ])
            }

            fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
                super::qwen3_5::isq_layer_regexes($moe)
            }
            fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes(config)
            }
            fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
                super::qwen3_5::isq_layer_regexes_moqe($moe)
            }
            fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
                self.isq_layer_regexes_moqe(config)
            }
        }

        impl DeviceMappedModelLoader for $loader {
            fn mapped_max_act_size_elems(
                &self,
                config: &str,
                params: &AutoDeviceMapParams,
            ) -> Result<usize> {
                let AutoDeviceMapParams::Text {
                    max_seq_len,
                    max_batch_size,
                } = params
                else {
                    anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
                };
                let cfg = parse_qwen35_text_config(config, $moe)?;
                Ok(max_batch_size
                    * cfg.num_attention_heads
                    * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2))
            }

            fn non_mapped_size_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<usize> {
                let cfg = parse_qwen35_text_config(config, $moe)?;
                let (embed_tokens_pack_factor, lm_head_pack_factor) =
                    super::language_model_pack_factors_with_aliases(
                        quantization,
                        &[
                            "model.language_model.embed_tokens.weight",
                            "language_model.model.embed_tokens.weight",
                            "model.embed_tokens.weight",
                        ],
                        &["lm_head.weight"],
                        cfg.tie_word_embeddings,
                        dtype,
                        weight_pack_factor,
                    )?;
                let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
                let lm_head = if cfg.tie_word_embeddings {
                    0
                } else {
                    cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
                };
                let mtp_head = if mtp_requested(config)? {
                    super::qwen3_5::mtp_head_elems(&cfg, weight_pack_factor)?
                } else {
                    0
                };
                Ok((embed_tokens + lm_head + cfg.hidden_size + mtp_head) * dtype.size_in_bytes())
            }
            fn layer_sizes_in_bytes(
                &self,
                config: &str,
                dtype: DType,
                weight_pack_factor: usize,
                _matformer_config: Option<&MatformerSliceConfig>,
            ) -> Result<Vec<usize>> {
                let cfg = parse_qwen35_text_config(config, $moe)?;
                cfg.layer_types()
                    .into_iter()
                    .map(|layer_type| {
                        Ok(super::qwen3_5::decoder_layer_elems(
                            &cfg,
                            layer_type,
                            weight_pack_factor,
                        )? * dtype.size_in_bytes())
                    })
                    .collect()
            }
            fn num_layers(&self, config: &str) -> Result<usize> {
                let cfg = parse_qwen35_text_config(config, $moe)?;
                Ok(cfg.num_hidden_layers)
            }
            fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
                let cfg = parse_qwen35_text_config(config, $moe)?;
                let mtp = mtp_requested(config)?;
                let base = ModelConfigMetadata {
                    max_seq_len: cfg.max_position_embeddings,
                    num_layers: cfg.num_hidden_layers + cfg.mtp_layers(mtp),
                    hidden_size: cfg.hidden_size,
                    num_kv_heads: cfg.num_key_value_heads,
                    num_attn_heads: cfg.num_attention_heads,
                    sliding_window: None,
                    k_head_dim: cfg.head_dim,
                    v_head_dim: cfg.head_dim,
                    kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
                };
                Ok(Box::new(
                    HybridPagedKvCacheConfig::new(base, cfg.paged_kv_layers(mtp))
                        .with_uniform_prefix_prefill_attention_features(Default::default()),
                ))
            }
        }
    };
}

qwen3_5_text_loader!(Qwen3_5TextLoader, false);
qwen3_5_text_loader!(Qwen3_5MoeTextLoader, true);
