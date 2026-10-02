use anyhow::Result;
use candle_core::DType;
use inference_quant::{IsqType, QuantizedWeightSource};

use crate::topology::Topology;

#[macro_export]
macro_rules! bias_if {
    ($cond:expr, $size:expr) => {
        if $cond { $size } else { 0 }
    };
}

#[derive(Clone, Copy)]
pub struct AutoDeviceMapQuantization<'a> {
    source: AutoDeviceMapQuantizationSource<'a>,
    topology: Option<&'a Topology>,
}

#[derive(Clone, Copy)]
enum AutoDeviceMapQuantizationSource<'a> {
    Isq(Option<IsqType>),
    WeightSource(&'a dyn QuantizedWeightSource),
}

impl<'a> AutoDeviceMapQuantization<'a> {
    pub fn isq(isq: Option<IsqType>, topology: Option<&'a Topology>) -> Self {
        Self {
            source: AutoDeviceMapQuantizationSource::Isq(isq),
            topology,
        }
    }

    pub fn weight_source(source: &'a dyn QuantizedWeightSource) -> Self {
        Self {
            source: AutoDeviceMapQuantizationSource::WeightSource(source),
            topology: None,
        }
    }

    pub fn weight_source_with_topology(
        source: &'a dyn QuantizedWeightSource,
        topology: Option<&'a Topology>,
    ) -> Self {
        Self {
            source: AutoDeviceMapQuantizationSource::WeightSource(source),
            topology,
        }
    }

    #[cfg(test)]
    fn unpromoted_pack_factor_for(
        &self,
        name: &str,
        dtype: DType,
        fallback: usize,
    ) -> Result<usize> {
        self.pack_factor_for_candidates(&[name], dtype, fallback, false)
    }

    pub fn promoted_pack_factor_for(
        &self,
        name: &str,
        dtype: DType,
        fallback: usize,
    ) -> Result<usize> {
        self.pack_factor_for_candidates(&[name], dtype, fallback, true)
    }

    pub fn conservative_pack_factor(&self, dtype: DType, fallback: usize) -> usize {
        let topology_pack_factors = self.topology.into_iter().flat_map(|topology| {
            topology
                .layers
                .iter()
                .filter_map(|entry| entry.as_ref().and_then(|entry| entry.isq))
                .chain(topology.patterns.iter().filter_map(|(_, entry)| entry.isq))
        });
        topology_pack_factors.fold(fallback, |factor, ty| factor.min(ty.pack_factor(dtype)))
    }

    pub fn conservative_moqe_pack_factor(
        &self,
        dtype: DType,
        source_pack_factor: usize,
        target: IsqType,
    ) -> usize {
        self.conservative_pack_factor(dtype, source_pack_factor.min(target.pack_factor(dtype)))
    }

    fn pack_factor_for_candidates(
        &self,
        names: &[&str],
        dtype: DType,
        fallback: usize,
        promote_default: bool,
    ) -> Result<usize> {
        let topology_ty = names.iter().find_map(|name| {
            self.topology
                .and_then(|topology| topology.match_for_name(name))
                .and_then(|topology| topology.isq)
        });
        match self.source {
            AutoDeviceMapQuantizationSource::WeightSource(source) => {
                if let Some(ty) = topology_ty {
                    return Ok(ty.pack_factor(dtype));
                }
                for name in names {
                    if let Some(pack_factor) = source.pack_factor_for(name, dtype)? {
                        return Ok(pack_factor);
                    }
                }
                Ok(1)
            }
            AutoDeviceMapQuantizationSource::Isq(default) => {
                let ty = topology_ty.or_else(|| {
                    default.map(|ty| {
                        if promote_default {
                            ty.promote_for_sensitive_tensor()
                        } else {
                            ty
                        }
                    })
                });
                Ok(ty.map(|ty| ty.pack_factor(dtype)).unwrap_or(fallback))
            }
        }
    }
}

pub fn promoted_tensor_pack_factor(
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    name: &str,
    dtype: DType,
    fallback: usize,
) -> Result<usize> {
    quantization.map_or(Ok(fallback), |quantization| {
        quantization.promoted_pack_factor_for(name, dtype, fallback)
    })
}

pub fn tied_promoted_tensor_pack_factor(
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    embedding_name: &str,
    legacy_head_name: &str,
    dtype: DType,
    fallback: usize,
) -> Result<usize> {
    quantization.map_or(Ok(fallback), |quantization| match quantization.source {
        AutoDeviceMapQuantizationSource::WeightSource(_) => quantization
            .pack_factor_for_candidates(&[embedding_name, legacy_head_name], dtype, fallback, true),
        AutoDeviceMapQuantizationSource::Isq(_) => {
            quantization.promoted_pack_factor_for(embedding_name, dtype, fallback)
        }
    })
}

pub struct LanguageModelEnds {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
}

/// Embeddings, untied LM head and final norm, the non-mapped weights of a plain decoder.
pub fn standard_non_mapped_size_in_bytes(
    ends: LanguageModelEnds,
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    dtype: DType,
    weight_pack_factor: usize,
) -> Result<usize> {
    let LanguageModelEnds {
        hidden_size,
        vocab_size,
        tie_word_embeddings,
    } = ends;
    let (embed_tokens_pack_factor, lm_head_pack_factor) = language_model_pack_factors(
        quantization,
        "model.embed_tokens.weight",
        "lm_head.weight",
        tie_word_embeddings,
        dtype,
        weight_pack_factor,
    )?;
    let embed_tokens = hidden_size * vocab_size / embed_tokens_pack_factor;
    let lm_head = if tie_word_embeddings {
        0
    } else {
        hidden_size * vocab_size / lm_head_pack_factor
    };
    Ok((embed_tokens + lm_head + hidden_size) * dtype.size_in_bytes())
}

pub fn language_model_pack_factors(
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    embedding_name: &str,
    head_name: &str,
    tied: bool,
    dtype: DType,
    fallback: usize,
) -> Result<(usize, usize)> {
    let embedding = if tied {
        tied_promoted_tensor_pack_factor(quantization, embedding_name, head_name, dtype, fallback)?
    } else {
        promoted_tensor_pack_factor(quantization, embedding_name, dtype, fallback)?
    };
    let head = promoted_tensor_pack_factor(quantization, head_name, dtype, fallback)?;
    Ok((embedding, head))
}

pub fn language_model_pack_factors_with_aliases(
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    embedding_names: &[&str],
    head_names: &[&str],
    tied: bool,
    dtype: DType,
    fallback: usize,
) -> Result<(usize, usize)> {
    let embedding = quantization.map_or(Ok(fallback), |quantization| {
        if tied
            && matches!(
                quantization.source,
                AutoDeviceMapQuantizationSource::WeightSource(_)
            )
        {
            let mut candidates = embedding_names.to_vec();
            candidates.extend_from_slice(head_names);
            quantization.pack_factor_for_candidates(&candidates, dtype, fallback, true)
        } else {
            quantization.pack_factor_for_candidates(embedding_names, dtype, fallback, true)
        }
    })?;
    let head = quantization.map_or(Ok(fallback), |quantization| {
        quantization.pack_factor_for_candidates(head_names, dtype, fallback, true)
    })?;
    Ok((embedding, head))
}

/// A decoder layer's MLP, as the device map sizes it.
#[derive(Clone, Copy, Debug)]
pub enum MlpShape {
    /// gate, up and down projections (SwiGLU and friends), stored split or as one `gate_up_proj`.
    Gated { intermediate_size: usize },
    /// an up and a down projection (`c_fc`/`c_proj`, `fc1`/`fc2`).
    Plain {
        intermediate_size: usize,
        bias: bool,
    },
}

/// One standard decoder layer: hidden-size norms, q/k/v/o attention and an MLP, with the model's own head dim.
#[derive(Clone, Copy, Debug)]
pub struct DecoderLayerShape {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub qkv_bias: bool,
    pub o_bias: bool,
    pub qk_norm: bool,
    /// Hidden-size norm vectors per layer: 2 for pre-attention and pre-MLP, 4 with post-norms or LayerNorm biases.
    pub norms: usize,
    pub mlp: MlpShape,
}

impl DecoderLayerShape {
    /// Elements of one layer; matrices shrink by `weight_pack_factor`, biases and norms do not.
    pub fn elems(&self, weight_pack_factor: usize) -> usize {
        let h = self.hidden_size;
        let q = self.num_attention_heads * self.head_dim;
        let kv = self.num_key_value_heads * self.head_dim;
        // each matrix packs on its own, as the layers are quantized one by one
        let packed = |rows: usize, cols: usize| rows * cols / weight_pack_factor;
        let attention = packed(h, q)
            + 2 * packed(h, kv)
            + packed(q, h)
            + bias_if!(self.qkv_bias, q + 2 * kv)
            + bias_if!(self.o_bias, h)
            + bias_if!(self.qk_norm, 2 * self.head_dim);
        let mlp = match self.mlp {
            MlpShape::Gated { intermediate_size } => 3 * packed(h, intermediate_size),
            MlpShape::Plain {
                intermediate_size,
                bias,
            } => 2 * packed(h, intermediate_size) + bias_if!(bias, intermediate_size + h),
        };
        self.norms * h + attention + mlp
    }

    pub fn layer_sizes_in_bytes(
        &self,
        num_layers: usize,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Vec<usize> {
        vec![self.elems(weight_pack_factor) * dtype.size_in_bytes(); num_layers]
    }

    /// The paged-KV metadata of a model made of these layers, so sizing and KV planning share one head dim.
    pub fn model_config(
        &self,
        num_layers: usize,
        max_seq_len: usize,
        sliding_window: Option<usize>,
    ) -> crate::paged_attention::ModelConfigMetadata {
        crate::paged_attention::ModelConfigMetadata {
            max_seq_len,
            num_layers,
            hidden_size: self.hidden_size,
            num_kv_heads: self.num_key_value_heads,
            num_attn_heads: self.num_attention_heads,
            sliding_window,
            k_head_dim: self.head_dim,
            v_head_dim: self.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use inference_quant::QuantizedWeightSource;

    use super::*;

    struct PackFactorWeightSource(usize);

    impl QuantizedWeightSource for PackFactorWeightSource {
        fn contains(&self, _name: &str) -> bool {
            true
        }

        fn load_linear(
            &self,
            _key: &str,
            _device: &Device,
            _shard: inference_quant::Shard,
        ) -> candle_core::Result<Option<std::sync::Arc<dyn inference_quant::QuantMethod>>> {
            unreachable!()
        }

        fn load_optional_tensor(
            &self,
            _name: &str,
            _device: &Device,
        ) -> candle_core::Result<Option<candle_core::Tensor>> {
            unreachable!()
        }

        fn shard_alignment(&self, _key: &str) -> candle_core::Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> candle_core::Result<usize> {
            Ok(self.0)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> candle_core::Result<Option<usize>> {
            Ok(Some(self.0))
        }
    }

    const EMBEDDING: &str = "model.embed_tokens.weight";
    const HEAD: &str = "lm_head.weight";

    #[test]
    fn explicit_promotion_and_topology_overrides_resolve_in_estimates() -> Result<()> {
        let dtype = DType::BF16;
        for (default, sensitive) in [
            (IsqType::AFQ4, IsqType::AFQ6),
            (IsqType::Q4K, IsqType::Q6K),
            (IsqType::Q5K, IsqType::Q8_0),
            (IsqType::Q6K, IsqType::Q8_0),
        ] {
            let automatic = AutoDeviceMapQuantization::isq(Some(default), None);
            assert_eq!(
                automatic.promoted_pack_factor_for(EMBEDDING, dtype, 1)?,
                sensitive.pack_factor(dtype),
                "{default}"
            );
            assert_eq!(
                automatic.unpromoted_pack_factor_for(EMBEDDING, dtype, 1)?,
                default.pack_factor(dtype),
                "{default}"
            );
            assert_eq!(
                automatic.unpromoted_pack_factor_for(
                    "model.layers.0.mlp.down_proj.weight",
                    dtype,
                    1,
                )?,
                default.pack_factor(dtype),
                "{default}"
            );
        }

        let topology = Topology::from_str(
            "'/^model\\.embed_tokens\\.weight$/':\n  isq: Q2K\n'/^lm_head\\.weight$/':\n  isq: Q8_0\n",
        )?;
        let overridden = AutoDeviceMapQuantization::isq(Some(IsqType::Q4K), Some(&topology));
        assert_eq!(
            overridden.unpromoted_pack_factor_for(EMBEDDING, dtype, 1)?,
            IsqType::Q2K.pack_factor(dtype)
        );
        assert_eq!(
            overridden.unpromoted_pack_factor_for(HEAD, dtype, 1)?,
            IsqType::Q8_0.pack_factor(dtype)
        );
        assert_eq!(
            tied_promoted_tensor_pack_factor(Some(&overridden), EMBEDDING, HEAD, dtype, 1,)?,
            IsqType::Q2K.pack_factor(dtype)
        );
        Ok(())
    }

    #[test]
    fn topology_only_quantization_uses_fallback_for_unmatched_tensors() -> Result<()> {
        let dtype = DType::BF16;
        let topology = Topology::from_str("'/^model\\.embed_tokens\\.weight$/':\n  isq: AFQ8\n")?;
        let quantization = AutoDeviceMapQuantization::isq(None, Some(&topology));
        assert_eq!(
            quantization.unpromoted_pack_factor_for(EMBEDDING, dtype, 1)?,
            IsqType::AFQ8.pack_factor(dtype)
        );
        assert_eq!(quantization.unpromoted_pack_factor_for(HEAD, dtype, 3)?, 3);
        Ok(())
    }

    #[test]
    fn topology_pack_factor_is_conservative_for_mapped_layers() -> Result<()> {
        let dtype = DType::BF16;
        let topology = Topology::from_str("'0':\n  isq: Q8_0\n")?;
        let quantization = AutoDeviceMapQuantization::isq(Some(IsqType::Q2K), Some(&topology));
        assert_eq!(
            quantization.conservative_pack_factor(dtype, IsqType::Q2K.pack_factor(dtype)),
            IsqType::Q8_0.pack_factor(dtype)
        );
        Ok(())
    }

    #[test]
    fn topology_isq_overlays_prepared_weight_source_sizing() -> Result<()> {
        let dtype = DType::BF16;
        let source = PackFactorWeightSource(IsqType::Q2K.pack_factor(dtype));
        let topology = Topology::from_str("'/^model\\.embed_tokens\\.weight$/':\n  isq: Q8_0\n")?;
        let quantization =
            AutoDeviceMapQuantization::weight_source_with_topology(&source, Some(&topology));

        assert_eq!(
            quantization.promoted_pack_factor_for(
                EMBEDDING,
                dtype,
                IsqType::Q2K.pack_factor(dtype),
            )?,
            IsqType::Q8_0.pack_factor(dtype)
        );
        assert_eq!(
            quantization.conservative_pack_factor(dtype, source.pack_factor(dtype)?),
            IsqType::Q8_0.pack_factor(dtype)
        );
        Ok(())
    }

    #[test]
    fn moqe_sizing_keeps_source_precision_for_the_unquantized_trunk() -> Result<()> {
        let dtype = DType::BF16;
        let source_factor = IsqType::Q4K.pack_factor(dtype);
        let source = PackFactorWeightSource(source_factor);
        let prepared = AutoDeviceMapQuantization::weight_source(&source);
        assert_eq!(
            prepared.conservative_moqe_pack_factor(dtype, source_factor, IsqType::Q2K),
            source_factor.min(IsqType::Q2K.pack_factor(dtype))
        );

        let checkpoint = AutoDeviceMapQuantization::isq(None, None);
        assert_eq!(
            checkpoint.conservative_moqe_pack_factor(dtype, 1, IsqType::Q2K),
            1
        );
        Ok(())
    }

    const HIDDEN: usize = 64;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 32;
    const INTERMEDIATE: usize = 96;

    fn llama_like() -> DecoderLayerShape {
        DecoderLayerShape {
            hidden_size: HIDDEN,
            num_attention_heads: HEADS,
            num_key_value_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            qkv_bias: false,
            o_bias: false,
            qk_norm: false,
            norms: 2,
            mlp: MlpShape::Gated {
                intermediate_size: INTERMEDIATE,
            },
        }
    }

    #[test]
    fn decoder_layer_elems_count_each_tensor_once() {
        let (q, kv) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
        let matrices = HIDDEN * q * 2 + HIDDEN * kv * 2 + 3 * HIDDEN * INTERMEDIATE;
        assert_eq!(llama_like().elems(1), 2 * HIDDEN + matrices);
        assert_eq!(llama_like().elems(2), 2 * HIDDEN + matrices / 2);
        let biased = DecoderLayerShape {
            qkv_bias: true,
            o_bias: true,
            qk_norm: true,
            norms: 4,
            mlp: MlpShape::Plain {
                intermediate_size: INTERMEDIATE,
                bias: true,
            },
            ..llama_like()
        };
        let plain = HIDDEN * q * 2 + HIDDEN * kv * 2 + 2 * HIDDEN * INTERMEDIATE;
        let extras = (q + 2 * kv) + HIDDEN + 2 * HEAD_DIM + INTERMEDIATE + HIDDEN;
        assert_eq!(biased.elems(1), 4 * HIDDEN + plain + extras);
    }

    #[test]
    fn decoder_model_config_reports_the_layer_head_dim() {
        let cfg = llama_like().model_config(3, 128, Some(16));
        assert_eq!((cfg.k_head_dim, cfg.v_head_dim), (HEAD_DIM, HEAD_DIM));
        assert_eq!(
            (cfg.num_kv_heads, cfg.num_attn_heads, cfg.num_layers),
            (KV_HEADS, HEADS, 3)
        );
    }
}
