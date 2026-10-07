use std::sync::Arc;

use inference_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};
use inference_tensor::{D, DType, Device, Module, Result, Tensor};

use super::{FamilyAttention, FamilyConfig, LayerCtx};
use crate::attention::{AttentionMask, SdpaParams};
use crate::kv_cache::KvCache;
use crate::layers::{
    DeepSeekV2RopeConfig, DeepSeekV2RopeScaling, DeepSeekV2RotaryEmbedding, RmsNorm, Sdpa,
};
use crate::mla::{
    MlaKvBProjection, MlaWeights, mla_cache_forward, mla_decode_forward, should_use_mla_cache,
    should_use_mla_decode,
};
use crate::model::ModelForwardContext;
use crate::ops::SplitOp;
use crate::paged_attention::{
    AttentionImplementation, ModelConfigMetadata, PagedAttention, PagedAttentionInputMetadata,
};
use crate::utils::unvarbuilder::UnVarBuilder;

/// When the paged KV cache takes the compressed MLA layout instead of the standard one.
#[derive(Clone, Copy, Debug)]
pub enum MlaKvLayout {
    /// Whenever paged attention is on (DeepSeek-V2/V3).
    Paged,
    /// Paged attention on a CUDA device (GLM4-MoE-Lite).
    PagedOnCudaDevice,
}

#[derive(Clone, Debug)]
pub struct MlaConfig {
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub attention_bias: bool,
    pub softmax_scale: f32,
    pub rope_theta: f32,
    pub rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub kv_layout: MlaKvLayout,
    // names the model in load errors
    pub label: &'static str,
}

impl MlaConfig {
    fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }
}

/// `1/sqrt(q_head_dim)`, times the squared yarn mscale when the rope scaling is yarn.
pub fn mla_softmax_scale(q_head_dim: usize, rope_scaling: Option<&DeepSeekV2RopeScaling>) -> f32 {
    let mut softmax_scale = 1.0 / (q_head_dim as f32).sqrt();
    if let Some(DeepSeekV2RopeScaling::Yarn {
        mscale_all_dim,
        factor,
        ..
    }) = rope_scaling
    {
        let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(*factor, *mscale_all_dim);
        softmax_scale = softmax_scale * mscale * mscale;
    }
    softmax_scale
}

#[derive(Clone, Copy)]
struct MlaDims {
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
}

enum QProj {
    Plain(Arc<dyn QuantMethod>),
    Lora {
        a: Arc<dyn QuantMethod>,
        norm: RmsNorm,
        b: Arc<dyn QuantMethod>,
    },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(xs)?)?),
            Self::Plain(lin) => lin.forward(xs),
        }
    }
}

/// Multi-head latent attention (DeepSeek-V2/V3, GLM4-MoE-Lite).
pub struct MlaAttention {
    q: QProj,
    kv_a_proj_with_mqa: Arc<dyn QuantMethod>,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: MlaKvBProjection,
    o_proj: Arc<dyn QuantMethod>,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    dims: MlaDims,
    q_head_dim: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    num_attention_heads: usize,
    mla_weights: MlaWeights,
}

impl FamilyAttention for MlaAttention {
    type Config = MlaConfig;
    type Rope = DeepSeekV2RotaryEmbedding;

    fn rope(
        cfg: &FamilyConfig<MlaConfig>,
        dtype: DType,
        device: &Device,
        _is_gptx: bool,
    ) -> Result<DeepSeekV2RotaryEmbedding> {
        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: cfg.attn.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.attn.rope_theta,
            qk_rope_head_dim: cfg.attn.qk_rope_head_dim,
        };
        DeepSeekV2RotaryEmbedding::new(&rope_cfg, dtype, device)
    }

    fn paged_head_dim(cfg: &FamilyConfig<MlaConfig>) -> usize {
        cfg.attn.v_head_dim
    }

    fn new(
        ctx: &LayerCtx<'_, MlaConfig>,
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let LayerCtx {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = *ctx;
        let mla = &cfg.attn;
        let q_head_dim = mla.q_head_dim();
        let q = match mla.q_lora_rank {
            Some(lora_rank) => {
                let a = ReplicatedLayer::new(
                    cfg.hidden_size,
                    lora_rank,
                    &cfg.quantization_config,
                    mla.attention_bias,
                    mapper.set_device(layer_idx, vb.pp("q_a_proj"), loading_isq),
                )?;
                let norm = RmsNorm::new(
                    lora_rank,
                    cfg.rms_norm_eps,
                    mapper.set_device(layer_idx, vb.pp("q_a_layernorm"), false),
                )?;
                let b = ColumnParallelLayer::new(
                    lora_rank,
                    cfg.num_attention_heads * q_head_dim,
                    &cfg.quantization_config,
                    false,
                    comm,
                    mapper.set_device(layer_idx, vb.pp("q_b_proj"), loading_isq),
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * q_head_dim,
                &cfg.quantization_config,
                false,
                comm,
                mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
            )?),
        };

        let kv_a_proj_with_mqa = ReplicatedLayer::new(
            cfg.hidden_size,
            mla.kv_lora_rank + mla.qk_rope_head_dim,
            &cfg.quantization_config,
            mla.attention_bias,
            mapper.set_device(layer_idx, vb.pp("kv_a_proj_with_mqa"), loading_isq),
        )?;
        let kv_a_layernorm = RmsNorm::new(
            mla.kv_lora_rank,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("kv_a_layernorm"), false),
        )?;
        let k_b_vb = vb.pp("k_b_proj");
        let v_b_vb = vb.pp("v_b_proj");
        let kv_b_proj = match (
            crate::layers::contains_tensor_or_weight_source(&k_b_vb, "weight"),
            crate::layers::contains_tensor_or_weight_source(&v_b_vb, "weight"),
        ) {
            (true, true) => MlaKvBProjection::split(
                ColumnParallelLayer::new(
                    mla.qk_nope_head_dim,
                    cfg.num_attention_heads * mla.kv_lora_rank,
                    &cfg.quantization_config,
                    false,
                    comm,
                    mapper.set_device(layer_idx, k_b_vb, loading_isq),
                )?,
                ColumnParallelLayer::new(
                    mla.kv_lora_rank,
                    cfg.num_attention_heads * mla.v_head_dim,
                    &cfg.quantization_config,
                    false,
                    comm,
                    mapper.set_device(layer_idx, v_b_vb, loading_isq),
                )?,
            ),
            (false, false) => MlaKvBProjection::fused(ColumnParallelLayer::new(
                mla.kv_lora_rank,
                cfg.num_attention_heads * (q_head_dim - mla.qk_rope_head_dim + mla.v_head_dim),
                &cfg.quantization_config,
                false,
                comm,
                mapper.set_device(layer_idx, vb.pp("kv_b_proj"), loading_isq),
            )?),
            _ => inference_tensor::bail!(
                "{} layer {layer_idx} has incomplete split MLA weights",
                mla.label
            ),
        };

        let o_proj = RowParallelLayer::new(
            cfg.num_attention_heads * mla.v_head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            mla.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        let mla_weights = MlaWeights::new(
            paged_attn.is_some(),
            mapper.device_for(layer_idx, loading_isq),
        );

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            dims: MlaDims {
                kv_lora_rank: mla.kv_lora_rank,
                qk_nope_head_dim: mla.qk_nope_head_dim,
                qk_rope_head_dim: mla.qk_rope_head_dim,
                v_head_dim: mla.v_head_dim,
            },
            q_head_dim,
            paged_attn,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            sdpa_params: SdpaParams {
                n_kv_groups: 1,
                softcap: None,
                softmax_scale: mla.softmax_scale,
                sliding_window: None,
                sinks: None,
                chunk: None,
            },
            mla_weights,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;

        let mut q = self.q.forward(xs)?;
        q = q
            .reshape((bs, seq_len, self.num_attention_heads, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_split = q.split(
            &[self.dims.qk_nope_head_dim, self.dims.qk_rope_head_dim],
            D::Minus1,
        )?;
        let q_nope = q_split[0].clone();
        let mut q_pe = q_split[1].clone();

        let mut compressed_kv = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv_split = compressed_kv.split(
            &[self.dims.kv_lora_rank, self.dims.qk_rope_head_dim],
            D::Minus1,
        )?;
        compressed_kv = ckv_split[0].clone();
        let mut k_pe = ckv_split[1].clone();
        k_pe = k_pe
            .reshape((bs, seq_len, 1, self.dims.qk_rope_head_dim))?
            .transpose(1, 2)?;

        let ckv = self.kv_a_layernorm.forward(&compressed_kv)?;

        let rope_positions = ctx
            .text_positions(q_pe.device(), q_pe.dim(2)?)?
            .ok_or_else(|| inference_tensor::Error::msg("missing RoPE positions"))?;
        (q_pe, k_pe) = self.rotary_emb.forward(&q_pe, &k_pe, rope_positions)?;
        let metadata = ctx.paged_layer(layer_idx);

        let use_mla_decode = should_use_mla_decode(
            attention_mask,
            seq_len,
            self.paged_attn.is_some(),
            q_nope.device(),
            &metadata,
            &self.kv_b_proj,
        );

        let mut attn_out = if use_mla_decode {
            mla_decode_forward(
                &q_nope,
                &q_pe,
                &ckv,
                &k_pe,
                &metadata,
                &self.mla_weights,
                &self.kv_b_proj,
                &self.sdpa_params,
                self.num_attention_heads,
                self.dims.kv_lora_rank,
                self.dims.qk_rope_head_dim,
                self.dims.qk_nope_head_dim,
                self.dims.v_head_dim,
                bs,
                seq_len,
            )?
        } else {
            let use_absorbed = self.kv_b_proj.is_split()
                && (self.paged_attn.is_none() || q_nope.device().is_cuda());
            let (k_nope, mut v, q_nope, value_head_dim) = if use_absorbed {
                let q_nope = self.kv_b_proj.project_query(&q_nope)?;
                let latent = ckv
                    .unsqueeze(1)?
                    .repeat((1, self.num_attention_heads, 1, 1))?;
                (latent.clone(), latent, q_nope, self.dims.kv_lora_rank)
            } else {
                let (k_nope, v) = self.kv_b_proj.expanded_kv(
                    &ckv,
                    self.num_attention_heads,
                    self.dims.qk_nope_head_dim,
                    self.dims.v_head_dim,
                )?;
                (k_nope, v, q_nope, self.dims.v_head_dim)
            };

            let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?.contiguous()?;
            let mut k = Tensor::cat(
                &[&k_nope, &k_pe.repeat((1, self.num_attention_heads, 1, 1))?],
                D::Minus1,
            )?
            .contiguous()?;

            let use_mla_cache =
                should_use_mla_cache(self.paged_attn.is_some(), q.device(), &self.kv_b_proj);

            if use_mla_cache {
                mla_cache_forward(
                    &q,
                    &k,
                    &v,
                    &ckv,
                    &k_pe,
                    attention_mask,
                    ctx.seqlen_offsets(),
                    &metadata,
                    ctx.flash_params(),
                    &self.kv_b_proj,
                    &self.sdpa_params,
                    self.num_attention_heads,
                    self.dims.kv_lora_rank,
                    self.dims.qk_rope_head_dim,
                    if use_absorbed {
                        self.dims.kv_lora_rank
                    } else {
                        self.dims.qk_nope_head_dim
                    },
                    value_head_dim,
                    bs,
                    seq_len,
                )?
            } else {
                let output = match &self.paged_attn {
                    Some(paged_attn) => match metadata {
                        Some(((key_cache, value_cache), input_metadata)) => {
                            let v = v
                                .pad_with_zeros(
                                    D::Minus1,
                                    0,
                                    self.q_head_dim - self.dims.v_head_dim,
                                )?
                                .contiguous()?;
                            paged_attn
                                .forward(
                                    &q,
                                    &k,
                                    &v,
                                    attention_mask,
                                    Some(key_cache),
                                    Some(value_cache),
                                    input_metadata,
                                    &self.sdpa_params,
                                    Some(ctx.flash_params()),
                                )?
                                .narrow(D::Minus1, 0, self.dims.v_head_dim)?
                        }
                        None => {
                            // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                            // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                            let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                            // Sanity check.
                            assert!(!attention_mask.is_none());
                            let v = v
                                .pad_with_zeros(
                                    D::Minus1,
                                    0,
                                    self.q_head_dim - self.dims.v_head_dim,
                                )?
                                .contiguous()?;
                            paged_attn
                                .forward(
                                    &q,
                                    &k,
                                    &v,
                                    attention_mask,
                                    None,
                                    None,
                                    &input_metadata,
                                    &self.sdpa_params,
                                    Some(ctx.flash_params()),
                                )?
                                .narrow(D::Minus1, 0, self.dims.v_head_dim)?
                        }
                    },
                    None => {
                        (k, v) = kv_cache.append(&k, &v)?;

                        Sdpa.run_attention(
                            &q,
                            &k,
                            &v,
                            attention_mask,
                            Some(ctx.flash_params()),
                            &self.sdpa_params,
                        )?
                    }
                };
                if use_absorbed {
                    self.kv_b_proj.project_value(&output)?
                } else {
                    output
                }
            }
        };

        attn_out = if !matches!(attention_mask, AttentionMask::None) {
            attn_out.transpose(1, 2)?.reshape((bs, seq_len, ()))?
        } else {
            attn_out.reshape((bs, seq_len, ()))?
        };

        self.o_proj.forward(&attn_out)
    }

    fn add_residual(&self, uvb: &UnVarBuilder) {
        uvb.pp("kv_a_layernorm").add(&self.kv_a_layernorm);
        if let QProj::Lora { norm, .. } = &self.q {
            uvb.pp("q_a_layernorm").add(norm);
        }
    }

    fn add_projections(&self, uvb: &UnVarBuilder) {
        match &self.q {
            QProj::Plain(q) => uvb.pp("q_proj").add(q),
            QProj::Lora { a, norm, b } => {
                uvb.pp("q_a_proj").add(a);
                uvb.pp("q_a_layernorm").add(norm);
                uvb.pp("q_b_proj").add(b);
            }
        }
        uvb.pp("kv_a_proj_with_mqa").add(&self.kv_a_proj_with_mqa);
        if let Some(projection) = self.kv_b_proj.fused_layer() {
            uvb.pp("kv_b_proj").add(projection);
        } else if let Some((key, value)) = self.kv_b_proj.split_projections() {
            uvb.pp("k_b_proj").add(key);
            uvb.pp("v_b_proj").add(value);
        }
        uvb.pp("o_proj").add(&self.o_proj);
    }

    #[cfg_attr(
        not(all(feature = "cuda", target_family = "unix")),
        allow(unused_variables)
    )]
    fn model_metadata(
        cfg: &FamilyConfig<MlaConfig>,
        attention_mechanism: &AttentionImplementation,
        real_device: &Device,
        world_size: usize,
    ) -> ModelConfigMetadata {
        let mla = &cfg.attn;
        let paged = matches!(attention_mechanism, AttentionImplementation::PagedAttention);
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        let mla_layout = paged
            && match mla.kv_layout {
                MlaKvLayout::Paged => true,
                MlaKvLayout::PagedOnCudaDevice => matches!(real_device, Device::Cuda(_)),
            };
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        let mla_layout = false;
        ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: (cfg.num_attention_heads / world_size).max(1),
            num_attn_heads: (cfg.num_attention_heads / world_size).max(1),
            sliding_window: None,
            k_head_dim: mla.q_head_dim(),
            v_head_dim: if paged {
                mla.q_head_dim()
            } else {
                mla.v_head_dim
            },
            kv_cache_layout: if mla_layout {
                crate::paged_attention::KvCacheLayout::Mla {
                    kv_lora_rank: mla.kv_lora_rank,
                    kpe_head_dim: mla.qk_rope_head_dim,
                }
            } else {
                crate::paged_attention::KvCacheLayout::Standard
            },
        }
    }
}
