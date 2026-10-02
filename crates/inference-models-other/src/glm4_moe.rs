#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use inference_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, RowParallelLayer, ShardedVarBuilder,
};
use serde::Deserialize;

use crate::deepseek_family::{
    FamilyAttention, FamilyConfig, FamilyModel, LayerCtx, MoeSpec, SharedExpert,
};
use crate::kv_cache::KvCache;
use crate::model::ModelForwardContext;
use crate::{
    attention::{AttentionDispatch, AttentionMask, SdpaParams},
    layers::{Activation, RmsNorm, apply_rotary_q},
    moe::{GroupedRouterConfig, RouterMethod, RouterRenorm, RouterScoring},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
};

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(usize, n_group, 1);
serde_default_fn!(usize, topk_group, 1);
serde_default_fn!(bool, norm_topk_prob, true);
serde_default_fn!(bool, use_qk_norm, false);
serde_default_fn!(bool, attention_bias, false);

#[derive(Deserialize, Clone, Debug)]
pub struct Glm4MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub partial_rotary_factor: f32,
    #[serde(default = "use_qk_norm")]
    pub use_qk_norm: bool,
    #[serde(default = "attention_bias")]
    pub attention_bias: bool,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default = "routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    #[serde(default = "n_group")]
    pub n_group: usize,
    #[serde(default = "topk_group")]
    pub topk_group: usize,
    #[serde(default = "norm_topk_prob")]
    pub norm_topk_prob: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    #[serde(default = "hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub head_dim: Option<usize>,
    #[serde(alias = "quantization")]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Glm4MoeConfig {
    fn router_config(&self) -> GroupedRouterConfig {
        GroupedRouterConfig {
            scoring: RouterScoring::Sigmoid,
            method: RouterMethod::NoAuxTc,
            n_group: self.n_group,
            topk_group: self.topk_group,
            routed_scaling_factor: self.routed_scaling_factor,
            renorm: RouterRenorm::TopkProb {
                norm_topk_prob: self.norm_topk_prob,
            },
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn family(&self) -> FamilyConfig<Glm4MoeAttnConfig> {
        FamilyConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            tie_word_embeddings: self.tie_word_embeddings,
            hidden_act: self.hidden_act,
            quantization_config: self.quantization_config.clone(),
            moe: Some(MoeSpec {
                n_routed_experts: self.n_routed_experts,
                num_experts_per_tok: Some(self.num_experts_per_tok),
                moe_intermediate_size: self.moe_intermediate_size,
                first_k_dense_replace: self.first_k_dense_replace,
                moe_layer_freq: None,
                shared_expert: (self.n_shared_experts > 0).then_some(SharedExpert::Replicated(
                    self.moe_intermediate_size * self.n_shared_experts,
                )),
                router: self.router_config(),
            }),
            attn: Glm4MoeAttnConfig {
                num_key_value_heads: self.num_key_value_heads,
                head_dim: self.head_dim(),
                partial_rotary_factor: self.partial_rotary_factor,
                use_qk_norm: self.use_qk_norm,
                attention_bias: self.attention_bias,
                rope_theta: self.rope_theta,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct Glm4MoeAttnConfig {
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub partial_rotary_factor: f32,
    pub use_qk_norm: bool,
    pub attention_bias: bool,
    pub rope_theta: f64,
}

pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    is_gpt_neox: bool,
}

impl RotaryEmbedding {
    fn new(
        rope_theta: f32,
        partial_rotary_factor: f32,
        head_dim: usize,
        max_seq_len: usize,
        dev: &Device,
        dtype: DType,
        is_gpt_neox: bool,
    ) -> Result<Self> {
        let rotary_dim = (partial_rotary_factor * head_dim as f32) as usize;

        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / rotary_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
            is_gpt_neox,
        })
    }

    fn apply_rotary_emb_positions(&self, xs: &Tensor, positions: &Tensor) -> Result<Tensor> {
        apply_rotary_q(xs, &self.cos, &self.sin, positions, self.is_gpt_neox)
    }

    fn forward_qk_norm(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_norm: &RmsNorm,
        k_norm: &RmsNorm,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        crate::layers::qk_rms_norm_rope(
            q,
            k,
            q_norm.weight(),
            k_norm.weight(),
            q_norm.eps(),
            k_norm.eps(),
            &self.cos,
            &self.sin,
            self.is_gpt_neox,
            positions,
        )
    }
}

/// GQA attention with partial rotary and optional QK norm.
pub struct Glm4MoeAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl FamilyAttention for Glm4MoeAttention {
    type Config = Glm4MoeAttnConfig;
    type Rope = RotaryEmbedding;

    const CUDA_DECODE_GRAPHS: bool = true;

    fn rope(
        cfg: &FamilyConfig<Glm4MoeAttnConfig>,
        dtype: DType,
        device: &Device,
        is_gptx: bool,
    ) -> Result<RotaryEmbedding> {
        RotaryEmbedding::new(
            cfg.attn.rope_theta as f32,
            cfg.attn.partial_rotary_factor,
            cfg.attn.head_dim,
            cfg.max_position_embeddings,
            device,
            dtype,
            is_gptx,
        )
    }

    fn paged_head_dim(cfg: &FamilyConfig<Glm4MoeAttnConfig>) -> usize {
        cfg.attn.head_dim
    }

    fn new(
        ctx: &LayerCtx<'_, Glm4MoeAttnConfig>,
        rotary_emb: Arc<RotaryEmbedding>,
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
        let attn = &cfg.attn;
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = attn.num_key_value_heads;
        let head_dim = attn.head_dim;

        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            attn.attention_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = inference_quant::compute_kv_shard(
            attn.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        )?;
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            attn.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            attn.attention_bias,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;

        let (q_norm, k_norm) = if attn.use_qk_norm {
            let q_norm = RmsNorm::new(
                head_dim,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("q_norm"), false),
            )?;
            let k_norm = RmsNorm::new(
                head_dim,
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("k_norm"), false),
            )?;
            (Some(q_norm), Some(k_norm))
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: inference_quant::compute_n_kv_groups(
                    attn.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
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
        let (b_sz, q_len, _) = xs.dims3()?;

        let (mut q, mut k, mut v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        {
            let positions = ctx
                .text_positions(q.device(), q.dim(2)?)?
                .ok_or_else(|| candle_core::Error::msg("missing RoPE positions"))?;
            if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
                (q, k) = self
                    .rotary_emb
                    .forward_qk_norm(&q, &k, q_norm, k_norm, positions)?;
            } else {
                q = self.rotary_emb.apply_rotary_emb_positions(&q, positions)?;
                k = self.rotary_emb.apply_rotary_emb_positions(&k, positions)?;
            }
        }

        let metadata = ctx.paged_layer(layer_idx);
        let flash_params = ctx.flash_params();
        let mut attn_output = AttentionDispatch {
            paged_attn: self.paged_attn.as_ref(),
            paged_layer: metadata,
            kv_cache,
            sdpa_params: &self.sdpa_params,
            flash_params,
        }
        .run(&q, &k, &v, attention_mask)?;

        attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let res = self.o_proj.forward(&attn_output)?;
        Ok(res)
    }

    fn add_residual(&self, uvb: &UnVarBuilder) {
        if let Some(ref q_norm) = self.q_norm {
            uvb.pp("q_norm").add(q_norm);
        }
        if let Some(ref k_norm) = self.k_norm {
            uvb.pp("k_norm").add(k_norm);
        }
    }

    fn add_projections(&self, uvb: &UnVarBuilder) {
        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("o_proj").add(&self.o_proj);
    }

    fn model_metadata(
        cfg: &FamilyConfig<Glm4MoeAttnConfig>,
        _attention_mechanism: &AttentionImplementation,
        _real_device: &Device,
        world_size: usize,
    ) -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: (cfg.attn.num_key_value_heads / world_size).max(1),
            num_attn_heads: (cfg.num_attention_heads / world_size).max(1),
            sliding_window: None,
            k_head_dim: cfg.attn.head_dim,
            v_head_dim: cfg.attn.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        }
    }
}

pub type Glm4Moe = FamilyModel<Glm4MoeAttention>;

#[cfg(test)]
#[path = "deepseek_family_tests/glm4_moe.rs"]
mod family_tests;
