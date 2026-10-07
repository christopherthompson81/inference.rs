//! The pre-norm attention + gated-MLP decoder most text models share, built from a [`DecoderSpec`].
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use inference_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use inference_tensor::{DType, Device, DeviceLocation, Module, Result, Tensor};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeLoraTarget, MlpLayer},
    attention::{AttentionDispatch, AttentionMask, FlashParams, Sdpa, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{EitherCache, KvCache, NormalCache, NormalCacheType},
    layers::{
        Activation, CausalMasker, Gemma3RopeScalingConfig, Gemma3RopeSpec, Gemma3RotaryEmbedding,
        GemmaRmsNorm, Llama3RopeConfig, Llama3RopeSpec, Llama3RotaryEmbedding, Mlp, RmsNorm,
        RotaryEmbedding, YarnRopeConfig, embedding, embedding_with_legacy_tied_uqff,
        masker::CausalMaskConfig, masker::PastKvLenCache,
    },
    model::{IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, KvCacheLayout, ModelConfigMetadata, PagedAttention,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

const DEFAULT_ROPE_THETA: f32 = 10_000.0;
const MERGED_GATE_UP_CHUNKS: usize = 2;
// Llama 3 checkpoints may carry per-frequency rope factors under this name
const ROPE_FREQS: &str = "rope_freqs.weight";
const QK_NORM_BEFORE_ROPE: (&str, &str) = ("q_norm", "k_norm");

const AMOE_LORA_TARGETS: &[AnyMoeLoraTarget] = &[
    AnyMoeLoraTarget::up("gate_proj"),
    AnyMoeLoraTarget::up("up_proj"),
    AnyMoeLoraTarget::down("down_proj"),
];

/// How the stack's RoPE tables are built.
#[derive(Clone, Debug)]
pub enum RopeKind {
    Default {
        theta: f32,
    },
    /// RoPE over the first `rotary_dim` features of each head, with its own pairing whatever the loader's.
    Partial {
        theta: f32,
        rotary_dim: usize,
        is_gpt_neox: bool,
    },
    /// Llama 3 or linear scaling, with the checkpoint's `rope_freqs.weight` factors when it has them.
    Llama3 {
        theta: f32,
        scaling: Option<Llama3RopeConfig>,
    },
    Yarn(YarnRopeConfig),
    /// Gemma 3's RoPE: frequencies in f64, with optional linear scaling.
    Gemma3 {
        theta: f64,
        scaling: Option<Gemma3RopeScalingConfig>,
    },
}

impl Default for RopeKind {
    fn default() -> Self {
        Self::Default {
            theta: DEFAULT_ROPE_THETA,
        }
    }
}

/// The layer and final norms: RMS, or Gemma's RMS over `1 + weight`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NormKind {
    #[default]
    Rms,
    Gemma,
}

/// A layer's norm names: before attention and the feed-forward, and after both in a sandwich layer (Gemma 2, GLM4).
#[derive(Clone, Copy, Debug)]
pub struct NormNames {
    pub input: &'static str,
    pub pre_ffn: &'static str,
    pub post_attn: Option<&'static str>,
    pub post_ffn: Option<&'static str>,
}

impl NormNames {
    pub const PRE: Self = Self {
        input: "input_layernorm",
        pre_ffn: "post_attention_layernorm",
        post_attn: None,
        post_ffn: None,
    };
}

impl Default for NormNames {
    fn default() -> Self {
        Self::PRE
    }
}

/// Per-head q/k RMS norm: fused into RoPE as `q_norm`/`k_norm`, or applied after it under the given names.
#[derive(Clone, Copy, Debug)]
pub enum QkNorm {
    BeforeRope,
    AfterRope { q: &'static str, k: &'static str },
}

/// Mistral's position-dependent query scaling, `1 + scale * ln(1 + floor(pos / floor_scale))`.
#[derive(Clone, Copy, Debug)]
pub struct AttentionTemperature {
    pub scale: f32,
    pub floor_scale: usize,
}

/// The shape and switches of one decoder stack; a model's config builds it.
#[derive(Clone, Debug, Default)]
pub struct DecoderSpec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope: RopeKind,
    pub max_position_embeddings: usize,
    pub qkv_bias: bool,
    pub qk_norm: Option<QkNorm>,
    /// Layers that skip RoPE (NoPE).
    pub no_rope_layers: Vec<usize>,
    /// The RoPE sliding layers take instead, as Gemma 3's local layers do.
    pub local_rope: Option<RopeKind>,
    pub attention_temperature: Option<AttentionTemperature>,
    /// One entry per layer: the sliding window it attends over, or `None` for full attention.
    pub layer_windows: Vec<Option<usize>>,
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
    pub norm: NormKind,
    pub norm_names: NormNames,
    pub o_bias: bool,
    /// `tanh(scores / cap) * cap` on the attention scores.
    pub attn_softcap: Option<f32>,
    /// The attention score scale; `1 / sqrt(head_dim)` when unset.
    pub softmax_scale: Option<f32>,
    /// `tanh(logits / cap) * cap` on the output logits.
    pub final_logit_softcap: Option<f32>,
    /// Multiplies the token embeddings, as Gemma scales them by `sqrt(hidden_size)`.
    pub embed_scale: Option<f64>,
    /// The MLP's gate and up projections are stored as one fused `gate_up_proj`.
    pub merged_gate_up: bool,
}

impl DecoderSpec {
    pub fn num_layers(&self) -> usize {
        self.layer_windows.len()
    }

    /// The window the stack's sliding layers share; the mask is built once for all of them.
    pub fn sliding_window(&self) -> Option<usize> {
        self.layer_windows.iter().flatten().next().copied()
    }

    fn rope(
        &self,
        kind: &RopeKind,
        vb_m: &ShardedVarBuilder,
        device: &Device,
        is_gptx: bool,
        dtype: DType,
    ) -> Result<RotaryEmbedding> {
        match kind {
            RopeKind::Default { theta } => RotaryEmbedding::new(
                *theta,
                self.head_dim,
                self.max_position_embeddings,
                device,
                is_gptx,
                dtype,
            ),
            RopeKind::Llama3 { theta, scaling } => {
                let freq_factors = vb_m
                    .contains_tensor(ROPE_FREQS)
                    .then(|| {
                        vb_m.clone()
                            .set_device(device.clone())
                            .get_unchecked_dtype(ROPE_FREQS, DType::F32)
                    })
                    .transpose()?;
                let spec = Llama3RopeSpec {
                    rope_theta: *theta,
                    head_dim: self.head_dim,
                    max_position_embeddings: self.max_position_embeddings,
                    scaling: scaling.as_ref(),
                };
                Ok(
                    Llama3RotaryEmbedding::new(
                        dtype,
                        spec,
                        device,
                        is_gptx,
                        freq_factors.as_ref(),
                    )?
                    .into_inner(),
                )
            }
            RopeKind::Partial {
                theta,
                rotary_dim,
                is_gpt_neox,
            } => RotaryEmbedding::new_partial(
                *theta,
                *rotary_dim,
                self.max_position_embeddings,
                device,
                *is_gpt_neox,
                dtype,
            ),
            RopeKind::Yarn(yarn) => RotaryEmbedding::new_yarn(yarn, device, is_gptx, dtype),
            RopeKind::Gemma3 { theta, scaling } => {
                let spec = Gemma3RopeSpec {
                    rope_theta: *theta,
                    head_dim: self.head_dim,
                    max_position_embeddings: self.max_position_embeddings,
                    scaling: scaling.as_ref(),
                };
                Ok(Gemma3RotaryEmbedding::new(is_gptx, dtype, spec, device)?.into_inner())
            }
        }
    }

    pub fn shape(&self) -> StackShape<'_> {
        StackShape {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            rms_norm_eps: self.rms_norm_eps,
            layer_windows: &self.layer_windows,
            tie_word_embeddings: self.tie_word_embeddings,
            quantization_config: &self.quantization_config,
            norm: self.norm,
            norm_names: self.norm_names,
            embed_scale: self.embed_scale,
            final_logit_softcap: self.final_logit_softcap,
        }
    }
}

/// Separate q/k/v/o projections, optional per-head q/k RMS norm, RoPE, and the engine's attention dispatch.
pub struct AttentionBlock {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    qk_norm: Option<(QkNorm, Norm, Norm)>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    attention_temperature: Option<AttentionTemperature>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

/// Where a layer's weights go: its index, device, communicator and attention implementation.
pub struct LayerLoad<'a> {
    pub mapper: &'a dyn DeviceMapper,
    pub layer_idx: usize,
    pub loading_isq: bool,
    pub comm: &'a Arc<inference_quant::Comm>,
    pub device: &'a Device,
    pub attention: &'a AttentionImplementation,
}

/// A sliding cache for each sliding layer and a full one, of `max_seq_len`, for the rest.
fn cache_types(layer_windows: &[Option<usize>], max_seq_len: usize) -> Vec<NormalCacheType> {
    layer_windows
        .iter()
        .map(|window| match window {
            Some(window) => NormalCacheType::SlidingWindow { window: *window },
            None => NormalCacheType::Normal { max_seq_len },
        })
        .collect()
}

/// The attention half of a decoder layer.
pub trait LayerAttention: Send + Sync {
    const CUDA_DECODE_GRAPHS: bool = false;

    /// Without a `kv_cache` the layer attends over this call's tokens only; `flash` overrides the ctx's flash params.
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: Option<&mut KvCache>,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
        flash: Option<&FlashParams>,
    ) -> Result<Tensor>;

    /// The window this layer slides over, which picks its mask.
    fn sliding_window(&self) -> Option<usize> {
        None
    }

    /// Whether the flash backend can run this layer's packed prefill.
    fn supports_packed_prefill(&self) -> bool {
        true
    }

    /// The tensors ISQ leaves alone, under `self_attn`.
    fn add_residual(&self, uvb: &UnVarBuilder);

    /// The projections, under `self_attn`, that a MoE-experts-only ISQ also leaves alone.
    fn add_projections(&self, _uvb: &UnVarBuilder) {}
}

/// The feed-forward half of a decoder layer.
pub trait LayerFfn: Send + Sync {
    /// Whether a MoE-experts-only ISQ applies, quantizing the experts and keeping the rest.
    const MOE_EXPERTS_ONLY_ISQ: bool = false;
    /// Whether AnyMoE can wrap this feed-forward in experts.
    const AMOE: bool = false;

    fn forward(&self, xs: &Tensor) -> Result<Tensor>;

    /// The tensors ISQ leaves alone, under `mlp`.
    fn add_residual(&self, _uvb: &UnVarBuilder) {}

    /// The dense MLP AnyMoE can wrap in experts, when this is one.
    fn as_mlp(&self) -> Option<&dyn MlpLayer> {
        None
    }

    fn as_mlp_mut(&mut self) -> Option<&mut Box<dyn MlpLayer>> {
        None
    }
}

impl LayerFfn for Box<dyn MlpLayer> {
    const AMOE: bool = true;

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        MlpLayer::forward(&**self, xs)
    }
    fn as_mlp(&self) -> Option<&dyn MlpLayer> {
        Some(&**self)
    }
    fn as_mlp_mut(&mut self) -> Option<&mut Box<dyn MlpLayer>> {
        Some(self)
    }
}

/// Builds each layer's attention and feed-forward halves from the layer's weights.
pub trait LayerBuilder: Sync {
    type Attention: LayerAttention;
    type Ffn: LayerFfn;

    fn build(
        &self,
        load: &LayerLoad<'_>,
        vb: ShardedVarBuilder,
    ) -> Result<(Self::Attention, Self::Ffn)>;
}

/// What the stack itself reads from a config, apart from its layers.
#[derive(Clone, Copy)]
pub struct StackShape<'a> {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub rms_norm_eps: f64,
    pub layer_windows: &'a [Option<usize>],
    pub tie_word_embeddings: bool,
    pub quantization_config: &'a Option<QuantizedConfig>,
    pub norm: NormKind,
    pub norm_names: NormNames,
    pub embed_scale: Option<f64>,
    pub final_logit_softcap: Option<f32>,
}

impl StackShape<'_> {
    fn sliding_window(&self) -> Option<usize> {
        self.layer_windows.iter().flatten().next().copied()
    }
}

/// [`AttentionBlock`] and the gated [`Mlp`], as a [`DecoderSpec`] describes them.
struct StandardLayers<'a> {
    spec: &'a DecoderSpec,
    ropes: HashMap<DeviceLocation, Arc<RotaryEmbedding>>,
    local_ropes: Option<HashMap<DeviceLocation, Arc<RotaryEmbedding>>>,
}

impl LayerBuilder for StandardLayers<'_> {
    type Attention = AttentionBlock;
    type Ffn = Box<dyn MlpLayer>;

    fn build(
        &self,
        load: &LayerLoad<'_>,
        vb: ShardedVarBuilder,
    ) -> Result<(AttentionBlock, Box<dyn MlpLayer>)> {
        let spec = self.spec;
        let ropes = match &self.local_ropes {
            Some(local) if spec.layer_windows[load.layer_idx].is_some() => local,
            _ => &self.ropes,
        };
        let rotary_emb = ropes
            .get(&load.device.location())
            .expect("No RoPE for device location!")
            .clone();
        let paged_attn = match load.attention {
            AttentionImplementation::Eager => None,
            AttentionImplementation::PagedAttention => {
                Some(PagedAttention::new(spec.head_dim, load.device, None)?)
            }
        };
        let place = |name| {
            load.mapper
                .set_device(load.layer_idx, vb.pp(name), load.loading_isq)
        };
        let attention =
            AttentionBlock::new(spec, load, place("self_attn"), rotary_emb, paged_attn)?;
        let mlp = if spec.merged_gate_up {
            Mlp::new_merged(
                place("mlp"),
                spec.hidden_size,
                spec.intermediate_size,
                MERGED_GATE_UP_CHUNKS,
                &spec.quantization_config,
                spec.hidden_act,
                load.comm,
            )?
        } else {
            Mlp::new(
                place("mlp"),
                spec.hidden_size,
                spec.intermediate_size,
                &spec.quantization_config,
                spec.hidden_act,
                load.comm,
            )?
        };
        Ok((attention, Box::new(mlp)))
    }
}

impl AttentionBlock {
    fn new(
        spec: &DecoderSpec,
        load: &LayerLoad<'_>,
        vb: ShardedVarBuilder,
        rotary_emb: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let LayerLoad {
            mapper,
            layer_idx,
            loading_isq,
            comm,
            ..
        } = *load;
        let (hidden, head_dim) = (spec.hidden_size, spec.head_dim);
        let qc = &spec.quantization_config;
        let q_proj = ColumnParallelLayer::new(
            hidden,
            spec.num_heads * head_dim,
            qc,
            spec.qkv_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = inference_quant::compute_kv_shard(spec.num_kv_heads, head_dim, comm)?;
        let kv = |name| {
            ColumnParallelLayer::new_with_shard(
                hidden,
                spec.num_kv_heads * head_dim,
                qc,
                spec.qkv_bias,
                comm,
                kv_shard,
                mapper.set_device(layer_idx, vb.pp(name), loading_isq),
            )
        };
        let (k_proj, v_proj) = (kv("k_proj")?, kv("v_proj")?);
        let o_proj = RowParallelLayer::new(
            spec.num_heads * head_dim,
            hidden,
            qc,
            spec.o_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let qk_norm = spec
            .qk_norm
            .map(|placement| -> Result<_> {
                let (q, k) = match placement {
                    QkNorm::BeforeRope => QK_NORM_BEFORE_ROPE,
                    QkNorm::AfterRope { q, k } => (q, k),
                };
                let norm = |name| {
                    Norm::new(
                        spec.norm,
                        head_dim,
                        spec.rms_norm_eps,
                        mapper.set_device(layer_idx, vb.pp(name), false),
                    )
                };
                Ok((placement, norm(q)?, norm(k)?))
            })
            .transpose()?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qk_norm,
            num_heads: spec.num_heads / comm.world_size(),
            num_kv_heads: (spec.num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb: (!spec.no_rope_layers.contains(&layer_idx)).then_some(rotary_emb),
            attention_temperature: spec.attention_temperature,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: inference_quant::compute_n_kv_groups(
                    spec.num_kv_heads,
                    spec.num_heads,
                    comm,
                )?,
                softcap: spec.attn_softcap,
                softmax_scale: spec
                    .softmax_scale
                    .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt()),
                sliding_window: spec.layer_windows[layer_idx],
                sinks: None,
                chunk: None,
            },
        })
    }

    fn attend(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: Option<&mut KvCache>,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
        flash: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let (q, k, v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        let heads = |t: Tensor, n: usize| -> Result<Tensor> {
            if q_len != 1 {
                t.reshape((b_sz, q_len, n, self.head_dim))?.transpose(1, 2)
            } else {
                t.reshape((b_sz, n, q_len, self.head_dim))
            }
        };
        let (q, k, v) = (
            heads(q, self.num_heads)?,
            heads(k, self.num_kv_heads)?,
            heads(v, self.num_kv_heads)?,
        );

        let (q, k) = self.rope_and_norm(q, k, ctx)?;

        let attn_output = match kv_cache {
            Some(kv_cache) => AttentionDispatch {
                paged_attn: self.paged_attn.as_ref(),
                paged_layer: ctx.paged_layer(layer_idx),
                kv_cache,
                sdpa_params: &self.sdpa_params,
                flash_params: flash.unwrap_or(ctx.flash_params()),
            }
            .run(&q, &k, &v, attention_mask)?,
            None => Sdpa.run_attention(
                &q,
                &k,
                &v,
                attention_mask,
                Some(flash.unwrap_or(ctx.flash_params())),
                &self.sdpa_params,
            )?,
        };
        let attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn_output)
    }

    fn rope_and_norm(
        &self,
        q: Tensor,
        k: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<(Tensor, Tensor)> {
        let Some(rotary_emb) = &self.rotary_emb else {
            return Ok((q, k));
        };
        let positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| inference_tensor::Error::msg("missing RoPE positions"))?;
        let (q, k) = match &self.qk_norm {
            Some((QkNorm::BeforeRope, q_norm, k_norm)) => rotary_emb.forward_qk_norm(
                &q,
                &k,
                q_norm.weight(),
                k_norm.weight(),
                q_norm.eps(),
                k_norm.eps(),
                positions,
            )?,
            Some((QkNorm::AfterRope { .. }, q_norm, k_norm)) => {
                let (q, k) = rotary_emb.forward(&q, &k, positions)?;
                (q_norm.forward(&q)?, k_norm.forward(&k)?)
            }
            None => rotary_emb.forward(&q, &k, positions)?,
        };
        let Some(AttentionTemperature { scale, floor_scale }) = self.attention_temperature else {
            return Ok((q, k));
        };
        let (b_sz, _, q_len, _) = q.dims4()?;
        let floor = (positions.to_dtype(DType::F32)? / floor_scale as f64)?.floor()?;
        let scales =
            ((((floor + 1.)?.log()? * f64::from(scale))? + 1.)?).reshape((b_sz, 1, q_len, 1))?;
        let q = q
            .to_dtype(DType::F32)?
            .broadcast_mul(&scales)?
            .to_dtype(q.dtype())?;
        Ok((q, k))
    }

    fn qk_norm_residual(&self, uvb: &UnVarBuilder) {
        if let Some((placement, q_norm, k_norm)) = &self.qk_norm {
            let (q, k) = match placement {
                QkNorm::BeforeRope => QK_NORM_BEFORE_ROPE,
                QkNorm::AfterRope { q, k } => (*q, *k),
            };
            q_norm.add_residual(&uvb.pp(q));
            k_norm.add_residual(&uvb.pp(k));
        }
    }
}

impl LayerAttention for AttentionBlock {
    const CUDA_DECODE_GRAPHS: bool = true;

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: Option<&mut KvCache>,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
        flash: Option<&FlashParams>,
    ) -> Result<Tensor> {
        self.attend(xs, attention_mask, kv_cache, ctx, layer_idx, flash)
    }
    fn sliding_window(&self) -> Option<usize> {
        self.sdpa_params.sliding_window
    }
    fn supports_packed_prefill(&self) -> bool {
        self.sdpa_params.softcap.is_none()
            || crate::attention::flash_backend_supports(self.head_dim, true)
    }
    fn add_residual(&self, uvb: &UnVarBuilder) {
        self.qk_norm_residual(uvb)
    }
    fn add_projections(&self, uvb: &UnVarBuilder) {
        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("o_proj").add(&self.o_proj);
    }
}

/// One layer or final norm of the stack's [`NormKind`].
pub enum Norm {
    Rms(RmsNorm),
    Gemma(GemmaRmsNorm),
}

impl Norm {
    fn new(kind: NormKind, size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(match kind {
            NormKind::Rms => Self::Rms(RmsNorm::new(size, eps, vb)?),
            NormKind::Gemma => Self::Gemma(GemmaRmsNorm::new(size, eps, vb)?),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(norm) => norm.forward(xs),
            Self::Gemma(norm) => norm.forward(xs),
        }
    }

    /// `self(x) + residual`, fused.
    fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(norm) => norm.forward_residual(x, residual),
            Self::Gemma(norm) => norm.forward_residual(x, residual),
        }
    }

    /// `self(x) + residual` and `next` of it, fused; both norms are of one kind.
    fn forward_residual_then_norm(
        &self,
        x: &Tensor,
        residual: &Tensor,
        next: &Self,
    ) -> Result<(Tensor, Tensor)> {
        match (self, next) {
            (Self::Rms(norm), Self::Rms(next)) => {
                norm.forward_residual_then_rms_norm(x, residual, next)
            }
            (Self::Gemma(norm), Self::Gemma(next)) => {
                norm.forward_residual_then_rms_norm(x, residual, next)
            }
            _ => unreachable!("a stack's norms share one kind"),
        }
    }

    fn add_residual(&self, uvb: &UnVarBuilder) {
        match self {
            Self::Rms(norm) => uvb.add(norm),
            Self::Gemma(norm) => uvb.add(norm),
        }
    }

    /// The weight the norm multiplies by, `1 + weight` for Gemma's.
    fn weight(&self) -> &Tensor {
        match self {
            Self::Rms(norm) => norm.weight(),
            Self::Gemma(norm) => norm.weight(),
        }
    }

    fn eps(&self) -> f64 {
        match self {
            Self::Rms(norm) => norm.eps(),
            Self::Gemma(norm) => norm.eps(),
        }
    }
}

/// A layer's norms under [`NormNames`]: pre-norm, or the sandwich with norms after attention and feed-forward too.
struct LayerNorms {
    input: Norm,
    pre_ffn: Norm,
    post: Option<(Norm, Norm)>,
}

/// Pre-norm residual attention then feed-forward, or the sandwich of Gemma 2 and GLM4.
pub struct DecoderLayer<A, F> {
    pub self_attn: A,
    pub mlp: F,
    norms: LayerNorms,
}

impl<A: LayerAttention, F: LayerFfn> DecoderLayer<A, F> {
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: Option<&mut KvCache>,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
        flash: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.norms.input.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx, flash)?;
        let Some((post_attn, post_ffn)) = &self.norms.post else {
            let xs = (xs + residual)?;
            let residual = &xs;
            let xs = self.mlp.forward(&self.norms.pre_ffn.forward(&xs)?)?;
            return residual + xs;
        };
        let (xs, mlp_in) =
            post_attn.forward_residual_then_norm(&xs, residual, &self.norms.pre_ffn)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        post_ffn.forward_residual(&mlp_out, &xs)
    }
}

/// Each layer attends through `full`, or through `sliding` when it has a window; either is built only if used.
pub struct LayerMasks {
    full: Option<DeviceMappedMask>,
    sliding: Option<DeviceMappedMask>,
    flash: Option<FlashParams>,
}

impl LayerMasks {
    /// Masks a model built itself, with the flash parameters they need when not the call's own.
    pub fn new(
        full: Option<DeviceMappedMask>,
        sliding: Option<DeviceMappedMask>,
        flash: Option<FlashParams>,
    ) -> Self {
        Self {
            full,
            sliding,
            flash,
        }
    }
}

/// Embeddings, layers and final norm: the part a causal LM and an embedder share.
pub struct DecoderStack<A = AttentionBlock, F = Box<dyn MlpLayer>> {
    pub embed_tokens: Arc<dyn QuantMethod>,
    pub layers: Vec<DecoderLayer<A, F>>,
    norm: Norm,
    norm_names: NormNames,
    embed_scale: Option<f64>,
    dtype: DType,
    sliding_window: Option<usize>,
    has_full_layers: bool,
    pub device: Device,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl DecoderStack {
    /// `tied_lm_head` names the head an untied-in-name, tied-in-fact UQFF may carry; `None` for an embedder.
    pub fn new(
        spec: &DecoderSpec,
        vb_m: ShardedVarBuilder,
        tied_lm_head: Option<ShardedVarBuilder>,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: &AttentionImplementation,
    ) -> Result<Self> {
        // a NoPE layer skips the RoPE step that carries q/k norm and the temperature
        if !spec.no_rope_layers.is_empty()
            && (spec.qk_norm.is_some() || spec.attention_temperature.is_some())
        {
            inference_tensor::bail!(
                "NoPE layers with q/k norm or attention temperature are not supported"
            );
        }
        if let Some(quant_cfg) = &spec.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let dtype = vb_m.dtype();
        let vb_rope = vb_m.clone();
        Self::new_with(
            spec.shape(),
            vb_m,
            tied_lm_head,
            normal_loading_metadata,
            attention_mechanism,
            |mapper, real_device| {
                let ropes_of = |kind: &RopeKind| {
                    crate::device_map::per_layer_device(
                        mapper,
                        spec.num_layers(),
                        real_device,
                        |device| spec.rope(kind, &vb_rope, device, is_gptx, dtype),
                    )
                };
                Ok(StandardLayers {
                    spec,
                    ropes: ropes_of(&spec.rope)?,
                    local_ropes: spec.local_rope.as_ref().map(ropes_of).transpose()?,
                })
            },
        )
    }
}

impl<A: LayerAttention, F: LayerFfn> DecoderStack<A, F> {
    /// The stack with layers from the builder `make` returns once the device mapper is known.
    pub fn new_with<B: LayerBuilder<Attention = A, Ffn = F>>(
        shape: StackShape<'_>,
        vb_m: ShardedVarBuilder,
        tied_lm_head: Option<ShardedVarBuilder>,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: &AttentionImplementation,
        make: impl FnOnce(&dyn DeviceMapper, &Device) -> Result<B>,
    ) -> Result<Self> {
        // one sliding mask serves every sliding layer
        if shape
            .layer_windows
            .iter()
            .flatten()
            .any(|window| Some(*window) != shape.sliding_window())
        {
            inference_tensor::bail!(
                "decoder layers slide over different windows: {:?}",
                shape.layer_windows
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let loading_isq = normal_loading_metadata.loading_isq;
        let real_device = &normal_loading_metadata.real_device;
        let dtype = vb_m.dtype();
        let embed_vb = mapper.set_nm_device(vb_m.pp("embed_tokens"), loading_isq);
        let embed_tokens = match tied_lm_head {
            Some(head) => embedding_with_legacy_tied_uqff(
                shape.vocab_size,
                shape.hidden_size,
                embed_vb,
                shape
                    .tie_word_embeddings
                    .then(|| mapper.set_nm_device(head, loading_isq)),
                shape.quantization_config,
            )?,
            None => embedding(
                shape.vocab_size,
                shape.hidden_size,
                embed_vb,
                shape.quantization_config,
            )?,
        };

        let builder = make(&*mapper, real_device)?;
        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..shape.layer_windows.len(),
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| -> Result<DecoderLayer<A, F>> {
            let device = mapper.device_for(layer_idx, false).unwrap_or(real_device);
            let comm = mapper.get_comm_for(layer_idx)?;
            let load = LayerLoad {
                mapper: &*mapper,
                layer_idx,
                loading_isq,
                comm: &comm,
                device,
                attention: attention_mechanism,
            };
            let vb = vb_l.pp(layer_idx);
            let (self_attn, mlp) = builder.build(&load, vb.clone())?;
            let norm = |name| {
                Norm::new(
                    shape.norm,
                    shape.hidden_size,
                    shape.rms_norm_eps,
                    mapper.set_device(layer_idx, vb.pp(name), false),
                )
            };
            let names = shape.norm_names;
            let post = match (names.post_attn, names.post_ffn) {
                (Some(post_attn), Some(post_ffn)) => Some((norm(post_attn)?, norm(post_ffn)?)),
                (None, None) => None,
                _ => inference_tensor::bail!("a sandwich layer names both post norms"),
            };
            Ok(DecoderLayer {
                self_attn,
                mlp,
                norms: LayerNorms {
                    input: norm(names.input)?,
                    pre_ffn: norm(names.pre_ffn)?,
                    post,
                },
            })
        })?;
        let norm = Norm::new(
            shape.norm,
            shape.hidden_size,
            shape.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            norm_names: shape.norm_names,
            embed_scale: shape.embed_scale,
            dtype,
            sliding_window: shape.sliding_window(),
            has_full_layers: shape.layer_windows.iter().any(Option::is_none),
            device: real_device.clone(),
            mapper,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
        match self.embed_scale {
            Some(scale) => xs * scale,
            None => Ok(xs),
        }
    }

    /// The full mask, and the sliding one when a layer slides, for this call's tokens after `past`.
    pub fn masks(
        &self,
        input_ids: &Tensor,
        dtype: DType,
        past: &dyn PastKvLenCache,
        is_first_prompt_chunk: bool,
    ) -> Result<LayerMasks> {
        let mask = |sliding_window| -> Result<DeviceMappedMask> {
            let mask = CausalMasker.make_causal_mask(
                input_ids,
                past,
                dtype,
                &CausalMaskConfig {
                    sliding_window,
                    ..Default::default()
                },
            )?;
            // PagedAttention prompt chunking
            let mask = if is_first_prompt_chunk {
                mask
            } else {
                AttentionMask::None
            };
            DeviceMappedMask::new(mask, &*self.mapper)
        };
        Ok(LayerMasks {
            full: self.has_full_layers.then(|| mask(None)).transpose()?,
            sliding: self
                .sliding_window
                .map(|window| mask(Some(window)))
                .transpose()?,
            flash: None,
        })
    }

    /// The normed hidden states; `cache` is `None` for an encoder pass over the call's tokens alone.
    pub fn forward(
        &self,
        mut xs: Tensor,
        masks: &LayerMasks,
        mut cache: Option<&mut [KvCache]>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let layer_mask = match (&masks.sliding, &masks.full) {
                (Some(sliding), _) if layer.self_attn.sliding_window().is_some() => sliding,
                (_, Some(full)) => full,
                (Some(sliding), None) => sliding,
                (None, None) => unreachable!("a stack has a full or a sliding layer"),
            };
            let kv_cache = cache.as_deref_mut().map(|cache| &mut cache[i]);
            xs = layer.forward(
                &xs,
                &layer_mask.get(xs.device()),
                kv_cache,
                ctx,
                i,
                masks.flash.as_ref(),
            )?;
        }
        self.norm.forward(&xs.to_device(&self.device)?)
    }

    /// The tensors ISQ leaves alone, under the stack's own prefix; `with_projections` for MoE-experts-only ISQ.
    pub fn residual_uvb(&self, uvb_m: &UnVarBuilder, with_projections: bool) {
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        self.norm.add_residual(&uvb_m.pp("norm"));
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            let names = self.norm_names;
            layer.norms.input.add_residual(&uvb_l.pp(names.input));
            layer.norms.pre_ffn.add_residual(&uvb_l.pp(names.pre_ffn));
            if let (Some((post_attn, post_ffn)), Some(attn_name), Some(ffn_name)) =
                (&layer.norms.post, names.post_attn, names.post_ffn)
            {
                post_attn.add_residual(&uvb_l.pp(attn_name));
                post_ffn.add_residual(&uvb_l.pp(ffn_name));
            }
            layer.self_attn.add_residual(&uvb_l.pp("self_attn"));
            layer.mlp.add_residual(&uvb_l.pp("mlp"));
            if with_projections {
                layer.self_attn.add_projections(&uvb_l.pp("self_attn"));
            }
        }
    }
}

/// A [`DecoderStack`] under `model.` with an `lm_head`, as the engine runs a text model.
pub struct CausalLm<A = AttentionBlock, F = Box<dyn MlpLayer>> {
    stack: DecoderStack<A, F>,
    lm_head: Arc<dyn QuantMethod>,
    final_logit_softcap: Option<f32>,
    cache: EitherCache,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
}

impl CausalLm {
    pub fn new(
        spec: &DecoderSpec,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        Self::new_inner(
            spec,
            vb.pp("model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    /// As [`CausalLm::new`] with the stack and head under the prefixes a multimodal checkpoint gives them.
    pub fn new_inner(
        spec: &DecoderSpec,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let loading_isq = normal_loading_metadata.loading_isq;
        let stack = DecoderStack::new(
            spec,
            vb_m,
            Some(vb_lm_head.clone()),
            is_gptx,
            normal_loading_metadata,
            &attention_mechanism,
        )?;
        let world_size = stack.mapper.get_comm_for(0)?.world_size();
        let cfg = ModelConfigMetadata {
            max_seq_len: spec.max_position_embeddings,
            num_layers: spec.num_layers(),
            hidden_size: spec.hidden_size,
            num_kv_heads: (spec.num_kv_heads / world_size).max(1),
            num_attn_heads: spec.num_heads / world_size,
            sliding_window: spec.sliding_window(),
            k_head_dim: spec.head_dim,
            v_head_dim: spec.head_dim,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        Self::with_stack(
            stack,
            spec.shape(),
            vb_lm_head,
            loading_isq,
            spec.max_position_embeddings,
            cfg,
        )
    }
}

impl<A: LayerAttention, F: LayerFfn> CausalLm<A, F> {
    /// The stack's model, with the `lm_head` (or the tied embedding) and the KV cache its layers' windows need.
    pub fn with_stack(
        stack: DecoderStack<A, F>,
        shape: StackShape<'_>,
        vb_lm_head: ShardedVarBuilder,
        loading_isq: bool,
        max_position_embeddings: usize,
        cfg: ModelConfigMetadata,
    ) -> Result<Self> {
        let lm_head = if shape.tie_word_embeddings {
            stack.embed_tokens.clone()
        } else {
            ReplicatedLayer::new(
                shape.hidden_size,
                shape.vocab_size,
                shape.quantization_config,
                false,
                stack.mapper.set_nm_device(vb_lm_head, loading_isq),
            )?
        };
        let cache_types = cache_types(shape.layer_windows, max_position_embeddings);
        Ok(Self {
            stack,
            lm_head,
            final_logit_softcap: shape.final_logit_softcap,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: max_position_embeddings,
            cfg,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.stack.embed(input_ids)
    }

    /// The device mapper a model maps the masks it builds itself with.
    pub fn stack_mapper(&self) -> &(dyn DeviceMapper + Send + Sync) {
        &*self.stack.mapper
    }

    pub fn embed_dtype(&self) -> DType {
        self.stack.dtype
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward_embeds(input_ids, self.stack.embed(input_ids)?, ctx)
    }

    /// Runs the stack over `xs`, embeddings a multimodal model may have spliced media into.
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        xs: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let masks = {
            let cache = &self.cache.normal().0;
            self.stack.masks(
                input_ids,
                xs.dtype(),
                &ctx.mask_cache(cache),
                ctx.is_first_prompt_chunk(),
            )?
        };
        self.forward_with_masks(xs, &masks, ctx)
    }

    /// Runs the stack over `xs` with masks the model built itself.
    pub fn forward_with_masks(
        &self,
        xs: Tensor,
        masks: &LayerMasks,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let xs = self.stack.forward(xs, masks, Some(cache), ctx)?;
        let xs = ctx.logits(&xs)?;
        let logits = ctx.lm_head(&*self.lm_head, &xs)?;
        match self.final_logit_softcap {
            Some(cap) => inference_quant::softcap(&logits, cap)?.to_dtype(logits.dtype()),
            None => Ok(logits),
        }
    }

    /// The tensors ISQ leaves alone, under `uvb_m`, the stack's prefix.
    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        self.stack.residual_uvb(&uvb_m, false);
        uvb_m.to_safetensors()
    }

    fn residual_uvb(&self, with_projections: bool) -> UnVarBuilder {
        let uvb = UnVarBuilder::new();
        self.stack.residual_uvb(&uvb.pp("model"), with_projections);
        uvb
    }
}

impl<A: LayerAttention, F: LayerFfn> IsqModel for CausalLm<A, F> {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.residual_uvb(false).to_safetensors()
    }
    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        F::MOE_EXPERTS_ONLY_ISQ.then(|| self.residual_uvb(true).to_safetensors())
    }
}

impl<A: LayerAttention, F: LayerFfn> crate::speculative::SpeculativeTargetMixin for CausalLm<A, F> {}

impl<A: LayerAttention, F: LayerFfn> NormalModel for CausalLm<A, F> {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.stack.device
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    fn supports_packed_prefill(&self) -> bool {
        self.stack
            .layers
            .iter()
            .all(|layer| layer.self_attn.supports_packed_prefill())
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        A::CUDA_DECODE_GRAPHS
    }
}

impl<A: LayerAttention, F: LayerFfn> AnyMoeBaseModelMixin for CausalLm<A, F> {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.stack
            .layers
            .iter()
            .filter_map(|layer| layer.mlp.as_mlp())
            .collect()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.stack
            .layers
            .iter_mut()
            .filter_map(|layer| layer.mlp.as_mlp_mut())
            .collect()
    }
    fn amoe_lora_targets(&self) -> &'static [AnyMoeLoraTarget] {
        AMOE_LORA_TARGETS
    }
    fn amoe_fine_tuned_expert(
        &self,
        layer: usize,
        base: &dyn MlpLayer,
        vb: ShardedVarBuilder,
    ) -> Result<Box<dyn MlpLayer>> {
        let (dtype, device) = base.dtype_device();
        Ok(Box::new(Mlp::replicate(
            base.get_params(),
            vb.set_dtype(dtype).set_device(device),
            base.hidden_act(),
            &self.stack.mapper.get_comm_for(layer)?,
        )?))
    }
    fn amoe_supported(&self) -> bool {
        F::AMOE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_SEQ: usize = 4096;
    const WINDOW: usize = 128;

    #[test]
    fn eager_cache_layout_follows_each_layers_window() {
        let spec = DecoderSpec {
            vocab_size: 8,
            hidden_size: 8,
            intermediate_size: 8,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 8,
            hidden_act: Activation::Silu,
            rms_norm_eps: 1e-6,
            rope: RopeKind::Default { theta: 1e4 },
            max_position_embeddings: MAX_SEQ,
            qkv_bias: false,
            qk_norm: None,
            no_rope_layers: Vec::new(),
            attention_temperature: None,
            layer_windows: vec![None, Some(WINDOW), None],
            tie_word_embeddings: false,
            quantization_config: None,
            ..Default::default()
        };
        let types = cache_types(&spec.layer_windows, spec.max_position_embeddings);
        assert!(matches!(
            types[0],
            NormalCacheType::Normal {
                max_seq_len: MAX_SEQ
            }
        ));
        assert!(matches!(
            types[1],
            NormalCacheType::SlidingWindow { window: WINDOW }
        ));
        assert!(matches!(
            types[2],
            NormalCacheType::Normal {
                max_seq_len: MAX_SEQ
            }
        ));
        assert_eq!(spec.sliding_window(), Some(WINDOW));
    }
}
