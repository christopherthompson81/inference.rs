//! The pre-norm attention + gated-MLP decoder most text models share, built from a [`DecoderSpec`].
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use inference_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use inference_tensor::{D, DType, Device, DeviceLocation, Module, Result, Tensor, nn::LayerNorm};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeLoraTarget, MlpLayer},
    attention::{AttentionDispatch, AttentionMask, FlashParams, Sdpa, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{EitherCache, KvCache, NormalCache, NormalCacheType},
    layers::{
        Activation, CausalMasker, F32RmsNorm, FusedGateUpMlp, Gemma3RopeScalingConfig,
        Gemma3RopeSpec, Gemma3RotaryEmbedding, GemmaRmsNorm, Llama3RopeConfig, Llama3RopeSpec,
        Llama3RotaryEmbedding, Mlp, PhiRopeConfig, PhiRotaryEmbedding, PlainMlp,
        Qwen2VLRotaryEmbedding, Qwen3VLRotaryEmbedding, RmsNorm, RotaryEmbedding, YarnRopeConfig,
        embedding, embedding_with_legacy_tied_uqff, layer_norm, masker::CausalMaskConfig,
        masker::PastKvLenCache,
    },
    model::{IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, KvCacheLayout, ModelConfigMetadata, PagedAttention,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

mod moe;

pub use moe::{BLOCK_SPARSE_MOE, MoeRouting, MoeSpec, SparseMoe};

/// A layer's feed-forward, dense or routed, as most checkpoints name it.
pub const MLP: &str = "mlp";
const DEFAULT_ROPE_THETA: f32 = 10_000.0;
const MERGED_GATE_UP_CHUNKS: usize = 2;
const O_PROJ: &str = "o_proj";
const QKV_PROJ: &str = "qkv_proj";
// Phi-3 GGUFs carry their LongRoPE factors as tensors
const PHI_ROPE_FACTORS: (&str, &str) = ("rope_factors_short.weight", "rope_factors_long.weight");
// Llama 3 checkpoints may carry per-frequency rope factors under this name
const ROPE_FREQS: &str = "rope_freqs.weight";

const AMOE_LORA_TARGETS: &[AnyMoeLoraTarget] = &[
    AnyMoeLoraTarget::up("gate_proj"),
    AnyMoeLoraTarget::up("up_proj"),
    AnyMoeLoraTarget::down("down_proj"),
];
const AMOE_MERGED_LORA_TARGETS: &[AnyMoeLoraTarget] = &[
    AnyMoeLoraTarget {
        name: "gate_up_proj",
        shape: |hidden, intermediate| (hidden, 2 * intermediate),
    },
    AnyMoeLoraTarget::down("down_proj"),
];

/// How the stack's RoPE tables are built.
#[derive(Clone, Debug)]
pub enum RopeKind {
    Default {
        theta: f32,
    },
    /// RoPE over the first `rotary_dim` features of each head; `is_gpt_neox` overrides the loader's pairing.
    Partial {
        theta: f32,
        rotary_dim: usize,
        is_gpt_neox: Option<bool>,
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
    /// Phi's LongRoPE, switching to its long factors once a sequence outgrows the original context.
    Phi(PhiRopeConfig),
    /// M-RoPE over (t, h, w) `sections`, chunked (Qwen2-VL) or interleaved (Qwen3-VL); the model sets the tables.
    MRope {
        theta: f32,
        sections: Vec<usize>,
        interleaved: bool,
    },
}

impl Default for RopeKind {
    fn default() -> Self {
        Self::Default {
            theta: DEFAULT_ROPE_THETA,
        }
    }
}

/// The layer and final norms: RMS, Gemma's RMS over `1 + weight`, LayerNorm with a bias, or RMS computed in f32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NormKind {
    #[default]
    Rms,
    Gemma,
    Layer,
    F32Rms,
}

/// A layer's norm names; a sandwich adds post norms (Gemma 2, GLM4), no `pre_ffn` makes it parallel (Phi-2).
#[derive(Clone, Copy, Debug)]
pub struct NormNames {
    pub input: &'static str,
    pub pre_ffn: Option<&'static str>,
    pub post_attn: Option<&'static str>,
    pub post_ffn: Option<&'static str>,
    /// The stack's final norm, under the stack's prefix.
    pub last: &'static str,
}

impl NormNames {
    pub const PRE: Self = Self {
        input: "input_layernorm",
        pre_ffn: Some("post_attention_layernorm"),
        post_attn: None,
        post_ffn: None,
        last: "norm",
    };
}

impl Default for NormNames {
    fn default() -> Self {
        Self::PRE
    }
}

/// Per-head q/k norm under the given names, before RoPE (fused into it for RMS norms) or after it.
#[derive(Clone, Copy, Debug)]
pub enum QkNorm {
    BeforeRope { q: &'static str, k: &'static str },
    AfterRope { q: &'static str, k: &'static str },
}

impl QkNorm {
    pub const BEFORE_ROPE: Self = Self::BeforeRope {
        q: "q_norm",
        k: "k_norm",
    };

    fn names(self) -> (&'static str, &'static str) {
        match self {
            Self::BeforeRope { q, k } | Self::AfterRope { q, k } => (q, k),
        }
    }
}

/// The feed-forward: a gated MLP, the same with `gate_up_proj` fused, or an ungated one.
#[derive(Clone, Copy, Debug, Default)]
pub enum MlpKind {
    #[default]
    Gated,
    /// A fused `gate_up_proj`, split into gate and up for tensor parallelism when the quantization allows.
    MergedGateUp,
    /// A fused `gate_up_proj` kept whole and replicated, one matmul for both halves.
    FusedGateUp,
    /// Up then down, named by the AnyMoE targets, which are those projections.
    Plain {
        projections: &'static [AnyMoeLoraTarget; 2],
        bias: bool,
    },
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
    /// The eps of every norm, whatever its [`NormKind`].
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
    /// The q/k norms' kind, when not the layer norms'.
    pub qk_norm_kind: Option<NormKind>,
    pub norm_names: NormNames,
    pub o_bias: bool,
    /// The attention output projection's name, when not `o_proj`.
    pub o_proj_name: Option<&'static str>,
    /// q, k and v come from one `qkv_proj`; it and the output projection are replicated, not split across ranks.
    pub fused_qkv: bool,
    /// `tanh(scores / cap) * cap` on the attention scores.
    pub attn_softcap: Option<f32>,
    /// The attention score scale; `1 / sqrt(head_dim)` when unset.
    pub softmax_scale: Option<f32>,
    /// `tanh(logits / cap) * cap` on the output logits.
    pub final_logit_softcap: Option<f32>,
    /// Multiplies the token embeddings, as Gemma scales them by `sqrt(hidden_size)`.
    pub embed_scale: Option<f64>,
    pub mlp: MlpKind,
    /// Experts in place of the dense MLP, except on the layers the spec lists as dense.
    pub moe: Option<MoeSpec>,
    pub lm_head_bias: bool,
    /// The lm_head is stored unquantized even in a quantized checkpoint.
    pub unquantized_lm_head: bool,
    /// Eager attention runs in f32 whatever the model's dtype.
    pub eager_attention_f32: bool,
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
    ) -> Result<LayerRope> {
        if let RopeKind::MRope {
            theta,
            sections,
            interleaved,
        } = kind
        {
            let (theta, sections) = (*theta, sections.clone());
            return Ok(if *interleaved {
                LayerRope::InterleavedMRope(Qwen3VLRotaryEmbedding::new(
                    theta,
                    self.head_dim,
                    device,
                    sections,
                )?)
            } else {
                LayerRope::MRope(Qwen2VLRotaryEmbedding::new(
                    theta,
                    self.head_dim,
                    device,
                    sections,
                )?)
            });
        }
        if let RopeKind::Phi(cfg) = kind {
            let factor = |name| {
                vb_m.contains_tensor(name)
                    .then(|| {
                        vb_m.clone()
                            .set_device(device.clone())
                            .get_unchecked_dtype(name, DType::F32)
                    })
                    .transpose()
            };
            let (short, long) = (factor(PHI_ROPE_FACTORS.0)?, factor(PHI_ROPE_FACTORS.1)?);
            return Ok(LayerRope::Phi(PhiRotaryEmbedding::new_with_factors(
                dtype,
                cfg.clone(),
                device,
                short.as_ref(),
                long.as_ref(),
            )?));
        }
        self.plain_rope(kind, vb_m, device, is_gptx, dtype)
            .map(LayerRope::Plain)
    }

    fn plain_rope(
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
                is_gpt_neox.unwrap_or(is_gptx),
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
            RopeKind::Phi(_) | RopeKind::MRope { .. } => unreachable!("built by `rope`"),
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
            mlp: self.mlp,
            lm_head_bias: self.lm_head_bias,
            unquantized_lm_head: self.unquantized_lm_head,
        }
    }
}

/// A layer's RoPE: plain tables, or Phi's LongRoPE, which picks its tables from the batch's positions on the host.
pub enum LayerRope {
    Plain(RotaryEmbedding),
    Phi(PhiRotaryEmbedding),
    MRope(Qwen2VLRotaryEmbedding),
    InterleavedMRope(Qwen3VLRotaryEmbedding),
}

impl LayerRope {
    /// The M-RoPE (cos, sin) for 3D `position_ids`; `None` for the RoPEs that build their own from positions.
    fn mrope_cos_sin(
        &self,
        position_ids: &Tensor,
        dtype: DType,
    ) -> Option<Result<(Tensor, Tensor)>> {
        match self {
            Self::MRope(rope) => Some(rope.compute_cos_sin(position_ids, dtype)),
            Self::InterleavedMRope(rope) => Some(rope.compute_cos_sin(position_ids, dtype)),
            Self::Plain(_) | Self::Phi(_) => None,
        }
    }

    /// Applies the forward's M-RoPE `tables`; `None` for the RoPEs that build their own from positions.
    fn apply_mrope(
        &self,
        tables: &(Tensor, Tensor),
        mut q: Tensor,
        mut k: Tensor,
    ) -> Option<Result<(Tensor, Tensor)>> {
        let applied = match self {
            Self::MRope(rope) => rope.forward(tables, &mut q, &mut k),
            Self::InterleavedMRope(rope) => rope.forward(tables, &mut q, &mut k),
            Self::Plain(_) | Self::Phi(_) => return None,
        };
        Some(applied.map(|()| (q, k)))
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Plain(rope) => rope.forward(q, k, positions),
            Self::Phi(rope) => rope.forward(q, k, positions, position_ids),
            Self::MRope(_) | Self::InterleavedMRope(_) => {
                unreachable!("M-RoPE reads the forward's tables")
            }
        }
    }
}

/// The q/k/v projections: separate and split across ranks, or one replicated `qkv_proj`.
enum QkvProj {
    Split {
        q: Arc<dyn QuantMethod>,
        k: Arc<dyn QuantMethod>,
        v: Arc<dyn QuantMethod>,
    },
    Fused(Arc<dyn QuantMethod>),
}

/// Separate q/k/v/o projections, optional per-head q/k RMS norm, RoPE, and the engine's attention dispatch.
pub struct AttentionBlock {
    qkv: QkvProj,
    o_proj: Arc<dyn QuantMethod>,
    o_proj_name: &'static str,
    qk_norm: Option<(QkNorm, Norm, Norm)>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Option<Arc<LayerRope>>,
    eager_attention_f32: bool,
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

    /// Whether a captured CUDA decode graph replays this layer correctly.
    fn cuda_decode_graphs(&self) -> bool {
        Self::CUDA_DECODE_GRAPHS
    }

    /// The tensors ISQ leaves alone, under `self_attn`.
    fn add_residual(&self, uvb: &UnVarBuilder);

    /// The projections, under `self_attn`, that a MoE-experts-only ISQ also leaves alone.
    fn add_projections(&self, _uvb: &UnVarBuilder) {}
}

/// The feed-forward half of a decoder layer.
pub trait LayerFfn: Send + Sync {
    /// Whether AnyMoE can wrap this feed-forward in experts.
    const AMOE: bool = false;

    fn forward(&self, xs: &Tensor) -> Result<Tensor>;

    /// The feed-forward's name in the layer.
    fn name(&self) -> &'static str {
        MLP
    }

    /// Whether this layer routes over experts, which a MoE-experts-only ISQ quantizes alone.
    fn moe_experts(&self) -> bool {
        false
    }

    /// The tensors ISQ leaves alone, under [`LayerFfn::name`].
    fn add_residual(&self, _uvb: &UnVarBuilder) {}

    /// The tensors, under [`LayerFfn::name`], that a MoE-experts-only ISQ also leaves alone.
    fn add_projections(&self, _uvb: &UnVarBuilder) {}

    /// Whether a captured CUDA decode graph replays this layer correctly.
    fn cuda_decode_graphs(&self) -> bool {
        true
    }

    /// The dense MLP AnyMoE can wrap in experts, when this is one.
    fn as_mlp(&self) -> Option<&dyn MlpLayer> {
        None
    }

    fn as_mlp_mut(&mut self) -> Option<&mut Box<dyn MlpLayer>> {
        None
    }
}

/// A layer's dense MLP or its experts.
pub enum Ffn {
    Dense(Box<dyn MlpLayer>),
    Moe(SparseMoe),
}

impl LayerFfn for Ffn {
    const AMOE: bool = true;

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Dense(_) => MLP,
            Self::Moe(moe) => moe.name(),
        }
    }
    fn moe_experts(&self) -> bool {
        matches!(self, Self::Moe(_))
    }
    fn cuda_decode_graphs(&self) -> bool {
        match self {
            Self::Dense(_) => true,
            Self::Moe(moe) => moe.cuda_decode_graphs(),
        }
    }
    fn add_residual(&self, uvb: &UnVarBuilder) {
        if let Self::Moe(moe) = self {
            moe.add_residual(uvb);
        }
    }
    fn add_projections(&self, uvb: &UnVarBuilder) {
        if let Self::Moe(moe) = self {
            moe.add_projections(uvb);
        }
    }
    fn as_mlp(&self) -> Option<&dyn MlpLayer> {
        match self {
            Self::Dense(mlp) => Some(&**mlp),
            Self::Moe(_) => None,
        }
    }
    fn as_mlp_mut(&mut self) -> Option<&mut Box<dyn MlpLayer>> {
        match self {
            Self::Dense(mlp) => Some(mlp),
            Self::Moe(_) => None,
        }
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
    pub mlp: MlpKind,
    pub lm_head_bias: bool,
    pub unquantized_lm_head: bool,
}

impl StackShape<'_> {
    fn sliding_window(&self) -> Option<usize> {
        self.layer_windows.iter().flatten().next().copied()
    }
}

/// [`AttentionBlock`] and the MLP or experts a [`DecoderSpec`] describes.
struct StandardLayers<'a> {
    spec: &'a DecoderSpec,
    ropes: HashMap<DeviceLocation, Arc<LayerRope>>,
    local_ropes: Option<HashMap<DeviceLocation, Arc<LayerRope>>>,
}

impl LayerBuilder for StandardLayers<'_> {
    type Attention = AttentionBlock;
    type Ffn = Ffn;

    fn build(&self, load: &LayerLoad<'_>, vb: ShardedVarBuilder) -> Result<(AttentionBlock, Ffn)> {
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
        let qc = &spec.quantization_config;
        if let Some(moe) = spec
            .moe
            .as_ref()
            .filter(|moe| !moe.dense_layers.contains(&load.layer_idx))
        {
            let experts = SparseMoe::new(
                moe,
                spec.hidden_size,
                qc,
                spec.hidden_act,
                load,
                place(moe.name),
            )?;
            return Ok((attention, Ffn::Moe(experts)));
        }
        let mlp_kind = match spec.mlp {
            MlpKind::MergedGateUp if qc.as_ref().is_some_and(|qc| !qc.loads_column_shards()) => {
                MlpKind::FusedGateUp
            }
            kind => kind,
        };
        let mlp: Box<dyn MlpLayer> = match mlp_kind {
            MlpKind::FusedGateUp => Box::new(FusedGateUpMlp::new(
                place(MLP),
                spec.hidden_size,
                spec.intermediate_size,
                qc,
                spec.hidden_act,
            )?),
            MlpKind::Gated => Box::new(Mlp::new(
                place(MLP),
                spec.hidden_size,
                spec.intermediate_size,
                &spec.quantization_config,
                spec.hidden_act,
                load.comm,
            )?),
            MlpKind::MergedGateUp => Box::new(Mlp::new_merged(
                place(MLP),
                spec.hidden_size,
                spec.intermediate_size,
                MERGED_GATE_UP_CHUNKS,
                &spec.quantization_config,
                spec.hidden_act,
                load.comm,
            )?),
            MlpKind::Plain { projections, bias } => Box::new(PlainMlp::new(
                place(MLP),
                &[spec.hidden_size, spec.intermediate_size],
                projections,
                bias,
                &spec.quantization_config,
                spec.hidden_act,
                load.comm,
            )?),
        };
        Ok((attention, Ffn::Dense(mlp)))
    }
}

impl AttentionBlock {
    fn new(
        spec: &DecoderSpec,
        load: &LayerLoad<'_>,
        vb: ShardedVarBuilder,
        rotary_emb: Arc<LayerRope>,
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
        let place = |name| mapper.set_device(layer_idx, vb.pp(name), loading_isq);
        let o_proj_name = spec.o_proj_name.unwrap_or(O_PROJ);
        let (q_size, kv_size) = (spec.num_heads * head_dim, spec.num_kv_heads * head_dim);
        let (qkv, o_proj, world_size, n_kv_groups) = if spec.fused_qkv {
            let qkv = inference_quant::linear_b(
                hidden,
                q_size + 2 * kv_size,
                spec.qkv_bias,
                qc,
                place(QKV_PROJ),
            )?;
            let o_proj =
                inference_quant::linear_b(q_size, hidden, spec.o_bias, qc, place(o_proj_name))?;
            let n_kv_groups = spec.num_heads / spec.num_kv_heads;
            (QkvProj::Fused(qkv), o_proj, 1, n_kv_groups)
        } else {
            let q =
                ColumnParallelLayer::new(hidden, q_size, qc, spec.qkv_bias, comm, place("q_proj"))?;
            let kv_shard = inference_quant::compute_kv_shard(spec.num_kv_heads, head_dim, comm)?;
            let kv = |name| {
                ColumnParallelLayer::new_with_shard(
                    hidden,
                    kv_size,
                    qc,
                    spec.qkv_bias,
                    comm,
                    kv_shard,
                    place(name),
                )
            };
            let (k, v) = (kv("k_proj")?, kv("v_proj")?);
            let o_proj =
                RowParallelLayer::new(q_size, hidden, qc, spec.o_bias, comm, place(o_proj_name))?;
            let n_kv_groups =
                inference_quant::compute_n_kv_groups(spec.num_kv_heads, spec.num_heads, comm)?;
            (
                QkvProj::Split { q, k, v },
                o_proj,
                comm.world_size(),
                n_kv_groups,
            )
        };
        let qk_norm = spec
            .qk_norm
            .map(|placement| -> Result<_> {
                let (q, k) = placement.names();
                let norm = |name| {
                    Norm::new(
                        spec.qk_norm_kind.unwrap_or(spec.norm),
                        head_dim,
                        spec.rms_norm_eps,
                        mapper.set_device(layer_idx, vb.pp(name), false),
                    )
                };
                Ok((placement, norm(q)?, norm(k)?))
            })
            .transpose()?;
        let num_heads = spec.num_heads / world_size;
        let num_kv_heads = (spec.num_kv_heads / world_size).max(1);
        Ok(Self {
            qkv,
            o_proj,
            o_proj_name,
            qk_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb: (!spec.no_rope_layers.contains(&layer_idx)).then_some(rotary_emb),
            eager_attention_f32: spec.eager_attention_f32,
            attention_temperature: spec.attention_temperature,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups,
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
        let (q, k, v) = match &self.qkv {
            QkvProj::Split { q, k, v } => crate::ops::qkv_projections(xs, &**q, &**k, &**v)?,
            QkvProj::Fused(qkv) => {
                let qkv = qkv.forward(xs)?;
                let q_size = self.num_heads * self.head_dim;
                let kv_size = self.num_kv_heads * self.head_dim;
                (
                    qkv.narrow(D::Minus1, 0, q_size)?,
                    qkv.narrow(D::Minus1, q_size, kv_size)?,
                    qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?,
                )
            }
        };
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
            Some(kv_cache) if self.eager_attention_f32 && self.paged_attn.is_none() => {
                let (k, v) = kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;
                let f32_mask = match attention_mask {
                    AttentionMask::Custom(mask) => {
                        AttentionMask::Custom(mask.to_dtype(DType::F32)?)
                    }
                    other => other.clone(),
                };
                Sdpa.run_attention(
                    &q.contiguous()?.to_dtype(DType::F32)?,
                    &k.contiguous()?.to_dtype(DType::F32)?,
                    &v.contiguous()?.to_dtype(DType::F32)?,
                    &f32_mask,
                    Some(flash.unwrap_or(ctx.flash_params())),
                    &self.sdpa_params,
                )?
                .to_dtype(q.dtype())?
            }
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
        let Some(rope) = &self.rotary_emb else {
            return Ok((q, k));
        };
        match (&**rope, &self.qk_norm) {
            (LayerRope::MRope(_) | LayerRope::InterleavedMRope(_), None) => {
                let tables = ctx.rope_tables(q.device())?;
                return rope.apply_mrope(tables, q, k).expect("an M-RoPE layer");
            }
            (
                LayerRope::InterleavedMRope(rope),
                Some((QkNorm::BeforeRope { .. }, q_norm, k_norm)),
            ) => {
                let (Some((q_weight, q_eps)), Some((k_weight, k_eps))) =
                    (q_norm.rms_params(), k_norm.rms_params())
                else {
                    inference_tensor::bail!("interleaved M-RoPE fuses only RMS q/k norms");
                };
                let tables = ctx.rope_tables(q.device())?;
                return rope.forward_qk_norm(tables, &q, &k, q_weight, k_weight, q_eps, k_eps);
            }
            (LayerRope::MRope(_) | LayerRope::InterleavedMRope(_), Some(_)) => {
                inference_tensor::bail!(
                    "M-RoPE takes q/k norm only fused before interleaved M-RoPE"
                )
            }
            _ => {}
        }
        let position_ids = match **rope {
            LayerRope::Phi(_) => ctx.position_ids_vec(),
            LayerRope::Plain(_) | LayerRope::MRope(_) | LayerRope::InterleavedMRope(_) => {
                Vec::new()
            }
        };
        let positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| inference_tensor::Error::msg("missing RoPE positions"))?;
        let (q, k) = match &self.qk_norm {
            Some((QkNorm::BeforeRope { .. }, q_norm, k_norm)) => {
                match (&**rope, q_norm.rms_params(), k_norm.rms_params()) {
                    (LayerRope::Plain(rope), Some((q_weight, q_eps)), Some((k_weight, k_eps))) => {
                        rope.forward_qk_norm(&q, &k, q_weight, k_weight, q_eps, k_eps, positions)?
                    }
                    _ => rope.forward(
                        &q_norm.forward(&q)?,
                        &k_norm.forward(&k)?,
                        positions,
                        &position_ids,
                    )?,
                }
            }
            Some((QkNorm::AfterRope { .. }, q_norm, k_norm)) => {
                let (q, k) = rope.forward(&q, &k, positions, &position_ids)?;
                (q_norm.forward(&q)?, k_norm.forward(&k)?)
            }
            None => rope.forward(&q, &k, positions, &position_ids)?,
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
            let (q, k) = placement.names();
            q_norm.add_residual(&uvb.pp(q));
            k_norm.add_residual(&uvb.pp(k));
        }
    }
}

impl LayerAttention for AttentionBlock {
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
    // LongRoPE and M-RoPE pick their tables on the host, which a replayed graph would freeze
    fn cuda_decode_graphs(&self) -> bool {
        !matches!(
            self.rotary_emb.as_deref(),
            Some(LayerRope::Phi(_) | LayerRope::MRope(_) | LayerRope::InterleavedMRope(_))
        )
    }
    fn add_residual(&self, uvb: &UnVarBuilder) {
        self.qk_norm_residual(uvb)
    }
    fn add_projections(&self, uvb: &UnVarBuilder) {
        match &self.qkv {
            QkvProj::Split { q, k, v } => {
                uvb.pp("q_proj").add(q);
                uvb.pp("k_proj").add(k);
                uvb.pp("v_proj").add(v);
            }
            QkvProj::Fused(qkv) => uvb.pp(QKV_PROJ).add(qkv),
        }
        uvb.pp(self.o_proj_name).add(&self.o_proj);
    }
}

/// One layer or final norm of the stack's [`NormKind`].
pub enum Norm {
    Rms(RmsNorm),
    Gemma(GemmaRmsNorm),
    Layer(LayerNorm),
    F32Rms(F32RmsNorm),
}

impl Norm {
    fn new(kind: NormKind, size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(match kind {
            NormKind::Rms => Self::Rms(RmsNorm::new(size, eps, vb)?),
            NormKind::Gemma => Self::Gemma(GemmaRmsNorm::new(size, eps, vb)?),
            NormKind::Layer => Self::Layer(layer_norm(size, eps, vb)?),
            NormKind::F32Rms => Self::F32Rms(F32RmsNorm::new(size, eps, vb)?),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(norm) => norm.forward(xs),
            Self::Gemma(norm) => norm.forward(xs),
            Self::Layer(norm) => norm.forward(xs),
            Self::F32Rms(norm) => norm.forward(xs),
        }
    }

    /// `self(x) + residual`, fused for the RMS norms.
    fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(norm) => norm.forward_residual(x, residual),
            Self::Gemma(norm) => norm.forward_residual(x, residual),
            Self::Layer(_) | Self::F32Rms(_) => self.forward(x)? + residual,
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
            (Self::Layer(_), Self::Layer(_)) | (Self::F32Rms(_), Self::F32Rms(_)) => {
                let xs = self.forward_residual(x, residual)?;
                let normed = next.forward(&xs)?;
                Ok((xs, normed))
            }
            _ => unreachable!("a stack's norms share one kind"),
        }
    }

    fn add_residual(&self, uvb: &UnVarBuilder) {
        match self {
            Self::Rms(norm) => uvb.add(norm),
            Self::Gemma(norm) => uvb.add(norm),
            Self::Layer(norm) => uvb.add(norm),
            Self::F32Rms(norm) => uvb.add(norm),
        }
    }

    /// An RMS norm's multiplier (`1 + weight` for Gemma's) and eps, for the RoPE kernel that fuses it.
    fn rms_params(&self) -> Option<(&Tensor, f64)> {
        match self {
            Self::Rms(norm) => Some((norm.weight(), norm.eps())),
            Self::Gemma(norm) => Some((norm.weight(), norm.eps())),
            Self::Layer(_) | Self::F32Rms(_) => None,
        }
    }
}

/// A layer's norms under [`NormNames`]: pre-norm, or the sandwich with norms after attention and feed-forward too.
struct LayerNorms {
    input: Norm,
    pre_ffn: Option<Norm>,
    post: Option<(Norm, Norm)>,
}

/// Pre-norm residual attention then feed-forward, the sandwich of Gemma 2 and GLM4, or Phi-2's parallel layer.
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
        let normed = self.norms.input.forward(xs)?;
        let xs =
            self.self_attn
                .forward(&normed, attention_mask, kv_cache, ctx, layer_idx, flash)?;
        let Some(pre_ffn) = &self.norms.pre_ffn else {
            return (xs + self.mlp.forward(&normed)?)? + residual;
        };
        let Some((post_attn, post_ffn)) = &self.norms.post else {
            let xs = (xs + residual)?;
            let residual = &xs;
            let xs = self.mlp.forward(&pre_ffn.forward(&xs)?)?;
            return residual + xs;
        };
        let (xs, mlp_in) = post_attn.forward_residual_then_norm(&xs, residual, pre_ffn)?;
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
pub struct DecoderStack<A = AttentionBlock, F = Ffn> {
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
        if let RopeKind::MRope { interleaved, .. } = spec.rope {
            // interleaved M-RoPE fuses an RMS q/k norm before rotating; nothing else rides along
            let fused_qk_norm = interleaved
                && matches!(spec.qk_norm, Some(QkNorm::BeforeRope { .. }))
                && matches!(
                    spec.qk_norm_kind.unwrap_or(spec.norm),
                    NormKind::Rms | NormKind::Gemma
                );
            if spec.attention_temperature.is_some() || (spec.qk_norm.is_some() && !fused_qk_norm) {
                inference_tensor::bail!(
                    "M-RoPE takes no attention temperature, and q/k norm only fused before interleaved M-RoPE"
                );
            }
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
            let post = match (names.post_attn, names.post_ffn, names.pre_ffn) {
                (Some(post_attn), Some(post_ffn), Some(_)) => {
                    Some((norm(post_attn)?, norm(post_ffn)?))
                }
                (None, None, _) => None,
                _ => inference_tensor::bail!("a sandwich layer names both post norms and pre_ffn"),
            };
            Ok(DecoderLayer {
                self_attn,
                mlp,
                norms: LayerNorms {
                    input: norm(names.input)?,
                    pre_ffn: names.pre_ffn.map(norm).transpose()?,
                    post,
                },
            })
        })?;
        let norm = Norm::new(
            shape.norm,
            shape.hidden_size,
            shape.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp(shape.norm_names.last), false),
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
        xs: Tensor,
        masks: &LayerMasks,
        cache: Option<&mut [KvCache]>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        self.forward_hooked(xs, masks, cache, ctx, &|_, xs| Ok(xs))
    }

    /// As [`DecoderStack::forward`], passing each layer's output through `after_layer` with the layer's index.
    pub fn forward_hooked(
        &self,
        mut xs: Tensor,
        masks: &LayerMasks,
        mut cache: Option<&mut [KvCache]>,
        ctx: &mut ModelForwardContext<'_>,
        after_layer: &dyn Fn(usize, Tensor) -> Result<Tensor>,
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
            xs = after_layer(i, xs)?;
        }
        self.norm.forward(&xs.to_device(&self.device)?)
    }

    /// The tensors ISQ leaves alone, under the stack's own prefix; `with_projections` for MoE-experts-only ISQ.
    pub fn residual_uvb(&self, uvb_m: &UnVarBuilder, with_projections: bool) {
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        self.norm.add_residual(&uvb_m.pp(self.norm_names.last));
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            let names = self.norm_names;
            layer.norms.input.add_residual(&uvb_l.pp(names.input));
            if let (Some(pre_ffn), Some(name)) = (&layer.norms.pre_ffn, names.pre_ffn) {
                pre_ffn.add_residual(&uvb_l.pp(name));
            }
            if let (Some((post_attn, post_ffn)), Some(attn_name), Some(ffn_name)) =
                (&layer.norms.post, names.post_attn, names.post_ffn)
            {
                post_attn.add_residual(&uvb_l.pp(attn_name));
                post_ffn.add_residual(&uvb_l.pp(ffn_name));
            }
            layer.self_attn.add_residual(&uvb_l.pp("self_attn"));
            let uvb_ffn = uvb_l.pp(layer.mlp.name());
            layer.mlp.add_residual(&uvb_ffn);
            if with_projections {
                layer.self_attn.add_projections(&uvb_l.pp("self_attn"));
                layer.mlp.add_projections(&uvb_ffn);
            }
        }
    }
}

/// A [`DecoderStack`] under `model.` with an `lm_head`, as the engine runs a text model.
pub struct CausalLm<A = AttentionBlock, F = Ffn> {
    stack: DecoderStack<A, F>,
    lm_head: Arc<dyn QuantMethod>,
    final_logit_softcap: Option<f32>,
    mlp: MlpKind,
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
        let world_size = if spec.fused_qkv {
            1
        } else {
            stack.mapper.get_comm_for(0)?.world_size()
        };
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

impl<F: LayerFfn> CausalLm<AttentionBlock, F> {
    /// This forward's M-RoPE (cos, sin) for `position_ids`, which [`ModelForwardContext::set_rope_tables`] takes.
    pub fn mrope_tables(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let tables = self.stack.layers.iter().find_map(|layer| {
            layer
                .self_attn
                .rotary_emb
                .as_deref()?
                .mrope_cos_sin(position_ids, dtype)
        });
        match tables {
            Some(tables) => tables,
            None => inference_tensor::bail!("the stack has no M-RoPE layer"),
        }
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
            if shape.lm_head_bias {
                inference_tensor::bail!("a tied lm_head with a bias is not supported");
            }
            stack.embed_tokens.clone()
        } else {
            ReplicatedLayer::new(
                shape.hidden_size,
                shape.vocab_size,
                if shape.unquantized_lm_head {
                    &None
                } else {
                    shape.quantization_config
                },
                shape.lm_head_bias,
                stack.mapper.set_nm_device(vb_lm_head, loading_isq),
            )?
        };
        let cache_types = cache_types(shape.layer_windows, max_position_embeddings);
        Ok(Self {
            stack,
            lm_head,
            final_logit_softcap: shape.final_logit_softcap,
            mlp: shape.mlp,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: max_position_embeddings,
            cfg,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.stack.embed(input_ids)
    }

    /// The token embedding, for a multimodal model that embeds its text tokens beside its media.
    pub fn embed_tokens(&self) -> &Arc<dyn QuantMethod> {
        &self.stack.embed_tokens
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
        self.forward_hooked(xs, masks, ctx, &|_, xs| Ok(xs))
    }

    /// As [`CausalLm::forward_with_masks`], with each layer's output through `after_layer` (Qwen3-VL's deepstack).
    pub fn forward_hooked(
        &self,
        xs: Tensor,
        masks: &LayerMasks,
        ctx: &mut ModelForwardContext<'_>,
        after_layer: &dyn Fn(usize, Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let xs = self
            .stack
            .forward_hooked(xs, masks, Some(cache), ctx, after_layer)?;
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
        self.stack
            .layers
            .iter()
            .any(|layer| layer.mlp.moe_experts())
            .then(|| self.residual_uvb(true).to_safetensors())
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
        self.stack
            .layers
            .iter()
            .all(|layer| layer.self_attn.cuda_decode_graphs() && layer.mlp.cuda_decode_graphs())
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
        match self.mlp {
            MlpKind::Plain { projections, .. } => projections,
            MlpKind::Gated => AMOE_LORA_TARGETS,
            MlpKind::MergedGateUp | MlpKind::FusedGateUp => AMOE_MERGED_LORA_TARGETS,
        }
    }
    fn amoe_fine_tuned_expert(
        &self,
        layer: usize,
        base: &dyn MlpLayer,
        vb: ShardedVarBuilder,
    ) -> Result<Box<dyn MlpLayer>> {
        let (dtype, device) = base.dtype_device();
        let vb = vb.set_dtype(dtype).set_device(device);
        let comm = self.stack.mapper.get_comm_for(layer)?;
        Ok(match self.mlp {
            MlpKind::Plain { projections, bias } => Box::new(PlainMlp::new(
                vb,
                base.get_params(),
                projections,
                bias,
                &None,
                base.hidden_act(),
                &comm,
            )?),
            MlpKind::Gated => Box::new(Mlp::replicate(
                base.get_params(),
                vb,
                base.hidden_act(),
                &comm,
            )?),
            MlpKind::MergedGateUp => Box::new(Mlp::new_merged(
                vb,
                base.get_params()[0],
                base.get_params()[1],
                MERGED_GATE_UP_CHUNKS,
                &None,
                base.hidden_act(),
                &comm,
            )?),
            MlpKind::FusedGateUp => Box::new(FusedGateUpMlp::new(
                vb,
                base.get_params()[0],
                base.get_params()[1],
                &None,
                base.hidden_act(),
            )?),
        })
    }
    fn amoe_supported(&self) -> bool {
        F::AMOE
            && self
                .stack
                .layers
                .iter()
                .all(|layer| layer.mlp.as_mlp().is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_gate_up_lora_shapes_follow_peft_in_out_features() {
        let (hidden, intermediate) = (3, 5);
        let shapes: Vec<_> = AMOE_MERGED_LORA_TARGETS
            .iter()
            .map(|t| (t.name, (t.shape)(hidden, intermediate)))
            .collect();
        // (in_features, out_features): gate_up maps hidden -> 2 * intermediate, down maps intermediate -> hidden
        assert_eq!(
            shapes,
            vec![("gate_up_proj", (3, 10)), ("down_proj", (5, 3))]
        );
    }

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
