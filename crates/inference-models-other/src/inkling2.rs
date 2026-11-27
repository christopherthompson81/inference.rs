//! Inkling 2 (`inkling2_loop`): a looped hybrid MoE. Every layer mixes its inputs from the earlier blocks' outputs
//! (block attention residuals), attends with a learned relative-position bias and a per-query sink in place of RoPE,
//! and runs short causal convolutions over K, V and both sublayer outputs; the middle layers run several passes with
//! shared weights, each later pass steered by low-rank controllers from the previous one. Text only; llama.cpp's
//! `src/models/inkling2{,-loop}.cpp` and transformers' `modeling_inkling2.py` are the references.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use inference_quant::{
    QuantMethod, QuantMethodConfig, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder,
    UnquantLinear,
};
use inference_tensor::{D, DType, Device, Module, Result, Tensor};
use serde::Serialize;

use crate::gdn::RecurrentBatchKind;

#[path = "inkling2_speculative.rs"]
mod speculative;
use crate::kv_cache::{
    AuxiliaryRecurrent, EitherCache, HybridCache, HybridCacheConfig, HybridLayerCache,
    HybridLayerType, KvCache, RecurrentLayerConfig, RecurrentStateSpec,
};
use crate::model::{IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{Activation, RmsNorm, embedding_with_legacy_tied_uqff},
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{
        AttentionImplementation, ModelConfigMetadata, PagedAttention, PagedCacheType, PagedKvLayout,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};
pub(crate) use speculative::INKLING_AUTO_DEPTHS;

serde_default_fn!(usize, default_max_position_embeddings, 1_048_576);

// the unweighted norms of the depth mix and the loop controllers
// the quantized cache widths the CUDA attention reads, K and V alike
const QUANT_CACHE_BITS: [u32; 6] = [2, 3, 4, 5, 6, 8];
const UNIT_NORM_EPS: f64 = 1e-6;
// the L2 clip's guard against a zero norm
const CLIP_EPS: f64 = 1e-6;
// the router's guard against a zero sum of the selected scores
const ROUTE_NORM_MIN: f32 = 1e-20;
// the controllers per looped layer, in their message order
const CONTROLLERS: [&str; 4] = ["q", "k", "v", "ffn"];
const FFN_CONTROLLER: usize = 3;
// the projections fused by their input where the checkpoint stores them so (as the GGUF bindings name them)
const Q_GROUP: &str = "q_group";
const K_GROUP: &str = "k_group";
// prefill rows a sliding visit writes and attends at once: a longer chunk goes through in pieces of this
const WINDOW_SUB_CHUNK: usize = 512;
const V_GROUP: &str = "v_group";
// the first special (control) token: the tokenizer's added specials are the vocabulary's tail from here
const CONTROL_TOKENS_START: usize = 199_998;

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    /// Trained over a range of pass counts (TML's varloop: 1 to 5), so any `run_passes` from 1 up runs
    #[serde(default)]
    pub is_varloop: bool,
    /// Passes to run, `num_passes` when unset: fewer trade quality for speed, more (varloop only) the reverse
    #[serde(default)]
    pub run_passes: Option<usize>,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: f64,
    pub log_scaling_alpha: f64,
    pub rms_norm_eps: f64,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,
    pub local_layer_ids: Vec<usize>,
    pub sconv_kernel_size: usize,
    pub final_logit_softcapping: f64,
    pub sliding_window_size: usize,
    pub n_layers_per_attn_res_block: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub intermediate_size: usize,
    pub mlp_gate_softcap: f64,
    pub mlp_up_softcap: f64,
    pub route_scale: f64,
    pub loop_config: LoopConfig,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// LoopCD-Logits: the coda also runs on the first pass's output, in cache slots of its own, so loop guidance
    /// contrasts the two passes' logits (adaptively) rather than their hidden states
    #[serde(default)]
    pub loop_guidance_logits: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    fn is_sliding(&self, layer: usize) -> bool {
        self.local_layer_ids.contains(&layer)
    }

    fn core_len(&self) -> usize {
        self.loop_config.core_end - self.loop_config.core_start
    }

    /// The passes of the looped core a forward runs.
    pub fn passes(&self) -> usize {
        self.loop_config
            .run_passes
            .unwrap_or(self.loop_config.num_passes)
    }

    fn coda_len(&self) -> usize {
        self.num_hidden_layers - self.loop_config.core_end
    }

    /// The layer visits a forward makes: every layer once and the core again for each later pass.
    pub fn visits(&self) -> usize {
        self.num_hidden_layers + (self.passes() - 1) * self.core_len()
    }

    /// The cache slots: a visit's each (a layer's first pass in its own, each later pass of the looped core in one
    /// after them), then with LoopCD-Logits the first pass's coda in its own.
    pub fn cache_slots(&self) -> usize {
        self.visits()
            + if self.loop_guidance_logits {
                self.coda_len()
            } else {
                0
            }
    }

    /// A layer visit of the trained layout (`num_passes` passes) as the run one's: the last trained pass and any
    /// later than it runs map to its last, the coda moves by the passes it skips or adds.
    pub fn run_visit(&self, trained: usize) -> usize {
        let (start, len) = (self.loop_config.core_start, self.core_len());
        let (trained_passes, passes) = (self.loop_config.num_passes, self.passes());
        let trained_core_end = start + trained_passes * len;
        if trained < start {
            trained
        } else if trained < trained_core_end {
            let (pass, layer) = ((trained - start) / len, (trained - start) % len);
            let pass = if pass + 1 == trained_passes {
                passes - 1
            } else {
                pass.min(passes - 1)
            };
            start + pass * len + layer
        } else {
            trained - trained_passes * len + passes * len
        }
    }

    /// The slots whose visits attend over a sliding window.
    pub fn sliding_slots(&self) -> Vec<usize> {
        (0..self.cache_slots())
            .filter(|&slot| self.is_sliding(physical_layer(self, slot)))
            .collect()
    }

    /// Each sliding slot's window ring for a paged cache of this layout on `device`, or `None` where the windows stay
    /// in the paged cache (off CUDA, without one, or a cache type the attention kernel doesn't read in a ring).
    pub fn window_ring(
        &self,
        paged: Option<PagedKvLayout>,
        device: &Device,
        act_dtype: DType,
    ) -> Option<WindowRing> {
        let paged = paged.filter(|_| device.is_cuda())?;
        let act = inference_nn::paged_attention::cache_activation_dtype(act_dtype, device);
        let (dtype, row) = paged.cache_type.row_shape(self.head_dim, act)?;
        // the window, a prefill sub-chunk written before it attends, and a block's slack for the window's start
        let tokens = self.sliding_window_size + WINDOW_SUB_CHUNK;
        Some(WindowRing {
            blocks: tokens.div_ceil(paged.block_size) + 1,
            block_size: paged.block_size,
            kv_heads: self.num_key_value_heads,
            dtype,
            row,
        })
    }

    /// A sequence's state beside its paged KV: the convolution states and any window rings, in bytes.
    pub fn sequence_state_bytes(&self, ring: Option<&WindowRing>) -> usize {
        self.conv_state_bytes() + ring.map_or(0, |ring| self.sliding_slots().len() * ring.bytes())
    }

    /// The paged cache's view of the model: every slot paged, but sliding ones when they keep rings.
    pub fn paged_cache_config(
        &self,
        base: ModelConfigMetadata,
        ring: Option<&WindowRing>,
    ) -> inference_nn::paged_attention::HybridPagedKvCacheConfig {
        let sliding = ring.map(|_| self.sliding_slots()).unwrap_or_default();
        let paged = (0..self.cache_slots())
            .map(|slot| !sliding.contains(&slot))
            .collect();
        // the biased attention reads the paged cache (or the rings) in place, through the block tables
        inference_nn::paged_attention::HybridPagedKvCacheConfig::new(base, paged)
            .with_recurrent_state_bytes(self.sequence_state_bytes(ring))
            .with_attention_in_place()
    }

    /// A sequence's convolution states over every slot, in bytes (f32).
    pub fn conv_state_bytes(&self) -> usize {
        self.cache_slots()
            * self.conv_channels()
            * (self.sconv_kernel_size - 1)
            * DType::F32.size_in_bytes()
    }

    fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// A slot's convolution channels: K, V, the attention output and the FFN output.
    pub fn conv_state_channels(&self) -> usize {
        self.conv_channels()
    }

    fn conv_channels(&self) -> usize {
        2 * self.kv_dim() + 2 * self.hidden_size
    }

    fn validate(&self) -> Result<()> {
        let lc = &self.loop_config;
        if let Some(run) = lc
            .run_passes
            .filter(|&run| run > lc.num_passes && !lc.is_varloop)
        {
            inference_tensor::bail!(
                "Inkling 2: {run} loop passes, but this model was trained at {} only (a varloop model takes more)",
                lc.num_passes
            );
        }
        let block = self.n_layers_per_attn_res_block;
        if lc.num_passes < 1
            || lc.run_passes.is_some_and(|run| run < 1)
            || lc.core_start == 0
            || lc.core_start >= lc.core_end
            || lc.core_end > self.num_hidden_layers
            || block == 0
            || !lc.core_start.is_multiple_of(block)
            || !lc.core_end.is_multiple_of(block)
            || self.sconv_kernel_size < 2
            || self.unpadded_vocab_size > self.vocab_size
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            inference_tensor::bail!("Inkling 2: invalid config {self:?}");
        }
        Ok(())
    }
}

fn unit_rms(x: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(out) = inference_nn::cuda::inkling::unit_rms(x, eps as f32)? {
        return Ok(out);
    }
    let x = x.to_dtype(DType::F32)?;
    if x.device().is_vulkan() {
        let mode = inference_tensor::vulkan::RowNormMode::Rms { eps: eps as f32 };
        if let Some(out) = inference_tensor::vulkan::row_norm(&x.contiguous()?, None, mode)? {
            return Ok(out);
        }
    }
    let norm = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&norm)
}

// the dtype a paged cache's writer takes its rows in: the cache's, or half precision for a quantized one, which the
// writer quantizes itself
#[cfg(feature = "cuda")]
fn paged_rows_dtype(cache: DType) -> DType {
    match cache {
        DType::F32 | DType::F16 | DType::BF16 => cache,
        _ => DType::BF16,
    }
}

// each last-axis row (after the optional per-row scale) down to an L2 norm of at most `threshold`
fn l2_clip(x: &Tensor, scale: Option<&Tensor>, threshold: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(out) =
        inference_nn::cuda::inkling::l2_clip(x, scale, threshold as f32, CLIP_EPS as f32)?
    {
        return Ok(out);
    }
    let x = x.to_dtype(DType::F32)?;
    if x.device().is_vulkan() {
        let mode = inference_tensor::vulkan::RowNormMode::Clip {
            threshold: threshold as f32,
            eps: CLIP_EPS as f32,
        };
        if let Some(out) = inference_tensor::vulkan::row_norm(&x.contiguous()?, scale, mode)? {
            return Ok(out);
        }
    }
    let x = match scale {
        Some(scale) => x.broadcast_mul(scale)?,
        None => x,
    };
    let norm = (x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()? + CLIP_EPS)?;
    let factor = norm
        .recip()?
        .affine(threshold, 0.0)?
        .clamp(0.0f32, 1.0f32)?;
    x.broadcast_mul(&factor)
}

fn linear(
    in_dim: usize,
    out_dim: usize,
    cfg: &Config,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    ReplicatedLayer::new(in_dim, out_dim, &cfg.quantization_config, false, vb)
}

/// The latest block stack for the depth mixes, by the stacked tensors' ids: a block's layers read the same blocks, so
/// they are stacked once. Only the latest is kept: every stack kept would hold n(n+1)/2 block states a prompt token.
// read by the CUDA mix only
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Default)]
struct BlockStacks {
    stacks: std::cell::RefCell<Vec<(Vec<inference_tensor::TensorId>, Tensor)>>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl BlockStacks {
    fn stacked(&self, blocks: &[&Tensor]) -> Result<Tensor> {
        let ids = blocks.iter().map(|b| b.id()).collect::<Vec<_>>();
        if let Some((_, hit)) = self.stacks.borrow().iter().find(|(k, _)| *k == ids) {
            return Ok(hit.clone());
        }
        let stacked = Tensor::stack(blocks, 0)?;
        *self.stacks.borrow_mut() = vec![(ids, stacked.clone())];
        Ok(stacked)
    }
}

/// The earlier blocks' outputs a layer mixes its inputs from, each `[b, t, hidden]`, with their stack (`[b * t, n,
/// hidden]`) kept for Vulkan's fused mix until the next push.
#[derive(Clone, Default)]
struct Blocks {
    states: Vec<Tensor>,
    stacked: Option<Tensor>,
}

impl Blocks {
    fn new(states: Vec<Tensor>) -> Self {
        Self {
            states,
            stacked: None,
        }
    }

    fn push(&mut self, state: Tensor) {
        self.states.push(state);
        self.stacked = None;
    }

    fn stacked(&mut self) -> Result<Option<Tensor>> {
        if self.stacked.is_none() && !self.states.is_empty() {
            let (b, t, hidden) = self.states[0].dims3()?;
            let stacked = Tensor::stack(&self.states, 2)?.to_dtype(DType::F32)?;
            self.stacked = Some(stacked.reshape((b * t, self.states.len(), hidden))?);
        }
        Ok(self.stacked.clone())
    }
}

/// The depth mix: softmax attention of learned queries over the unit-normed blocks and `cur`, `[b, t, queries,
/// hidden]`.
fn mix(
    blocks: &mut Blocks,
    cur: &Tensor,
    queries: &Tensor,
    out: DType,
    stacks: &BlockStacks,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if cur.device().is_cuda() {
        let states = blocks.states.iter().collect::<Vec<_>>();
        let stacked = (!states.is_empty())
            .then(|| stacks.stacked(&states))
            .transpose()?;
        if let Some(mixed) = inference_nn::cuda::inkling::depth_mix(
            stacked.as_ref(),
            cur,
            queries,
            UNIT_NORM_EPS as f32,
            out,
        )? {
            return Ok(mixed);
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = stacks;
    let (b, t, hidden) = cur.dims3()?;
    if cur.device().is_vulkan() {
        let flat = cur.to_dtype(DType::F32)?.reshape((b * t, hidden))?;
        let stacked = blocks.stacked()?;
        if let Some(mixed) = inference_tensor::vulkan::depth_mix(
            stacked.as_ref(),
            &flat,
            queries,
            UNIT_NORM_EPS as f32,
        )? {
            return mixed
                .reshape((b, t, queries.dim(0)?, hidden))?
                .to_dtype(out);
        }
    }
    let owned = blocks
        .states
        .iter()
        .chain(std::iter::once(cur))
        // a block's output is its weights' dtype (BF16 from unquantized ones, F32 from quantized) and on its device
        .map(|state| state.to_device(cur.device())?.to_dtype(DType::F32))
        .collect::<Result<Vec<_>>>()?;
    let states = &owned.iter().collect::<Vec<_>>();
    let stacked = Tensor::stack(states, 2)?;
    let stacked = unit_rms(&stacked, UNIT_NORM_EPS)?.reshape((b * t, states.len(), hidden))?;
    let scores = stacked.broadcast_matmul(&queries.t()?)?;
    let scores = (scores / (hidden as f64).sqrt())?;
    let weights = inference_tensor::nn::ops::softmax(&scores, 1)?;
    weights
        .transpose(1, 2)?
        .contiguous()?
        .matmul(&stacked)?
        .reshape((b, t, queries.dim(0)?, hidden))?
        .to_dtype(out)
}

fn mixed(
    blocks: &mut Blocks,
    cur: &Tensor,
    queries: &Tensor,
    i: usize,
    out: DType,
    stacks: &BlockStacks,
) -> Result<Tensor> {
    mix(blocks, cur, queries, out, stacks)?
        .narrow(2, i, 1)?
        .squeeze(2)
}

/// Projections of one input: one product over their rows when the checkpoint stores them concatenated (a GGUF's
/// bindings do), each its own otherwise.
enum ProjGroup {
    Fused(Arc<dyn QuantMethod>, Vec<usize>),
    Separate(Vec<Arc<dyn QuantMethod>>),
}

impl ProjGroup {
    fn load(
        cfg: &Config,
        vb: &ShardedVarBuilder,
        fused: &str,
        parts: &[(&str, usize)],
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let rows = parts.iter().map(|(_, rows)| *rows).collect::<Vec<_>>();
        if vb.contains_tensor(&format!("{fused}.weight")) {
            let total = rows.iter().sum();
            return Ok(Self::Fused(linear(hidden, total, cfg, vb.pp(fused))?, rows));
        }
        let parts = parts
            .iter()
            .map(|(name, rows)| linear(hidden, *rows, cfg, vb.pp(*name)))
            .collect::<Result<_>>()?;
        Ok(Self::Separate(parts))
    }

    fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        match self {
            Self::Fused(proj, rows) => {
                let all = proj.forward(x)?;
                let mut start = 0;
                rows.iter()
                    .map(|&len| {
                        let part = all.narrow(D::Minus1, start, len);
                        start += len;
                        part
                    })
                    .collect()
            }
            Self::Separate(parts) => parts.iter().map(|proj| proj.forward(x)).collect(),
        }
    }
}

/// A sliding slot's window on CUDA: each sequence's ring of `blocks` paged-cache blocks for K, then as many for V,
/// kept as an auxiliary recurrent state so a prefix snapshot holds it. Key block k lives in ring block k % blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowRing {
    pub blocks: usize,
    pub block_size: usize,
    pub kv_heads: usize,
    pub dtype: DType,
    // a row's elements (bytes for a quantized cache)
    pub row: usize,
}

impl WindowRing {
    /// A sequence's K and V rings as the state's dims: `[2 * blocks, kv_heads, block_size, row]`.
    pub fn dims(&self) -> Vec<usize> {
        vec![2 * self.blocks, self.kv_heads, self.block_size, self.row]
    }

    pub fn bytes(&self) -> usize {
        self.dims().iter().product::<usize>() * self.dtype.size_in_bytes()
    }
}

/// A slot's convolution states: its rows gathered from the pool (`[b, channels, hist]`, a fresh copy), or the pool
/// itself and the batch's rows of it, which a Vulkan decode step updates in place. A verify step's rows also collect
/// each conv's input, in the states' channel order, for its rollback.
#[derive(Clone, Copy)]
enum ConvState<'a> {
    Rows {
        rows: &'a Tensor,
        inputs: Option<&'a ConvInputs>,
    },
    Pool {
        pool: &'a Tensor,
        slots: &'a Tensor,
        rollback: Option<&'a ConvRollback>,
    },
}

/// A verify step's states as read `[b, channels, hist]` F32 and its conv inputs `[b, t, channels]` in x's dtype, the
/// CUDA pool kernel's to fill.
struct ConvRollback {
    before: Tensor,
    stash: Tensor,
}

/// A slot's conv inputs `[b, t, channels]` as its convs take them, in their dtype (the rollback widens them to F32).
type ConvInputs = std::cell::RefCell<Vec<Tensor>>;

/// One conv's channels of a slot's states.
enum ConvPart<'a> {
    Rows {
        state: Tensor,
        inputs: Option<&'a ConvInputs>,
    },
    Pool {
        pool: &'a Tensor,
        slots: &'a Tensor,
        channel_off: usize,
        rollback: Option<&'a ConvRollback>,
    },
}

impl<'a> ConvState<'a> {
    fn part(self, channel_off: usize, channels: usize) -> Result<ConvPart<'a>> {
        Ok(match self {
            Self::Rows { rows, inputs } => ConvPart::Rows {
                state: rows.narrow(1, channel_off, channels)?,
                inputs,
            },
            Self::Pool {
                pool,
                slots,
                rollback,
            } => ConvPart::Pool {
                pool,
                slots,
                channel_off,
                rollback,
            },
        })
    }
}

/// A causal depthwise convolution with a residual, its last `width - 1` inputs carried in `[b, c, width - 1]`
/// states; the global layers' input-dependent taps (a lag each, `heads` sets of them) are shared within a head.
struct ShortConv {
    // `[c, width]`, the last tap on the current input
    weight: Tensor,
    tap_heads: Option<usize>,
}

impl ShortConv {
    fn load(channels: usize, width: usize, vb: &ShardedVarBuilder) -> Result<Tensor> {
        vb.get((channels, width), "weight")?.to_dtype(DType::F32)
    }

    // `out_dtype`: the output's, f32 or x's; the new states from gathered rows (a pool updates in place)
    fn forward(
        &self,
        x: &Tensor,
        taps: Option<&Tensor>,
        state: ConvPart<'_>,
        out_dtype: DType,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (b, t, c) = x.dims3()?;
        let width = self.weight.dim(1)?;
        let taps = match (self.tap_heads, taps) {
            (Some(heads), Some(taps)) => Some((
                taps.to_dtype(DType::F32)?.reshape((b, t, heads, width))?,
                heads,
            )),
            _ => None,
        };
        let terms = 1 + width + taps.as_ref().map_or(0, |_| width);
        let state = match state {
            ConvPart::Rows { state, inputs } => {
                if let Some(inputs) = inputs {
                    inputs.borrow_mut().push(x.clone());
                }
                state
            }
            ConvPart::Pool {
                pool,
                slots,
                channel_off,
                rollback,
            } => {
                #[cfg(feature = "cuda")]
                if x.device().is_cuda() {
                    let op = inference_nn::cuda::inkling::ShortConvPool {
                        x,
                        pool,
                        slots,
                        channel_off,
                        weight: &self.weight,
                        taps: taps.as_ref().map(|(taps, heads)| (taps, *heads)),
                        out_like_x: out_dtype != DType::F32,
                        before: rollback.map(|r| &r.before),
                        stash: rollback.map(|r| &r.stash),
                    };
                    let Some(out) = inference_nn::cuda::inkling::short_conv_pool(&op)? else {
                        inference_tensor::bail!(
                            "Inkling 2: the pool convolution does not take this layout"
                        );
                    };
                    return Ok((out, None));
                }
                let _ = rollback;
                let taps = taps
                    .as_ref()
                    .map(|(taps, heads)| taps.reshape((b, *heads, width))?.contiguous())
                    .transpose()?;
                let out = inference_tensor::vulkan::short_conv_step(
                    &inference_tensor::vulkan::ShortConvStep {
                        x: &x.to_dtype(DType::F32)?.reshape((b, c))?.contiguous()?,
                        weight: &self.weight,
                        taps: taps.as_ref(),
                        pool,
                        channel_off,
                        slots,
                        scale: (1.0 / (terms as f64).sqrt()) as f32,
                    },
                )?;
                return Ok((out.reshape((b, t, c))?.to_dtype(out_dtype)?, None));
            }
        };
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() {
            let taps = taps.as_ref().map(|(taps, heads)| (taps, *heads));
            if let Some((out, new_state)) = inference_nn::cuda::inkling::short_conv(
                x,
                &state,
                &self.weight,
                taps,
                out_dtype != DType::F32,
            )? {
                return Ok((out, Some(new_state)));
            }
        }
        let x = x.to_dtype(DType::F32)?;
        let ctx = Tensor::cat(&[&state.to_dtype(DType::F32)?.transpose(1, 2)?, &x], 1)?;
        let mut out = x.clone();
        for k in 0..width {
            let w = self.weight.narrow(1, k, 1)?.squeeze(1)?;
            out = (out + ctx.narrow(1, k, t)?.broadcast_mul(&w)?)?;
        }
        if let Some((taps, heads)) = &taps {
            for lag in 0..width {
                let window =
                    ctx.narrow(1, width - 1 - lag, t)?
                        .reshape((b, t, *heads, c / heads))?;
                let tap = taps.narrow(3, lag, 1)?;
                out = (out + window.broadcast_mul(&tap)?.reshape((b, t, c))?)?;
            }
        }
        let new_state = ctx.narrow(1, t, width - 1)?.transpose(1, 2)?.contiguous()?;
        Ok((
            (out / (terms as f64).sqrt())?.to_dtype(out_dtype)?,
            Some(new_state),
        ))
    }
}

struct Attention {
    // q, gate, r and the sinks; k and v, each with the global layers' convolution taps
    q_group: ProjGroup,
    k_group: ProjGroup,
    v_group: ProjGroup,
    o_proj: Arc<dyn QuantMethod>,
    // `[heads, head_dim]` and `[heads, d_rel]`
    q_clip: Tensor,
    rel_clip: Tensor,
    // `[d_rel, extent + 1]`, the last column zero: the bias past the extent
    rel_bias: Tensor,
    k_conv: ShortConv,
    v_conv: ShortConv,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    d_rel: usize,
    extent: usize,
    sliding: bool,
}

/// The positions a layer kind's queries see: an additive mask (`[t, s]`, 0 or -inf) and each pair's bias column.
struct RelativeInputs {
    mask: Tensor,
    // `[t, s]` u32, the distance where it is below the extent, else the extent (the zero column)
    columns: Tensor,
}

impl RelativeInputs {
    fn new(
        past: usize,
        t: usize,
        extent: usize,
        window: Option<usize>,
        device: &Device,
    ) -> Result<Self> {
        let s = past + t;
        let mut mask = vec![0f32; t * s];
        let mut columns = vec![extent as u32; t * s];
        for i in 0..t {
            let pos = past + i;
            for j in 0..s {
                let at = i * s + j;
                let visible = j <= pos && window.is_none_or(|w| pos - j < w);
                if !visible {
                    mask[at] = f32::NEG_INFINITY;
                } else if pos - j < extent {
                    columns[at] = (pos - j) as u32;
                }
            }
        }
        Ok(Self {
            mask: Tensor::from_vec(mask, (t, s), device)?,
            columns: Tensor::from_vec(columns, (t, s), device)?,
        })
    }
}

struct AttentionInputs<'a> {
    q: &'a Tensor,
    k: &'a Tensor,
    v: &'a Tensor,
    // `[b, t, 1, 1]`, the global layers' log scale of q and r; `None` where the fused CUDA attention scales itself
    tau: Option<&'a Tensor>,
}

/// Where a paged step's rows go and what its queries see: the slot mapping, each sequence's block table row, and
/// its past (`lens - minus`, u32 on the device, so a captured decode graph reads them).
// read by the CUDA attention only; a build without it never runs paged
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct PagedPositions {
    slots: Tensor,
    tables: Tensor,
    lens: Tensor,
    minus: usize,
}

/// A slot's paged cache for one step (CUDA, the fused attention).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct PagedStep<'a> {
    k_cache: &'a Tensor,
    v_cache: &'a Tensor,
    positions: &'a PagedPositions,
    writer: &'a PagedAttention,
    // a sliding slot's window ring in place of the paged cache: the caches above are its K and V views
    ring: Option<(&'a RingRows, usize)>,
}

/// A step's rows in the window rings, alike for every sliding slot: each batch row's first K ring block (the
/// attention's table) and, per prefill piece of at most WINDOW_SUB_CHUNK rows, its offset, rows, ring slots and the
/// pasts its queries start from.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct RingRows {
    bases: Tensor,
    pieces: Vec<RingPiece>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct RingPiece {
    offset: usize,
    rows: usize,
    slots: Tensor,
    lens: Tensor,
}

#[cfg(feature = "cuda")]
impl RingRows {
    // `capacity`: the ring pools' rows, past which a state slot marks a padded row
    fn new(
        ring: &WindowRing,
        positions: &PagedPositions,
        states: &Tensor,
        capacity: usize,
        t: usize,
    ) -> Result<Self> {
        let states = states.to_dtype(DType::U32)?;
        let mut bases = None;
        let mut pieces = Vec::new();
        for offset in (0..t).step_by(WINDOW_SUB_CHUNK) {
            let rows = WINDOW_SUB_CHUNK.min(t - offset);
            let (slots, first) = inference_nn::cuda::inkling::ring_slots(
                &inference_nn::cuda::inkling::RingSlots {
                    states: &states,
                    capacity,
                    lens: &positions.lens,
                    minus: positions.minus,
                    offset,
                    slots: &positions.slots,
                    slot_stride: t,
                    rows,
                    ring: ring.blocks,
                    block_size: ring.block_size,
                },
                offset == 0,
            )?;
            bases = bases.or(first);
            // a later piece's queries start that many rows on (only a prefill has more than one, and no minus)
            let lens = if offset == 0 {
                positions.lens.clone()
            } else {
                (positions.lens.to_dtype(DType::F32)? + offset as f64)?.to_dtype(DType::U32)?
            };
            pieces.push(RingPiece {
                offset,
                rows,
                slots,
                lens,
            });
        }
        Ok(Self {
            bases: bases
                .ok_or_else(|| inference_tensor::Error::msg("Inkling 2: a step without rows"))?,
            pieces,
        })
    }
}

impl Attention {
    fn new(cfg: &Config, layer: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let sliding = cfg.is_sliding(layer);
        let (h, hkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let hidden = cfg.hidden_size;
        let extent = if sliding {
            cfg.sliding_window_size
        } else {
            cfg.rel_extent
        };
        let width = cfg.sconv_kernel_size;
        let rel_bias = vb
            .pp("rel_logits_proj")
            .get((cfg.d_rel, extent), "proj")?
            .to_dtype(DType::F32)?;
        let rel_bias = Tensor::cat(
            &[
                &rel_bias,
                &Tensor::zeros((cfg.d_rel, 1), DType::F32, rel_bias.device())?,
            ],
            1,
        )?;
        let conv = |name: &str| -> Result<ShortConv> {
            Ok(ShortConv {
                weight: ShortConv::load(hkv * hd, width, &vb.pp(name))?,
                tap_heads: (!sliding).then_some(hkv),
            })
        };
        let kv_group = |group: &str, proj: &str, conv: &str| -> Result<ProjGroup> {
            let taps = format!("{conv}.weight_proj");
            let mut parts = vec![(proj, hkv * hd)];
            if !sliding {
                parts.push((taps.as_str(), hkv * width));
            }
            ProjGroup::load(cfg, &vb, group, &parts)
        };
        let q_parts = [
            ("q_proj", h * hd),
            ("gate_proj", h * hd),
            ("r_proj", h * cfg.d_rel),
            ("sink_proj", h),
        ];
        Ok(Self {
            q_group: ProjGroup::load(cfg, &vb, Q_GROUP, &q_parts)?,
            k_group: kv_group(K_GROUP, "k_proj", "k_sconv")?,
            v_group: kv_group(V_GROUP, "v_proj", "v_sconv")?,
            o_proj: linear(h * hd, hidden, cfg, vb.pp("o_proj"))?,
            q_clip: vb
                .pp("q_l2_clip")
                .get((h, hd), "scale")?
                .to_dtype(DType::F32)?,
            rel_clip: vb
                .pp("rel_l2_clip")
                .get((h, cfg.d_rel), "scale")?
                .to_dtype(DType::F32)?,
            rel_bias,
            k_conv: conv("k_sconv")?,
            v_conv: conv("v_sconv")?,
            heads: h,
            kv_heads: hkv,
            head_dim: hd,
            d_rel: cfg.d_rel,
            extent,
            sliding,
        })
    }

    fn forward(
        &self,
        cfg: &Config,
        inputs: AttentionInputs<'_>,
        rel: Option<&RelativeInputs>,
        kv_cache: &mut KvCache,
        paged: Option<&PagedStep<'_>>,
        conv_state: ConvState<'_>,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        let (b, t, _) = inputs.q.dims3()?;
        let (h, hkv, hd) = (self.heads, self.kv_heads, self.head_dim);
        let kv_dim = hkv * hd;
        let k_state = conv_state.part(0, kv_dim)?;
        let v_state = conv_state.part(kv_dim, kv_dim)?;

        let [q, raw_gate, r, sinks] = <[Tensor; 4]>::try_from(self.q_group.forward(inputs.q)?)
            .map_err(|_| inference_tensor::Error::msg("Inkling 2: the q group has four parts"))?;
        let q = q.reshape((b, t, h, hd))?;
        let k = self.k_group.forward(inputs.k)?;
        let (k, k_state) = self.k_conv.forward(&k[0], k.get(1), k_state, DType::F32)?;
        let v = self.v_group.forward(inputs.v)?;
        let (v, v_state) = self.v_conv.forward(&v[0], v.get(1), v_state, DType::F32)?;
        let r = r.reshape((b, t, h, self.d_rel))?;
        let raw_gate = raw_gate.contiguous()?;
        let gate = || raw_gate.to_dtype(DType::F32);
        let (k, v) = (k.reshape((b, t, hkv, hd))?, v.reshape((b, t, hkv, hd))?);
        // a paged CUDA step whose log scale the attention applies: the clips, the norm and the casts in one launch,
        // k and v in the cache writer's dtype
        #[cfg(feature = "cuda")]
        let rows = match paged {
            Some(step) if self.sliding || inputs.tau.is_none() => {
                inference_nn::cuda::inkling::attention_rows(
                    &inference_nn::cuda::inkling::AttentionRows {
                        q: &q,
                        q_scale: &self.q_clip,
                        r: &r,
                        r_scale: &self.rel_clip,
                        k: &k,
                        v: &v,
                        sinks: &sinks,
                        qk_threshold: cfg.qk_l2_clip_threshold as f32,
                        rel_threshold: cfg.rel_l2_clip_threshold as f32,
                        clip_eps: CLIP_EPS as f32,
                        rms_eps: cfg.rms_norm_eps as f32,
                        kv_dtype: paged_rows_dtype(step.k_cache.dtype()),
                    },
                )?
            }
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let rows: Option<[Tensor; 5]> = None;
        let (mut q, k, v, mut r, sinks) = match rows {
            Some([q, r, k, v, sinks]) => (q, k, v, r, sinks),
            None => (
                l2_clip(&q, Some(&self.q_clip), cfg.qk_l2_clip_threshold)?,
                l2_clip(&k, None, cfg.qk_l2_clip_threshold)?,
                unit_rms(&v, cfg.rms_norm_eps)?,
                l2_clip(&r, Some(&self.rel_clip), cfg.rel_l2_clip_threshold)?,
                sinks.to_dtype(DType::F32)?,
            ),
        };
        // the log scale of q and r is the score's (bar the sink's), which the fused CUDA attention applies itself
        if !self.sliding
            && let Some(tau) = inputs.tau
        {
            q = q.broadcast_mul(tau)?;
            r = r.broadcast_mul(tau)?;
        }
        #[cfg(feature = "cuda")]
        let log_scale = (!self.sliding)
            .then_some((cfg.log_scaling_alpha as f32, cfg.log_scaling_n_floor as f32));
        let window = if self.sliding {
            cfg.sliding_window_size
        } else {
            0
        };
        let scale = 1.0 / (hd as f64).sqrt();
        // each query's bias over the distances, `[b, t, h, extent + 1]`, the last column zero (past the extent)
        let token_table = r
            .reshape((b * t * h, self.d_rel))?
            .matmul(&self.rel_bias)?
            .reshape((b, t, h, self.extent + 1))?;
        #[cfg(feature = "cuda")]
        let output_gate = |out: Tensor| -> Result<Tensor> {
            if let Some(gated) =
                inference_nn::cuda::inkling::sigmoid_gate(&out, &raw_gate, inputs.q.dtype())?
            {
                return self.o_proj.forward(&gated);
            }
            let out = out.broadcast_mul(&inference_tensor::nn::ops::sigmoid(&gate()?)?)?;
            self.o_proj.forward(&out.to_dtype(inputs.q.dtype())?)
        };

        // paged: this step's rows into the cache, the queries over it through the block tables
        if let Some(step) = paged {
            #[cfg(feature = "cuda")]
            if let Some((ring_rows, blocks)) = step.ring {
                use inference_nn::cuda::biased_attention::{Keys, Past};
                let dtype = step.k_cache.dtype();
                let rows_dtype = paged_rows_dtype(dtype);
                let (k, v) = (k.to_dtype(rows_dtype)?, v.to_dtype(rows_dtype)?);
                let mut outs = Vec::with_capacity(ring_rows.pieces.len());
                // each piece's rows go into the ring before its queries read it, so the ring holds a piece past the
                // window, not a whole chunk
                for piece in &ring_rows.pieces {
                    let (o, n) = (piece.offset, piece.rows);
                    step.writer.write_rows(
                        &k.narrow(1, o, n)?.transpose(1, 2)?,
                        &v.narrow(1, o, n)?.transpose(1, 2)?,
                        step.k_cache,
                        step.v_cache,
                        &piece.slots,
                    )?;
                    let q = q.narrow(1, o, n)?.contiguous()?;
                    let table = token_table.narrow(1, o, n)?.contiguous()?;
                    let sinks = sinks.narrow(1, o, n)?.contiguous()?;
                    let op = inference_nn::cuda::biased_attention::BiasedAttention {
                        q: &q,
                        keys: Keys::Paged {
                            k_cache: step.k_cache,
                            v_cache: step.v_cache,
                            block_tables: &ring_rows.bases,
                            ring: blocks,
                        },
                        past: Past::Lens {
                            lens: &piece.lens,
                            minus: step.positions.minus,
                        },
                        table: &table,
                        sinks: &sinks,
                        window,
                        scale: scale as f32,
                        log_scale,
                    };
                    outs.push(inference_nn::cuda::biased_attention::biased_attention(&op)?.ok_or_else(|| {
                        inference_tensor::Error::msg(format!(
                            "Inkling 2: the windowed attention does not take a {dtype:?} ring"
                        ))
                    })?);
                }
                let out = if outs.len() == 1 {
                    outs.remove(0)
                } else {
                    Tensor::cat(&outs, 1)?
                };
                return Ok((output_gate(out)?, k_state, v_state));
            }
            #[cfg(feature = "cuda")]
            {
                use inference_nn::cuda::biased_attention::{Keys, Past};
                let dtype = step.k_cache.dtype();
                let rows_dtype = paged_rows_dtype(dtype);
                step.writer.write_rows(
                    &k.to_dtype(rows_dtype)?.transpose(1, 2)?,
                    &v.to_dtype(rows_dtype)?.transpose(1, 2)?,
                    step.k_cache,
                    step.v_cache,
                    &step.positions.slots,
                )?;
                let op = inference_nn::cuda::biased_attention::BiasedAttention {
                    q: &q,
                    keys: Keys::Paged {
                        k_cache: step.k_cache,
                        v_cache: step.v_cache,
                        block_tables: &step.positions.tables,
                        ring: 0,
                    },
                    past: Past::Lens {
                        lens: &step.positions.lens,
                        minus: step.positions.minus,
                    },
                    table: &token_table,
                    sinks: &sinks,
                    window,
                    scale: scale as f32,
                    log_scale,
                };
                let out = inference_nn::cuda::biased_attention::biased_attention(&op)?.ok_or_else(
                    || {
                        inference_tensor::Error::msg(format!(
                            "Inkling 2: the paged attention does not take a {dtype:?} cache"
                        ))
                    },
                )?;
                return Ok((output_gate(out)?, k_state, v_state));
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = step;
                inference_tensor::bail!("Inkling 2 runs paged attention on CUDA only");
            }
        }

        // heads first for the cache; one position is that layout already
        let heads_first = |x: Tensor| -> Result<Tensor> {
            if t == 1 {
                x.reshape((b, hkv, 1, hd))
            } else {
                x.transpose(1, 2)?.contiguous()
            }
        };
        let (k, v) = kv_cache.append(&heads_first(k)?, &heads_first(v)?)?;
        let s = k.dim(2)?;
        // `[b, h, t, extent + 1]` for the paths that take heads first
        let table = || token_table.transpose(1, 2)?.contiguous();

        // one position on Vulkan: the fused kernel over the cache in place, the sink in the zero column's place
        if t == 1 && q.device().is_vulkan() {
            let bias = Tensor::cat(
                &[
                    &table()?.squeeze(2)?.narrow(2, 0, self.extent)?,
                    &sinks.transpose(1, 2)?,
                ],
                2,
            )?;
            let q = q.transpose(1, 2)?;
            let op = inference_tensor::vulkan::BiasedDecode {
                q: &q,
                k: &k,
                v: &v,
                bias: &bias,
                window,
                scale: scale as f32,
            };
            if let Some(out) = inference_tensor::vulkan::biased_decode(&op)? {
                let out = out.reshape((b, t, h * hd))?;
                let out = out.broadcast_mul(&inference_tensor::nn::ops::sigmoid(&gate()?)?)?;
                let out = self.o_proj.forward(&out.to_dtype(inputs.q.dtype())?)?;
                return Ok((out, k_state, v_state));
            }
        }

        #[cfg(feature = "cuda")]
        if q.device().is_cuda() && inputs.tau.is_none() {
            use inference_nn::cuda::biased_attention::{Keys, Past};
            let op = inference_nn::cuda::biased_attention::BiasedAttention {
                q: &q,
                keys: Keys::View { k: &k, v: &v },
                past: Past::Fixed(s - t),
                table: &token_table,
                sinks: &sinks,
                window,
                scale: scale as f32,
                log_scale,
            };
            if let Some(out) = inference_nn::cuda::biased_attention::biased_attention(&op)? {
                return Ok((output_gate(out)?, k_state, v_state));
            }
        }

        let rel = rel.ok_or_else(|| {
            inference_tensor::Error::msg("Inkling 2: the fused attention did not take these inputs")
        })?;
        // the bias of every (query, key) pair, `[b, h, t, s]`, gathered by distance
        let columns = rel.columns.broadcast_as((b, h, t, s))?.contiguous()?;
        let bias = table()?.gather(&columns, 3)?.broadcast_add(&rel.mask)?;

        // the query heads of a kv head together, so K and V are read once
        let group = h / hkv;
        let q = q
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, hkv, group * t, hd))?;
        let scores = (q.matmul(&k.to_dtype(DType::F32)?.t()?)? * scale)?;
        let scores = (scores + bias.reshape((b, hkv, group * t, s))?)?;
        // the sink: a logit of its own per (query, head) with a zero value
        let sink = sinks
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, hkv, group * t, 1))?;
        let probs =
            inference_tensor::nn::ops::softmax_last_dim(&Tensor::cat(&[&scores, &sink], 3)?)?;
        let out = probs
            .narrow(3, 0, s)?
            .contiguous()?
            .matmul(&v.to_dtype(DType::F32)?)?
            .reshape((b, h, t, hd))?
            .transpose(1, 2)?
            .reshape((b, t, h * hd))?;
        let out = out.broadcast_mul(&inference_tensor::nn::ops::sigmoid(&gate()?)?)?;
        let out = self.o_proj.forward(&out.to_dtype(inputs.q.dtype())?)?;
        Ok((out, k_state, v_state))
    }
}

struct Moe {
    gate: Arc<dyn QuantMethod>,
    selection_bias: Tensor,
    experts: MoEExperts,
    top_k: usize,
    route_scale: f32,
}

impl Moe {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<inference_quant::Comm>,
        loading_isq: bool,
    ) -> Result<Self> {
        let gate_vb = vb.pp("gate");
        // the router reads f32 inputs, so its weight stays f32 (as stored) whatever the model's dtype
        let gate = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.n_routed_experts,
            &None,
            false,
            gate_vb.clone().set_dtype(DType::F32),
        )?;
        let selection_bias = gate_vb
            .get(cfg.n_routed_experts, "e_score_correction_bias")?
            .to_dtype(DType::F32)?;
        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.n_routed_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.intermediate_size,
            expert_proj_names: crate::moe::ExpertProjNames::DEFAULT,
            gated: true,
        };
        let experts = MoEExperts::new(
            &moe_cfg,
            vb,
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            Activation::SoftcapSilu(cfg.mlp_gate_softcap, cfg.mlp_up_softcap),
        )?;
        Ok(Self {
            gate,
            selection_bias,
            experts,
            top_k: cfg.num_experts_per_tok,
            route_scale: cfg.route_scale as f32,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, hidden) = xs.dims3()?;
        let flat = xs.reshape((b * t, hidden))?;
        // transformers routes in f32
        let logits = self.gate.forward(&flat.to_dtype(DType::F32)?)?;
        let topk = crate::ops::moe_router_topk(
            &logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.top_k,
                score_function: crate::ops::MoeRouterScoreFunction::Sigmoid,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: true,
                norm_min: ROUTE_NORM_MIN,
                output_scale: self.route_scale,
                logit_clip: None,
            },
            Some(&self.selection_bias),
            None,
        )?;
        self.experts
            .forward(xs, topk.values, &topk.indices)?
            .reshape((b, t, hidden))
    }
}

/// A looped layer's controller: `(1 - forget) * x + gain * message`, both gated on the unit-normed pair.
struct Controller {
    // `[rank, 2 hidden]`, then the gain's rows over the forget's, `[2 hidden, rank]`, applied transposed: each output
    // reads its own contiguous row; f32 as stored, on f32 unit-normed inputs whatever the model's dtype
    down: Arc<dyn QuantMethod>,
    gain_forget: Arc<dyn QuantMethod>,
    // the same weights for the fused controller of a few rows
    #[cfg(feature = "cuda")]
    weights: (Tensor, Tensor),
    hidden: usize,
}

impl Controller {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let (hidden, rank) = (cfg.hidden_size, cfg.loop_config.gate_rank);
        let vb = vb.set_dtype(DType::F32);
        let weight = |name: &str, shape: (usize, usize)| vb.pp(name).get(shape, "weight");
        // UnquantLinear: decode on CUDA takes its gemv, whose sums are llama.cpp's order (cuBLAS's drift greedy)
        let proj = |w: Tensor| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(inference_tensor::nn::Linear::new(w, None)),
            )?))
        };
        let gain = weight("gain", (hidden, rank))?;
        let forget = weight("forget", (hidden, rank))?;
        let down = weight("down", (rank, 2 * hidden))?;
        let gain_forget = Tensor::cat(&[&gain, &forget], 0)?;
        Ok(Self {
            #[cfg(feature = "cuda")]
            weights: (down.clone(), gain_forget.clone()),
            down: proj(down)?,
            gain_forget: proj(gain_forget)?,
            hidden,
        })
    }

    // the gain's and the forget's pre-activations, from the gate's down projection
    fn gain_forget(&self, gate: &Tensor) -> Result<(Tensor, Tensor)> {
        let hidden = self.hidden;
        let both = self
            .gain_forget
            .forward(&self.down.forward(&gate.to_dtype(DType::F32)?)?)?;
        Ok((
            both.narrow(D::Minus1, 0, hidden)?.contiguous()?,
            both.narrow(D::Minus1, hidden, hidden)?.contiguous()?,
        ))
    }

    fn forward(&self, x: &Tensor, message: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        #[cfg(feature = "cuda")]
        if let Some(out) = inference_nn::cuda::inkling::controller(
            x,
            message,
            &self.weights.0,
            &self.weights.1,
            UNIT_NORM_EPS as f32,
        )? {
            return Ok(out);
        }
        #[cfg(feature = "cuda")]
        if let Some(gate) =
            inference_nn::cuda::inkling::controller_in(x, message, UNIT_NORM_EPS as f32)?
        {
            let (gain, forget) = self.gain_forget(&gate)?;
            if let Some(out) =
                inference_nn::cuda::inkling::controller_out(x, message, &gain, &forget)?
            {
                return Ok(out);
            }
        }
        let gate = Tensor::cat(
            &[
                &unit_rms(x, UNIT_NORM_EPS)?,
                &unit_rms(message, UNIT_NORM_EPS)?,
            ],
            D::Minus1,
        )?;
        let (gain, forget) = self.gain_forget(&gate)?;
        let (gain, forget) = (gain.tanh()?, forget.tanh()?);
        let x = x.to_dtype(DType::F32)?;
        let kept = (&x - (forget * &x)?)?;
        (kept + (gain * message.to_dtype(DType::F32)?)?)?.to_dtype(dtype)
    }
}

/// A looped layer's weights for its later passes: the message queries over the previous pass and the controllers.
struct LoopLayer {
    // `[4, hidden]`: the q, k, v and FFN messages
    queries: Tensor,
    controllers: Vec<Controller>,
}

enum AttentionNorms {
    Sliding(RmsNorm),
    // q, k, v
    Global([RmsNorm; 3]),
}

struct DecoderLayer {
    // `[1 or 3, hidden]`; the first layer takes its input as is
    attention_query: Option<Tensor>,
    mlp_query: Tensor,
    norms: AttentionNorms,
    attn: Attention,
    post_attention_conv: ShortConv,
    ffn_norm: RmsNorm,
    moe: Moe,
    post_mlp_conv: ShortConv,
    block_start: bool,
    loop_layer: Option<LoopLayer>,
}

/// Where one pass of one layer reads and writes: its cache slot's KV and convolution state.
struct LayerStep<'a> {
    kv_cache: &'a mut KvCache,
    paged: Option<PagedStep<'a>>,
    conv_state: ConvState<'a>,
    // `None` where the fused CUDA kernel computes the mask and the bias's columns itself
    rel: Option<&'a RelativeInputs>,
    tau: Option<&'a Tensor>,
    stacks: &'a BlockStacks,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        layer: usize,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
        comm: &Arc<inference_quant::Comm>,
    ) -> Result<Self> {
        let vb = mapper.set_device(layer, vb, loading_isq);
        let hidden = cfg.hidden_size;
        let sliding = cfg.is_sliding(layer);
        let width = cfg.sconv_kernel_size;
        let eps = cfg.rms_norm_eps;
        let attention_query = if layer == 0 {
            None
        } else {
            let n = if sliding { 1 } else { 3 };
            Some(
                vb.pp("attention_input")
                    .get((n, hidden), "query")?
                    .to_dtype(DType::F32)?,
            )
        };
        let norms = if sliding {
            AttentionNorms::Sliding(RmsNorm::new(hidden, eps, vb.pp("input_layernorm"))?)
        } else {
            AttentionNorms::Global([
                RmsNorm::new(hidden, eps, vb.pp("q_input_layernorm"))?,
                RmsNorm::new(hidden, eps, vb.pp("k_input_layernorm"))?,
                RmsNorm::new(hidden, eps, vb.pp("v_input_layernorm"))?,
            ])
        };
        let lc = &cfg.loop_config;
        let loop_layer = if (lc.core_start..lc.core_end).contains(&layer) && lc.num_passes > 1 {
            let lvb = vb.pp("loop");
            Some(LoopLayer {
                queries: lvb
                    .get((CONTROLLERS.len(), hidden), "msg_query")?
                    .to_dtype(DType::F32)?,
                controllers: CONTROLLERS
                    .iter()
                    .map(|name| Controller::new(cfg, lvb.pp(name)))
                    .collect::<Result<_>>()?,
            })
        } else {
            None
        };
        let layer_device = vb.device().clone();
        Ok(Self {
            attention_query,
            mlp_query: vb
                .pp("mlp_input")
                .get((1, hidden), "query")?
                .to_dtype(DType::F32)?,
            norms,
            attn: Attention::new(cfg, layer, vb.pp("self_attn"))?,
            post_attention_conv: ShortConv {
                weight: ShortConv::load(hidden, width, &vb.pp("post_attention_convolution"))?,
                tap_heads: None,
            },
            ffn_norm: RmsNorm::new(hidden, eps, vb.pp("post_attention_layernorm"))?,
            moe: Moe::new(cfg, vb.pp("mlp"), layer_device, comm, loading_isq)?,
            post_mlp_conv: ShortConv {
                weight: ShortConv::load(hidden, width, &vb.pp("post_mlp_convolution"))?,
                tap_heads: None,
            },
            block_start: layer.is_multiple_of(cfg.n_layers_per_attn_res_block),
            loop_layer,
        })
    }

    fn forward(
        &self,
        cfg: &Config,
        x: &Tensor,
        blocks: &mut Blocks,
        messages: Option<&Tensor>,
        step: LayerStep<'_>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let LayerStep {
            kv_cache,
            paged,
            conv_state,
            rel,
            tau,
            stacks,
        } = step;
        let dtype = x.dtype();
        let span = |name| inference_tensor::profile::span(name, Some(x.device()));
        let mix_span = span("inkling.attention_inputs");
        let mut inputs: [Tensor; 3] = match &self.attention_query {
            Some(query) => {
                let m = mix(blocks, x, query, dtype, stacks)?;
                let pick = |j: usize| -> Result<Tensor> {
                    let j = if query.dim(0)? == 1 { 0 } else { j };
                    m.narrow(2, j, 1)?.squeeze(2)?.to_dtype(dtype)
                };
                [pick(0)?, pick(1)?, pick(2)?]
            }
            None => [x.clone(), x.clone(), x.clone()],
        };
        let controllers = match (messages, &self.loop_layer) {
            (Some(messages), Some(lp)) => Some((messages, lp)),
            (None, _) => None,
            (Some(_), None) => inference_tensor::bail!("Inkling 2: messages for an unlooped layer"),
        };
        if let Some((messages, lp)) = controllers {
            for (j, input) in inputs.iter_mut().enumerate() {
                let message = messages.narrow(2, j, 1)?.squeeze(2)?;
                *input = lp.controllers[j].forward(input, &message)?;
            }
        }
        let [q_in, k_in, v_in] = match &self.norms {
            AttentionNorms::Sliding(norm) if controllers.is_some() => {
                let [q, k, v] = inputs;
                [norm.forward(&q)?, norm.forward(&k)?, norm.forward(&v)?]
            }
            AttentionNorms::Sliding(norm) => {
                let normed = norm.forward(&inputs[0])?;
                [normed.clone(), normed.clone(), normed]
            }
            AttentionNorms::Global(norms) => {
                let [q, k, v] = inputs;
                [
                    norms[0].forward(&q)?,
                    norms[1].forward(&k)?,
                    norms[2].forward(&v)?,
                ]
            }
        };

        drop(mix_span);

        let kv_dim = cfg.kv_dim();
        let hidden = cfg.hidden_size;
        let attn_span = span("inkling.attention");
        let (attn, k_state, v_state) = self.attn.forward(
            cfg,
            AttentionInputs {
                q: &q_in,
                k: &k_in,
                v: &v_in,
                tau,
            },
            rel,
            kv_cache,
            paged.as_ref(),
            conv_state,
        )?;
        drop(attn_span);
        let conv_span = span("inkling.post_attention_conv");
        let (attn, attn_state) = self.post_attention_conv.forward(
            &attn,
            None,
            conv_state.part(2 * kv_dim, hidden)?,
            dtype,
        )?;
        let attn = attn.to_dtype(dtype)?;
        drop(conv_span);
        let ffn_mix_span = span("inkling.ffn_inputs");

        // a block's first layer starts a new residual stream; the one before it joins the mixed blocks
        let ffn_in = if self.block_start {
            blocks.push(x.clone());
            attn
        } else {
            (x + attn)?
        };
        let mut h = mixed(blocks, &ffn_in, &self.mlp_query, 0, dtype, stacks)?;
        if let Some((messages, lp)) = controllers {
            let message = messages.narrow(2, FFN_CONTROLLER, 1)?.squeeze(2)?;
            h = lp.controllers[FFN_CONTROLLER].forward(&h, &message)?;
        }
        let h = self.ffn_norm.forward(&h)?;
        drop(ffn_mix_span);
        let moe_span = span("inkling.moe");
        let h = self.moe.forward(&h)?;
        drop(moe_span);
        let tail_span = span("inkling.post_mlp_conv");
        let (h, ffn_state) = self.post_mlp_conv.forward(
            &h,
            None,
            conv_state.part(2 * kv_dim + hidden, hidden)?,
            dtype,
        )?;
        let out = (ffn_in + h.to_dtype(dtype)?)?;
        let state = match (k_state, v_state, attn_state, ffn_state) {
            (Some(k), Some(v), Some(attn), Some(ffn)) => {
                Some(Tensor::cat(&[&k, &v, &attn, &ffn], 1)?)
            }
            _ => None,
        };
        drop(tail_span);
        Ok((out, state))
    }
}

pub struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    embed_norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    final_query: Tensor,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    cfg: Config,
    dtype: DType,
    cache: EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    meta: ModelConfigMetadata,
    // under paged attention: what writes each step's K/V rows into the paged cache
    paged_writer: Option<PagedAttention>,
    // the sliding slots' window rings, and each slot's ring state in the hybrid cache
    ring: Option<WindowRing>,
    ring_caches: Vec<Option<usize>>,
    spec: speculative::SpecState,
    // added to each layer visit's output, the looped core's every pass alike
    control_vectors: inference_nn::control_vectors::ControlVectorSlot,
    control_devices: Vec<Device>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        // the paged cache is read by the fused CUDA attention, which nothing else stands in for
        let paged_writer = match attention_mechanism {
            AttentionImplementation::PagedAttention => {
                #[cfg(feature = "cuda")]
                let fused = normal_loading_metadata.real_device.is_cuda()
                    && inference_nn::cuda::biased_attention::HEAD_DIMS.contains(&cfg.head_dim);
                #[cfg(not(feature = "cuda"))]
                let fused = false;
                if !fused {
                    inference_tensor::bail!(
                        "Inkling 2 runs paged attention on CUDA only; pass --paged-attn off"
                    );
                }
                Some(PagedAttention::new(
                    cfg.head_dim,
                    &normal_loading_metadata.real_device,
                    None,
                )?)
            }
            AttentionImplementation::Eager => None,
        };
        let vb = crate::inkling2_official::bind(cfg, vb)?;
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata.mapper;
        let loading_isq = normal_loading_metadata.loading_isq;
        let dtype = vb_m.dtype();
        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), loading_isq),
            None,
            &cfg.quantization_config,
        )?;
        let embed_norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("embed_norm"), false),
        )?;
        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer| {
            let comm = mapper.get_comm_for(layer)?;
            DecoderLayer::new(cfg, layer, vb_l.pp(layer), &*mapper, loading_isq, &comm)
        })?;
        let final_query = mapper
            .set_nm_device(vb_m.pp("final_input"), false)
            .get((1, cfg.hidden_size), "query")?
            .to_dtype(DType::F32)?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = ReplicatedLayer::new(
            cfg.hidden_size,
            cfg.vocab_size,
            &cfg.quantization_config,
            false,
            mapper.set_nm_device(vb.pp("lm_head"), loading_isq),
        )?;

        // the slots' KV caches, then their convolution states
        let slots = cfg.cache_slots();
        let device_of = |slot: usize| {
            mapper
                .device_for(physical_layer(cfg, slot), false)
                .unwrap_or(&normal_loading_metadata.real_device)
                .clone()
        };
        let layer_types = (0..2 * slots)
            .map(|i| {
                if i < slots {
                    HybridLayerType::Attention
                } else {
                    HybridLayerType::Recurrent
                }
            })
            .collect::<Vec<_>>();
        let layer_devices = (0..2 * slots)
            .map(|i| device_of(i % slots))
            .collect::<Vec<_>>();
        // the CUDA attention reads one width for K and V; others would load and then fail every request
        if let Some(PagedKvLayout {
            cache_type: PagedCacheType::Quant { k, v },
            ..
        }) = normal_loading_metadata.paged_kv
            && normal_loading_metadata.real_device.is_cuda()
            && (k != v || !QUANT_CACHE_BITS.contains(&k))
        {
            inference_tensor::bail!(
                "Inkling 2 on CUDA reads an auto, f8e4m3 or quantized paged cache with K and V at one width (q2..q6 \
                 or q8), not q{k}/q{v}"
            );
        }
        // on CUDA a sliding slot keeps its window in a ring beside the convolution states, after them in the cache
        let ring = matches!(attention_mechanism, AttentionImplementation::PagedAttention)
            .then(|| {
                cfg.window_ring(
                    normal_loading_metadata.paged_kv,
                    &normal_loading_metadata.real_device,
                    dtype,
                )
            })
            .flatten();
        let mut ring_caches = vec![None; slots];
        let mut auxiliary = Vec::new();
        if let Some(ring) = &ring {
            for slot in cfg.sliding_slots() {
                ring_caches[slot] = Some(2 * slots + auxiliary.len());
                auxiliary.push(AuxiliaryRecurrent {
                    layer: slot,
                    config: RecurrentLayerConfig {
                        conv_dim: 0,
                        conv_width: 1,
                        state: RecurrentStateSpec::Opaque { dims: ring.dims() },
                        recurrent_dtype: Some(ring.dtype),
                    },
                });
            }
        }
        let cache = HybridCache::new(
            HybridCacheConfig {
                // a ring slot holds a window for every sliding slot: the load's reservation sizes the pool, which
                // starts as small as a sequence and the graph pad
                initial_capacity: ring
                    .is_some()
                    .then_some(1 + inference_nn::kv_cache::RECURRENT_GRAPH_PAD_SLOTS),
                layer_types,
                max_seq_len: cfg.max_position_embeddings,
                recurrent: RecurrentLayerConfig {
                    conv_dim: cfg.conv_channels(),
                    conv_width: cfg.sconv_kernel_size - 1,
                    state: RecurrentStateSpec::Opaque { dims: Vec::new() },
                    recurrent_dtype: None,
                },
                auxiliary,
            },
            DType::F32,
            &layer_devices,
        )?;
        let world = mapper.get_comm_for(0)?.world_size();
        let control_devices = (0..cfg.num_hidden_layers)
            .map(|layer| {
                mapper
                    .device_for(layer, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone()
            })
            .collect();
        Ok(Self {
            control_vectors: Default::default(),
            control_devices,
            embed_tokens,
            embed_norm,
            layers,
            final_query,
            norm,
            lm_head,
            dtype,
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(cache))),
            device: normal_loading_metadata.real_device,
            meta: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: slots,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / world).max(1),
                num_attn_heads: cfg.num_attention_heads / world,
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            cfg: cfg.clone(),
            mapper,
            paged_writer,
            ring,
            ring_caches,
            spec: speculative::SpecState::default(),
        })
    }

    // a paged step's slot mapping, block tables and pasts, or `None` off paged attention
    fn paged_positions(
        &self,
        ctx: &ModelForwardContext<'_>,
        b: usize,
        t: usize,
    ) -> Result<Option<PagedPositions>> {
        let Some((_, metadata)) = ctx.paged_metadata() else {
            return Ok(None);
        };
        let loc = self.device.location();
        let on_device = |map: Option<&std::collections::HashMap<_, Tensor>>, what: &str| {
            map.and_then(|m| m.get(&loc)).cloned().ok_or_else(|| {
                inference_tensor::Error::msg(format!("Inkling 2: paged metadata has no {what}"))
            })
        };
        let slots = on_device(Some(&metadata.slot_mappings), "slot mapping")?;
        if metadata.is_decode_step() {
            // a row a query: each sequence's first row's table, its last row's length (past + t)
            let tables = on_device(metadata.block_tables.as_ref(), "block tables")?;
            let lens = on_device(metadata.context_lens.as_ref(), "context lengths")?;
            let (tables, lens) = if t > 1 {
                let width = tables.dim(1)?;
                (
                    tables.reshape((b, t, width))?.narrow(1, 0, 1)?.squeeze(1)?,
                    lens.reshape((b, t))?.narrow(1, t - 1, 1)?.squeeze(1)?,
                )
            } else {
                (tables, lens.flatten_all()?)
            };
            return Ok(Some(PagedPositions {
                slots,
                tables,
                lens,
                minus: t,
            }));
        }
        // a prompt chunk: its sequences' whole tables, their pasts from the host
        let tables = on_device(
            metadata
                .full_block_tables
                .as_ref()
                .or(metadata.block_tables.as_ref()),
            "block tables",
        )?;
        let pasts = ctx
            .seqlen_offsets()
            .iter()
            .map(|&p| u32::try_from(p).map_err(inference_tensor::Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        if pasts.len() != b {
            inference_tensor::bail!("Inkling 2: {} offsets for {b} sequences", pasts.len());
        }
        Ok(Some(PagedPositions {
            slots,
            tables,
            lens: Tensor::from_vec(pasts, b, &self.device)?,
            minus: 0,
        }))
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward_embeds(self.embed(input_ids)?, ctx)
    }

    // `x` is the normed token embeddings, image rows already in place
    pub fn forward_embeds(&self, x: Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let _span = inference_tensor::profile::span("inkling.forward", Some(x.device()));
        let cfg = &self.cfg;
        let (b, t, _) = x.dims3()?;
        let recurrent = ctx
            .recurrent_metadata()
            .cloned()
            .ok_or_else(|| inference_tensor::Error::msg("Inkling 2 needs recurrent metadata"))?;
        let fresh_state =
            recurrent.batch_kind() == RecurrentBatchKind::Prefill && ctx.is_first_prompt_chunk();
        let paged = match &self.paged_writer {
            Some(writer) => self
                .paged_positions(ctx, b, t)?
                .map(|positions| (writer, positions, ctx.paged_metadata().map(|(c, _)| c))),
            None => None,
        };
        let mut cache = self.cache.hybrid();
        let slots = cfg.cache_slots();
        let past = match cache.get_mut(0) {
            Some(HybridLayerCache::Attention(kv)) => kv.current_seq_len(),
            _ => inference_tensor::bail!("Inkling 2: slot 0 is not an attention cache"),
        };
        let device = x.device().clone();
        // the fused CUDA attention computes the mask and the bias's columns itself
        #[cfg(feature = "cuda")]
        let fused = device.is_cuda()
            && inference_nn::cuda::biased_attention::HEAD_DIMS.contains(&cfg.head_dim);
        #[cfg(not(feature = "cuda"))]
        let fused = false;
        let rels = (!fused)
            .then(|| -> Result<_> {
                Ok((
                    RelativeInputs::new(past, t, cfg.rel_extent, None, &device)?,
                    RelativeInputs::new(
                        past,
                        t,
                        cfg.sliding_window_size,
                        Some(cfg.sliding_window_size),
                        &device,
                    )?,
                ))
            })
            .transpose()?;
        // 1 + alpha log(max(1, (pos + 1) / floor)), which the fused CUDA attention computes itself
        let tau = (!fused)
            .then(|| -> Result<Tensor> {
                let tau = (0..t)
                    .map(|i| {
                        let ratio = (past + i + 1) as f64 / cfg.log_scaling_n_floor;
                        (1.0 + cfg.log_scaling_alpha * ratio.max(1.0).ln()) as f32
                    })
                    .collect::<Vec<_>>();
                Tensor::from_vec(tau, (1, t, 1, 1), &device)?.broadcast_as((b, t, 1, 1))
            })
            .transpose()?;
        let decode = t == 1 && recurrent.batch_kind() == RecurrentBatchKind::Decode;

        let stacks = BlockStacks::default();
        // a verify step keeps each slot's state and conv inputs, for the commit to cut back to the accepted rows
        let verifying = self.spec.drafter().is_some()
            && recurrent.batch_kind() == RecurrentBatchKind::SpeculativeDecode
            && t > 1;
        // a decode step updates the pools' rows in place rather than gathering and scattering them; on CUDA a verify
        // too, its kernels keeping the states as read and the inputs for the rollback
        let in_place =
            (device.is_vulkan() && decode) || (device.is_cuda() && (decode || verifying));
        let replays = std::cell::RefCell::new(Vec::new());
        #[cfg(feature = "cuda")]
        let ring_rows = match (&self.ring, &paged) {
            (Some(ring), Some((_, positions, Some(_)))) => {
                if cache.checkpoint_lanes() != 1 {
                    inference_tensor::bail!("Inkling 2's window rings keep one lane a sequence");
                }
                let states = cache.state_indices_for_layer(slots)?.ok_or_else(|| {
                    inference_tensor::Error::msg("Inkling 2: missing recurrent state indices")
                })?;
                let capacity = match self
                    .ring_caches
                    .iter()
                    .flatten()
                    .next()
                    .and_then(|&index| cache.get(index))
                {
                    Some(HybridLayerCache::Recurrent(pool)) => pool.recurrent_state.dim(0)?,
                    _ => inference_tensor::bail!("Inkling 2: a window ring without its pool"),
                };
                Some(RingRows::new(ring, positions, &states, capacity, t)?)
            }
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let ring_rows: Option<RingRows> = None;
        let mut run = |layer: usize,
                       slot: usize,
                       x: &Tensor,
                       blocks: &mut Blocks,
                       messages: Option<&Tensor>|
         -> Result<Tensor> {
            let indices = cache
                .state_indices_for_layer(slots + slot)?
                .ok_or_else(|| {
                    inference_tensor::Error::msg("Inkling 2: missing recurrent state indices")
                })?;
            let (pool, gathered) = {
                let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(slots + slot) else {
                    inference_tensor::bail!("Inkling 2: slot {slot} has no convolution state");
                };
                if in_place {
                    (Some(pool.conv_state.clone()), None)
                } else {
                    let state = pool.gather_conv_state(&indices)?;
                    let state = if fresh_state {
                        state.zeros_like()?
                    } else {
                        state
                    };
                    (None, Some(state))
                }
            };
            let conv_inputs = ConvInputs::default();
            let pool_slots = if in_place {
                Some(indices.to_dtype(DType::U32)?)
            } else {
                None
            };
            let rollback = match (&pool, verifying) {
                (Some(pool), true) => {
                    let (_, channels, hist) = pool.dims3()?;
                    // SAFETY: the convs write every channel of both before they are read
                    Some(unsafe {
                        ConvRollback {
                            before: Tensor::empty((b, channels, hist), DType::F32, &device)?,
                            stash: Tensor::empty((b, t, channels), x.dtype(), &device)?,
                        }
                    })
                }
                _ => None,
            };
            let conv_state = match (&pool, &pool_slots, &gathered) {
                (Some(pool), Some(slots), _) => ConvState::Pool {
                    pool,
                    slots,
                    rollback: rollback.as_ref(),
                },
                (_, _, Some(rows)) => ConvState::Rows {
                    rows,
                    inputs: verifying.then_some(&conv_inputs),
                },
                _ => unreachable!("a slot's states are pooled or gathered"),
            };
            let x = self.mapper.map(x.clone(), layer)?;
            // a sliding slot's ring: its pool's rows as cache blocks, V the K view shifted by a ring
            let ring_views = match (&self.ring, self.ring_caches[slot], &ring_rows) {
                (Some(ring), Some(index), Some(_)) => {
                    let Some(HybridLayerCache::Recurrent(pool)) = cache.get(index) else {
                        inference_tensor::bail!("Inkling 2: slot {slot} has no window ring");
                    };
                    let state = &pool.recurrent_state;
                    let blocks = state.dim(0)? * 2 * ring.blocks;
                    let all = state.reshape((blocks, ring.kv_heads, ring.block_size, ring.row))?;
                    Some((
                        all.narrow(0, 0, blocks - ring.blocks)?,
                        all.narrow(0, ring.blocks, blocks - ring.blocks)?,
                    ))
                }
                _ => None,
            };
            let Some(HybridLayerCache::Attention(kv_cache)) = cache.get_mut(slot) else {
                inference_tensor::bail!("Inkling 2: slot {slot} has no KV cache");
            };
            let rel = rels.as_ref().map(|(global, sliding)| {
                if cfg.is_sliding(layer) {
                    sliding
                } else {
                    global
                }
            });
            let paged_step = match &paged {
                Some((writer, positions, Some(_))) if ring_views.is_some() => {
                    let (k_cache, v_cache) = ring_views.as_ref().expect("checked above");
                    Some(PagedStep {
                        k_cache,
                        v_cache,
                        positions,
                        writer,
                        ring: ring_rows.as_ref().zip(self.ring.map(|ring| ring.blocks)),
                    })
                }
                Some((writer, positions, Some(caches))) => {
                    let (k_cache, v_cache) = &caches[slot];
                    Some(PagedStep {
                        k_cache,
                        v_cache,
                        positions,
                        writer,
                        ring: None,
                    })
                }
                Some((_, _, None)) => {
                    inference_tensor::bail!("Inkling 2: paged metadata without its caches")
                }
                None => None,
            };
            let (out, state) = self.layers[layer].forward(
                cfg,
                &x,
                blocks,
                messages,
                LayerStep {
                    kv_cache,
                    paged: paged_step,
                    conv_state,
                    rel,
                    tau: tau.as_ref(),
                    stacks: &stacks,
                },
            )?;
            if let Some(ConvRollback { before, stash }) = rollback {
                replays.borrow_mut().push(speculative::ConvReplay {
                    slot: slots + slot,
                    indices: indices.clone(),
                    host_indices: recurrent.state_indices_host().map(<[u32]>::to_vec),
                    state: before,
                    inputs: stash,
                });
            } else if verifying && let Some(before) = &gathered {
                replays.borrow_mut().push(speculative::ConvReplay {
                    slot: slots + slot,
                    indices: indices.clone(),
                    host_indices: recurrent.state_indices_host().map(<[u32]>::to_vec),
                    state: before.clone(),
                    inputs: {
                        let parts = conv_inputs.take();
                        let dtype = parts[0].dtype();
                        let parts = parts
                            .iter()
                            .map(|p| p.to_dtype(dtype))
                            .collect::<Result<Vec<_>>>()?;
                        Tensor::cat(&parts, 2)?
                    },
                });
            }
            if let Some(state) = state {
                let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(slots + slot) else {
                    inference_tensor::bail!("Inkling 2: slot {slot} has no convolution state");
                };
                pool.scatter_conv_state_with_host_indices(
                    &indices,
                    recurrent.state_indices_host(),
                    &state,
                )?;
            }
            Ok(out)
        };

        // the drafter's taps: the residual stream before each tapped layer visit (the looped core's per pass)
        let tap_visits = self.spec.tap_visits();
        let taps = std::cell::RefCell::new(Vec::new());
        let visit = std::cell::Cell::new(0usize);
        let mut run = |layer: usize,
                       slot: usize,
                       x: &Tensor,
                       blocks: &mut Blocks,
                       messages: Option<&Tensor>|
         -> Result<Tensor> {
            if tap_visits.contains(&visit.get()) {
                taps.borrow_mut().push(x.clone());
            }
            visit.set(visit.get() + 1);
            let out = run(layer, slot, x, blocks, messages)?;
            self.control_vectors.apply(layer, out)
        };
        let lc = &cfg.loop_config;
        let mut blocks = Blocks::default();
        let mut x = x;
        for layer in 0..lc.core_start {
            x = run(layer, layer, &x, &mut blocks, None)?;
        }
        // every pass starts from the prelude's output
        let prelude = x.clone();
        let prelude_blocks = blocks.clone();
        // LoopCD: the first pass's output (and its blocks) kept; LoopCD-Hidden pushes the last pass's away from it,
        // LoopCD-Logits runs the coda on it too (always, so its cache slots stay whole) and contrasts the logits
        let guidance = self.control_vectors.loop_guidance();
        let logits_mode = cfg.loop_guidance_logits && cfg.passes() > 1;
        let hidden = guidance.clone().filter(|_| !cfg.loop_guidance_logits);
        let mut first_pass = None;
        for pass in 0..cfg.passes() {
            // the previous pass's bank: its blocks after the first core block (the prelude's output), and its output
            let mut bank = (pass > 0).then(|| {
                let states = blocks.states[prelude_blocks.states.len() + 1..].to_vec();
                (Blocks::new(states), x.clone())
            });
            if pass > 0 {
                x = prelude.clone();
                blocks = prelude_blocks.clone();
            }
            let core = &self.layers[lc.core_start..lc.core_end];
            for (layer, decoder) in (lc.core_start..).zip(core) {
                let slot = slot_of(cfg, pass, layer);
                let messages = match &mut bank {
                    Some((bank, bank_out)) => {
                        let lp = decoder.loop_layer.as_ref().ok_or_else(|| {
                            inference_tensor::Error::msg(
                                "Inkling 2: a looped layer has no loop weights",
                            )
                        })?;
                        Some(mix(bank, bank_out, &lp.queries, DType::F32, &stacks)?)
                    }
                    None => None,
                };
                x = run(layer, slot, &x, &mut blocks, messages.as_ref())?;
            }
            if pass == 0 && cfg.passes() > 1 && (hidden.is_some() || logits_mode) {
                first_pass = Some((x.clone(), blocks.clone()));
            }
        }
        if let (Some(w), Some((first, _))) = (&hidden, &first_pass) {
            let w = w.to_device(x.device())?.to_dtype(x.dtype())?;
            x = (&x + (&x - first)?.broadcast_mul(&w)?)?;
        }
        for layer in lc.core_end..cfg.num_hidden_layers {
            x = run(layer, layer, &x, &mut blocks, None)?;
        }
        let reference = match first_pass.filter(|_| logits_mode) {
            Some((mut x1, mut blocks1)) => {
                for layer in lc.core_end..cfg.num_hidden_layers {
                    x1 = run(layer, reference_slot(cfg, layer), &x1, &mut blocks1, None)?;
                }
                Some((x1, blocks1))
            }
            None => None,
        };
        drop(cache);

        let head = |x: &Tensor, blocks: &mut Blocks| -> Result<Tensor> {
            let x = self.mapper.cast_nm_device(x, false)?;
            let x = mixed(blocks, &x, &self.final_query, 0, self.dtype, &stacks)?;
            self.norm.forward(&x)
        };
        let x = head(&x, &mut blocks)?;
        let mut taps = taps.into_inner();
        if self.spec.taps_output() {
            taps.push(x.clone());
        }
        self.spec.store(
            taps,
            replays.into_inner(),
            recurrent.batch_kind() == RecurrentBatchKind::Prefill,
            |tap| ctx.logits(tap),
        )?;
        let x = ctx.logits(&x)?;
        let logits = ctx.lm_head(&*self.lm_head, &x)?;
        if logits.dim(D::Minus1)? != cfg.vocab_size {
            return Ok(logits);
        }
        let cap = cfg.final_logit_softcapping;
        let capped = |logits: Tensor| -> Result<Tensor> {
            ((logits.to_dtype(DType::F32)? / cap)?.tanh()? * cap)?
                .narrow(D::Minus1, 0, cfg.unpadded_vocab_size)?
                .contiguous()
        };
        let logits = capped(logits)?;
        match (reference, guidance) {
            (Some((x1, mut blocks1)), Some(w)) => {
                let x1 = ctx.logits(&head(&x1, &mut blocks1)?)?;
                let first = capped(ctx.lm_head(&*self.lm_head, &x1)?)?;
                loop_contrast(&logits, &first, &w)
            }
            _ => Ok(logits),
        }
    }
}

/// LoopCD-Logits with the adaptive strength: `z + w (z - z1)`, `w = w_max (1 - (p1 - p2))` from the last pass's top
/// two probabilities, so guidance is strongest where the choice is close; the control tokens (the vocabulary's tail
/// from CONTROL_TOKENS_START: message, block and tool-call markers) keep the last pass's logits.
fn loop_contrast(z: &Tensor, z1: &Tensor, w_max: &Tensor) -> Result<Tensor> {
    let vocab = z.dim(D::Minus1)?;
    // p1 - p2 = (1 - e^(z2 - z1)) / sum e^(z - z1): reductions, not a one-row softmax over the vocabulary (~2.4 ms)
    let shifted = z.broadcast_sub(&z.max_keepdim(D::Minus1)?)?;
    let total = shifted.exp()?.sum_keepdim(D::Minus1)?;
    let below = shifted.lt(0.0)?;
    let second = below
        .where_cond(&shifted, &shifted.affine(0.0, f64::from(f32::MIN))?)?
        .max_keepdim(D::Minus1)?;
    let margin = ((second.exp()? * -1.0)? + 1.0)?.div(&total)?;
    let w = ((margin * -1.0)? + 1.0)?
        .broadcast_mul(&w_max.to_device(z.device())?.to_dtype(DType::F32)?)?;
    let text = CONTROL_TOKENS_START.min(vocab);
    let steered = z.narrow(D::Minus1, 0, text)?.add(
        &(z.narrow(D::Minus1, 0, text)? - z1.narrow(D::Minus1, 0, text)?)?.broadcast_mul(&w)?,
    )?;
    if text == vocab {
        return Ok(steered);
    }
    Tensor::cat(
        &[&steered, &z.narrow(D::Minus1, text, vocab - text)?],
        D::Minus1,
    )
}

/// The weights a cache slot runs: its own layer's, or, for a later pass of the core, the core layer's.
fn physical_layer(cfg: &Config, slot: usize) -> usize {
    let n = cfg.num_hidden_layers;
    if slot < n {
        slot
    } else if slot < cfg.visits() {
        cfg.loop_config.core_start + (slot - n) % cfg.core_len()
    } else {
        cfg.loop_config.core_end + (slot - cfg.visits())
    }
}

// a coda layer's slot for its run on the first pass's output (LoopCD-Logits)
fn reference_slot(cfg: &Config, layer: usize) -> usize {
    cfg.visits() + (layer - cfg.loop_config.core_end)
}

fn slot_of(cfg: &Config, pass: usize, layer: usize) -> usize {
    if pass == 0 {
        layer
    } else {
        cfg.num_hidden_layers + (pass - 1) * cfg.core_len() + (layer - cfg.loop_config.core_start)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("embed_norm").add(&self.embed_norm);
        uvb_m.pp("norm").add(&self.norm);
        uvb.to_safetensors()
    }
}

impl NormalModel for Model {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }

    fn set_control_vectors(
        &self,
        set: inference_nn::control_vectors::ControlVectorSet,
    ) -> Result<bool> {
        // the guidance applies at the coda's input, on its first layer's device
        let coda = &self.control_devices[self.cfg.loop_config.core_end];
        self.control_vectors.replace(
            set,
            &self.control_devices,
            self.cfg.hidden_size,
            self.dtype,
            Some(coda),
        )
    }

    fn control_vectors(&self) -> inference_nn::control_vectors::ControlVectorSet {
        self.control_vectors.current()
    }

    fn begin_layer_capture(&self) -> bool {
        self.control_vectors
            .begin_capture(self.cfg.num_hidden_layers);
        true
    }

    fn take_layer_capture(&self) -> Result<Option<Vec<Tensor>>> {
        self.control_vectors.take_capture()
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }

    fn max_seq_len(&self) -> usize {
        self.cfg.max_position_embeddings
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.meta
    }

    fn model_config(
        &self,
    ) -> Arc<dyn inference_nn::paged_attention::ModelConfigLike + Send + Sync> {
        Arc::new(
            self.cfg
                .paged_cache_config(self.meta.clone(), self.ring.as_ref()),
        )
    }

    // a decode step reads its positions and lengths from the paged metadata's device tensors (a drafter's taps and
    // verify replays ride the speculative graph state)
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        self.paged_writer.is_some()
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_inputs_mask_the_future_and_the_window_and_zero_the_far_bias() {
        let rel = RelativeInputs::new(2, 2, 2, Some(3), &Device::Cpu).unwrap();
        let mask = rel.mask.to_vec2::<f32>().unwrap();
        let columns = rel.columns.to_vec2::<u32>().unwrap();
        // query positions 2 and 3 over keys 0..4
        assert_eq!(mask[0], [0.0, 0.0, 0.0, f32::NEG_INFINITY]);
        assert_eq!(mask[1], [f32::NEG_INFINITY, 0.0, 0.0, 0.0]);
        assert_eq!(columns[0], [2, 1, 0, 2]);
        assert_eq!(columns[1], [2, 2, 1, 0]);
    }
}
