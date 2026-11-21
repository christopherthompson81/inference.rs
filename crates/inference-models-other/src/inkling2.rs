//! Inkling 2 (`inkling2_loop`): a looped hybrid MoE. Every layer mixes its inputs from the earlier blocks' outputs
//! (block attention residuals), attends with a learned relative-position bias and a per-query sink in place of RoPE,
//! and runs short causal convolutions over K, V and both sublayer outputs; the middle layers run several passes with
//! shared weights, each later pass steered by low-rank controllers from the previous one. Text only; llama.cpp's
//! `src/models/inkling2{,-loop}.cpp` and transformers' `modeling_inkling2.py` are the references.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::{D, DType, Device, Module, Result, Tensor};
use serde::Serialize;

use crate::gdn::RecurrentBatchKind;
use crate::kv_cache::{
    EitherCache, HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, KvCache,
    RecurrentLayerConfig, RecurrentStateSpec,
};
use crate::model::{IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{Activation, RmsNorm, embedding_with_legacy_tied_uqff},
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

serde_default_fn!(usize, default_max_position_embeddings, 1_048_576);

// the unweighted norms of the depth mix and the loop controllers
const UNIT_NORM_EPS: f64 = 1e-6;
// the L2 clip's guard against a zero norm
const CLIP_EPS: f64 = 1e-6;
// the router's guard against a zero sum of the selected scores
const ROUTE_NORM_MIN: f32 = 1e-20;
// the controllers per looped layer, in their message order
const CONTROLLERS: [&str; 4] = ["q", "k", "v", "ffn"];
const FFN_CONTROLLER: usize = 3;

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
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

    /// The cache slots: a layer's first pass in its own, each later pass of the looped core in one after them.
    pub fn cache_slots(&self) -> usize {
        self.num_hidden_layers + (self.loop_config.num_passes - 1) * self.core_len()
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
    fn conv_channels(&self) -> usize {
        2 * self.kv_dim() + 2 * self.hidden_size
    }

    fn validate(&self) -> Result<()> {
        let lc = &self.loop_config;
        let block = self.n_layers_per_attn_res_block;
        if lc.num_passes < 1
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
    let norm = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&norm)
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

/// One forward's block stacks for the depth mixes, by the stacked tensors' ids: a block's layers read the same
/// blocks, so they are stacked once. Kept for the forward only (a captured graph frees nothing it did not allocate).
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
        self.stacks.borrow_mut().push((ids, stacked.clone()));
        Ok(stacked)
    }
}

/// The depth mix: softmax attention of learned queries over the unit-normed states, `[b, t, queries, hidden]`.
fn mix(states: &[&Tensor], queries: &Tensor, out: DType, stacks: &BlockStacks) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some((current, blocks)) = states.split_last()
        && current.device().is_cuda()
    {
        let stacked = (!blocks.is_empty())
            .then(|| stacks.stacked(blocks))
            .transpose()?;
        if let Some(mixed) = inference_nn::cuda::inkling::depth_mix(
            stacked.as_ref(),
            current,
            queries,
            UNIT_NORM_EPS as f32,
            out,
        )? {
            return Ok(mixed);
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = stacks;
    let (b, t, hidden) = states[0].dims3()?;
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
    states: &[&Tensor],
    queries: &Tensor,
    i: usize,
    out: DType,
    stacks: &BlockStacks,
) -> Result<Tensor> {
    mix(states, queries, out, stacks)?
        .narrow(2, i, 1)?
        .squeeze(2)
}

/// A causal depthwise convolution with a residual, its last `width - 1` inputs carried in `[b, c, width - 1]`
/// states; `taps` adds input-dependent weights (a lag each) shared within a head.
struct ShortConv {
    // `[c, width]`, the last tap on the current input
    weight: Tensor,
    taps: Option<(Arc<dyn QuantMethod>, usize)>,
}

impl ShortConv {
    fn load(channels: usize, width: usize, vb: &ShardedVarBuilder) -> Result<Tensor> {
        vb.get((channels, width), "weight")?.to_dtype(DType::F32)
    }

    // `out_dtype`: the output's, f32 or x's
    fn forward(
        &self,
        x: &Tensor,
        dyn_input: &Tensor,
        state: &Tensor,
        out_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let (b, t, c) = x.dims3()?;
        let width = self.weight.dim(1)?;
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() {
            let taps = self
                .taps
                .as_ref()
                .map(|(proj, heads)| -> Result<_> {
                    Ok((
                        proj.forward(dyn_input)?
                            .to_dtype(DType::F32)?
                            .reshape((b, t, *heads, width))?,
                        *heads,
                    ))
                })
                .transpose()?;
            let taps = taps.as_ref().map(|(taps, heads)| (taps, *heads));
            if let Some(convolved) = inference_nn::cuda::inkling::short_conv(
                x,
                state,
                &self.weight,
                taps,
                out_dtype != DType::F32,
            )? {
                return Ok(convolved);
            }
        }
        let x = x.to_dtype(DType::F32)?;
        let ctx = Tensor::cat(&[&state.to_dtype(DType::F32)?.transpose(1, 2)?, &x], 1)?;
        let mut out = x.clone();
        for k in 0..width {
            let w = self.weight.narrow(1, k, 1)?.squeeze(1)?;
            out = (out + ctx.narrow(1, k, t)?.broadcast_mul(&w)?)?;
        }
        let mut terms = 1 + width;
        if let Some((proj, heads)) = &self.taps {
            let taps = proj
                .forward(dyn_input)?
                .to_dtype(DType::F32)?
                .reshape((b, t, *heads, width))?;
            for lag in 0..width {
                let window =
                    ctx.narrow(1, width - 1 - lag, t)?
                        .reshape((b, t, *heads, c / heads))?;
                let tap = taps.narrow(3, lag, 1)?;
                out = (out + window.broadcast_mul(&tap)?.reshape((b, t, c))?)?;
            }
            terms += width;
        }
        let new_state = ctx.narrow(1, t, width - 1)?.transpose(1, 2)?.contiguous()?;
        Ok((
            (out / (terms as f64).sqrt())?.to_dtype(out_dtype)?,
            new_state,
        ))
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    r_proj: Arc<dyn QuantMethod>,
    gate_proj: Arc<dyn QuantMethod>,
    sink_proj: Arc<dyn QuantMethod>,
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
            let cvb = vb.pp(name);
            let taps = if sliding {
                None
            } else {
                Some((
                    linear(hidden, hkv * width, cfg, cvb.pp("weight_proj"))?,
                    hkv,
                ))
            };
            Ok(ShortConv {
                weight: ShortConv::load(hkv * hd, width, &cvb)?,
                taps,
            })
        };
        Ok(Self {
            q_proj: linear(hidden, h * hd, cfg, vb.pp("q_proj"))?,
            k_proj: linear(hidden, hkv * hd, cfg, vb.pp("k_proj"))?,
            v_proj: linear(hidden, hkv * hd, cfg, vb.pp("v_proj"))?,
            r_proj: linear(hidden, h * cfg.d_rel, cfg, vb.pp("r_proj"))?,
            gate_proj: linear(hidden, h * hd, cfg, vb.pp("gate_proj"))?,
            sink_proj: linear(hidden, h, cfg, vb.pp("sink_proj"))?,
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
        conv_state: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, _) = inputs.q.dims3()?;
        let (h, hkv, hd) = (self.heads, self.kv_heads, self.head_dim);
        let kv_dim = hkv * hd;
        let k_state = conv_state.narrow(1, 0, kv_dim)?;
        let v_state = conv_state.narrow(1, kv_dim, kv_dim)?;

        let q = self.q_proj.forward(inputs.q)?.reshape((b, t, h, hd))?;
        let (k, k_state) = self.k_conv.forward(
            &self.k_proj.forward(inputs.k)?,
            inputs.k,
            &k_state,
            DType::F32,
        )?;
        let (v, v_state) = self.v_conv.forward(
            &self.v_proj.forward(inputs.v)?,
            inputs.v,
            &v_state,
            DType::F32,
        )?;
        let r = self
            .r_proj
            .forward(inputs.q)?
            .reshape((b, t, h, self.d_rel))?;
        let raw_gate = self.gate_proj.forward(inputs.q)?;
        let gate = || raw_gate.to_dtype(DType::F32);
        let sinks = self.sink_proj.forward(inputs.q)?.to_dtype(DType::F32)?;

        let mut q = l2_clip(&q, Some(&self.q_clip), cfg.qk_l2_clip_threshold)?;
        let k = l2_clip(&k.reshape((b, t, hkv, hd))?, None, cfg.qk_l2_clip_threshold)?;
        let v = unit_rms(&v.reshape((b, t, hkv, hd))?, cfg.rms_norm_eps)?;
        let mut r = l2_clip(&r, Some(&self.rel_clip), cfg.rel_l2_clip_threshold)?;
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
            {
                use inference_nn::cuda::biased_attention::{Keys, Past};
                let dtype = step.k_cache.dtype();
                step.writer.write_rows(
                    &k.to_dtype(dtype)?.transpose(1, 2)?,
                    &v.to_dtype(dtype)?.transpose(1, 2)?,
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
    down: Arc<dyn QuantMethod>,
    gain: Arc<dyn QuantMethod>,
    forget: Arc<dyn QuantMethod>,
}

impl Controller {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let (hidden, rank) = (cfg.hidden_size, cfg.loop_config.gate_rank);
        // f32 as stored: the controller runs on f32 unit-normed inputs whatever the model's dtype
        let vb = vb.set_dtype(DType::F32);
        Ok(Self {
            down: linear(2 * hidden, rank, cfg, vb.pp("down"))?,
            gain: linear(rank, hidden, cfg, vb.pp("gain"))?,
            forget: linear(rank, hidden, cfg, vb.pp("forget"))?,
        })
    }

    fn forward(&self, x: &Tensor, message: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        #[cfg(feature = "cuda")]
        if let Some(gate) =
            inference_nn::cuda::inkling::controller_in(x, message, UNIT_NORM_EPS as f32)?
        {
            let gate = self.down.forward(&gate)?;
            let gain = self.gain.forward(&gate)?;
            let forget = self.forget.forward(&gate)?;
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
        let gate = self.down.forward(&gate)?;
        let gain = self.gain.forward(&gate)?.to_dtype(DType::F32)?.tanh()?;
        let forget = self.forget.forward(&gate)?.to_dtype(DType::F32)?.tanh()?;
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
    conv_state: Tensor,
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
                taps: None,
            },
            ffn_norm: RmsNorm::new(hidden, eps, vb.pp("post_attention_layernorm"))?,
            moe: Moe::new(cfg, vb.pp("mlp"), layer_device, comm, loading_isq)?,
            post_mlp_conv: ShortConv {
                weight: ShortConv::load(hidden, width, &vb.pp("post_mlp_convolution"))?,
                taps: None,
            },
            block_start: layer.is_multiple_of(cfg.n_layers_per_attn_res_block),
            loop_layer,
        })
    }

    fn forward(
        &self,
        cfg: &Config,
        x: &Tensor,
        blocks: &mut Vec<Tensor>,
        messages: Option<&Tensor>,
        step: LayerStep<'_>,
    ) -> Result<(Tensor, Tensor)> {
        let LayerStep {
            kv_cache,
            paged,
            conv_state,
            rel,
            tau,
            stacks,
        } = step;
        let dtype = x.dtype();
        let states = |x: &Tensor, blocks: &[Tensor]| -> Vec<Tensor> {
            blocks.iter().chain(std::iter::once(x)).cloned().collect()
        };
        let mut inputs: [Tensor; 3] = match &self.attention_query {
            Some(query) => {
                let all = states(x, blocks);
                let all: Vec<&Tensor> = all.iter().collect();
                let m = mix(&all, query, dtype, stacks)?;
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

        let kv_dim = cfg.kv_dim();
        let hidden = cfg.hidden_size;
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
            &conv_state,
        )?;
        let (attn, attn_state) = self.post_attention_conv.forward(
            &attn,
            &attn,
            &conv_state.narrow(1, 2 * kv_dim, hidden)?,
            dtype,
        )?;
        let attn = attn.to_dtype(dtype)?;

        // a block's first layer starts a new residual stream; the one before it joins the mixed blocks
        let ffn_in = if self.block_start {
            blocks.push(x.clone());
            attn
        } else {
            (x + attn)?
        };
        let all = states(&ffn_in, blocks);
        let all: Vec<&Tensor> = all.iter().collect();
        let mut h = mixed(&all, &self.mlp_query, 0, dtype, stacks)?;
        if let Some((messages, lp)) = controllers {
            let message = messages.narrow(2, FFN_CONTROLLER, 1)?.squeeze(2)?;
            h = lp.controllers[FFN_CONTROLLER].forward(&h, &message)?;
        }
        let h = self.moe.forward(&self.ffn_norm.forward(&h)?)?;
        let (h, ffn_state) = self.post_mlp_conv.forward(
            &h,
            &h,
            &conv_state.narrow(1, 2 * kv_dim + hidden, hidden)?,
            dtype,
        )?;
        let out = (ffn_in + h.to_dtype(dtype)?)?;
        let state = Tensor::cat(&[&k_state, &v_state, &attn_state, &ffn_state], 1)?;
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
        let cache = HybridCache::new(
            HybridCacheConfig {
                layer_types,
                max_seq_len: cfg.max_position_embeddings,
                recurrent: RecurrentLayerConfig {
                    conv_dim: cfg.conv_channels(),
                    conv_width: cfg.sconv_kernel_size - 1,
                    state: RecurrentStateSpec::Opaque { dims: Vec::new() },
                    recurrent_dtype: None,
                },
                auxiliary: Vec::new(),
            },
            DType::F32,
            &layer_devices,
        )?;
        let world = mapper.get_comm_for(0)?.world_size();
        Ok(Self {
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
        let cfg = &self.cfg;
        let x = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
        let x = self.embed_norm.forward(&x)?;
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

        let stacks = BlockStacks::default();
        let mut run = |layer: usize,
                       slot: usize,
                       x: &Tensor,
                       blocks: &mut Vec<Tensor>,
                       messages: Option<&Tensor>|
         -> Result<Tensor> {
            let indices = cache
                .state_indices_for_layer(slots + slot)?
                .ok_or_else(|| {
                    inference_tensor::Error::msg("Inkling 2: missing recurrent state indices")
                })?;
            let conv_state = {
                let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(slots + slot) else {
                    inference_tensor::bail!("Inkling 2: slot {slot} has no convolution state");
                };
                let state = pool.gather_conv_state(&indices)?;
                if fresh_state {
                    state.zeros_like()?
                } else {
                    state
                }
            };
            let x = self.mapper.map(x.clone(), layer)?;
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
                Some((writer, positions, Some(caches))) => {
                    let (k_cache, v_cache) = &caches[slot];
                    Some(PagedStep {
                        k_cache,
                        v_cache,
                        positions,
                        writer,
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
            let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(slots + slot) else {
                inference_tensor::bail!("Inkling 2: slot {slot} has no convolution state");
            };
            pool.scatter_conv_state_with_host_indices(
                &indices,
                recurrent.state_indices_host(),
                &state,
            )?;
            Ok(out)
        };

        let lc = &cfg.loop_config;
        let mut blocks = Vec::new();
        let mut x = x;
        for layer in 0..lc.core_start {
            x = run(layer, layer, &x, &mut blocks, None)?;
        }
        // every pass starts from the prelude's output
        let prelude = x.clone();
        let prelude_blocks = blocks.clone();
        for pass in 0..lc.num_passes {
            // the previous pass's bank: its blocks after the first core block (the prelude's output), and its output
            let bank = (pass > 0).then(|| {
                let mut bank = blocks[prelude_blocks.len() + 1..].to_vec();
                bank.push(x.clone());
                bank
            });
            if pass > 0 {
                x = prelude.clone();
                blocks = prelude_blocks.clone();
            }
            let core = &self.layers[lc.core_start..lc.core_end];
            for (layer, decoder) in (lc.core_start..).zip(core) {
                let slot = slot_of(cfg, pass, layer);
                let messages = match &bank {
                    Some(bank) => {
                        let lp = decoder.loop_layer.as_ref().ok_or_else(|| {
                            inference_tensor::Error::msg(
                                "Inkling 2: a looped layer has no loop weights",
                            )
                        })?;
                        let bank: Vec<&Tensor> = bank.iter().collect();
                        Some(mix(&bank, &lp.queries, DType::F32, &stacks)?)
                    }
                    None => None,
                };
                x = run(layer, slot, &x, &mut blocks, messages.as_ref())?;
            }
        }
        for layer in lc.core_end..cfg.num_hidden_layers {
            x = run(layer, layer, &x, &mut blocks, None)?;
        }
        drop(cache);

        let x = self.mapper.cast_nm_device(&x, false)?;
        let mut states: Vec<&Tensor> = blocks.iter().collect();
        states.push(&x);
        let x = mixed(&states, &self.final_query, 0, self.dtype, &stacks)?;
        let x = self.norm.forward(&x)?;
        let x = ctx.logits(&x)?;
        let logits = ctx.lm_head(&*self.lm_head, &x)?;
        if logits.dim(D::Minus1)? != cfg.vocab_size {
            return Ok(logits);
        }
        let cap = cfg.final_logit_softcapping;
        let logits = ((logits.to_dtype(DType::F32)? / cap)?.tanh()? * cap)?;
        logits
            .narrow(D::Minus1, 0, cfg.unpadded_vocab_size)?
            .contiguous()
    }
}

/// The weights a cache slot runs: its own layer's, or, for a later pass of the core, the core layer's.
fn physical_layer(cfg: &Config, slot: usize) -> usize {
    let n = cfg.num_hidden_layers;
    if slot < n {
        slot
    } else {
        cfg.loop_config.core_start + (slot - n) % cfg.core_len()
    }
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

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
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

    // a decode step reads its positions and lengths from the paged metadata's device tensors
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
