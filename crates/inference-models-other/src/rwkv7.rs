//! RWKV-7 ("Goose", `RWKV7ForCausalLM` in flash-linear-attention's layout): every layer a time-mix (the WKV-7
//! generalized delta rule over per-head states, with a decay per key channel) and a channel-mix (a squared-ReLU MLP),
//! each reading the previous token through a token shift. No attention: each layer's state is its two shifted inputs
//! (the hybrid cache's conv slot, `2 * hidden` wide and one step deep) and its WKV matrices.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::nn::LayerNorm;
use inference_tensor::{D, DType, Device, Module, Result, Tensor};
use serde::Serialize;

use crate::gdn::RecurrentBatchKind;
use crate::kv_cache::{
    EitherCache, HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType,
    RecurrentLayerConfig, RecurrentStateSpec,
};
use crate::layers::embedding_with_legacy_tied_uqff;
use crate::model::{IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel};
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};
use crate::serde_default_fn;
use crate::utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder};

// fla's `w = -exp(-0.5) * sigmoid(..)`, so the decay `exp(w)` stays in (0.545, 1)
const DECAY_SCALE: f64 = -0.606_530_659_712_633_4;
// F.normalize's floor on a head's key norm
const KK_NORM_EPS: f64 = 1e-12;
/// The context the planner may offer a sequence: RWKV has no positions.
pub const MAX_POSITIONS: usize = 1 << 20;
// the six token-shift mixes in the order the time-mix reads them
const TIME_MIXES: [&str; 6] = ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"];

serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(usize, default_head_dim, 64);
serde_default_fn!(bool, default_true, true);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    pub decay_low_rank_dim: usize,
    pub a_low_rank_dim: usize,
    pub v_low_rank_dim: usize,
    pub gate_low_rank_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_true")]
    pub norm_bias: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn heads(&self) -> usize {
        self.hidden_size / self.head_dim
    }

    pub fn ffn_size(&self) -> usize {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
    }

    /// A sequence's state per layer: the two shifted inputs and the heads' WKV matrices in f32.
    pub fn recurrent_state_bytes(&self, dtype: DType) -> usize {
        self.num_hidden_layers
            * (2 * self.hidden_size * dtype.size_in_bytes()
                + self.heads() * self.head_dim * self.head_dim * DType::F32.size_in_bytes())
    }
}

/// A low-rank projection, `up(act(down(x))) (+ bias)`, as fla's `LoRA` holds it.
struct LoRA {
    down: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
}

impl LoRA {
    fn load(cfg: &Config, rank: usize, bias: bool, vb: ShardedVarBuilder) -> Result<Self> {
        let vb = vb.pp("lora");
        let h = cfg.hidden_size;
        Ok(Self {
            down: ReplicatedLayer::new(h, rank, &cfg.quantization_config, false, vb.pp(0))?,
            up: ReplicatedLayer::new(rank, h, &cfg.quantization_config, bias, vb.pp(2))?,
        })
    }

    fn forward(&self, x: &Tensor, act: Option<fn(&Tensor) -> Result<Tensor>>) -> Result<Tensor> {
        let mid = self.down.forward(x)?;
        let mid = match act {
            Some(act) => act(&mid)?,
            None => mid,
        };
        self.up.forward(&mid)
    }
}

struct TimeMix {
    mixes: [Tensor; 6],
    r_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    w_lora: LoRA,
    a_lora: LoRA,
    // the first layer's value is every later layer's residual, so it has no mix of its own
    v_lora: Option<LoRA>,
    g_lora: LoRA,
    k_k: Tensor,
    k_a: Tensor,
    r_k: Tensor,
    g_norm: (Tensor, Tensor),
    g_norm_eps: f64,
    heads: usize,
    head_dim: usize,
}

impl TimeMix {
    fn load(cfg: &Config, layer: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let q = &cfg.quantization_config;
        let proj = |name: &str| ReplicatedLayer::new(h, h, q, false, vb.pp(name));
        let mixes = TIME_MIXES.map(|name| vb.get((1, 1, h), name));
        let [r, w, k, v, a, g] = mixes;
        // the vectors the f32 part of a step reads, converted once
        let f32 = |t: Tensor| t.to_dtype(DType::F32);
        Ok(Self {
            mixes: [r?, w?, k?, v?, a?, g?],
            r_proj: proj("r_proj")?,
            k_proj: proj("k_proj")?,
            v_proj: proj("v_proj")?,
            o_proj: proj("o_proj")?,
            w_lora: LoRA::load(cfg, cfg.decay_low_rank_dim, true, vb.pp("w_lora"))?,
            a_lora: LoRA::load(cfg, cfg.a_low_rank_dim, true, vb.pp("a_lora"))?,
            v_lora: (layer > 0)
                .then(|| LoRA::load(cfg, cfg.v_low_rank_dim, true, vb.pp("v_lora")))
                .transpose()?,
            g_lora: LoRA::load(cfg, cfg.gate_low_rank_dim, false, vb.pp("g_lora"))?,
            k_k: f32(vb.get(h, "k_k")?)?,
            k_a: f32(vb.get(h, "k_a")?)?,
            r_k: f32(vb.get((cfg.heads(), cfg.head_dim), "r_k")?)?,
            g_norm: (
                f32(vb.pp("g_norm").get(h, "weight")?)?,
                f32(vb.pp("g_norm").get(h, "bias")?)?,
            ),
            g_norm_eps: cfg.head_dim as f64 * cfg.norm_eps,
            heads: cfg.heads(),
            head_dim: cfg.head_dim,
        })
    }

    // `x` normed `(b, s, h)`, `prev` the inputs shifted by one token, `state` the heads' WKV matrices (advanced),
    // `v_first` the first layer's value (set by it)
    fn forward(
        &self,
        x: &Tensor,
        prev: &Tensor,
        state: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let shifted = (prev - x)?;
        let mixed = |i: usize| x + shifted.broadcast_mul(&self.mixes[i])?;
        let [xr, xw, xk, xv, xa, xg] = [0, 1, 2, 3, 4, 5].map(mixed);
        let (xr, xw, xk, xv, xa, xg) = (xr?, xw?, xk?, xv?, xa?, xg?);
        let f32 = |t: Tensor| t.to_dtype(DType::F32);
        let r = f32(self.r_proj.forward(&xr)?)?;
        let w = f32(self.w_lora.forward(&xw, Some(Tensor::tanh))?)?;
        let w = (inference_tensor::nn::ops::sigmoid(&w)? * DECAY_SCALE)?.exp()?;
        let k = f32(self.k_proj.forward(&xk)?)?;
        let mut v = f32(self.v_proj.forward(&xv)?)?;
        match (&self.v_lora, v_first.as_ref()) {
            (Some(lora), Some(first)) => {
                let mix = inference_tensor::nn::ops::sigmoid(&f32(lora.forward(&xv, None)?)?)?;
                v = (&v + (first - &v)?.mul(&mix)?)?;
            }
            _ => *v_first = Some(v.clone()),
        }
        let a = inference_tensor::nn::ops::sigmoid(&f32(self.a_lora.forward(&xa, None)?)?)?;
        let g = f32(self
            .g_lora
            .forward(&xg, Some(inference_tensor::nn::ops::sigmoid))?)?;
        let heads = |t: &Tensor| t.reshape((b, s, self.heads, self.head_dim));
        let kk = heads(&k.broadcast_mul(&self.k_k)?)?;
        let norm = kk
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .maximum(KK_NORM_EPS)?;
        let kk = kk.broadcast_div(&norm)?;
        // k (1 + (a - 1) k_a)
        let k = (&k + (&a - 1.0)?.broadcast_mul(&self.k_a)?.mul(&k)?)?;
        let (r, w, k, v, a) = (heads(&r)?, heads(&w)?, heads(&k)?, heads(&v)?, heads(&a)?);
        let y = crate::ops::rwkv7_wkv(
            crate::ops::Wkv7 {
                r: &r.contiguous()?,
                w: &w.contiguous()?,
                k: &k.contiguous()?,
                v: &v.contiguous()?,
                a: &kk.neg()?.contiguous()?,
                b: &kk.mul(&a)?.contiguous()?,
            },
            state,
        )?;
        // the group norm over each head
        let mean = y.mean_keepdim(D::Minus1)?;
        let centered = y.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let y = centered.broadcast_div(&(var + self.g_norm_eps)?.sqrt()?)?;
        let y = y.reshape((b, s, h))?;
        let y = y
            .broadcast_mul(&self.g_norm.0)?
            .broadcast_add(&self.g_norm.1)?;
        // the bonus: each head's (r . k r_k) times its value
        let rk = r
            .mul(&k)?
            .broadcast_mul(&self.r_k)?
            .sum_keepdim(D::Minus1)?;
        let y = (y + rk.broadcast_mul(&v)?.reshape((b, s, h))?)?.mul(&g)?;
        self.o_proj.forward(&y.to_dtype(x.dtype())?)
    }
}

struct ChannelMix {
    mix: Tensor,
    key: Arc<dyn QuantMethod>,
    value: Arc<dyn QuantMethod>,
}

impl ChannelMix {
    fn load(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let (h, ffn) = (cfg.hidden_size, cfg.ffn_size());
        let q = &cfg.quantization_config;
        Ok(Self {
            mix: vb.get(h, "x_k")?,
            key: ReplicatedLayer::new(h, ffn, q, false, vb.pp("key"))?,
            value: ReplicatedLayer::new(ffn, h, q, false, vb.pp("value"))?,
        })
    }

    fn forward(&self, x: &Tensor, prev: &Tensor) -> Result<Tensor> {
        let xk = (x + (prev - x)?.broadcast_mul(&self.mix)?)?;
        self.value.forward(&self.key.forward(&xk)?.relu()?.sqr()?)
    }
}

struct Layer {
    pre_norm: Option<LayerNorm>,
    attn_norm: LayerNorm,
    ffn_norm: LayerNorm,
    attn: TimeMix,
    ffn: ChannelMix,
}

fn layer_norm(cfg: &Config, vb: ShardedVarBuilder) -> Result<LayerNorm> {
    let config = inference_tensor::nn::LayerNormConfig {
        eps: cfg.norm_eps,
        affine: cfg.norm_bias,
        ..Default::default()
    };
    crate::layers::layer_norm(cfg.hidden_size, config, vb)
}

// the inputs shifted by one token: the state's last token, then all but the sequence's last
fn shift(x: &Tensor, last: &Tensor) -> Result<Tensor> {
    let s = x.dim(1)?;
    let last = last.unsqueeze(1)?.to_dtype(x.dtype())?;
    if s == 1 {
        return Ok(last);
    }
    Tensor::cat(&[last, x.narrow(1, 0, s - 1)?], 1)
}

pub struct Model {
    embeddings: Arc<dyn QuantMethod>,
    layers: Vec<Layer>,
    norm: LayerNorm,
    lm_head: Arc<dyn QuantMethod>,
    dtype: DType,
    hidden: usize,
    cache: EitherCache,
    device: Device,
    mapper: Box<dyn crate::device_map::DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !cfg.hidden_size.is_multiple_of(cfg.head_dim) {
            inference_tensor::bail!(
                "RWKV-7 hidden size {} is not a whole number of {}-wide heads",
                cfg.hidden_size,
                cfg.head_dim
            );
        }
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        let mapper = normal_loading_metadata.mapper;
        let loading_isq = normal_loading_metadata.loading_isq;
        let dtype = vb_m.dtype();
        let embeddings = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embeddings"), loading_isq),
            cfg.tie_word_embeddings
                .then(|| mapper.set_nm_device(vb_lm_head.clone(), loading_isq)),
            &cfg.quantization_config,
        )?;
        let norm = layer_norm(cfg, mapper.set_nm_device(vb_m.pp("norm"), false))?;
        let lm_head = if cfg.tie_word_embeddings {
            embeddings.clone()
        } else {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, loading_isq),
            )?
        };
        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer| {
            let vb = mapper.set_device(layer, vb_l.pp(layer), loading_isq);
            Ok(Layer {
                pre_norm: (layer == 0)
                    .then(|| layer_norm(cfg, vb.pp("pre_norm")))
                    .transpose()?,
                attn_norm: layer_norm(cfg, vb.pp("attn_norm"))?,
                ffn_norm: layer_norm(cfg, vb.pp("ffn_norm"))?,
                attn: TimeMix::load(cfg, layer, vb.pp("attn"))?,
                ffn: ChannelMix::load(cfg, vb.pp("ffn"))?,
            })
        })?;
        let layer_devices = (0..cfg.num_hidden_layers)
            .map(|layer| {
                mapper
                    .device_for(layer, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone()
            })
            .collect::<Vec<_>>();
        let cache = HybridCache::new(
            HybridCacheConfig {
                layer_types: vec![HybridLayerType::Recurrent; cfg.num_hidden_layers],
                max_seq_len: MAX_POSITIONS,
                recurrent: RecurrentLayerConfig {
                    conv_dim: 2 * cfg.hidden_size,
                    conv_width: 1,
                    state: RecurrentStateSpec::Opaque {
                        dims: vec![cfg.heads(), cfg.head_dim, cfg.head_dim],
                    },
                    recurrent_dtype: Some(DType::F32),
                },
                auxiliary: Vec::new(),
            },
            dtype,
            &layer_devices,
        )?;
        Ok(Self {
            embeddings,
            layers,
            norm,
            lm_head,
            dtype,
            hidden: cfg.hidden_size,
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(cache))),
            device: normal_loading_metadata.real_device,
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: MAX_POSITIONS,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: 1,
                num_attn_heads: 1,
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
        })
    }

    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let mut x = self.embeddings.embedding_forward(input_ids, self.dtype)?;
        let metadata = ctx.recurrent_metadata().cloned().ok_or_else(|| {
            inference_tensor::Error::msg("RWKV-7 needs the hybrid cache's recurrent metadata")
        })?;
        let fresh =
            metadata.batch_kind() == RecurrentBatchKind::Prefill && ctx.is_first_prompt_chunk();
        let mut cache = self.cache.hybrid();
        let mut v_first = None;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;
            let indices = cache.state_indices_for_layer(layer_idx)?.ok_or_else(|| {
                inference_tensor::Error::msg(format!(
                    "RWKV-7 layer {layer_idx} has no recurrent state indices"
                ))
            })?;
            let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(layer_idx) else {
                inference_tensor::bail!("RWKV-7 layer {layer_idx} is not recurrent in the cache");
            };
            // the shifts `(b, 2h, 1)` and the WKV matrices, copied so the update never writes a snapshot's rows
            let (mut shifts, state) = if fresh {
                let rows = indices.dim(0)?;
                (
                    Tensor::zeros((rows, 2 * self.hidden, 1), x.dtype(), x.device())?,
                    Tensor::zeros(
                        pool.recurrent_state.dims()[1..].to_vec(),
                        DType::F32,
                        x.device(),
                    )?
                    .unsqueeze(0)?
                    .repeat((rows, 1, 1, 1))?,
                )
            } else {
                (
                    pool.gather_conv_state(&indices)?,
                    pool.gather_recurrent_state(&indices)?.copy()?,
                )
            };
            if let Some(pre_norm) = &layer.pre_norm {
                x = pre_norm.forward(&x)?;
            }
            let last = |t: &Tensor| t.narrow(1, t.dim(1)? - 1, 1)?.squeeze(1);
            let xn = layer.attn_norm.forward(&x)?;
            let prev = shift(&xn, &shifts.narrow(1, 0, self.hidden)?.squeeze(2)?)?;
            let attn_last = last(&xn)?;
            x = (&x + layer.attn.forward(&xn, &prev, &state, &mut v_first)?)?;
            let xn = layer.ffn_norm.forward(&x)?;
            let prev = shift(
                &xn,
                &shifts.narrow(1, self.hidden, self.hidden)?.squeeze(2)?,
            )?;
            let ffn_last = last(&xn)?;
            x = (&x + layer.ffn.forward(&xn, &prev)?)?;
            shifts = Tensor::cat(&[attn_last, ffn_last], 1)?
                .unsqueeze(2)?
                .to_dtype(shifts.dtype())?;
            pool.scatter_conv_state_with_host_indices(
                &indices,
                metadata.state_indices_host(),
                &shifts,
            )?;
            pool.scatter_recurrent_state_with_host_indices(
                &indices,
                metadata.state_indices_host(),
                &state,
            )?;
        }
        let x = self.mapper.cast_nm_device(&x, false)?;
        let x = self.norm.forward(&x)?;
        let x = ctx.logits(&x)?;
        ctx.lm_head(&*self.lm_head, &x)
    }
}

impl IsqModel for Model {
    // the norms, the token-shift mixes, the LoRAs and the small vectors stay as the checkpoint has them
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embeddings").add(&self.embeddings);
        uvb_m.pp("norm").add(&self.norm);
        for (i, layer) in self.layers.iter().enumerate() {
            let l = uvb_m.pp("layers").pp(i);
            if let Some(pre_norm) = &layer.pre_norm {
                l.pp("pre_norm").add(pre_norm);
            }
            l.pp("attn_norm").add(&layer.attn_norm);
            l.pp("ffn_norm").add(&layer.ffn_norm);
            let a = l.pp("attn");
            for (name, mix) in TIME_MIXES.iter().zip(&layer.attn.mixes) {
                a.add_tensor(name, mix.clone());
            }
            for (name, lora) in [
                ("w_lora", Some(&layer.attn.w_lora)),
                ("a_lora", Some(&layer.attn.a_lora)),
                ("v_lora", layer.attn.v_lora.as_ref()),
                ("g_lora", Some(&layer.attn.g_lora)),
            ] {
                if let Some(lora) = lora {
                    let p = a.pp(name).pp("lora");
                    p.pp(0).add(&lora.down);
                    p.pp(2).add(&lora.up);
                }
            }
            a.add_tensor("k_k", layer.attn.k_k.clone());
            a.add_tensor("k_a", layer.attn.k_a.clone());
            a.add_tensor("r_k", layer.attn.r_k.clone());
            a.pp("g_norm")
                .add_tensor("weight", layer.attn.g_norm.0.clone());
            a.pp("g_norm")
                .add_tensor("bias", layer.attn.g_norm.1.clone());
            l.pp("ffn").add_tensor("x_k", layer.ffn.mix.clone());
        }
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
        MAX_POSITIONS
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for Model {}
