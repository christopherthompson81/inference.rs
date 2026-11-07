//! RWKV-7 ("Goose", `RWKV7ForCausalLM` in flash-linear-attention's layout): every layer a time-mix (the WKV-7
//! generalized delta rule over per-head states, with a decay per key channel) and a channel-mix (a squared-ReLU MLP),
//! each reading the previous token through a token shift. No attention: each layer's state is its two shifted inputs
//! (the hybrid cache's conv slot, `2 * hidden` wide and one step deep) and its WKV matrices.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::nn::LayerNorm;
use inference_tensor::{DType, Device, Module, Result, Tensor};
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
    // the six token-shift mixes stacked `(6, hidden)`, in `TIME_MIXES`' order
    mixes: Tensor,
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
        let mixes = TIME_MIXES
            .iter()
            .map(|name| vb.get((1, 1, h), name)?.reshape(h))
            .collect::<Result<Vec<_>>>()?;
        // the vectors the f32 part of a step reads, converted once
        let f32 = |t: Tensor| t.to_dtype(DType::F32);
        Ok(Self {
            mixes: Tensor::stack(&mixes, 0)?,
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

    // `x` normed `(b, s, h)`, `last` the token before it per sequence `(b, h)`, `state` the heads' WKV matrices
    // (advanced), `v_first` the first layer's value (set by it); returns the output and `x`'s last token
    fn forward(
        &self,
        x: &Tensor,
        last: &Tensor,
        state: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (mixed, new_last) = crate::ops::rwkv7_token_shift(x, last, &self.mixes)?;
        let [xr, xw, xk, xv, xa, xg] = [0, 1, 2, 3, 4, 5].map(|i| mixed.get(i));
        let (xr, xw, xk, xv, xa, xg) = (xr?, xw?, xk?, xv?, xa?, xg?);
        let r = self.r_proj.forward(&xr)?;
        let w = self.w_lora.forward(&xw, Some(Tensor::tanh))?;
        let k = self.k_proj.forward(&xk)?;
        let v = self.v_proj.forward(&xv)?;
        let a = self.a_lora.forward(&xa, None)?;
        let g = self
            .g_lora
            .forward(&xg, Some(inference_tensor::nn::ops::sigmoid))?;
        let v_mix = match (&self.v_lora, v_first.as_ref()) {
            (Some(lora), Some(_)) => Some(lora.forward(&xv, None)?),
            _ => {
                *v_first = Some(v.clone());
                None
            }
        };
        let y = crate::ops::rwkv7_time_mix(
            crate::ops::TimeMix7 {
                r: &r,
                w: &w,
                k: &k,
                v: &v,
                a: &a,
                g: &g,
                v_mix: v_mix
                    .as_ref()
                    .zip(v_first.as_ref())
                    .map(|(mix, first)| (first, mix)),
                k_k: &self.k_k,
                k_a: &self.k_a,
                r_k: &self.r_k,
                norm_w: &self.g_norm.0,
                norm_b: &self.g_norm.1,
                norm_eps: self.g_norm_eps,
                heads: self.heads,
                head_dim: self.head_dim,
            },
            state,
        )?;
        Ok((self.o_proj.forward(&y.to_dtype(x.dtype())?)?, new_last))
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
            mix: vb.get(h, "x_k")?.reshape((1, h))?,
            key: ReplicatedLayer::new(h, ffn, q, false, vb.pp("key"))?,
            value: ReplicatedLayer::new(ffn, h, q, false, vb.pp("value"))?,
        })
    }

    // returns the output and `x`'s last token
    fn forward(&self, x: &Tensor, last: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mixed, new_last) = crate::ops::rwkv7_token_shift(x, last, &self.mix)?;
        let out = self
            .value
            .forward(&self.key.forward(&mixed.get(0)?)?.relu()?.sqr()?)?;
        Ok((out, new_last))
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
            let xn = layer.attn_norm.forward(&x)?;
            let shift = |at: usize| shifts.narrow(1, at, self.hidden)?.squeeze(2);
            let (out, attn_last) = layer.attn.forward(&xn, &shift(0)?, &state, &mut v_first)?;
            x = (&x + out)?;
            let xn = layer.ffn_norm.forward(&x)?;
            let (out, ffn_last) = layer.ffn.forward(&xn, &shift(self.hidden)?)?;
            x = (&x + out)?;
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
            for (i, name) in TIME_MIXES.iter().enumerate() {
                let mix = layer
                    .attn
                    .mixes
                    .get(i)
                    .and_then(|m| m.reshape((1, 1, m.dim(0)?)));
                if let Ok(mix) = mix {
                    a.add_tensor(name, mix);
                }
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
            if let Ok(mix) = layer.ffn.mix.flatten_all() {
                l.pp("ffn").add_tensor("x_k", mix);
            }
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
