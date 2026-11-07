//! RWKV-7 ("Goose", `RWKV7ForCausalLM` in flash-linear-attention's layout): every layer a time-mix (the WKV-7
//! generalized delta rule over per-head states, with a decay per key channel) and a channel-mix (a squared-ReLU MLP),
//! each reading the previous token through a token shift. No attention: each layer's state is its two shifted inputs
//! (the hybrid cache's conv slot, `2 * hidden` wide and one step deep) and its WKV matrices.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::nn::LayerNorm;
use inference_tensor::{DType, Module, Result, Tensor};
use serde::Serialize;

use crate::model::NormalLoadingMetadata;
use crate::rwkv::{RwkvBlock, RwkvModel, RwkvNames, RwkvShape};
use crate::serde_default_fn;
use crate::utils::unvarbuilder::UnVarBuilder;
// the six token-shift mixes in the order the time-mix reads them
const TIME_MIXES: [&str; 6] = ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"];

serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(usize, default_head_dim, 64);

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
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn heads(&self) -> usize {
        self.hidden_size / self.head_dim
    }

    pub fn ffn_size(&self) -> usize {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
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
        let (mixed, new_last) = crate::ops::rwkv_token_shift(x, last, &self.mixes, None)?;
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
        let (mixed, new_last) = crate::ops::rwkv_token_shift(x, last, &self.mix, None)?;
        let out = self
            .value
            .forward(&self.key.forward(&mixed.get(0)?)?.relu()?.sqr()?)?;
        Ok((out, new_last))
    }
}

struct Layer {
    attn_norm: LayerNorm,
    ffn_norm: LayerNorm,
    attn: TimeMix,
    ffn: ChannelMix,
}

impl RwkvBlock for Layer {
    fn forward(
        &self,
        x: &Tensor,
        [attn_shift, ffn_shift]: [&Tensor; 2],
        state: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<[Tensor; 3]> {
        let xn = self.attn_norm.forward(x)?;
        let (out, attn_last) = self.attn.forward(&xn, attn_shift, state, v_first)?;
        let x = (x + out)?;
        let xn = self.ffn_norm.forward(&x)?;
        let (out, ffn_last) = self.ffn.forward(&xn, ffn_shift)?;
        Ok([(x + out)?, attn_last, ffn_last])
    }

    fn residual_tensors(&self, l: &UnVarBuilder) {
        l.pp("attn_norm").add(&self.attn_norm);
        l.pp("ffn_norm").add(&self.ffn_norm);
        let a = l.pp("attn");
        for (i, name) in TIME_MIXES.iter().enumerate() {
            let mix = self
                .attn
                .mixes
                .get(i)
                .and_then(|m| m.reshape((1, 1, m.dim(0)?)));
            if let Ok(mix) = mix {
                a.add_tensor(name, mix);
            }
        }
        for (name, lora) in [
            ("w_lora", Some(&self.attn.w_lora)),
            ("a_lora", Some(&self.attn.a_lora)),
            ("v_lora", self.attn.v_lora.as_ref()),
            ("g_lora", Some(&self.attn.g_lora)),
        ] {
            if let Some(lora) = lora {
                let p = a.pp(name).pp("lora");
                p.pp(0).add(&lora.down);
                p.pp(2).add(&lora.up);
            }
        }
        a.add_tensor("k_k", self.attn.k_k.clone());
        a.add_tensor("k_a", self.attn.k_a.clone());
        a.add_tensor("r_k", self.attn.r_k.clone());
        a.pp("g_norm")
            .add_tensor("weight", self.attn.g_norm.0.clone());
        a.pp("g_norm")
            .add_tensor("bias", self.attn.g_norm.1.clone());
        if let Ok(mix) = self.ffn.mix.flatten_all() {
            l.pp("ffn").add_tensor("x_k", mix);
        }
    }
}

const NAMES: RwkvNames = RwkvNames {
    root: "model",
    embeddings: "embeddings",
    norm: "norm",
    layers: "layers",
    head: "lm_head",
};

impl Config {
    pub(crate) fn shape(&self) -> RwkvShape<'_> {
        RwkvShape {
            vocab: self.vocab_size,
            hidden: self.hidden_size,
            heads: self.heads(),
            head_dim: self.head_dim,
            layers: self.num_hidden_layers,
            tie_word_embeddings: self.tie_word_embeddings,
            norm_eps: self.norm_eps,
            quantization_config: &self.quantization_config,
            rescaled_every: None,
        }
    }
}

/// RWKV-7 as the shared RWKV stack, in flash-linear-attention's layout.
pub fn model(
    cfg: &Config,
    vb: ShardedVarBuilder,
    meta: NormalLoadingMetadata,
) -> Result<RwkvModel> {
    let shape = cfg.shape();
    RwkvModel::new(&shape, NAMES, "pre_norm", vb, meta, |layer, vb| {
        Ok(Box::new(Layer {
            attn_norm: shape.layer_norm(vb.pp("attn_norm"))?,
            ffn_norm: shape.layer_norm(vb.pp("ffn_norm"))?,
            attn: TimeMix::load(cfg, layer, vb.pp("attn"))?,
            ffn: ChannelMix::load(cfg, vb.pp("ffn"))?,
        }) as Box<dyn RwkvBlock>)
    })
}
