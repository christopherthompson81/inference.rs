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

/// A time-mix's names: fla's (`r_proj`, `Linear` LoRAs) or BlinkDL's (`receptance`, `w1`/`w2` matrices), as ARWKV's.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    Fla,
    Blink,
}

impl Layout {
    // the receptance, key, value and output projections
    fn projections(self) -> [&'static str; 4] {
        match self {
            Self::Fla => ["r_proj", "k_proj", "v_proj", "o_proj"],
            Self::Blink => ["receptance", "key", "value", "output"],
        }
    }

    fn norm(self) -> &'static str {
        match self {
            Self::Fla => "g_norm",
            Self::Blink => "ln_x",
        }
    }
}

/// What a time-mix holds beyond its widths.
pub(crate) struct TimeMixSpec<'a> {
    pub layout: Layout,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub gated: bool,
    pub group_norm: bool,
    pub group_norm_eps: f64,
    /// fla's decay, rate, value and gate LoRA ranks (BlinkDL's matrices carry their own)
    pub ranks: [usize; 4],
    pub quantization_config: &'a Option<QuantizedConfig>,
}

/// A low-rank projection, `up(act(down(x))) (+ bias)`: fla's `LoRA` (`Linear`s) or BlinkDL's matrices (`x @ w1`).
enum LoRA {
    Linear {
        down: Arc<dyn QuantMethod>,
        up: Arc<dyn QuantMethod>,
    },
    Matrix {
        down: Tensor,
        up: Tensor,
        bias: Option<Tensor>,
    },
}

impl LoRA {
    // `name` the LoRA's letter (`w`, `a`, `v`, `g`)
    fn load(
        spec: &TimeMixSpec<'_>,
        name: &str,
        rank: usize,
        bias: bool,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let h = spec.hidden;
        Ok(match spec.layout {
            Layout::Fla => {
                let q = spec.quantization_config;
                let vb = vb.pp(format!("{name}_lora")).pp("lora");
                Self::Linear {
                    down: ReplicatedLayer::new(h, rank, q, false, vb.pp(0))?,
                    up: ReplicatedLayer::new(rank, h, q, bias, vb.pp(2))?,
                }
            }
            Layout::Blink => Self::Matrix {
                down: vb.get_unchecked(&format!("{name}1"))?,
                up: vb.get_unchecked(&format!("{name}2"))?,
                bias: bias
                    .then(|| vb.get((1, 1, h), &format!("{name}0"))?.reshape(h))
                    .transpose()?,
            },
        })
    }

    fn forward(&self, x: &Tensor, act: Option<fn(&Tensor) -> Result<Tensor>>) -> Result<Tensor> {
        let mid = match self {
            Self::Linear { down, .. } => down.forward(x)?,
            Self::Matrix { down, .. } => x.broadcast_matmul(down)?,
        };
        let mid = match act {
            Some(act) => act(&mid)?,
            None => mid,
        };
        match self {
            Self::Linear { up, .. } => up.forward(&mid),
            Self::Matrix { up, bias, .. } => {
                let out = mid.broadcast_matmul(up)?;
                match bias {
                    Some(bias) => out.broadcast_add(bias),
                    None => Ok(out),
                }
            }
        }
    }

    fn residual_tensors(&self, name: &str, a: &UnVarBuilder) {
        match self {
            Self::Linear { down, up } => {
                let p = a.pp(format!("{name}_lora")).pp("lora");
                p.pp(0).add(down);
                p.pp(2).add(up);
            }
            Self::Matrix { down, up, bias } => {
                a.add_tensor(format!("{name}1"), down.clone());
                a.add_tensor(format!("{name}2"), up.clone());
                if let Some(Ok(bias)) = bias.as_ref().map(|b| b.unsqueeze(0)?.unsqueeze(0)) {
                    a.add_tensor(format!("{name}0"), bias);
                }
            }
        }
    }
}

pub(crate) struct TimeMix {
    layout: Layout,
    // the token-shift mixes stacked `(n, hidden)`, in `TIME_MIXES`' order (no `x_g` without a gate)
    mixes: Tensor,
    r_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    w_lora: LoRA,
    a_lora: LoRA,
    // the first layer's value is every later layer's residual, so it has no mix of its own
    v_lora: Option<LoRA>,
    g_lora: Option<LoRA>,
    k_k: Tensor,
    k_a: Tensor,
    r_k: Tensor,
    g_norm: Option<(Tensor, Tensor)>,
    g_norm_eps: f64,
    heads: usize,
    head_dim: usize,
}

impl TimeMix {
    pub(crate) fn load(
        spec: &TimeMixSpec<'_>,
        layer: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let h = spec.hidden;
        let q = spec.quantization_config;
        let proj = |name: &str| ReplicatedLayer::new(h, h, q, false, vb.pp(name));
        let [r_proj, k_proj, v_proj, o_proj] = spec.layout.projections();
        let mixes = TIME_MIXES[..TIME_MIXES.len() - usize::from(!spec.gated)]
            .iter()
            .map(|name| vb.get((1, 1, h), name)?.reshape(h))
            .collect::<Result<Vec<_>>>()?;
        // the vectors the f32 part of a step reads, converted once
        let f32 = |t: Tensor| t.to_dtype(DType::F32);
        let vector = |name: &str| match spec.layout {
            Layout::Fla => f32(vb.get(h, name)?),
            Layout::Blink => f32(vb.get((1, 1, h), name)?.reshape(h)?),
        };
        let [w_rank, a_rank, v_rank, g_rank] = spec.ranks;
        let norm = vb.pp(spec.layout.norm());
        Ok(Self {
            layout: spec.layout,
            mixes: Tensor::stack(&mixes, 0)?,
            r_proj: proj(r_proj)?,
            k_proj: proj(k_proj)?,
            v_proj: proj(v_proj)?,
            o_proj: proj(o_proj)?,
            w_lora: LoRA::load(spec, "w", w_rank, true, &vb)?,
            a_lora: LoRA::load(spec, "a", a_rank, true, &vb)?,
            v_lora: (layer > 0)
                .then(|| LoRA::load(spec, "v", v_rank, true, &vb))
                .transpose()?,
            g_lora: spec
                .gated
                .then(|| LoRA::load(spec, "g", g_rank, false, &vb))
                .transpose()?,
            k_k: vector("k_k")?,
            k_a: vector("k_a")?,
            r_k: f32(vb.get((spec.heads, spec.head_dim), "r_k")?)?,
            g_norm: spec
                .group_norm
                .then(|| -> Result<(Tensor, Tensor)> {
                    Ok((f32(norm.get(h, "weight")?)?, f32(norm.get(h, "bias")?)?))
                })
                .transpose()?,
            g_norm_eps: spec.group_norm_eps,
            heads: spec.heads,
            head_dim: spec.head_dim,
        })
    }

    /// `x` normed `(b, s, h)`, `last` the token before it per sequence `(b, h)`, `state` the heads' WKV matrices
    /// (advanced), `v_first` the first layer's value (set by it); returns the output and `x`'s last token.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        last: &Tensor,
        state: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (mixed, new_last) = crate::ops::rwkv_token_shift(x, last, &self.mixes, None)?;
        let [xr, xw, xk, xv, xa] = [0, 1, 2, 3, 4].map(|i| mixed.get(i));
        let (xr, xw, xk, xv, xa) = (xr?, xw?, xk?, xv?, xa?);
        let r = self.r_proj.forward(&xr)?;
        let w = self.w_lora.forward(&xw, Some(Tensor::tanh))?;
        let k = self.k_proj.forward(&xk)?;
        let v = self.v_proj.forward(&xv)?;
        let a = self.a_lora.forward(&xa, None)?;
        let mut g = self
            .g_lora
            .as_ref()
            .map(|lora| lora.forward(&mixed.get(5)?, Some(inference_tensor::nn::ops::sigmoid)))
            .transpose()?;
        // BlinkDL's gate is `sigmoid(xg w1) w2 + 1`
        if self.layout == Layout::Blink {
            g = g.map(|g| g + 1.0).transpose()?;
        }
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
                g: g.as_ref(),
                v_mix: v_mix
                    .as_ref()
                    .zip(v_first.as_ref())
                    .map(|(mix, first)| (first, mix)),
                k_k: &self.k_k,
                k_a: &self.k_a,
                r_k: &self.r_k,
                norm: self.g_norm.as_ref().map(|(w, b)| (w, b)),
                norm_eps: self.g_norm_eps,
                heads: self.heads,
                head_dim: self.head_dim,
            },
            state,
        )?;
        Ok((self.o_proj.forward(&y.to_dtype(x.dtype())?)?, new_last))
    }

    /// The tensors ISQ does not take, under the time-mix's prefix.
    pub(crate) fn residual_tensors(&self, a: &UnVarBuilder) {
        let vector = |t: &Tensor| match self.layout {
            Layout::Fla => Ok(t.clone()),
            Layout::Blink => t.reshape((1, 1, t.dim(0)?)),
        };
        for (i, name) in TIME_MIXES.iter().enumerate() {
            let mix = self.mixes.get(i).and_then(|m| m.reshape((1, 1, m.dim(0)?)));
            if let Ok(mix) = mix {
                a.add_tensor(name, mix);
            }
        }
        for (name, lora) in [
            ("w", Some(&self.w_lora)),
            ("a", Some(&self.a_lora)),
            ("v", self.v_lora.as_ref()),
            ("g", self.g_lora.as_ref()),
        ] {
            if let Some(lora) = lora {
                lora.residual_tensors(name, a);
            }
        }
        for (name, t) in [("k_k", &self.k_k), ("k_a", &self.k_a)] {
            if let Ok(t) = vector(t) {
                a.add_tensor(name, t);
            }
        }
        a.add_tensor("r_k", self.r_k.clone());
        if let Some((w, b)) = &self.g_norm {
            let norm = a.pp(self.layout.norm());
            norm.add_tensor("weight", w.clone());
            norm.add_tensor("bias", b.clone());
        }
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
        shifts: &[Tensor],
        state: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let [attn_shift, ffn_shift] = shifts else {
            inference_tensor::bail!("an RWKV layer reads two token shifts");
        };
        let xn = self.attn_norm.forward(x)?;
        let (out, attn_last) = self.attn.forward(&xn, attn_shift, state, v_first)?;
        let x = (x + out)?;
        let xn = self.ffn_norm.forward(&x)?;
        let (out, ffn_last) = self.ffn.forward(&xn, ffn_shift)?;
        Ok(((x + out)?, vec![attn_last, ffn_last]))
    }

    fn residual_tensors(&self, l: &UnVarBuilder) {
        l.pp("attn_norm").add(&self.attn_norm);
        l.pp("ffn_norm").add(&self.ffn_norm);
        self.attn.residual_tensors(&l.pp("attn"));
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
    fn time_mix_spec(&self) -> TimeMixSpec<'_> {
        TimeMixSpec {
            layout: Layout::Fla,
            hidden: self.hidden_size,
            heads: self.heads(),
            head_dim: self.head_dim,
            gated: true,
            group_norm: true,
            group_norm_eps: self.head_dim as f64 * self.norm_eps,
            ranks: [
                self.decay_low_rank_dim,
                self.a_low_rank_dim,
                self.v_low_rank_dim,
                self.gate_low_rank_dim,
            ],
            quantization_config: &self.quantization_config,
        }
    }

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
            rms_norm: false,
            shifts: 2,
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
    RwkvModel::new(&shape, NAMES, Some("pre_norm"), vb, meta, |layer, vb, _| {
        Ok(Box::new(Layer {
            attn_norm: shape.layer_norm(vb.pp("attn_norm"))?,
            ffn_norm: shape.layer_norm(vb.pp("ffn_norm"))?,
            attn: TimeMix::load(&cfg.time_mix_spec(), layer, vb.pp("attn"))?,
            ffn: ChannelMix::load(cfg, vb.pp("ffn"))?,
        }) as Box<dyn RwkvBlock>)
    })
}
