//! RWKV-6 ("Finch", `Rwkv6ForCausalLM` in RWKV's HF layout): every layer a time-mix (data-dependent token shifts,
//! a decay per channel from a LoRA, the WKV-6 recurrence with its bonus for the current token) and a channel-mix (a
//! squared-ReLU MLP gated by a sigmoid receptance), on the stack RWKV-7 shares.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::nn::LayerNorm;
use inference_tensor::{Module, Result, Tensor};
use serde::Serialize;

use crate::model::NormalLoadingMetadata;
use crate::rwkv::{RwkvBlock, RwkvModel, RwkvNames, RwkvShape};
use crate::serde_default_fn;
use crate::utils::unvarbuilder::UnVarBuilder;

// the data-dependent mixes in the order `time_maa_w2` stacks them
const TIME_MIXES: [&str; 5] = [
    "time_maa_w",
    "time_maa_k",
    "time_maa_v",
    "time_maa_r",
    "time_maa_g",
];
// the channel-mix's two mixes, key then receptance
const CHANNEL_MIXES: [&str; 2] = ["time_maa_k", "time_maa_r"];
// the FFN width when the config leaves it unset: 3.5x the hidden size, to a multiple of 32
const FFN_RATIO_NUM: usize = 7;
const FFN_RATIO_DEN: usize = 2;
const FFN_ROUND: usize = 32;

serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(usize, default_head_size, 64);
serde_default_fn!(usize, default_head_size_divisor, 8);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_head_size")]
    pub head_size: usize,
    #[serde(default = "default_head_size_divisor")]
    pub head_size_divisor: usize,
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_norm_eps")]
    pub layer_norm_epsilon: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// llama.cpp's GGUF rescaling (its synthesized config's own key; a checkpoint's `rescale_every` is fp16 only and
    /// its weights are as trained).
    pub rescaled_every: Option<usize>,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn heads(&self) -> usize {
        self.hidden_size / self.head_size
    }

    pub fn ffn_size(&self) -> usize {
        self.intermediate_size
            .unwrap_or(self.hidden_size * FFN_RATIO_NUM / FFN_RATIO_DEN / FFN_ROUND * FFN_ROUND)
    }

    pub(crate) fn shape(&self) -> RwkvShape<'_> {
        RwkvShape {
            vocab: self.vocab_size,
            hidden: self.hidden_size,
            heads: self.heads(),
            head_dim: self.head_size,
            layers: self.num_hidden_layers,
            tie_word_embeddings: self.tie_word_embeddings,
            norm_eps: self.layer_norm_epsilon,
            quantization_config: &self.quantization_config,
            rescaled_every: self.rescaled_every,
        }
    }
}

struct TimeMix {
    maa_x: Tensor,
    // `(5, hidden)` in `TIME_MIXES`' order
    maa: Tensor,
    // the mixes' LoRA, `(hidden, 5 * rank)` then `(5, rank, hidden)`
    maa_w1: Tensor,
    maa_w2: Tensor,
    time_decay: Tensor,
    decay_w1: Tensor,
    decay_w2: Tensor,
    time_first: Tensor,
    receptance: Arc<dyn QuantMethod>,
    key: Arc<dyn QuantMethod>,
    value: Arc<dyn QuantMethod>,
    gate: Arc<dyn QuantMethod>,
    output: Arc<dyn QuantMethod>,
    ln_x: (Tensor, Tensor),
    ln_x_eps: f64,
    heads: usize,
    head_size: usize,
}

impl TimeMix {
    fn load(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let q = &cfg.quantization_config;
        let proj = |name: &str| ReplicatedLayer::new(h, h, q, false, vb.pp(name));
        let maa = TIME_MIXES
            .iter()
            .map(|name| vb.get((1, 1, h), name)?.reshape(h))
            .collect::<Result<Vec<_>>>()?;
        let maa_w1 = vb.get_unchecked("time_maa_w1")?;
        let rank = maa_w1.dim(1)? / TIME_MIXES.len();
        let decay_w1 = vb.get_unchecked("time_decay_w1")?;
        let decay_rank = decay_w1.dim(1)?;
        let f32 = |t: Tensor| t.to_dtype(inference_tensor::DType::F32);
        Ok(Self {
            maa_x: vb.get((1, 1, h), "time_maa_x")?.reshape((1, h))?,
            maa: Tensor::stack(&maa, 0)?,
            maa_w1,
            maa_w2: vb.get((TIME_MIXES.len(), rank, h), "time_maa_w2")?,
            time_decay: vb.get((1, 1, h), "time_decay")?.reshape(h)?,
            decay_w1,
            decay_w2: vb.get((decay_rank, h), "time_decay_w2")?,
            time_first: f32(vb.get((cfg.heads(), cfg.head_size), "time_faaaa")?)?,
            receptance: proj("receptance")?,
            key: proj("key")?,
            value: proj("value")?,
            gate: proj("gate")?,
            output: proj("output")?,
            ln_x: (
                f32(vb.pp("ln_x").get(h, "weight")?)?,
                f32(vb.pp("ln_x").get(h, "bias")?)?,
            ),
            ln_x_eps: cfg.layer_norm_epsilon
                * (cfg.head_size_divisor * cfg.head_size_divisor) as f64,
            heads: cfg.heads(),
            head_size: cfg.head_size,
        })
    }

    fn forward(&self, x: &Tensor, last: &Tensor, state: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, s, h) = x.dims3()?;
        let mixes = TIME_MIXES.len();
        let (xxx, new_last) = crate::ops::rwkv_token_shift(x, last, &self.maa_x, None)?;
        // the mixes' data-dependent parts: tanh(xxx w1), split per mix, through each mix's w2
        let lora = xxx.get(0)?.broadcast_matmul(&self.maa_w1)?.tanh()?;
        let rank = lora.dim(2)? / mixes;
        let lora = lora
            .reshape((b * s, mixes, rank))?
            .transpose(0, 1)?
            .contiguous()?;
        let extra = lora.matmul(&self.maa_w2)?.reshape((mixes, b, s, h))?;
        let (mixed, _) = crate::ops::rwkv_token_shift(x, last, &self.maa, Some(&extra))?;
        let [xw, xk, xv, xr, xg] = [0, 1, 2, 3, 4].map(|i| mixed.get(i));
        let (xw, xk, xv, xr, xg) = (xw?, xk?, xv?, xr?, xg?);
        let w = xw
            .broadcast_matmul(&self.decay_w1)?
            .tanh()?
            .broadcast_matmul(&self.decay_w2)?
            .broadcast_add(&self.time_decay)?;
        let y = crate::ops::rwkv6_time_mix(
            crate::ops::TimeMix6 {
                r: &self.receptance.forward(&xr)?,
                k: &self.key.forward(&xk)?,
                v: &self.value.forward(&xv)?,
                w: &w,
                g: &self.gate.forward(&xg)?,
                u: &self.time_first,
                norm_w: &self.ln_x.0,
                norm_b: &self.ln_x.1,
                norm_eps: self.ln_x_eps,
                heads: self.heads,
                head_dim: self.head_size,
            },
            state,
        )?;
        Ok((self.output.forward(&y.to_dtype(x.dtype())?)?, new_last))
    }
}

struct ChannelMix {
    // `(2, hidden)`: the key's mix, then the receptance's
    maa: Tensor,
    key: Arc<dyn QuantMethod>,
    receptance: Arc<dyn QuantMethod>,
    value: Arc<dyn QuantMethod>,
}

impl ChannelMix {
    fn load(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let (h, ffn) = (cfg.hidden_size, cfg.ffn_size());
        let q = &cfg.quantization_config;
        let maa = CHANNEL_MIXES
            .iter()
            .map(|name| vb.get((1, 1, h), name)?.reshape(h))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            maa: Tensor::stack(&maa, 0)?,
            key: ReplicatedLayer::new(h, ffn, q, false, vb.pp("key"))?,
            receptance: ReplicatedLayer::new(h, h, q, false, vb.pp("receptance"))?,
            value: ReplicatedLayer::new(ffn, h, q, false, vb.pp("value"))?,
        })
    }

    fn forward(&self, x: &Tensor, last: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mixed, new_last) = crate::ops::rwkv_token_shift(x, last, &self.maa, None)?;
        let k = self.key.forward(&mixed.get(0)?)?.relu()?.sqr()?;
        let r = inference_tensor::nn::ops::sigmoid(&self.receptance.forward(&mixed.get(1)?)?)?;
        Ok(((r * self.value.forward(&k)?)?, new_last))
    }
}

struct Layer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    attention: TimeMix,
    feed_forward: ChannelMix,
}

impl RwkvBlock for Layer {
    fn forward(
        &self,
        x: &Tensor,
        [attn_shift, ffn_shift]: [&Tensor; 2],
        state: &Tensor,
        _v_first: &mut Option<Tensor>,
    ) -> Result<[Tensor; 3]> {
        let (out, attn_last) = self
            .attention
            .forward(&self.ln1.forward(x)?, attn_shift, state)?;
        let x = (x + out)?;
        let (out, ffn_last) = self
            .feed_forward
            .forward(&self.ln2.forward(&x)?, ffn_shift)?;
        Ok([(x + out)?, attn_last, ffn_last])
    }

    fn residual_tensors(&self, l: &UnVarBuilder) {
        l.pp("ln1").add(&self.ln1);
        l.pp("ln2").add(&self.ln2);
        let a = l.pp("attention");
        let t = &self.attention;
        let h = t.time_decay.dim(0).unwrap_or(0);
        let row = |m: &Tensor| m.reshape((1, 1, h));
        if let Ok(x) = row(&t.maa_x) {
            a.add_tensor("time_maa_x", x);
        }
        for (i, name) in TIME_MIXES.iter().enumerate() {
            if let Ok(mix) = t.maa.get(i).and_then(|m| row(&m)) {
                a.add_tensor(name, mix);
            }
        }
        a.add_tensor("time_maa_w1", t.maa_w1.clone());
        a.add_tensor("time_maa_w2", t.maa_w2.clone());
        if let Ok(decay) = row(&t.time_decay) {
            a.add_tensor("time_decay", decay);
        }
        a.add_tensor("time_decay_w1", t.decay_w1.clone());
        a.add_tensor("time_decay_w2", t.decay_w2.clone());
        a.add_tensor("time_faaaa", t.time_first.clone());
        a.pp("ln_x").add_tensor("weight", t.ln_x.0.clone());
        a.pp("ln_x").add_tensor("bias", t.ln_x.1.clone());
        let f = l.pp("feed_forward");
        for (i, name) in CHANNEL_MIXES.iter().enumerate() {
            if let Ok(mix) = self.feed_forward.maa.get(i).and_then(|m| row(&m)) {
                f.add_tensor(name, mix);
            }
        }
    }
}

const NAMES: RwkvNames = RwkvNames {
    root: "rwkv",
    embeddings: "embeddings",
    norm: "ln_out",
    layers: "blocks",
    head: "head",
};

/// RWKV-6 as the shared RWKV stack, in RWKV's HF layout.
pub fn model(
    cfg: &Config,
    vb: ShardedVarBuilder,
    meta: NormalLoadingMetadata,
) -> Result<RwkvModel> {
    let shape = cfg.shape();
    RwkvModel::new(&shape, NAMES, "pre_ln", vb, meta, |_, vb| {
        Ok(Box::new(Layer {
            ln1: shape.layer_norm(vb.pp("ln1"))?,
            ln2: shape.layer_norm(vb.pp("ln2"))?,
            attention: TimeMix::load(cfg, vb.pp("attention"))?,
            feed_forward: ChannelMix::load(cfg, vb.pp("feed_forward"))?,
        }) as Box<dyn RwkvBlock>)
    })
}
