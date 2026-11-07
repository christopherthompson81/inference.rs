//! RWKV time-mixes in Qwen2's decoder: ARWKV (`RwkvHybridForCausalLM`, RWKV-7's time-mix in BlinkDL's layout under
//! `self_attn.time_mixer`) and QRWKV6 (`RWKV6Qwen2ForCausalLM`, RWKV-6's data-dependent token shift and decay over
//! gated linear attention, Qwen2's projections with their biases and grouped key/value heads). Each layer is Qwen2's:
//! an RMSNorm, the time-mix in attention's place, an RMSNorm and the SwiGLU MLP; one token shift per layer.

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::{Module, Result, Tensor};
use serde::Serialize;

use crate::layers::{Activation, Mlp};
use crate::model::NormalLoadingMetadata;
use crate::rwkv::{RwkvBlock, RwkvModel, RwkvNames, RwkvNorm, RwkvShape};
use crate::rwkv6::DataDependentShift;
use crate::rwkv7::{Layout, TimeMix, TimeMixSpec};
use crate::serde_default_fn;
use crate::utils::unvarbuilder::UnVarBuilder;

// QRWKV6's data-dependent mixes in the order `time_maa_w2` stacks them
const QRWKV6_MIXES: [&str; 5] = [
    "time_maa_r",
    "time_maa_k",
    "time_maa_v",
    "time_maa_w",
    "time_maa_g",
];
// the WKV version ARWKV's time-mix is
const ARWKV_WKV_VERSION: usize = 7;
// QRWKV6's only gate form: a full-rank `gate` projection
const QRWKV6_GATE_RANK_TYPE: usize = 1;
// RWKV's group norm eps before its `head_size_divisor^2` scale
const GROUP_NORM_EPS: f64 = 1e-5;

serde_default_fn!(f64, default_rms_norm_eps, 1e-6);
serde_default_fn!(usize, default_head_size, 64);
serde_default_fn!(usize, default_head_size_divisor, 8);
serde_default_fn!(usize, default_wkv_version, ARWKV_WKV_VERSION);
serde_default_fn!(usize, default_gate_rank_type, QRWKV6_GATE_RANK_TYPE);
serde_default_fn!(bool, default_true, true);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct ArwkvConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_head_size")]
    pub head_size: usize,
    #[serde(default = "default_head_size_divisor")]
    pub head_size_divisor: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_wkv_version")]
    pub wkv_version: usize,
    #[serde(default)]
    pub wkv_has_gate: bool,
    #[serde(default = "default_true")]
    pub wkv_has_group_norm: bool,
    #[serde(default = "default_true")]
    pub wkv_use_vfirst: bool,
    /// the layers whose attention is a time-mix; all of them unless listed
    pub wkv_layers: Option<Vec<usize>>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl ArwkvConfig {
    pub fn heads(&self) -> usize {
        self.hidden_size / self.head_size
    }

    pub(crate) fn shape(&self) -> RwkvShape<'_> {
        qwen2_shape(
            [
                self.vocab_size,
                self.hidden_size,
                self.heads(),
                self.head_size,
            ],
            self.num_hidden_layers,
            self.rms_norm_eps,
            self.tie_word_embeddings,
            &self.quantization_config,
        )
    }

    fn check(&self) -> Result<()> {
        if self.wkv_version != ARWKV_WKV_VERSION || !self.wkv_use_vfirst {
            inference_tensor::bail!(
                "ARWKV runs RWKV-7 time-mixes with the first layer's value (wkv_version {}, wkv_use_vfirst {})",
                self.wkv_version,
                self.wkv_use_vfirst
            );
        }
        if let Some(layers) = &self.wkv_layers
            && (0..self.num_hidden_layers).any(|l| !layers.contains(&l))
        {
            inference_tensor::bail!(
                "ARWKV with attention layers (wkv_layers {layers:?}) is not supported"
            );
        }
        Ok(())
    }

    fn time_mix_spec(&self) -> TimeMixSpec<'_> {
        TimeMixSpec {
            layout: Layout::Blink,
            hidden: self.hidden_size,
            heads: self.heads(),
            head_dim: self.head_size,
            gated: self.wkv_has_gate,
            group_norm: self.wkv_has_group_norm,
            group_norm_eps: GROUP_NORM_EPS
                * (self.head_size_divisor * self.head_size_divisor) as f64,
            ranks: [0; 4],
            quantization_config: &self.quantization_config,
        }
    }
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Qrwkv6Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_output_bias: bool,
    #[serde(default = "default_gate_rank_type")]
    pub gate_rank_type: usize,
    #[serde(default = "default_true")]
    pub balance_state: bool,
    #[serde(default)]
    pub groupnorm_att: bool,
    #[serde(default = "default_true")]
    pub use_tokenshift: bool,
    #[serde(default)]
    pub use_rope: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Qrwkv6Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub(crate) fn shape(&self) -> RwkvShape<'_> {
        qwen2_shape(
            [
                self.vocab_size,
                self.hidden_size,
                self.num_attention_heads,
                self.head_dim(),
            ],
            self.num_hidden_layers,
            self.rms_norm_eps,
            self.tie_word_embeddings,
            &self.quantization_config,
        )
    }

    fn check(&self) -> Result<()> {
        if self.gate_rank_type != QRWKV6_GATE_RANK_TYPE
            || !self.balance_state
            || self.groupnorm_att
            || !self.use_tokenshift
            || self.use_rope
        {
            inference_tensor::bail!(
                "QRWKV6 runs a full-rank gate, a balanced state, token shifts and no group norm or RoPE \
                 (gate_rank_type {}, balance_state {}, groupnorm_att {}, use_tokenshift {}, use_rope {})",
                self.gate_rank_type,
                self.balance_state,
                self.groupnorm_att,
                self.use_tokenshift,
                self.use_rope
            );
        }
        Ok(())
    }
}

// `[vocab, hidden, heads, head_dim]`
fn qwen2_shape(
    [vocab, hidden, heads, head_dim]: [usize; 4],
    layers: usize,
    norm_eps: f64,
    tie_word_embeddings: bool,
    quantization_config: &Option<QuantizedConfig>,
) -> RwkvShape<'_> {
    RwkvShape {
        vocab,
        hidden,
        heads,
        head_dim,
        layers,
        tie_word_embeddings,
        norm_eps,
        rms_norm: true,
        shifts: 1,
        quantization_config,
        rescaled_every: None,
    }
}

struct Qrwkv6TimeMix {
    maa_x: Tensor,
    // `(5, hidden)` in `QRWKV6_MIXES`' order
    maa: Tensor,
    maa_w1: Tensor,
    maa_w2: Tensor,
    time_decay: Tensor,
    decay_w1: Tensor,
    decay_w2: Tensor,
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    gate: Arc<dyn QuantMethod>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Qrwkv6TimeMix {
    fn load(cfg: &Qrwkv6Config, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let (n, kv) = (cfg.head_dim(), cfg.kv_heads());
        let (attn, kv_width) = (cfg.num_attention_heads * n, kv * n);
        let q = &cfg.quantization_config;
        let bias = cfg.attention_bias;
        let maa = QRWKV6_MIXES
            .iter()
            .map(|name| vb.get((1, 1, h), name)?.reshape(h))
            .collect::<Result<Vec<_>>>()?;
        let maa_w1 = vb.get_unchecked("time_maa_w1")?;
        let rank = maa_w1.dim(1)? / QRWKV6_MIXES.len();
        let decay_w1 = vb.get_unchecked("time_decay_w1")?;
        let decay_rank = decay_w1.dim(1)?;
        Ok(Self {
            maa_x: vb.get((1, 1, h), "time_maa_x")?.reshape((1, h))?,
            maa: Tensor::stack(&maa, 0)?,
            maa_w1,
            maa_w2: vb.get((QRWKV6_MIXES.len(), rank, h), "time_maa_w2")?,
            time_decay: vb.get((1, 1, attn), "time_decay")?.reshape(attn)?,
            decay_w1,
            decay_w2: vb.get((decay_rank, attn), "time_decay_w2")?,
            q_proj: ReplicatedLayer::new(h, attn, q, bias, vb.pp("q_proj"))?,
            k_proj: ReplicatedLayer::new(h, kv_width, q, bias, vb.pp("k_proj"))?,
            v_proj: ReplicatedLayer::new(h, kv_width, q, bias, vb.pp("v_proj"))?,
            o_proj: ReplicatedLayer::new(attn, h, q, cfg.attention_output_bias, vb.pp("o_proj"))?,
            gate: ReplicatedLayer::new(h, attn, q, false, vb.pp("gate"))?,
            heads: cfg.num_attention_heads,
            kv_heads: kv,
            head_dim: n,
        })
    }

    fn forward(&self, x: &Tensor, last: &Tensor, state: &Tensor) -> Result<(Tensor, Tensor)> {
        let shift = DataDependentShift {
            maa_x: &self.maa_x,
            maa: &self.maa,
            maa_w1: &self.maa_w1,
            maa_w2: &self.maa_w2,
        };
        let (mixed, new_last) = shift.forward(x, last)?;
        let [xr, xk, xv, xw, xg] = [0, 1, 2, 3, 4].map(|i| mixed.get(i));
        let (xr, xk, xv, xw, xg) = (xr?, xk?, xv?, xw?, xg?);
        let w = xw
            .broadcast_matmul(&self.decay_w1)?
            .tanh()?
            .broadcast_matmul(&self.decay_w2)?
            .broadcast_add(&self.time_decay)?;
        let y = crate::ops::qrwkv6_time_mix(
            crate::ops::TimeMixQrwkv6 {
                r: &self.q_proj.forward(&xr)?,
                k: &self.k_proj.forward(&xk)?,
                v: &self.v_proj.forward(&xv)?,
                w: &w,
                g: &self.gate.forward(&xg)?,
                heads: self.heads,
                kv_heads: self.kv_heads,
                head_dim: self.head_dim,
            },
            state,
        )?;
        Ok((self.o_proj.forward(&y.to_dtype(x.dtype())?)?, new_last))
    }

    fn residual_tensors(&self, a: &UnVarBuilder) {
        let row = |m: &Tensor| m.reshape((1, 1, m.dim(m.rank() - 1)?));
        if let Ok(x) = row(&self.maa_x) {
            a.add_tensor("time_maa_x", x);
        }
        for (i, name) in QRWKV6_MIXES.iter().enumerate() {
            if let Ok(mix) = self.maa.get(i).and_then(|m| row(&m)) {
                a.add_tensor(name, mix);
            }
        }
        a.add_tensor("time_maa_w1", self.maa_w1.clone());
        a.add_tensor("time_maa_w2", self.maa_w2.clone());
        if let Ok(decay) = row(&self.time_decay) {
            a.add_tensor("time_decay", decay);
        }
        a.add_tensor("time_decay_w1", self.decay_w1.clone());
        a.add_tensor("time_decay_w2", self.decay_w2.clone());
    }
}

enum Mixer {
    Arwkv(TimeMix),
    Qrwkv6(Qrwkv6TimeMix),
}

struct Layer {
    input_layernorm: RwkvNorm,
    post_attention_layernorm: RwkvNorm,
    mixer: Mixer,
    mlp: Mlp,
}

impl RwkvBlock for Layer {
    fn forward(
        &self,
        x: &Tensor,
        shifts: &[Tensor],
        state: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let [shift] = shifts else {
            inference_tensor::bail!("a hybrid RWKV layer reads one token shift");
        };
        let xn = self.input_layernorm.forward(x)?;
        let (out, last) = match &self.mixer {
            Mixer::Arwkv(tm) => tm.forward(&xn, shift, state, v_first)?,
            Mixer::Qrwkv6(tm) => tm.forward(&xn, shift, state)?,
        };
        let x = (x + out)?;
        let out = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?;
        Ok(((x + out)?, vec![last]))
    }

    fn residual_tensors(&self, l: &UnVarBuilder) {
        self.input_layernorm
            .residual_tensors(&l.pp("input_layernorm"));
        self.post_attention_layernorm
            .residual_tensors(&l.pp("post_attention_layernorm"));
        let a = l.pp("self_attn");
        match &self.mixer {
            Mixer::Arwkv(tm) => tm.residual_tensors(&a.pp(ARWKV_TIME_MIXER)),
            Mixer::Qrwkv6(tm) => tm.residual_tensors(&a),
        }
    }
}

// ARWKV's attention module holds its time-mix under this name
const ARWKV_TIME_MIXER: &str = "time_mixer";

const NAMES: RwkvNames = RwkvNames {
    root: "model",
    embeddings: "embed_tokens",
    norm: "norm",
    layers: "layers",
    head: "lm_head",
};

fn hybrid_model(
    shape: &RwkvShape<'_>,
    ffn: usize,
    vb: ShardedVarBuilder,
    meta: NormalLoadingMetadata,
    mixer: impl Fn(usize, ShardedVarBuilder) -> Result<Mixer> + Sync,
) -> Result<RwkvModel> {
    RwkvModel::new(shape, NAMES, None, vb, meta, |layer, vb, comm| {
        Ok(Box::new(Layer {
            input_layernorm: shape.norm(vb.pp("input_layernorm"))?,
            post_attention_layernorm: shape.norm(vb.pp("post_attention_layernorm"))?,
            mixer: mixer(layer, vb.pp("self_attn"))?,
            mlp: Mlp::new(
                vb.pp("mlp"),
                shape.hidden,
                ffn,
                shape.quantization_config,
                Activation::Silu,
                comm,
            )?,
        }) as Box<dyn RwkvBlock>)
    })
}

/// ARWKV on the shared RWKV stack.
pub fn arwkv_model(
    cfg: &ArwkvConfig,
    vb: ShardedVarBuilder,
    meta: NormalLoadingMetadata,
) -> Result<RwkvModel> {
    cfg.check()?;
    let spec = cfg.time_mix_spec();
    hybrid_model(
        &cfg.shape(),
        cfg.intermediate_size,
        vb,
        meta,
        |layer, vb| {
            Ok(Mixer::Arwkv(TimeMix::load(
                &spec,
                layer,
                vb.pp(ARWKV_TIME_MIXER),
            )?))
        },
    )
}

/// QRWKV6 on the shared RWKV stack.
pub fn qrwkv6_model(
    cfg: &Qrwkv6Config,
    vb: ShardedVarBuilder,
    meta: NormalLoadingMetadata,
) -> Result<RwkvModel> {
    cfg.check()?;
    hybrid_model(&cfg.shape(), cfg.intermediate_size, vb, meta, |_, vb| {
        Ok(Mixer::Qrwkv6(Qrwkv6TimeMix::load(cfg, vb)?))
    })
}
