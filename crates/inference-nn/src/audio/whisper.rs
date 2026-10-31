#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Whisper's encoder layer: pre-norm LayerNorm, bidirectional multi-head attention, a two-layer FFN, under
//! transformers' `WhisperEncoderLayer` names. Whisper's encoders and the Qwen audio towers built on them share it.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{Result, Tensor};

use crate::attention::{AttentionMask, FlashParams, SdpaParams};
use crate::layers::{Activation, Sdpa, layer_norm, linear_b};
use crate::utils::unvarbuilder::UnVarBuilder;

#[derive(Clone, Copy, Debug)]
pub struct WhisperLayerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub activation: Activation,
    pub layer_norm_eps: f64,
    // Whisper's k_proj has no bias; the Qwen audio towers' does
    pub k_bias: bool,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &WhisperLayerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            q_proj: linear_b(d, d, true, vb.pp("q_proj"))?,
            k_proj: linear_b(d, d, cfg.k_bias, vb.pp("k_proj"))?,
            v_proj: linear_b(d, d, true, vb.pp("v_proj"))?,
            out_proj: linear_b(d, d, true, vb.pp("out_proj"))?,
            num_heads: cfg.num_heads,
            head_dim: d / cfg.num_heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let heads = |proj: &Linear| -> Result<Tensor> {
            proj.forward(xs)?
                .reshape((b, t, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (
            heads(&self.q_proj)?,
            heads(&self.k_proj)?,
            heads(&self.v_proj)?,
        );
        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        // empty seqlens: the flash path's non-varlen kernel, bidirectional
        let out = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::None,
            Some(&FlashParams::empty(false)),
            &params,
        )?;
        self.out_proj
            .forward(&out.transpose(1, 2)?.reshape((b, t, d))?)
    }

    fn residual_tensors(&self, uvb: &UnVarBuilder) {
        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("out_proj").add(&self.out_proj);
    }
}

pub struct WhisperEncoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}

impl WhisperEncoderLayer {
    pub fn new(cfg: &WhisperLayerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            self_attn_layer_norm: layer_norm(d, cfg.layer_norm_eps, vb.pp("self_attn_layer_norm"))?,
            final_layer_norm: layer_norm(d, cfg.layer_norm_eps, vb.pp("final_layer_norm"))?,
            fc1: linear_b(d, cfg.ffn_dim, true, vb.pp("fc1"))?,
            fc2: linear_b(cfg.ffn_dim, d, true, vb.pp("fc2"))?,
            activation: cfg.activation,
        })
    }

    /// `(batch, frames, d_model)` in and out; every frame of a batch row attends to every other.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attended = self
            .self_attn
            .forward(&self.self_attn_layer_norm.forward(xs)?)?;
        let xs = (xs + attended)?;
        let ffn = self.fc2.forward(
            &self
                .activation
                .forward(&self.fc1.forward(&self.final_layer_norm.forward(&xs)?)?)?,
        )?;
        xs + ffn
    }

    pub fn residual_tensors(&self, uvb: &UnVarBuilder) {
        self.self_attn.residual_tensors(&uvb.pp("self_attn"));
        uvb.pp("self_attn_layer_norm")
            .add(&self.self_attn_layer_norm);
        uvb.pp("final_layer_norm").add(&self.final_layer_norm);
        uvb.pp("fc1").add(&self.fc1);
        uvb.pp("fc2").add(&self.fc2);
    }
}

/// Whisper's sinusoidal position table: `length` rows of `sin` then `cos` over `channels / 2` log-spaced timescales
/// up to 10000.
pub fn sinusoid_positions(length: usize, channels: usize) -> Vec<f32> {
    const MAX_TIMESCALE: f64 = 10_000.0;
    let half = channels / 2;
    let increment = MAX_TIMESCALE.ln() / (half - 1) as f64;
    let inv: Vec<f32> = (0..half)
        .map(|i| (-increment * i as f64).exp() as f32)
        .collect();
    let mut table = Vec::with_capacity(length * channels);
    for position in 0..length {
        let scaled = inv.iter().map(|&inv| position as f32 * inv);
        table.extend(scaled.clone().map(f32::sin));
        table.extend(scaled.map(f32::cos));
    }
    table
}
