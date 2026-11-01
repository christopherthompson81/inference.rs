#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Whisper's encoder: two conv1d downsampling the log-mel, fixed sinusoidal positions, pre-norm layers (LayerNorm,
//! bidirectional attention, a two-layer FFN) and a final LayerNorm, under transformers' `WhisperEncoder` names. The
//! Whisper recognizer, the audio LLMs built on its encoder and the Qwen audio towers share its pieces.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module};
use inference_tensor::{Result, Tensor};

use crate::attention::{AttentionMask, FlashParams, SdpaParams};
use crate::layers::{Activation, Sdpa, conv1d, layer_norm, linear_b};
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

const CONV_KERNEL: usize = 3;
// the second conv halves the frames: 3000 mel frames of a 30 s window to 1500 positions
const DOWNSAMPLE: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct WhisperEncoderConfig {
    pub layer: WhisperLayerConfig,
    pub num_layers: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
}

pub struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positions: Tensor,
    layers: Vec<WhisperEncoderLayer>,
    layer_norm: LayerNorm,
    activation: Activation,
}

impl WhisperEncoder {
    pub fn new(cfg: &WhisperEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.layer.d_model;
        let conv = |stride| Conv1dConfig {
            padding: 1,
            stride,
            ..Default::default()
        };
        Ok(Self {
            conv1: conv1d(cfg.num_mel_bins, d, CONV_KERNEL, conv(1), vb.pp("conv1"))?,
            conv2: conv1d(d, d, CONV_KERNEL, conv(DOWNSAMPLE), vb.pp("conv2"))?,
            // the checkpoint stores the sinusoid table; fine-tunes are read as they ship
            positions: vb.get((cfg.max_source_positions, d), "embed_positions.weight")?,
            layers: (0..cfg.num_layers)
                .map(|i| WhisperEncoderLayer::new(&cfg.layer, vb.pp("layers").pp(i)))
                .collect::<Result<Vec<_>>>()?,
            layer_norm: layer_norm(d, cfg.layer.layer_norm_eps, vb.pp("layer_norm"))?,
            activation: cfg.layer.activation,
        })
    }

    /// `(batch, n_mels, frames)` log-mel to `(batch, frames / 2, d_model)`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let xs = self.encode(&self.embed(mel)?)?;
        self.norm(&xs)
    }

    /// The convolutions and positions: `(batch, n_mels, frames)` to `(batch, frames / 2, d_model)`.
    pub fn embed(&self, mel: &Tensor) -> Result<Tensor> {
        let xs = self.activation.forward(&self.conv1.forward(mel)?)?;
        let xs = self.activation.forward(&self.conv2.forward(&xs)?)?;
        let xs = xs.transpose(1, 2)?;
        let positions = self.positions.narrow(0, 0, xs.dim(1)?)?;
        xs.broadcast_add(&positions)
    }

    /// The layers over embedded positions, before the final norm.
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }

    pub fn norm(&self, xs: &Tensor) -> Result<Tensor> {
        self.layer_norm.forward(xs)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("conv1").add(&self.conv1);
        uvb.pp("conv2").add(&self.conv2);
        uvb.add_tensor("embed_positions.weight", self.positions.clone());
        for (i, layer) in self.layers.iter().enumerate() {
            layer.residual_tensors(&uvb.pp("layers").pp(i));
        }
        uvb.pp("layer_norm").add(&self.layer_norm);
        uvb.to_safetensors()
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
