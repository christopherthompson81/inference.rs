#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Whisper's encoder: two conv1d downsampling the log-mel, fixed sinusoidal positions, pre-norm layers (LayerNorm,
//! bidirectional attention, a two-layer FFN) and a final LayerNorm, under transformers' `WhisperEncoder` names. The
//! Whisper recognizer, the audio LLMs built on its encoder and the Qwen audio towers share its pieces; GLM-ASR's
//! tower is the same layers with rotary positions in place of the table, under Llama-style names.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::attention::{AttentionMask, FlashParams, SdpaParams};
use crate::layers::{Activation, Sdpa, conv1d, layer_norm, linear_b};
use crate::utils::unvarbuilder::UnVarBuilder;

/// The checkpoint names a layer's weights go by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhisperLayerNames {
    /// `self_attn_layer_norm`, `self_attn.out_proj`, `final_layer_norm`, `fc1`, `fc2`
    Whisper,
    /// `input_layernorm`, `self_attn.o_proj`, `post_attention_layernorm`, `mlp.fc1`, `mlp.fc2`
    Llama,
}

impl WhisperLayerNames {
    fn attn_norm(self) -> &'static str {
        match self {
            Self::Whisper => "self_attn_layer_norm",
            Self::Llama => "input_layernorm",
        }
    }

    fn ffn_norm(self) -> &'static str {
        match self {
            Self::Whisper => "final_layer_norm",
            Self::Llama => "post_attention_layernorm",
        }
    }

    fn out_proj(self) -> &'static str {
        match self {
            Self::Whisper => "out_proj",
            Self::Llama => "o_proj",
        }
    }

    fn ffn(self, vb: &ShardedVarBuilder) -> ShardedVarBuilder {
        match self {
            Self::Whisper => vb.clone(),
            Self::Llama => vb.pp("mlp"),
        }
    }

    fn final_norm(self) -> &'static str {
        match self {
            Self::Whisper => "layer_norm",
            Self::Llama => "norm",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WhisperLayerConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub activation: Activation,
    pub layer_norm_eps: f64,
    // Whisper's k_proj has no bias; the Qwen audio towers' does
    pub k_bias: bool,
    pub names: WhisperLayerNames,
}

/// How positions enter the encoder.
#[derive(Clone, Copy, Debug)]
pub enum WhisperPositions {
    /// The checkpoint's `embed_positions` table, added after the convolutions.
    Learned,
    /// Rotary positions on the first `dims` of each head's queries and keys, rotated by halves (GLM-ASR).
    Rotary { theta: f32, dims: usize },
}

/// A rotary tower's cos and sin over `dims`, by position.
pub struct WhisperRotary {
    cos: Tensor,
    sin: Tensor,
    dims: usize,
}

impl WhisperRotary {
    fn new(
        theta: f32,
        dims: usize,
        positions: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv: Vec<f32> = (0..dims)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dims as f32))
            .collect();
        let mut angles = Vec::with_capacity(positions * dims);
        for position in 0..positions {
            let row = inv.iter().map(|&inv| position as f32 * inv);
            angles.extend(row.clone().chain(row));
        }
        let angles = Tensor::from_vec(angles, (positions, dims), device)?;
        Ok(Self {
            cos: angles.cos()?.to_dtype(dtype)?,
            sin: angles.sin()?.to_dtype(dtype)?,
            dims,
        })
    }

    // `(batch, heads, frames, head_dim)`: the first `dims` rotated by halves, the rest passed through
    fn apply(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, frames, head_dim) = xs.dims4()?;
        let (cos, sin) = (
            self.cos.narrow(0, 0, frames)?,
            self.sin.narrow(0, 0, frames)?,
        );
        let half = self.dims / 2;
        let rot = xs.narrow(D::Minus1, 0, self.dims)?;
        let turned = Tensor::cat(
            &[
                &rot.narrow(D::Minus1, half, half)?.neg()?,
                &rot.narrow(D::Minus1, 0, half)?,
            ],
            D::Minus1,
        )?;
        let rot = (rot.broadcast_mul(&cos)? + turned.broadcast_mul(&sin)?)?;
        if self.dims == head_dim {
            return Ok(rot);
        }
        Tensor::cat(
            &[
                &rot,
                &xs.narrow(D::Minus1, self.dims, head_dim - self.dims)?,
            ],
            D::Minus1,
        )
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    names: WhisperLayerNames,
}

impl Attention {
    fn new(cfg: &WhisperLayerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            q_proj: linear_b(d, d, true, vb.pp("q_proj"))?,
            k_proj: linear_b(d, d, cfg.k_bias, vb.pp("k_proj"))?,
            v_proj: linear_b(d, d, true, vb.pp("v_proj"))?,
            out_proj: linear_b(d, d, true, vb.pp(cfg.names.out_proj()))?,
            num_heads: cfg.num_heads,
            head_dim: d / cfg.num_heads,
            names: cfg.names,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rotary: Option<&WhisperRotary>,
        keys: Option<usize>,
    ) -> Result<Tensor> {
        let (b, t, d) = xs.dims3()?;
        let heads = |proj: &Linear| -> Result<Tensor> {
            proj.forward(xs)?
                .reshape((b, t, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (mut q, mut k, v) = (
            heads(&self.q_proj)?,
            heads(&self.k_proj)?,
            heads(&self.v_proj)?,
        );
        if let Some(rotary) = rotary {
            q = rotary.apply(&q)?.contiguous()?;
            k = rotary.apply(&k)?.contiguous()?;
        }
        // every position queries, only the first `keys` are attended to: a padding mask over the keys
        let (k, v) = match keys {
            Some(keys) if keys < t => (
                k.narrow(2, 0, keys)?.contiguous()?,
                v.narrow(2, 0, keys)?.contiguous()?,
            ),
            _ => (k, v),
        };
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
        uvb.pp(self.names.out_proj()).add(&self.out_proj);
    }
}

pub struct WhisperEncoderLayer {
    self_attn: Attention,
    self_attn_layer_norm: LayerNorm,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
    names: WhisperLayerNames,
}

impl WhisperEncoderLayer {
    pub fn new(cfg: &WhisperLayerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let names = cfg.names;
        let ffn = names.ffn(&vb);
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            self_attn_layer_norm: layer_norm(d, cfg.layer_norm_eps, vb.pp(names.attn_norm()))?,
            final_layer_norm: layer_norm(d, cfg.layer_norm_eps, vb.pp(names.ffn_norm()))?,
            fc1: linear_b(d, cfg.ffn_dim, true, ffn.pp("fc1"))?,
            fc2: linear_b(cfg.ffn_dim, d, true, ffn.pp("fc2"))?,
            activation: cfg.activation,
            names,
        })
    }

    /// `(batch, frames, d_model)` in and out; every frame of a batch row attends to every other.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_rotary(xs, None, None)
    }

    fn forward_rotary(
        &self,
        xs: &Tensor,
        rotary: Option<&WhisperRotary>,
        keys: Option<usize>,
    ) -> Result<Tensor> {
        let attended =
            self.self_attn
                .forward(&self.self_attn_layer_norm.forward(xs)?, rotary, keys)?;
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
        uvb.pp(self.names.attn_norm())
            .add(&self.self_attn_layer_norm);
        uvb.pp(self.names.ffn_norm()).add(&self.final_layer_norm);
        let mlp;
        let ffn = match self.names {
            WhisperLayerNames::Whisper => uvb,
            WhisperLayerNames::Llama => {
                mlp = uvb.pp("mlp");
                &mlp
            }
        };
        ffn.pp("fc1").add(&self.fc1);
        ffn.pp("fc2").add(&self.fc2);
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
    pub positions: WhisperPositions,
}

pub struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positions: Option<Tensor>,
    rotary: Option<WhisperRotary>,
    layers: Vec<WhisperEncoderLayer>,
    layer_norm: LayerNorm,
    activation: Activation,
    names: WhisperLayerNames,
}

impl WhisperEncoder {
    pub fn new(cfg: &WhisperEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let d = cfg.layer.d_model;
        let conv = |stride| Conv1dConfig {
            padding: 1,
            stride,
            ..Default::default()
        };
        let (positions, rotary) = match cfg.positions {
            // the checkpoint stores the sinusoid table; fine-tunes are read as they ship
            WhisperPositions::Learned => (
                Some(vb.get((cfg.max_source_positions, d), "embed_positions.weight")?),
                None,
            ),
            WhisperPositions::Rotary { theta, dims } => (
                None,
                Some(WhisperRotary::new(
                    theta,
                    dims,
                    cfg.max_source_positions,
                    vb.dtype(),
                    vb.device(),
                )?),
            ),
        };
        let names = cfg.layer.names;
        Ok(Self {
            conv1: conv1d(cfg.num_mel_bins, d, CONV_KERNEL, conv(1), vb.pp("conv1"))?,
            conv2: conv1d(d, d, CONV_KERNEL, conv(DOWNSAMPLE), vb.pp("conv2"))?,
            positions,
            rotary,
            layers: (0..cfg.num_layers)
                .map(|i| WhisperEncoderLayer::new(&cfg.layer, vb.pp("layers").pp(i)))
                .collect::<Result<Vec<_>>>()?,
            layer_norm: layer_norm(d, cfg.layer.layer_norm_eps, vb.pp(names.final_norm()))?,
            activation: cfg.layer.activation,
            names,
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
        match &self.positions {
            Some(positions) => xs.broadcast_add(&positions.narrow(0, 0, xs.dim(1)?)?),
            None => Ok(xs),
        }
    }

    /// The layers over embedded positions, before the final norm.
    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        self.encode_keys(xs, None)
    }

    /// [`Self::encode`] with every position attending to the first `keys` only, as a padding mask over a window's
    /// padded tail has it: the padded positions still come out.
    pub fn encode_keys(&self, xs: &Tensor, keys: Option<usize>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward_rotary(&xs, self.rotary.as_ref(), keys)?;
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
        if let Some(positions) = &self.positions {
            uvb.add_tensor("embed_positions.weight", positions.clone());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            layer.residual_tensors(&uvb.pp("layers").pp(i));
        }
        uvb.pp(self.names.final_norm()).add(&self.layer_norm);
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_tensor::{DType, Device, Tensor};

    use super::*;
    use crate::testing::load_synthesized;

    const POSITIONS: usize = 12;
    const KEYS: usize = 7;

    // a key limit is a padding mask: the positions before it come out as the encoder over those positions alone
    #[test]
    fn a_key_limit_leaves_the_real_positions_as_encoding_them_alone() -> anyhow::Result<()> {
        let cfg = WhisperEncoderConfig {
            layer: WhisperLayerConfig {
                d_model: 8,
                num_heads: 2,
                ffn_dim: 16,
                activation: Activation::Gelu,
                layer_norm_eps: 1e-5,
                k_bias: false,
                names: WhisperLayerNames::Whisper,
            },
            num_layers: 2,
            num_mel_bins: 4,
            max_source_positions: POSITIONS,
            positions: WhisperPositions::Learned,
        };
        let (encoder, _) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
            Ok(WhisperEncoder::new(&cfg, vb)?)
        })?;
        let xs = Tensor::randn(0f32, 1.0, (1, POSITIONS, 8), &Device::Cpu)?;
        let limited = encoder.encode_keys(&xs, Some(KEYS))?.narrow(1, 0, KEYS)?;
        let alone = encoder.encode(&xs.narrow(1, 0, KEYS)?)?;
        let worst = (limited - alone)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(worst < 1e-5, "{worst}");
        // the padded positions still come out, unlike the encoder over the real positions alone
        assert_eq!(encoder.encode_keys(&xs, Some(KEYS))?.dim(1)?, POSITIONS);
        Ok(())
    }
}
