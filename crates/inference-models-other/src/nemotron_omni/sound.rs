//! Nemotron-H Omni's sound tower: a Parakeet FastConformer over a clip's log-mel, then RMSNorm, an MLP with squared
//! ReLU into the decoder's width (the reference's `SoundEncoder` and `SoundProjection`).

use std::collections::HashMap;
use std::sync::Arc;

use inference_audio::nemo::NemoMelConfig;
use inference_models_speech::parakeet::{Encoder, EncoderConfig};
use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Module, VarBuilder};
use inference_tensor::{DType, Result, Tensor};
use serde::Serialize;

use crate::layers::RmsNorm;

pub const ENCODER_PREFIX: &str = "sound_encoder.encoder";
pub const PROJECTION_PREFIX: &str = "sound_projection";
// transformers' `ParakeetFeatureExtractor` defaults, which the reference builds with only the rate and bins
pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const PREEMPHASIS: f32 = 0.97;
// `audio_model.py`'s RMSNorm
const PROJECTION_EPS: f64 = 1e-5;

/// `sound_config`: transformers' `ParakeetEncoderConfig` fields with the projector's.
#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct SoundConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub conv_kernel_size: usize,
    #[serde(default)]
    pub convolution_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub subsampling_factor: usize,
    pub num_mel_bins: usize,
    pub projection_hidden_size: usize,
    #[serde(default)]
    pub projection_bias: bool,
    pub sampling_rate: u32,
}

impl SoundConfig {
    pub fn encoder(&self) -> EncoderConfig {
        EncoderConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_mel_bins: self.num_mel_bins,
            conv_kernel_size: self.conv_kernel_size,
            subsampling_conv_channels: self.subsampling_conv_channels,
            subsampling_conv_kernel_size: self.subsampling_conv_kernel_size,
            subsampling_conv_stride: self.subsampling_conv_stride,
            subsampling_factor: self.subsampling_factor,
            attention_bias: self.attention_bias,
            convolution_bias: self.convolution_bias,
            scale_input: false,
            model_type: None,
            sliding_window: None,
            default_num_lookahead_tokens: None,
        }
    }

    pub fn mel(&self) -> NemoMelConfig {
        NemoMelConfig {
            sample_rate: self.sampling_rate,
            n_fft: N_FFT,
            win_length: WIN_LENGTH,
            hop_length: HOP_LENGTH,
            n_mels: self.num_mel_bins,
            preemphasis: Some(PREEMPHASIS),
            normalize: true,
            bf16_constants: false,
        }
    }

    /// The tower's weights, encoder and projector, for sizing.
    pub fn weight_elements(&self, lm_hidden: usize) -> usize {
        let (h, inter, k) = (
            self.hidden_size,
            self.intermediate_size,
            self.conv_kernel_size,
        );
        let (ch, sk) = (
            self.subsampling_conv_channels,
            self.subsampling_conv_kernel_size,
        );
        // two half-step FFNs, q/k/v/o and the position projection, the conv module's 2h and h pointwise and depthwise
        let layer = 4 * h * inter + 8 * h * h + h * k;
        let freq = self.num_mel_bins / self.subsampling_factor;
        let subsampling = ch * sk * sk + 2 * (ch * sk * sk + ch * ch) + ch * freq * h;
        let projection = h * self.projection_hidden_size + self.projection_hidden_size * lm_hidden;
        self.num_hidden_layers * layer + subsampling + projection
    }

    /// The encoder frames (and sound tokens) of `mel_frames` log-mel frames: three stride-2 convs, padded by one.
    pub fn frames(&self, mel_frames: usize) -> usize {
        let mut frames = mel_frames;
        let mut factor = self.subsampling_factor;
        while factor > 1 {
            frames = frames.div_ceil(self.subsampling_conv_stride);
            factor /= self.subsampling_conv_stride;
        }
        frames
    }
}

pub struct SoundTower {
    encoder: Encoder,
    norm: RmsNorm,
    linear1: Arc<dyn QuantMethod>,
    linear2: Arc<dyn QuantMethod>,
    dtype: DType,
}

// the encoder takes the plain var builder its speech models load through
fn prefixed_tensors(vb: &ShardedVarBuilder, prefix: &str) -> Result<HashMap<String, Tensor>> {
    let names = vb.tensor_names().unwrap_or_default();
    let dot = format!("{prefix}.");
    names
        .iter()
        .filter_map(|name| name.strip_prefix(&dot).map(|rest| (name, rest)))
        .map(|(name, rest)| Ok((rest.to_string(), vb.get_unchecked(name)?)))
        .collect()
}

impl SoundTower {
    pub fn new(cfg: &SoundConfig, vb: ShardedVarBuilder, lm_hidden: usize) -> Result<Self> {
        let dtype = vb.dtype();
        let weights = prefixed_tensors(&vb, ENCODER_PREFIX)?;
        let encoder = Encoder::new(
            &cfg.encoder(),
            VarBuilder::from_tensors(weights, dtype, vb.device()),
        )?;
        let projection = vb.pp(PROJECTION_PREFIX);
        let h = cfg.hidden_size;
        let linear = |i: usize, o: usize, name: &str| {
            inference_quant::linear_b(i, o, cfg.projection_bias, &None, projection.pp(name))
        };
        Ok(Self {
            encoder,
            norm: RmsNorm::new(h, PROJECTION_EPS, projection.pp("norm"))?,
            linear1: linear(h, cfg.projection_hidden_size, "linear1")?,
            linear2: linear(cfg.projection_hidden_size, lm_hidden, "linear2")?,
            dtype,
        })
    }

    /// The encoder's `(encoder frames, hidden)` output for one clip's `(frames, mels)` normalized log-mel.
    pub fn encode(&self, features: &Tensor) -> Result<Tensor> {
        self.encoder
            .forward(&features.unsqueeze(0)?.to_dtype(self.dtype)?)?
            .squeeze(0)
    }

    /// Encoder frames into the decoder's width.
    pub fn project(&self, encoded: &Tensor) -> Result<Tensor> {
        let xs = self.norm.forward(encoded)?;
        let xs = self.linear1.forward(&xs)?.relu()?.sqr()?;
        self.linear2.forward(&xs)
    }

    /// One clip's `(encoder frames, lm hidden)` embeddings.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        self.project(&self.encode(features)?)
    }
}

#[cfg(test)]
mod tests {
    use super::SoundConfig;

    fn config(factor: usize) -> SoundConfig {
        serde_json::from_value(serde_json::json!({
            "hidden_size": 1024, "num_attention_heads": 8, "num_hidden_layers": 24, "intermediate_size": 4096,
            "conv_kernel_size": 9, "subsampling_conv_channels": 256, "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2, "subsampling_factor": factor, "num_mel_bins": 128,
            "projection_hidden_size": 4096, "sampling_rate": 16000,
        }))
        .unwrap()
    }

    // transformers' `_get_subsampling_output_length`: each stride-2 conv (kernel 3, padding 1) rounds up
    #[test]
    fn a_clip_takes_one_token_per_encoder_frame() {
        let cfg = config(8);
        assert_eq!(cfg.frames(907), 114);
        assert_eq!(cfg.frames(908), 114);
        assert_eq!(cfg.frames(50), 7);
        assert_eq!(cfg.frames(1), 1);
        assert_eq!(config(4).frames(907), 227);
    }
}
