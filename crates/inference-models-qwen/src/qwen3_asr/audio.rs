//! The AuT audio encoder of Qwen3-ASR and Qwen3-Omni: per-chunk conv2d downsampling of the log-mel, Whisper encoder
//! layers attending within windows of chunks, and a two-layer projector to the text width.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module};
use inference_tensor::{Device, Result, Tensor};

use crate::audio::whisper::{WhisperEncoderLayer, WhisperLayerConfig, sinusoid_positions};
use crate::layers::{Activation, conv2d, layer_norm, linear_b, linear_no_bias};
use crate::utils::unvarbuilder::UnVarBuilder;

use super::config::AudioConfig;

const CONV_LAYERS: usize = 3;
const CONV_KERNEL: usize = 3;
const CONV_STRIDE: usize = 2;
const LAYER_NORM_EPS: f64 = 1e-5;

// one stride-2, padding-1 convolution's output length; an empty input stays empty
fn conv_length(length: usize) -> usize {
    if length == 0 {
        0
    } else {
        (length - 1) / CONV_STRIDE + 1
    }
}

fn downsampled(length: usize) -> usize {
    (0..CONV_LAYERS).fold(length, |length, _| conv_length(length))
}

/// The encoder tokens of a clip's `mel_frames`: its full chunks downsample to the same count each, the last to its own.
pub fn audio_token_count(mel_frames: usize, n_window: usize) -> usize {
    let chunk = 2 * n_window;
    mel_frames / chunk * downsampled(chunk) + downsampled(mel_frames % chunk)
}

pub struct AudioEncoder {
    convs: Vec<Conv2d>,
    conv_out: Linear,
    positions: Tensor,
    layers: Vec<WhisperEncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    activation: Activation,
    chunk: usize,
    chunks_per_window: usize,
}

impl AudioEncoder {
    pub fn new(cfg: &AudioConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: CONV_STRIDE,
            ..Default::default()
        };
        let channels = cfg.downsample_hidden_size;
        let convs = (1..=CONV_LAYERS)
            .map(|i| {
                let input = if i == 1 { 1 } else { channels };
                conv2d(
                    input,
                    channels,
                    CONV_KERNEL,
                    conv_cfg,
                    vb.pp(format!("conv2d{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let freq = downsampled(cfg.num_mel_bins);
        let chunk = 2 * cfg.n_window;
        let tokens_per_chunk = downsampled(chunk);
        let positions = Tensor::from_vec(
            sinusoid_positions(tokens_per_chunk, cfg.d_model),
            (tokens_per_chunk, cfg.d_model),
            device,
        )?
        .to_dtype(vb.dtype())?;
        let layer = WhisperLayerConfig {
            d_model: cfg.d_model,
            num_heads: cfg.encoder_attention_heads,
            ffn_dim: cfg.encoder_ffn_dim,
            activation: cfg.activation_function,
            layer_norm_eps: LAYER_NORM_EPS,
            k_bias: true,
        };
        let layers = (0..cfg.encoder_layers)
            .map(|i| WhisperEncoderLayer::new(&layer, vb.pp("layers").pp(i)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            convs,
            conv_out: linear_no_bias(channels * freq, cfg.d_model, vb.pp("conv_out"))?,
            positions,
            layers,
            ln_post: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("ln_post"))?,
            proj1: linear_b(cfg.d_model, cfg.d_model, true, vb.pp("proj1"))?,
            proj2: linear_b(cfg.d_model, cfg.output_dim, true, vb.pp("proj2"))?,
            activation: cfg.activation_function,
            chunk,
            chunks_per_window: cfg.n_window_infer / chunk,
        })
    }

    /// One clip's `(n_mels, frames)` log-mel, `frames` a multiple of the chunk with the last `frames - valid` zero
    /// padding, to its `(audio_token_count(valid), output_dim)` text-width embeddings.
    pub fn forward(&self, mel: &Tensor, valid: usize) -> Result<Tensor> {
        let (n_mels, frames) = mel.dims2()?;
        let chunks = frames / self.chunk;
        let mut xs = mel
            .reshape((n_mels, chunks, self.chunk))?
            .transpose(0, 1)?
            .unsqueeze(1)?
            .contiguous()?
            .to_dtype(self.positions.dtype())?;
        for conv in &self.convs {
            xs = self.activation.forward(&conv.forward(&xs)?)?;
        }
        // (chunks, channels, freq, time) -> (chunks, time, channels * freq)
        let (_, channels, freq, time) = xs.dims4()?;
        let xs =
            xs.permute((0, 3, 1, 2))?
                .contiguous()?
                .reshape((chunks, time, channels * freq))?;
        let xs = self.conv_out.forward(&xs)?.broadcast_add(&self.positions)?;
        // keep each chunk's valid positions, in order
        let kept = (0..chunks)
            .map(|chunk| {
                (
                    chunk,
                    downsampled(valid.saturating_sub(chunk * self.chunk).min(self.chunk)),
                )
            })
            .filter(|&(_, length)| length > 0)
            .map(|(chunk, length)| xs.get(chunk)?.narrow(0, 0, length))
            .collect::<Result<Vec<_>>>()?;
        let xs = Tensor::cat(&kept, 0)?;
        // attention is block-diagonal over windows of chunks: each window runs the layers on its own
        let window = self.chunks_per_window * downsampled(self.chunk.min(valid));
        let tokens = xs.dim(0)?;
        let full = tokens / window;
        let mut windows = Vec::with_capacity(2);
        if full > 0 {
            windows.push(
                xs.narrow(0, 0, full * window)?
                    .reshape((full, window, ()))?,
            );
        }
        if tokens % window != 0 {
            windows.push(xs.narrow(0, full * window, tokens % window)?.unsqueeze(0)?);
        }
        let encoded = windows
            .into_iter()
            .map(|mut window| {
                for layer in &self.layers {
                    window = layer.forward(&window)?;
                }
                window.flatten_to(1)
            })
            .collect::<Result<Vec<_>>>()?;
        let xs = self.ln_post.forward(&Tensor::cat(&encoded, 0)?)?;
        let xs = self.activation.forward(&self.proj1.forward(&xs)?)?;
        self.proj2.forward(&xs)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        for (i, conv) in self.convs.iter().enumerate() {
            uvb.pp(format!("conv2d{}", i + 1)).add(conv);
        }
        uvb.pp("conv_out").add(&self.conv_out);
        for (i, layer) in self.layers.iter().enumerate() {
            layer.residual_tensors(&uvb.pp("layers").pp(i));
        }
        uvb.pp("ln_post").add(&self.ln_post);
        uvb.pp("proj1").add(&self.proj1);
        uvb.pp("proj2").add(&self.proj2);
        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod tests {
    use super::audio_token_count;

    const N_WINDOW: usize = 50;

    // transformers' _get_feat_extract_output_lengths: 13 per full chunk, then the remainder's three halvings
    #[test]
    fn token_counts_follow_the_chunked_downsampling() {
        let reference = |frames: usize| {
            let leave = frames % 100;
            let feat = if leave == 0 { 0 } else { (leave - 1) / 2 + 1 };
            let feat = if feat == 0 { 0 } else { (feat - 1) / 2 + 1 };
            let feat = if feat == 0 { 0 } else { (feat - 1) / 2 + 1 };
            feat + frames / 100 * 13
        };
        for frames in [1, 7, 8, 9, 99, 100, 101, 250, 800, 801, 3000] {
            assert_eq!(
                audio_token_count(frames, N_WINDOW),
                reference(frames),
                "{frames} frames"
            );
        }
    }
}
