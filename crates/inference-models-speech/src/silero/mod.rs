//! Silero VAD (v5/v6, 16 kHz): a speech probability per 32 ms chunk, and segments as `get_speech_timestamps` cuts them.

mod gguf;
mod segments;

use inference_tensor::nn::{Conv1d, Conv1dConfig, Module, VarBuilder};
use inference_tensor::{D, Device, Error, Result, Tensor};

pub use gguf::{ARCHITECTURE, VadConfig, is_silero_gguf, read_gguf, write_gguf};
pub use segments::{SegmentOptions, speech_segments};

const FILTER_LENGTH: usize = 256;
const HOP_LENGTH: usize = 128;
// a reflect pad past the window, so the STFT conv sees whole frames to the chunk's end
const STFT_PAD: usize = 64;
const FREQ_BINS: usize = FILTER_LENGTH / 2 + 1;
const KERNEL: usize = 3;
// (in, out, stride) of the four encoder convs
const ENCODER: [(usize, usize, usize); 4] =
    [(FREQ_BINS, 128, 1), (128, 64, 2), (64, 64, 2), (64, 128, 1)];
const HIDDEN: usize = 128;
// chunks per encoder pass, about 4 minutes of audio
const ENCODER_BLOCK: usize = 8192;

struct LstmState {
    h: Vec<f32>,
    c: Vec<f32>,
}

impl Default for LstmState {
    fn default() -> Self {
        Self {
            h: vec![0.; HIDDEN],
            c: vec![0.; HIDDEN],
        }
    }
}

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

/// A span of speech, in seconds from the start of the audio.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SpeechSpan {
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct VoiceActivity {
    /// Seconds of audio.
    pub duration: f64,
    /// Seconds each probability covers.
    pub chunk_seconds: f64,
    /// Speech probability per chunk.
    pub probabilities: Vec<f32>,
    pub segments: Vec<SpeechSpan>,
}

pub struct SileroVad {
    stft: Tensor,
    encoder: Vec<Conv1d>,
    w_ih_t: Tensor,
    // the recurrence and head run on the host: one step per chunk is far too small for a device launch each
    w_hh: Vec<f32>,
    bias: Vec<f32>,
    head: Vec<f32>,
    head_bias: f32,
    config: VadConfig,
    device: Device,
}

impl SileroVad {
    pub fn new(config: VadConfig, vb: VarBuilder) -> Result<Self> {
        let stft = vb
            .pp("stft_conv")
            .get((2 * FREQ_BINS, 1, FILTER_LENGTH), "weight")?;
        let encoder = ENCODER
            .iter()
            .enumerate()
            .map(|(i, &(inp, out, stride))| {
                let cfg = Conv1dConfig {
                    padding: KERNEL / 2,
                    stride,
                    ..Default::default()
                };
                {
                    let vb = vb.pp(format!("conv{}", i + 1));
                    let weight = vb.get((out, inp, KERNEL), "weight")?;
                    Ok(Conv1d::new(weight, Some(vb.get(out, "bias")?), cfg))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let lstm = vb.pp("lstm_cell");
        let gates = 4 * HIDDEN;
        let host = |t: Tensor| t.flatten_all()?.to_vec1::<f32>();
        let bias = (lstm.get(gates, "bias_ih")? + lstm.get(gates, "bias_hh")?)?;
        let head = vb.pp("final_conv");
        Ok(Self {
            stft,
            encoder,
            w_ih_t: lstm.get((gates, HIDDEN), "weight_ih")?.t()?.contiguous()?,
            w_hh: host(lstm.get((gates, HIDDEN), "weight_hh")?)?,
            bias: host(bias)?,
            head: host(head.get((1, HIDDEN, 1), "weight")?)?,
            head_bias: host(head.get(1, "bias")?)?[0],
            config,
            device: vb.device().clone(),
        })
    }

    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Speech probability per `chunk_size` samples of mono `pcm` at the model's rate; the last chunk is zero-padded.
    pub fn probabilities(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let chunks = pcm.len().div_ceil(self.config.chunk_size);
        let mut state = LstmState::default();
        let mut out = Vec::with_capacity(chunks);
        // the encoder runs in blocks so device memory stays bounded on hours of audio; only the LSTM spans them
        for first in (0..chunks).step_by(ENCODER_BLOCK) {
            let gx = self.encode(pcm, first, ENCODER_BLOCK.min(chunks - first))?;
            out.extend(self.recurrence(&gx, &mut state));
        }
        Ok(out)
    }

    // LSTM input for chunks first..first+count, each read with the context before it (zeros at first), reflect-padded
    fn encode(&self, pcm: &[f32], first: usize, count: usize) -> Result<Vec<Vec<f32>>> {
        let (chunk, context) = (self.config.chunk_size, self.config.context_size);
        let window = context + chunk;
        let padded = window + STFT_PAD;
        let mut frames = vec![0f32; count * padded];
        for (i, row) in frames.chunks_mut(padded).enumerate() {
            let c = first + i;
            for (j, slot) in row[..window].iter_mut().enumerate() {
                let at = (c * chunk + j) as isize - context as isize;
                if at >= 0 && (at as usize) < pcm.len() {
                    *slot = pcm[at as usize];
                }
            }
            for j in 0..STFT_PAD {
                row[window + j] = row[window - 2 - j];
            }
        }
        let xs = Tensor::from_vec(frames, (count, 1, padded), &self.device)?;
        let spec = xs.conv1d(&self.stft, 0, HOP_LENGTH, 1, 1)?;
        let re = spec.narrow(1, 0, FREQ_BINS)?;
        let im = spec.narrow(1, FREQ_BINS, FREQ_BINS)?;
        let mut xs = (re.sqr()? + im.sqr()?)?.sqrt()?;
        for conv in &self.encoder {
            xs = conv.forward(&xs)?.relu()?;
        }
        // one frame per chunk is left; the reference averages over it
        xs.mean(D::Minus1)?.matmul(&self.w_ih_t)?.to_vec2::<f32>()
    }

    fn recurrence(&self, gx: &[Vec<f32>], state: &mut LstmState) -> Vec<f32> {
        let sigmoid = |x: f32| 1. / (1. + (-x).exp());
        let LstmState { h, c } = state;
        let mut g = vec![0f32; 4 * HIDDEN];
        gx.iter()
            .map(|x| {
                for (j, gj) in g.iter_mut().enumerate() {
                    let row = &self.w_hh[j * HIDDEN..(j + 1) * HIDDEN];
                    *gj = x[j]
                        + self.bias[j]
                        + row.iter().zip(h.iter()).map(|(w, h)| w * h).sum::<f32>();
                }
                for j in 0..HIDDEN {
                    let (i, f) = (sigmoid(g[j]), sigmoid(g[HIDDEN + j]));
                    let (cell, o) = (g[2 * HIDDEN + j].tanh(), sigmoid(g[3 * HIDDEN + j]));
                    c[j] = f * c[j] + i * cell;
                    h[j] = o * c[j].tanh();
                }
                let logit: f32 = self
                    .head
                    .iter()
                    .zip(h.iter())
                    .map(|(w, h)| w * h.max(0.))
                    .sum();
                sigmoid(logit + self.head_bias)
            })
            .collect()
    }

    /// Probabilities and speech segments of mono `pcm` at `sample_rate`, resampled to the model's rate first.
    pub fn detect(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        options: &SegmentOptions,
    ) -> Result<VoiceActivity> {
        let rate = self.config.sample_rate;
        let pcm = inference_audio::mel::resample(pcm, sample_rate, rate).map_err(msg)?;
        let probabilities = self.probabilities(&pcm)?;
        let seconds = |samples: usize| samples as f64 / f64::from(rate);
        let segments = speech_segments(
            &probabilities,
            pcm.len(),
            rate,
            self.config.chunk_size,
            options,
        )
        .into_iter()
        .map(|(start, end)| SpeechSpan {
            start: seconds(start),
            end: seconds(end),
        })
        .collect();
        Ok(VoiceActivity {
            duration: seconds(pcm.len()),
            chunk_seconds: seconds(self.config.chunk_size),
            probabilities,
            segments,
        })
    }
}
