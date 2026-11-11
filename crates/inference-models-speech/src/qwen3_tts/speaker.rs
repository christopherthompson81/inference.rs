//! The Base checkpoints' speaker encoder: an ECAPA-TDNN (SE-Res2Net blocks, multi-layer aggregation, attentive
//! statistics pooling) over a 24 kHz log-mel, giving the x-vector the talker's prompt carries in place of a voice id.

use inference_audio::fft::plan_forward_f32;
use inference_audio::mel::slaney_filterbank;
use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::ops::{sigmoid, softmax_last_dim};
use inference_tensor::{D, Device, Result, Tensor};
use rustfft::num_complex::Complex;

use super::config::SpeakerEncoderConfig;

// the reference's `mel_spectrogram` settings for the speaker encoder
const N_FFT: usize = 1024;
const HOP: usize = 256;
const MAG_EPS: f32 = 1e-9;
const LOG_FLOOR: f32 = 1e-5;
const STATS_EPS: f64 = 1e-12;

/// `[frames][n_mels]` log-mel: reflect padding of `(n_fft - hop) / 2` each side, uncentered periodic-Hann STFT
/// magnitudes, Slaney filters, natural log.
pub fn log_mel(pcm: &[f32], sample_rate: u32, n_mels: usize) -> Vec<Vec<f32>> {
    let pad = (N_FFT - HOP) / 2;
    let len = pcm.len();
    let at = |i: isize| -> f32 {
        let last = len as isize - 1;
        let i = if i < 0 {
            -i
        } else if i > last {
            2 * last - i
        } else {
            i
        };
        pcm[i.clamp(0, last) as usize]
    };
    #[allow(clippy::cast_precision_loss)]
    let window: Vec<f32> = (0..N_FFT)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos())
        .collect();
    let filters = slaney_filterbank(sample_rate, N_FFT, n_mels);
    let fft = plan_forward_f32(N_FFT);
    let padded = len + 2 * pad;
    let frames = if padded < N_FFT {
        0
    } else {
        (padded - N_FFT) / HOP + 1
    };
    let mut buffer = vec![Complex::new(0f32, 0f32); N_FFT];
    (0..frames)
        .map(|t| {
            let start = (t * HOP) as isize - pad as isize;
            for (n, slot) in buffer.iter_mut().enumerate() {
                *slot = Complex::new(at(start + n as isize) * window[n], 0.0);
            }
            fft.process(&mut buffer);
            let mags: Vec<f32> = buffer[..N_FFT / 2 + 1]
                .iter()
                .map(|c| (c.norm_sqr() + MAG_EPS).sqrt())
                .collect();
            filters
                .iter()
                .map(|f| {
                    f.iter()
                        .zip(&mags)
                        .map(|(w, m)| w * m)
                        .sum::<f32>()
                        .max(LOG_FLOOR)
                        .ln()
                })
                .collect()
        })
        .collect()
}

// a 'same' conv with reflect padding, as the reference's `padding="same", padding_mode="reflect"`
struct Conv {
    weight: Tensor,
    bias: Tensor,
    dilation: usize,
    pad: usize,
}

impl Conv {
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            weight: vb.get((out_ch, in_ch, kernel), "weight")?,
            bias: vb.get(out_ch, "bias")?,
            dilation,
            pad: dilation * (kernel - 1) / 2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.pad > 0 {
            let len = xs.dim(D::Minus1)?;
            let index: Vec<u32> = (0..len + 2 * self.pad)
                .map(|i| {
                    let i = i as isize - self.pad as isize;
                    let last = len as isize - 1;
                    let i = if i < 0 {
                        -i
                    } else if i > last {
                        2 * last - i
                    } else {
                        i
                    };
                    u32::try_from(i).unwrap_or(0)
                })
                .collect();
            let index = Tensor::from_vec(index, len + 2 * self.pad, xs.device())?;
            xs.index_select(&index, 2)?
        } else {
            xs.clone()
        };
        xs.conv1d(&self.weight, 0, 1, self.dilation, 1)?
            .broadcast_add(&self.bias.reshape((1, (), 1))?)
    }

    fn relu(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)?.relu()
    }
}

struct SeRes2Net {
    tdnn1: Conv,
    res2net: Vec<Conv>,
    tdnn2: Conv,
    se1: Conv,
    se2: Conv,
    scale: usize,
}

impl SeRes2Net {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.tdnn1.relu(xs)?;
        let parts = h.chunk(self.scale, 1)?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);
        for (i, part) in parts.iter().enumerate() {
            let out = match i {
                0 => part.clone(),
                1 => self.res2net[0].relu(part)?,
                _ => {
                    self.res2net[i - 1].relu(&(part + outputs.last().expect("an earlier part"))?)?
                }
            };
            outputs.push(out);
        }
        let h = self.tdnn2.relu(&Tensor::cat(&outputs, 1)?)?;
        let gate = self.se1.relu(&h.mean_keepdim(2)?)?;
        let gate = sigmoid(&self.se2.forward(&gate)?)?;
        h.broadcast_mul(&gate)? + xs
    }
}

// mean and std over time of `xs` `(1, c, t)` under weights `w` that sum to 1 per channel
fn statistics(xs: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
    let mean = xs.broadcast_mul(w)?.sum_keepdim(2)?;
    let var = xs
        .broadcast_sub(&mean)?
        .sqr()?
        .broadcast_mul(w)?
        .sum_keepdim(2)?;
    Ok((mean, var.clamp(STATS_EPS, f64::INFINITY)?.sqrt()?))
}

pub struct SpeakerEncoder {
    first: Conv,
    blocks: Vec<SeRes2Net>,
    mfa: Conv,
    asp_tdnn: Conv,
    asp_conv: Conv,
    fc: Conv,
    n_mels: usize,
    pub sample_rate: u32,
}

impl SpeakerEncoder {
    /// `vb` is the checkpoint's `speaker_encoder` prefix.
    pub fn new(cfg: &SpeakerEncoderConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let ch = &cfg.enc_channels;
        let last = ch.len() - 1;
        let blocks = (1..last)
            .map(|i| {
                let vb = vb.pp("blocks").pp(i);
                let width = ch[i] / cfg.enc_res2net_scale;
                Ok(SeRes2Net {
                    tdnn1: Conv::new(ch[i - 1], ch[i], 1, 1, &vb.pp("tdnn1").pp("conv"))?,
                    res2net: (0..cfg.enc_res2net_scale - 1)
                        .map(|j| {
                            Conv::new(
                                width,
                                width,
                                cfg.enc_kernel_sizes[i],
                                cfg.enc_dilations[i],
                                &vb.pp("res2net_block").pp("blocks").pp(j).pp("conv"),
                            )
                        })
                        .collect::<Result<_>>()?,
                    tdnn2: Conv::new(ch[i], ch[i], 1, 1, &vb.pp("tdnn2").pp("conv"))?,
                    se1: Conv::new(
                        ch[i],
                        cfg.enc_se_channels,
                        1,
                        1,
                        &vb.pp("se_block").pp("conv1"),
                    )?,
                    se2: Conv::new(
                        cfg.enc_se_channels,
                        ch[i],
                        1,
                        1,
                        &vb.pp("se_block").pp("conv2"),
                    )?,
                    scale: cfg.enc_res2net_scale,
                })
            })
            .collect::<Result<_>>()?;
        let asp = vb.pp("asp");
        Ok(Self {
            first: Conv::new(
                cfg.mel_dim,
                ch[0],
                cfg.enc_kernel_sizes[0],
                cfg.enc_dilations[0],
                &vb.pp("blocks").pp(0).pp("conv"),
            )?,
            blocks,
            mfa: Conv::new(
                ch[last],
                ch[last],
                cfg.enc_kernel_sizes[last],
                cfg.enc_dilations[last],
                &vb.pp("mfa").pp("conv"),
            )?,
            asp_tdnn: Conv::new(
                3 * ch[last],
                cfg.enc_attention_channels,
                1,
                1,
                &asp.pp("tdnn").pp("conv"),
            )?,
            asp_conv: Conv::new(cfg.enc_attention_channels, ch[last], 1, 1, &asp.pp("conv"))?,
            fc: Conv::new(2 * ch[last], cfg.enc_dim, 1, 1, &vb.pp("fc"))?,
            n_mels: cfg.mel_dim,
            sample_rate: cfg.sample_rate,
        })
    }

    /// The x-vector `(enc_dim,)` of mono samples at [`SpeakerEncoder::sample_rate`].
    pub fn embed(
        &self,
        pcm: &[f32],
        device: &Device,
        dtype: inference_tensor::DType,
    ) -> Result<Tensor> {
        let mel = log_mel(pcm, self.sample_rate, self.n_mels);
        let frames = mel.len();
        if frames == 0 {
            inference_tensor::bail!("the reference audio is too short");
        }
        let flat: Vec<f32> = mel.into_iter().flatten().collect();
        let xs = Tensor::from_vec(flat, (1, frames, self.n_mels), device)?
            .to_dtype(dtype)?
            .transpose(1, 2)?
            .contiguous()?;
        let mut h = self.first.relu(&xs)?;
        let mut outputs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            h = block.forward(&h)?;
            outputs.push(h.clone());
        }
        let h = self.mfa.relu(&Tensor::cat(&outputs, 1)?)?;
        #[allow(clippy::cast_precision_loss)]
        let uniform = (Tensor::ones((1, 1, frames), h.dtype(), device)? / frames as f64)?;
        let (mean, std) = statistics(&h, &uniform)?;
        let context = Tensor::cat(
            &[
                h.clone(),
                mean.repeat((1, 1, frames))?,
                std.repeat((1, 1, frames))?,
            ],
            1,
        )?;
        let attention = self
            .asp_conv
            .forward(&self.asp_tdnn.relu(&context)?.tanh()?)?;
        let attention = softmax_last_dim(&attention.contiguous()?)?;
        let (mean, std) = statistics(&h, &attention)?;
        let pooled = Tensor::cat(&[mean, std], 1)?;
        self.fc.forward(&pooled)?.flatten_all()
    }
}
