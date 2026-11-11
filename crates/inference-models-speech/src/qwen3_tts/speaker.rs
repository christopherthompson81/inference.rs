//! The Base checkpoints' speaker encoder: an ECAPA-TDNN over a 24 kHz log-mel, giving the x-vector the talker's
//! prompt carries in place of a voice id.

use inference_audio::fft::plan_forward_f32;
use inference_audio::mel::slaney_filterbank;
use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};
use rustfft::num_complex::Complex;

use super::config::SpeakerEncoderConfig;
use crate::ecapa::{Ecapa, EcapaConfig, EcapaLayout};

// the reference's `mel_spectrogram` settings for the speaker encoder
const N_FFT: usize = 1024;
const HOP: usize = 256;
const MAG_EPS: f32 = 1e-9;
const LOG_FLOOR: f32 = 1e-5;

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

pub struct SpeakerEncoder {
    ecapa: Ecapa,
    n_mels: usize,
    pub sample_rate: u32,
}

impl SpeakerEncoder {
    /// `vb` is the checkpoint's `speaker_encoder` prefix.
    pub fn new(cfg: &SpeakerEncoderConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let ecapa = EcapaConfig {
            input_dim: cfg.mel_dim,
            channels: cfg.enc_channels.clone(),
            kernel_sizes: cfg.enc_kernel_sizes.clone(),
            dilations: cfg.enc_dilations.clone(),
            attention_channels: cfg.enc_attention_channels,
            res2net_scale: cfg.enc_res2net_scale,
            se_channels: cfg.enc_se_channels,
            out_dim: cfg.enc_dim,
            layout: EcapaLayout::Bare,
        };
        Ok(Self {
            ecapa: Ecapa::new(&ecapa, vb)?,
            n_mels: cfg.mel_dim,
            sample_rate: cfg.sample_rate,
        })
    }

    /// The x-vector `(enc_dim,)` of mono samples at [`SpeakerEncoder::sample_rate`].
    pub fn embed(&self, pcm: &[f32], device: &Device, dtype: DType) -> Result<Tensor> {
        let mel = log_mel(pcm, self.sample_rate, self.n_mels);
        if mel.is_empty() {
            inference_tensor::bail!("the reference audio is too short");
        }
        self.ecapa.embed(&mel, device, dtype)
    }
}
