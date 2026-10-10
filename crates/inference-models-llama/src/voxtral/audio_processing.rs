#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use anyhow::Result;
use inference_audio::AudioInput;
use inference_audio::fft::{Complex32, plan_forward_f32};
use inference_tensor::{Device, Tensor};

use super::config::AudioEncodingArgs;

/// Number of silence tokens to left-pad audio (matches voxmlx reference).
const N_LEFT_PAD_TOKENS: usize = 32;
/// Number of silence tokens to right-pad audio (matches voxmlx reference).
const N_RIGHT_PAD_TOKENS: usize = 17;

/// Whisper-style mel spectrogram processor for Voxtral audio encoder.
pub struct VoxtralAudioProcessor {
    sampling_rate: u32,
    frame_rate: f32,
    num_mel_bins: usize,
    hop_length: usize,
    window_size: usize,
    global_log_mel_max: f32,
}

impl VoxtralAudioProcessor {
    pub fn new(cfg: &AudioEncodingArgs) -> Self {
        Self {
            sampling_rate: cfg.sampling_rate,
            frame_rate: cfg.frame_rate as f32,
            num_mel_bins: cfg.num_mel_bins,
            hop_length: cfg.hop_length,
            window_size: cfg.window_size,
            global_log_mel_max: cfg.global_log_mel_max as f32,
        }
    }

    /// Number of samples per streaming token (sampling_rate / frame_rate).
    fn samples_per_token(&self) -> usize {
        (self.sampling_rate as f32 / self.frame_rate) as usize
    }

    /// Process audio input into a mel spectrogram tensor.
    /// Left-pads with 32 tokens of silence and right-pads with 17 tokens of silence
    /// to match the reference implementation.
    /// Returns [1, T, num_mel_bins] tensor.
    pub fn process_audio(&self, audio: &AudioInput, device: &Device) -> Result<Tensor> {
        let mono = audio.to_mono();

        // Resample if necessary
        let samples = if audio.sample_rate != self.sampling_rate {
            inference_audio::mel::resample(&mono, audio.sample_rate, self.sampling_rate)?
        } else {
            mono
        };

        // Pad audio with silence: left_pad + audio + right_pad
        let spt = self.samples_per_token();
        let left_pad = N_LEFT_PAD_TOKENS * spt;
        let right_pad = N_RIGHT_PAD_TOKENS * spt;
        let mut padded = vec![0.0f32; left_pad + samples.len() + right_pad];
        padded[left_pad..left_pad + samples.len()].copy_from_slice(&samples);

        let mel = self.compute_mel_spectrogram(&padded)?;
        let num_frames = mel.len();
        if num_frames == 0 {
            anyhow::bail!("Audio too short to produce mel frames");
        }

        let data: Vec<f32> = mel.into_iter().flatten().collect();

        let tensor = Tensor::from_vec(data, (1, num_frames, self.num_mel_bins), device)?;
        Ok(tensor)
    }

    /// Centered STFT mel spectrogram matching `torch.stft(center=True)`.
    /// Applies reflection padding of n_fft//2 on each side, then drops the last STFT frame.
    fn compute_mel_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        let n_fft = self.window_size;
        let hop = self.hop_length;
        let n_freqs = n_fft / 2 + 1;
        let pad = n_fft / 2;

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Reflection-pad the input by n_fft//2 on each side (matching torch.stft center=True)
        let padded_len = pad + samples.len() + pad;
        let mut padded = vec![0.0f32; padded_len];
        // Left reflection: samples[pad], samples[pad-1], ..., samples[1]
        for (i, p) in padded.iter_mut().enumerate().take(pad) {
            let src_idx = (pad - i).min(samples.len() - 1);
            *p = samples[src_idx];
        }
        // Center: copy original samples
        padded[pad..pad + samples.len()].copy_from_slice(samples);
        // Right reflection: samples[len-2], samples[len-3], ...
        for i in 0..pad {
            let src_idx = samples.len().saturating_sub(2 + i);
            padded[pad + samples.len() + i] = samples[src_idx];
        }

        let total_frames = (padded_len - n_fft) / hop + 1;
        // Drop last frame (matching HF: stft[..., :-1])
        let num_frames = total_frames.saturating_sub(1);

        // Hann window (periodic: w[n] = 0.5*(1 - cos(2*pi*n/N)))
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
            .collect();

        let mel_filters =
            inference_audio::mel::slaney_filterbank(self.sampling_rate, n_fft, self.num_mel_bins);

        let fft = plan_forward_f32(n_fft);

        let mut mel_features = Vec::with_capacity(num_frames);
        let log_mel_floor = self.global_log_mel_max - 8.0;

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;

            let mut buf: Vec<Complex32> = padded[start..start + n_fft]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex32::new(s * w, 0.0))
                .collect();

            fft.process(&mut buf);

            let power: Vec<f32> = buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();

            let mut mel_frame = vec![0.0f32; self.num_mel_bins];
            for (mel_idx, filter) in mel_filters.iter().enumerate() {
                let mut sum = 0.0f32;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    if freq_idx < power.len() {
                        sum += power[freq_idx] * coeff;
                    }
                }
                let log_val = sum.max(1e-10).log10();
                let clamped = log_val.max(log_mel_floor);
                mel_frame[mel_idx] = (clamped + 4.0) / 4.0;
            }

            mel_features.push(mel_frame);
        }

        Ok(mel_features)
    }
}
