#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use anyhow::Result;
use inference_audio::AudioInput;
use inference_audio::whisper::{LogMelMax, WhisperLogMel};
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

    fn compute_mel_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        let extractor = WhisperLogMel {
            sample_rate: self.sampling_rate,
            n_fft: self.window_size,
            hop_length: self.hop_length,
            n_mels: self.num_mel_bins,
        };
        Ok(extractor.frames(samples, LogMelMax::Fixed(self.global_log_mel_max)))
    }
}
