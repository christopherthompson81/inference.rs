//! Whisper's log-mel front end, which Whisper-encoder audio models share: a centered periodic-Hann STFT, Slaney mels,
//! log10 floored 8 below a maximum, then `(x + 4) / 4`.

use crate::fft::{Complex32, plan_forward_f32};
use crate::mel::slaney_filterbank;

const POWER_FLOOR: f32 = 1e-10;
const DYNAMIC_RANGE: f32 = 8.0;
const LOG_OFFSET: f32 = 4.0;
const LOG_SCALE: f32 = 4.0;

/// The maximum the log-mel floor sits [`DYNAMIC_RANGE`] below.
#[derive(Clone, Copy, Debug)]
pub enum LogMelMax {
    /// A fixed maximum, as a streaming model sets in its config.
    Fixed(f32),
    /// The clip's own maximum, as transformers' Whisper feature extractors take it.
    Clip,
}

#[derive(Clone, Copy, Debug)]
pub struct WhisperLogMel {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
}

impl WhisperLogMel {
    /// `samples / hop_length` frames of `n_mels` values, frame-major, as `torch.stft(center=True)` without its last
    /// frame gives them.
    pub fn frames(&self, samples: &[f32], max: LogMelMax) -> Vec<Vec<f32>> {
        let n_fft = self.n_fft;
        let pad = n_fft / 2;
        if samples.is_empty() {
            return Vec::new();
        }
        // reflection padding, as torch.stft's center=True
        let mut padded = vec![0.0f32; pad + samples.len() + pad];
        for (i, p) in padded.iter_mut().enumerate().take(pad) {
            *p = samples[(pad - i).min(samples.len() - 1)];
        }
        padded[pad..pad + samples.len()].copy_from_slice(samples);
        for i in 0..pad {
            padded[pad + samples.len() + i] = samples[samples.len().saturating_sub(2 + i)];
        }
        let num_frames = ((padded.len() - n_fft) / self.hop_length + 1).saturating_sub(1);
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
            .collect();
        let filters = slaney_filterbank(self.sample_rate, n_fft, self.n_mels);
        let fft = plan_forward_f32(n_fft);
        let n_freqs = n_fft / 2 + 1;
        let mut log_mels = Vec::with_capacity(num_frames);
        for frame in 0..num_frames {
            let start = frame * self.hop_length;
            let mut buf: Vec<Complex32> = padded[start..start + n_fft]
                .iter()
                .zip(&window)
                .map(|(&s, &w)| Complex32::new(s * w, 0.0))
                .collect();
            fft.process(&mut buf);
            let power: Vec<f32> = buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();
            log_mels.push(
                filters
                    .iter()
                    .map(|filter| {
                        let sum: f32 = filter.iter().zip(&power).map(|(&c, &p)| p * c).sum();
                        sum.max(POWER_FLOOR).log10()
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let max = match max {
            LogMelMax::Fixed(max) => max,
            LogMelMax::Clip => log_mels
                .iter()
                .flatten()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
        };
        let floor = max - DYNAMIC_RANGE;
        for value in log_mels.iter_mut().flatten() {
            *value = (value.max(floor) + LOG_OFFSET) / LOG_SCALE;
        }
        log_mels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;
    const TONE_HZ: f32 = 440.0;

    fn extractor() -> WhisperLogMel {
        WhisperLogMel {
            sample_rate: RATE,
            n_fft: 400,
            hop_length: 160,
            n_mels: 128,
        }
    }

    #[test]
    fn a_clip_has_one_frame_per_hop() {
        let samples = vec![0.1f32; 1600];
        let frames = extractor().frames(&samples, LogMelMax::Clip);
        assert_eq!(frames.len(), 10);
        assert!(frames.iter().all(|frame| frame.len() == 128));
    }

    // the clip's maximum maps to (max + 4) / 4 and nothing sits more than 8 below it
    #[test]
    fn the_clip_floor_is_eight_below_its_peak() {
        let samples: Vec<f32> = (0..RATE as usize)
            .map(|i| (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / RATE as f32).sin())
            .collect();
        let frames = extractor().frames(&samples, LogMelMax::Clip);
        let values = frames.iter().flatten().copied().collect::<Vec<_>>();
        let top = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let bottom = values.iter().copied().fold(f32::INFINITY, f32::min);
        assert!((top - bottom - DYNAMIC_RANGE / LOG_SCALE).abs() < 1e-5);
    }
}
