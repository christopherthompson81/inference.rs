//! Whisper's log-mel front end, which Whisper-encoder audio models share: a centered periodic-Hann STFT, Slaney mels,
//! log10 floored 8 below a maximum, then `(x + 4) / 4`.

use crate::fft::{Complex32, plan_forward_f32};
use crate::mel::slaney_filterbank;

const POWER_FLOOR: f32 = 1e-10;
const DYNAMIC_RANGE: f32 = 8.0;
const LOG_OFFSET: f32 = 4.0;
const LOG_SCALE: f32 = 4.0;
// frames a peak pass computes at once
const PEAK_BLOCK_FRAMES: usize = 3000;

/// The maximum the log-mel floor sits `DYNAMIC_RANGE` below.
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
        let mut frames = self.log_frames(samples, 0..self.frame_count(samples.len()));
        let max = match max {
            LogMelMax::Fixed(max) => max,
            LogMelMax::Clip => frames
                .iter()
                .flatten()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
        };
        self.scale(&mut frames, max);
        frames
    }

    /// The frames `samples` hold.
    pub fn frame_count(&self, samples: usize) -> usize {
        if samples == 0 {
            return 0;
        }
        let padded = samples + 2 * (self.n_fft / 2);
        ((padded - self.n_fft) / self.hop_length + 1).saturating_sub(1)
    }

    /// The largest log10 mel power of all of `samples`' frames, the [`LogMelMax::Clip`] maximum, read a frame at a time
    /// so a long recording's log-mel is never held whole.
    pub fn peak(&self, samples: &[f32]) -> f32 {
        let count = self.frame_count(samples.len());
        let mut peak = f32::NEG_INFINITY;
        let mut start = 0;
        while start < count {
            let end = (start + PEAK_BLOCK_FRAMES).min(count);
            peak = self
                .log_frames(samples, start..end)
                .iter()
                .flatten()
                .copied()
                .fold(peak, f32::max);
            start = end;
        }
        peak
    }

    /// Frames `range` of `samples` floored `DYNAMIC_RANGE` below `max` and scaled, as [`Self::frames`] gives them.
    pub fn frames_in(
        &self,
        samples: &[f32],
        range: std::ops::Range<usize>,
        max: f32,
    ) -> Vec<Vec<f32>> {
        let mut frames = self.log_frames(samples, range);
        self.scale(&mut frames, max);
        frames
    }

    // log10 mel powers of frames `range`, each from the reflection-padded signal around it
    fn log_frames(&self, samples: &[f32], range: std::ops::Range<usize>) -> Vec<Vec<f32>> {
        let n_fft = self.n_fft;
        let pad = n_fft / 2;
        let len = samples.len();
        // torch.stft's center=True padding, read where needed rather than copied
        let at = |p: usize| {
            if p < pad {
                samples[(pad - p).min(len - 1)]
            } else if p < pad + len {
                samples[p - pad]
            } else {
                samples[len.saturating_sub(2 + (p - pad - len))]
            }
        };
        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
            .collect();
        let filters = slaney_filterbank(self.sample_rate, n_fft, self.n_mels);
        let fft = plan_forward_f32(n_fft);
        let n_freqs = n_fft / 2 + 1;
        range
            .map(|frame| {
                let start = frame * self.hop_length;
                let mut buf: Vec<Complex32> = (0..n_fft)
                    .zip(&window)
                    .map(|(k, &w)| Complex32::new(at(start + k) * w, 0.0))
                    .collect();
                fft.process(&mut buf);
                let power: Vec<f32> = buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();
                filters
                    .iter()
                    .map(|filter| {
                        let sum: f32 = filter.iter().zip(&power).map(|(&c, &p)| p * c).sum();
                        sum.max(POWER_FLOOR).log10()
                    })
                    .collect()
            })
            .collect()
    }

    fn scale(&self, frames: &mut [Vec<f32>], max: f32) {
        let floor = max - DYNAMIC_RANGE;
        for value in frames.iter_mut().flatten() {
            *value = (value.max(floor) + LOG_OFFSET) / LOG_SCALE;
        }
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

    // a recording's frames read a window at a time, against its peak, are the whole recording's frames
    #[test]
    fn frames_read_in_windows_match_the_whole() {
        let samples: Vec<f32> = (0..3 * RATE as usize + 77)
            .map(|i| (i as f32 * 0.37).sin() * (1.0 + (i as f32 * 1e-4).cos()))
            .collect();
        let mel = extractor();
        let whole = mel.frames(&samples, LogMelMax::Clip);
        let peak = mel.peak(&samples);
        let count = mel.frame_count(samples.len());
        assert_eq!(count, whole.len());
        let mut pieced = Vec::new();
        for start in (0..count).step_by(97) {
            pieced.extend(mel.frames_in(&samples, start..(start + 97).min(count), peak));
        }
        assert_eq!(pieced, whole);
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
