//! Granite Speech's front end: torchaudio's `MelSpectrogram` (HTK mel scale, no filter normalization), log10 with
//! an 8-decade floor under the clip's peak, and frame pairs stacked into one feature.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use inference_audio::fft::{Complex32, plan_forward_f32};

pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;
// frames stacked into one encoder input
pub const STACKED_FRAMES: usize = 2;
const LOG_FLOOR: f32 = 1e-10;
// decades kept under the clip's loudest bin
const DYNAMIC_RANGE: f32 = 8.0;
const LOG_SCALE: f32 = 4.0;

fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// torchaudio's `melscale_fbanks(n_fft / 2 + 1, 0, sr / 2, n_mels, sr, norm=None, mel_scale="htk")`, `[freq][mel]`.
fn htk_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1;
    let nyquist = f64::from(SAMPLE_RATE / 2);
    let freqs = (0..n_freqs)
        .map(|i| nyquist * i as f64 / (n_freqs - 1) as f64)
        .collect::<Vec<_>>();
    let (m_min, m_max) = (hz_to_mel_htk(0.0), hz_to_mel_htk(nyquist));
    let points = (0..N_MELS + 2)
        .map(|i| mel_to_hz_htk(m_min + (m_max - m_min) * i as f64 / (N_MELS + 1) as f64))
        .collect::<Vec<_>>();
    freqs
        .iter()
        .map(|&f| {
            (0..N_MELS)
                .map(|m| {
                    let down = (f - points[m]) / (points[m + 1] - points[m]);
                    let up = (points[m + 2] - f) / (points[m + 2] - points[m + 1]);
                    down.min(up).max(0.0) as f32
                })
                .collect()
        })
        .collect()
}

/// Tokens a clip of `samples` 16 kHz samples becomes: windows of encoder frames, `window / downsample` tokens each.
pub fn audio_token_count(samples: usize, window: usize, downsample: usize) -> usize {
    let mel_frames = samples / HOP_LENGTH + 1;
    let encoder_frames = mel_frames / STACKED_FRAMES;
    encoder_frames.div_ceil(window) * (window / downsample)
}

/// `(frames, N_MELS * STACKED_FRAMES)` row-major features of a mono 16 kHz clip.
pub fn log_mel_features(samples: &[f32]) -> (Vec<f32>, usize) {
    let pad = N_FFT / 2;
    // torch.stft's centered reflect padding; a clip shorter than the pad reflects what it has
    let reflect = |i: isize| -> f32 {
        let n = samples.len() as isize;
        if n == 0 {
            return 0.0;
        }
        let period = 2 * (n - 1).max(1);
        let mut j = i.rem_euclid(period);
        if j >= n {
            j = period - j;
        }
        samples[j as usize]
    };
    let frames = samples.len() / HOP_LENGTH + 1;
    let offset = (N_FFT - WIN_LENGTH) / 2;
    let window = (0..WIN_LENGTH)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / WIN_LENGTH as f32).cos())
        .collect::<Vec<_>>();
    let filters = htk_filterbank();
    let fft = plan_forward_f32(N_FFT);
    let mut buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
    let mut mels = vec![0f32; frames * N_MELS];
    for frame in 0..frames {
        let start = (frame * HOP_LENGTH) as isize - pad as isize;
        for (k, slot) in buffer.iter_mut().enumerate() {
            let w = if (offset..offset + WIN_LENGTH).contains(&k) {
                window[k - offset]
            } else {
                0.0
            };
            *slot = Complex32::new(reflect(start + k as isize) * w, 0.0);
        }
        fft.process(&mut buffer);
        let row = &mut mels[frame * N_MELS..(frame + 1) * N_MELS];
        for (bin, filter) in buffer.iter().take(N_FFT / 2 + 1).zip(&filters) {
            let power = bin.norm_sqr();
            for (mel, weight) in row.iter_mut().zip(filter) {
                *mel += power * weight;
            }
        }
    }
    for mel in &mut mels {
        *mel = mel.max(LOG_FLOOR).log10();
    }
    let peak = mels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for mel in &mut mels {
        *mel = mel.max(peak - DYNAMIC_RANGE) / LOG_SCALE + 1.0;
    }
    let kept = frames / STACKED_FRAMES * STACKED_FRAMES;
    mels.truncate(kept * N_MELS);
    (mels, kept / STACKED_FRAMES)
}

#[cfg(test)]
mod tests {
    use super::{audio_token_count, log_mel_features};

    #[test]
    fn frames_pair_up_and_count_as_transformers_does() {
        // 1 s: 101 mel frames, 50 encoder frames, 4 windows of 15, 3 tokens each
        let samples = vec![0.1f32; 16_000];
        let (features, frames) = log_mel_features(&samples);
        assert_eq!(frames, 50);
        assert_eq!(features.len(), 50 * 160);
        assert_eq!(audio_token_count(samples.len(), 15, 5), 12);
    }
}
