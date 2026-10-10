use anyhow::Result;

const MIN_LOG_HERTZ: f32 = 1000.0;
const MIN_LOG_MEL: f32 = 15.0;
// 27 / ln(6.4) and its inverse, the Slaney scale's log steps above 1 kHz
const MELS_PER_LOG_HERTZ: f32 = 27.0 / 1.856_298;
const LOG_HERTZ_PER_MEL: f32 = 1.856_298 / 27.0;

/// Slaney mel scale: Hz to mel.
pub fn hertz_to_mel(freq: f32) -> f32 {
    if freq >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * MELS_PER_LOG_HERTZ
    } else {
        3.0 * freq / 200.0
    }
}

/// Slaney mel scale: mel to Hz.
pub fn mel_to_hertz(mel: f32) -> f32 {
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (LOG_HERTZ_PER_MEL * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// `[n_mels][n_fft / 2 + 1]` Slaney filters over 0 to Nyquist with Slaney area normalization, as librosa's
/// `filters.mel(norm="slaney")` and `mistral_common`'s `mel_filter_bank` build them.
pub fn slaney_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * nyquist / (n_freqs - 1) as f32)
        .collect();
    let (mel_min, mel_max) = (hertz_to_mel(0.0), hertz_to_mel(nyquist));
    let filter_freqs: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hertz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let filter_diff: Vec<f32> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();
    (0..n_mels)
        .map(|m| {
            let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
            fft_freqs
                .iter()
                .map(|&f| {
                    let rising = (f - filter_freqs[m]) / filter_diff[m];
                    let falling = (filter_freqs[m + 2] - f) / filter_diff[m + 1];
                    0f32.max(rising.min(falling)) * enorm
                })
                .collect()
        })
        .collect()
}

// windowed-sinc settings every model's audio path has used: a long filter, cut just under Nyquist
const SINC_LEN: usize = 256;
const SINC_CUTOFF: f32 = 0.95;
const SINC_OVERSAMPLING: usize = 256;
const MAX_RATIO_RELATIVE: f64 = 2.0;

/// Band-limited sinc resampling of mono `samples`; returns them unchanged at the same rate.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    let sinc = rubato::SincInterpolationParameters {
        sinc_len: SINC_LEN,
        f_cutoff: SINC_CUTOFF,
        interpolation: rubato::SincInterpolationType::Linear,
        oversampling_factor: SINC_OVERSAMPLING,
        window: rubato::WindowFunction::BlackmanHarris2,
    };
    let mut resampler = rubato::SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        MAX_RATIO_RELATIVE,
        sinc,
        samples.len(),
        1,
    )?;
    Ok(resampler.process(&[samples], None)?.swap_remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_scale_round_trips_and_is_linear_below_one_kilohertz() {
        for hz in [0.0, 200.0, 999.0, 1000.0, 4321.0, 8000.0] {
            assert!((mel_to_hertz(hertz_to_mel(hz)) - hz).abs() < 1e-2, "{hz}");
        }
        assert_eq!(hertz_to_mel(600.0), 9.0);
    }

    // librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney"): filter 0's one tap, filter 127's peak
    #[test]
    fn filterbank_matches_librosa() {
        let fb = slaney_filterbank(16000, 512, 128);
        assert_eq!((fb.len(), fb[0].len()), (128, 257));
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * b.abs();
        assert!(close(fb[0][1], 0.028_377_542), "{}", fb[0][1]);
        assert_eq!(fb[0].iter().filter(|&&v| v > 0.0).count(), 1);
        let peak = fb[127].iter().cloned().fold(0f32, f32::max);
        assert!(close(peak, 0.005_223_188_5), "{peak}");
    }
}
