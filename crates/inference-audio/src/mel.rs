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
// input frames per resampler call when streaming
const STREAM_CHUNK: usize = 1024;

/// Band-limited sinc resampling of mono `samples`; returns them unchanged at the same rate.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    let mut resampler = rubato::SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        MAX_RATIO_RELATIVE,
        sinc_parameters(),
        samples.len(),
        1,
    )?;
    Ok(resampler.process(&[samples], None)?.swap_remove(0))
}

/// [`resample`] over audio that arrives in pieces: rubato's sinc resampler fed fixed chunks, its state carried
/// across pushes, so a stream at another rate reaches a model at its own.
pub struct StreamResampler {
    // rubato's sinc tables are not Sync; a resampler is used by one stream at a time
    inner: Option<std::sync::Mutex<rubato::SincFixedIn<f32>>>,
    pending: Vec<f32>,
    ratio: f64,
    taken: usize,
    given: usize,
}

impl StreamResampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Result<Self> {
        let ratio = to_rate as f64 / from_rate as f64;
        let inner = (from_rate != to_rate)
            .then(|| {
                rubato::SincFixedIn::<f32>::new(
                    ratio,
                    MAX_RATIO_RELATIVE,
                    sinc_parameters(),
                    STREAM_CHUNK,
                    1,
                )
                .map(std::sync::Mutex::new)
            })
            .transpose()?;
        Ok(Self {
            inner,
            pending: Vec::new(),
            ratio,
            taken: 0,
            given: 0,
        })
    }

    /// The resampled audio `samples` complete; the rest waits for the next push or `finish`.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        use rubato::Resampler;
        self.taken += samples.len();
        let Some(inner) = self.inner.as_mut() else {
            self.given += samples.len();
            return Ok(samples.to_vec());
        };
        let inner = inner.get_mut().expect("get_mut takes no lock to poison");
        self.pending.extend_from_slice(samples);
        let mut out = Vec::new();
        while self.pending.len() >= inner.input_frames_next() {
            let n = inner.input_frames_next();
            out.extend(inner.process(&[&self.pending[..n]], None)?.swap_remove(0));
            self.pending.drain(..n);
        }
        Ok(self.deliver(out))
    }

    /// The rest of the audio, its length rounded to the whole stream's at the new rate.
    pub fn finish(&mut self) -> Result<Vec<f32>> {
        use rubato::Resampler;
        let Some(inner) = self.inner.as_mut() else {
            return Ok(Vec::new());
        };
        let inner = inner.get_mut().expect("get_mut takes no lock to poison");
        let mut out = inner
            .process_partial(Some(&[&self.pending]), None)?
            .swap_remove(0);
        self.pending.clear();
        // the input still inside the filter comes out with the silence pushed after it
        out.extend(inner.process_partial::<&[f32]>(None, None)?.swap_remove(0));
        let before = self.given;
        let mut out = self.deliver(out);
        let total = (self.taken as f64 * self.ratio).round() as usize;
        out.truncate(total.saturating_sub(before));
        self.given = before + out.len();
        Ok(out)
    }

    fn deliver(&mut self, out: Vec<f32>) -> Vec<f32> {
        self.given += out.len();
        out
    }
}

fn sinc_parameters() -> rubato::SincInterpolationParameters {
    rubato::SincInterpolationParameters {
        sinc_len: SINC_LEN,
        f_cutoff: SINC_CUTOFF,
        interpolation: rubato::SincInterpolationType::Linear,
        oversampling_factor: SINC_OVERSAMPLING,
        window: rubato::WindowFunction::BlackmanHarris2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // output samples at 16 kHz by which the sinc filter lags its 24 kHz input
    const FILTER_DELAY: f32 = 36.0;

    #[test]
    fn a_streamed_resample_lasts_as_long_as_its_input_and_keeps_its_tone() -> Result<()> {
        let tone: Vec<f32> = (0..24_000 * 3 + 17)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 24_000.0).sin() * 0.5)
            .collect();
        for piece in [7, 480, 4096] {
            let mut stream = StreamResampler::new(24_000, 16_000)?;
            let mut out = Vec::new();
            for chunk in tone.chunks(piece) {
                out.extend(stream.push(chunk)?);
            }
            out.extend(stream.finish()?);
            assert_eq!(
                out.len(),
                (tone.len() as f64 * 2.0 / 3.0).round() as usize,
                "{piece}"
            );
            // past the sinc's warm-up, a 440 Hz tone at 16 kHz still peaks near its amplitude
            let peak = out[4000..40_000].iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!((peak - 0.5).abs() < 0.02, "{piece}: {peak}");
            // the filter delays the output about 2 ms, 36 samples at 16 kHz, as the batch resample does
            let drift = (4000..40_000)
                .map(|i| {
                    let t = (i as f32 - FILTER_DELAY) * 1.5 / 24_000.0;
                    (out[i] - (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5).abs()
                })
                .fold(0f32, f32::max);
            assert!(drift < 0.02, "{piece}: {drift}");
        }
        let mut same = StreamResampler::new(16_000, 16_000)?;
        assert_eq!(same.push(&tone[..100])?, tone[..100]);
        assert!(same.finish()?.is_empty());
        Ok(())
    }

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
