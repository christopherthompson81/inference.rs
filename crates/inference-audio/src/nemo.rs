use anyhow::{Result, bail};

use crate::fft::{Complex64, Fft, plan_forward_f64};
use crate::mel::slaney_filterbank;

// NeMo's guard inside the log, and the one added to each feature's standard deviation
const LOG_GUARD: f64 = 5.960_464_477_539_063e-8;
const NORMALIZE_EPSILON: f64 = 1e-5;

/// NeMo's `AudioToMelSpectrogramPreprocessor` settings, as Parakeet's `processor_config.json` gives them.
#[derive(Debug, Clone, PartialEq)]
pub struct NemoMelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemphasis: Option<f32>,
    /// Per-feature normalisation over the utterance's frames.
    pub normalize: bool,
}

/// Log-mel features as NeMo computes them: preemphasis, a centred zero-padded STFT under a symmetric Hann window,
/// Slaney mel power, `ln(x + 2^-24)`, then optionally per-feature mean and variance normalisation.
pub struct NemoMel {
    config: NemoMelConfig,
    filters: Vec<Vec<f64>>,
    window: Vec<f64>,
    fft: std::sync::Arc<dyn Fft<f64>>,
}

impl NemoMel {
    pub fn new(config: NemoMelConfig) -> Self {
        let filters = slaney_filterbank(config.sample_rate, config.n_fft, config.n_mels)
            .into_iter()
            .map(|row| row.into_iter().map(f64::from).collect())
            .collect();
        // torch.hann_window(periodic=False) centred inside the FFT frame, as torch.stft pads a short window
        let offset = (config.n_fft - config.win_length) / 2;
        let last = (config.win_length - 1) as f64;
        let mut window = vec![0f64; config.n_fft];
        for n in 0..config.win_length {
            window[offset + n] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / last).cos();
        }
        let fft = plan_forward_f64(config.n_fft);
        Self {
            config,
            filters,
            window,
            fft,
        }
    }

    // one STFT frame of `n_fft` padded samples to its `n_mels` log-mel energies
    fn frame(&self, frame: &[f64], out: &mut [f64], scratch: &mut FrameScratch) {
        for ((b, &x), &w) in scratch.buffer.iter_mut().zip(frame).zip(&self.window) {
            *b = Complex64::new(x * w, 0.0);
        }
        self.fft.process(&mut scratch.buffer);
        for (p, b) in scratch.power.iter_mut().zip(&scratch.buffer) {
            *p = b.norm_sqr();
        }
        for (o, filter) in out.iter_mut().zip(&self.filters) {
            let energy: f64 = filter.iter().zip(&scratch.power).map(|(f, p)| f * p).sum();
            *o = (energy + LOG_GUARD).ln();
        }
    }

    pub fn config(&self) -> &NemoMelConfig {
        &self.config
    }

    /// `(frames, n_mels)` row-major features over the utterance's valid frames, and the frame count.
    pub fn features(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let NemoMelConfig {
            n_fft,
            hop_length,
            n_mels,
            ..
        } = self.config;
        let half = n_fft / 2;
        // torch.stft(center=True) pads n_fft/2 zeros each side; NeMo keeps (len + 2 * (n_fft/2) - n_fft) / hop frames
        let frames = (samples.len() + 2 * half - n_fft) / hop_length;
        if frames < 2 {
            bail!(
                "audio is too short: {} samples give {frames} feature frames",
                samples.len()
            )
        }
        let mut padded = vec![0f64; samples.len() + 2 * half];
        let coefficient = f64::from(self.config.preemphasis.unwrap_or(0.0));
        let mut previous = 0f64;
        for (i, &s) in samples.iter().enumerate() {
            let s = f64::from(s);
            padded[half + i] = if i == 0 {
                s
            } else {
                s - coefficient * previous
            };
            previous = s;
        }
        let mut features = vec![0f64; frames * n_mels];
        let mut scratch = FrameScratch::new(n_fft);
        for t in 0..frames {
            self.frame(
                &padded[t * hop_length..t * hop_length + n_fft],
                &mut features[t * n_mels..(t + 1) * n_mels],
                &mut scratch,
            );
        }
        if self.config.normalize {
            for m in 0..n_mels {
                let column = (0..frames).map(|t| features[t * n_mels + m]);
                let mean = column.clone().sum::<f64>() / frames as f64;
                let variance =
                    column.map(|v| (v - mean).powi(2)).sum::<f64>() / (frames - 1) as f64;
                let scale = 1.0 / (variance.sqrt() + NORMALIZE_EPSILON);
                for t in 0..frames {
                    let v = &mut features[t * n_mels + m];
                    *v = (*v - mean) * scale;
                }
            }
        }
        Ok((features.into_iter().map(|v| v as f32).collect(), frames))
    }
}

struct FrameScratch {
    buffer: Vec<Complex64>,
    power: Vec<f64>,
}

impl FrameScratch {
    fn new(n_fft: usize) -> Self {
        Self {
            buffer: vec![Complex64::default(); n_fft],
            power: vec![0f64; n_fft / 2 + 1],
        }
    }
}

/// [`NemoMel::features`] over audio that arrives in pieces: each frame is emitted once its samples are all in, and
/// the frames together equal the offline features of the whole audio. Normalisation needs the whole utterance, so
/// a normalising config is refused.
pub struct NemoMelStream {
    // preemphasised samples with the leading zero pad, from padded index `base` on
    padded: Vec<f64>,
    base: usize,
    previous: Option<f64>,
    samples: usize,
    next_frame: usize,
    scratch: FrameScratch,
}

impl NemoMel {
    pub fn stream(&self) -> Result<NemoMelStream> {
        if self.config.normalize {
            bail!("normalised features need the whole utterance; they cannot be streamed");
        }
        Ok(NemoMelStream {
            padded: vec![0f64; self.config.n_fft / 2],
            base: 0,
            previous: None,
            samples: 0,
            next_frame: 0,
            scratch: FrameScratch::new(self.config.n_fft),
        })
    }
}

impl NemoMelStream {
    /// Appends `samples` and returns the `(frames, n_mels)` features they complete, row-major; `mel` is the
    /// extractor the stream came from.
    pub fn push(&mut self, mel: &NemoMel, samples: &[f32]) -> Vec<f32> {
        let coefficient = f64::from(mel.config.preemphasis.unwrap_or(0.0));
        for &s in samples {
            let s = f64::from(s);
            self.padded.push(match self.previous {
                None => s,
                Some(previous) => s - coefficient * previous,
            });
            self.previous = Some(s);
        }
        self.samples += samples.len();
        let NemoMelConfig {
            n_fft, hop_length, ..
        } = mel.config;
        let available = self.base + self.padded.len();
        let ready = (available + 1).saturating_sub(n_fft).div_ceil(hop_length);
        // never past the frames the offline count keeps for the audio so far
        let kept = (self.samples + 2 * (n_fft / 2)).saturating_sub(n_fft) / hop_length;
        self.emit(mel, ready.min(kept))
    }

    /// The features of the frames the end of the audio completes; the stream is done after this.
    pub fn finish(&mut self, mel: &NemoMel) -> Vec<f32> {
        let NemoMelConfig {
            n_fft, hop_length, ..
        } = mel.config;
        self.padded.extend(std::iter::repeat_n(0f64, n_fft / 2));
        let total = (self.samples + 2 * (n_fft / 2)).saturating_sub(n_fft) / hop_length;
        self.emit(mel, total)
    }

    pub fn frames(&self) -> usize {
        self.next_frame
    }

    fn emit(&mut self, mel: &NemoMel, until: usize) -> Vec<f32> {
        let NemoMelConfig {
            n_fft,
            hop_length,
            n_mels,
            ..
        } = mel.config;
        let count = until.saturating_sub(self.next_frame);
        let mut features = vec![0f64; count * n_mels];
        for (i, out) in features.chunks_mut(n_mels).enumerate() {
            let start = (self.next_frame + i) * hop_length - self.base;
            mel.frame(&self.padded[start..start + n_fft], out, &mut self.scratch);
        }
        self.next_frame += count;
        // samples before the next frame's start are never read again
        let keep_from = (self.next_frame * hop_length - self.base).min(self.padded.len());
        self.padded.drain(..keep_from);
        self.base += keep_from;
        features.into_iter().map(|v| v as f32).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parakeet() -> NemoMel {
        NemoMel::new(NemoMelConfig {
            sample_rate: 16000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            preemphasis: Some(0.97),
            normalize: true,
        })
    }

    #[test]
    fn frame_count_follows_the_hop_and_short_audio_is_refused() -> Result<()> {
        let mel = parakeet();
        let (features, frames) = mel.features(&vec![0.1; 16000])?;
        assert_eq!(frames, 100);
        assert_eq!(features.len(), 100 * 128);
        assert!(mel.features(&[0.0; 200]).is_err());
        Ok(())
    }

    #[test]
    fn streamed_features_equal_the_offline_ones_in_any_pieces() -> Result<()> {
        let mel = NemoMel::new(NemoMelConfig {
            normalize: false,
            ..parakeet().config().clone()
        });
        let samples: Vec<f32> = (0..16_337)
            .map(|i| (i as f32 * 0.0137).sin() * 0.3 + (i as f32 * 0.31).sin() * 0.05)
            .collect();
        let (offline, frames) = mel.features(&samples)?;
        for piece in [1, 159, 160, 1000, 16_337] {
            let mut stream = mel.stream()?;
            let mut streamed = Vec::new();
            for chunk in samples.chunks(piece) {
                streamed.extend(stream.push(&mel, chunk));
            }
            streamed.extend(stream.finish(&mel));
            assert_eq!(stream.frames(), frames, "{piece}");
            assert_eq!(streamed, offline, "{piece}");
        }
        assert!(parakeet().stream().is_err());
        Ok(())
    }

    #[test]
    fn normalised_features_are_zero_mean_unit_variance_per_bin() -> Result<()> {
        let samples: Vec<f32> = (0..24000)
            .map(|i| (i as f32 * 0.0137).sin() * 0.3 + (i as f32 * 0.31).sin() * 0.05)
            .collect();
        let (features, frames) = parakeet().features(&samples)?;
        for m in [0, 40, 127] {
            let column: Vec<f64> = (0..frames).map(|t| features[t * 128 + m] as f64).collect();
            let mean = column.iter().sum::<f64>() / frames as f64;
            let var = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (frames - 1) as f64;
            assert!(mean.abs() < 1e-4, "{m}: {mean}");
            assert!((var - 1.0).abs() < 1e-2, "{m}: {var}");
        }
        Ok(())
    }
}
