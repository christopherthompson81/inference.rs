//! NVIDIA's Parakeet speech recognisers: a FastConformer encoder under a CTC, RNN-T or TDT head, loaded from the
//! transformers layout (`config.json`, `processor_config.json`, `tokenizer.json`, `model.safetensors`).

mod config;
mod decoder;
mod encoder;

use std::path::{Path, PathBuf};

use inference_audio::nemo::{NemoMel, NemoMelConfig};
use inference_tensor::nn::VarBuilder;
use inference_tensor::{DType, Device, Error, Result, Tensor};
use tokenizers::Tokenizer;

use crate::{TimedText, Transcription};
pub use config::{HeadKind, MODEL_TYPES, ParakeetConfig, ProcessorConfig};
pub use decoder::Emission;
use decoder::Head;
use encoder::Encoder;

// NeMo's bound for full attention on these models, which attend over the whole recording at once
const MAX_SECONDS: f64 = 24.0 * 60.0;
// tokens transformers' processor pins to the end of the token before them, as NeMo does for TDT
const ATTACHED_PUNCTUATION: [&str; 11] = [
    "?", "'", "\u{a1}", "\u{bf}", "-", ":", ",", "%", "/", ".", "!",
];

/// The files of a Parakeet checkpoint.
#[derive(Debug, Clone)]
pub struct ParakeetFiles {
    pub config: PathBuf,
    pub processor_config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

pub struct Parakeet {
    mel: NemoMel,
    encoder: Encoder,
    head: Head,
    tokenizer: Tokenizer,
    kind: HeadKind,
    frame_seconds: f64,
    device: Device,
    dtype: DType,
}

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).map_err(|e| msg(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| msg(format!("{}: {e}", path.display())))
}

impl Parakeet {
    pub fn load(files: &ParakeetFiles, device: &Device, dtype: DType) -> Result<Self> {
        let config: ParakeetConfig = read_json(&files.config)?;
        let processor: ProcessorConfig = read_json(&files.processor_config)?;
        // SAFETY: the weights are read-only for the model's life; a file changed underneath it is undefined
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, device)? };
        Self::new(
            config,
            processor,
            Tokenizer::from_file(&files.tokenizer).map_err(msg)?,
            vb,
        )
    }

    pub fn new(
        config: ParakeetConfig,
        processor: ProcessorConfig,
        tokenizer: Tokenizer,
        vb: VarBuilder,
    ) -> Result<Self> {
        let kind = config.head().ok_or_else(|| {
            msg(format!(
                "`{}` is not a Parakeet model_type",
                config.model_type
            ))
        })?;
        let enc = &config.encoder_config;
        let fe = &processor.feature_extractor;
        if fe.feature_size != enc.num_mel_bins {
            return Err(msg(format!(
                "the features have {} mel bins but the encoder takes {}",
                fe.feature_size, enc.num_mel_bins
            )));
        }
        let mel = NemoMel::new(NemoMelConfig {
            sample_rate: fe.sampling_rate,
            n_fft: fe.n_fft,
            win_length: fe.win_length,
            hop_length: fe.hop_length,
            n_mels: fe.feature_size,
            preemphasis: fe.preemphasis,
            normalize: true,
        });
        let frame_seconds =
            (fe.hop_length * enc.subsampling_factor) as f64 / f64::from(fe.sampling_rate);
        Ok(Self {
            encoder: Encoder::new(enc, vb.pp("encoder"))?,
            head: Head::new(&config, kind, enc.hidden_size, vb.clone())?,
            mel,
            tokenizer,
            kind,
            frame_seconds,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.mel.config().sample_rate
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Transcribes mono `pcm` at `sample_rate`, resampling to the model's rate first.
    pub fn transcribe(&self, pcm: &[f32], sample_rate: u32) -> Result<Transcription> {
        // checked before resampling, which would copy hours of audio the model then refuses
        let seconds = pcm.len() as f64 / f64::from(sample_rate);
        if seconds > MAX_SECONDS {
            return Err(msg(format!(
                "the audio is {seconds:.0} s; Parakeet's full attention takes at most {MAX_SECONDS:.0} s"
            )));
        }
        let pcm =
            inference_audio::mel::resample(pcm, sample_rate, self.sample_rate()).map_err(msg)?;
        let duration = pcm.len() as f64 / f64::from(self.sample_rate());
        let encoded = self.encode(&pcm)?;
        let emissions = self.head.decode(&encoded)?;
        self.text(&emissions, duration)
    }

    /// `(frames, mels)` row-major log-mel features of mono `pcm` at the model's rate, and the frame count.
    pub fn features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        self.mel.features(pcm).map_err(msg)
    }

    /// `(1, frames, hidden)` encoder output for mono `pcm` at the model's rate.
    pub fn encode(&self, pcm: &[f32]) -> Result<Tensor> {
        let (features, frames) = self.features(pcm)?;
        let n_mels = self.mel.config().n_mels;
        let features =
            Tensor::from_vec(features, (1, frames, n_mels), &self.device)?.to_dtype(self.dtype)?;
        self.encoder.forward(&features)
    }

    /// Token ids and their frames, before detokenisation.
    pub fn emissions(&self, encoded: &Tensor) -> Result<Vec<Emission>> {
        self.head.decode(encoded)
    }

    fn text(&self, emissions: &[Emission], duration: f64) -> Result<Transcription> {
        let ids: Vec<u32> = emissions.iter().map(|e| e.token).collect();
        let text = self.tokenizer.decode(&ids, true).map_err(msg)?;
        let mut stream = self.tokenizer.decode_stream(true);
        let mut tokens: Vec<TimedText> = Vec::new();
        for e in emissions {
            let Some(piece) = stream.step(e.token).map_err(msg)? else {
                continue;
            };
            // a span predicted past the last frame is clamped to the audio
            let (mut start, mut end) = (
                (e.frame as f64 * self.frame_seconds).min(duration),
                ((e.frame + e.frames) as f64 * self.frame_seconds).min(duration),
            );
            if self.kind == HeadKind::Tdt
                && ATTACHED_PUNCTUATION.contains(&piece.as_str())
                && let Some(previous) = tokens.last()
            {
                (start, end) = (previous.end, previous.end);
            }
            tokens.push(TimedText {
                text: piece,
                start,
                end,
            });
        }
        Ok(Transcription {
            words: words(&tokens),
            text: text.trim().to_string(),
            tokens,
            duration,
        })
    }
}

// a piece opening with whitespace starts a word; any other joins the word before it
fn words(tokens: &[TimedText]) -> Vec<TimedText> {
    let mut words: Vec<TimedText> = Vec::new();
    for t in tokens {
        let starts_word = t.text.starts_with(char::is_whitespace) || words.is_empty();
        let piece = t.text.trim();
        if piece.is_empty() {
            continue;
        }
        match words.last_mut() {
            Some(word) if !starts_word => {
                word.text.push_str(piece);
                word.end = word.end.max(t.end);
            }
            _ => words.push(TimedText {
                text: piece.to_string(),
                start: t.start,
                end: t.end,
            }),
        }
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(text: &str, start: f64, end: f64) -> TimedText {
        TimedText {
            text: text.into(),
            start,
            end,
        }
    }

    #[test]
    fn pieces_join_into_words_with_punctuation_attached() {
        let tokens = [
            piece("W", 0.32, 0.4),
            piece("ell", 0.4, 0.56),
            piece(",", 0.56, 0.56),
            piece(" I", 0.64, 0.8),
            piece(" don", 0.8, 0.88),
            piece("'", 0.88, 0.88),
            piece("t", 0.96, 1.04),
        ];
        let w = words(&tokens);
        assert_eq!(
            w,
            [
                piece("Well,", 0.32, 0.56),
                piece("I", 0.64, 0.8),
                piece("don't", 0.8, 1.04)
            ]
        );
    }
}
