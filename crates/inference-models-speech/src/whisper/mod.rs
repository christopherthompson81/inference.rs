//! OpenAI's Whisper speech recognizer: 30 s windows of log-mel through the shared Whisper encoder, a text decoder
//! decoding greedily with timestamp tokens, long recordings window by window from the last timestamp, and word
//! times from the cross-attention of the alignment heads.

mod config;
mod decode;
mod languages;
mod model;
mod timing;

use std::path::{Path, PathBuf};

use inference_audio::whisper::{LogMelMax, WhisperLogMel};
use inference_nn::audio::whisper::WhisperEncoder;
use inference_quant::{ShardedSafeTensors, ShardedVarBuilder};
use inference_tensor::{DType, Device, Error, Result, Tensor};
use tokenizers::Tokenizer;

pub use config::{WhisperConfig, WhisperGenerationConfig};
use decode::{
    SECONDS_PER_TIMESTAMP, SpecialTokens, WindowSegment, apply_rules, argmax, split_segments,
};
use model::Decoder;
use timing::WordTiming;

use crate::{TimedText, Transcription, TranscriptionOptions};

pub const MODEL_TYPE: &str = "whisper";
pub const SAMPLE_RATE: u32 = 16_000;
const ENGLISH: &str = "en";
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const WINDOW_SECONDS: usize = 30;
const WINDOW_SAMPLES: usize = WINDOW_SECONDS * SAMPLE_RATE as usize;
const WINDOW_FRAMES: usize = WINDOW_SAMPLES / HOP_LENGTH;
// mel frames per encoder position (the second conv's stride)
const FRAMES_PER_POSITION: usize = 2;
// OpenAI's `sample_len`: half the decoder's positions for one window's text
const MAX_WINDOW_TOKENS: usize = 224;
// languages written without spaces between words: word times go per character
const UNSPACED_LANGUAGES: [&str; 6] = ["zh", "ja", "th", "lo", "my", "yue"];
const FRAMES_PER_SECOND: f64 = SAMPLE_RATE as f64 / HOP_LENGTH as f64;

fn msg(error: impl std::fmt::Display) -> Error {
    Error::Msg(error.to_string())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_str(&std::fs::read_to_string(path).map_err(msg)?).map_err(msg)
}

/// A transformers Whisper checkpoint's files.
#[derive(Clone, Debug)]
pub struct WhisperFiles {
    pub config: PathBuf,
    pub generation_config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

/// Whether a `config.json` is a Whisper recognizer's.
pub fn is_whisper_config(config: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(config)
        .ok()
        .and_then(|value| value.get("model_type")?.as_str().map(|t| t == MODEL_TYPE))
        .unwrap_or(false)
}

pub struct Whisper {
    encoder: WhisperEncoder,
    decoder: Decoder,
    tokenizer: Tokenizer,
    special: SpecialTokens,
    generation: WhisperGenerationConfig,
    log_mel: WhisperLogMel,
    device: Device,
    dtype: DType,
}

impl Whisper {
    pub fn load(files: &WhisperFiles, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: the weights are read-only for the model's life; a file changed underneath it is undefined
        let vb = unsafe {
            ShardedSafeTensors::sharded(
                &files.weights,
                dtype,
                device,
                None,
                std::sync::Arc::new(|_| true),
            )?
        };
        Self::new(
            &read_json(&files.config)?,
            read_json(&files.generation_config)?,
            Tokenizer::from_file(&files.tokenizer).map_err(msg)?,
            vb,
        )
    }

    pub fn new(
        config: &WhisperConfig,
        generation: WhisperGenerationConfig,
        tokenizer: Tokenizer,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let (device, dtype) = (vb.device().clone(), vb.dtype());
        let vb = vb.pp("model");
        Ok(Self {
            encoder: WhisperEncoder::new(&config.encoder(), vb.pp("encoder"))?,
            decoder: Decoder::new(config, vb.pp("decoder"))?,
            tokenizer,
            special: SpecialTokens::new(&generation),
            log_mel: WhisperLogMel {
                sample_rate: SAMPLE_RATE,
                n_fft: N_FFT,
                hop_length: HOP_LENGTH,
                n_mels: config.num_mel_bins,
            },
            generation,
            device,
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn multilingual(&self) -> bool {
        !self.generation.lang_to_id.is_empty()
    }

    // a request's language, an ISO-639-1 code or Whisper's name for it, as Whisper's code
    fn language_code(&self, language: &str) -> Result<String> {
        let wanted = language.to_lowercase();
        let code = languages::LANGUAGES
            .iter()
            .find(|(code, name)| *code == wanted || *name == wanted)
            .map(|(code, _)| (*code).to_string())
            .ok_or_else(|| msg(format!("Whisper does not transcribe language `{language}`")))?;
        // an English-only checkpoint has no language tokens and transcribes English only
        let known = if self.multilingual() {
            self.generation
                .lang_to_id
                .contains_key(&format!("<|{code}|>"))
        } else {
            code == ENGLISH
        };
        if !known {
            return Err(msg(format!(
                "this Whisper checkpoint does not transcribe language `{language}`"
            )));
        }
        Ok(code)
    }

    // OpenAI's `detect_language`: one step from the start token, the likeliest language token
    fn detect_language(&self, encoded: &Tensor) -> Result<String> {
        let mut cache = self.decoder.cache(encoded)?;
        let out = self
            .decoder
            .forward(&[self.special.start], &mut cache, false)?;
        let logits = Decoder::last_logits(&out)?;
        let (token, _) = self
            .generation
            .lang_to_id
            .iter()
            .map(|(token, &id)| (token, logits[id as usize]))
            .fold((None, f32::NEG_INFINITY), |best, (token, logit)| {
                if logit > best.1 {
                    (Some(token), logit)
                } else {
                    best
                }
            });
        token
            .map(|token| {
                token
                    .trim_start_matches("<|")
                    .trim_end_matches("|>")
                    .to_string()
            })
            .ok_or_else(|| msg("this Whisper checkpoint has no language tokens"))
    }

    // the start of every window's prompt: the start token, then the language and the task for a multilingual model
    fn start_sequence(&self, language: Option<&str>) -> Vec<u32> {
        let mut tokens = vec![self.special.start];
        if let Some(code) = language.filter(|_| self.multilingual())
            && let Some(&id) = self.generation.lang_to_id.get(&format!("<|{code}|>"))
        {
            tokens.push(id);
            tokens.extend(self.special.transcribe);
        }
        tokens
    }

    fn decode_window(&self, encoded: &Tensor, prompt: &[u32]) -> Result<Vec<u32>> {
        let mut cache = self.decoder.cache(encoded)?;
        let mut out = self.decoder.forward(prompt, &mut cache, false)?;
        let mut generated = Vec::new();
        let budget = MAX_WINDOW_TOKENS.min(self.decoder.max_positions() - prompt.len());
        while generated.len() < budget {
            let mut logits = Decoder::last_logits(&out)?;
            apply_rules(&self.special, &mut logits, &generated, true);
            let token = argmax(&logits);
            if token == self.special.eot {
                break;
            }
            generated.push(token);
            if cache.len() >= self.decoder.max_positions() {
                break;
            }
            out = self.decoder.forward(&[token], &mut cache, false)?;
        }
        Ok(generated)
    }

    fn token_text(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens, false).map_err(msg)
    }

    // OpenAI's `split_to_word_tokens`: words begin at a space or a punctuation mark, or, unspaced, at each character
    fn split_words(&self, tokens: &[u32], unspaced: bool) -> Result<Vec<(String, usize)>> {
        let mut words: Vec<(String, usize)> = Vec::new();
        let mut pending: Vec<u32> = Vec::new();
        for &token in tokens {
            pending.push(token);
            let text = self.token_text(&pending)?;
            // a token cutting a multi-byte character waits for the rest of it
            if text.contains('\u{fffd}') {
                continue;
            }
            let count = pending.len();
            pending.clear();
            let punctuation =
                !text.trim().is_empty() && text.trim().chars().all(|c| c.is_ascii_punctuation());
            match words.last_mut() {
                Some(last) if !unspaced && !text.starts_with(' ') && !punctuation => {
                    last.0.push_str(&text);
                    last.1 += count;
                }
                _ => words.push((text, count)),
            }
        }
        if !pending.is_empty() {
            words.push((self.token_text(&pending)?, pending.len()));
        }
        Ok(words)
    }

    // the window's alignment words (punctuation apart) and their times from its start, by DTW over the alignment
    // heads' cross-attention; without alignment heads each word spans its segment
    fn align_words(
        &self,
        encoded: &Tensor,
        start: &[u32],
        segments: &[WindowSegment],
        content_frames: usize,
        unspaced: bool,
    ) -> Result<Vec<WordTiming>> {
        let text: Vec<u32> = segments
            .iter()
            .flat_map(|s| s.tokens.iter().copied())
            .collect();
        let words = self.split_words(&text, unspaced)?;
        let heads = &self.generation.alignment_heads;
        let positions = (content_frames / FRAMES_PER_POSITION).max(1);
        let times = if heads.is_empty() || text.is_empty() {
            None
        } else {
            let mut tokens = start.to_vec();
            tokens.push(self.special.no_timestamps);
            tokens.extend(&text);
            tokens.push(self.special.eot);
            let mut cache = self.decoder.cache(encoded)?;
            let out = self.decoder.forward(&tokens, &mut cache, true)?;
            let scores = heads
                .iter()
                .map(|head| {
                    out.cross_scores[head[0]]
                        .get(head[1])?
                        .narrow(1, 0, positions)?
                        .flatten_all()?
                        .to_vec1::<f32>()
                })
                .collect::<Result<Vec<_>>>()?;
            // standardized over every row, as OpenAI does, then the rows from no-timestamps to the last text token
            let matrix = timing::alignment_matrix(&scores, tokens.len(), positions);
            let (first, rows) = (start.len(), text.len() + 1);
            let matrix = matrix[first * positions..(first + rows) * positions].to_vec();
            let counts: Vec<usize> = words.iter().map(|(_, count)| *count).collect();
            Some(timing::word_times(&matrix, rows, positions, &counts))
        };
        let mut token_index = 0;
        Ok(words
            .into_iter()
            .enumerate()
            .map(|(i, (word, tokens))| {
                let (start, end) = match &times {
                    Some(times) => times[i],
                    None => {
                        let segment = segment_of(segments, token_index);
                        (segment.start, segment.end)
                    }
                };
                token_index += tokens;
                WordTiming {
                    word,
                    tokens,
                    start,
                    end,
                }
            })
            .collect())
    }

    pub fn transcribe(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        options: &TranscriptionOptions,
    ) -> Result<Transcription> {
        let pcm = if sample_rate == SAMPLE_RATE {
            pcm.to_vec()
        } else {
            inference_audio::mel::resample(pcm, sample_rate, SAMPLE_RATE).map_err(msg)?
        };
        let duration = pcm.len() as f64 / f64::from(SAMPLE_RATE);
        let content_frames = pcm.len() / HOP_LENGTH;
        // OpenAI pads a window of silence after the audio before its log-mel, whose floor is the whole clip's peak
        let mut padded = pcm;
        padded.resize(padded.len() + WINDOW_SAMPLES, 0.0);
        let frames = self.log_mel.frames(&padded, LogMelMax::Clip);
        let n_mels = self.log_mel.n_mels;
        let mut language = options
            .language
            .as_deref()
            .map(|l| self.language_code(l))
            .transpose()?;
        if !self.multilingual() {
            language.get_or_insert_with(|| ENGLISH.to_string());
        }
        let mut words = Vec::new();
        let mut text = String::new();
        let mut seek = 0usize;
        let mut last_speech = 0.0;
        while seek < content_frames {
            let window_frames = WINDOW_FRAMES.min(content_frames - seek);
            let mut mel = vec![0.0f32; n_mels * WINDOW_FRAMES];
            for (t, frame) in frames[seek..seek + WINDOW_FRAMES].iter().enumerate() {
                for (m, &value) in frame.iter().enumerate() {
                    mel[m * WINDOW_FRAMES + t] = value;
                }
            }
            let mel = Tensor::from_vec(mel, (1, n_mels, WINDOW_FRAMES), &self.device)?
                .to_dtype(self.dtype)?;
            let encoded = self.encoder.forward(&mel)?;
            if language.is_none() && self.multilingual() {
                language = Some(self.detect_language(&encoded)?);
            }
            let start = self.start_sequence(language.as_deref());
            let generated = self.decode_window(&encoded, &start)?;
            let window_seconds = window_frames as f64 * HOP_LENGTH as f64 / f64::from(SAMPLE_RATE);
            let (segments, consumed) = split_segments(&self.special, &generated, window_seconds);
            let offset = seek as f64 * HOP_LENGTH as f64 / f64::from(SAMPLE_RATE);
            let unspaced = language
                .as_deref()
                .is_some_and(|l| UNSPACED_LANGUAGES.contains(&l));
            for segment in &segments {
                text.push_str(&self.token_text(&segment.tokens)?);
            }
            let alignment =
                self.align_words(&encoded, &start, &segments, window_frames, unspaced)?;
            let placed = timing::place_words(alignment, &segments, offset, last_speech);
            let previous_seek = seek;
            seek += match consumed {
                Some(steps) if steps > 0 => {
                    (f64::from(steps) * SECONDS_PER_TIMESTAMP * FRAMES_PER_SECOND).round() as usize
                }
                _ => window_frames,
            };
            if let Some(last) = placed.last() {
                last_speech = last.end;
                // OpenAI's word-timestamp seek: a window cut at a timestamp pair resumes after its last word
                let shift = ((last.end - offset) * FRAMES_PER_SECOND).round();
                if consumed.is_some() && shift > 0.0 {
                    seek = previous_seek + shift as usize;
                }
            }
            words.extend(placed.into_iter().map(|word| TimedText {
                text: word.text.trim().to_string(),
                start: word.start.min(duration),
                end: word.end.min(duration),
            }));
        }
        words.retain(|word| !word.text.is_empty());
        Ok(Transcription {
            text: text.trim().to_string(),
            tokens: Vec::new(),
            words,
            duration,
            language: language.map(|code| {
                languages::LANGUAGES
                    .iter()
                    .find(|(known, _)| *known == code)
                    .map_or(code.clone(), |(_, name)| (*name).to_string())
            }),
        })
    }
}

fn segment_of(segments: &[WindowSegment], token_index: usize) -> &WindowSegment {
    let mut seen = 0;
    for segment in segments {
        seen += segment.tokens.len();
        if token_index < seen {
            return segment;
        }
    }
    &segments[segments.len() - 1]
}
