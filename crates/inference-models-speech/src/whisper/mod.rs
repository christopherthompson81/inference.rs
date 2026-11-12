//! OpenAI's Whisper speech recognizer: 30 s windows of log-mel through the shared Whisper encoder, a text decoder
//! decoding greedily with timestamp tokens, long recordings window by window from the last timestamp, and word
//! times from the cross-attention of the alignment heads.

mod config;
mod decode;
mod languages;
mod timing;

use std::path::{Path, PathBuf};

use inference_audio::whisper::{LogMelMax, WhisperLogMel};
use inference_nn::audio::whisper::WhisperEncoder;
use inference_quant::{ShardedSafeTensors, ShardedVarBuilder};
use inference_tensor::{DType, Device, Error, Result, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokenizers::Tokenizer;

use crate::text_decoder::{Decoder, DecoderCache};
pub use config::{WhisperConfig, WhisperGenerationConfig};
use decode::{
    SECONDS_PER_TIMESTAMP, SpecialTokens, WindowSegment, apply_rules, argmax, compression_ratio,
    log_softmax, sample, split_segments,
};
use timing::WordTiming;

use crate::{TimedText, Transcription, TranscriptionOptions};

pub const MODEL_TYPE: &str = "whisper";
pub const SAMPLE_RATE: u32 = 16_000;
const ENGLISH: &str = "en";
const TRANSLATE: &str = "translate";
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
// OpenAI's `transcribe` defaults: a window is redecoded hotter when its text compresses this well or its tokens
// average this little log-probability, and skipped as silence when its no-speech probability also passes this
const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;
const LOGPROB_THRESHOLD: f32 = -1.0;
const NO_SPEECH_THRESHOLD: f32 = 0.6;
const FALLBACK_TEMPERATURES: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
// a window decoded hotter than this is not carried into the next one's prompt
const PROMPT_RESET_TEMPERATURE: f32 = 0.5;
// the sampling at a fallback temperature draws from this seed, so a transcript repeats
const SAMPLING_SEED: u64 = 0;
const START_OF_PREVIOUS: &str = "<|startofprev|>";
// `<|nocaptions|>` in the first multilingual checkpoints
const NO_SPEECH: [&str; 2] = ["<|nospeech|>", "<|nocaptions|>"];

/// How one window is decoded.
#[derive(Clone, Copy, Debug)]
struct WindowDecoding {
    temperature: f32,
    beam_size: usize,
    patience: f32,
}

/// One window's decoded tokens (no end token) and what OpenAI judges a decode by.
struct Decoded {
    tokens: Vec<u32>,
    avg_logprob: f32,
    no_speech: f32,
    compression_ratio: f64,
    temperature: f32,
}

// one beam: its tokens after the prompt, their summed log-probability, its cache and next logits
struct Beam {
    tokens: Vec<u32>,
    sum: f32,
    cache: DecoderCache,
    logits: Vec<f32>,
}

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
    start_of_previous: Option<u32>,
    no_speech: Option<u32>,
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
            decoder: Decoder::new(&config.decoder(), vb.pp("decoder"), None)?,
            start_of_previous: tokenizer.token_to_id(START_OF_PREVIOUS),
            no_speech: NO_SPEECH.iter().find_map(|t| tokenizer.token_to_id(t)),
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
    fn start_sequence(&self, language: Option<&str>, task: Option<u32>) -> Vec<u32> {
        let mut tokens = vec![self.special.start];
        if let Some(code) = language.filter(|_| self.multilingual())
            && let Some(&id) = self.generation.lang_to_id.get(&format!("<|{code}|>"))
        {
            tokens.push(id);
            tokens.extend(task);
        }
        tokens
    }

    // OpenAI's `DecodingTask.run` on one window: `previous` as `<|startofprev|>` context before `start`, then greedy,
    // sampled or beam decoding under the timestamp rules
    fn decode_window(
        &self,
        encoded: &Tensor,
        previous: &[u32],
        start: &[u32],
        how: WindowDecoding,
        rng: &mut StdRng,
    ) -> Result<Decoded> {
        let mut prompt = Vec::new();
        if let Some(marker) = self.start_of_previous.filter(|_| !previous.is_empty()) {
            let keep = self.decoder.max_positions() / 2 - 1;
            prompt.push(marker);
            prompt.extend(&previous[previous.len().saturating_sub(keep)..]);
        }
        let start_index = prompt.len();
        prompt.extend(start);
        let mut cache = self.decoder.cache(encoded)?;
        let out = self.decoder.forward(&prompt, &mut cache, false)?;
        let no_speech = match self.no_speech {
            Some(id) => {
                let row = out.logits.get(start_index)?.to_vec1::<f32>()?;
                log_softmax(&row)[id as usize].exp()
            }
            None => 0.0,
        };
        let logits = Decoder::last_logits(&out)?;
        let budget = MAX_WINDOW_TOKENS.min(self.decoder.max_positions() - prompt.len());
        let (tokens, sum) = if how.beam_size > 1 && how.temperature == 0.0 {
            self.beam_search(cache, logits, budget, how)?
        } else {
            self.sample_window(cache, logits, budget, how.temperature, rng)?
        };
        let text: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|&t| !self.special.is_timestamp(t))
            .collect();
        let compression_ratio = compression_ratio(self.token_text(&text)?.trim());
        Ok(Decoded {
            avg_logprob: sum / (tokens.len() + 1) as f32,
            tokens,
            no_speech,
            compression_ratio,
            temperature: how.temperature,
        })
    }

    // OpenAI's `GreedyDecoder`: the likeliest token at 0, a draw at a temperature; the sum counts the end token's
    // log-probability, as OpenAI's does
    fn sample_window(
        &self,
        mut cache: DecoderCache,
        mut logits: Vec<f32>,
        budget: usize,
        temperature: f32,
        rng: &mut StdRng,
    ) -> Result<(Vec<u32>, f32)> {
        let mut generated = Vec::new();
        let mut sum = 0.0;
        while generated.len() < budget {
            apply_rules(&self.special, &mut logits, &generated, true);
            let token = if temperature == 0.0 {
                argmax(&logits)
            } else {
                sample(&logits, temperature, rng)
            };
            sum += log_softmax(&logits)[token as usize];
            if token == self.special.eot {
                break;
            }
            generated.push(token);
            if cache.len() >= self.decoder.max_positions() {
                break;
            }
            let out = self.decoder.forward(&[token], &mut cache, false)?;
            logits = Decoder::last_logits(&out)?;
        }
        Ok((generated, sum))
    }

    // OpenAI's `BeamSearchDecoder` and `MaximumLikelihoodRanker`: each step the best `beam_size` unfinished
    // continuations go on, finished ones collect until `beam_size * patience` have, and the best finished sequence by
    // log-probability over its length wins
    fn beam_search(
        &self,
        cache: DecoderCache,
        logits: Vec<f32>,
        budget: usize,
        how: WindowDecoding,
    ) -> Result<(Vec<u32>, f32)> {
        let width = how.beam_size;
        let max_finished = ((width as f32 * how.patience).round() as usize).max(1);
        let mut beams = vec![Beam {
            tokens: Vec::new(),
            sum: 0.0,
            cache,
            logits,
        }];
        let mut finished: Vec<(Vec<u32>, f32)> = Vec::new();
        for _ in 0..budget {
            let mut candidates: Vec<(Vec<u32>, f32, usize)> = Vec::new();
            for (j, beam) in beams.iter().enumerate() {
                let mut logits = beam.logits.clone();
                apply_rules(&self.special, &mut logits, &beam.tokens, true);
                let logprobs = log_softmax(&logits);
                let mut order: Vec<usize> = (0..logprobs.len()).collect();
                let top = (width + 1).min(order.len());
                order.select_nth_unstable_by(top - 1, |&a, &b| logprobs[b].total_cmp(&logprobs[a]));
                order.truncate(top);
                order.sort_by(|&a, &b| logprobs[b].total_cmp(&logprobs[a]));
                for token in order {
                    let mut sequence = beam.tokens.clone();
                    sequence.push(u32::try_from(token).expect("a vocabulary index fits u32"));
                    let score = beam.sum + logprobs[token];
                    match candidates.iter_mut().find(|c| c.0 == sequence) {
                        Some(known) => (known.1, known.2) = (score, j),
                        None => candidates.push((sequence, score, j)),
                    }
                }
            }
            candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mut next = Vec::new();
            for (sequence, score, source) in candidates {
                if sequence.last() == Some(&self.special.eot) {
                    if finished.len() < max_finished {
                        finished.push((sequence[..sequence.len() - 1].to_vec(), score));
                    }
                } else {
                    next.push((sequence, score, source));
                    if next.len() == width {
                        break;
                    }
                }
            }
            if finished.len() >= max_finished || next.is_empty() {
                break;
            }
            let mut stepped = Vec::with_capacity(next.len());
            for (tokens, sum, source) in next {
                let mut cache = beams[source].cache.clone();
                if cache.len() >= self.decoder.max_positions() {
                    continue;
                }
                let token = tokens[tokens.len() - 1];
                let out = self.decoder.forward(&[token], &mut cache, false)?;
                stepped.push(Beam {
                    tokens,
                    sum,
                    cache,
                    logits: Decoder::last_logits(&out)?,
                });
            }
            if stepped.is_empty() {
                break;
            }
            beams = stepped;
        }
        if finished.len() < width {
            beams.sort_by(|a, b| b.sum.total_cmp(&a.sum));
            for beam in beams {
                if finished.len() >= width {
                    break;
                }
                finished.push((beam.tokens, beam.sum));
            }
        }
        let score = |(tokens, sum): &(Vec<u32>, f32)| sum / tokens.len() as f32;
        let best = (0..finished.len())
            .fold(None::<usize>, |best, i| match best {
                Some(b) if score(&finished[b]) >= score(&finished[i]) => Some(b),
                _ => Some(i),
            })
            .expect("a beam search ends with a hypothesis");
        Ok(finished.swap_remove(best))
    }

    // OpenAI's `decode_with_fallback`: hotter until the window is neither repetitive nor improbable, unless it is
    // silence
    fn decode_with_fallback(
        &self,
        encoded: &Tensor,
        previous: &[u32],
        start: &[u32],
        options: &TranscriptionOptions,
        rng: &mut StdRng,
    ) -> Result<Decoded> {
        let fixed = options.temperature.filter(|&t| t > 0.0);
        let temperatures: &[f32] = match &fixed {
            Some(t) => std::slice::from_ref(t),
            None => &FALLBACK_TEMPERATURES,
        };
        let mut last = None;
        for &temperature in temperatures {
            let how = WindowDecoding {
                temperature,
                beam_size: options.beam_size.unwrap_or(1),
                patience: options.patience.unwrap_or(1.0),
            };
            let decoded = self.decode_window(encoded, previous, start, how, rng)?;
            let silence =
                decoded.no_speech > NO_SPEECH_THRESHOLD && decoded.avg_logprob < LOGPROB_THRESHOLD;
            let failed = decoded.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                || decoded.avg_logprob < LOGPROB_THRESHOLD;
            let done = silence || !failed;
            last = Some(decoded);
            if done {
                break;
            }
        }
        Ok(last.expect("at least one temperature"))
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
            inference_audio::mel::resample(pcm, sample_rate, SAMPLE_RATE)
                .map_err(msg)?
                .into_owned()
        };
        let duration = pcm.len() as f64 / f64::from(SAMPLE_RATE);
        let content_frames = pcm.len() / HOP_LENGTH;
        // OpenAI pads a window of silence after the audio before its log-mel, whose floor is the whole clip's peak
        let mut padded = pcm;
        padded.resize(padded.len() + WINDOW_SAMPLES, 0.0);
        let frames = self.log_mel.frames(&padded, LogMelMax::Clip);
        let n_mels = self.log_mel.n_mels;
        let mut language = match options.language.as_deref() {
            Some(l) if options.language_identified => self.language_code(l).ok(),
            Some(l) => Some(self.language_code(l)?),
            None => None,
        };
        if !self.multilingual() {
            language.get_or_insert_with(|| ENGLISH.to_string());
        }
        let task = if options.translate {
            Some(self.generation.task_to_id.get(TRANSLATE).copied().ok_or_else(|| {
                msg("this Whisper checkpoint transcribes English only; it does not translate")
            })?)
        } else {
            self.special.transcribe
        };
        let mut words = Vec::new();
        let mut text = String::new();
        let mut seek = 0usize;
        let mut last_speech = 0.0;
        // the text so far as the next window's context, from `carried` on (OpenAI's `all_tokens`, `prompt_reset_since`)
        let mut history: Vec<u32> = match options.prompt.as_deref().map(str::trim) {
            Some(prompt) if !prompt.is_empty() => self
                .tokenizer
                .encode(format!(" {prompt}"), false)
                .map_err(msg)?
                .get_ids()
                .to_vec(),
            _ => Vec::new(),
        };
        let mut carried = 0;
        let condition = options.condition_on_previous_text.unwrap_or(true);
        let mut rng = StdRng::seed_from_u64(SAMPLING_SEED);
        while seek < content_frames {
            let window_frames = WINDOW_FRAMES.min(content_frames - seek);
            // OpenAI's `pad_or_trim`: a window short of 30 s is zeros past the audio, not the padding's log-mel
            let mut mel = vec![0.0f32; n_mels * WINDOW_FRAMES];
            for (t, frame) in frames[seek..seek + window_frames].iter().enumerate() {
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
            let start = self.start_sequence(language.as_deref(), task);
            let decoded = self.decode_with_fallback(
                &encoded,
                &history[carried..],
                &start,
                options,
                &mut rng,
            )?;
            // silence, unless the text is probable enough despite the no-speech probability
            if decoded.no_speech > NO_SPEECH_THRESHOLD && decoded.avg_logprob <= LOGPROB_THRESHOLD {
                seek += window_frames;
                continue;
            }
            let generated = decoded.tokens;
            let window_seconds = window_frames as f64 * HOP_LENGTH as f64 / f64::from(SAMPLE_RATE);
            let (segments, consumed) = split_segments(&self.special, &generated, window_seconds);
            let offset = seek as f64 * HOP_LENGTH as f64 / f64::from(SAMPLE_RATE);
            // a translation is English, whatever the speech's language
            let unspaced = !options.translate
                && language
                    .as_deref()
                    .is_some_and(|l| UNSPACED_LANGUAGES.contains(&l));
            for segment in &segments {
                text.push_str(&self.token_text(&segment.tokens)?);
                // an instantaneous segment carries nothing forward, as in OpenAI's
                if segment.end > segment.start {
                    history.extend(&segment.decoded);
                }
            }
            if !condition || decoded.temperature > PROMPT_RESET_TEMPERATURE {
                carried = history.len();
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
            segments: Vec::new(),
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
