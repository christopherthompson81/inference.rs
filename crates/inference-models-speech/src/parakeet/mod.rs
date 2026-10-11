//! NVIDIA's Parakeet speech recognisers: a FastConformer encoder under a CTC, RNN-T or TDT head, loaded from the
//! transformers layout (`config.json`, `processor_config.json`, `tokenizer.json`, `model.safetensors`).

mod config;
mod decoder;
mod encoder;
mod gguf_checkpoint;
mod gguf_writer;
mod nemo_checkpoint;

pub use gguf_checkpoint::is_asr_gguf;
pub use gguf_writer::{PRESETS, QuantPolicy, TYPE_NAMES, write_asr_gguf};
pub use nemo_checkpoint::is_asr;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference_audio::nemo::{NemoMel, NemoMelConfig};
use inference_tensor::nn::{Linear, Module, VarBuilder, linear};
use inference_tensor::{DType, Device, Error, Result, Tensor};
use tokenizers::Tokenizer;

use crate::silero::{SegmentOptions, SileroVad, speech_segments};
use crate::{TimedText, Transcription, TranscriptionOptions};
pub use config::{
    EncoderConfig, FeatureConfig, HeadKind, MODEL_TYPES, ParakeetConfig, ProcessorConfig,
    STREAMING_ENCODER_TYPE,
};
pub use decoder::Emission;
use decoder::Head;
pub use encoder::{Encoder, relative_positions};

// NeMo's bound for full attention on these models, which attend over the whole recording at once
const MAX_SECONDS: f64 = 24.0 * 60.0;
// with a VAD, longer audio is windowed: full attention is quadratic in length, and silence costs as much as speech
const LONG_FORM_SECONDS: f64 = 5.0 * 60.0;
const WINDOW_SECONDS: f64 = 2.0 * 60.0;
// how far a window reaches into the gaps around it, for speech the VAD scored just under its threshold
const WINDOW_EDGE_SECONDS: f64 = 1.0;
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

// Nemotron-3.5's language conditioning: a one-hot prompt joins each encoder frame, through a two-layer projector
struct Prompts {
    linear_1: Linear,
    linear_2: Linear,
    slots: usize,
    default: usize,
    ids: HashMap<String, usize>,
    // the tag tokens the model emits for the language it identifies, as `<de-DE>`
    tags: HashMap<u32, String>,
}

impl Prompts {
    fn new(
        config: &ParakeetConfig,
        processor: &ProcessorConfig,
        tokenizer: &Tokenizer,
        vb: VarBuilder,
    ) -> Result<Option<Self>> {
        let (Some(slots), Some(inner), Some(default)) = (
            config.num_prompts,
            config.prompt_intermediate_size,
            config.default_prompt_id,
        ) else {
            return Ok(None);
        };
        let h = config.encoder_config.hidden_size;
        Ok(Some(Self {
            linear_1: linear(h + slots, inner, vb.pp("linear_1"))?,
            linear_2: linear(inner, h, vb.pp("linear_2"))?,
            slots,
            default,
            ids: processor.prompt_dictionary.clone(),
            tags: processor
                .prompt_dictionary
                .keys()
                .filter_map(|l| Some((tokenizer.token_to_id(&format!("<{l}>"))?, l.clone())))
                .collect(),
        }))
    }

    fn id(&self, language: Option<&str>) -> Result<usize> {
        let Some(language) = language else {
            return Ok(self.default);
        };
        self.ids.get(language).copied().ok_or_else(|| {
            let mut known: Vec<&str> = self.ids.keys().map(String::as_str).collect();
            known.sort_unstable();
            msg(format!(
                "`{language}` is not a language this model takes; it takes {}",
                known.join(", ")
            ))
        })
    }

    fn one_hot(&self, id: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let mut one_hot = vec![0f32; self.slots];
        one_hot[id] = 1.;
        Tensor::from_vec(one_hot, (1, 1, self.slots), device)?.to_dtype(dtype)
    }

    // `(1, T, hidden)` encoder frames, each joined by the language's one-hot
    fn forward(&self, encoded: &Tensor, id: usize) -> Result<Tensor> {
        self.forward_with(
            encoded,
            &self.one_hot(id, encoded.dtype(), encoded.device())?,
        )
    }

    fn forward_with(&self, encoded: &Tensor, one_hot: &Tensor) -> Result<Tensor> {
        let t = encoded.dim(1)?;
        let one_hot = one_hot.broadcast_as((1, t, self.slots))?;
        let joined = Tensor::cat(&[encoded, &one_hot], 2)?;
        self.linear_2
            .forward(&self.linear_1.forward(&joined)?.relu()?)
    }
}

pub struct Parakeet {
    mel: NemoMel,
    encoder: Encoder,
    prompts: Option<Prompts>,
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
            // streaming checkpoints see each chunk's raw log-mel, so offline ones normalize alone
            normalize: !enc.is_streaming(),
        });
        let frame_seconds =
            (fe.hop_length * enc.subsampling_factor) as f64 / f64::from(fe.sampling_rate);
        Ok(Self {
            encoder: Encoder::new(enc, vb.pp("encoder"))?,
            prompts: Prompts::new(&config, &processor, &tokenizer, vb.pp("prompt_projector"))?,
            head: Head::new(&config, kind, enc.hidden_size, vb.clone())?,
            mel,
            tokenizer,
            kind,
            frame_seconds,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn head(&self) -> HeadKind {
        self.kind
    }

    pub fn sample_rate(&self) -> u32 {
        self.mel.config().sample_rate
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Transcribes mono `pcm` at `sample_rate`, resampling to the model's rate first.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        options: &TranscriptionOptions,
    ) -> Result<Transcription> {
        let language = self.language(options)?;
        // checked before resampling, which would copy hours of audio the model then refuses
        let seconds = pcm.len() as f64 / f64::from(sample_rate);
        if seconds > MAX_SECONDS {
            return Err(msg(format!(
                "the audio is {seconds:.0} s; Parakeet's full attention takes at most {MAX_SECONDS:.0} s \
                 (a transcription model loaded with a VAD transcribes longer audio)"
            )));
        }
        let pcm =
            inference_audio::mel::resample(pcm, sample_rate, self.sample_rate()).map_err(msg)?;
        self.transcribe_samples(&pcm, language, options.beam_size)
    }

    fn identifies(&self, language: &str) -> bool {
        self.prompts
            .as_ref()
            .is_some_and(|p| p.id(Some(language)).ok() == Some(p.default))
    }

    // the request's language for a model that takes one, and its beam, checked before any audio is touched
    fn language<'a>(&self, options: &'a TranscriptionOptions) -> Result<Option<&'a str>> {
        if self.kind == HeadKind::Ctc && options.beam_size.is_some_and(|b| b > 1) {
            return Err(msg(
                "beam search decodes transducer (RNN-T, TDT) heads; this one is CTC",
            ));
        }
        let Some(prompts) = &self.prompts else {
            return Ok(None);
        };
        let language = options.language.as_deref();
        prompts.id(language)?;
        Ok(language)
    }

    /// As `transcribe`, but audio past `LONG_FORM_SECONDS` is cut at the VAD's silences into windows of at most
    /// `WINDOW_SECONDS`, each transcribed alone; stretches without speech are skipped.
    pub fn transcribe_with_vad(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        vad: &SileroVad,
        options: &TranscriptionOptions,
    ) -> Result<Transcription> {
        let language = self.language(options)?;
        let beam = options.beam_size;
        let rate = self.sample_rate();
        if vad.config().sample_rate != rate {
            return Err(msg(format!(
                "the VAD runs at {} Hz and Parakeet at {rate} Hz",
                vad.config().sample_rate
            )));
        }
        let pcm = inference_audio::mel::resample(pcm, sample_rate, rate).map_err(msg)?;
        let duration = pcm.len() as f64 / f64::from(rate);
        if duration <= LONG_FORM_SECONDS {
            return self.transcribe_samples(&pcm, language, beam);
        }
        let options = SegmentOptions {
            max_speech_duration_s: Some(WINDOW_SECONDS),
            ..SegmentOptions::default()
        };
        let probabilities = vad.probabilities(&pcm)?;
        let speech = speech_segments(
            &probabilities,
            pcm.len(),
            rate,
            vad.config().chunk_size,
            &options,
        );
        let window_samples = (WINDOW_SECONDS * f64::from(rate)) as usize;
        let mut merged = Transcription {
            text: String::new(),
            tokens: Vec::new(),
            words: Vec::new(),
            duration,
            language: None,
        };
        let edge = (WINDOW_EDGE_SECONDS * f64::from(rate)) as usize;
        for (start, end) in widen(&windows(&speech, window_samples), pcm.len(), edge) {
            let offset = start as f64 / f64::from(rate);
            let part = self.transcribe_samples(&pcm[start..end], language, beam)?;
            merged.language = merged.language.or(part.language);
            let shift = |t: TimedText| TimedText {
                start: t.start + offset,
                end: t.end + offset,
                ..t
            };
            merged.tokens.extend(part.tokens.into_iter().map(shift));
            merged.words.extend(part.words.into_iter().map(shift));
            if !part.text.is_empty() {
                if !merged.text.is_empty() {
                    merged.text.push(' ');
                }
                merged.text.push_str(&part.text);
            }
        }
        Ok(merged)
    }

    fn transcribe_samples(
        &self,
        pcm: &[f32],
        language: Option<&str>,
        beam: Option<usize>,
    ) -> Result<Transcription> {
        let duration = pcm.len() as f64 / f64::from(self.sample_rate());
        let encoded = self.encode(pcm, language)?;
        let emissions = self.head.decode(&encoded, beam)?;
        self.transcription(&emissions, duration, language)
    }

    fn transcription(
        &self,
        emissions: &[Emission],
        duration: f64,
        language: Option<&str>,
    ) -> Result<Transcription> {
        let mut transcription = self.text(emissions, duration)?;
        let identified = self.prompts.as_ref().and_then(|prompts| {
            emissions
                .iter()
                .find_map(|e| prompts.tags.get(&e.token).cloned())
        });
        // a model left to its default prompt (`auto`) reports the language it identified, not the prompt's name
        transcription.language = match language {
            Some(l) if !self.identifies(l) => Some(l.to_string()),
            _ => identified,
        };
        Ok(transcription)
    }

    /// Whether the checkpoint is a cache-aware streaming one, which [`Parakeet::stream`] transcribes as audio
    /// arrives.
    pub fn can_stream(&self) -> bool {
        self.encoder.can_stream()
    }

    /// A transcription fed audio as it arrives, chunk by chunk, making the offline pass's decisions over the same
    /// audio. Greedy only; the language is a Nemotron-3.5 prompt as in `transcribe`.
    pub fn stream(&self, options: &TranscriptionOptions) -> Result<ParakeetStream> {
        if options.beam_size.is_some_and(|b| b > 1) {
            return Err(msg(
                "a streamed transcription decodes greedily; it takes no beam",
            ));
        }
        let language = self.language(options)?.map(str::to_string);
        let prompt = self
            .prompts
            .as_ref()
            .map(|p| p.one_hot(p.id(language.as_deref())?, self.dtype, &self.device))
            .transpose()?;
        Ok(ParakeetStream {
            mel: self.mel.stream().map_err(msg)?,
            encoder: self.encoder.stream(self.dtype, &self.device)?,
            decoder: self.head.stream(),
            prompt,
            language,
            emissions: Vec::new(),
            samples: 0,
            text: TextStream::default(),
        })
    }

    /// Feeds mono `pcm` at the model's rate to `stream`; returns the text the chunks it completes add.
    pub fn stream_push(&self, stream: &mut ParakeetStream, pcm: &[f32]) -> Result<String> {
        stream.samples += pcm.len();
        let mel = stream.mel.push(&self.mel, pcm);
        let n_mels = self.mel.config().n_mels;
        let chunks = self
            .encoder
            .stream_push(&mut stream.encoder, &mel, n_mels)?;
        self.stream_decode(stream, chunks)
    }

    /// Ends `stream`: the text its last frames add, and the whole transcription, as `transcribe` gives it.
    pub fn stream_finish(&self, mut stream: ParakeetStream) -> Result<(String, Transcription)> {
        let n_mels = self.mel.config().n_mels;
        let mel = stream.mel.finish(&self.mel);
        let chunks = self
            .encoder
            .stream_push(&mut stream.encoder, &mel, n_mels)?;
        let mut delta = self.stream_decode(&mut stream, chunks)?;
        let chunks = self.encoder.stream_finish(&mut stream.encoder, n_mels)?;
        delta.push_str(&self.stream_decode(&mut stream, chunks)?);
        let duration = stream.samples as f64 / f64::from(self.sample_rate());
        let transcription =
            self.transcription(&stream.emissions, duration, stream.language.as_deref())?;
        Ok((delta, transcription))
    }

    // decodes encoder chunks, each with its first frame's index, into the text they add to the stream's transcript
    fn stream_decode(
        &self,
        stream: &mut ParakeetStream,
        chunks: Vec<(usize, Tensor)>,
    ) -> Result<String> {
        let mut delta = String::new();
        for (offset, encoded) in chunks {
            let encoded = match (&self.prompts, &stream.prompt) {
                (Some(prompts), Some(one_hot)) => prompts.forward_with(&encoded, one_hot)?,
                _ => encoded,
            };
            for e in self
                .head
                .decode_chunk(&encoded, offset, &mut stream.decoder)?
            {
                stream.emissions.push(e);
                delta.push_str(&stream.text.step(&self.tokenizer, e.token)?);
            }
        }
        Ok(delta)
    }

    /// `(frames, mels)` row-major log-mel features of mono `pcm` at the model's rate, and the frame count.
    pub fn features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        self.mel.features(pcm).map_err(msg)
    }

    /// `(1, frames, hidden)` encoder output for mono `pcm` at the model's rate, conditioned on the language for a
    /// model that takes one (`None` lets it identify the language).
    pub fn encode(&self, pcm: &[f32], language: Option<&str>) -> Result<Tensor> {
        let (features, frames) = self.features(pcm)?;
        let n_mels = self.mel.config().n_mels;
        let features =
            Tensor::from_vec(features, (1, frames, n_mels), &self.device)?.to_dtype(self.dtype)?;
        let encoded = self.encoder.forward(&features)?;
        match &self.prompts {
            Some(prompts) => prompts.forward(&encoded, prompts.id(language)?),
            None => Ok(encoded),
        }
    }

    /// Token ids and their frames, before detokenisation: greedy, or a beam of `beam` hypotheses.
    pub fn emissions(&self, encoded: &Tensor, beam: Option<usize>) -> Result<Vec<Emission>> {
        self.head.decode(encoded, beam)
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
            language: None,
        })
    }
}

/// A transcription in progress, fed through [`Parakeet::stream_push`]; it holds no borrow of the model, so an
/// engine can keep it between requests.
pub struct ParakeetStream {
    mel: inference_audio::nemo::NemoMelStream,
    encoder: encoder::EncoderStream,
    decoder: decoder::DecodeStream,
    // Nemotron-3.5's language one-hot, `(1, 1, slots)`
    prompt: Option<Tensor>,
    language: Option<String>,
    emissions: Vec<Emission>,
    samples: usize,
    text: TextStream,
}

// tokenizers' `DecodeStream` without its borrow of the tokenizer: each token's new text from a short prefix
// window, so a stream's cost per token stays flat; the transcript's leading whitespace is trimmed, as offline
#[derive(Default)]
struct TextStream {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    started: bool,
}

impl TextStream {
    fn step(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<String> {
        self.ids.push(id);
        let text = tokenizer.decode(&self.ids, true).map_err(msg)?;
        // a partial UTF-8 sequence decodes to the replacement character until the rest arrives
        if text.len() <= self.prefix.len() || text.ends_with(char::REPLACEMENT_CHARACTER) {
            return Ok(String::new());
        }
        let Some(new) = text.strip_prefix(self.prefix.as_str()) else {
            return Err(msg(
                "the tokenizer's decoding of a longer sequence changed its prefix",
            ));
        };
        let mut new = new.to_string();
        let new_prefix_index = self.ids.len() - self.prefix_index;
        self.ids.drain(..self.prefix_index);
        self.prefix = tokenizer.decode(&self.ids, true).map_err(msg)?;
        self.prefix_index = new_prefix_index;
        if !self.started {
            new = new.trim_start().to_string();
            self.started = !new.is_empty();
        }
        Ok(new)
    }
}

// consecutive segments grouped into ranges of at most `max` samples; a longer segment is a window of its own
fn windows(speech: &[(usize, usize)], max: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in speech {
        match out.last_mut() {
            Some(window) if end - window.0 <= max => window.1 = end,
            _ => out.push((start, end)),
        }
    }
    out
}

// each window reaches up to `edge` samples into the gaps around it, never past a gap's midpoint or the audio
fn widen(windows: &[(usize, usize)], len: usize, edge: usize) -> Vec<(usize, usize)> {
    windows
        .iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let before = i.checked_sub(1).map_or(0, |p| windows[p].1.midpoint(start));
            let after = windows.get(i + 1).map_or(len, |n| end.midpoint(n.0));
            (
                start.saturating_sub(edge).max(before),
                (end + edge).min(after),
            )
        })
        .collect()
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
    fn speech_groups_into_bounded_windows() {
        let speech = [(0, 10), (20, 40), (45, 90), (95, 300), (310, 320)];
        assert_eq!(windows(&speech, 100), [(0, 90), (95, 300), (310, 320)]);
        assert_eq!(windows(&[], 100), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn windows_reach_into_their_gaps_up_to_the_midpoint() {
        let windows = [(100, 200), (210, 400), (1000, 1100)];
        assert_eq!(
            widen(&windows, 1150, 50),
            [(50, 205), (205, 450), (950, 1150)]
        );
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
