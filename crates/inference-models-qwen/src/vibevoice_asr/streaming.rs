//! VibeVoice-ASR Streaming, from Microsoft's layout (`VibeVoiceForASRStreamingTraining`): the offline model's encoders
//! and projector over windows of audio, each encoded alone, and a Qwen2 decoder whose cache runs the whole recording.
//! Each window is the speech start, its embeddings and the speech end; the decoder writes that window's text, with
//! `\n Speaker N:` where a turn begins, up to `<|text_chunk_end|>`, which it is then fed, as the reference's
//! `streaming_generate` runs.

use inference_nn::decoder::DecoderStack;
use inference_nn::kv_cache::KvCache;
use inference_nn::layers::Activation;
use inference_nn::loaders::TimedSegment;
use inference_nn::model::NormalLoadingMetadata;
use inference_nn::paged_attention::AttentionImplementation;
use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Linear, Module};
use inference_tensor::{DType, Device, IndexOp, Result, Tensor};
use tokenizers::Tokenizer;

use super::config::EncoderConfig;
use super::{AudioEmbedder, AudioSpec, EncoderNames};
use crate::qwen2_audio::config::TextConfig;

/// The streaming checkpoints' class.
pub const ARCHITECTURE: &str = "VibeVoiceForASRStreamingTraining";
const PROMPT: &str = "You are a helpful assistant that transcribes audio input into text output. Please transcribe \
                      the following audios streamingly with these keys: speaker, content";
const SPEECH_START: &str = "<|object_ref_start|>";
const SPEECH_END: &str = "<|object_ref_end|>";
const CHUNK_END: &str = "<|text_chunk_end|>";
const EOS: &str = "<|endoftext|>";
// the special tokens the reference strips from each window's text
const STRIPPED: [&str; 7] = [
    "<|text_chunk_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|speech_start|>",
    "<|speech_end|>",
    "<|speech_pad|>",
];
// the reference's budget per window
const MAX_NEW_TOKENS: usize = 256;
const SPEAKER_MARK: &str = "Speaker ";
// the upstream encoders' fixed sizes, which their config does not state
const ENCODER_KERNEL: usize = 7;
const ENCODER_FFN_EXPANSION: usize = 4;
const KV_DIM: usize = 2;
// the cache's first allocation, in positions; it grows on demand
const CACHE_CAPACITY: usize = 4096;
const DEPTH_SEPARATOR: char = '-';

#[derive(Debug, Clone, inference_json::FromJson)]
struct UpstreamTokenizer {
    channels: usize,
    encoder_ratios: Vec<usize>,
    encoder_depths: String,
    encoder_n_filters: usize,
    vae_dim: usize,
    layernorm_eps: f64,
}

impl UpstreamTokenizer {
    // as transformers' port names it: its ratios run stem-first, the reverse of how the upstream config lists them
    fn encoder(&self) -> Result<EncoderConfig> {
        let depths = self
            .encoder_depths
            .split(DEPTH_SEPARATOR)
            .map(|d| d.trim().parse::<usize>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| inference_tensor::Error::Msg(format!("`encoder_depths`: {e}")))?;
        Ok(EncoderConfig {
            channels: self.channels,
            depths,
            downsampling_ratios: self.encoder_ratios.iter().rev().copied().collect(),
            ffn_expansion: ENCODER_FFN_EXPANSION,
            hidden_act: Activation::Gelu,
            hidden_size: self.vae_dim,
            kernel_size: ENCODER_KERNEL,
            num_filters: self.encoder_n_filters,
            rms_norm_eps: self.layernorm_eps,
        })
    }
}

/// A streaming checkpoint's `config.json`, the fields the model reads.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct StreamingConfig {
    acoustic_tokenizer_config: UpstreamTokenizer,
    semantic_tokenizer_config: UpstreamTokenizer,
    decoder_config: TextConfig,
}

/// `preprocessor_config.json`: the window the checkpoint was trained on, in latent frames.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct StreamingFrames {
    pub speech_tok_compress_ratio: usize,
    pub target_sample_rate: u32,
    pub chunk_frames: usize,
    pub lookahead_frames: usize,
}

/// Whether a `config.json` is a streaming checkpoint's.
pub fn is_streaming_config(config: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(config)
        .ok()
        .and_then(|v| {
            v.get("architectures")?
                .as_array()?
                .iter()
                .find_map(|a| a.as_str().map(|a| a == ARCHITECTURE))
        })
        .unwrap_or(false)
}

/// One recording's decoding: its cache, the positions in it, and the audio not yet in a window.
pub struct StreamingSession {
    cache: Vec<KvCache>,
    offset: usize,
    // the prompt each pass starts from
    prompt: Vec<u32>,
    pass: usize,
    // samples from the next window's start on
    pending: Vec<f32>,
    // samples before `pending`
    consumed: usize,
    windows: Vec<WindowText>,
}

impl StreamingSession {
    /// The positions in this pass's cache.
    pub fn offset(&self) -> usize {
        self.offset
    }
}

/// One window's text and the seconds its text covers.
#[derive(Clone, Debug)]
pub struct WindowText {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// The pass it was decoded in: each starts a fresh cache, and its speaker numbers are its own.
    pub pass: usize,
}

/// A streamed transcript: the windows' text joined, cut into speaker turns timed by the windows they span.
#[derive(Clone, Debug)]
pub struct StreamingTranscript {
    pub text: String,
    pub segments: Vec<TimedSegment>,
    pub duration: f64,
}

pub struct VibeVoiceStreaming {
    audio: AudioEmbedder,
    stack: DecoderStack,
    head: Linear,
    tokenizer: Tokenizer,
    prompt: Vec<u32>,
    speech_start: u32,
    speech_end: u32,
    chunk_end: u32,
    eos: u32,
    chunk_samples: usize,
    window_samples: usize,
    window_frames: usize,
    sample_rate: u32,
    layers: usize,
    max_positions: usize,
    pass_positions: usize,
    device: Device,
}

impl VibeVoiceStreaming {
    /// `vb` is the checkpoint's root.
    pub fn new(
        config: &StreamingConfig,
        frames: &StreamingFrames,
        tokenizer: Tokenizer,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let text = config.decoder_config.qwen2(None);
        let acoustic = config.acoustic_tokenizer_config.encoder()?;
        let semantic = config.semantic_tokenizer_config.encoder()?;
        let spec = AudioSpec {
            acoustic: &acoustic,
            semantic: &semantic,
            hidden: text.hidden_size,
            names: EncoderNames::Upstream,
        };
        let audio = AudioEmbedder::from_spec(&spec, &vb)?;
        let vb_m = vb.pp("model").pp("language_model");
        let stack = DecoderStack::new(
            &text.decoder_spec()?,
            vb_m.clone(),
            None,
            true,
            NormalLoadingMetadata::single_device(&device),
            &AttentionImplementation::Eager,
        )?;
        // a tied checkpoint may still carry its head
        let head = if vb.contains_tensor("lm_head.weight") {
            vb.pp("lm_head")
                .get((text.vocab_size, text.hidden_size), "weight")?
        } else {
            vb_m.pp("embed_tokens")
                .get((text.vocab_size, text.hidden_size), "weight")?
        };
        let id = |token: &str| {
            tokenizer.token_to_id(token).ok_or_else(|| {
                inference_tensor::Error::Msg(format!("the tokenizer has no `{token}`"))
            })
        };
        let prompt = prompt_ids(&tokenizer, None)?;
        let hop = frames.speech_tok_compress_ratio;
        Ok(Self {
            audio,
            stack,
            head: Linear::new(head, None),
            speech_start: id(SPEECH_START)?,
            speech_end: id(SPEECH_END)?,
            chunk_end: id(CHUNK_END)?,
            eos: id(EOS)?,
            tokenizer,
            prompt,
            chunk_samples: frames.chunk_frames * hop,
            window_samples: (frames.chunk_frames + frames.lookahead_frames) * hop,
            window_frames: frames.chunk_frames + frames.lookahead_frames,
            sample_rate: frames.target_sample_rate,
            layers: text.num_hidden_layers,
            max_positions: text.max_position_embeddings,
            pass_positions: text.max_position_embeddings,
            device,
        })
    }

    /// Caps each pass's cache at `positions` (the decoder's maximum when larger): a window that might not fit starts
    /// a new pass, its cache fresh from the prompt.
    pub fn with_pass_positions(mut self, positions: usize) -> Self {
        self.pass_positions = positions.min(self.max_positions);
        self
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The prompt's token ids.
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt
    }

    /// One window's embeddings `(1, frames, hidden)`: `samples` zero-padded to a window, encoded alone.
    pub fn embed_window(&self, samples: &[f32]) -> Result<Tensor> {
        let mut padded = samples.to_vec();
        padded.resize(self.window_samples, 0.0);
        let len = padded.len();
        self.audio
            .embed(&Tensor::from_vec(padded, (1, 1, len), &self.device)?)?
            .to_dtype(self.dtype())?
            .unsqueeze(0)
    }

    fn dtype(&self) -> DType {
        self.head.weight().dtype()
    }

    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        self.stack
            .embed(&Tensor::new(ids, &self.device)?.unsqueeze(0)?)
    }

    // the logits `(vocab,)` after `xs` `(1, tokens, hidden)`, which join the session's cache
    fn step(&self, session: &mut StreamingSession, xs: Tensor) -> Result<Tensor> {
        let tokens = xs.dim(1)?;
        let hidden = self
            .stack
            .forward_cached(xs, &mut session.cache, session.offset)?;
        session.offset += tokens;
        self.head
            .forward(&hidden.i((0, tokens - 1))?.unsqueeze(0)?)?
            .squeeze(0)
    }

    /// A session with the prompt in its cache; a `context` (hotwords, names, the topic) joins the prompt as the
    /// reference's `context_info` does.
    pub fn start(&self, context: Option<&str>) -> Result<StreamingSession> {
        let mut session = StreamingSession {
            cache: Vec::new(),
            offset: 0,
            prompt: prompt_ids(&self.tokenizer, context)?,
            pass: 0,
            pending: Vec::new(),
            consumed: 0,
            windows: Vec::new(),
        };
        self.prefill(&mut session)?;
        Ok(session)
    }

    fn prefill(&self, session: &mut StreamingSession) -> Result<()> {
        session.cache = (0..self.layers)
            .map(|_| KvCache::new_normal(KV_DIM, self.max_positions, CACHE_CAPACITY))
            .collect();
        session.offset = 0;
        let prompt = self.embed(&session.prompt)?;
        self.step(session, prompt)?;
        Ok(())
    }

    /// One window's token ids and text: `samples` (the window and its lookahead, zero-padded at the end) encoded
    /// alone.
    pub fn window(
        &self,
        session: &mut StreamingSession,
        samples: &[f32],
    ) -> Result<(Vec<u32>, String)> {
        let audio = self.embed_window(samples)?;
        let xs = Tensor::cat(
            &[
                &self.embed(&[self.speech_start])?,
                &audio,
                &self.embed(&[self.speech_end])?,
            ],
            1,
        )?;
        let mut logits = self.step(session, xs)?;
        let mut ids = Vec::new();
        while ids.len() < MAX_NEW_TOKENS {
            let token = logits.argmax(0)?.to_scalar::<u32>()?;
            if token == self.chunk_end || token == self.eos {
                break;
            }
            ids.push(token);
            logits = self.step(session, self.embed(&[token])?)?;
        }
        self.step(session, self.embed(&[self.chunk_end])?)?;
        let mut text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
        for special in STRIPPED {
            text = text.replace(special, "");
        }
        Ok((ids, text))
    }

    // the window starting at `start` samples, its text over the chunk it advances
    fn run_window(&self, session: &mut StreamingSession, samples: &[f32]) -> Result<WindowText> {
        let seconds = |s: usize| self.seconds(s);
        // the window, its markers, the longest text and the closing marker
        let room = self.window_frames + 2 + MAX_NEW_TOKENS + 1;
        if session.offset + room > self.pass_positions {
            session.pass += 1;
            self.prefill(session)?;
        }
        let (_, text) = self.window(session, samples)?;
        let start = session.consumed;
        let window = WindowText {
            start: seconds(start),
            end: seconds(start + self.chunk_samples.min(samples.len())),
            text,
            pass: session.pass,
        };
        session.windows.push(window.clone());
        Ok(window)
    }

    // sample counts stay far below f64's 52-bit mantissa
    #[allow(clippy::cast_precision_loss)]
    fn seconds(&self, samples: usize) -> f64 {
        samples as f64 / f64::from(self.sample_rate)
    }

    /// The samples a window takes (its chunk and lookahead) and the chunk it advances by.
    pub fn window_shape(&self) -> (usize, usize) {
        (self.window_samples, self.chunk_samples)
    }

    /// Appends audio at the model's rate and decodes every window it completes; their texts in order.
    pub fn push(&self, session: &mut StreamingSession, pcm: &[f32]) -> Result<Vec<WindowText>> {
        session.pending.extend_from_slice(pcm);
        let mut out = Vec::new();
        while session.pending.len() >= self.window_samples {
            let window = session.pending[..self.window_samples].to_vec();
            out.push(self.run_window(session, &window)?);
            session.pending.drain(..self.chunk_samples);
            session.consumed += self.chunk_samples;
        }
        Ok(out)
    }

    /// Decodes what is left, the last window zero-padded, and the session's whole transcript.
    pub fn finish(
        &self,
        mut session: StreamingSession,
    ) -> Result<(Vec<WindowText>, StreamingTranscript)> {
        let mut out = Vec::new();
        while !session.pending.is_empty() {
            let take = session.pending.len().min(self.window_samples);
            let window = session.pending[..take].to_vec();
            out.push(self.run_window(&mut session, &window)?);
            let advance = session.pending.len().min(self.chunk_samples);
            session.pending.drain(..advance);
            session.consumed += advance;
        }
        let duration = self.seconds(session.consumed);
        Ok((out, transcript(&session.windows, duration)))
    }

    /// The whole of `pcm` at the model's rate, window by window.
    pub fn transcribe(&self, pcm: &[f32], context: Option<&str>) -> Result<StreamingTranscript> {
        let mut session = self.start(context)?;
        self.push(&mut session, pcm)?;
        Ok(self.finish(session)?.1)
    }
}

fn prompt_ids(tokenizer: &Tokenizer, context: Option<&str>) -> Result<Vec<u32>> {
    let prompt = match context.map(str::trim).filter(|c| !c.is_empty()) {
        Some(context) => format!("{PROMPT} and extra info: {context}\n"),
        None => format!("{PROMPT}\n"),
    };
    Ok(tokenizer
        .encode_fast(prompt, false)
        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?
        .get_ids()
        .to_vec())
}

/// The windows' text cut at each `\n Speaker N:` into turns: a turn starts with the window its mark is in and ends
/// with the last window it reaches into; text before any mark is speaker-less. A later pass numbers its speakers
/// after every earlier pass's, since it cannot tell they are the same people.
pub fn transcript(windows: &[WindowText], duration: f64) -> StreamingTranscript {
    let mut segments: Vec<TimedSegment> = Vec::new();
    let mut open: Option<TimedSegment> = None;
    let (mut pass, mut base, mut fresh) = (0, 0, 0);
    let close = |open: &mut Option<TimedSegment>, segments: &mut Vec<TimedSegment>| {
        if let Some(mut segment) = open.take() {
            segment.text = segment.text.trim().to_string();
            if !segment.text.is_empty() {
                segments.push(segment);
            }
        }
    };
    for window in windows {
        if window.pass != pass {
            close(&mut open, &mut segments);
            (pass, base) = (window.pass, fresh);
        }
        let mut rest = window.text.as_str();
        while !rest.is_empty() {
            let mark = rest.find(SPEAKER_MARK).and_then(|at| {
                let after = &rest[at + SPEAKER_MARK.len()..];
                let digits = after.find(':')?;
                let speaker = base + after[..digits].trim().parse::<u32>().ok()?;
                Some((at, at + SPEAKER_MARK.len() + digits + 1, speaker))
            });
            let (text, next) = match mark {
                Some((at, end, _)) => (&rest[..at], &rest[end..]),
                None => (rest, ""),
            };
            let segment = open.get_or_insert_with(|| TimedSegment {
                start: window.start,
                end: window.end,
                speaker: None,
                text: String::new(),
            });
            if !text.trim().is_empty() {
                segment.text.push_str(text);
                segment.end = window.end;
            }
            if let Some((_, _, speaker)) = mark {
                fresh = fresh.max(speaker + 1);
                close(&mut open, &mut segments);
                open = Some(TimedSegment {
                    start: window.start,
                    end: window.end,
                    speaker: Some(speaker),
                    text: String::new(),
                });
            }
            rest = next;
        }
    }
    close(&mut open, &mut segments);
    for segment in &mut segments {
        segment.end = segment.end.min(duration);
    }
    let text = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    StreamingTranscript {
        text,
        segments,
        duration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start: f64, text: &str) -> WindowText {
        WindowText {
            start,
            end: start + 2.0,
            text: text.to_string(),
            pass: 0,
        }
    }

    #[test]
    fn windows_cut_into_speaker_turns_at_their_marks() {
        let windows = [
            window(0.0, " \n Speaker 0:Did you finish?"),
            window(2.0, " \n Speaker 1:Almost. I just need "),
            window(4.0, "the numbers. \n Speaker 0:Great. "),
            window(6.0, "Send it."),
        ];
        let out = transcript(&windows, 7.5);
        let turns: Vec<(Option<u32>, &str, f64, f64)> = out
            .segments
            .iter()
            .map(|s| (s.speaker, s.text.as_str(), s.start, s.end))
            .collect();
        assert_eq!(
            turns,
            [
                (Some(0), "Did you finish?", 0.0, 2.0),
                (Some(1), "Almost. I just need the numbers.", 2.0, 6.0),
                (Some(0), "Great. Send it.", 4.0, 7.5),
            ]
        );
        assert_eq!(
            out.text,
            "Did you finish? Almost. I just need the numbers. Great. Send it."
        );
    }

    #[test]
    fn a_later_pass_numbers_its_speakers_after_the_earlier_ones() {
        let windows = [
            window(0.0, " \n Speaker 0:One. \n Speaker 1:Two."),
            WindowText {
                pass: 1,
                ..window(2.0, "still two. \n Speaker 0:Three.")
            },
        ];
        let turns: Vec<(Option<u32>, String)> = transcript(&windows, 4.0)
            .segments
            .into_iter()
            .map(|s| (s.speaker, s.text))
            .collect();
        assert_eq!(
            turns,
            [
                (Some(0), "One.".to_string()),
                (Some(1), "Two.".to_string()),
                (None, "still two.".to_string()),
                (Some(2), "Three.".to_string()),
            ]
        );
    }
}
