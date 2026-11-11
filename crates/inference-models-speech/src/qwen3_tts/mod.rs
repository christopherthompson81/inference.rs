//! Qwen3-TTS (12 Hz): a Qwen3 talker predicts each 80 ms frame's first codec code, a small code predictor fills the
//! other 15, and the codec decoder turns the frames into 24 kHz audio. Preset speakers (CustomVoice) and a voice
//! described in an instruction (VoiceDesign); the reference is Qwen's `qwen-tts` package.

mod codec;
pub mod config;
mod talker;
mod tokenizer;

use std::path::Path;
use std::sync::Arc;

use inference_quant::{ShardedSafeTensors, ShardedVarBuilder};
use inference_tensor::{DType, Device, Tensor};
use rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;

use crate::{SpeechGenerationOutput, SpeechOptions};
use codec::CodecDecoder;
pub use config::CodecConfig;
pub use config::Qwen3TtsConfig;
use talker::{StackCache, Talker};

pub const CONFIG: &str = "config.json";
const WEIGHTS: &str = "model.safetensors";
const CODEC_DIR: &str = "speech_tokenizer";
/// The files a checkpoint loads from beside `config.json`.
pub const FILES: [&str; 6] = [
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
];
const TALKER_PREFIX: &str = "talker";
const CODEC_PREFIX: &str = "decoder";
// `<|im_start|>assistant\n` before the text, `<|im_end|>\n<|im_start|>assistant\n` after it
const ROLE_TOKENS: usize = 3;
const TAIL_TOKENS: usize = 5;
// the codec vocabulary's last 1024 ids are control ids the talker must not emit, EOS aside
const CONTROL_IDS: u32 = 1024;
// the reference's `min_new_tokens`
const MIN_FRAMES: usize = 2;
const LANGUAGE_AUTO: &str = "auto";
const CHINESE: &str = "chinese";
const CUSTOM_VOICE: &str = "custom_voice";
const VOICE_DESIGN: &str = "voice_design";
// the 0.6B checkpoints take no instruction
const SMALL_MODEL: &str = "0b6";

/// Sampling for the talker and the code predictor; a temperature of 0 decodes greedily.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub sub_temperature: f32,
    pub sub_top_k: usize,
    pub sub_top_p: f32,
    pub max_frames: usize,
}

impl Default for Sampling {
    // the checkpoints' generation_config.json
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            sub_temperature: 0.9,
            sub_top_k: 50,
            sub_top_p: 1.0,
            max_frames: 8192,
        }
    }
}

fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, rng: &mut Isaac64Rng) -> u32 {
    let argmax = || {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    if temperature <= 0.0 {
        return u32::try_from(argmax()).expect("vocab fits u32");
    }
    let mut ranked: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, l)| l.is_finite())
        .map(|(i, &l)| (i, l / temperature))
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    if top_k > 0 {
        ranked.truncate(top_k);
    }
    let max = ranked.first().map_or(0.0, |r| r.1);
    let mut probs: Vec<f32> = ranked.iter().map(|r| (r.1 - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= total);
    if top_p < 1.0 {
        let mut mass = 0.0;
        let keep = probs
            .iter()
            .position(|&p| {
                mass += p;
                mass >= top_p
            })
            .map_or(probs.len(), |i| i + 1);
        probs.truncate(keep);
        ranked.truncate(keep);
        let total: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);
    }
    let mut draw: f32 = rng.random();
    for (p, (i, _)) in probs.iter().zip(&ranked) {
        draw -= p;
        if draw <= 0.0 {
            return u32::try_from(*i).expect("vocab fits u32");
        }
    }
    u32::try_from(ranked.last().map_or_else(argmax, |r| r.0)).expect("vocab fits u32")
}

/// What one request asks the talker for, checked against the checkpoint.
struct Voice {
    speaker: Option<u32>,
    language: Option<u32>,
    instruction: Option<String>,
}

pub struct Qwen3Tts {
    cfg: Qwen3TtsConfig,
    talker: Talker,
    codec: CodecDecoder,
    tokenizer: Tokenizer,
    sample_rate: usize,
    device: Device,
}

impl Qwen3Tts {
    /// A checkpoint directory: `config.json`, `model.safetensors`, the tokenizer files and `speech_tokenizer/`.
    pub fn load(dir: &Path, dtype: DType, device: &Device) -> anyhow::Result<Self> {
        let cfg = Qwen3TtsConfig::from_json(&std::fs::read_to_string(dir.join(CONFIG))?)?;
        let codec_dir = dir.join(CODEC_DIR);
        let codec_cfg: CodecConfig =
            serde_json::from_str(&std::fs::read_to_string(codec_dir.join(CONFIG))?)?;
        // SAFETY: the weights are read-only for the model's life; a file changed underneath it is undefined
        let vb = unsafe {
            ShardedSafeTensors::sharded(
                &[dir.join(WEIGHTS)],
                dtype,
                device,
                None,
                Arc::new(|_| true),
            )?
        };
        // the codec is stored in f32 and the reference runs it so
        let codec_vb = unsafe {
            ShardedSafeTensors::sharded(
                &[codec_dir.join(WEIGHTS)],
                DType::F32,
                device,
                None,
                Arc::new(|name: String| name.starts_with(CODEC_PREFIX)),
            )?
        };
        Self::new(
            cfg,
            &codec_cfg,
            &vb,
            &codec_vb,
            tokenizer::load(dir)?,
            device,
        )
    }

    /// The model over its checkpoint's root (`vb`) and its codec's (`codec_vb`).
    pub fn new(
        cfg: Qwen3TtsConfig,
        codec: &CodecConfig,
        vb: &ShardedVarBuilder,
        codec_vb: &ShardedVarBuilder,
        tokenizer: Tokenizer,
        device: &Device,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            cfg.tts_model_type == CUSTOM_VOICE || cfg.tts_model_type == VOICE_DESIGN,
            "Qwen3-TTS `{}` checkpoints (voice cloning) are not supported yet",
            cfg.tts_model_type
        );
        Ok(Self {
            talker: Talker::new(&cfg.talker_config, &vb.pp(TALKER_PREFIX), device)?,
            codec: CodecDecoder::new(&codec.decoder_config, &codec_vb.pp(CODEC_PREFIX), device)?,
            tokenizer,
            sample_rate: codec.output_sample_rate,
            device: device.clone(),
            cfg,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The preset speakers, lowercased as the checkpoint names them.
    pub fn speakers(&self) -> impl Iterator<Item = &str> {
        self.cfg.talker_config.spk_id.keys().map(String::as_str)
    }

    fn voice(&self, options: &SpeechOptions) -> anyhow::Result<Voice> {
        let talker = &self.cfg.talker_config;
        let speaker_name = options.voice.as_deref().map(str::to_lowercase);
        let speaker = match (self.cfg.tts_model_type.as_str(), &speaker_name) {
            (CUSTOM_VOICE, Some(name)) => Some(*talker.spk_id.get(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown voice `{name}`; this model speaks as {}",
                    self.speakers().collect::<Vec<_>>().join(", ")
                )
            })?),
            (CUSTOM_VOICE, None) => anyhow::bail!(
                "this model needs a `voice`: {}",
                self.speakers().collect::<Vec<_>>().join(", ")
            ),
            // a voice design model is described by its instruction; OpenAI clients always send some voice
            _ => None,
        };
        let instruction = options
            .instructions
            .clone()
            .filter(|i| !i.trim().is_empty());
        if self.cfg.tts_model_type == VOICE_DESIGN && instruction.is_none() {
            anyhow::bail!("a voice design model needs `instructions` describing the voice");
        }
        if instruction.is_some() && self.cfg.tts_model_size == SMALL_MODEL {
            anyhow::bail!("the 0.6B Qwen3-TTS models take no `instructions`");
        }
        let language = options
            .language
            .as_deref()
            .map(str::to_lowercase)
            .unwrap_or_else(|| LANGUAGE_AUTO.to_string());
        let dialect = speaker_name.as_deref().and_then(|s| self.cfg.dialect(s));
        let language = match (language.as_str(), dialect) {
            (CHINESE | LANGUAGE_AUTO, Some(dialect)) => Some(dialect.to_string()),
            (LANGUAGE_AUTO, None) => None,
            (other, _) => Some(other.to_string()),
        };
        let language = language
            .map(|name| {
                talker.codec_language_id.get(&name).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "unknown language `{name}`; this model speaks {}",
                        talker
                            .codec_language_id
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
            })
            .transpose()?;
        Ok(Voice {
            speaker,
            language,
            instruction,
        })
    }

    /// Checks a request's options without generating.
    pub fn validate(&self, options: &SpeechOptions) -> anyhow::Result<()> {
        anyhow::ensure!(
            options.phonemes.is_none(),
            "Qwen3-TTS speaks text; it does not take `phonemes`"
        );
        self.voice(options).map(|_| ())
    }

    fn ids(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec())
    }

    fn tensor(&self, ids: &[u32]) -> inference_tensor::Result<Tensor> {
        Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device)
    }

    /// The talker's prompt embeddings `(1, n, hidden)`, as the reference's non-streaming text input lays them out.
    fn prompt(&self, text: &str, voice: &Voice) -> anyhow::Result<Tensor> {
        let talker = &self.cfg.talker_config;
        let ids = self.ids(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ))?;
        anyhow::ensure!(ids.len() > ROLE_TOKENS + TAIL_TOKENS, "nothing to speak");
        let specials = self.talker.text(&self.tensor(&[
            self.cfg.tts_bos_token_id,
            self.cfg.tts_eos_token_id,
            self.cfg.tts_pad_token_id,
        ])?)?;
        let (tts_bos, tts_eos, tts_pad) = (
            specials.narrow(1, 0, 1)?,
            specials.narrow(1, 1, 1)?,
            specials.narrow(1, 2, 1)?,
        );
        let mut codec_ids = match voice.language {
            Some(language) => vec![
                talker.codec_think_id,
                talker.codec_think_bos_id,
                language,
                talker.codec_think_eos_id,
            ],
            None => vec![
                talker.codec_nothink_id,
                talker.codec_think_bos_id,
                talker.codec_think_eos_id,
            ],
        };
        codec_ids.extend(voice.speaker);
        codec_ids.extend([talker.codec_pad_id, talker.codec_bos_id]);
        let codec = self.talker.codec(&self.tensor(&codec_ids)?)?;
        let n = codec_ids.len();
        // every codec tag but bos rides on tts_pad, the last of them on tts_bos
        let tags = Tensor::cat(&[tts_pad.repeat((1, n - 2, 1))?, tts_bos], 1)?;
        let tags = (tags + codec.narrow(1, 0, n - 1)?)?;
        let body = &ids[ROLE_TOKENS..ids.len() - TAIL_TOKENS];
        let text_rows = Tensor::cat(&[self.talker.text(&self.tensor(body)?)?, tts_eos], 1)?;
        let pads = self
            .talker
            .codec(&self.tensor(&vec![talker.codec_pad_id; body.len() + 1])?)?;
        let mut parts = Vec::new();
        if let Some(instruction) = &voice.instruction {
            let ids = self.ids(&format!("<|im_start|>user\n{instruction}<|im_end|>\n"))?;
            parts.push(self.talker.text(&self.tensor(&ids)?)?);
        }
        parts.push(self.talker.text(&self.tensor(&ids[..ROLE_TOKENS])?)?);
        parts.push(tags);
        parts.push((text_rows + pads)?);
        parts.push((tts_pad + codec.narrow(1, n - 1, 1)?)?);
        Ok(Tensor::cat(&parts, 1)?)
    }

    fn logits_vec(logits: &Tensor) -> inference_tensor::Result<Vec<f32>> {
        logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
    }

    /// The frames' codes `(frames, groups)`, row-major.
    fn codes(
        &self,
        prompt: Tensor,
        sampling: &Sampling,
        rng: &mut Isaac64Rng,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let talker = &self.cfg.talker_config;
        let eos = talker.codec_eos_token_id;
        let first_control = u32::try_from(talker.vocab_size)? - CONTROL_IDS;
        let predictor = &self.talker.predictor;
        let groups = predictor.groups();
        let tts_pad = self
            .talker
            .text(&self.tensor(&[self.cfg.tts_pad_token_id])?)?;
        let mut cache = self.talker.new_cache();
        let mut sub_cache: StackCache = predictor.new_cache();
        let mut offset = prompt.dim(1)?;
        let (mut hidden, mut logits) = self.talker.step(prompt, &mut cache, 0)?;
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut history: Vec<u32> = Vec::new();
        while frames.len() < sampling.max_frames {
            let mut scores = Self::logits_vec(&logits)?;
            for &token in &history {
                let s = &mut scores[token as usize];
                *s = if *s < 0.0 {
                    *s * sampling.repetition_penalty
                } else {
                    *s / sampling.repetition_penalty
                };
            }
            for (id, s) in scores.iter_mut().enumerate().skip(first_control as usize) {
                let early_eos = frames.len() < MIN_FRAMES;
                if id != eos as usize || early_eos {
                    *s = f32::NEG_INFINITY;
                }
            }
            let code0 = sample(
                &scores,
                sampling.temperature,
                sampling.top_k,
                sampling.top_p,
                rng,
            );
            if code0 == eos {
                break;
            }
            if !history.contains(&code0) {
                history.push(code0);
            }
            let code0_ids = self.tensor(&[code0])?;
            let code0_embed = self.talker.codec(&code0_ids)?;
            sub_cache.reset();
            let mut frame = vec![code0];
            let mut sum = code0_embed.clone();
            let mut input = Tensor::cat(&[hidden, code0_embed], 1)?;
            for group in 1..groups {
                let sub_offset = if group == 1 { 0 } else { group };
                let logits = predictor.step(&input, group, &mut sub_cache, sub_offset)?;
                let code = sample(
                    &Self::logits_vec(&logits)?,
                    sampling.sub_temperature,
                    sampling.sub_top_k,
                    sampling.sub_top_p,
                    rng,
                );
                frame.push(code);
                input = predictor.embed(group, &self.tensor(&[code])?)?;
                sum = (sum + &input)?;
            }
            frames.push(frame);
            let next = (sum + &tts_pad)?;
            (hidden, logits) = self.talker.step(next, &mut cache, offset)?;
            offset += 1;
        }
        Ok(frames)
    }

    /// Speaks `text`; `seed` drives both samplers.
    pub fn generate(
        &self,
        text: &str,
        options: &SpeechOptions,
        sampling: &Sampling,
        seed: u64,
    ) -> anyhow::Result<SpeechGenerationOutput> {
        let voice = self.voice(options)?;
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let frames = self.codes(self.prompt(text, &voice)?, sampling, &mut rng)?;
        let pcm = self.decode(&frames)?;
        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: self.sample_rate,
            channels: 1,
            words: None,
        })
    }

    /// Frames of codes to audio.
    pub fn decode(&self, frames: &[Vec<u32>]) -> anyhow::Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let groups = frames[0].len();
        let flat: Vec<u32> = frames.iter().flatten().copied().collect();
        let codes = Tensor::from_vec(flat, (frames.len(), groups), &self.device)?
            .t()?
            .contiguous()?;
        Ok(self.codec.decode(&codes)?)
    }

    /// The prompt's text ids, as the reference tokenizes them (for parity checks).
    pub fn prompt_ids(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        self.ids(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ))
    }

    /// Greedy codes for `text` (for parity checks).
    pub fn greedy_codes(
        &self,
        text: &str,
        options: &SpeechOptions,
        max_frames: usize,
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        let voice = self.voice(options)?;
        let sampling = Sampling {
            temperature: 0.0,
            sub_temperature: 0.0,
            repetition_penalty: 1.0,
            max_frames,
            ..Sampling::default()
        };
        let mut rng = Isaac64Rng::seed_from_u64(0);
        self.codes(self.prompt(text, &voice)?, &sampling, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_top_k_sampling() {
        let mut rng = Isaac64Rng::seed_from_u64(7);
        let logits = [0.0, 5.0, 1.0, f32::NEG_INFINITY];
        assert_eq!(sample(&logits, 0.0, 0, 1.0, &mut rng), 1);
        for _ in 0..50 {
            assert_eq!(sample(&logits, 1.0, 1, 1.0, &mut rng), 1);
            assert_ne!(sample(&logits, 1.0, 3, 1.0, &mut rng), 3);
        }
    }
}
