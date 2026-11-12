//! Cohere Transcribe: NeMo's FastConformer encoder (Parakeet's), projected into NeMo's transformer decoder (the text
//! decoder Whisper shares), prompted with the language and task flags. It does not identify the language itself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inference_audio::nemo::NemoMel;
use inference_nn::layers::Activation;
use inference_quant::{ShardedSafeTensors, ShardedVarBuilder};
use inference_tensor::nn::{Linear, Module, VarBuilder};
use inference_tensor::{DType, Device, Error, Result, Tensor};
use tokenizers::Tokenizer;

use crate::nemo::{NemoEncoderConfig, NemoPreprocessorConfig, parakeet_encoder_name};
use crate::parakeet::Encoder;
use crate::text_decoder::{Decoder, DecoderConfig, NEMO_NAMES};
use crate::{TimedText, Transcription, TranscriptionOptions};

pub const MODEL_TYPE: &str = "cohere_asr";
const SAMPLE_RATE: u32 = 16_000;
// the reference's budget per chunk of at most 35 s
const MAX_NEW_TOKENS: usize = 256;
// the feature extractor's `min_energy_window_samples`: the quietest window it splits long audio at
const ENERGY_WINDOW: usize = 1600;
const PROJECTION: &str = "encoder_decoder_proj";
const HEAD: &str = "log_softmax.mlp.layer0";
const DECODER: &str = "transf_decoder";
const ENCODER: &str = "encoder";
const EOS: &str = "<|endoftext|>";
// the decoder prompt after the language: punctuation on, no inverse normalization, timestamps or diarization
const PROMPT_START: [&str; 4] = [
    "\u{2581}",
    "<|startofcontext|>",
    "<|startoftranscript|>",
    "<|emo:undefined|>",
];
const PROMPT_FLAGS: [&str; 4] = ["<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"];
// languages written without spaces between words: chunk texts join with none
const NO_SPACE_LANGUAGES: [&str; 2] = ["ja", "zh"];

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

/// Whether a `config.json` is Cohere Transcribe's.
pub fn is_cohere_asr_config(config: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(config)
        .ok()
        .and_then(|v| v.get("model_type")?.as_str().map(|t| t == MODEL_TYPE))
        .unwrap_or(false)
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct DecoderDict {
    hidden_size: usize,
    inner_size: usize,
    num_attention_heads: usize,
    num_layers: usize,
    max_sequence_length: usize,
    hidden_act: Activation,
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct TransfDecoder {
    config_dict: DecoderDict,
}

#[derive(Debug, Clone, inference_json::FromJson)]
struct HeadConfig {
    num_classes: usize,
}

/// The release's `config.json`, the fields the model reads.
#[derive(Debug, Clone, inference_json::FromJson)]
pub struct CohereAsrConfig {
    encoder: NemoEncoderConfig,
    preprocessor: NemoPreprocessorConfig,
    transf_decoder: TransfDecoder,
    head: HeadConfig,
    supported_languages: Vec<String>,
    max_audio_clip_s: f64,
    overlap_chunk_second: f64,
}

#[derive(Clone, Debug)]
pub struct CohereAsrFiles {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

pub struct CohereAsr {
    mel: NemoMel,
    encoder: Encoder,
    projection: Option<Linear>,
    decoder: Decoder,
    tokenizer: Tokenizer,
    languages: Vec<String>,
    prompt_start: Vec<u32>,
    prompt_flags: Vec<u32>,
    eos: u32,
    chunk_samples: usize,
    search_samples: usize,
    max_positions: usize,
    device: Device,
    dtype: DType,
}

// the quietest `ENERGY_WINDOW` start in `pcm[start..end]` by RMS, or the middle of a span shorter than one window
fn quietest(pcm: &[f32], start: usize, end: usize) -> usize {
    let segment = &pcm[start..end];
    if segment.len() <= ENERGY_WINDOW {
        return (start + end) / 2;
    }
    let mut best = (f32::INFINITY, start);
    for i in (0..segment.len() - ENERGY_WINDOW).step_by(ENERGY_WINDOW) {
        let window = &segment[i..i + ENERGY_WINDOW];
        #[allow(clippy::cast_precision_loss)]
        let energy = (window.iter().map(|s| s * s).sum::<f32>() / ENERGY_WINDOW as f32).sqrt();
        if energy < best.0 {
            best = (energy, start + i);
        }
    }
    best.1
}

/// The reference's chunks of long audio: `chunk` samples at most, each cut at the quietest point of its last
/// `search` samples; audio within one chunk is one.
fn chunks(pcm: &[f32], chunk: usize, search: usize) -> Vec<(usize, usize)> {
    let total = pcm.len();
    if total <= chunk {
        return vec![(0, total)];
    }
    let mut spans = Vec::new();
    let mut at = 0;
    while at < total {
        if at + chunk >= total {
            spans.push((at, total));
            break;
        }
        let from = at.max(at + chunk - search);
        let to = (at + chunk).min(total);
        let split = if to <= from {
            at + chunk
        } else {
            quietest(pcm, from, to)
        };
        let split = split.clamp(at + 1, total);
        spans.push((at, split));
        at = split;
    }
    spans
}

impl CohereAsr {
    pub fn load(files: &CohereAsrFiles, device: &Device, dtype: DType) -> Result<Self> {
        let config: CohereAsrConfig =
            serde_json::from_str(&std::fs::read_to_string(&files.config)?).map_err(msg)?;
        let mut tensors = HashMap::new();
        for file in &files.weights {
            for (name, tensor) in inference_tensor::safetensors::load(file, &Device::Cpu)? {
                tensors.insert(parakeet_encoder_name(&name), tensor);
            }
        }
        let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(msg)?;
        let encoder_vb = VarBuilder::from_tensors(tensors.clone(), dtype, device);
        let vb = ShardedSafeTensors::wrap(tensors, dtype, device.clone());
        Self::new(&config, tokenizer, encoder_vb, vb)
    }

    /// `encoder_vb` holds the encoder under the transformers names Parakeet's encoder reads, `vb` the rest under the
    /// checkpoint's own; both over the same weights.
    pub fn new(
        config: &CohereAsrConfig,
        tokenizer: Tokenizer,
        encoder_vb: VarBuilder,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let (device, dtype) = (vb.device().clone(), vb.dtype());
        let encoder_config = config.encoder.parakeet().map_err(msg)?;
        let d = &config.transf_decoder.config_dict;
        let decoder_config = DecoderConfig {
            d_model: d.hidden_size,
            num_layers: d.num_layers,
            num_heads: d.num_attention_heads,
            ffn_dim: d.inner_size,
            vocab_size: config.head.num_classes,
            max_positions: d.max_sequence_length,
            activation: d.hidden_act,
            key_bias: true,
            names: &NEMO_NAMES,
            model: "Cohere Transcribe",
        };
        let linear = |in_dim: usize, out_dim: usize, name: &str| -> Result<Linear> {
            let vb = vb.pp(name);
            Ok(Linear::new(
                vb.get((out_dim, in_dim), "weight")?,
                Some(vb.get(out_dim, "bias")?),
            ))
        };
        let projection = (encoder_config.hidden_size != d.hidden_size)
            .then(|| linear(encoder_config.hidden_size, d.hidden_size, PROJECTION))
            .transpose()?;
        let head = linear(d.hidden_size, config.head.num_classes, HEAD)?;
        let id = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| msg(format!("the tokenizer has no `{token}`")))
        };
        let ids = |tokens: &[&str]| tokens.iter().map(|t| id(t)).collect::<Result<Vec<_>>>();
        let seconds = |s: f64| (s * f64::from(SAMPLE_RATE)).round() as usize;
        Ok(Self {
            mel: NemoMel::new(config.preprocessor.mel().map_err(msg)?),
            encoder: Encoder::new(&encoder_config, encoder_vb.pp(ENCODER))?,
            projection,
            decoder: Decoder::new(&decoder_config, vb.pp(DECODER), Some(head))?,
            prompt_start: ids(&PROMPT_START)?,
            prompt_flags: ids(&PROMPT_FLAGS)?,
            eos: id(EOS)?,
            languages: config.supported_languages.clone(),
            chunk_samples: seconds(config.max_audio_clip_s),
            search_samples: seconds(config.overlap_chunk_second),
            max_positions: d.max_sequence_length,
            tokenizer,
            device,
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    // the request's language as one of the model's codes
    fn language(&self, options: &TranscriptionOptions) -> Result<String> {
        let known = || self.languages.join(", ");
        let Some(language) = options.language.as_deref() else {
            return Err(msg(format!(
                "Cohere Transcribe does not identify the language; name one of {} (or a `language_model`)",
                known()
            )));
        };
        let code = language.split(['-', '_']).next().unwrap_or(language);
        self.languages
            .iter()
            .find(|l| l.eq_ignore_ascii_case(code))
            .cloned()
            .ok_or_else(|| {
                let how = if options.language_identified {
                    "the identified language"
                } else {
                    "language"
                };
                msg(format!(
                    "Cohere Transcribe does not transcribe {how} `{language}`; it takes {}",
                    known()
                ))
            })
    }

    /// The decoder prompt for `language`, one of the model's codes.
    pub fn prompt(&self, language: &str) -> Result<Vec<u32>> {
        let token = format!("<|{language}|>");
        let id = self
            .tokenizer
            .token_to_id(&token)
            .ok_or_else(|| msg(format!("the tokenizer has no `{token}`")))?;
        Ok([self.prompt_start.as_slice(), &[id, id], &self.prompt_flags].concat())
    }

    /// `(frames, n_mels)` row-major log-mel features of 16 kHz `pcm`, and the frame count.
    pub fn features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        self.mel.features(pcm).map_err(msg)
    }

    /// The encoder's `(1, frames, hidden)` output for 16 kHz `pcm`, before the projection into the decoder.
    pub fn encode(&self, pcm: &[f32]) -> Result<Tensor> {
        let (features, frames) = self.features(pcm)?;
        let n_mels = self.mel.config().n_mels;
        let features =
            Tensor::from_vec(features, (1, frames, n_mels), &self.device)?.to_dtype(self.dtype)?;
        self.encoder.forward(features)
    }

    /// The `(start, end)` sample spans 16 kHz `pcm` is decoded in.
    pub fn chunks(&self, pcm: &[f32]) -> Vec<(usize, usize)> {
        chunks(pcm, self.chunk_samples, self.search_samples)
    }

    /// Greedy token ids after `prompt` for one chunk of 16 kHz `pcm`, end of text excluded.
    pub fn decode_chunk(&self, pcm: &[f32], prompt: &[u32]) -> Result<Vec<u32>> {
        let encoded = self.encode(pcm)?;
        let encoded = match &self.projection {
            Some(projection) => projection.forward(&encoded)?,
            None => encoded,
        };
        let mut cache = self.decoder.cache(&encoded)?;
        let mut step = prompt.to_vec();
        let mut ids = Vec::new();
        // the position table bounds a chunk's tokens as well as the reference's budget
        let budget = MAX_NEW_TOKENS.min(self.max_positions.saturating_sub(prompt.len()));
        while ids.len() < budget {
            let out = self.decoder.forward(&step, &mut cache, false)?;
            let logits = Decoder::last_logits(&out)?;
            let token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| u32::try_from(i).expect("a vocabulary id"))
                .expect("a vocabulary");
            if token == self.eos {
                break;
            }
            ids.push(token);
            step = vec![token];
        }
        Ok(ids)
    }

    /// Transcribes mono `pcm` at `sample_rate`; audio past a chunk (35 s) is cut at quiet points into chunks, each
    /// decoded alone and their texts joined.
    pub fn transcribe(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        options: &TranscriptionOptions,
    ) -> Result<Transcription> {
        if options.translate {
            return Err(msg("Cohere Transcribe transcribes; it does not translate"));
        }
        if options.beam_size.is_some_and(|b| b > 1) {
            return Err(msg(
                "Cohere Transcribe decodes greedily; it has no beam search",
            ));
        }
        let language = self.language(options)?;
        let prompt = self.prompt(&language)?;
        let pcm = inference_audio::mel::resample(pcm, sample_rate, SAMPLE_RATE).map_err(msg)?;
        let mut texts = Vec::new();
        for (start, end) in self.chunks(&pcm) {
            let ids = self.decode_chunk(&pcm[start..end], &prompt)?;
            let text = self.tokenizer.decode(&ids, true).map_err(msg)?;
            if !text.trim().is_empty() {
                texts.push(text.trim().to_string());
            }
        }
        let separator = if NO_SPACE_LANGUAGES.contains(&language.as_str()) {
            ""
        } else {
            " "
        };
        Ok(Transcription {
            segments: Vec::new(),
            text: texts.join(separator),
            tokens: Vec::<TimedText>::new(),
            words: Vec::new(),
            duration: pcm.len() as f64 / f64::from(SAMPLE_RATE),
            language: Some(language),
        })
    }
}

/// The files of a transformers-layout checkpoint directory.
pub fn files_in(dir: &Path) -> CohereAsrFiles {
    CohereAsrFiles {
        config: dir.join("config.json"),
        tokenizer: dir.join("tokenizer.json"),
        weights: vec![dir.join("model.safetensors")],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_audio_splits_at_its_quietest_window() {
        // 10 units of chunk, 3 of search: a silent window at 8 is where the first chunk ends
        let unit = ENERGY_WINDOW;
        let mut pcm = vec![0.5f32; 25 * unit];
        pcm[8 * unit..9 * unit].fill(0.0);
        let spans = chunks(&pcm, 10 * unit, 3 * unit);
        assert_eq!(spans[0], (0, 8 * unit));
        assert_eq!(spans.last().unwrap().1, pcm.len());
        assert!(spans.windows(2).all(|w| w[0].1 == w[1].0));
        assert_eq!(
            chunks(&pcm[..5 * unit], 10 * unit, 3 * unit),
            [(0, 5 * unit)]
        );
    }
}
