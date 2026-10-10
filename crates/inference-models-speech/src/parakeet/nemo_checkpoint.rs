//! Parakeet-family recognisers from their `.nemo`: NeMo's ASR config read into transformers' Parakeet config, the
//! weights renamed onto its names, and a decoding tokenizer from the archive's SentencePiece model.

use std::collections::HashMap;
use std::path::Path;

use inference_tensor::{DType, Device, Error, Result};
use serde::Deserialize;

use super::Parakeet;
use super::config::{FeatureConfig, ParakeetConfig, ProcessorConfig};
use crate::nemo::{
    NemoArchive, NemoEncoderConfig, NemoPreprocessorConfig, parakeet_name, piece_tokenizer,
    sentencepiece_pieces,
};

const RNNT_DECODER: &str = "RNNTDecoder";
const CTC_DECODER: &str = "ConvASRDecoder";
// NeMo's tokenizer kinds: SentencePiece BPE is the one this decodes
const SENTENCEPIECE: &str = "bpe";
// the prompted 3.5 class, whose prompt module this does not map
const PROMPTED_CLASS: &str = "WithPrompt";
const DEFAULT_MAX_SYMBOLS: usize = 10;
// NeMo's ASR model classes all open `EncDec`; Sortformer's does not
const ASR_CLASS: &str = ".EncDec";
const CONFORMER: &str = "ConformerEncoder";

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Msg(e.to_string())
}

#[derive(Debug, Clone, Deserialize)]
struct Prednet {
    pred_hidden: usize,
    pred_rnn_layers: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct DecoderConfig {
    #[serde(rename = "_target_")]
    target: String,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    num_classes: Option<i64>,
    #[serde(default)]
    prednet: Option<Prednet>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct GreedyConfig {
    #[serde(default)]
    max_symbols: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct DecodingConfig {
    #[serde(default)]
    durations: Option<Vec<usize>>,
    /// Multi-blank RNN-T's extra blanks, which this does not decode.
    #[serde(default)]
    big_blank_durations: Option<Vec<usize>>,
    #[serde(default)]
    greedy: Option<GreedyConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct TokenizerConfig {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    model_path: Option<String>,
}

/// The parts of a NeMo ASR `model_config.yaml` a Parakeet-family model needs.
#[derive(Debug, Clone, Deserialize)]
struct AsrConfig {
    target: String,
    preprocessor: NemoPreprocessorConfig,
    encoder: NemoEncoderConfig,
    decoder: DecoderConfig,
    #[serde(default)]
    decoding: Option<DecodingConfig>,
    tokenizer: TokenizerConfig,
}

impl AsrConfig {
    // transformers' `model_type`, vocabulary (blank included) and TDT durations
    fn head(&self) -> Result<(&'static str, usize, Vec<usize>)> {
        let d = &self.decoder;
        if d.target.ends_with(CTC_DECODER) {
            let classes = d
                .num_classes
                .filter(|&c| c > 0)
                .ok_or_else(|| msg("a CTC decoder without `num_classes`"))?;
            return Ok(("parakeet_ctc", classes as usize + 1, Vec::new()));
        }
        if !d.target.ends_with(RNNT_DECODER) {
            return Err(msg(format!(
                "a `{}` decoder is not one this loads",
                d.target
            )));
        }
        let vocab = d
            .vocab_size
            .ok_or_else(|| msg("an RNN-T decoder without `vocab_size`"))?
            + 1;
        let decoding = self.decoding.clone().unwrap_or_default();
        if decoding.big_blank_durations.is_some_and(|d| !d.is_empty()) {
            return Err(msg("a multi-blank RNN-T is not one this decodes"));
        }
        // NeMo's decoder takes a model as TDT exactly when its decoding names durations
        match decoding.durations.filter(|d| !d.is_empty()) {
            Some(durations) => Ok(("parakeet_tdt", vocab, durations)),
            None => Ok(("parakeet_rnnt", vocab, Vec::new())),
        }
    }
}

#[derive(Deserialize)]
struct ModuleTarget {
    #[serde(rename = "_target_")]
    target: Option<String>,
}

#[derive(Deserialize)]
struct Probe {
    target: Option<String>,
    encoder: Option<ModuleTarget>,
    decoder: Option<ModuleTarget>,
    tokenizer: Option<serde::de::IgnoredAny>,
}

/// Whether `archive` is a recogniser this loads: an ASR class over a Conformer with an RNN-T or CTC decoder.
pub fn is_asr(archive: &NemoArchive) -> Result<bool> {
    let probe: Probe = archive.config()?;
    let module = |m: Option<ModuleTarget>| m.and_then(|m| m.target).unwrap_or_default();
    let decoder = module(probe.decoder);
    Ok(probe.target.is_some_and(|t| t.contains(ASR_CLASS))
        && probe.tokenizer.is_some()
        && module(probe.encoder).ends_with(CONFORMER)
        && (decoder.ends_with(RNNT_DECODER) || decoder.ends_with(CTC_DECODER)))
}

impl Parakeet {
    /// A Parakeet, hybrid or streaming recogniser from its `.nemo`.
    pub fn load_nemo(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let archive = NemoArchive::open(path)?;
        let nemo: AsrConfig = archive.config()?;
        if nemo.target.contains(PROMPTED_CLASS) {
            return Err(msg(
                "a prompted `.nemo` (Nemotron-3.5) loads from its transformers layout instead",
            ));
        }
        let (model_type, vocab_size, durations) = nemo.head()?;
        let encoder_config = nemo.encoder.parakeet()?;
        let streaming = encoder_config.is_streaming();
        let mel = nemo.preprocessor.mel()?;
        if mel.normalize == streaming {
            return Err(msg(
                "a streaming encoder takes raw log-mel features and an offline one normalized ones",
            ));
        }
        let prednet = nemo.decoder.prednet.as_ref();
        let max_symbols = nemo
            .decoding
            .as_ref()
            .and_then(|d| d.greedy.as_ref()?.max_symbols)
            .unwrap_or(DEFAULT_MAX_SYMBOLS);
        let config = ParakeetConfig {
            model_type: model_type.to_string(),
            encoder_config,
            vocab_size,
            pad_token_id: None,
            blank_token_id: None,
            decoder_hidden_size: prednet.map(|p| p.pred_hidden),
            num_decoder_layers: prednet.map(|p| p.pred_rnn_layers),
            durations,
            max_symbols_per_step: max_symbols,
            num_prompts: None,
            prompt_intermediate_size: None,
            default_prompt_id: None,
        };
        let processor = ProcessorConfig {
            feature_extractor: FeatureConfig {
                feature_size: mel.n_mels,
                sampling_rate: mel.sample_rate,
                n_fft: mel.n_fft,
                win_length: mel.win_length,
                hop_length: mel.hop_length,
                preemphasis: mel.preemphasis,
            },
            prompt_dictionary: HashMap::new(),
        };
        let model_path = match (&nemo.tokenizer.kind, &nemo.tokenizer.model_path) {
            (kind, Some(path)) if kind == SENTENCEPIECE => path,
            (kind, _) => {
                return Err(msg(format!(
                    "a `{kind}` tokenizer is not one this decodes; SentencePiece BPE is"
                )));
            }
        };
        let pieces = sentencepiece_pieces(&archive.member(model_path)?)?;
        let tokenizer = piece_tokenizer(&pieces)?;
        let vb = archive.var_builder(|name| parakeet_name(name, streaming), dtype, device)?;
        Self::new(config, processor, tokenizer, vb)
    }
}
