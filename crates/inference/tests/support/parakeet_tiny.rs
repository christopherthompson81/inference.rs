//! Builds a tiny random-weight Parakeet checkpoint (safetensors, configs, tokenizer) for any of its heads at test time.
#![allow(dead_code)]

use std::path::Path;

use inference_models_speech::parakeet::{Parakeet, ParakeetConfig, ProcessorConfig};
use inference_tensor::Device;

#[path = "recording.rs"]
mod recording;

// through the crates directory, so the server crate's tests resolve it too
const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/parakeet_tiny"
);
const CONFIG: &str = "config.json";
const PROCESSOR_CONFIG: &str = "processor_config.json";
const TOKENIZER: &str = "tokenizer.json";
const WEIGHTS: &str = "model.safetensors";
const RUNNING_VAR: &str = "running_var";
// the tiny tokenizer's `<blank>`, which a CTC checkpoint names as its pad token
const BLANK_ID: u64 = 32;

/// The heads a Parakeet checkpoint can carry, as `model_type`s, then the streaming Nemotron ASR ones (RNN-T over a
/// chunked-limited causal encoder; 3.5 also takes a language prompt).
pub const HEADS: [&str; 5] = [
    "parakeet_tdt",
    "parakeet_rnnt",
    "parakeet_ctc",
    "nemotron_asr_streaming",
    "nemotron3_5_asr",
];
pub const PROMPTED: &str = "nemotron3_5_asr";
// the tiny streaming encoder: 4 frames back and a 2-frame chunk, so a few seconds cross many chunk boundaries
const STREAMING_WINDOW: u64 = 5;
const STREAMING_LOOKAHEAD: u64 = 1;
const PROMPTS: u64 = 4;
const PROMPT_INNER: u64 = 8;
/// The tiny 3.5 checkpoint's languages and its auto prompt.
pub const LANGUAGES: [(&str, u64); 3] = [("en-US", 0), ("de-DE", 1), ("auto", 3)];

/// The committed fixtures with `model_type` set to `head`, and random weights for every tensor the model loads.
pub fn tiny_parakeet_checkpoint(head: &str) -> anyhow::Result<tempfile::TempDir> {
    let fixtures = Path::new(FIXTURES);
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join(CONFIG))?)?;
    config["model_type"] = head.into();
    if head == "parakeet_ctc" {
        config["pad_token_id"] = BLANK_ID.into();
    }
    if head != "parakeet_tdt" {
        config["durations"] = serde_json::json!([]);
    }
    let streaming = head.starts_with("nemotron");
    if streaming {
        let encoder = &mut config["encoder_config"];
        encoder["model_type"] = "nemotron_asr_streaming_encoder".into();
        encoder["sliding_window"] = STREAMING_WINDOW.into();
        encoder["default_num_lookahead_tokens"] = STREAMING_LOOKAHEAD.into();
    }
    let mut processor_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join(PROCESSOR_CONFIG))?)?;
    if head == PROMPTED {
        config["num_prompts"] = PROMPTS.into();
        config["prompt_intermediate_size"] = PROMPT_INNER.into();
        config["default_prompt_id"] = LANGUAGES[2].1.into();
        processor_json["prompt_dictionary"] = LANGUAGES
            .iter()
            .map(|(l, id)| (l.to_string(), serde_json::Value::from(*id)))
            .collect::<serde_json::Map<_, _>>()
            .into();
    }
    let staged = tempfile::tempdir()?;
    let staged_config = staged.path().join(CONFIG);
    std::fs::write(&staged_config, serde_json::to_string_pretty(&config)?)?;
    let staged_processor = staged.path().join(PROCESSOR_CONFIG);
    std::fs::write(
        &staged_processor,
        serde_json::to_string_pretty(&processor_json)?,
    )?;
    let processor: ProcessorConfig = serde_json::from_value(processor_json)?;
    let parsed: ParakeetConfig = serde_json::from_value(config)?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(fixtures.join(TOKENIZER)).map_err(anyhow::Error::msg)?;
    let dir = recording::record_plain_checkpoint(
        &[
            staged_config.as_path(),
            &staged_processor,
            &fixtures.join(TOKENIZER),
        ],
        |vb| Parakeet::new(parsed, processor, tokenizer, vb).map(|_| ()),
    )?;
    // drawn from a normal, a batch norm's variance would be negative half the time
    let weights = dir.path().join(WEIGHTS);
    let mut tensors = inference_tensor::safetensors::load(&weights, &Device::Cpu)?;
    for (name, t) in tensors.iter_mut() {
        if name.ends_with(RUNNING_VAR) {
            *t = (t.abs()? + 1.0)?;
        }
    }
    inference_tensor::safetensors::save(&tensors, &weights)?;
    Ok(dir)
}

const NEMO_CHECKPOINT: &str = "tiny_parakeet.nemo";
const NEMO_TOKENIZER: &str = "tiny_tokenizer.model";

// SentencePiece's unknown and control piece types, as the tiny tokenizer's `<unk>` and `<pad>` are special
const PIECE_TYPES: [(&str, u8); 2] = [("<unk>", 2), ("<pad>", 3)];
const PIECE_NORMAL: u8 = 1;

// a SentencePiece ModelProto holding `pieces`: field 1 per piece, its text in field 1 and its type in field 3
fn sentencepiece_model(pieces: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    for piece in pieces {
        let kind = PIECE_TYPES
            .iter()
            .find(|(p, _)| p == piece)
            .map_or(PIECE_NORMAL, |(_, k)| *k);
        let mut inner = vec![0x0a, piece.len() as u8];
        inner.extend_from_slice(piece.as_bytes());
        inner.extend_from_slice(&[0x18, kind]);
        out.extend_from_slice(&[0x0a, inner.len() as u8]);
        out.extend(inner);
    }
    out
}

// the tiny transformers config as NeMo's `model_config.yaml` lays it out
fn nemo_yaml(config: &serde_json::Value, head: &str) -> String {
    let enc = &config["encoder_config"];
    let num = |v: &serde_json::Value| v.as_u64().expect("a number");
    let streaming = head.starts_with("nemotron");
    let (target, decoder) = if head == "parakeet_ctc" {
        (
            "ctc_bpe_models.EncDecCTCModelBPE",
            format!(
                "  _target_: nemo.collections.asr.modules.ConvASRDecoder\n  num_classes: {}\n",
                num(&config["vocab_size"]) - 1
            ),
        )
    } else {
        (
            "rnnt_bpe_models.EncDecRNNTBPEModel",
            format!(
                "  _target_: nemo.collections.asr.modules.RNNTDecoder\n  vocab_size: {}\n  prednet:\n    \
                 pred_hidden: {}\n    pred_rnn_layers: {}\n",
                num(&config["vocab_size"]) - 1,
                num(&config["decoder_hidden_size"]),
                num(&config["num_decoder_layers"])
            ),
        )
    };
    let decoding = if head == "parakeet_tdt" {
        format!(
            "decoding:\n  model_type: tdt\n  durations: {}\n",
            config["durations"]
        )
    } else {
        String::new()
    };
    let streaming_encoder = if streaming {
        format!(
            "  causal_downsampling: true\n  att_context_style: chunked_limited\n  att_context_size:\n  - - {}\n    - {}\n  \
             conv_norm_type: layer_norm\n  conv_context_size: causal\n",
            STREAMING_WINDOW - 1,
            STREAMING_LOOKAHEAD
        )
    } else {
        String::new()
    };
    format!(
        "target: nemo.collections.asr.models.{target}\n\
         tokenizer:\n  type: bpe\n  model_path: nemo:{NEMO_TOKENIZER}\n\
         preprocessor:\n  sample_rate: 16000\n  window_size: 0.025\n  window_stride: 0.01\n  features: {mels}\n  \
         n_fft: 512\n  normalize: {normalize}\n\
         encoder:\n  _target_: nemo.collections.asr.modules.ConformerEncoder\n  feat_in: {mels}\n  n_layers: {layers}\n  \
         d_model: {hidden}\n  n_heads: {heads}\n  ff_expansion_factor: {ff}\n  subsampling: dw_striding\n  \
         subsampling_factor: {factor}\n  subsampling_conv_channels: {channels}\n  self_attention_model: rel_pos\n  \
         conv_kernel_size: {kernel}\n  xscaling: false\n  use_bias: false\n{streaming_encoder}\
         decoder:\n{decoder}{decoding}",
        mels = num(&enc["num_mel_bins"]),
        normalize = if streaming { "NA" } else { "per_feature" },
        layers = num(&enc["num_hidden_layers"]),
        hidden = num(&enc["hidden_size"]),
        heads = num(&enc["num_attention_heads"]),
        ff = num(&enc["intermediate_size"]) / num(&enc["hidden_size"]),
        factor = num(&enc["subsampling_factor"]),
        channels = num(&enc["subsampling_conv_channels"]),
        kernel = num(&enc["conv_kernel_size"]),
    )
}

/// The tiny checkpoint for `head` (not the prompted one) and the same model as a `.nemo` beside it: NeMo's config,
/// its weights under NeMo's names, and a SentencePiece model of the tokenizer's pieces.
pub fn tiny_parakeet_nemo(head: &str) -> anyhow::Result<(tempfile::TempDir, std::path::PathBuf)> {
    let dir = tiny_parakeet_checkpoint(head)?;
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join(CONFIG))?)?;
    let tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join(TOKENIZER))?)?;
    let mut vocab: Vec<(String, u64)> = tokenizer["model"]["vocab"]
        .as_object()
        .expect("a BPE vocab")
        .iter()
        .map(|(piece, id)| (piece.clone(), id.as_u64().expect("an id")))
        .collect();
    vocab.sort_by_key(|(_, id)| *id);
    let pieces: Vec<String> = vocab.into_iter().map(|(piece, _)| piece).collect();
    let streaming = head.starts_with("nemotron");
    let tensors = inference_tensor::safetensors::load(dir.path().join(WEIGHTS), &Device::Cpu)?;
    let mut renamed: Vec<(String, inference_tensor::Tensor)> = tensors
        .into_iter()
        .map(|(n, t)| (inference_models_speech::nemo::nemo_name(&n, streaming), t))
        .collect();
    renamed.sort_by(|a, b| a.0.cmp(&b.0));
    let named: Vec<(&str, &inference_tensor::Tensor)> =
        renamed.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let nemo = dir.path().join(NEMO_CHECKPOINT);
    let model = sentencepiece_model(&pieces);
    inference_models_speech::nemo::write_nemo(
        &nemo,
        &nemo_yaml(&config, head),
        &named,
        &[(NEMO_TOKENIZER, model.as_slice())],
    )?;
    Ok((dir, nemo))
}
