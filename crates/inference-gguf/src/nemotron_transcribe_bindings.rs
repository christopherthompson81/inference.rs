//! Nemotron 3.5 Transcribe from a `nemotron_h_moe` text GGUF beside a `nemotron_transcribe` audio GGUF: the decoder
//! through the Nemotron-H bindings under the model's `llm.` names, the audio tensors (`perception.*`) under their
//! checkpoint names, and the checkpoint's `config.json` carried in the audio GGUF's metadata.

use std::path::Path;

use anyhow::{Context, Result};
use inference_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};
use inference_tensor::quantized::gguf_file::Value;

use super::multimodal_binding_utils::{metadata_string, validate_projector};
use super::normal_registry::CanonicalGgufArchitecture;

pub const PROJECTOR: &str = "nemotron_transcribe";
pub const TEXT_ARCHITECTURE: &str = "nemotron_h_moe";
/// The audio GGUF's copy of the checkpoint's `config.json`.
pub const CONFIG_KEY: &str = "nemotron_transcribe.config";
/// The audio GGUF's copy of the checkpoint's `chat_template.jinja`, which places audio parts; the text GGUF's
/// (llama.cpp's converter takes `tokenizer_config.json`'s) renders content as a string.
pub const CHAT_TEMPLATE_KEY: &str = "nemotron_transcribe.chat_template";
/// The checkpoint tensors the audio GGUF holds: the speech encoder, speaker branch, preprocessor and projector.
pub const AUDIO_PREFIX: &str = "perception.";
const AUDIO_TOKEN: &str = "<|audio|>";

// Nemotron-H's transformers names to the transcription model's (NeMo's ends, under `llm.`)
const RENAMES: [(&str, &str); 3] = [
    ("model.embeddings.weight", "llm.model.embed_tokens.weight"),
    ("model.norm_f.weight", "llm.model.norm.weight"),
    ("lm_head.weight", "llm.lm_head.weight"),
];
const TEXT_ROOT: &str = "model.";
// llama.cpp's architecture for a projector file
const PROJECTOR_ARCHITECTURE: &str = "clip";
const CHECKPOINT_WEIGHTS: &str = "model.safetensors";
const CHECKPOINT_CONFIG: &str = "config.json";
const CHECKPOINT_TEMPLATE: &str = "chat_template.jinja";
const LLM_ROOT: &str = "llm.model.";

pub fn build_nemotron_transcribe_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_projector(archive, PROJECTOR)?;
    let text = super::nemotron_h_bindings::build_nemotron_h_bindings(
        archive,
        CanonicalGgufArchitecture::NemotronHMoe,
    )?;
    let mut bindings = GgufBindingMap::new();
    for (native, binding) in text.iter() {
        let renamed = RENAMES
            .iter()
            .find(|(from, _)| *from == native)
            .map(|(_, to)| to.to_string())
            .or_else(|| {
                native
                    .strip_prefix(TEXT_ROOT)
                    .map(|rest| format!("{LLM_ROOT}{rest}"))
            })
            .with_context(|| format!("Nemotron Transcribe: no `llm.` name for `{native}`"))?;
        bindings.insert(renamed, binding.clone());
    }
    for name in archive.tensors().keys() {
        if name.starts_with(AUDIO_PREFIX) {
            bindings.insert(name.clone(), GgufTensorBinding::tensor(name.clone()));
        }
    }
    Ok(bindings)
}

pub fn nemotron_transcribe_chat_template(archive: &GgufArchive) -> Result<Option<String>> {
    Ok(metadata_string(archive, CHAT_TEMPLATE_KEY)?.map(str::to_string))
}

pub fn nemotron_transcribe_config(archive: &GgufArchive) -> Result<String> {
    metadata_string(archive, CONFIG_KEY)?
        .map(str::to_string)
        .with_context(|| format!("Nemotron Transcribe audio GGUF has no `{CONFIG_KEY}`"))
}

/// The audio GGUF beside a `nemotron_h_moe` text GGUF, from NVIDIA's checkpoint directory: its `perception.*` tensors
/// as they are and its `config.json`. Returns the number of tensors written.
pub fn write_nemotron_transcribe_audio_gguf(checkpoint: &Path, out: &Path) -> Result<usize> {
    let config = std::fs::read_to_string(checkpoint.join(CHECKPOINT_CONFIG))
        .with_context(|| format!("reading {}", checkpoint.join(CHECKPOINT_CONFIG).display()))?;
    let template = std::fs::read_to_string(checkpoint.join(CHECKPOINT_TEMPLATE))
        .with_context(|| format!("reading {}", checkpoint.join(CHECKPOINT_TEMPLATE).display()))?;
    let values = [
        Value::String(PROJECTOR_ARCHITECTURE.to_string()),
        Value::String(PROJECTOR.to_string()),
        Value::Bool(true),
        Value::String(config),
        Value::String(template),
    ];
    let metadata = [
        ("general.architecture", &values[0]),
        (super::multimodal_binding_utils::PROJECTOR_TYPE, &values[1]),
        ("clip.has_audio_encoder", &values[2]),
        (CONFIG_KEY, &values[3]),
        (CHAT_TEMPLATE_KEY, &values[4]),
    ];
    let written = inference_quant::requantize::write_safetensors_gguf(
        &checkpoint.join(CHECKPOINT_WEIGHTS),
        out,
        |name| name.starts_with(AUDIO_PREFIX),
        &metadata,
    )?;
    if written == 0 {
        anyhow::bail!("{}: no `{AUDIO_PREFIX}*` tensors", checkpoint.display());
    }
    Ok(written)
}

/// The audio placeholder, which llama.cpp's converter leaves out (it is the first id past the text vocabulary).
pub fn ensure_nemotron_transcribe_audio_token(
    tokenizer: &mut tokenizers::Tokenizer,
    config: &str,
) -> Result<()> {
    let audio_token_id = serde_json::from_str::<serde_json::Value>(config)?
        .get("audio_token_id")
        .and_then(serde_json::Value::as_u64)
        .context("Nemotron Transcribe config has no `audio_token_id`")?;
    if tokenizer.token_to_id(AUDIO_TOKEN).is_none() {
        tokenizer
            .add_special_tokens([tokenizers::AddedToken::from(AUDIO_TOKEN, true)])
            .map_err(anyhow::Error::msg)?;
    }
    anyhow::ensure!(
        tokenizer.token_to_id(AUDIO_TOKEN).map(u64::from) == Some(audio_token_id),
        "Nemotron Transcribe tokenizer maps `{AUDIO_TOKEN}` to {:?}, but the config's `audio_token_id` is \
         {audio_token_id}",
        tokenizer.token_to_id(AUDIO_TOKEN)
    );
    Ok(())
}
