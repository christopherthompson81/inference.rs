//! Builds a tiny random-weight Voxtral Realtime checkpoint at test time, in Mistral's consolidated layout or
//! transformers'.
#![allow(dead_code)]

use std::collections::HashMap;

use inference_models_llama::voxtral::{VoxtralModel, config::VoxtralConfig};
use inference_nn::paged_attention::AttentionImplementation;
use inference_tensor::Tensor;

#[path = "recording.rs"]
mod recording;

pub const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/voxtral/tiny"
);
// the tiny Voxtral's byte-level vocabulary, which holds the ids the Realtime prompt uses (BOS 1, [STREAMING_PAD] 32)
pub const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../inference/tests/fixtures/voxtral_chat/tiny/tokenizer.json"
);
const ENCODER: &str = "mm_streams_embeddings.embedding_module.whisper_encoder.";
const EMBEDDING_MODULE: &str = "mm_streams_embeddings.embedding_module.";
const TRANSFORMERS_EMBEDDING: &str = "language_model.model.embed_tokens.weight";
// transformers' first convolution, which tells the model a checkpoint is in its layout
const TRANSFORMERS_CONV: &str = "audio_tower.embedder.conv1.weight";
const WEIGHTS: &str = "model.safetensors";

fn config() -> anyhow::Result<VoxtralConfig> {
    Ok(VoxtralConfig::from_json(&std::fs::read_to_string(
        format!("{TINY}/params.json"),
    )?)?)
}

/// The committed params and the tiny tokenizer, with random weights under the consolidated names.
pub fn consolidated_checkpoint() -> anyhow::Result<tempfile::TempDir> {
    let cfg = config()?;
    let params = std::path::PathBuf::from(format!("{TINY}/params.json"));
    let tokenizer = std::path::PathBuf::from(TOKENIZER);
    recording::record_checkpoint_seeded_by_name(
        &[params.as_path(), tokenizer.as_path()],
        cfg.n_layers,
        &[TRANSFORMERS_CONV],
        |vb, metadata| {
            VoxtralModel::new(&cfg, vb, false, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}

fn shared_names(name: &str) -> String {
    [
        ("attention.wq", "self_attn.q_proj"),
        ("attention.wk", "self_attn.k_proj"),
        ("attention.wv", "self_attn.v_proj"),
        ("attention.wo", "self_attn.o_proj"),
        ("feed_forward.w1", "mlp.gate_proj"),
        ("feed_forward.w2", "mlp.down_proj"),
        ("feed_forward.w3", "mlp.up_proj"),
    ]
    .iter()
    .fold(name.to_string(), |name, (from, to)| name.replace(from, to))
}

/// transformers' name for a consolidated tensor, as its conversion script renames them.
pub fn transformers_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix(ENCODER) {
        let rest = rest
            .replace("conv_layers.0.conv", "embedder.conv1")
            .replace("conv_layers.1.conv", "embedder.conv2")
            .replace("transformer.", "");
        let rest = shared_names(&rest)
            .replace("attention_norm", "self_attn_layer_norm")
            .replace("ffn_norm", "final_layer_norm");
        return format!("audio_tower.{rest}");
    }
    if let Some(rest) = name.strip_prefix(EMBEDDING_MODULE) {
        return match rest {
            "tok_embeddings.weight" => TRANSFORMERS_EMBEDDING.to_string(),
            rest => rest
                .replace(
                    "audio_language_projection.0",
                    "multi_modal_projector.linear_1",
                )
                .replace(
                    "audio_language_projection.2",
                    "multi_modal_projector.linear_2",
                ),
        };
    }
    let rest = shared_names(name)
        .replace("attention_norm", "input_layernorm")
        .replace("ffn_norm", "post_attention_layernorm")
        .replace("ada_rms_norm_t_cond.0", "ada_rms_norm.linear1")
        .replace("ada_rms_norm_t_cond.2", "ada_rms_norm.linear2");
    format!("language_model.model.{rest}")
}

// consolidated interleaves each head's rotary pairs; transformers stores the halves apart
fn half_split(t: &Tensor, heads: usize) -> anyhow::Result<Tensor> {
    let dims = t.dims().to_vec();
    let head_dim = dims[0] / heads;
    let rest = dims[1..].iter().product::<usize>();
    Ok(t.reshape((heads, head_dim / 2, 2, rest))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape(dims)?)
}

/// The consolidated checkpoint's weights renamed and re-laid as transformers' `model.safetensors` holds them.
pub fn transformers_tensors(
    consolidated: &std::path::Path,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let cfg = config()?;
    let enc = &cfg.multimodal.whisper_model_args.encoder_args;
    let tensors = inference_tensor::safetensors::load(
        consolidated.join(WEIGHTS),
        &inference_tensor::Device::Cpu,
    )?;
    tensors
        .into_iter()
        .map(|(name, t)| {
            let heads = match (
                name.starts_with(ENCODER),
                name.contains("attention.wq"),
                name.contains("attention.wk"),
            ) {
                (true, true, _) => Some(enc.n_heads),
                (true, _, true) => Some(enc.n_kv_heads),
                (false, true, _) => Some(cfg.n_heads),
                (false, _, true) => Some(cfg.n_kv_heads),
                _ => None,
            };
            let t = match heads {
                Some(heads) => half_split(&t, heads)?,
                None => t,
            };
            Ok((transformers_name(&name), t))
        })
        .collect()
}

/// The same weights in transformers' layout, beside the same params and tokenizer.
pub fn transformers_checkpoint(
    consolidated: &std::path::Path,
) -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    for file in ["params.json", "tokenizer.json"] {
        std::fs::copy(consolidated.join(file), dir.path().join(file))?;
    }
    inference_tensor::safetensors::save(
        &transformers_tensors(consolidated)?,
        dir.path().join(WEIGHTS),
    )?;
    Ok(dir)
}
