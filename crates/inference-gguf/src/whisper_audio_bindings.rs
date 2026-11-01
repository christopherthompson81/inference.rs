//! The Whisper tower llama.cpp's audio projectors (`voxtral`, `ultravox`) store under `a.*`: its bindings to
//! transformers' names and the config its metadata gives.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use inference_quant::{GgufBindingMap, GgufTensorBinding};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::{Value as Json, json};

use super::multimodal_binding_utils::{
    TensorInventory, bind_required, bind_required_linear, bind_required_with, map_usize,
};

pub(super) const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
pub(super) const POSITIONS: &str = "a.position_embd.weight";

/// The tower's tensors under `audio_tower.*`.
pub(super) fn bind_whisper_tower(
    inventory: &TensorInventory<'_>,
    bindings: &mut GgufBindingMap,
    family: &str,
) -> Result<()> {
    for (native, source) in [("conv1", "a.conv1d.1"), ("conv2", "a.conv1d.2")] {
        bind_required(
            inventory,
            bindings,
            format!("audio_tower.{native}.weight"),
            format!("{source}.weight"),
        )?;
        // llama.cpp stores the conv bias as (1, channels)
        bind_required_with(
            inventory,
            bindings,
            format!("audio_tower.{native}.bias"),
            format!("{source}.bias"),
            |source, shape| {
                Ok(GgufTensorBinding::tensor(source).reshape(vec![shape.iter().product()]))
            },
        )?;
    }
    bind_required(
        inventory,
        bindings,
        "audio_tower.embed_positions.weight",
        POSITIONS,
    )?;
    bind_required_linear(inventory, bindings, "audio_tower.layer_norm", "a.post_ln")?;
    for layer in inventory.require_layers("a.blk.", family)? {
        let native = format!("audio_tower.layers.{layer}");
        let source = format!("a.blk.{layer}");
        for (to, from) in [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.out_proj", "attn_out"),
            ("self_attn_layer_norm", "ln1"),
            ("fc1", "ffn_up"),
            ("fc2", "ffn_down"),
            ("final_layer_norm", "ln2"),
        ] {
            bind_required_linear(
                inventory,
                bindings,
                &format!("{native}.{to}"),
                &format!("{source}.{from}"),
            )?;
        }
    }
    Ok(())
}

/// The tower's `audio_config` from the projector's metadata, under transformers' Whisper names.
pub(super) fn whisper_tower_config(
    metadata: &HashMap<String, Value>,
    shapes: &HashMap<String, Vec<usize>>,
    family: &str,
) -> Result<Json> {
    let hidden = map_usize(metadata, "clip.audio.embedding_length")?;
    let positions = shapes
        .get(POSITIONS)
        .with_context(|| format!("{family} projector has no `{POSITIONS}`"))?
        .iter()
        .product::<usize>()
        / hidden;
    Ok(json!({
        "hidden_size": hidden,
        "intermediate_size": map_usize(metadata, "clip.audio.feed_forward_length")?,
        "num_attention_heads": map_usize(metadata, "clip.audio.attention.head_count")?,
        "num_hidden_layers": map_usize(metadata, "clip.audio.block_count")?,
        "num_mel_bins": map_usize(metadata, "clip.audio.num_mel_bins")?,
        "max_source_positions": positions,
        "activation_function": "gelu",
    }))
}

/// `token`'s id in the text GGUF's vocabulary.
pub(super) fn token_id(
    metadata: &HashMap<String, Value>,
    token: &str,
    family: &str,
) -> Result<usize> {
    let Some(Value::Array(tokens)) = metadata.get(TOKENS_KEY) else {
        bail!("{family} GGUF has no `{TOKENS_KEY}` list");
    };
    tokens
        .iter()
        .position(|candidate| matches!(candidate, Value::String(text) if text == token))
        .with_context(|| format!("{family} GGUF vocabulary has no `{token}` token"))
}
