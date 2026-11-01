//! Ultravox from llama.cpp's GGUFs: a `llama` text GGUF beside an `ultravox` projector holding the Whisper encoder and
//! the stacked-frame SwiGLU projector.

use std::collections::HashMap;

use anyhow::{Context, Result};
use inference_nn::loaders::NormalLoaderType;
use inference_quant::{GgufArchive, GgufBindingMap};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::multimodal_binding_utils::{
    TensorInventory, bind_llama_text, bind_required, bind_required_linear, map_usize,
    validate_projector,
};
use super::normal_config::synthesize_normal_config_value;
use super::whisper_audio_bindings::{
    WHISPER_LAYOUT, bind_whisper_tower, token_id, whisper_tower_config,
};

pub const PROJECTOR: &str = "ultravox";
pub const TEXT_ARCHITECTURE: &str = "llama";
const FAMILY: &str = "Ultravox";
// the processor's placeholder; the reference writes EOS there and swaps in the audio, the same input
const AUDIO_TOKEN: &str = "<|reserved_special_token_0|>";
const FFN_IN: &str = "mm.a.mlp.1.weight";

pub fn build_ultravox_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_projector(archive, PROJECTOR)?;
    build_ultravox_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_ultravox_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_llama_text(inventory, &mut bindings, "model", "lm_head", FAMILY)?;
    bind_whisper_tower(inventory, &mut bindings, FAMILY, &WHISPER_LAYOUT)?;
    for (native, source) in [("linear_1", "mm.a.mlp.1"), ("linear_2", "mm.a.mlp.2")] {
        bind_required_linear(
            inventory,
            &mut bindings,
            &format!("multi_modal_projector.{native}"),
            source,
        )?;
    }
    for (native, source) in [("ln_pre", "mm.a.norm_pre"), ("ln_mid", "mm.a.norm_mid")] {
        bind_required(
            inventory,
            &mut bindings,
            format!("multi_modal_projector.{native}.weight"),
            format!("{source}.weight"),
        )?;
    }
    Ok(bindings)
}

/// The config the merged text and projector GGUFs describe: the text half from the `llama` metadata, the audio half
/// and the projector from the projector's.
pub fn synthesize_ultravox_config(archive: &GgufArchive) -> Result<String> {
    let shapes = archive
        .tensors()
        .iter()
        .map(|(name, info)| (name.clone(), info.shape().to_vec()))
        .collect();
    ultravox_config(archive.metadata(), &shapes)
}

fn ultravox_config(
    metadata: &HashMap<String, Value>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<String> {
    let tensor_names = shapes.keys().cloned().collect::<Vec<_>>();
    let text = synthesize_normal_config_value(&NormalLoaderType::Llama, metadata, &tensor_names)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    // (out, in), as the archive reports it
    let inner = *shapes
        .get(FFN_IN)
        .and_then(|shape| shape.first())
        .with_context(|| format!("{FAMILY} projector has no `{FFN_IN}`"))?;
    let config = json!({
        "architectures": ["UltravoxModel"],
        "model_type": "ultravox",
        "audio_token_id": token_id(metadata, AUDIO_TOKEN, FAMILY)?,
        "stack_factor": map_usize(metadata, "clip.audio.projector.stack_factor")?,
        "hidden_size": inner,
        "audio_config": whisper_tower_config(metadata, shapes, FAMILY)?,
        "text_config": text,
    });
    Ok(config.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal_binding_utils::binding_sources;
    use crate::whisper_audio_bindings::{POSITIONS, TOKENS_KEY};

    fn inventory_tensors() -> Vec<(String, Vec<usize>)> {
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 4]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 4]),
            ("a.conv1d.1.weight".to_string(), vec![3, 2, 4]),
            ("a.conv1d.1.bias".to_string(), vec![1, 4]),
            ("a.conv1d.2.weight".to_string(), vec![3, 4, 4]),
            ("a.conv1d.2.bias".to_string(), vec![1, 4]),
            (POSITIONS.to_string(), vec![4, 6]),
            ("a.post_ln.weight".to_string(), vec![4]),
            ("a.post_ln.bias".to_string(), vec![4]),
            ("mm.a.norm_pre.weight".to_string(), vec![8]),
            (FFN_IN.to_string(), vec![8, 12]),
            ("mm.a.norm_mid.weight".to_string(), vec![6]),
            ("mm.a.mlp.2.weight".to_string(), vec![6, 8]),
        ];
        for name in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            tensors.push((format!("blk.0.{name}.weight"), vec![8, 8]));
        }
        tensors.push(("blk.0.attn_norm.weight".to_string(), vec![8]));
        tensors.push(("blk.0.ffn_norm.weight".to_string(), vec![8]));
        for name in [
            "attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down",
        ] {
            tensors.push((format!("a.blk.0.{name}.weight"), vec![4, 4]));
        }
        for name in [
            "attn_q", "attn_v", "attn_out", "ffn_up", "ffn_down", "ln1", "ln2",
        ] {
            tensors.push((format!("a.blk.0.{name}.bias"), vec![4]));
        }
        for name in ["ln1", "ln2"] {
            tensors.push((format!("a.blk.0.{name}.weight"), vec![4]));
        }
        tensors
    }

    #[test]
    fn every_ultravox_tensor_is_bound_to_its_transformers_name() {
        let tensors = inventory_tensors();
        let inventory =
            TensorInventory::new(tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice())));
        let bindings = build_ultravox_bindings_from_inventory(&inventory).unwrap();
        assert_eq!(
            binding_sources(&bindings),
            tensors.iter().map(|(name, _)| name.clone()).collect()
        );
        for native in [
            "audio_tower.conv1.bias",
            "audio_tower.layers.0.fc1.weight",
            "multi_modal_projector.ln_pre.weight",
            "multi_modal_projector.linear_1.weight",
            "multi_modal_projector.ln_mid.weight",
            "multi_modal_projector.linear_2.weight",
            "lm_head.weight",
        ] {
            assert!(bindings.get(native).is_some(), "`{native}` is not bound");
        }
    }

    #[test]
    fn an_ultravox_config_comes_from_the_text_and_projector_metadata() {
        let mut metadata = HashMap::new();
        let mut put = |key: &str, value: Value| {
            metadata.insert(key.to_string(), value);
        };
        put("general.architecture", Value::String("llama".to_string()));
        for (key, value) in [
            ("llama.embedding_length", 48),
            ("llama.feed_forward_length", 96),
            ("llama.block_count", 2),
            ("llama.attention.head_count", 2),
            ("llama.attention.head_count_kv", 1),
            ("llama.attention.key_length", 24),
            ("llama.attention.value_length", 24),
            ("llama.context_length", 2048),
            ("clip.audio.embedding_length", 16),
            ("clip.audio.feed_forward_length", 64),
            ("clip.audio.block_count", 2),
            ("clip.audio.attention.head_count", 2),
            ("clip.audio.num_mel_bins", 128),
            ("clip.audio.projector.stack_factor", 8),
        ] {
            put(key, Value::U32(value));
        }
        put("llama.rope.freq_base", Value::F32(5e5));
        put("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5));
        let mut tokens = (0..300)
            .map(|i| Value::String(format!("<t{i}>")))
            .collect::<Vec<_>>();
        tokens[258] = Value::String(AUDIO_TOKEN.to_string());
        put(TOKENS_KEY, Value::Array(tokens));
        let shapes = inventory_tensors()
            .into_iter()
            .chain([
                (POSITIONS.to_string(), vec![16, 1500]),
                (FFN_IN.to_string(), vec![64, 128]),
            ])
            .collect::<HashMap<_, _>>();
        let json = ultravox_config(&metadata, &shapes).unwrap();
        inference_models_llama::ultravox::UltravoxConfig::from_json(&json).unwrap();
        let config: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(config["audio_token_id"], 258);
        assert_eq!(config["stack_factor"], 8);
        assert_eq!(config["hidden_size"], 64);
        assert_eq!(config["audio_config"]["max_source_positions"], 1500);
        assert_eq!(config["text_config"]["hidden_size"], 48);
        assert_eq!(config["text_config"]["vocab_size"], 300);
    }
}
