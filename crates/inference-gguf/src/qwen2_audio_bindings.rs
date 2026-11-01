//! Qwen2-Audio from llama.cpp's GGUFs: a `qwen2` text GGUF beside a `qwen2a` projector holding the Whisper encoder
//! and the one linear projector.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::NormalLoaderType;
use inference_quant::{GgufArchive, GgufBindingMap};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::multimodal_binding_utils::{
    TensorInventory, bind_llama_text, bind_required_linear, validate_projector,
};
use super::normal_config::synthesize_normal_config_value;
use super::whisper_audio_bindings::{
    WHISPER_LAYOUT, bind_whisper_tower, token_id, whisper_tower_config,
};

pub const PROJECTOR: &str = "qwen2a";
pub const TEXT_ARCHITECTURE: &str = "qwen2";
const FAMILY: &str = "Qwen2-Audio";
const AUDIO_TOKEN: &str = "<|AUDIO|>";

pub fn build_qwen2_audio_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_projector(archive, PROJECTOR)?;
    build_qwen2_audio_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_qwen2_audio_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_llama_text(
        inventory,
        &mut bindings,
        "language_model.model",
        "language_model.lm_head",
        FAMILY,
    )?;
    bind_whisper_tower(inventory, &mut bindings, FAMILY, &WHISPER_LAYOUT)?;
    bind_required_linear(
        inventory,
        &mut bindings,
        "multi_modal_projector.linear",
        "mm.a.fc",
    )?;
    Ok(bindings)
}

/// The `Qwen2AudioForConditionalGeneration` config the merged text and projector GGUFs describe.
pub fn synthesize_qwen2_audio_config(archive: &GgufArchive) -> Result<String> {
    let shapes = archive
        .tensors()
        .iter()
        .map(|(name, info)| (name.clone(), info.shape().to_vec()))
        .collect();
    qwen2_audio_config(archive.metadata(), &shapes)
}

fn qwen2_audio_config(
    metadata: &HashMap<String, Value>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<String> {
    let tensor_names = shapes.keys().cloned().collect::<Vec<_>>();
    let text = synthesize_normal_config_value(&NormalLoaderType::Qwen2, metadata, &tensor_names)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let config = json!({
        "architectures": ["Qwen2AudioForConditionalGeneration"],
        "model_type": "qwen2_audio",
        "audio_token_index": token_id(metadata, AUDIO_TOKEN, FAMILY)?,
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
            ("mm.a.fc.weight".to_string(), vec![4, 8]),
            ("mm.a.fc.bias".to_string(), vec![8]),
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
        for name in ["attn_q", "attn_k", "attn_v"] {
            tensors.push((format!("blk.0.{name}.bias"), vec![8]));
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
    fn every_qwen2_audio_tensor_is_bound_to_its_transformers_name() {
        let tensors = inventory_tensors();
        let inventory =
            TensorInventory::new(tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice())));
        let bindings = build_qwen2_audio_bindings_from_inventory(&inventory).unwrap();
        assert_eq!(
            binding_sources(&bindings),
            tensors.iter().map(|(name, _)| name.clone()).collect()
        );
        for native in [
            "audio_tower.embed_positions.weight",
            "audio_tower.layer_norm.weight",
            "multi_modal_projector.linear.bias",
            "language_model.model.layers.0.self_attn.q_proj.bias",
        ] {
            assert!(bindings.get(native).is_some(), "`{native}` is not bound");
        }
    }

    #[test]
    fn a_qwen2_audio_config_comes_from_the_text_and_projector_metadata() {
        let mut metadata = HashMap::new();
        let mut put = |key: &str, value: Value| {
            metadata.insert(key.to_string(), value);
        };
        put("general.architecture", Value::String("qwen2".to_string()));
        for (key, value) in [
            ("qwen2.embedding_length", 48),
            ("qwen2.feed_forward_length", 96),
            ("qwen2.block_count", 2),
            ("qwen2.attention.head_count", 2),
            ("qwen2.attention.head_count_kv", 2),
            ("qwen2.context_length", 2048),
            ("clip.audio.embedding_length", 16),
            ("clip.audio.feed_forward_length", 64),
            ("clip.audio.block_count", 2),
            ("clip.audio.attention.head_count", 2),
            ("clip.audio.num_mel_bins", 128),
        ] {
            put(key, Value::U32(value));
        }
        put("qwen2.rope.freq_base", Value::F32(1e4));
        put("qwen2.attention.layer_norm_rms_epsilon", Value::F32(1e-5));
        let mut tokens = (0..300)
            .map(|i| Value::String(format!("<t{i}>")))
            .collect::<Vec<_>>();
        tokens[264] = Value::String(AUDIO_TOKEN.to_string());
        put(TOKENS_KEY, Value::Array(tokens));
        let shapes = inventory_tensors()
            .into_iter()
            .chain([(POSITIONS.to_string(), vec![16, 1500])])
            .collect::<HashMap<_, _>>();
        let json = qwen2_audio_config(&metadata, &shapes).unwrap();
        inference_models_qwen::qwen2_audio::Qwen2AudioConfig::from_json(&json).unwrap();
        let config: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(config["audio_token_index"], 264);
        assert_eq!(config["audio_config"]["max_source_positions"], 1500);
        assert_eq!(config["text_config"]["hidden_size"], 48);
    }
}
