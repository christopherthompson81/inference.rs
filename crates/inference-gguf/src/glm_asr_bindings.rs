//! GLM-ASR from llama.cpp's GGUFs: a `llama` text GGUF beside a `glma` projector. llama.cpp converts the original
//! checkpoint, so the GGUFs bind onto transformers' GLM-ASR, the layout the safetensors loader reads: Llama-style
//! layer names, q/k rows back in half-split rotary order, and the original's unused position table and learned
//! audio markers (`v.boi`, `v.eoi`, which transformers' checkpoint has no counterpart for) left unbound.

use std::collections::HashMap;

use anyhow::{Result, bail};
use inference_nn::loaders::NormalLoaderType;
use inference_quant::{GgufArchive, GgufBindingMap};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::multimodal_binding_utils::{
    TensorInventory, bind_llama_text, bind_required_linear, map_usize, validate_projector,
};
use super::normal_config::synthesize_normal_config_value;
use super::whisper_audio_bindings::{TowerLayout, bind_whisper_tower, token_id};

pub const PROJECTOR: &str = "glma";
pub const TEXT_ARCHITECTURE: &str = "llama";
const FAMILY: &str = "GLM-ASR";
const AUDIO_TOKEN: &str = "<|pad|>";
// the GGUFs record no encoder rotary; transformers' config: theta 10000 over half of each 64-dim head
const ROPE_THETA: f64 = 10_000.0;
const PARTIAL_ROTARY_FACTOR: f64 = 0.5;
// the original encoder's (unused) Whisper table, as many positions as a 30 s window
const MAX_POSITIONS: usize = 1500;

fn layout(heads: usize, head_dim: usize) -> TowerLayout {
    TowerLayout {
        layer: [
            ("self_attn.q_proj", "attn_q"),
            ("self_attn.k_proj", "attn_k"),
            ("self_attn.v_proj", "attn_v"),
            ("self_attn.o_proj", "attn_out"),
            ("input_layernorm", "ln1"),
            ("mlp.fc1", "ffn_up"),
            ("mlp.fc2", "ffn_down"),
            ("post_attention_layernorm", "ln2"),
        ],
        final_norm: ("norm", "mm.a.norm_pre"),
        learned_positions: false,
        interleaved_rotary: Some((heads, rotary_dims(head_dim))),
    }
}

fn rotary_dims(head_dim: usize) -> usize {
    (head_dim as f64 * PARTIAL_ROTARY_FACTOR) as usize
}

pub fn build_glm_asr_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_projector(archive, PROJECTOR)?;
    let metadata = archive.metadata();
    let heads = map_usize(metadata, "clip.audio.attention.head_count")?;
    let hidden = map_usize(metadata, "clip.audio.embedding_length")?;
    build_glm_asr_bindings_from_inventory(
        &TensorInventory::from_archive(archive),
        heads,
        hidden / heads,
    )
}

fn build_glm_asr_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
    heads: usize,
    head_dim: usize,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_llama_text(
        inventory,
        &mut bindings,
        "language_model.model",
        "language_model.lm_head",
        FAMILY,
    )?;
    bind_whisper_tower(inventory, &mut bindings, FAMILY, &layout(heads, head_dim))?;
    for (native, source) in [("linear_1", "mm.a.mlp.1"), ("linear_2", "mm.a.mlp.2")] {
        bind_required_linear(
            inventory,
            &mut bindings,
            &format!("multi_modal_projector.{native}"),
            source,
        )?;
    }
    Ok(bindings)
}

/// The `GlmAsrForConditionalGeneration` config the merged text and projector GGUFs describe.
pub fn synthesize_glm_asr_config(archive: &GgufArchive) -> Result<String> {
    let tensor_names = archive.tensors().keys().cloned().collect::<Vec<_>>();
    glm_asr_config(archive.metadata(), &tensor_names)
}

fn glm_asr_config(metadata: &HashMap<String, Value>, tensor_names: &[String]) -> Result<String> {
    let text = synthesize_normal_config_value(&NormalLoaderType::Llama, metadata, tensor_names)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let hidden = map_usize(metadata, "clip.audio.embedding_length")?;
    let ffn = map_usize(metadata, "clip.audio.feed_forward_length")?;
    let stack = map_usize(metadata, "clip.audio.projector.stack_factor")?;
    // transformers regroups the encoder output by the tower's FFN width
    if stack * hidden != ffn {
        bail!(
            "{FAMILY} projector stacks {stack} positions of {hidden}, but its encoder FFN is {ffn} wide"
        );
    }
    let config = json!({
        "architectures": ["GlmAsrForConditionalGeneration"],
        "model_type": "glmasr",
        "audio_token_id": token_id(metadata, AUDIO_TOKEN, FAMILY)?,
        "projector_hidden_act": "gelu",
        "audio_config": {
            "hidden_size": hidden,
            "intermediate_size": ffn,
            "num_attention_heads": map_usize(metadata, "clip.audio.attention.head_count")?,
            "num_hidden_layers": map_usize(metadata, "clip.audio.block_count")?,
            "num_mel_bins": map_usize(metadata, "clip.audio.num_mel_bins")?,
            "max_position_embeddings": MAX_POSITIONS,
            "hidden_act": "gelu",
            "rope_parameters": {
                "rope_theta": ROPE_THETA,
                "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
            },
        },
        "text_config": text,
    });
    Ok(config.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal_binding_utils::binding_sources;
    use crate::whisper_audio_bindings::{POSITIONS, TOKENS_KEY};

    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;

    fn inventory_tensors() -> Vec<(String, Vec<usize>)> {
        let d = HEADS * HEAD_DIM;
        let mut tensors = vec![
            ("token_embd.weight".to_string(), vec![8, 4]),
            ("output_norm.weight".to_string(), vec![8]),
            ("output.weight".to_string(), vec![8, 4]),
            ("a.conv1d.1.weight".to_string(), vec![3, 2, d]),
            ("a.conv1d.1.bias".to_string(), vec![1, d]),
            ("a.conv1d.2.weight".to_string(), vec![3, d, d]),
            ("a.conv1d.2.bias".to_string(), vec![1, d]),
            (POSITIONS.to_string(), vec![d, 6]),
            ("mm.a.norm_pre.weight".to_string(), vec![d]),
            ("mm.a.norm_pre.bias".to_string(), vec![d]),
            ("mm.a.mlp.1.weight".to_string(), vec![16, 32]),
            ("mm.a.mlp.1.bias".to_string(), vec![16]),
            ("mm.a.mlp.2.weight".to_string(), vec![8, 16]),
            ("mm.a.mlp.2.bias".to_string(), vec![8]),
            ("v.boi".to_string(), vec![8]),
            ("v.eoi".to_string(), vec![8]),
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
            tensors.push((format!("a.blk.0.{name}.weight"), vec![d, d]));
        }
        for name in [
            "attn_q", "attn_v", "attn_out", "ffn_up", "ffn_down", "ln1", "ln2",
        ] {
            tensors.push((format!("a.blk.0.{name}.bias"), vec![d]));
        }
        for name in ["ln1", "ln2"] {
            tensors.push((format!("a.blk.0.{name}.weight"), vec![d]));
        }
        tensors
    }

    // the unused position table and the original's audio markers are the only tensors left out
    #[test]
    fn every_glm_asr_tensor_transformers_reads_is_bound_to_its_name() {
        let tensors = inventory_tensors();
        let inventory =
            TensorInventory::new(tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice())));
        let bindings = build_glm_asr_bindings_from_inventory(&inventory, HEADS, HEAD_DIM).unwrap();
        let unbound = [POSITIONS, "v.boi", "v.eoi"];
        assert_eq!(
            binding_sources(&bindings),
            tensors
                .iter()
                .map(|(name, _)| name.clone())
                .filter(|name| !unbound.contains(&name.as_str()))
                .collect()
        );
        for native in [
            "audio_tower.conv1.bias",
            "audio_tower.layers.0.self_attn.o_proj.weight",
            "audio_tower.layers.0.mlp.fc1.weight",
            "audio_tower.layers.0.input_layernorm.weight",
            "audio_tower.norm.weight",
            "multi_modal_projector.linear_2.bias",
            "language_model.lm_head.weight",
        ] {
            assert!(bindings.get(native).is_some(), "`{native}` is not bound");
        }
        // q and k (weight and q's bias) come back through the rotary permutation, not as stored
        for native in [
            "audio_tower.layers.0.self_attn.q_proj.weight",
            "audio_tower.layers.0.self_attn.q_proj.bias",
            "audio_tower.layers.0.self_attn.k_proj.weight",
        ] {
            assert!(
                !matches!(
                    bindings.get(native),
                    Some(inference_quant::GgufTensorBinding::Tensor(_))
                ),
                "`{native}` is bound as stored"
            );
        }
        assert!(
            bindings
                .get("audio_tower.layers.0.self_attn.k_proj.bias")
                .is_none()
        );
    }

    #[test]
    fn a_glm_asr_config_comes_from_the_text_and_projector_metadata() {
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
            ("llama.context_length", 2048),
            ("clip.audio.embedding_length", 16),
            ("clip.audio.feed_forward_length", 64),
            ("clip.audio.block_count", 2),
            ("clip.audio.attention.head_count", 2),
            ("clip.audio.num_mel_bins", 128),
            ("clip.audio.projector.stack_factor", 4),
        ] {
            put(key, Value::U32(value));
        }
        put("llama.rope.freq_base", Value::F32(1e4));
        put("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5));
        let mut tokens = (0..300)
            .map(|i| Value::String(format!("<t{i}>")))
            .collect::<Vec<_>>();
        tokens[280] = Value::String(AUDIO_TOKEN.to_string());
        put(TOKENS_KEY, Value::Array(tokens));
        let names = inventory_tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        let json = glm_asr_config(&metadata, &names).unwrap();
        inference_models_llama::glm_asr::GlmAsrConfig::from_json(&json).unwrap();
        let config: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(config["audio_token_id"], 280);
        assert_eq!(config["audio_config"]["intermediate_size"], 64);
        assert_eq!(
            config["audio_config"]["rope_parameters"]["partial_rotary_factor"],
            0.5
        );
        assert_eq!(config["text_config"]["hidden_size"], 48);
        assert_eq!(config["text_config"]["vocab_size"], 300);
    }
}
