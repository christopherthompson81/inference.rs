//! MERaLiON-2 from its official GGUFs (made with the repo's patched llama.cpp): a `gemma2` text GGUF beside a
//! `meralion` projector holding the Whisper encoder, `ln_speech` and the gated adapter.

use std::collections::HashMap;

use anyhow::{Result, bail};
use inference_nn::loaders::NormalLoaderType;
use inference_quant::{GgufArchive, GgufBindingMap};
use inference_tensor::quantized::gguf_file::Value;
use serde_json::json;

use super::gemma3_bindings::bind_text;
use super::multimodal_binding_utils::{
    TensorInventory, bind_required_linear, map_usize, validate_projector,
};
use super::normal_config::synthesize_normal_config_value;
use super::whisper_audio_bindings::{
    TowerLayout, WHISPER_LAYOUT, bind_whisper_tower, token_id, whisper_tower_config,
};

pub const PROJECTOR: &str = "meralion";
const FAMILY: &str = "MERaLiON-2";
const SPEECH_TOKEN: &str = "<SpeechHere>";
// the patch's `mm.a.mlp.N`: frame compression, the gate, the pool, out
const ADAPTER: [(&str, &str); 4] = [
    ("mlp_adapter.0", "mm.a.mlp.0"),
    ("gate_proj", "mm.a.mlp.1"),
    ("pool_proj", "mm.a.mlp.2"),
    ("out_proj", "mm.a.mlp.3"),
];
const LAYOUT: TowerLayout = TowerLayout {
    root: "speech_encoder",
    ..WHISPER_LAYOUT
};

pub fn build_meralion2_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    validate_projector(archive, PROJECTOR)?;
    build_meralion2_bindings_from_inventory(&TensorInventory::from_archive(archive))
}

fn build_meralion2_bindings_from_inventory(
    inventory: &TensorInventory<'_>,
) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    bind_text(
        inventory,
        &mut bindings,
        "text_decoder.model",
        "text_decoder.lm_head",
        false,
    )?;
    bind_whisper_tower(inventory, &mut bindings, FAMILY, &LAYOUT)?;
    bind_required_linear(inventory, &mut bindings, "ln_speech", "mm.a.norm_pre")?;
    for (native, source) in ADAPTER {
        bind_required_linear(
            inventory,
            &mut bindings,
            &format!("speech_audio_adapter.{native}"),
            source,
        )?;
    }
    Ok(bindings)
}

/// The `MERaLiON2ForConditionalGeneration` config the merged text and projector GGUFs describe.
pub fn synthesize_meralion2_config(archive: &GgufArchive) -> Result<String> {
    let shapes = archive
        .tensors()
        .iter()
        .map(|(name, info)| (name.clone(), info.shape().to_vec()))
        .collect();
    meralion2_config(archive.metadata(), &shapes)
}

fn meralion2_config(
    metadata: &HashMap<String, Value>,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<String> {
    let tensor_names = shapes.keys().cloned().collect::<Vec<_>>();
    let text = synthesize_normal_config_value(&NormalLoaderType::Gemma2, metadata, &tensor_names)
        .map_err(|error| anyhow::anyhow!("{error}"))?;
    let hidden = map_usize(metadata, "clip.audio.embedding_length")?;
    let stack = map_usize(metadata, "clip.audio.projector.stack_factor")?;
    // (out, in), as the archive reports it
    let adapter_in = shapes
        .get("mm.a.mlp.0.weight")
        .and_then(|shape| shape.last());
    if adapter_in != Some(&(stack * hidden)) {
        bail!("{FAMILY} adapter does not read {stack} stacked positions of {hidden}");
    }
    let config = json!({
        "architectures": ["MERaLiON2ForConditionalGeneration"],
        "model_type": "meralion2",
        "speech_token_index": token_id(metadata, SPEECH_TOKEN, FAMILY)?,
        "speech_mlp_scale_factor": stack,
        "speech_config": whisper_tower_config(metadata, shapes, FAMILY)?,
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
            ("a.conv1d.1.weight".to_string(), vec![3, 2, 4]),
            ("a.conv1d.1.bias".to_string(), vec![1, 4]),
            ("a.conv1d.2.weight".to_string(), vec![3, 4, 4]),
            ("a.conv1d.2.bias".to_string(), vec![1, 4]),
            (POSITIONS.to_string(), vec![4, 6]),
            ("a.post_ln.weight".to_string(), vec![4]),
            ("a.post_ln.bias".to_string(), vec![4]),
            ("mm.a.norm_pre.weight".to_string(), vec![4]),
            ("mm.a.norm_pre.bias".to_string(), vec![4]),
        ];
        for (_, source) in ADAPTER {
            tensors.push((format!("{source}.weight"), vec![8, 8]));
            tensors.push((format!("{source}.bias"), vec![8]));
        }
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
        for name in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            tensors.push((format!("blk.0.{name}.weight"), vec![8]));
        }
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
    fn every_meralion2_tensor_is_bound_to_its_transformers_name() {
        let tensors = inventory_tensors();
        let inventory =
            TensorInventory::new(tensors.iter().map(|(n, s)| (n.as_str(), s.as_slice())));
        let bindings = build_meralion2_bindings_from_inventory(&inventory).unwrap();
        assert_eq!(
            binding_sources(&bindings),
            tensors.iter().map(|(name, _)| name.clone()).collect()
        );
        for native in [
            "speech_encoder.embed_positions.weight",
            "speech_encoder.layers.0.self_attn.out_proj.weight",
            "speech_encoder.layer_norm.bias",
            "ln_speech.weight",
            "speech_audio_adapter.mlp_adapter.0.weight",
            "speech_audio_adapter.out_proj.bias",
            "text_decoder.model.layers.0.post_feedforward_layernorm.weight",
        ] {
            assert!(bindings.get(native).is_some(), "`{native}` is not bound");
        }
        // the head is tied
        assert!(bindings.get("text_decoder.lm_head.weight").is_none());
    }

    #[test]
    fn a_meralion2_config_comes_from_the_text_and_projector_metadata() {
        let mut metadata = HashMap::new();
        let mut put = |key: &str, value: Value| {
            metadata.insert(key.to_string(), value);
        };
        put("general.architecture", Value::String("gemma2".to_string()));
        for (key, value) in [
            ("gemma2.embedding_length", 48),
            ("gemma2.feed_forward_length", 96),
            ("gemma2.block_count", 2),
            ("gemma2.attention.head_count", 2),
            ("gemma2.attention.head_count_kv", 1),
            ("gemma2.attention.key_length", 24),
            ("gemma2.attention.value_length", 24),
            ("gemma2.attention.sliding_window", 4096),
            ("gemma2.context_length", 2048),
            ("clip.audio.embedding_length", 16),
            ("clip.audio.feed_forward_length", 64),
            ("clip.audio.block_count", 2),
            ("clip.audio.attention.head_count", 2),
            ("clip.audio.num_mel_bins", 128),
            ("clip.audio.projector.stack_factor", 15),
        ] {
            put(key, Value::U32(value));
        }
        put("gemma2.attn_logit_softcapping", Value::F32(50.0));
        put("gemma2.final_logit_softcapping", Value::F32(30.0));
        put("gemma2.attention.layer_norm_rms_epsilon", Value::F32(1e-6));
        let mut tokens = (0..300)
            .map(|i| Value::String(format!("<t{i}>")))
            .collect::<Vec<_>>();
        tokens[299] = Value::String(SPEECH_TOKEN.to_string());
        put(TOKENS_KEY, Value::Array(tokens));
        let shapes = inventory_tensors()
            .into_iter()
            .chain([
                (POSITIONS.to_string(), vec![16, 1500]),
                ("mm.a.mlp.0.weight".to_string(), vec![80, 240]),
            ])
            .collect::<HashMap<_, _>>();
        let json = meralion2_config(&metadata, &shapes).unwrap();
        let cfg = inference_models_gemma::meralion2::MERaLiON2Config::from_json(&json).unwrap();
        assert_eq!(cfg.speech_token_index, 299);
        assert_eq!(cfg.speech_mlp_scale_factor, 15);
        assert_eq!(cfg.speech_config.d_model, 16);
        assert_eq!(cfg.text_config.hidden_size, 48);
    }
}
