//! Builds tiny random-weight Qwen-VL checkpoints at test time, for the image and video input paths.
#![allow(dead_code)]

use inference_models_qwen::exaone4_5::{Config as Exaone4_5Config, exaone4_5_model};
use inference_models_qwen::qwen2_5_vl::{Config as Qwen2_5VLConfig, Qwen2_5VLModel};
use inference_models_qwen::qwen2vl::{Config as Qwen2VLConfig, Qwen2VLModel};
use inference_models_qwen::qwen3_5::{Config as Qwen3_5Config, Qwen3_5Model};
use inference_models_qwen::qwen3_vl::{Config as Qwen3VLConfig, Qwen3VLModel};
use inference_models_qwen::qwen3_vl_moe::{Config as Qwen3VLMoEConfig, Qwen3VLMoEModel};
use inference_nn::paged_attention::AttentionImplementation;

#[path = "recording.rs"]
mod recording;

const QWEN2_VL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_vl/qwen2_vl"
);
const QWEN2_5_VL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_vl/qwen2_5_vl"
);
const QWEN3_VL_MOE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_vl/qwen3_vl_moe"
);
const QWEN3_VL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_vl/qwen3_vl"
);
const QWEN3_5_MOE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen_vl/qwen3_5_moe"
);
const TEXT_LAYERS: usize = 2;
// shorter than every prompt, so the sliding layer drops keys
const EXAONE_SLIDING_WINDOW: usize = 2;
// From qwen3_5_moe/ and qwen3_vl_moe/config.json, written by make_tiny.py.
const MOE_LAYERS: usize = 4;
const MOE_EXPERTS: usize = 4;
const MOE_HIDDEN: usize = 128;
// moe_gemm takes expert widths in multiples of 64
const MOE_INTERMEDIATE: usize = 64;
const DENSE_INTERMEDIATE: usize = 128;
// above every test's context, so QSA keeps every key
const FLASH_NEXT_INDEXER_BUDGET: usize = 1024;
// the fixture tokenizer's EOS, which also cuts the PLE n-grams
const FLASH_NEXT_EOS: u32 = 259;
// The MLX layout the constructors probe for; reporting it absent selects the HF tensor names.
const MLX_PROBES: &[&str] = &[
    "vision_tower.patch_embed.proj.weight",
    "language_model.model.embed_tokens.weight",
];

fn files(dir: &str) -> anyhow::Result<Vec<std::path::PathBuf>> {
    recording::fixture_files(dir)
}

pub fn tiny_qwen2_vl() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Qwen2VLConfig =
        serde_json::from_str(&std::fs::read_to_string(format!("{QWEN2_VL}/config.json"))?)?;
    let files = files(QWEN2_VL)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint(&files, TEXT_LAYERS, MLX_PROBES, |vb, metadata| {
        Qwen2VLModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}

pub fn tiny_qwen3_vl() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Qwen3VLConfig =
        serde_json::from_str(&std::fs::read_to_string(format!("{QWEN3_VL}/config.json"))?)?;
    let files = files(QWEN3_VL)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint(&files, TEXT_LAYERS, MLX_PROBES, |vb, metadata| {
        Qwen3VLModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}

pub fn tiny_qwen2_5_vl() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Qwen2_5VLConfig = serde_json::from_str(&std::fs::read_to_string(format!(
        "{QWEN2_5_VL}/config.json"
    ))?)?;
    let files = files(QWEN2_5_VL)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint(&files, TEXT_LAYERS, MLX_PROBES, |vb, metadata| {
        Qwen2_5VLModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
    })
}

/// Qwen2.5-VL's tokenizer and template under an EXAONE 4.5 config: a GQA tower (one kv head for two) beside EXAONE 4
/// text layers, a sliding one and a NoPE global one.
pub fn tiny_exaone4_5() -> anyhow::Result<tempfile::TempDir> {
    record_exaone4_5(false)
}

/// As [`tiny_exaone4_5`] with its MTP layer, at the checkpoint's root as EXAONE 4.5 keeps it.
pub fn tiny_exaone4_5_mtp() -> anyhow::Result<tempfile::TempDir> {
    record_exaone4_5(true)
}

fn record_exaone4_5(mtp: bool) -> anyhow::Result<tempfile::TempDir> {
    let qwen: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{QWEN2_5_VL}/config.json"
    ))?)?;
    let mut vision = qwen["vision_config"].clone();
    vision["num_key_value_heads"] = 1.into();
    let config = serde_json::json!({
        "architectures": ["Exaone4_5_ForConditionalGeneration"],
        "model_type": "exaone4_5",
        "image_token_id": qwen["image_token_id"],
        "video_token_id": qwen["video_token_id"],
        "tie_word_embeddings": false,
        "eos_token_id": qwen["eos_token_id"],
        "vision_config": vision,
        "text_config": {
            "model_type": "exaone4_5_text",
            "vocab_size": qwen["vocab_size"],
            "hidden_size": qwen["hidden_size"],
            "intermediate_size": qwen["intermediate_size"],
            "num_hidden_layers": TEXT_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "rope_scaling": {
                "factor": 16.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                "original_max_position_embeddings": 256, "rope_theta": 1000000.0, "rope_type": "llama3"
            },
            "sliding_window": EXAONE_SLIDING_WINDOW,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_nextn_predict_layers": usize::from(mtp),
        },
    });
    let mut cfg: Exaone4_5Config = serde_json::from_value(config.clone())?;
    cfg.mtp = mtp;
    let staging = tempfile::tempdir()?;
    let config_path = staging.path().join("config.json");
    std::fs::write(&config_path, config.to_string())?;
    let mut files = files(QWEN2_5_VL)?;
    files.retain(|path| path.file_name().is_some_and(|name| name != "config.json"));
    files.push(config_path);
    // EXAONE's delimiters under the ids Qwen's take
    for path in &mut files {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        if [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ]
        .contains(&name.as_str())
        {
            let text = std::fs::read_to_string(&path)?
                .replace("<|vision_start|>", "<vision>")
                .replace("<|vision_end|>", "</vision>");
            let renamed = staging.path().join(&name);
            std::fs::write(&renamed, text)?;
            *path = renamed;
        }
    }
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    // llama.cpp's precomputed llama3 factors; a safetensors checkpoint has none
    recording::record_checkpoint(
        &files,
        TEXT_LAYERS,
        &["model.language_model.rope_freqs.weight"],
        |vb, metadata| {
            exaone4_5_model(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}

pub fn tiny_qwen3_vl_moe() -> anyhow::Result<tempfile::TempDir> {
    let cfg: Qwen3VLMoEConfig = serde_json::from_str(&std::fs::read_to_string(format!(
        "{QWEN3_VL_MOE}/config.json"
    ))?)?;
    let files = files(QWEN3_VL_MOE)?;
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    // HF stacks these experts transposed, gate_up [E, H, 2I] and down [E, I, H]; the layout is detected from shapes.
    let shapes = (0..TEXT_LAYERS)
        .flat_map(|layer| {
            let p = format!("model.language_model.layers.{layer}.mlp.experts");
            [
                (
                    format!("{p}.gate_up_proj"),
                    vec![MOE_EXPERTS, MOE_HIDDEN, 2 * MOE_INTERMEDIATE],
                ),
                (
                    format!("{p}.down_proj"),
                    vec![MOE_EXPERTS, MOE_INTERMEDIATE, MOE_HIDDEN],
                ),
            ]
        })
        .collect();
    recording::record_checkpoint_with_shapes(
        &files,
        TEXT_LAYERS,
        MLX_PROBES,
        shapes,
        |vb, metadata| {
            Qwen3VLMoEModel::new(&cfg, vb, true, metadata, AttentionImplementation::Eager)
                .map(|_| ())
        },
    )
}

/// The Qwen3.5-MoE checkpoint as a Qwen3.8-Flash-Next (qwen4exp): hyper-connections, PLE on a GDN layer, a QSA
/// indexer on the full-attention layer, fused experts and a vision tower without deepstack.
pub fn tiny_flash_next_vl() -> anyhow::Result<tempfile::TempDir> {
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{QWEN3_5_MOE}/config.json"
    ))?)?;
    config["architectures"] = serde_json::json!(["Qwen4ExpForConditionalGeneration"]);
    config["model_type"] = "qwen4_exp".into();
    config["vision_config"]["deepstack_visual_indexes"] = serde_json::json!([]);
    let text = config["text_config"].as_object_mut().unwrap();
    let serde_json::Value::Object(flash_next) = serde_json::json!({
        "model_type": "qwen4_exp_text",
        "mtp_num_hidden_layers": 0,
        "hc_count": 4,
        "hc_lowrank": 8,
        "output_gate_type": "sigmoid",
        "ple_layer_ids": [2],
        "ple_embed_dim": 64,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ngram_vocab_size_base": 1000,
        "make_ngram_vocab_size_divisible_by": 128,
        "split_ngram_parts": 1,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 16,
        "indexer_budget": FLASH_NEXT_INDEXER_BUDGET,
        "indexer_compress_ratio": 4,
        "eos_token_id": FLASH_NEXT_EOS
    }) else {
        unreachable!("a json! object literal")
    };
    text.extend(flash_next);
    let cfg: Qwen3_5Config = serde_json::from_value(config.clone())?;
    let shapes = (0..MOE_LAYERS)
        .flat_map(|layer| {
            let p = format!("model.language_model.layers.{layer}.mlp.experts");
            [
                (
                    format!("{p}.gate_up_proj"),
                    vec![MOE_EXPERTS, 2 * MOE_INTERMEDIATE, MOE_HIDDEN],
                ),
                (
                    format!("{p}.down_proj"),
                    vec![MOE_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE],
                ),
            ]
        })
        .collect();
    let scratch = tempfile::tempdir()?;
    let config_path = scratch.path().join("config.json");
    std::fs::write(&config_path, config.to_string())?;
    let mut files = files(QWEN3_5_MOE)?;
    files.retain(|path| path.file_name() != Some("config.json".as_ref()));
    files.push(config_path);
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_seeded_with_shapes(
        &files,
        MOE_LAYERS,
        MLX_PROBES,
        shapes,
        |vb, metadata| {
            Qwen3_5Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}

// HF stores the Qwen3.5 MoE experts one by one; the layout detection reads these shapes before any tensor.
fn qwen3_5_moe_expert_shapes() -> std::collections::HashMap<String, Vec<usize>> {
    (0..MOE_LAYERS)
        .flat_map(|layer| (0..MOE_EXPERTS).map(move |expert| (layer, expert)))
        .flat_map(|(layer, expert)| {
            let p = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
            [
                (
                    format!("{p}.gate_proj.weight"),
                    vec![MOE_INTERMEDIATE, MOE_HIDDEN],
                ),
                (
                    format!("{p}.up_proj.weight"),
                    vec![MOE_INTERMEDIATE, MOE_HIDDEN],
                ),
                (
                    format!("{p}.down_proj.weight"),
                    vec![MOE_HIDDEN, MOE_INTERMEDIATE],
                ),
            ]
        })
        .collect()
}

pub fn tiny_qwen3_5_moe() -> anyhow::Result<tempfile::TempDir> {
    record_qwen3_5(true, false)
}

/// The same checkpoint plus the built-in MTP head, whose experts are stacked as in the released MoE checkpoints.
pub fn tiny_qwen3_5_moe_mtp() -> anyhow::Result<tempfile::TempDir> {
    record_qwen3_5(true, true)
}

/// The MoE checkpoint with a dense MLP in place of the experts, plus the built-in MTP head.
pub fn tiny_qwen3_5_mtp() -> anyhow::Result<tempfile::TempDir> {
    record_qwen3_5(false, true)
}

fn record_qwen3_5(moe: bool, mtp: bool) -> anyhow::Result<tempfile::TempDir> {
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{QWEN3_5_MOE}/config.json"
    ))?)?;
    if !moe {
        config["architectures"] = serde_json::json!(["Qwen3_5ForConditionalGeneration"]);
        config["model_type"] = "qwen3_5".into();
        let text = config["text_config"].as_object_mut().unwrap();
        for key in [
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
        ] {
            text.remove(key);
        }
        text.insert("intermediate_size".into(), DENSE_INTERMEDIATE.into());
    }
    let mut cfg: Qwen3_5Config = serde_json::from_value(config.clone())?;
    cfg.mtp = mtp;
    let mut shapes = if moe {
        qwen3_5_moe_expert_shapes()
    } else {
        Default::default()
    };
    if moe && mtp {
        let p = "mtp.layers.0.mlp.experts";
        shapes.insert(
            format!("{p}.gate_up_proj"),
            vec![MOE_EXPERTS, 2 * MOE_INTERMEDIATE, MOE_HIDDEN],
        );
        shapes.insert(
            format!("{p}.down_proj"),
            vec![MOE_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE],
        );
    }
    let scratch = tempfile::tempdir()?;
    let config_path = scratch.path().join("config.json");
    std::fs::write(&config_path, config.to_string())?;
    let mut files = files(QWEN3_5_MOE)?;
    files.retain(|path| path.file_name() != Some("config.json".as_ref()));
    files.push(config_path);
    let files = files.iter().map(|path| path.as_path()).collect::<Vec<_>>();
    recording::record_checkpoint_with_shapes(
        &files,
        MOE_LAYERS + usize::from(mtp),
        MLX_PROBES,
        shapes,
        |vb, metadata| {
            Qwen3_5Model::new(&cfg, vb, true, metadata, AttentionImplementation::Eager).map(|_| ())
        },
    )
}
