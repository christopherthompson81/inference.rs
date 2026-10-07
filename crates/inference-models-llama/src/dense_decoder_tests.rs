//! Locks the prefill of Llama, SmolLM3 and Mistral on tiny made-up weights, each with its rope or window variant.

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest, patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::{LlamaLoader, MistralLoader, SmolLm3Loader};

const VOCAB: usize = 64;
// Llama 3 checkpoints may carry per-frequency rope factors; these use plain rope.
const ABSENT: &[&str] = &["model.rope_freqs.weight"];

fn prefill(
    loader: &dyn NormalModelLoader,
    config: &Value,
    expected_names: u64,
    expected: &Snapshot,
) -> Result<()> {
    let (model, names) = load_synthesized(ABSENT, Default::default(), DType::F32, |vb| {
        loader.load(
            &config.to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    assert_eq!(
        names_digest(names.keys()),
        expected_names,
        "tensor names moved: {:#x}",
        names_digest(names.keys())
    );
    assert_snapshot(&forward_normal(model.as_ref())?, VOCAB, expected)
}

fn llama_config() -> Value {
    json!({
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 48,
        "vocab_size": VOCAB,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 64,
        "rope_scaling": null,
        "quantization_config": null,
        "tie_word_embeddings": false,
    })
}

#[test]
fn llama_prefill() -> Result<()> {
    prefill(
        &LlamaLoader,
        &llama_config(),
        0x4b03_3f00_d15a_edcc,
        &Snapshot {
            probes: [-0.31660226, 1.502106, 0.9029181, 1.3898151],
            sum: 48.40179,
            l2: 18.073221,
        },
    )
}

#[test]
fn llama_prefill_with_llama3_rope_scaling() -> Result<()> {
    let config = patched(
        llama_config(),
        json!({"rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 16,
        }}),
    );
    prefill(
        &LlamaLoader,
        &config,
        0x4b03_3f00_d15a_edcc,
        &Snapshot {
            probes: [-0.31660226, 1.5027776, 1.0367024, 1.4200451],
            sum: 45.768364,
            l2: 17.914343,
        },
    )
}

// Every second layer skips RoPE.
#[test]
fn smollm3_prefill_with_nope_layers() -> Result<()> {
    let config = patched(
        llama_config(),
        json!({"no_rope_layers": null, "no_rope_layer_interval": 2}),
    );
    prefill(
        &SmolLm3Loader,
        &config,
        0x4b03_3f00_d15a_edcc,
        &Snapshot {
            probes: [-0.31660226, 1.4981068, 0.8552188, 1.3966383],
            sum: 47.77395,
            l2: 18.072855,
        },
    )
}

fn mistral_config() -> Value {
    patched(
        llama_config(),
        json!({"sliding_window": null, "head_dim": 8, "rope_parameters": null}),
    )
}

#[test]
fn mistral_prefill_with_a_sliding_window() -> Result<()> {
    let config = patched(mistral_config(), json!({"sliding_window": 2}));
    prefill(
        &MistralLoader,
        &config,
        0x4b03_3f00_d15a_edcc,
        &Snapshot {
            probes: [-0.31660226, 1.502106, 0.96187055, 0.52486664],
            sum: 43.443687,
            l2: 17.74289,
        },
    )
}

#[test]
fn mistral_prefill_with_yarn_rope() -> Result<()> {
    let config = patched(
        mistral_config(),
        json!({"rope_parameters": {
            "rope_theta": 10000.0,
            "rope_type": "yarn",
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 16,
            "llama_4_scaling_beta": 0.1,
        }}),
    );
    prefill(
        &MistralLoader,
        &config,
        0x4b03_3f00_d15a_edcc,
        &Snapshot {
            probes: [-0.31660226, 1.505108, 0.86720014, 1.4266213],
            sum: 48.629845,
            l2: 18.120821,
        },
    )
}
