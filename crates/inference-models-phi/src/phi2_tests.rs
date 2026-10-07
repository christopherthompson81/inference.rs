//! Locks Phi-2's prefill on tiny made-up weights: parallel attention and MLP off one LayerNorm, partial RoPE.

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest, patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::Phi2Loader;

const VOCAB: usize = 64;

fn prefill(config: &Value, expected_names: u64, expected: &Snapshot) -> Result<()> {
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
        Phi2Loader.load(
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

fn config() -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": null,
        "hidden_act": "gelu_new",
        "max_position_embeddings": 64,
        "layer_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "qk_layernorm": false,
        "tie_word_embeddings": false,
    })
}

#[test]
fn phi2_prefill() -> Result<()> {
    prefill(
        &config(),
        0xd247_68ad_62b7_b4a6,
        &Snapshot {
            probes: [-0.025898028, -0.8811163, -1.2798675, -1.0146405],
            sum: -78.0925,
            l2: 15.501827,
        },
    )
}

#[test]
fn phi2_prefill_full_rotary() -> Result<()> {
    prefill(
        &patched(config(), json!({"partial_rotary_factor": 1.0})),
        0xd247_68ad_62b7_b4a6,
        &Snapshot {
            probes: [-0.025898028, -0.7319666, -1.5905572, -1.1711627],
            sum: -75.82963,
            l2: 15.516547,
        },
    )
}
