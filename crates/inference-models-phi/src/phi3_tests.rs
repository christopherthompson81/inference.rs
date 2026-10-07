//! Locks Phi-3's prefill on tiny made-up weights: fused qkv and gate/up, LongRoPE from the config or GGUF factors.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest, patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::Phi3Loader;

const VOCAB: usize = 64;
// half the head dim: one factor per rotated pair
const ROPE_FACTORS: usize = 4;
const GGUF_ROPE_FACTORS: &str = "model.rope_factors";

fn prefill(
    config: &Value,
    absent: &[&str],
    shapes: HashMap<String, Vec<usize>>,
    expected_names: u64,
    expected: &Snapshot,
) -> Result<()> {
    let (model, names) = load_synthesized(absent, shapes, DType::F32, |vb| {
        Phi3Loader.load(
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
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "rope_scaling": null,
        "max_position_embeddings": 64,
        "sliding_window": null,
        "original_max_position_embeddings": 4,
        "tie_word_embeddings": false,
        "partial_rotary_factor": null,
    })
}

#[test]
fn phi3_prefill() -> Result<()> {
    prefill(
        &config(),
        &[GGUF_ROPE_FACTORS],
        HashMap::new(),
        0x7957_4782_1e57_6c5d,
        &Snapshot {
            probes: [-0.8364335, 0.619232, 0.7776145, -0.97568667],
            sum: 33.117744,
            l2: 17.976223,
        },
    )
}

// the prompt runs past `original_max_position_embeddings`, so the long factors apply
#[test]
fn phi3_prefill_longrope_sliding_tied() -> Result<()> {
    let config = patched(
        config(),
        json!({
            "rope_scaling": {
                "type": "longrope",
                "short_factor": [1.0, 1.5, 2.0, 2.5],
                "long_factor": [3.0, 4.0, 5.0, 6.0],
            },
            "sliding_window": 3,
            "tie_word_embeddings": true,
        }),
    );
    prefill(
        &config,
        &[GGUF_ROPE_FACTORS],
        HashMap::new(),
        0xc458_3230_8b90_7117,
        &Snapshot {
            probes: [0.38963628, -1.399997, 0.44013375, -1.4508058],
            sum: 11.721778,
            l2: 18.198391,
        },
    )
}

#[test]
fn phi3_prefill_gguf_rope_factors() -> Result<()> {
    let config = patched(config(), json!({"rope_scaling_attn_factor": 1.2}));
    let shapes = ["short", "long"]
        .map(|kind| {
            (
                format!("{GGUF_ROPE_FACTORS}_{kind}.weight"),
                vec![ROPE_FACTORS],
            )
        })
        .into();
    prefill(
        &config,
        &[],
        shapes,
        0x3e1c_9859_922b_23bd,
        &Snapshot {
            probes: [-0.8364335, 0.5416922, 1.167644, -1.1458325],
            sum: 36.420128,
            l2: 18.426968,
        },
    )
}
