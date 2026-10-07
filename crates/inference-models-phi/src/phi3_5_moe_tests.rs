//! Locks Phi-3.5-MoE's prefill on tiny made-up weights: LayerNorm layers, biases, LongRoPE and sparsemixer routing.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest, patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::Phi3_5MoELoader;

const VOCAB: usize = 64;
const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 16;
const EXPERTS: usize = 4;
const LAYERS: usize = 2;
const GGUF_ROPE_FACTORS: &str = "model.rope_factors";

// Expert layout detection reads these shapes before any tensor; the checkpoint stores the experts one by one.
fn expert_shapes() -> HashMap<String, Vec<usize>> {
    (0..LAYERS)
        .flat_map(|layer| (0..EXPERTS).map(move |expert| (layer, expert)))
        .flat_map(|(layer, expert)| {
            let p = format!("model.layers.{layer}.block_sparse_moe.experts.{expert}");
            [
                (format!("{p}.w1.weight"), vec![INTERMEDIATE, HIDDEN]),
                (format!("{p}.w3.weight"), vec![INTERMEDIATE, HIDDEN]),
                (format!("{p}.w2.weight"), vec![HIDDEN, INTERMEDIATE]),
            ]
        })
        .collect()
}

fn prefill(config: &Value, expected_names: u64, expected: &Snapshot) -> Result<()> {
    let (model, names) =
        load_synthesized(&[GGUF_ROPE_FACTORS], expert_shapes(), DType::F32, |vb| {
            Phi3_5MoELoader.load(
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
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": null,
        "max_position_embeddings": 64,
        "sliding_window": null,
        "original_max_position_embeddings": 4,
        "quantization_config": null,
        "lm_head_bias": true,
        "attention_bias": true,
        "num_local_experts": EXPERTS,
        "router_jitter_noise": 0.01,
        "tie_word_embeddings": false,
    })
}

#[test]
fn phi3_5_moe_prefill() -> Result<()> {
    prefill(
        &config(),
        0xca46_6baa_6c4b_5b79,
        &Snapshot {
            probes: [-0.35455522, 1.0136878, -0.32101417, -0.2829165],
            sum: 3.575695,
            l2: 17.15991,
        },
    )
}

// the prompt runs past `original_max_position_embeddings`, so the long factors apply
#[test]
fn phi3_5_moe_prefill_longrope_unbiased() -> Result<()> {
    let config = patched(
        config(),
        json!({
            "rope_scaling": {
                "type": "longrope",
                "short_factor": [1.0, 1.5, 2.0, 2.5],
                "long_factor": [3.0, 4.0, 5.0, 6.0],
            },
            "lm_head_bias": false,
            "attention_bias": false,
        }),
    );
    prefill(
        &config,
        0x90ca_bfcc_b3c5_2bee,
        &Snapshot {
            probes: [0.17627852, 1.3460411, 0.15425546, 0.680068],
            sum: 46.988384,
            l2: 18.34637,
        },
    )
}
