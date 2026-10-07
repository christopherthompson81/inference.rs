//! Locks Mixtral's prefill on tiny made-up weights: every layer routes over `block_sparse_moe` experts.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest, patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::MixtralLoader;

const VOCAB: usize = 64;
const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 16;
const EXPERTS: usize = 4;
const LAYERS: usize = 2;

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
    let (model, names) = load_synthesized(&[], expert_shapes(), DType::F32, |vb| {
        MixtralLoader.load(
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
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "sliding_window": null,
        "num_experts_per_tok": 2,
        "num_local_experts": EXPERTS,
        "quantization_config": null,
        "tie_word_embeddings": false,
    })
}

#[test]
fn mixtral_prefill() -> Result<()> {
    prefill(
        &config(),
        0xf335_292b_6929_5dd4,
        &Snapshot {
            probes: [0.64179504, 1.2688609, 0.61056864, -0.36173704],
            sum: 16.686935,
            l2: 18.332682,
        },
    )
}

#[test]
fn mixtral_prefill_sliding_tied() -> Result<()> {
    prefill(
        &patched(
            config(),
            json!({"sliding_window": 3, "tie_word_embeddings": true}),
        ),
        0x8c37_784a_e891_bcc6,
        &Snapshot {
            probes: [-0.8626528, -0.72250557, -0.4146694, 1.2747489],
            sum: -3.2047634,
            l2: 17.519428,
        },
    )
}

// an experts-only ISQ quantizes the experts alone; the router and attention projections are residuals
#[test]
fn mixtral_experts_only_residuals_keep_router_and_attention() -> Result<()> {
    let (model, _) = load_synthesized(&[], expert_shapes(), DType::F32, |vb| {
        MixtralLoader.load(
            &config().to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    let residuals = model
        .residual_tensors_moe_experts_only()
        .expect("Mixtral routes over experts");
    let names: Vec<_> = residuals.iter().map(|(name, _)| name.as_str()).collect();
    for kept in [
        "model.layers.0.block_sparse_moe.gate.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ] {
        assert!(names.contains(&kept), "{kept} missing from {names:?}");
    }
    assert!(!names.iter().any(|name| name.contains(".experts.")));
    Ok(())
}
