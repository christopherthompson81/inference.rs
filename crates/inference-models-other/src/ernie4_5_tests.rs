//! Locks ERNIE 4.5's prefill on tiny made-up weights: the dense model, and the MoE with layer 0 dense and layer 1
//! routing over experts by softmax plus selection bias, beside its ungated shared experts.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::{Ernie4_5Loader, Ernie4_5MoeLoader};

const VOCAB: usize = 64;
const HIDDEN: usize = 32;
const MOE_INTERMEDIATE: usize = 16;
const EXPERTS: usize = 4;
const MOE_LAYER: usize = 1;

fn dense_config() -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": true,
        "quantization_config": null,
    })
}

// Expert layout detection reads these shapes before any tensor; the checkpoints store the experts one by one.
fn expert_shapes() -> HashMap<String, Vec<usize>> {
    (0..EXPERTS)
        .flat_map(|expert| {
            let p = format!("model.layers.{MOE_LAYER}.mlp.experts.{expert}");
            [
                (
                    format!("{p}.gate_proj.weight"),
                    vec![MOE_INTERMEDIATE, HIDDEN],
                ),
                (
                    format!("{p}.up_proj.weight"),
                    vec![MOE_INTERMEDIATE, HIDDEN],
                ),
                (
                    format!("{p}.down_proj.weight"),
                    vec![HIDDEN, MOE_INTERMEDIATE],
                ),
            ]
        })
        .collect()
}

fn prefill(
    loader: &dyn NormalModelLoader,
    config: &Value,
    shapes: HashMap<String, Vec<usize>>,
    expected_names: u64,
    expected: &Snapshot,
) -> Result<()> {
    let (model, names) = load_synthesized(&[], shapes, DType::F32, |vb| {
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

#[test]
fn ernie4_5_prefill() -> Result<()> {
    prefill(
        &Ernie4_5Loader,
        &dense_config(),
        HashMap::new(),
        0x55d2_5ca4_fbf1_9e2e,
        &Snapshot {
            probes: [-0.62108856, -0.5534645, 1.2946829, -0.6743976],
            sum: -48.14716,
            l2: 16.944386,
        },
    )
}

#[test]
fn ernie4_5_moe_prefill() -> Result<()> {
    let mut config = dense_config();
    config.as_object_mut().unwrap().extend(
        json!({
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "moe_num_experts": EXPERTS,
            "moe_k": 2,
            "moe_num_shared_experts": 2,
            "moe_layer_start_index": MOE_LAYER,
        })
        .as_object()
        .unwrap()
        .clone(),
    );
    prefill(
        &Ernie4_5MoeLoader,
        &config,
        expert_shapes(),
        0x606b_e654_4332_7c3c,
        &Snapshot {
            probes: [-1.8124458, -1.145232, 1.0855446, -0.49902478],
            sum: -83.17132,
            l2: 17.973629,
        },
    )
}
