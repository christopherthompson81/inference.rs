//! Locks StarCoder2's prefill on tiny made-up weights: biased LayerNorm layers, a plain MLP, full or sliding attention.

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest, patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::Starcoder2Loader;

const VOCAB: usize = 64;

fn prefill(config: &Value, expected_names: u64, expected: &Snapshot) -> Result<()> {
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
        Starcoder2Loader.load(
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
        "num_key_value_heads": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "max_position_embeddings": 64,
        "norm_epsilon": 1e-5,
        "rope_theta": 10000.0,
        "use_bias": true,
        "sliding_window": null,
        "tie_word_embeddings": false,
    })
}

#[test]
fn starcoder2_prefill() -> Result<()> {
    prefill(
        &config(),
        0x748c_876e_1879_d235,
        &Snapshot {
            probes: [-0.34178913, 1.401826, 1.5548364, 0.19749717],
            sum: 48.636597,
            l2: 19.636234,
        },
    )
}

#[test]
fn starcoder2_prefill_sliding_tied_unbiased() -> Result<()> {
    let config = patched(
        config(),
        json!({"sliding_window": 3, "tie_word_embeddings": true, "use_bias": false}),
    );
    prefill(
        &config,
        0x85e5_c47c_8ac2_16d5,
        &Snapshot {
            probes: [-1.2587255, -0.79244566, 0.061945267, 0.3642503],
            sum: -3.988884,
            l2: 19.110067,
        },
    )
}
