//! Locks GLM4's prefill on tiny made-up weights: sandwich norms, partial rotary and q/k/v bias.

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::GLM4Loader;

const VOCAB: usize = 64;

fn prefill(
    loader: &dyn NormalModelLoader,
    config: &Value,
    expected_names: u64,
    expected: &Snapshot,
) -> Result<()> {
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
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
fn glm4_prefill() -> Result<()> {
    let config = json!({
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": null,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 64,
        "attention_bias": true,
        "head_dim": 8,
        "quantization_config": null,
        "tie_word_embeddings": false,
    });
    prefill(
        &GLM4Loader,
        &config,
        0x2924_1afa_bd9a_1bb6,
        &Snapshot {
            probes: [-0.16945215, 1.921854, 0.6041116, -0.025098458],
            sum: 43.15165,
            l2: 17.503899,
        },
    )
}
