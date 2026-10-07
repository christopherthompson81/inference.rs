//! Locks Qwen2's prefill on tiny made-up weights, with full attention and with a sliding layer.

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest, patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::Qwen2Loader;

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

fn qwen2_config() -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "sliding_window": null,
        "use_sliding_window": false,
        "max_window_layers": 2,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "quantization_config": null,
        "tie_word_embeddings": false,
    })
}

#[test]
fn qwen2_prefill() -> Result<()> {
    prefill(
        &Qwen2Loader,
        &qwen2_config(),
        0xcbc6_ffeb_dece_a351,
        &Snapshot {
            probes: [0.10115375, 1.5709375, 0.59350157, 0.6359584],
            sum: 35.15997,
            l2: 17.545107,
        },
    )
}

// Layer 1 slides over a window shorter than the prompt; layer 0 stays full.
#[test]
fn qwen2_prefill_with_a_sliding_layer() -> Result<()> {
    let config = patched(
        qwen2_config(),
        json!({"sliding_window": 2, "use_sliding_window": true, "max_window_layers": 1}),
    );
    prefill(
        &Qwen2Loader,
        &config,
        0xcbc6_ffeb_dece_a351,
        &Snapshot {
            probes: [0.10115375, 1.5709375, 0.6692579, 0.95223856],
            sum: 42.161827,
            l2: 17.714283,
        },
    )
}
