//! Locks Qwen3's prefill and the Qwen3 embedder's hidden states on tiny made-up weights.

use anyhow::Result;
use inference_nn::attention::FlashParams;
use inference_nn::loaders::{EmbeddingModelLoader, NormalModelLoader};
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    PROMPT, Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest,
    patched,
};
use inference_tensor::{DType, Device, Tensor};
use serde_json::{Value, json};

use crate::loaders::{Qwen3EmbeddingLoader, Qwen3Loader};

const VOCAB: usize = 64;
// The embedder's hidden states stand in for logits, so its width has to cover the snapshot probes
const EMBEDDING_HIDDEN: usize = 64;

fn qwen3_config() -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": null,
        "head_dim": 8,
        "quantization_config": null,
        "tie_word_embeddings": false,
        "max_window_layers": 2,
        "use_sliding_window": false,
    })
}

fn prefill(config: &Value, expected_names: u64, expected: &Snapshot) -> Result<()> {
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
        Qwen3Loader.load(
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
fn qwen3_prefill() -> Result<()> {
    prefill(
        &qwen3_config(),
        0xbfc5_9a52_43b3_772a,
        &Snapshot {
            probes: [-0.31660226, 1.5481849, 1.0827407, 1.3690944],
            sum: 50.453697,
            l2: 18.20922,
        },
    )
}

// Layer 1 slides over a window shorter than the prompt; layer 0 stays full.
#[test]
fn qwen3_prefill_with_a_sliding_layer() -> Result<()> {
    let config = patched(
        qwen3_config(),
        json!({"sliding_window": 2, "use_sliding_window": true, "max_window_layers": 1}),
    );
    prefill(
        &config,
        0xbfc5_9a52_43b3_772a,
        &Snapshot {
            probes: [-0.31660226, 1.5481849, 0.91872513, 1.5498589],
            sum: 56.253574,
            l2: 18.407532,
        },
    )
}

#[test]
fn qwen3_embedding_hidden_states() -> Result<()> {
    let config = patched(qwen3_config(), json!({"hidden_size": EMBEDDING_HIDDEN}));
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
        Qwen3EmbeddingLoader.load(
            &config.to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    assert_eq!(
        names_digest(names.keys()),
        0xe78d_bbb2_7202_81ba,
        "tensor names moved: {:#x}",
        names_digest(names.keys())
    );
    let input = Tensor::new(&PROMPT, &Device::Cpu)?.unsqueeze(0)?;
    let hidden = model
        .forward(&input, &FlashParams::empty(false))?
        .to_dtype(DType::F32)?;
    let expected = Snapshot {
        probes: [0.3702855, -0.797998, -1.040941, 1.9468331],
        sum: -46.403706,
        l2: 17.512182,
    };
    assert_snapshot(&hidden, EMBEDDING_HIDDEN, &expected)
}
