//! Locks Phi-4MM's text prefill on tiny made-up weights: static vision LoRA merged into every projection, partial
//! LongRoPE past the original context.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::MultimodalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_multimodal, load_synthesized, metadata, names_digest,
    patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::Phi4MMLoader;

const VOCAB: usize = 64;
const GGUF_ROPE_FACTORS: &str = "model.rope_factors";

fn prefill(config: &Value, dtype: DType, expected_names: u64, expected: &Snapshot) -> Result<()> {
    let (model, names) = load_synthesized(&[GGUF_ROPE_FACTORS], HashMap::new(), dtype, |vb| {
        Phi4MMLoader.load(
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
    assert_snapshot(&forward_multimodal(model.as_ref())?, VOCAB, expected)
}

fn config() -> Value {
    let lora = json!({ "layer": "layers", "lora_alpha": 4.0, "r": 2 });
    json!({
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "original_max_position_embeddings": 4,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "use_cache": true,
        "tie_word_embeddings": false,
        "rope_theta": 10000.0,
        "rope_scaling": null,
        "partial_rotary_factor": 0.75,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "sliding_window": null,
        "embd_layer": { "image_embd_layer": null, "audio_embd_layer": null },
        "vision_lora": lora,
        "speech_lora": lora,
        "quantization_config": null,
    })
}

#[test]
fn phi4mm_prefill() -> Result<()> {
    prefill(
        &config(),
        DType::F32,
        0xe240_69f3_cf3f_93c3,
        &Snapshot {
            probes: [2.077187, 0.37962377, -1.8465322, -0.23485968],
            sum: -20.548685,
            l2: 17.561396,
        },
    )
}

// the prompt runs past `original_max_position_embeddings`, so the long factors apply over the rotated 6 of 8 dims
#[test]
fn phi4mm_prefill_longrope_sliding_tied() -> Result<()> {
    let config = patched(
        config(),
        json!({
            "rope_scaling": {
                "type": "longrope",
                "short_factor": [1.0, 1.5, 2.0],
                "long_factor": [3.0, 4.0, 5.0],
            },
            "sliding_window": 3,
            "tie_word_embeddings": true,
        }),
    );
    prefill(
        &config,
        DType::F32,
        0xf039_eee9_cb89_9425,
        &Snapshot {
            probes: [-0.18887675, 0.11563873, -0.76021427, 0.6737891],
            sum: 38.493706,
            l2: 17.991816,
        },
    )
}
