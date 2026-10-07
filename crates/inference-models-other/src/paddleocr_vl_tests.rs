//! Locks PaddleOCR-VL's text prefill (ERNIE-4.5 with chunked M-RoPE) on tiny made-up weights, in F32 and BF16.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::MultimodalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_multimodal, load_synthesized, metadata, names_digest,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::PaddleOcrVlLoader;

const VOCAB: usize = 64;

fn prefill(dtype: DType, expected_names: u64, expected: &Snapshot) -> Result<()> {
    let (model, names) = load_synthesized(&[], HashMap::new(), dtype, |vb| {
        PaddleOcrVlLoader.load(
            &config().to_string(),
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

// head_dim is twice the M-RoPE sections' sum
fn config() -> Value {
    json!({
        "head_dim": 16,
        "hidden_size": 32,
        "intermediate_size": 48,
        "max_position_embeddings": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_scaling": { "mrope_section": [2, 3, 3] },
        "rope_theta": 500000.0,
        "vocab_size": VOCAB,
        "image_token_id": 61,
        "vision_config": {
            "hidden_size": 16,
            "image_size": 56,
            "intermediate_size": 32,
            "layer_norm_eps": 1e-6,
            "num_attention_heads": 2,
            "num_channels": 3,
            "num_hidden_layers": 1,
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
    })
}

#[test]
fn paddleocr_vl_prefill() -> Result<()> {
    prefill(
        DType::F32,
        0xc6a2_f574_7978_1a94,
        &Snapshot {
            probes: [1.1266398, -0.47756937, 0.7152879, -0.59420884],
            sum: -25.586395,
            l2: 16.525724,
        },
    )
}

#[test]
fn paddleocr_vl_prefill_bf16() -> Result<()> {
    prefill(
        DType::BF16,
        0xc6a2_f574_7978_1a94,
        &Snapshot {
            probes: [1.1328125, -0.4765625, 0.71484375, -0.5703125],
            sum: -24.725338,
            l2: 16.513906,
        },
    )
}
