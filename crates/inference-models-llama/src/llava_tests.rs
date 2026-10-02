//! Locks LLaVA 1.5's text-only prefill (CLIP tower, projector, Llama LLM) on tiny made-up weights.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::DType;
use inference_nn::loaders::MultimodalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_multimodal, load_synthesized, metadata, names_digest,
};
use serde_json::json;

use crate::loaders::LLaVALoader;

const VOCAB: usize = 64;

#[test]
fn llava_llama_prefill() -> Result<()> {
    let config = json!({
        "image_grid_pinpoints": null,
        "projector_hidden_act": "gelu",
        "text_config": {
            "hidden_size": 32,
            "intermediate_size": 48,
            "max_length": 64,
            "max_position_embeddings": 64,
            "model_type": "llama",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "vocab_size": VOCAB,
            "sliding_window": null,
            "rope_scaling": null,
            "quantization_config": null,
        },
        "vision_config": {
            "hidden_size": 16,
            "image_size": 8,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "patch_size": 4,
        },
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
    });
    let (model, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
        LLaVALoader.load(
            &config.to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    assert_eq!(
        names_digest(&names),
        0x4089_c6c3_0a5b_bc83,
        "tensor names moved: {:#x}",
        names_digest(&names)
    );
    let expected = Snapshot {
        probes: [-0.44500175, 0.26666418, 0.32315516, -0.076289274],
        sum: 86.59143,
        l2: 17.094133,
    };
    assert_snapshot(&forward_multimodal(model.as_ref())?, VOCAB, &expected)
}
