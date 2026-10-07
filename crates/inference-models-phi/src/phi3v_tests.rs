//! Locks Phi-3-Vision's text-only prefill (CLIP tower, HD image embedding, Phi-3 text model) on made-up weights.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::MultimodalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_multimodal, load_synthesized, metadata, names_digest,
};
use inference_tensor::DType;
use serde_json::json;

use crate::loaders::Phi3VLoader;

const VOCAB: usize = 64;

#[test]
fn phi3v_text_prefill() -> Result<()> {
    // The CLIP tower is fixed at ViT-L/14-336 (PHI3V_CLIP_CONFIG), so image_dim_out must be its 1024.
    let config = json!({
        "vocab_size": VOCAB,
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "rope_scaling": null,
        "max_position_embeddings": 64,
        "sliding_window": null,
        "original_max_position_embeddings": 64,
        "embd_layer": {
            "embedding_cls": "image",
            "hd_transform_order": "sub_glb",
            "projection_cls": "mlp",
            "use_hd_transform": true,
            "with_learnable_separator": true
        },
        "img_processor": {
            "image_dim_out": 1024,
            "model_name": "openai/clip-vit-large-patch14-336",
            "name": "clip_vision_model",
            "num_img_tokens": 144
        },
        "quantization_config": null,
        "tie_word_embeddings": false,
    });
    let (model, names) = load_synthesized(&[], HashMap::new(), DType::F32, |vb| {
        Phi3VLoader.load(
            &config.to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    assert_eq!(
        names_digest(names.keys()),
        0x0675_75b5_9300_56f5,
        "tensor names moved: {:#x}",
        names_digest(names.keys())
    );
    let expected = Snapshot {
        probes: [-0.36735478, -0.5964175, 2.117749, 0.979436],
        sum: 1.168644,
        l2: 16.548435,
    };
    assert_snapshot(&forward_multimodal(model.as_ref())?, VOCAB, &expected)
}
