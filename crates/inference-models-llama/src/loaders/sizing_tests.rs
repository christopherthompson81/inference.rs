//! Pins each loader's device-map layer sizes and KV planning metadata, so sharing the sizing code moves nothing
//! silently; a value that changes is a drift against the model code.

use serde_json::{Value, json};

use super::*;

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
// differs from HIDDEN / HEADS, so a loader that derives the head dim instead of reading it shows
const HEAD_DIM: usize = 32;
const INTERMEDIATE: usize = 96;
const LAYERS: usize = 3;
const PACK: usize = 2;

/// Per-layer bytes at F32 unpacked and with `PACK`, then (kv heads, k head dim, v head dim).
fn sizing(
    loader: &dyn DeviceMappedModelLoader,
    config: &Value,
) -> ((usize, usize), (usize, usize, usize)) {
    let config = config.to_string();
    let sizes = |pack| {
        let sizes = loader
            .layer_sizes_in_bytes(&config, DType::F32, pack, None)
            .unwrap();
        assert_eq!(sizes.len(), LAYERS);
        assert!(
            sizes.iter().all(|&size| size == sizes[0]),
            "layers differ: {sizes:?}"
        );
        sizes[0]
    };
    let meta = loader.model_config(&config).unwrap();
    (
        (sizes(1), sizes(PACK)),
        (meta.num_kv_heads(), meta.k_head_dim(), meta.v_head_dim()),
    )
}

fn llama_text() -> Value {
    json!({
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": 128,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 256,
        "rope_scaling": null,
        "quantization_config": null,
        "tie_word_embeddings": false,
    })
}

#[test]
fn llama_sizing() {
    assert_eq!(
        sizing(&LlamaLoader, &llama_text()),
        ((123392, 61952), (2, 16, 16))
    );
}

#[test]
fn mistral_sizing() {
    let mut config = llama_text();
    config["sliding_window"] = json!(null);
    config["head_dim"] = json!(HEAD_DIM);
    assert_eq!(
        sizing(&MistralLoader, &config),
        // sized with the config head_dim the model builds with; master used hidden_size / heads (123392)
        ((172544, 86528), (2, 32, 32))
    );
}

fn mistral_text() -> Value {
    let mut config = llama_text();
    config["sliding_window"] = json!(null);
    config["head_dim"] = json!(HEAD_DIM);
    config["model_type"] = json!("mistral");
    config
}

#[test]
fn smollm3_sizing() {
    let mut config = llama_text();
    config["no_rope_layers"] = json!(null);
    config["no_rope_layer_interval"] = json!(4);
    config["head_dim"] = json!(HEAD_DIM);
    assert_eq!(
        sizing(&SmolLm3Loader, &config),
        ((123392, 61952), (2, 16, 16))
    );
}

#[test]
fn idefics2_sizing() {
    let config = json!({
        "perceiver_config": {},
        "vision_config": {},
        "text_config": mistral_text(),
    });
    assert_eq!(
        sizing(&Idefics2Loader, &config),
        ((123392, 61952), (2, 16, 16))
    );
}

#[test]
fn idefics3_sizing() {
    let mut text = llama_text();
    text["head_dim"] = json!(HEAD_DIM);
    let config = json!({
        "image_token_id": 7,
        "scale_factor": 2,
        "text_config": text,
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_channels": 3,
            "image_size": 32,
            "patch_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
        },
    });
    assert_eq!(
        sizing(&Idefics3Loader, &config),
        ((123392, 61952), (2, 16, 16))
    );
}

#[test]
fn mistral3_sizing() {
    let config = json!({
        "image_token_index": 7,
        "multimodal_projector_bias": false,
        "projector_hidden_act": "gelu",
        "spatial_merge_size": 2,
        "vision_feature_layer": -1,
        "text_config": mistral_text(),
        "vision_config": { "image_size": 32, "patch_size": 8, "head_dim": null },
    });
    assert_eq!(
        sizing(&Mistral3Loader, &config),
        // the Mistral text stack builds with the config head_dim; master sized with hidden_size / heads (123392)
        ((172544, 86528), (2, 32, 32))
    );
}

fn llava(model_type: &str) -> Value {
    let mut text = llama_text();
    text["model_type"] = json!(model_type);
    text["sliding_window"] = json!(null);
    text["head_dim"] = json!(HEAD_DIM);
    json!({
        "image_grid_pinpoints": [[32, 32]],
        "projector_hidden_act": "gelu",
        "text_config": text,
        "vision_config": {
            "hidden_size": 32,
            "image_size": 32,
            "intermediate_size": 48,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "patch_size": 8,
        },
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
    })
}

#[test]
fn llava_sizing() {
    for model_type in ["llama", "mistral"] {
        assert_eq!(
            sizing(&LLaVALoader, &llava(model_type)),
            ((123392, 61952), (2, 16, 16))
        );
    }
}

#[test]
fn llava_next_sizing() {
    for model_type in ["llama", "mistral"] {
        assert_eq!(
            sizing(&LLaVANextLoader, &llava(model_type)),
            ((123392, 61952), (2, 16, 16))
        );
    }
}
