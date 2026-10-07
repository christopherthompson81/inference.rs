//! Pins each loader's device-map layer sizes and KV planning metadata; a value that moves is a drift to verify.

use serde_json::{Value, json};

use inference_nn::testing::{LoaderSizing, loader_sizing};

use super::*;

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
// differs from HIDDEN / HEADS, so a loader that derives the head dim instead of reading it shows
const HEAD_DIM: usize = 32;
const INTERMEDIATE: usize = 96;
const LAYERS: usize = 3;
const PACK: usize = 2;

fn sizing(loader: &dyn DeviceMappedModelLoader, config: &Value) -> LoaderSizing {
    loader_sizing(loader, config, PACK)
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
        ((123392, 61952), (2, 16, 16), (256, 3, 64, 4))
    );
}

#[test]
fn llama_sizing_reads_head_dim() {
    let mut config = llama_text();
    config["head_dim"] = json!(HEAD_DIM);
    assert_eq!(
        sizing(&LlamaLoader, &config),
        ((172544, 86528), (2, 32, 32), (256, 3, 64, 4))
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
        ((172544, 86528), (2, 32, 32), (256, 3, 64, 4))
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
        ((172544, 86528), (2, 32, 32), (256, 3, 64, 4))
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
        ((123392, 61952), (2, 16, 16), (256, 3, 64, 4))
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
        ((172544, 86528), (2, 32, 32), (256, 3, 64, 4))
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
        ((172544, 86528), (2, 32, 32), (256, 3, 64, 4))
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
            ((172544, 86528), (2, 32, 32), (256, 3, 64, 4))
        );
    }
}

#[test]
fn llava_next_sizing() {
    for model_type in ["llama", "mistral"] {
        assert_eq!(
            sizing(&LLaVANextLoader, &llava(model_type)),
            ((172544, 86528), (2, 32, 32), (256, 3, 64, 4))
        );
    }
}

// The vision tower is non-mapped, so the device map sees it only through non_mapped_size_in_bytes.
#[test]
fn idefics2_sizes_every_vision_layer() {
    const VISION_HIDDEN: usize = 32;
    const VISION_INTERMEDIATE: usize = 48;
    let non_mapped = |layers: usize| {
        let config = json!({
            "perceiver_config": {},
            "vision_config": {
                "hidden_size": VISION_HIDDEN,
                "intermediate_size": VISION_INTERMEDIATE,
                "num_hidden_layers": layers,
                "num_attention_heads": 2,
                "image_size": 32,
                "patch_size": 8,
            },
            "text_config": mistral_text(),
        });
        Idefics2Loader
            .non_mapped_size_in_bytes(&config.to_string(), DType::F32, 1, None, None)
            .unwrap()
    };
    let (h, i) = (VISION_HIDDEN, VISION_INTERMEDIATE);
    // two biased layer norms, biased fc1/fc2 and four biased attention projections
    let layer_elems = 4 * h + (h * i + i) + (i * h + h) + 4 * (h * h + h);
    assert_eq!(
        non_mapped(3) - non_mapped(1),
        2 * layer_elems * DType::F32.size_in_bytes()
    );
}
