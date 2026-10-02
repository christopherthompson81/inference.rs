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

fn qwen2_text() -> Value {
    json!({
        "vocab_size": 128,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "max_position_embeddings": 256,
        "sliding_window": null,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "quantization_config": null,
        "tie_word_embeddings": false,
    })
}

fn qwen3_text() -> Value {
    let mut config = qwen2_text();
    config["head_dim"] = json!(HEAD_DIM);
    config["max_window_layers"] = json!(LAYERS);
    config["use_sliding_window"] = json!(false);
    config
}

fn qwen_vl(vision_config: Value) -> Value {
    let mut config = qwen2_text();
    config["vision_config"] = vision_config;
    config["rope_scaling"] = json!({ "mrope_section": [2, 3, 3] });
    config["image_token_id"] = json!(100);
    config["video_token_id"] = json!(101);
    config
}

#[test]
fn qwen2_sizing() {
    assert_eq!(
        sizing(&Qwen2Loader, &qwen2_text()),
        ((123904, 62464), (2, 16, 16), (256, 3, 64, 4))
    );
}

#[test]
fn qwen3_sizing() {
    assert_eq!(
        sizing(&Qwen3Loader, &qwen3_text()),
        // q/k/v are built without bias and KV is planned with the config head_dim; master: ((173824, 87808), 16)
        ((172800, 86784), (2, 32, 32), (256, 3, 64, 4))
    );
}

#[test]
fn qwen3_embedding_sizing() {
    assert_eq!(
        sizing(&Qwen3EmbeddingLoader, &qwen3_text()),
        // q/k/v are built without bias and KV is planned with the config head_dim; master: ((173824, 87808), 16)
        ((172800, 86784), (2, 32, 32), (256, 3, 64, 4))
    );
}

#[test]
fn qwen2vl_sizing() {
    let config = qwen_vl(json!({
        "depth": 2,
        "embed_dim": 32,
        "hidden_size": HIDDEN,
        "mlp_ratio": 2.0,
        "num_heads": 2,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }));
    assert_eq!(
        sizing(&Qwen2VLLoader, &config),
        ((123904, 62464), (2, 16, 16), (256, 3, 64, 4))
    );
}

#[test]
fn qwen2_5vl_sizing() {
    let config = qwen_vl(json!({ "depth": 2, "hidden_size": 32, "out_hidden_size": HIDDEN }));
    assert_eq!(
        sizing(&Qwen2_5VLLoader, &config),
        ((123904, 62464), (2, 16, 16), (256, 3, 64, 4))
    );
}

#[test]
fn qwen3vl_sizing() {
    let mut text_config = qwen3_text();
    text_config["rope_scaling"] = json!({ "mrope_section": [8, 4, 4] });
    let config = json!({
        "text_config": text_config,
        "vision_config": { "depth": 2, "hidden_size": 32, "out_hidden_size": HIDDEN },
        "image_token_id": 100,
        "video_token_id": 101,
        "vision_start_token_id": 102,
        "vision_end_token_id": 103,
        "tie_word_embeddings": false,
        "quantization_config": null,
    });
    assert_eq!(
        sizing(&Qwen3VLLoader, &config),
        ((172800, 86784), (2, 32, 32), (256, 3, 64, 4))
    );
}

#[test]
fn minicpm_o_sizing() {
    let mut config = qwen2_text();
    config["vision_config"] = json!({ "hidden_size": 32, "num_hidden_layers": 2 });
    config["vision_batch_size"] = json!(1);
    config["query_num"] = json!(4);
    assert_eq!(
        sizing(&MiniCpmOLoader, &config),
        // the Qwen2 LLM it builds has q/k/v biases master left out (123392, 61952)
        ((123904, 62464), (2, 16, 16), (256, 3, 64, 4))
    );
}
