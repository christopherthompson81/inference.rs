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

fn decoder_text() -> Value {
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
fn glm4_sizing() {
    let mut config = decoder_text();
    config["sliding_window"] = json!(null);
    config["partial_rotary_factor"] = json!(0.5);
    config["head_dim"] = json!(HEAD_DIM);
    config["attention_bias"] = json!(true);
    assert_eq!(
        sizing(&GLM4Loader, &config),
        // KV metadata reports the config head_dim the model builds with; master reported hidden_size / heads (16)
        ((174080, 88064), (2, 32, 32), (256, 3, 64, 4))
    );
    config["attention_bias"] = json!(null);
    assert_eq!(
        sizing(&GLM4Loader, &config),
        // q/k/v biases only when attention_bias is set, as the model loads them; master always counted them (174080)
        ((173056, 87040), (2, 32, 32), (256, 3, 64, 4))
    );
}

#[test]
fn starcoder2_sizing() {
    let mut config = decoder_text();
    config["hidden_act"] = json!("gelu_pytorch_tanh");
    config["norm_epsilon"] = json!(1e-5);
    config["use_bias"] = json!(true);
    config["sliding_window"] = json!(null);
    assert_eq!(
        sizing(&Starcoder2Loader, &config),
        ((100736, 51584), (2, 16, 16), (256, 3, 64, 4))
    );
}

#[test]
fn hunyuan_v1_dense_sizing() {
    let mut config = decoder_text();
    config["head_dim"] = json!(HEAD_DIM);
    assert_eq!(
        sizing(&HunYuanDenseV1Loader, &config),
        ((172800, 86784), (2, 32, 32), (256, 3, 64, 4))
    );
}

#[test]
fn paddleocr_vl_text_sizing() {
    let mut config: Value =
        serde_json::from_str(include_str!("../paddleocr_vl/reference_config.json")).unwrap();
    for (key, value) in decoder_text().as_object().unwrap() {
        config[key] = value.clone();
    }
    config["head_dim"] = json!(HEAD_DIM);
    config["rope_scaling"] = json!({"mrope_section": [4, 6, 6], "rope_type": "default"});
    assert_eq!(
        sizing(&PaddleOcrVlLoader, &config),
        ((172544, 86528), (2, 32, 32), (256, 3, 64, 4))
    );
}
