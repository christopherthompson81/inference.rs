//! Pins each loader's device-map layer sizes and KV planning metadata, so sharing the sizing code moves nothing
//! silently; a value that changes is a drift against the model code.

use serde_json::{Value, json};

use super::*;

const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
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

fn phi2_text(qk_layernorm: bool) -> Value {
    json!({
        "vocab_size": 128,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "hidden_act": "gelu_new",
        "max_position_embeddings": 256,
        "layer_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "qk_layernorm": qk_layernorm,
        "quantization_config": null,
    })
}

fn phi3_text() -> Value {
    json!({
        "vocab_size": 128,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": null,
        "max_position_embeddings": 256,
        "sliding_window": null,
        "original_max_position_embeddings": 256,
        "quantization_config": null,
        "partial_rotary_factor": null,
    })
}

#[test]
fn phi2_sizing() {
    assert_eq!(
        sizing(&Phi2Loader, &phi2_text(false)),
        // counts the fc1/fc2 biases the model loads; master left them out (99584, 50432)
        ((100224, 51072), (2, 16, 16))
    );
    assert_eq!(
        sizing(&Phi2Loader, &phi2_text(true)),
        // also counts the q/k layernorm biases; master had (99712, 50560)
        ((100480, 51328), (2, 16, 16))
    );
}

#[test]
fn phi3_sizing() {
    assert_eq!(
        sizing(&Phi3Loader, &phi3_text()),
        // o_proj is built without a bias; master counted one (123648, 62208)
        ((123392, 61952), (2, 16, 16))
    );
}

#[test]
fn phi3v_sizing() {
    let mut config = phi3_text();
    config["embd_layer"] = json!({ "projection_cls": "linear" });
    config["img_processor"] = json!({
        "image_dim_out": 32,
        "name": "clip_vision_model",
        "num_img_tokens": 16,
    });
    assert_eq!(
        sizing(&Phi3VLoader, &config),
        ((123392, 61952), (2, 16, 16))
    );
}

#[test]
fn phi4mm_sizing() {
    let lora = json!({ "layer": "layers", "lora_alpha": 1.0, "r": 0 });
    let mut config = phi3_text();
    config["resid_pdrop"] = json!(0.0);
    config["embd_pdrop"] = json!(0.0);
    config["attention_dropout"] = json!(0.0);
    config["initializer_range"] = json!(0.02);
    config["use_cache"] = json!(true);
    config["tie_word_embeddings"] = json!(true);
    config["partial_rotary_factor"] = json!(0.75);
    config["bos_token_id"] = json!(1);
    config["eos_token_id"] = json!(2);
    config["pad_token_id"] = json!(0);
    config["embd_layer"] = json!({ "image_embd_layer": null, "audio_embd_layer": null });
    config["vision_lora"] = lora.clone();
    config["speech_lora"] = lora;
    assert_eq!(
        sizing(&Phi4MMLoader, &config),
        ((123392, 61952), (2, 16, 16))
    );
}
