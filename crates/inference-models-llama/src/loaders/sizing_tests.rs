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
        ((123392, 61952), (2, 32, 32))
    );
}
