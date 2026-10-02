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

fn gemma_text() -> Value {
    json!({
        "attention_bias": false,
        "head_dim": HEAD_DIM,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "quantization_config": null,
        "tie_word_embeddings": true,
    })
}

fn gemma2_text() -> Value {
    let mut config = gemma_text();
    config["sliding_window"] = json!(16);
    config["attn_logit_softcapping"] = json!(50.0);
    config["final_logit_softcapping"] = json!(30.0);
    config["query_pre_attn_scalar"] = json!(HEAD_DIM);
    config
}

#[test]
fn gemma_sizing() {
    let mut config = gemma_text();
    config["attention_bias"] = json!(true);
    assert_eq!(
        sizing(&GemmaLoader, &config),
        ((173824, 87808), (2, 32, 32))
    );
}

#[test]
fn gemma2_sizing() {
    assert_eq!(
        sizing(&Gemma2Loader, &gemma2_text()),
        // pre/post feedforward norms counted and KV sized with the config head_dim; master: 172544 and 16
        ((173056, 87040), (2, 32, 32))
    );
}

#[test]
fn embedding_gemma_sizing() {
    assert_eq!(
        sizing(&EmbeddingGemmaLoader, &gemma2_text()),
        // feedforward and q/k norms counted, KV sized with the config head_dim; master: 172544 and 16
        ((173312, 87296), (2, 32, 32))
    );
}

#[test]
fn gemma3_sizing() {
    assert_eq!(
        sizing(&Gemma3Loader, &gemma2_text()),
        // pre/post feedforward and q/k norms counted; master: 172544
        ((173312, 87296), (2, 32, 32))
    );
}
