//! Locks the prefill of Gemma, Gemma 2 and Gemma 3 text, and EmbeddingGemma's hidden states, on tiny made-up weights.

use anyhow::Result;
use inference_nn::attention::FlashParams;
use inference_nn::loaders::{EmbeddingModelLoader, MultimodalModelLoader, NormalModelLoader};
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    PROMPT, Snapshot, assert_snapshot, forward_multimodal, forward_normal, load_synthesized,
    metadata, names_digest, patched,
};
use inference_tensor::{DType, Device, Tensor};
use serde_json::{Value, json};

use crate::loaders::{EmbeddingGemmaLoader, Gemma2Loader, Gemma3Loader, GemmaLoader};

const VOCAB: usize = 64;

fn prefill(
    loader: &dyn NormalModelLoader,
    config: &Value,
    expected_names: u64,
    expected: &Snapshot,
) -> Result<()> {
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
        loader.load(
            &config.to_string(),
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
    assert_snapshot(&forward_normal(model.as_ref())?, VOCAB, expected)
}

// The embedder's hidden states stand in for logits, so its width has to cover the snapshot probes
const EMBEDDING_HIDDEN: usize = 64;
// Shorter than the prompt, so the sliding layers see less than the full ones
const WINDOW: usize = 2;

fn gemma_config() -> Value {
    json!({
        "attention_bias": false,
        "head_dim": 8,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "quantization_config": null,
        "tie_word_embeddings": true,
    })
}

fn gemma2_config() -> Value {
    patched(
        gemma_config(),
        json!({
            "sliding_window": WINDOW,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 8,
        }),
    )
}

// Layer 0 slides with the local rope, layer 1 attends fully with the global one.
fn gemma3_config() -> Value {
    patched(
        gemma2_config(),
        json!({"sliding_window_pattern": 2, "rope_local_base_freq": 1000.0, "rope_scaling": null}),
    )
}

#[test]
fn gemma_prefill() -> Result<()> {
    prefill(
        &GemmaLoader,
        &gemma_config(),
        0x55d2_5ca4_fbf1_9e2e,
        &Snapshot {
            probes: [-1.2407782, -0.8033775, 0.032351747, -1.8356842],
            sum: -40.83593,
            l2: 36.104527,
        },
    )
}

#[test]
fn gemma2_prefill_with_softcaps_and_a_sliding_layer() -> Result<()> {
    prefill(
        &Gemma2Loader,
        &gemma2_config(),
        0x0225_2f88_a6bc_e06a,
        &Snapshot {
            probes: [-2.0220714, 0.31683192, 1.3942876, 0.14704038],
            sum: -7.529931,
            l2: 33.83025,
        },
    )
}

#[test]
fn gemma3_text_prefill() -> Result<()> {
    let config = gemma3_config().to_string();
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
        Gemma3Loader.load(&config, vb, metadata(), AttentionImplementation::Eager)
    })?;
    assert_eq!(
        names_digest(names.keys()),
        0xa0fa_41ef_411d_7524,
        "tensor names moved: {:#x}",
        names_digest(names.keys())
    );
    assert_snapshot(
        &forward_multimodal(model.as_ref())?,
        VOCAB,
        &Snapshot {
            probes: [-2.0220714, 0.3051516, 1.041176, 0.13264762],
            sum: -7.0928936,
            l2: 34.50471,
        },
    )
}

#[test]
fn embedding_gemma_hidden_states() -> Result<()> {
    let config = patched(gemma3_config(), json!({"hidden_size": EMBEDDING_HIDDEN})).to_string();
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
        EmbeddingGemmaLoader.load(&config, vb, metadata(), AttentionImplementation::Eager)
    })?;
    assert_eq!(
        names_digest(names.keys()),
        0x9628_17a6_3ecd_ae98,
        "tensor names moved: {:#x}",
        names_digest(names.keys())
    );
    let input = Tensor::new(&PROMPT, &Device::Cpu)?.unsqueeze(0)?;
    let hidden = model
        .forward(&input, &FlashParams::empty(false))?
        .to_dtype(DType::F32)?;
    assert_snapshot(
        &hidden,
        EMBEDDING_HIDDEN,
        &Snapshot {
            probes: [1.1804714, -2.3619092, -2.6457171, 1.6340134],
            sum: 22.992924,
            l2: 35.195293,
        },
    )
}
