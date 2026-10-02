//! Locks Qwen3.5 dense and MoE text-only prefill on tiny made-up weights, so sharing their text model moves nothing
//! it does not mean to.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Tensor};
use inference_nn::loaders::MultimodalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_multimodal, load_synthesized, metadata, names_digest,
    patched,
};
use serde_json::{Value, json};

use crate::loaders::{Qwen3_5Loader, Qwen3_5MoeLoader};

const VOCAB: usize = 64;
const HIDDEN: usize = 64;
const LAYERS: usize = 4;
const MOE_INTERMEDIATE: usize = 32;
const EXPERTS: usize = 4;
// The alternate MLX layout the loaders probe for; leaving it absent selects the HF names.
const MLX_NAMES: &[&str] = &["vision_tower.", "language_model."];

fn text() -> Value {
    json!({
        "head_dim": 32,
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "hidden_act": "silu",
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1]
        },
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "tie_word_embeddings": false,
    })
}

fn moe_text() -> Value {
    patched(
        text(),
        json!({
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "shared_expert_intermediate_size": 64,
            "num_experts": EXPERTS,
            "num_experts_per_tok": 2,
        }),
    )
}

fn config(text: Value) -> Value {
    json!({
        "text_config": text,
        "vision_config": {
            "depth": 2,
            "hidden_size": 32,
            "out_hidden_size": HIDDEN,
            "intermediate_size": 48,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        "image_token_id": 60,
        "video_token_id": 61,
        "vision_start_token_id": 62,
        "vision_end_token_id": 63,
        "tie_word_embeddings": false,
        "quantization_config": null,
    })
}

// HF stores the Qwen3.5 MoE experts one by one; layout detection reads these shapes before any tensor.
fn expert_shapes() -> HashMap<String, Vec<usize>> {
    (0..LAYERS)
        .flat_map(|layer| (0..EXPERTS).map(move |expert| (layer, expert)))
        .flat_map(|(layer, expert)| {
            let p = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
            [
                (
                    format!("{p}.gate_proj.weight"),
                    vec![MOE_INTERMEDIATE, HIDDEN],
                ),
                (
                    format!("{p}.up_proj.weight"),
                    vec![MOE_INTERMEDIATE, HIDDEN],
                ),
                (
                    format!("{p}.down_proj.weight"),
                    vec![HIDDEN, MOE_INTERMEDIATE],
                ),
            ]
        })
        .collect()
}

fn prefill(
    loader: &dyn MultimodalModelLoader,
    config: &Value,
    shapes: HashMap<String, Vec<usize>>,
    dtype: DType,
) -> Result<(Tensor, u64)> {
    let (model, seen) = load_synthesized(MLX_NAMES, shapes, dtype, |vb| {
        loader.load(
            &config.to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    Ok((forward_multimodal(model.as_ref())?, names_digest(&seen)))
}

fn check(logits: Tensor, digest: u64, names: u64, expected: Snapshot) -> Result<()> {
    assert_eq!(digest, names, "tensor names moved: {digest:#x}");
    assert_snapshot(&logits, VOCAB, &expected)
}

#[test]
fn qwen3_5_dense_prefill() -> Result<()> {
    let (logits, digest) = prefill(&Qwen3_5Loader, &config(text()), HashMap::new(), DType::F32)?;
    check(
        logits,
        digest,
        0xf2c1_663c_5a1e_bf86,
        Snapshot {
            probes: [-0.94613516, 0.67169124, 3.330963, 4.801997],
            sum: 21.30183,
            l2: 50.22795,
        },
    )
}

#[test]
fn qwen3_5_moe_prefill() -> Result<()> {
    let (logits, digest) = prefill(
        &Qwen3_5MoeLoader,
        &config(moe_text()),
        expert_shapes(),
        DType::F32,
    )?;
    check(
        logits,
        digest,
        0xdcb1_500e_458e_6eb2,
        Snapshot {
            probes: [-1.1597177, -0.7348413, 4.277955, 6.7439075],
            sum: 32.445145,
            l2: 52.39406,
        },
    )
}

#[test]
fn qwen3_5_moe_prefill_bf16() -> Result<()> {
    let (logits, digest) = prefill(
        &Qwen3_5MoeLoader,
        &config(moe_text()),
        expert_shapes(),
        DType::BF16,
    )?;
    check(
        logits,
        digest,
        0xdcb1_500e_458e_6eb2,
        Snapshot {
            probes: [-1.1015625, -0.6640625, 4.3125, 6.625],
            sum: 32.989548,
            l2: 52.179066,
        },
    )
}
