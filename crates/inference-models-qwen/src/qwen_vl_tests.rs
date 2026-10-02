//! Locks the Qwen-VL family's text-only prefill on tiny made-up weights, so sharing its code cannot move a logit.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Tensor};
use inference_nn::loaders::MultimodalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_err_contains, assert_snapshot, forward_multimodal, load_synthesized, metadata,
    names_digest, patched,
};
use serde_json::{Value, json};

use crate::loaders::{Qwen2_5VLLoader, Qwen2VLLoader, Qwen3VLLoader, Qwen3VLMoELoader};

const VOCAB: usize = 64;
const HIDDEN: usize = 32;
const HEAD_DIM: usize = 8;
const MOE_INTERMEDIATE: usize = 16;
const EXPERTS: usize = 4;
// The alternate MLX layout the loaders probe for; leaving it absent selects the HF names.
const MLX_NAMES: &[&str] = &["vision_tower.", "language_model."];

fn vision_config() -> Value {
    json!({
        "depth": 2,
        "hidden_size": 16,
        "out_hidden_size": HIDDEN,
        "intermediate_size": 24,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
    })
}

fn qwen3_vl_text() -> Value {
    json!({
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": null,
        "rope_scaling": {"mrope_section": [2, 1, 1]},
    })
}

fn qwen3_vl_config(text: Value) -> Value {
    json!({
        "text_config": text,
        "vision_config": vision_config(),
        "image_token_id": 60,
        "video_token_id": 61,
        "vision_start_token_id": 62,
        "vision_end_token_id": 63,
        "tie_word_embeddings": false,
        "quantization_config": null,
    })
}

/// Logits of a text-only prefill and the digest of every tensor name the load read.
fn prefill_as(
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
    // ISQ and UQFF serialise residual tensors under these names, so each must be one the load read
    let stray = model
        .residual_tensors()
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| !seen.contains(name))
        .collect::<Vec<_>>();
    assert!(stray.is_empty(), "residual tensors never loaded: {stray:?}");
    Ok((forward_multimodal(model.as_ref())?, names_digest(&seen)))
}

fn prefill(
    loader: &dyn MultimodalModelLoader,
    config: &Value,
    shapes: HashMap<String, Vec<usize>>,
    names: u64,
) -> Result<Tensor> {
    let (logits, digest) = prefill_as(loader, config, shapes, DType::F32)?;
    assert_eq!(digest, names, "tensor names moved: {digest:#x}");
    Ok(logits)
}

fn moe_text() -> Value {
    patched(
        qwen3_vl_text(),
        json!({
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "num_experts": EXPERTS,
            "num_experts_per_tok": 2,
            "mlp_only_layers": [0],
        }),
    )
}

// HF stacks the experts transposed: gate_up [E, H, 2I], down [E, I, H].
fn moe_shapes() -> HashMap<String, Vec<usize>> {
    let experts = "model.language_model.layers.1.mlp.experts";
    HashMap::from([
        (
            format!("{experts}.gate_up_proj"),
            vec![EXPERTS, HIDDEN, 2 * MOE_INTERMEDIATE],
        ),
        (
            format!("{experts}.down_proj"),
            vec![EXPERTS, MOE_INTERMEDIATE, HIDDEN],
        ),
    ])
}

#[test]
fn qwen3_vl_dense_prefill() -> Result<()> {
    let logits = prefill(
        &Qwen3VLLoader,
        &qwen3_vl_config(qwen3_vl_text()),
        HashMap::new(),
        0xf25b_496e_18c5_1e04,
    )?;
    let expected = Snapshot {
        probes: [1.2218488, -0.56189346, 0.6537559, -0.30086958],
        sum: -27.892378,
        l2: 17.84988,
    };
    assert_snapshot(&logits, VOCAB, &expected)
}

#[test]
fn qwen3_vl_moe_prefill() -> Result<()> {
    let logits = prefill(
        &Qwen3VLMoELoader,
        &qwen3_vl_config(moe_text()),
        moe_shapes(),
        0xc590_3089_c86b_c974,
    )?;
    let expected = Snapshot {
        probes: [1.3302258, 0.18510357, 0.27132797, -0.688689],
        sum: -38.30021,
        l2: 17.858477,
    };
    assert_snapshot(&logits, VOCAB, &expected)
}

fn qwen2_vl_config(vision: Value) -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
        "rope_theta": 10000.0,
        "sliding_window": null,
        "vision_config": vision,
        "rope_scaling": {"mrope_section": [2, 1, 1]},
        "quantization_config": null,
        "image_token_id": 60,
        "video_token_id": 61,
    })
}

#[test]
fn qwen2_vl_prefill() -> Result<()> {
    let vision = json!({
        "depth": 2,
        "embed_dim": 16,
        "hidden_size": HIDDEN,
        "mlp_ratio": 2.0,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    });
    let logits = prefill(
        &Qwen2VLLoader,
        &qwen2_vl_config(vision),
        HashMap::new(),
        0xdc45_cab7_7534_b3cd,
    )?;
    let expected = Snapshot {
        probes: [0.10101704, 1.5707272, 0.5929547, 0.636466],
        sum: 35.159973,
        l2: 17.545603,
    };
    assert_snapshot(&logits, VOCAB, &expected)
}

#[test]
fn qwen2_5_vl_prefill() -> Result<()> {
    let vision = json!({
        "depth": 2,
        "hidden_size": 16,
        "out_hidden_size": HIDDEN,
        "intermediate_size": 24,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 8,
        "fullatt_block_indexes": [1],
    });
    let logits = prefill(
        &Qwen2_5VLLoader,
        &qwen2_vl_config(vision),
        HashMap::new(),
        0x5059_ed84_eb91_5e82,
    )?;
    let expected = Snapshot {
        probes: [0.10101704, 1.5707272, 0.5929547, 0.636466],
        sum: 35.159973,
        l2: 17.545603,
    };
    assert_snapshot(&logits, VOCAB, &expected)
}

// In BF16 the dense model's F32 norms and the MoE model's fused ones round differently, so these pin which each gets.
#[test]
fn qwen3_vl_dense_prefill_bf16() -> Result<()> {
    let config = qwen3_vl_config(qwen3_vl_text());
    let (logits, _) = prefill_as(&Qwen3VLLoader, &config, HashMap::new(), DType::BF16)?;
    let expected = Snapshot {
        probes: [1.2421875, -0.55859375, 0.66015625, -0.29296875],
        sum: -27.897491,
        l2: 17.8422,
    };
    assert_snapshot(&logits, VOCAB, &expected)
}

#[test]
fn qwen3_vl_moe_prefill_bf16() -> Result<()> {
    let config = qwen3_vl_config(moe_text());
    let (logits, _) = prefill_as(&Qwen3VLMoELoader, &config, moe_shapes(), DType::BF16)?;
    let expected = Snapshot {
        probes: [1.34375, 0.20703125, 0.28125, -0.68359375],
        sum: -38.45215,
        l2: 17.85445,
    };
    assert_snapshot(&logits, VOCAB, &expected)
}

#[test]
fn qwen3_vl_loaders_reject_the_other_expert_layout() {
    let load = |loader: &dyn MultimodalModelLoader, text: Value| {
        prefill_as(loader, &qwen3_vl_config(text), moe_shapes(), DType::F32)
    };
    assert_err_contains(load(&Qwen3VLLoader, moe_text()), "load it as qwen3vlmoe");
    assert_err_contains(
        load(&Qwen3VLMoELoader, qwen3_vl_text()),
        "needs nonzero num_experts",
    );
    let no_top_k = patched(moe_text(), json!({"num_experts_per_tok": 0}));
    assert_err_contains(
        load(&Qwen3VLMoELoader, no_top_k),
        "needs nonzero num_experts",
    );
    let no_step = patched(moe_text(), json!({"decoder_sparse_step": 0}));
    assert_err_contains(
        load(&Qwen3VLMoELoader, no_step),
        "needs nonzero num_experts",
    );
}
