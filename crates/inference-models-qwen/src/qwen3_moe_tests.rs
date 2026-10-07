//! Locks Qwen3-MoE's prefill on tiny made-up weights: layer 0 is a dense MLP, layer 1 routes over experts.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest,
};
use inference_tensor::DType;
use serde_json::json;

use crate::loaders::Qwen3MoELoader;

const VOCAB: usize = 64;
const HIDDEN: usize = 32;
const MOE_INTERMEDIATE: usize = 16;
const EXPERTS: usize = 4;
const MOE_LAYER: usize = 1;

// Expert layout detection reads these shapes before any tensor; HF stores the experts one by one.
fn expert_shapes() -> HashMap<String, Vec<usize>> {
    (0..EXPERTS)
        .flat_map(|expert| {
            let p = format!("model.layers.{MOE_LAYER}.mlp.experts.{expert}");
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

#[test]
fn qwen3_moe_prefill() -> Result<()> {
    let config = json!({
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
        "head_dim": 8,
        "quantization_config": null,
        "tie_word_embeddings": false,
        "max_window_layers": 0,
        "use_sliding_window": false,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_experts": EXPERTS,
        "mlp_only_layers": [0],
        "decoder_sparse_step": 1,
        "norm_topk_prob": true,
        "num_experts_per_tok": 2,
    });
    let (model, names) = load_synthesized(&[], expert_shapes(), DType::F32, |vb| {
        Qwen3MoELoader.load(
            &config.to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    assert_eq!(
        names_digest(names.keys()),
        0xf93e_432d_ce05_497d,
        "tensor names moved: {:#x}",
        names_digest(names.keys())
    );
    let expected = Snapshot {
        probes: [0.40193462, 1.4729915, 1.4494103, 1.1353827],
        sum: 48.4729,
        l2: 19.265102,
    };
    assert_snapshot(&forward_normal(model.as_ref())?, VOCAB, &expected)
}
