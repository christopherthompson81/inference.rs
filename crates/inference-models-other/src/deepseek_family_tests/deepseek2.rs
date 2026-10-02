use anyhow::Result;
use candle_core::Tensor;
use serde_json::{Value, json};

use super::DeepSeekV2Config;
use crate::deepseek_family::MoeGate;
use crate::deepseek_family_tests::{
    Checkpoint, EXPERTS, HEADS, HIDDEN, INTERMEDIATE, KV_LORA, LAYERS, MAX_POS, MOE_INTERMEDIATE,
    N_SHARED, NARROW_V_DIM, NOPE_DIM, Q_LORA, ROPE_DIM, SCALE, Snapshot, TOP_K, V_DIM, VOCAB,
    assert_err_contains, assert_routes, assert_snapshot, load_and_forward, patched, router_input,
    router_vb,
};
use crate::loaders::DeepSeekV2Loader;

fn base_config() -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "n_shared_experts": N_SHARED,
        "n_routed_experts": EXPERTS,
        "routed_scaling_factor": SCALE,
        "topk_method": "greedy",
        "num_experts_per_tok": TOP_K,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "norm_topk_prob": false,
        "scoring_func": "softmax",
        "hidden_act": "silu",
        "max_position_embeddings": MAX_POS,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
        "rope_theta": 10000.0,
        "rope_scaling": null,
        "attention_bias": false,
        "q_lora_rank": null,
        "qk_rope_head_dim": ROPE_DIM,
        "kv_lora_rank": KV_LORA,
        "v_head_dim": V_DIM,
        "qk_nope_head_dim": NOPE_DIM,
        "n_group": 1,
        "topk_group": 1,
    })
}

fn gate(patch: Value) -> Result<MoeGate> {
    let cfg: DeepSeekV2Config = serde_json::from_value(patched(
        base_config(),
        patched(json!({"hidden_size": EXPERTS}), patch),
    ))?;
    Ok(MoeGate::new(&cfg.family(), router_vb(false)?, EXPERTS)?)
}

#[test]
fn router_greedy_scales_softmax_scores() -> Result<()> {
    let routes = gate(json!({}))?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [
            [(1, 0.956_273), (4, 0.580_009)],
            [(3, 1.118_709), (6, 0.613_961)],
        ],
    )
}

#[test]
fn router_greedy_norm_topk_prob_renormalises_and_skips_scale() -> Result<()> {
    let routes = gate(json!({"norm_topk_prob": true}))?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [
            [(1, 0.622_459), (4, 0.377_541)],
            [(3, 0.645_656), (6, 0.354_344)],
        ],
    )
}

#[test]
fn router_group_limited_greedy() -> Result<()> {
    // pins current behaviour; see Run 44
    // the inverted u8 mask is all ones, so every token lands on experts 0 and 1 with zero weight
    let cfg = json!({"topk_method": "group_limited_greedy", "n_group": 4, "topk_group": 2});
    let routes = gate(cfg)?.forward(&router_input()?)?;
    assert_routes(routes, [[(0, 0.0), (1, 0.0)], [(0, 0.0), (1, 0.0)]])
}

#[test]
fn router_group_limited_greedy_norm_topk_prob_errors() -> Result<()> {
    // pins current behaviour; see Run 44
    let cfg = json!({
        "topk_method": "group_limited_greedy",
        "n_group": 4,
        "topk_group": 2,
        "norm_topk_prob": true,
    });
    assert_err_contains(
        gate(cfg)?.forward(&router_input()?),
        "shape mismatch in div",
    );
    Ok(())
}

fn forward_case(q_lora: Option<usize>, split_kv_b: bool, v_dim: usize) -> Result<Tensor> {
    let config = patched(
        base_config(),
        json!({"q_lora_rank": q_lora, "v_head_dim": v_dim}),
    );
    let mut checkpoint = Checkpoint::default();
    checkpoint.skeleton(MOE_INTERMEDIATE * N_SHARED, false);
    checkpoint.mla_attention(q_lora, split_kv_b, v_dim);
    load_and_forward(&DeepSeekV2Loader, &config, &checkpoint)
}

#[test]
fn forward_plain_q() -> Result<()> {
    let expected = Snapshot {
        probes: [-0.3501127, -0.8086991, -0.17928225, 0.2732258],
        sum: -13.745773,
        l2: 16.48927,
    };
    assert_snapshot(&forward_case(None, false, V_DIM)?, &expected)
}

#[test]
fn forward_lora_q() -> Result<()> {
    let expected = Snapshot {
        probes: [-0.3501127, -0.6735035, -0.6895249, -0.4464788],
        sum: -8.021072,
        l2: 16.533241,
    };
    assert_snapshot(&forward_case(Some(Q_LORA), false, V_DIM)?, &expected)
}

#[test]
fn forward_split_kv_b_errors_on_2d_weights() -> Result<()> {
    // pins current behaviour: 2-D k_b/v_b load from safetensors, then the split projection wants 3-D (GGUF) weights
    assert_err_contains(forward_case(Some(Q_LORA), true, V_DIM), "unexpected rank");
    Ok(())
}

// DeepSeek-V2/V3 value heads are narrower than their query heads (128 vs 192).
#[test]
fn forward_narrow_v_head() -> Result<()> {
    let expected = Snapshot {
        probes: [-0.2601225, 1.2632831, -1.8173112, 0.48527986],
        sum: 6.293726,
        l2: 17.189129,
    };
    assert_snapshot(&forward_case(None, false, NARROW_V_DIM)?, &expected)
}

#[test]
fn load_rejects_half_split_kv_b() -> Result<()> {
    let mut checkpoint = Checkpoint::default();
    checkpoint.skeleton(MOE_INTERMEDIATE * N_SHARED, false);
    checkpoint.mla_attention(None, true, V_DIM);
    checkpoint
        .0
        .retain(|name, _| !name.ends_with("v_b_proj.weight"));
    assert_err_contains(
        load_and_forward(&DeepSeekV2Loader, &base_config(), &checkpoint),
        "incomplete split MLA weights",
    );
    Ok(())
}
