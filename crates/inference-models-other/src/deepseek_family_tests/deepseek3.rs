use anyhow::Result;
use candle_core::Tensor;
use serde_json::{Value, json};

use super::DeepSeekV3Config;
use crate::deepseek_family::MoeGate;
use crate::deepseek_family_tests::{
    Checkpoint, EXPERTS, FamilyCheckpoint, HEADS, HIDDEN, INTERMEDIATE, KV_LORA, LAYERS, MAX_POS,
    MOE_INTERMEDIATE, N_SHARED, NOPE_DIM, Q_LORA, ROPE_DIM, SCALE, Snapshot, TOP_K, V_DIM, VOCAB,
    assert_err_contains, assert_routes, assert_snapshot, load_and_forward, patched, router_input,
    router_vb,
};
use crate::loaders::DeepSeekV3Loader;

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
        "topk_method": "noaux_tc",
        "num_experts_per_tok": TOP_K,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "scoring_func": "sigmoid",
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
        "n_group": 4,
        "topk_group": 2,
    })
}

fn gate(patch: Value, bias: bool) -> Result<MoeGate> {
    let cfg: DeepSeekV3Config = serde_json::from_value(patched(
        base_config(),
        patched(json!({"hidden_size": EXPERTS}), patch),
    ))?;
    Ok(MoeGate::new(&cfg.family(), router_vb(bias)?, EXPERTS)?)
}

#[test]
fn router_greedy_softmax_scales_without_renormalising() -> Result<()> {
    let cfg = json!({
        "topk_method": "greedy",
        "scoring_func": "softmax",
        "norm_topk_prob": false,
    });
    let routes = gate(cfg, false)?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [
            [(1, 0.956273), (4, 0.580009)],
            [(3, 1.118709), (6, 0.613961)],
        ],
    )
}

#[test]
fn router_greedy_softmax_norm_topk_prob_renormalises_then_scales() -> Result<()> {
    let cfg = json!({"topk_method": "greedy", "scoring_func": "softmax"});
    let routes = gate(cfg, false)?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [
            [(1, 1.556_148), (4, 0.943_852)],
            [(3, 1.614_14), (6, 0.885_86)],
        ],
    )
}

#[test]
fn router_greedy_sigmoid_renormalises_then_scales() -> Result<()> {
    let routes = gate(json!({"topk_method": "greedy"}), false)?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [
            [(1, 1.296532), (4, 1.203468)],
            [(3, 1.287799), (6, 1.212201)],
        ],
    )
}

#[test]
fn router_noaux_tc_sigmoid_with_bias() -> Result<()> {
    let routes = gate(json!({}), true)?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [
            [(4, 1.551287), (6, 0.948713)],
            [(2, 1.105823), (6, 1.394177)],
        ],
    )
}

#[test]
fn router_noaux_tc_requires_bias() -> Result<()> {
    assert_err_contains(gate(json!({}), false), "e_score_correction_bias");
    Ok(())
}

#[test]
fn router_group_limited_greedy_picks_within_the_best_group() -> Result<()> {
    let cfg = json!({
        "topk_method": "group_limited_greedy",
        "scoring_func": "softmax",
        "norm_topk_prob": false,
        "n_group": 4,
        "topk_group": 1,
    });
    let routes = gate(cfg, false)?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [
            [(0, 0.143_028), (1, 0.956_273)],
            [(2, 0.204_37), (3, 1.118_709)],
        ],
    )
}

fn forward_case(q_lora: Option<usize>, split_kv_b: bool) -> Result<Tensor> {
    let config = patched(base_config(), json!({"q_lora_rank": q_lora}));
    let mut checkpoint = Checkpoint::default();
    checkpoint.skeleton(MOE_INTERMEDIATE * N_SHARED, true);
    checkpoint.mla_attention(q_lora, split_kv_b, V_DIM);
    load_and_forward(&DeepSeekV3Loader, &config, &checkpoint)
}

#[test]
fn forward_plain_q() -> Result<()> {
    let expected = Snapshot {
        probes: [-0.6860722, -0.36114687, -0.17502864, 0.42829013],
        sum: -2.8865871,
        l2: 16.895712,
    };
    assert_snapshot(&forward_case(None, false)?, &expected)
}

#[test]
fn forward_lora_q() -> Result<()> {
    let expected = Snapshot {
        probes: [-0.6860722, -0.22463873, -0.6333456, -0.32590142],
        sum: 0.9085331,
        l2: 17.034044,
    };
    assert_snapshot(&forward_case(Some(Q_LORA), false)?, &expected)
}

#[test]
fn forward_split_kv_b_errors_on_2d_weights() -> Result<()> {
    // pins current behaviour: 2-D k_b/v_b load from safetensors, then the split projection wants 3-D (GGUF) weights
    assert_err_contains(forward_case(None, true), "unexpected rank");
    Ok(())
}
