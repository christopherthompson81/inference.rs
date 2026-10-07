use anyhow::Result;
use inference_tensor::Tensor;
use serde_json::{Value, json};

use super::Glm4MoeLiteConfig;
use crate::deepseek_family::MoeGate;
use crate::deepseek_family_tests::{
    Checkpoint, EXPERTS, FamilyCheckpoint, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, KV_LORA, LAYERS,
    MAX_POS, MOE_INTERMEDIATE, N_SHARED, NOPE_DIM, Q_LORA, ROPE_DIM, SCALE, Snapshot, TOP_K, V_DIM,
    VOCAB, assert_err_contains, assert_routes, assert_snapshot, load_and_forward, patched,
    router_input, router_vb,
};
use crate::loaders::GLM4MoeLiteLoader;

fn base_config() -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA,
        "qk_nope_head_dim": NOPE_DIM,
        "qk_rope_head_dim": ROPE_DIM,
        "v_head_dim": V_DIM,
        "n_routed_experts": EXPERTS,
        "n_shared_experts": N_SHARED,
        "num_experts_per_tok": TOP_K,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": SCALE,
        "n_group": 4,
        "topk_group": 2,
        "moe_layer_freq": 1,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": MAX_POS,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
    })
}

fn gate(bias: bool) -> Result<MoeGate> {
    let cfg: Glm4MoeLiteConfig =
        serde_json::from_value(patched(base_config(), json!({"hidden_size": EXPERTS})))?;
    Ok(MoeGate::new(&cfg.family(), router_vb(bias)?, EXPERTS)?)
}

#[test]
fn router_noaux_tc_renormalises_then_scales() -> Result<()> {
    let routes = gate(true)?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [
            [(4, 1.551287), (6, 0.948713)],
            [(2, 1.105823), (6, 1.394177)],
        ],
    )
}

#[test]
fn router_requires_bias() -> Result<()> {
    assert_err_contains(gate(false), "e_score_correction_bias");
    Ok(())
}

fn forward_case(split_kv_b: bool) -> Result<Tensor> {
    let mut checkpoint = Checkpoint::default();
    checkpoint.skeleton(MOE_INTERMEDIATE * N_SHARED, true);
    checkpoint.mla_attention(Some(Q_LORA), split_kv_b, V_DIM);
    load_and_forward(&GLM4MoeLiteLoader, &base_config(), &checkpoint)
}

#[test]
fn forward_fused_kv_b() -> Result<()> {
    let expected = Snapshot {
        probes: [-0.6860662, -0.2246428, -0.63331395, -0.32600886],
        sum: 0.9202633,
        l2: 17.03426,
    };
    assert_snapshot(&forward_case(false)?, &expected)
}

#[test]
fn forward_split_kv_b_errors_on_2d_weights() -> Result<()> {
    // pins current behaviour: 2-D k_b/v_b load from safetensors, then the split projection wants 3-D (GGUF) weights
    assert_err_contains(forward_case(true), "unexpected rank");
    Ok(())
}

#[test]
fn a_zero_moe_layer_freq_is_a_config_error() {
    let config = patched(base_config(), json!({"moe_layer_freq": 0}));
    assert_err_contains(
        serde_json::from_value::<Glm4MoeLiteConfig>(config),
        "moe_layer_freq must be at least 1",
    );
}
