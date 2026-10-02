use anyhow::Result;
use candle_core::Tensor;
use serde_json::{Value, json};

use super::Glm4MoeConfig;
use crate::deepseek_family::MoeGate;
use crate::deepseek_family_tests::{
    Checkpoint, EXPERTS, HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS, MAX_POS,
    MOE_INTERMEDIATE, N_SHARED, SCALE, Snapshot, TOP_K, VOCAB, assert_err_contains, assert_routes,
    assert_snapshot, load_and_forward, patched, router_input, router_vb,
};
use crate::loaders::GLM4MoeLoader;

fn base_config() -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "partial_rotary_factor": 0.5,
        "use_qk_norm": true,
        "attention_bias": true,
        "n_routed_experts": EXPERTS,
        "n_shared_experts": N_SHARED,
        "num_experts_per_tok": TOP_K,
        "first_k_dense_replace": 1,
        "routed_scaling_factor": SCALE,
        "n_group": 4,
        "topk_group": 2,
        "norm_topk_prob": true,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": MAX_POS,
        "hidden_act": "silu",
        "tie_word_embeddings": false,
    })
}

fn gate(patch: Value, bias: bool) -> Result<MoeGate> {
    let cfg: Glm4MoeConfig = serde_json::from_value(patched(
        base_config(),
        patched(json!({"hidden_size": EXPERTS}), patch),
    ))?;
    Ok(MoeGate::new(&cfg.family(), router_vb(bias)?, EXPERTS)?)
}

#[test]
fn router_noaux_tc_norm_topk_prob() -> Result<()> {
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
fn router_noaux_tc_without_norm_topk_prob() -> Result<()> {
    let routes = gate(json!({"norm_topk_prob": false}), true)?.forward(&router_input()?)?;
    assert_routes(
        routes,
        [[(4, 2.043936), (6, 1.25)], [(2, 1.724936), (6, 2.174729)]],
    )
}

#[test]
fn router_requires_bias() -> Result<()> {
    assert_err_contains(gate(json!({}), false), "e_score_correction_bias");
    Ok(())
}

fn gqa_attention(checkpoint: &mut Checkpoint, qk_norm: bool, bias: bool) {
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}.self_attn");
        for (proj, heads) in [
            ("q_proj", HEADS),
            ("k_proj", KV_HEADS),
            ("v_proj", KV_HEADS),
        ] {
            checkpoint.linear(&format!("{p}.{proj}"), heads * HEAD_DIM, HIDDEN);
            if bias {
                checkpoint.add(format!("{p}.{proj}.bias"), &[heads * HEAD_DIM]);
            }
        }
        checkpoint.linear(&format!("{p}.o_proj"), HIDDEN, HEADS * HEAD_DIM);
        if qk_norm {
            checkpoint.add(format!("{p}.q_norm.weight"), &[HEAD_DIM]);
            checkpoint.add(format!("{p}.k_norm.weight"), &[HEAD_DIM]);
        }
    }
}

fn forward_case(qk_norm: bool, bias: bool) -> Result<Tensor> {
    let config = patched(
        base_config(),
        json!({"use_qk_norm": qk_norm, "attention_bias": bias}),
    );
    let mut checkpoint = Checkpoint::default();
    // shared expert width ignores n_shared_experts today
    checkpoint.skeleton(MOE_INTERMEDIATE, true);
    gqa_attention(&mut checkpoint, qk_norm, bias);
    load_and_forward(&GLM4MoeLoader, &config, &checkpoint)
}

#[test]
fn forward_qk_norm_with_bias() -> Result<()> {
    let expected = Snapshot {
        probes: [1.2364452, 1.4078764, 0.6092725, 0.45861205],
        sum: 63.415905,
        l2: 19.252851,
    };
    assert_snapshot(&forward_case(true, true)?, &expected)
}

#[test]
fn forward_plain_attention() -> Result<()> {
    let expected = Snapshot {
        probes: [1.2770244, 1.4468794, 0.3916812, 1.2174238],
        sum: 56.134796,
        l2: 19.152615,
    };
    assert_snapshot(&forward_case(false, false)?, &expected)
}
