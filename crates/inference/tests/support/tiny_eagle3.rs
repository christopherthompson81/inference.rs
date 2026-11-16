//! A tiny random EAGLE-3 drafter (SpecForge layout, a reduced vocabulary mapped through `d2t`) for a tiny target.
#![allow(dead_code)]

use std::collections::HashMap;

use inference_tensor::{DType, Device, Tensor};

// draft id i is target id 3i
const DRAFT_ID_STRIDE: i64 = 3;
const WEIGHT_STD: f32 = 0.1;

/// The drafter's shape; `eagle31` adds EAGLE 3.1's per-tap norms and chains its normed output.
pub struct Eagle3Shape {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub draft_vocab: usize,
    pub tap_layer_ids: [usize; 3],
    pub eagle31: bool,
}

pub fn tiny_eagle3(shape: &Eagle3Shape) -> anyhow::Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLMEagle3"],
        "hidden_size": shape.hidden,
        "intermediate_size": shape.intermediate,
        "num_attention_heads": shape.heads,
        "num_key_value_heads": shape.kv_heads,
        "head_dim": shape.head_dim,
        "num_hidden_layers": 1,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": shape.vocab,
        "draft_vocab_size": shape.draft_vocab,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": shape.tap_layer_ids},
        "fc_norm": shape.eagle31,
        "norm_output": shape.eagle31,
    });
    std::fs::write(dir.path().join("config.json"), config.to_string())?;
    let random = |dims: &[usize]| -> anyhow::Result<Tensor> {
        Ok(Tensor::randn(0f32, WEIGHT_STD, dims, &Device::Cpu)?.to_dtype(DType::BF16)?)
    };
    let ones = |width: usize| Tensor::ones(width, DType::BF16, &Device::Cpu);
    let hidden = shape.hidden;
    let (q, kv) = (
        shape.heads * shape.head_dim,
        shape.kv_heads * shape.head_dim,
    );
    let mut tensors = HashMap::from([
        ("fc.weight".to_string(), random(&[hidden, 3 * hidden])?),
        ("norm.weight".to_string(), ones(hidden)?),
        (
            "lm_head.weight".to_string(),
            random(&[shape.draft_vocab, hidden])?,
        ),
        (
            "d2t".to_string(),
            Tensor::from_vec(
                (0..shape.draft_vocab as i64)
                    .map(|i| i * (DRAFT_ID_STRIDE - 1))
                    .collect::<Vec<_>>(),
                shape.draft_vocab,
                &Device::Cpu,
            )?,
        ),
    ]);
    if shape.eagle31 {
        for tap in 0..3 {
            tensors.insert(format!("fcs.{tap}.weight"), ones(hidden)?);
        }
    }
    for (name, dims) in [
        ("self_attn.q_proj.weight", vec![q, 2 * hidden]),
        ("self_attn.k_proj.weight", vec![kv, 2 * hidden]),
        ("self_attn.v_proj.weight", vec![kv, 2 * hidden]),
        ("self_attn.o_proj.weight", vec![hidden, q]),
        ("mlp.gate_proj.weight", vec![shape.intermediate, hidden]),
        ("mlp.up_proj.weight", vec![shape.intermediate, hidden]),
        ("mlp.down_proj.weight", vec![hidden, shape.intermediate]),
    ] {
        tensors.insert(format!("midlayer.{name}"), random(&dims)?);
    }
    for norm in ["input_layernorm", "hidden_norm", "post_attention_layernorm"] {
        tensors.insert(format!("midlayer.{norm}.weight"), ones(hidden)?);
    }
    inference_tensor::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
    Ok(dir)
}
