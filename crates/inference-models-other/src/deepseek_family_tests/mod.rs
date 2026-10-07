//! Shared fixtures for the DeepSeek-V2/V3 and GLM4-MoE(-Lite) behaviour-locking tests beside each model.

use std::collections::HashMap;

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Device, Tensor};
use serde_json::Value;

pub use inference_nn::testing::{
    Checkpoint, Snapshot, assert_err_contains, metadata, patched, wrap,
};

pub const VOCAB: usize = 64;
pub const HIDDEN: usize = 32;
pub const INTERMEDIATE: usize = 48;
pub const MOE_INTERMEDIATE: usize = 16;
pub const LAYERS: usize = 2;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 8;
pub const Q_LORA: usize = 16;
pub const KV_LORA: usize = 16;
pub const ROPE_DIM: usize = 8;
pub const NOPE_DIM: usize = 8;
pub const V_DIM: usize = 16;
pub const NARROW_V_DIM: usize = 8;
pub const EXPERTS: usize = 8;
pub const TOP_K: usize = 2;
pub const N_SHARED: usize = 2;
pub const MAX_POS: usize = 64;
pub const SCALE: f64 = 2.5;

// Router logits per token: the gate weight is the identity, so these are the gate logits directly.
pub const ROUTER_LOGITS: [[f32; EXPERTS]; 2] = [
    [0.1, 2.0, -1.0, 0.5, 1.5, -0.5, 0.0, 1.0],
    [1.2, -0.3, 0.8, 2.5, -1.5, 0.3, 1.9, -0.7],
];
pub const ROUTER_BIAS: [f32; EXPERTS] = [0.0, -0.5, 0.3, 0.0, 0.2, 0.0, 0.4, 0.1];

const ROUTE_TOL: f32 = 1e-5;

/// The DeepSeek-family layouts on top of the shared checkpoint builder.
pub trait FamilyCheckpoint {
    fn mlp(&mut self, prefix: &str, intermediate: usize);
    fn skeleton(&mut self, shared_intermediate: usize, gate_bias: bool);
    fn mla_attention(&mut self, q_lora: Option<usize>, split_kv_b: bool, v_dim: usize);
}

impl FamilyCheckpoint for Checkpoint {
    fn mlp(&mut self, prefix: &str, intermediate: usize) {
        self.linear(&format!("{prefix}.gate_proj"), intermediate, HIDDEN);
        self.linear(&format!("{prefix}.up_proj"), intermediate, HIDDEN);
        self.linear(&format!("{prefix}.down_proj"), HIDDEN, intermediate);
    }

    /// Embeddings, final norm, lm_head and per-layer norms plus the dense layer 0 MLP and the MoE layer 1 MLP.
    fn skeleton(&mut self, shared_intermediate: usize, gate_bias: bool) {
        self.add("model.embed_tokens.weight", &[VOCAB, HIDDEN]);
        self.add("model.norm.weight", &[HIDDEN]);
        self.add("lm_head.weight", &[VOCAB, HIDDEN]);
        for layer in 0..LAYERS {
            let p = format!("model.layers.{layer}");
            self.add(format!("{p}.input_layernorm.weight"), &[HIDDEN]);
            self.add(format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]);
        }
        self.mlp("model.layers.0.mlp", INTERMEDIATE);
        let p = "model.layers.1.mlp";
        self.add(format!("{p}.gate.weight"), &[EXPERTS, HIDDEN]);
        if gate_bias {
            self.add(format!("{p}.gate.e_score_correction_bias"), &[EXPERTS]);
        }
        for expert in 0..EXPERTS {
            self.mlp(&format!("{p}.experts.{expert}"), MOE_INTERMEDIATE);
        }
        self.mlp(&format!("{p}.shared_experts"), shared_intermediate);
    }

    /// The MLA attention tensors of every layer (DeepSeek-V2/V3, GLM4-MoE-Lite).
    fn mla_attention(&mut self, q_lora: Option<usize>, split_kv_b: bool, v_dim: usize) {
        let q_head = NOPE_DIM + ROPE_DIM;
        for layer in 0..LAYERS {
            let p = format!("model.layers.{layer}.self_attn");
            match q_lora {
                Some(rank) => {
                    self.linear(&format!("{p}.q_a_proj"), rank, HIDDEN);
                    self.add(format!("{p}.q_a_layernorm.weight"), &[rank]);
                    self.linear(&format!("{p}.q_b_proj"), HEADS * q_head, rank);
                }
                None => self.linear(&format!("{p}.q_proj"), HEADS * q_head, HIDDEN),
            }
            self.linear(
                &format!("{p}.kv_a_proj_with_mqa"),
                KV_LORA + ROPE_DIM,
                HIDDEN,
            );
            self.add(format!("{p}.kv_a_layernorm.weight"), &[KV_LORA]);
            if split_kv_b {
                self.linear(&format!("{p}.k_b_proj"), HEADS * KV_LORA, NOPE_DIM);
                self.linear(&format!("{p}.v_b_proj"), HEADS * v_dim, KV_LORA);
            } else {
                self.linear(
                    &format!("{p}.kv_b_proj"),
                    HEADS * (NOPE_DIM + v_dim),
                    KV_LORA,
                );
            }
            self.linear(&format!("{p}.o_proj"), HIDDEN, HEADS * v_dim);
        }
    }
}

pub fn load_and_forward(
    loader: &dyn NormalModelLoader,
    config: &Value,
    checkpoint: &Checkpoint,
) -> Result<Tensor> {
    let model = inference_nn::testing::load_checked(checkpoint, |vb| {
        loader.load(
            &config.to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    inference_nn::testing::forward_normal(model.as_ref())
}

pub fn assert_snapshot(logits: &Tensor, expected: &Snapshot) -> Result<()> {
    inference_nn::testing::assert_snapshot(logits, VOCAB, expected)
}

/// A gate checkpoint whose weight is the identity, so the router logits are the hidden states.
pub fn router_vb(bias: bool) -> Result<ShardedVarBuilder> {
    let mut tensors = HashMap::from([(
        "weight".to_string(),
        Tensor::eye(EXPERTS, DType::F32, &Device::Cpu)?,
    )]);
    if bias {
        tensors.insert(
            "e_score_correction_bias".to_string(),
            Tensor::new(&ROUTER_BIAS, &Device::Cpu)?,
        );
    }
    Ok(wrap(tensors))
}

pub fn router_input() -> Result<Tensor> {
    Ok(Tensor::new(&ROUTER_LOGITS, &Device::Cpu)?.unsqueeze(0)?)
}

/// Compares routes per token as (expert, weight) pairs sorted by expert, so output order is free.
pub fn assert_routes(
    routes: (Tensor, Tensor),
    expected: [[(u32, f32); TOP_K]; ROUTER_LOGITS.len()],
) -> Result<()> {
    let (indices, weights) = routes;
    let indices = indices.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let weights = weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let actual = indices
        .iter()
        .zip(&weights)
        .map(|(idx, w)| {
            let mut pairs = idx
                .iter()
                .copied()
                .zip(w.iter().copied())
                .collect::<Vec<_>>();
            pairs.sort_by_key(|(i, _)| *i);
            pairs
        })
        .collect::<Vec<_>>();
    let ok = actual.iter().zip(&expected).all(|(a, e)| {
        a.len() == e.len()
            && a.iter()
                .zip(e)
                .all(|((ai, aw), (ei, ew))| ai == ei && (aw - ew).abs() <= ROUTE_TOL)
    });
    assert!(ok, "routes moved: {actual:?}");
    Ok(())
}
