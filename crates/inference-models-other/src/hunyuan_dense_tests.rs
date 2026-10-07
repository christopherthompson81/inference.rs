//! Locks HunYuan dense's prefill on tiny made-up weights, with plain and dynamic-alpha RoPE.

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest, patched,
};
use inference_tensor::DType;
use serde_json::{Value, json};

use crate::loaders::HunYuanDenseV1Loader;

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

fn hunyuan_config() -> Value {
    json!({
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "head_dim": 8,
        "rope_scaling": null,
        "use_cla": false,
        "cla_share_factor": null,
        "attention_bias": false,
        "mlp_bias": false,
        "pretraining_tp": 1,
        "add_classification_head": false,
        "tie_word_embeddings": false,
        "quantization_config": null,
    })
}

#[test]
fn hunyuan_dense_prefill() -> Result<()> {
    prefill(
        &HunYuanDenseV1Loader,
        &hunyuan_config(),
        0x24c4_54cc_9a00_65a6,
        &Snapshot {
            probes: [-0.31660226, 1.5865619, 1.1089734, 1.2527826],
            sum: 49.484287,
            l2: 18.22231,
        },
    )
}

#[test]
fn hunyuan_dense_prefill_with_dynamic_rope() -> Result<()> {
    let config = patched(
        hunyuan_config(),
        json!({"rope_scaling": {"type": "dynamic", "alpha": 1000.0}}),
    );
    prefill(
        &HunYuanDenseV1Loader,
        &config,
        0x24c4_54cc_9a00_65a6,
        &Snapshot {
            probes: [-0.31660226, 1.5896238, 1.078223, 1.2904422],
            sum: 49.911858,
            l2: 18.263638,
        },
    )
}
