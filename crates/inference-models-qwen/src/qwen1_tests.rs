//! Locks Qwen 1's prefill on tiny made-up weights under its own names (`transformer.h.N.attn.c_attn`, `mlp.w1`).

use anyhow::Result;
use inference_nn::loaders::NormalModelLoader;
use inference_nn::paged_attention::AttentionImplementation;
use inference_nn::testing::{
    Snapshot, assert_snapshot, forward_normal, load_synthesized, metadata, names_digest,
};
use inference_tensor::DType;
use serde_json::json;

use crate::loaders::QwenLoader;

const VOCAB: usize = 64;

#[test]
fn qwen1_prefill() -> Result<()> {
    let config = json!({
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "kv_channels": 8,
        "layer_norm_epsilon": 1e-6,
        "rotary_emb_base": 10000.0,
        "seq_length": 64,
        "quantization_config": null,
    });
    let (model, names) = load_synthesized(&[], Default::default(), DType::F32, |vb| {
        QwenLoader.load(
            &config.to_string(),
            vb,
            metadata(),
            AttentionImplementation::Eager,
        )
    })?;
    assert_eq!(
        names_digest(names.keys()),
        0x4823_c225_96f6_1cec,
        "tensor names moved: {:#x}",
        names_digest(names.keys())
    );
    let expected = Snapshot {
        probes: [0.4616803, -0.057436243, 0.22844815, 0.115322],
        sum: -1.4004626,
        l2: 3.4705162,
    };
    assert_snapshot(&forward_normal(model.as_ref())?, VOCAB, &expected)
}
