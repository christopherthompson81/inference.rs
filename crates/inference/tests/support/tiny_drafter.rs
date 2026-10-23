//! A tiny DFlash-family drafter written at test time: one ramp-weighted layer over a target's hidden size, reusing its
//! embeddings and head.
#![allow(dead_code)]

const INTERMEDIATE: usize = 128;
const HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const WEIGHT_SCALE: f32 = 0.05;
const WEIGHT_STEP: f32 = 0.37;

/// What a drafter needs of its target: the hidden size, the vocabulary and the layers it taps.
pub struct DrafterTarget {
    pub hidden: usize,
    pub vocab: usize,
    pub taps: &'static [usize],
}

impl DrafterTarget {
    // the tokenizer's last id, never produced by the prompts
    fn mask_token(&self) -> u32 {
        u32::try_from(self.vocab - 1).expect("a tiny vocabulary")
    }
}

pub fn tiny_drafter(
    target: &DrafterTarget,
    overrides: serde_json::Value,
    markov_rank: Option<usize>,
) -> anyhow::Result<tempfile::TempDir> {
    let (hidden, vocab) = (target.hidden, target.vocab);
    let dir = tempfile::tempdir()?;
    let mut config = serde_json::json!({
        "hidden_size": hidden,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 1,
        "num_attention_heads": HEADS,
        "num_key_value_heads": 1,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "rope_theta": 10000,
        "mask_token_id": target.mask_token(),
        "dflash_config": { "target_layer_ids": target.taps }
    });
    for (key, value) in overrides
        .as_object()
        .expect("drafter overrides are an object")
    {
        config[key] = value.clone();
    }
    std::fs::write(dir.path().join("config.json"), config.to_string())?;
    let device = inference_tensor::Device::Cpu;
    let ramp = |rows: usize, cols: usize| {
        let data = (0..rows * cols)
            .map(|i| ((i as f32) * WEIGHT_STEP).sin() * WEIGHT_SCALE)
            .collect::<Vec<_>>();
        inference_tensor::Tensor::from_vec(data, (rows, cols), &device)
    };
    let ones = |n: usize| inference_tensor::Tensor::ones(n, inference_tensor::DType::F32, &device);
    let (q, kv) = (HEADS * HEAD_DIM, HEAD_DIM);
    let mut tensors = std::collections::HashMap::new();
    let layer = "layers.0";
    for (name, rows, cols) in [
        ("self_attn.q_proj", q, hidden),
        ("self_attn.k_proj", kv, hidden),
        ("self_attn.v_proj", kv, hidden),
        ("self_attn.o_proj", hidden, q),
        ("mlp.gate_proj", INTERMEDIATE, hidden),
        ("mlp.up_proj", INTERMEDIATE, hidden),
        ("mlp.down_proj", hidden, INTERMEDIATE),
    ] {
        tensors.insert(format!("{layer}.{name}.weight"), ramp(rows, cols)?);
    }
    for (name, n) in [
        ("self_attn.q_norm", HEAD_DIM),
        ("self_attn.k_norm", HEAD_DIM),
        ("input_layernorm", hidden),
        ("post_attention_layernorm", hidden),
    ] {
        tensors.insert(format!("{layer}.{name}.weight"), ones(n)?);
    }
    tensors.insert(
        "fc.weight".to_string(),
        ramp(hidden, target.taps.len() * hidden)?,
    );
    tensors.insert("hidden_norm.weight".to_string(), ones(hidden)?);
    tensors.insert("norm.weight".to_string(), ones(hidden)?);
    if let Some(rank) = markov_rank {
        for name in [
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
        ] {
            tensors.insert(name.to_string(), ramp(vocab, rank)?);
        }
    }
    inference_tensor::safetensors::save(&tensors, dir.path().join("model.safetensors"))?;
    Ok(dir)
}
