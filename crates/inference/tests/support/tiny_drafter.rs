//! A tiny DFlash-family drafter written at test time: one ramp-weighted layer over a target's hidden size, reusing its
//! embeddings and head.
#![allow(dead_code)]

const INTERMEDIATE: usize = 128;
const HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const WEIGHT_SCALE: f32 = 0.05;
const WEIGHT_STEP: f32 = 0.37;
// DSpine's code bridge: its code width, message rank and refinement candidates
const DSPINE_CODE: usize = 32;
const DSPINE_RANK: usize = 16;
const DSPINE_TOPK: usize = 4;
const DSPINE_LAYERS: usize = 2;
const DSPINE_VERSIONS: [&str; 4] = [
    "draft_block_causal_version",
    "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
];

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
    let layers = config["num_hidden_layers"]
        .as_u64()
        .and_then(|layers| usize::try_from(layers).ok())
        .expect("drafter layer count");
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
    for layer in (0..layers).map(|layer| format!("layers.{layer}")) {
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

/// A tiny DSpine drafter: two layers, each ending in a code write, and the serial refinement over the head's top-k.
pub fn tiny_dspine(
    target: &DrafterTarget,
    overrides: serde_json::Value,
) -> anyhow::Result<tempfile::TempDir> {
    let mut config = serde_json::json!({
        "architectures": ["DSpineDraftModel"],
        "num_hidden_layers": DSPINE_LAYERS,
        "is_causal": true,
        "code_bridge_second_pass": {
            "mode": "serial_table",
            "first_exact": DSPINE_LAYERS - 1,
            "head_topk": DSPINE_TOPK
        },
        "dflash_config": {
            "target_layer_ids": target.taps,
            "draft_block_causal": true,
            "selected_prefix_code_bridge": { "code_size": DSPINE_CODE, "rank": DSPINE_RANK, "eps": 1e-6 },
            "selected_prefix_decode_mode": "serial",
            "selected_prefix_feature_layers": (1..=DSPINE_LAYERS).collect::<Vec<_>>(),
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": true,
            "selected_prefix_last_write_refine": {
                "head_topk": DSPINE_TOPK, "kind": "exact_sender", "version": 1, "weight": 1.0
            },
            "selected_prefix_terminal": false
        }
    });
    for (key, value) in overrides
        .as_object()
        .expect("drafter overrides are an object")
    {
        config[key] = value.clone();
    }
    let dir = tiny_drafter(target, config, None)?;
    let path = dir.path().join("model.safetensors");
    let mut tensors = inference_tensor::safetensors::load(&path, &inference_tensor::Device::Cpu)?;
    let device = inference_tensor::Device::Cpu;
    let ramp = |shape: &[usize], salt: f32| {
        let len = shape.iter().product::<usize>();
        let data = (0..len)
            .map(|i| ((i as f32) * WEIGHT_STEP + salt).sin() * WEIGHT_SCALE)
            .collect::<Vec<_>>();
        inference_tensor::Tensor::from_vec(data, shape, &device)
    };
    let (hidden, code, rank) = (target.hidden, DSPINE_CODE, DSPINE_RANK);
    let bridge = |name: &str| format!("selected_prefix_bridge.{name}");
    for (name, shape, salt) in [
        ("codebook", vec![target.vocab, code], 1.0),
        (
            "early_read_weights",
            vec![DSPINE_LAYERS - 1, code, hidden],
            2.0,
        ),
        ("read_weight", vec![code, hidden], 3.0),
        (
            "feature_write_weights",
            vec![DSPINE_LAYERS, hidden, rank],
            4.0,
        ),
        ("gate.weight", vec![rank, code], 5.0),
        ("gate.bias", vec![rank], 6.0),
        ("predecessor_gate.weight", vec![rank, code], 7.0),
        ("prev.weight", vec![rank, code], 8.0),
    ] {
        tensors.insert(bridge(name), ramp(&shape, salt)?);
    }
    tensors.insert(
        bridge("out.weight"),
        inference_tensor::Tensor::zeros((code, rank), inference_tensor::DType::F32, &device)?,
    );
    for name in DSPINE_VERSIONS {
        tensors.insert(
            name.to_string(),
            inference_tensor::Tensor::new(1i64, &device)?,
        );
    }
    inference_tensor::safetensors::save(&tensors, &path)?;
    Ok(dir)
}
