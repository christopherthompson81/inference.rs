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

// our tensor names to llama.cpp's (its `convert_hf_to_gguf.py`), the q/k ones in its adjacent-pair RoPE row order
const GGUF_NAMES: &[(&str, &str)] = &[
    ("fc.weight", "fc.weight"),
    ("norm.weight", "output_norm.weight"),
    ("lm_head.weight", "output.weight"),
    ("midlayer.input_layernorm.weight", "blk.0.attn_norm.weight"),
    ("midlayer.hidden_norm.weight", "blk.0.attn_norm_2.weight"),
    (
        "midlayer.post_attention_layernorm.weight",
        "blk.0.ffn_norm.weight",
    ),
    ("midlayer.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
    ("midlayer.self_attn.k_proj.weight", "blk.0.attn_k.weight"),
    ("midlayer.self_attn.v_proj.weight", "blk.0.attn_v.weight"),
    (
        "midlayer.self_attn.o_proj.weight",
        "blk.0.attn_output.weight",
    ),
    ("midlayer.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
    ("midlayer.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
    ("midlayer.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
];
const GGML_F32: u32 = 0;
const GGML_I64: u32 = 27;

// llama.cpp's `LlamaModel.permute`: each head's rows from [first half, second half] to interleaved pairs
fn permute_rows(rows: &[f32], cols: usize, heads: usize) -> Vec<f32> {
    let head_rows = rows.len() / cols / heads;
    let half = head_rows / 2;
    let mut out = vec![0f32; rows.len()];
    for head in 0..heads {
        for pair in 0..half {
            for side in 0..2 {
                let from = head * head_rows + side * half + pair;
                let to = head * head_rows + 2 * pair + side;
                out[to * cols..(to + 1) * cols]
                    .copy_from_slice(&rows[from * cols..(from + 1) * cols]);
            }
        }
    }
    out
}

/// `drafter` (a [`tiny_eagle3`] directory) as llama.cpp's converter writes it: an `eagle3` GGUF with F32 weights,
/// q/k rows permuted for its RoPE, and `d2t` as the target ids.
pub fn tiny_eagle3_gguf(
    shape: &Eagle3Shape,
    drafter: &std::path::Path,
) -> anyhow::Result<tempfile::NamedTempFile> {
    use inference_tensor::quantized::gguf_file::{self, Value};
    let tensors =
        inference_tensor::safetensors::load(drafter.join("model.safetensors"), &Device::Cpu)?;
    let mut entries = Vec::new();
    for (ours, gguf) in GGUF_NAMES {
        let tensor = tensors[*ours].to_dtype(DType::F32)?;
        let dims = tensor.dims().to_vec();
        let mut values = tensor.flatten_all()?.to_vec1::<f32>()?;
        if ours.ends_with("q_proj.weight") {
            values = permute_rows(&values, dims[1], shape.heads);
        } else if ours.ends_with("k_proj.weight") {
            values = permute_rows(&values, dims[1], shape.kv_heads);
        }
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        entries.push((gguf.to_string(), dims, GGML_F32, bytes));
    }
    let d2t = tensors["d2t"].to_vec1::<i64>()?;
    let targets = d2t.iter().zip(0i64..).map(|(offset, draft)| offset + draft);
    entries.push((
        "d2t".to_string(),
        vec![d2t.len()],
        GGML_I64,
        targets.flat_map(i64::to_le_bytes).collect(),
    ));
    let u32_value = |v: usize| Value::U32(u32::try_from(v).expect("tiny dims"));
    let metadata = [
        ("general.architecture", Value::String("eagle3".to_string())),
        (
            "eagle3.target_layers",
            Value::Array(
                shape
                    .tap_layer_ids
                    .iter()
                    .map(|&id| u32_value(id))
                    .collect(),
            ),
        ),
        ("eagle3.target_hidden_size", u32_value(shape.hidden)),
        ("eagle3.norm_before_residual", Value::Bool(false)),
        ("eagle3.norm_before_fc", Value::Bool(false)),
        ("eagle3.block_count", u32_value(1)),
        ("eagle3.context_length", u32_value(512)),
        ("eagle3.embedding_length", u32_value(shape.hidden)),
        ("eagle3.feed_forward_length", u32_value(shape.intermediate)),
        ("eagle3.attention.head_count", u32_value(shape.heads)),
        ("eagle3.attention.head_count_kv", u32_value(shape.kv_heads)),
        ("eagle3.attention.key_length", u32_value(shape.head_dim)),
        ("eagle3.attention.value_length", u32_value(shape.head_dim)),
        ("eagle3.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("eagle3.rope.freq_base", Value::F32(10000.0)),
        ("eagle3.vocab_size", u32_value(shape.vocab)),
    ];
    let metadata = metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>();
    let mut file = tempfile::Builder::new().suffix(".gguf").tempfile()?;
    let headers = entries
        .iter()
        .map(|(name, dims, ty, bytes)| gguf_file::TensorEntry {
            name,
            dims,
            ggml_dtype: *ty,
            size_in_bytes: bytes.len(),
        })
        .collect::<Vec<_>>();
    gguf_file::write_header(
        file.as_file_mut(),
        metadata.len(),
        &gguf_file::metadata_bytes(&metadata)?,
        &headers,
    )?;
    for (_, _, _, bytes) in &entries {
        gguf_file::write_tensor_data(file.as_file_mut(), bytes)?;
    }
    Ok(file)
}
