//! The tiny random-weight Llama checkpoint as a GGUF, laid out as llama.cpp's converter writes one.
#![allow(dead_code)]

use std::path::Path;

use inference_tensor::quantized::{GgmlDType, QTensor, gguf_file};
use inference_tensor::{Device, Tensor};

// From tests/fixtures/llama_tiny/config.json
pub const LAYERS: usize = 2;
pub const HIDDEN: usize = 32;
pub const HEADS: usize = 2;
pub const KV_HEADS: usize = 1;
pub const HEAD_DIM: usize = HIDDEN / HEADS;
const INTERMEDIATE: u32 = 64;
const VOCAB: u32 = 266;
const CONTEXT: u32 = 512;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10_000.0;
// tests/fixtures/llama_tiny/tokenizer_config.json: <unk>, <s>, </s>
const UNK_ID: u32 = 0;
const BOS_ID: u32 = 1;
const EOS_ID: u32 = 2;

// llama.cpp's convert_hf_to_gguf.py permute: each head's half-split rows become interleaved pairs.
fn to_adjacent_rope_rows(weight: &Tensor, heads: usize) -> anyhow::Result<Tensor> {
    let (rows, cols) = weight.dims2()?;
    Ok(weight
        .reshape((heads, 2, rows / heads / 2, cols))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((rows, cols))?)
}

fn gguf_name(hf: &str) -> Option<String> {
    let fixed = match hf {
        "model.embed_tokens.weight" => Some("token_embd.weight"),
        "model.norm.weight" => Some("output_norm.weight"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(name) = fixed {
        return Some(name.to_string());
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (layer, tensor) = rest.split_once('.')?;
    let tensor = match tensor {
        "input_layernorm.weight" => "attn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{tensor}"))
}

// The checkpoint's tokens by id: an external tokenizer must match the vocabulary the GGUF embeds.
fn tokens_by_id(tokenizer_json: &Path) -> anyhow::Result<Vec<gguf_file::Value>> {
    let tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tokenizer_json)?)?;
    let mut by_id = std::collections::BTreeMap::new();
    let vocab = tokenizer["model"]["vocab"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no model.vocab"))?;
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("non-integer id for {token:?}"))?;
        by_id.insert(id, token.clone());
    }
    for added in tokenizer["added_tokens"].as_array().into_iter().flatten() {
        let id = added["id"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("non-integer added token id"))?;
        by_id.insert(
            id,
            added["content"].as_str().unwrap_or_default().to_string(),
        );
    }
    anyhow::ensure!(
        by_id.keys().copied().eq(0..by_id.len() as u64),
        "tokenizer ids are not contiguous"
    );
    Ok(by_id.into_values().map(gguf_file::Value::String).collect())
}

/// The tiny checkpoint as an f32 GGUF, with Q/K reordered as llama.cpp's converter writes them, and `extra` metadata.
pub fn write_tiny_gguf(
    checkpoint: &Path,
    out: &Path,
    extra: &[(&str, gguf_file::Value)],
) -> anyhow::Result<()> {
    write_tiny_gguf_with_head(checkpoint, out, extra, GgmlDType::F32)
}

/// As [`write_tiny_gguf`], with the output head stored as `head`.
pub fn write_tiny_gguf_with_head(
    checkpoint: &Path,
    out: &Path,
    extra: &[(&str, gguf_file::Value)],
    head: GgmlDType,
) -> anyhow::Result<()> {
    let weights =
        inference_tensor::safetensors::load(checkpoint.join("model.safetensors"), &Device::Cpu)?;
    let mut tensors = Vec::new();
    for (name, tensor) in &weights {
        let gguf = gguf_name(name).ok_or_else(|| anyhow::anyhow!("no GGUF name for `{name}`"))?;
        let tensor = if gguf.ends_with("attn_q.weight") {
            to_adjacent_rope_rows(tensor, HEADS)?
        } else if gguf.ends_with("attn_k.weight") {
            to_adjacent_rope_rows(tensor, KV_HEADS)?
        } else {
            tensor.clone()
        };
        let dtype = if gguf == "output.weight" {
            head
        } else {
            GgmlDType::F32
        };
        tensors.push((gguf, QTensor::quantize(&tensor, dtype)?));
    }
    let metadata = [
        (
            "general.architecture",
            gguf_file::Value::String("llama".into()),
        ),
        ("llama.context_length", gguf_file::Value::U32(CONTEXT)),
        ("llama.vocab_size", gguf_file::Value::U32(VOCAB)),
        (
            "llama.embedding_length",
            gguf_file::Value::U32(HIDDEN as u32),
        ),
        ("llama.block_count", gguf_file::Value::U32(LAYERS as u32)),
        (
            "llama.feed_forward_length",
            gguf_file::Value::U32(INTERMEDIATE),
        ),
        (
            "llama.attention.head_count",
            gguf_file::Value::U32(HEADS as u32),
        ),
        (
            "llama.attention.head_count_kv",
            gguf_file::Value::U32(KV_HEADS as u32),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            gguf_file::Value::F32(RMS_EPS),
        ),
        ("llama.rope.freq_base", gguf_file::Value::F32(ROPE_THETA)),
        (
            "llama.rope.dimension_count",
            gguf_file::Value::U32(HEAD_DIM as u32),
        ),
        (
            "tokenizer.ggml.tokens",
            gguf_file::Value::Array(tokens_by_id(&checkpoint.join("tokenizer.json"))?),
        ),
        (
            "tokenizer.ggml.unknown_token_id",
            gguf_file::Value::U32(UNK_ID),
        ),
        ("tokenizer.ggml.bos_token_id", gguf_file::Value::U32(BOS_ID)),
        ("tokenizer.ggml.eos_token_id", gguf_file::Value::U32(EOS_ID)),
    ];
    let metadata = metadata
        .iter()
        .chain(extra)
        .map(|(k, v)| (*k, v))
        .collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let mut file = std::fs::File::create(out)?;
    gguf_file::write(&mut file, &metadata, &tensors)?;
    Ok(())
}
