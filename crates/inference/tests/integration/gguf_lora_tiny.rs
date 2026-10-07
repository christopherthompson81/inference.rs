//! Runtime LoRA on a GGUF whose Q/K rows a converter reordered to adjacent RoPE pairs, against the safetensors source.

use std::collections::HashMap;
use std::path::Path;

use inference::{
    GgufModelBuilder, LoraModelBuilder, Model, ModelDType, RequestBuilder, TextMessageRole,
    TextMessages, TextModelBuilder,
};
use inference_tensor::quantized::{GgmlDType, QTensor, gguf_file};
use inference_tensor::{Device, Tensor};

#[path = "../support/llama_tiny.rs"]
mod support;
use support::tiny_llama_checkpoint;

const PROMPT: &str = "hello";
const MAX_LEN: usize = 8;
const ADAPTER: &str = "qk-adapter";
const ADAPTER_RANK: usize = 2;
const GGUF_FILE: &str = "tiny.gguf";
// From tests/fixtures/llama_tiny/config.json
const LAYERS: usize = 2;
const HIDDEN: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = HIDDEN / HEADS;
const INTERMEDIATE: u32 = 64;
const VOCAB: u32 = 266;
const CONTEXT: u32 = 512;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10_000.0;
// tests/fixtures/llama_tiny/tokenizer_config.json: <unk>, <s>, </s>
const UNK_ID: u32 = 0;
const BOS_ID: u32 = 1;
const EOS_ID: u32 = 2;
// Adjacent and half-split RoPE are the same rotation; only f32 summation order differs between the two loads
const LOGPROB_TOLERANCE: f32 = 1e-3;

async fn greedy_trace(model: &Model, adapter: Option<&str>) -> anyhow::Result<Vec<(u32, f32)>> {
    let mut request =
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, PROMPT))
            .set_sampler_max_len(MAX_LEN)
            .set_sampler_topk(1)
            .return_logprobs(true)
            .set_sampler_topn_logprobs(1);
    if let Some(adapter) = adapter {
        request = request.set_adapter(adapter);
    }
    let response = model.send_chat_request(request).await?;
    let trace = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::ensure!(!trace.is_empty(), "the model generated nothing");
    Ok(trace)
}

fn assert_same_trace(
    gguf: &[(u32, f32)],
    safetensors: &[(u32, f32)],
    what: &str,
) -> anyhow::Result<()> {
    let same = gguf.len() == safetensors.len()
        && gguf
            .iter()
            .zip(safetensors)
            .all(|(g, s)| g.0 == s.0 && (g.1 - s.1).abs() < LOGPROB_TOLERANCE);
    anyhow::ensure!(
        same,
        "{what}: GGUF {gguf:?} differs from safetensors {safetensors:?}"
    );
    Ok(())
}

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

/// The tiny checkpoint as an f32 GGUF, with Q/K reordered as llama.cpp's converter writes them.
fn write_tiny_gguf(checkpoint: &Path, out: &Path) -> anyhow::Result<()> {
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
        tensors.push((gguf, QTensor::quantize(&tensor, GgmlDType::F32)?));
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
    let metadata = metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let mut file = std::fs::File::create(out)?;
    gguf_file::write(&mut file, &metadata, &tensors)?;
    Ok(())
}

// Touches both Q (two heads) and K (one KV head), in the half-split layout a Hugging Face adapter is trained in.
fn write_qk_adapter(dir: &Path) -> anyhow::Result<()> {
    std::fs::write(
        dir.join("adapter_config.json"),
        format!(
            r#"{{"r":{ADAPTER_RANK},"lora_alpha":{ADAPTER_RANK},"target_modules":["q_proj","k_proj"]}}"#
        ),
    )?;
    let ramp = |rows: usize, cols: usize, scale: f32| {
        let data = (0..rows * cols)
            .map(|i| ((i % 7) as f32 - 3.0) * scale)
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (rows, cols), &Device::Cpu)
    };
    let mut tensors = HashMap::new();
    for layer in 0..LAYERS {
        for (proj, out) in [
            ("q_proj", HEADS * HEAD_DIM),
            ("k_proj", KV_HEADS * HEAD_DIM),
        ] {
            let prefix = format!("base_model.model.model.layers.{layer}.self_attn.{proj}");
            tensors.insert(
                format!("{prefix}.lora_A.weight"),
                ramp(ADAPTER_RANK, HIDDEN, 0.3)?,
            );
            tensors.insert(
                format!("{prefix}.lora_B.weight"),
                ramp(out, ADAPTER_RANK, 0.5)?,
            );
        }
    }
    inference_tensor::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
    Ok(())
}

#[tokio::test]
async fn a_lora_adapter_on_an_adjacent_rope_gguf_matches_the_safetensors_model()
-> anyhow::Result<()> {
    let checkpoint = tiny_llama_checkpoint()?;
    let gguf_dir = tempfile::tempdir()?;
    write_tiny_gguf(checkpoint.path(), &gguf_dir.path().join(GGUF_FILE))?;
    let adapter = tempfile::tempdir()?;
    write_qk_adapter(adapter.path())?;

    let safetensors = LoraModelBuilder::from_text_model_builder(
        TextModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu(),
    )
    .with_adapter(ADAPTER, adapter.path().to_string_lossy())
    .build()
    .await?;
    let gguf = GgufModelBuilder::new(gguf_dir.path().to_string_lossy(), vec![GGUF_FILE])
        .with_tok_model_id(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_lora_adapter(ADAPTER, adapter.path().to_string_lossy())
        .build()
        .await?;

    let base = greedy_trace(&safetensors, None).await?;
    assert_same_trace(&greedy_trace(&gguf, None).await?, &base, "base model")?;
    let adapted = greedy_trace(&safetensors, Some(ADAPTER)).await?;
    anyhow::ensure!(adapted != base, "the adapter did not change the decode");
    assert_same_trace(
        &greedy_trace(&gguf, Some(ADAPTER)).await?,
        &adapted,
        "with the adapter",
    )
}
