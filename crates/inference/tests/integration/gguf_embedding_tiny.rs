//! Tiny BERT and Nomic BERT embedders written as GGUFs under llama.cpp's tensor names and metadata, against the
//! safetensors checkpoints they came from.

use std::collections::BTreeMap;
use std::path::Path;

use inference::{EmbeddingModelBuilder, GgufModelBuilder, Model, ModelDType};
use inference_tensor::Device;
use inference_tensor::quantized::{GgmlDType, QTensor, gguf_file};

#[path = "../support/bert_embedding_tiny.rs"]
mod bert;
#[path = "../support/rope_encoders_tiny.rs"]
mod rope;

const PROMPT: &str = "hello there, how are you";
const GGUF_FILE: &str = "tiny.gguf";
const F32_TOLERANCE: f32 = 1e-5;
// Q8_0 on 32-wide layers keeps the embedding close, not equal
const Q8_MIN_COSINE: f32 = 0.99;
// llama.cpp's LLAMA_POOLING_TYPE_MEAN / _CLS
const POOLING_MEAN: u32 = 1;
const POOLING_CLS: u32 = 2;
// From the committed tiny configs
const HIDDEN: u32 = 32;
const LAYERS: u32 = 2;
const HEADS: u32 = 2;
const BERT_INTERMEDIATE: u32 = 64;
const BERT_POSITIONS: u32 = 64;
const NOMIC_INTERMEDIATE: u32 = 64;
const NOMIC_POSITIONS: u32 = 64;
const BERT_EPS: f32 = 1e-12;
const NOMIC_ROPE_BASE: f32 = 1000.0;

/// The GGUF name llama.cpp's converter gives each native tensor.
fn gguf_name(native: &str, nomic: bool) -> Option<String> {
    let root = match native {
        "embeddings.word_embeddings.weight" => Some("token_embd.weight"),
        "embeddings.position_embeddings.weight" => Some("position_embd.weight"),
        "embeddings.token_type_embeddings.weight" => Some("token_types.weight"),
        "embeddings.LayerNorm.weight" | "emb_ln.weight" => Some("token_embd_norm.weight"),
        "embeddings.LayerNorm.bias" | "emb_ln.bias" => Some("token_embd_norm.bias"),
        _ => None,
    };
    if let Some(root) = root {
        return Some(root.to_string());
    }
    let layers = if nomic {
        "encoder.layers."
    } else {
        "encoder.layer."
    };
    let rest = native.strip_prefix(layers)?;
    let (block, rest) = rest.split_once('.')?;
    let (module, part) = rest.rsplit_once('.')?;
    let role = match module {
        "attention.self.query" => "attn_q",
        "attention.self.key" => "attn_k",
        "attention.self.value" => "attn_v",
        "attention.output.dense" | "attn.out_proj" => "attn_output",
        "attention.output.LayerNorm" | "norm1" => "attn_output_norm",
        "intermediate.dense" | "mlp.fc11" => "ffn_up",
        "mlp.fc12" => "ffn_gate",
        "output.dense" | "mlp.fc2" => "ffn_down",
        "output.LayerNorm" | "norm2" => "layer_output_norm",
        "attn.Wqkv" => "attn_qkv",
        _ => return None,
    };
    Some(format!("blk.{block}.{role}.{part}"))
}

fn tokens_by_id(tokenizer_json: &Path) -> anyhow::Result<Vec<gguf_file::Value>> {
    let tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(tokenizer_json)?)?;
    let mut by_id = BTreeMap::new();
    for (token, id) in tokenizer["model"]["vocab"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no model.vocab"))?
    {
        by_id.insert(id.as_u64().unwrap_or_default(), token.clone());
    }
    for added in tokenizer["added_tokens"].as_array().into_iter().flatten() {
        by_id.insert(
            added["id"].as_u64().unwrap_or_default(),
            added["content"].as_str().unwrap_or_default().to_string(),
        );
    }
    Ok(by_id.into_values().map(gguf_file::Value::String).collect())
}

/// The checkpoint's weights as a GGUF, matrices in `matrix_dtype`, with the metadata llama.cpp writes.
fn write_gguf(
    checkpoint: &Path,
    nomic: bool,
    matrix_dtype: GgmlDType,
    out: &Path,
) -> anyhow::Result<()> {
    let weights =
        inference_tensor::safetensors::load(checkpoint.join("model.safetensors"), &Device::Cpu)?;
    let mut tensors = Vec::new();
    for (name, tensor) in &weights {
        let Some(gguf) = gguf_name(name, nomic) else {
            continue;
        };
        let dtype = if tensor.rank() == 2 && !gguf.starts_with("token_types") {
            matrix_dtype
        } else {
            GgmlDType::F32
        };
        tensors.push((gguf, QTensor::quantize(tensor, dtype)?));
    }
    let arch = if nomic { "nomic-bert" } else { "bert" };
    let key = |k: &str| format!("{arch}.{k}");
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            gguf_file::Value::String(arch.into()),
        ),
        (key("block_count"), gguf_file::Value::U32(LAYERS)),
        (key("embedding_length"), gguf_file::Value::U32(HIDDEN)),
        (key("attention.head_count"), gguf_file::Value::U32(HEADS)),
        (
            key("attention.layer_norm_epsilon"),
            gguf_file::Value::F32(BERT_EPS),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            gguf_file::Value::Array(tokens_by_id(&checkpoint.join("tokenizer.json"))?),
        ),
    ];
    if nomic {
        metadata.extend([
            (
                key("feed_forward_length"),
                gguf_file::Value::U32(NOMIC_INTERMEDIATE),
            ),
            (
                key("context_length"),
                gguf_file::Value::U32(NOMIC_POSITIONS),
            ),
            (
                key("rope.freq_base"),
                gguf_file::Value::F32(NOMIC_ROPE_BASE),
            ),
            (key("pooling_type"), gguf_file::Value::U32(POOLING_MEAN)),
        ]);
    } else {
        metadata.extend([
            (
                key("feed_forward_length"),
                gguf_file::Value::U32(BERT_INTERMEDIATE),
            ),
            (key("context_length"), gguf_file::Value::U32(BERT_POSITIONS)),
            (key("pooling_type"), gguf_file::Value::U32(POOLING_CLS)),
        ]);
    }
    let metadata = metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    let tensors = tensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();
    gguf_file::write(&mut std::fs::File::create(out)?, &metadata, &tensors)?;
    Ok(())
}

async fn safetensors_embedding(checkpoint: &Path) -> anyhow::Result<Vec<f32>> {
    let model = EmbeddingModelBuilder::new(checkpoint.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    Ok(model.generate_embedding(PROMPT).await?)
}

async fn gguf_model(checkpoint: &Path, dir: &Path) -> anyhow::Result<Model> {
    // the tokenizer comes from the checkpoint; the GGUF vocabularies have their own conversion tests
    Ok(
        GgufModelBuilder::new(dir.to_string_lossy(), vec![GGUF_FILE])
            .with_tok_model_id(checkpoint.to_string_lossy())
            .with_dtype(ModelDType::F32)
            .with_force_cpu()
            .build()
            .await?,
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

async fn assert_gguf_embeds_as_its_source(checkpoint: &Path, nomic: bool) -> anyhow::Result<()> {
    let expected = safetensors_embedding(checkpoint).await?;
    let f32_dir = tempfile::tempdir()?;
    write_gguf(
        checkpoint,
        nomic,
        GgmlDType::F32,
        &f32_dir.path().join(GGUF_FILE),
    )?;
    let f32_embedding = gguf_model(checkpoint, f32_dir.path())
        .await?
        .generate_embedding(PROMPT)
        .await?;
    let diff = expected
        .iter()
        .zip(&f32_embedding)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    anyhow::ensure!(diff < F32_TOLERANCE, "F32 GGUF differs by {diff}");

    let q8_dir = tempfile::tempdir()?;
    write_gguf(
        checkpoint,
        nomic,
        GgmlDType::Q8_0,
        &q8_dir.path().join(GGUF_FILE),
    )?;
    let q8_embedding = gguf_model(checkpoint, q8_dir.path())
        .await?
        .generate_embedding(PROMPT)
        .await?;
    let cos = cosine(&expected, &q8_embedding);
    anyhow::ensure!(cos > Q8_MIN_COSINE, "Q8_0 GGUF cosine {cos}");
    Ok(())
}

#[tokio::test]
async fn a_bert_gguf_embeds_as_its_safetensors_source() -> anyhow::Result<()> {
    assert_gguf_embeds_as_its_source(bert::tiny_bert_checkpoint(|_| {})?.path(), false).await
}

#[tokio::test]
async fn a_nomic_bert_gguf_embeds_as_its_safetensors_source() -> anyhow::Result<()> {
    assert_gguf_embeds_as_its_source(rope::tiny_nomic_bert_checkpoint(|_| {})?.path(), true).await
}
