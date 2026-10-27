//! A tiny random-weight BERT embedder: encoder detection, bidirectional attention, tokenizer and length limits, ISQ.

use std::path::Path;

use inference::{
    EmbeddingModelBuilder, EmbeddingRequestBuilder, IsqType, Model, ModelDType,
    UqffEmbeddingModelBuilder,
};

#[path = "../support/bert_embedding_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::tiny_bert_checkpoint;
use traces::uqff_files;

const HIDDEN_SIZE: usize = 32;
// The Normalize module scales every embedding to unit length.
const NORM_TOLERANCE: f32 = 1e-4;
const EMBEDDING_TOLERANCE: f32 = 1e-5;
// Q8_0 on 32-wide layers keeps the embedding close, not equal
const ISQ_MIN_COSINE: f32 = 0.99;
// Prompts that differ only after their first token: CLS pooling sees the change only through bidirectional attention.
const PROMPT: &str = "hello there";
const PROMPT_CHANGED_LATER: &str = "hello where";
// A tokenizer padding as sentence-transformers exports it, longer than any prompt here.
const SAVED_PADDING: usize = 48;
const SENTENCE_MAX_SEQ_LENGTH: usize = 8;
const TOKENS: [u32; 9] = [40, 41, 42, 43, 44, 45, 46, 47, 48];
// RoBERTa positions start at pad_token_id + 1, so 10 position rows hold 8 tokens.
const ROBERTA_PAD_TOKEN_ID: usize = 1;
const ROBERTA_POSITIONS: usize = 10;

fn cpu_embedding_builder(dir: &Path) -> EmbeddingModelBuilder {
    EmbeddingModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

async fn embed_tokens(model: &Model, tokens: &[u32], truncate: bool) -> anyhow::Result<Vec<f32>> {
    let request = EmbeddingRequestBuilder::new()
        .add_tokens(tokens.to_vec())
        .with_truncate_sequence(truncate)
        .build()?;
    let response = serde_json::to_value(model.generate_embeddings(request).await?)?;
    Ok(serde_json::from_value(
        response["data"][0]["embedding"].clone(),
    )?)
}

#[tokio::test]
async fn an_encoder_class_is_detected_and_embeds_to_unit_length() -> anyhow::Result<()> {
    // no config_sentence_transformers.json: the BertModel class alone routes to the embedding loader
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let embedding = model.generate_embedding(PROMPT).await?;
    assert_eq!(embedding.len(), HIDDEN_SIZE);
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < NORM_TOLERANCE, "norm {norm}");
    Ok(())
}

#[tokio::test]
async fn the_first_token_attends_to_later_ones() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let first = model.generate_embedding(PROMPT).await?;
    let changed = model.generate_embedding(PROMPT_CHANGED_LATER).await?;
    let diff = max_diff(&first, &changed);
    assert!(
        diff > EMBEDDING_TOLERANCE,
        "a later token left the CLS embedding unchanged"
    );
    Ok(())
}

#[tokio::test]
async fn a_saved_tokenizer_padding_does_not_reach_the_prompt() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let expected = cpu_embedding_builder(checkpoint.path())
        .build()
        .await?
        .generate_embedding(PROMPT)
        .await?;

    let tokenizer_path = checkpoint.path().join("tokenizer.json");
    let mut tokenizer: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer_path)?)?;
    tokenizer["padding"] = serde_json::json!({
        "strategy": {"Fixed": SAVED_PADDING},
        "direction": "Right",
        "pad_to_multiple_of": null,
        "pad_id": 0,
        "pad_type_id": 0,
        "pad_token": "<unk>"
    });
    std::fs::write(&tokenizer_path, serde_json::to_string(&tokenizer)?)?;
    let actual = cpu_embedding_builder(checkpoint.path())
        .build()
        .await?
        .generate_embedding(PROMPT)
        .await?;
    let diff = max_diff(&expected, &actual);
    assert!(
        diff < EMBEDDING_TOLERANCE,
        "the saved padding moved the embedding by {diff}"
    );
    Ok(())
}

#[tokio::test]
async fn sentence_transformers_max_seq_length_limits_the_prompt() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    std::fs::write(
        checkpoint.path().join("sentence_bert_config.json"),
        serde_json::to_string(&serde_json::json!({"max_seq_length": SENTENCE_MAX_SEQ_LENGTH}))?,
    )?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    assert!(
        embed_tokens(&model, &TOKENS, false).await.is_err(),
        "a prompt past max_seq_length was accepted"
    );
    let truncated = embed_tokens(&model, &TOKENS, true).await?;
    let head = embed_tokens(&model, &TOKENS[..SENTENCE_MAX_SEQ_LENGTH], false).await?;
    assert!(max_diff(&truncated, &head) < EMBEDDING_TOLERANCE);
    Ok(())
}

#[tokio::test]
async fn roberta_positions_leave_room_for_the_pad_offset() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|config| {
        config["architectures"] = serde_json::json!(["RobertaModel"]);
        config["model_type"] = "roberta".into();
        config["pad_token_id"] = ROBERTA_PAD_TOKEN_ID.into();
        config["max_position_embeddings"] = ROBERTA_POSITIONS.into();
    })?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let longest = ROBERTA_POSITIONS - ROBERTA_PAD_TOKEN_ID - 1;
    assert_eq!(
        embed_tokens(&model, &TOKENS[..longest], false).await?.len(),
        HIDDEN_SIZE
    );
    assert!(
        embed_tokens(&model, &TOKENS[..=longest], false)
            .await
            .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn an_isq_encoder_and_a_reload_of_its_uqff_embed_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let unquantized = cpu_embedding_builder(checkpoint.path())
        .build()
        .await?
        .generate_embedding(PROMPT)
        .await?;
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_embedding_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let quantized_embedding = quantized.generate_embedding(PROMPT).await?;
    drop(quantized);
    let cosine: f32 = unquantized
        .iter()
        .zip(&quantized_embedding)
        .map(|(a, b)| a * b)
        .sum();
    assert!(cosine > ISQ_MIN_COSINE, "Q8_0 cosine {cosine}");

    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(!written.is_empty(), "no UQFF written");
    let reloaded = UqffEmbeddingModelBuilder::new(
        checkpoint.path().to_string_lossy(),
        vec![written[0].clone()],
    )
    .into_inner()
    .with_dtype(ModelDType::F32)
    .with_force_cpu()
    .build()
    .await?;
    assert_eq!(
        reloaded.generate_embedding(PROMPT).await?,
        quantized_embedding
    );
    Ok(())
}

// A GGUF-type ISQ keeps its biases in F32; a BF16 CUDA matmul has to add them in its own dtype.
#[tokio::test]
async fn a_bf16_gpu_encoder_with_biased_isq_layers_embeds() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let checkpoint = tiny_bert_checkpoint(|_| {})?;
    let model = EmbeddingModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_isq(IsqType::Q8_0)
        .build()
        .await?;
    assert_eq!(model.generate_embedding(PROMPT).await?.len(), HIDDEN_SIZE);
    Ok(())
}
