//! Tiny random-weight Nomic BERT and ModernBERT embedders: detection, ModernBERT's sliding window, ISQ and UQFF.

use std::path::Path;

use inference::{
    EmbeddingModelBuilder, EmbeddingRequestBuilder, IsqType, Model, ModelDType,
    UqffEmbeddingModelBuilder,
};

#[path = "../support/rope_encoders_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::{prefix_weights, tiny_modern_bert_checkpoint, tiny_nomic_bert_checkpoint};
use traces::uqff_files;

const PROMPT: &str = "hello there";
const HIDDEN_SIZE: usize = 32;
// The Normalize module scales every embedding to unit length.
const NORM_TOLERANCE: f32 = 1e-4;
const EMBEDDING_TOLERANCE: f32 = 1e-5;
// The fixture's local_attention of 4 attends 2 tokens each way; 3 tokens all see each other, 9 do not.
const INSIDE_WINDOW: [u32; 3] = [40, 41, 42];
const PAST_WINDOW: [u32; 9] = [40, 41, 42, 43, 44, 45, 46, 47, 48];
const WIDE_LOCAL_ATTENTION: usize = 64;

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

async fn embed_tokens(model: &Model, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
    let request = EmbeddingRequestBuilder::new()
        .add_tokens(tokens.to_vec())
        .build()?;
    let response = serde_json::to_value(model.generate_embeddings(request).await?)?;
    Ok(serde_json::from_value(
        response["data"][0]["embedding"].clone(),
    )?)
}

async fn assert_unit_embedding(dir: &Path) -> anyhow::Result<()> {
    let model = cpu_embedding_builder(dir).build().await?;
    let embedding = model.generate_embedding(PROMPT).await?;
    assert_eq!(embedding.len(), HIDDEN_SIZE);
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < NORM_TOLERANCE, "norm {norm}");
    Ok(())
}

async fn assert_uqff_reload_matches(dir: &Path) -> anyhow::Result<()> {
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_embedding_builder(dir)
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let expected = quantized.generate_embedding(PROMPT).await?;
    drop(quantized);
    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(!written.is_empty(), "no UQFF written");
    let reloaded = UqffEmbeddingModelBuilder::new(dir.to_string_lossy(), vec![written[0].clone()])
        .into_inner()
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    assert_eq!(reloaded.generate_embedding(PROMPT).await?, expected);
    Ok(())
}

#[tokio::test]
async fn rope_encoders_are_detected_by_class_and_embed_to_unit_length() -> anyhow::Result<()> {
    assert_unit_embedding(tiny_nomic_bert_checkpoint()?.path()).await?;
    assert_unit_embedding(tiny_modern_bert_checkpoint(|_| {})?.path()).await
}

#[tokio::test]
async fn modern_bert_sliding_layers_see_only_their_window() -> anyhow::Result<()> {
    let narrow = tiny_modern_bert_checkpoint(|_| {})?;
    let narrow = cpu_embedding_builder(narrow.path()).build().await?;
    // the same weights, with a window wider than any prompt here
    let wide = tiny_modern_bert_checkpoint(|config| {
        config["local_attention"] = WIDE_LOCAL_ATTENTION.into();
    })?;
    let wide = cpu_embedding_builder(wide.path()).build().await?;

    let inside = max_diff(
        &embed_tokens(&narrow, &INSIDE_WINDOW).await?,
        &embed_tokens(&wide, &INSIDE_WINDOW).await?,
    );
    assert!(
        inside < EMBEDDING_TOLERANCE,
        "a prompt inside the window moved by {inside}"
    );
    let past = max_diff(
        &embed_tokens(&narrow, &PAST_WINDOW).await?,
        &embed_tokens(&wide, &PAST_WINDOW).await?,
    );
    assert!(
        past > EMBEDDING_TOLERANCE,
        "the window left a long prompt unchanged"
    );
    Ok(())
}

#[tokio::test]
async fn an_isq_nomic_bert_and_a_reload_of_its_uqff_embed_alike() -> anyhow::Result<()> {
    assert_uqff_reload_matches(tiny_nomic_bert_checkpoint()?.path()).await
}

#[tokio::test]
async fn a_prefixed_isq_modern_bert_and_a_reload_of_its_uqff_embed_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_modern_bert_checkpoint(|config| {
        config["architectures"] = serde_json::json!(["ModernBertForMaskedLM"]);
    })?;
    prefix_weights(checkpoint.path(), "model")?;
    assert_uqff_reload_matches(checkpoint.path()).await
}
