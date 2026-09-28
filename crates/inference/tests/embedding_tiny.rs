//! The embedding load paths on a tiny random-weight Qwen3 embedder: plain, in-situ quantized, and reloaded from UQFF.

use std::path::{Path, PathBuf};

use inference::{EmbeddingModelBuilder, IsqType, ModelDType, UqffEmbeddingModelBuilder};

#[path = "support/qwen3_embedding_tiny.rs"]
mod support;
use support::tiny_embedding_checkpoint;

const PROMPT: &str = "hello";
const HIDDEN_SIZE: usize = 32;
const UQFF_EXTENSION: &str = "uqff";
// The Normalize module scales every embedding to unit length.
const NORM_TOLERANCE: f32 = 1e-4;

fn cpu_embedding_builder(dir: &Path) -> EmbeddingModelBuilder {
    EmbeddingModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

fn uqff_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|path| path.extension().is_some_and(|ext| ext == UQFF_EXTENSION));
    files.sort();
    Ok(files)
}

#[tokio::test]
async fn an_embedding_checkpoint_loads_and_embeds_to_unit_length() -> anyhow::Result<()> {
    let checkpoint = tiny_embedding_checkpoint()?;
    let model = cpu_embedding_builder(checkpoint.path()).build().await?;
    let embedding = model.generate_embedding(PROMPT).await?;
    assert_eq!(embedding.len(), HIDDEN_SIZE);
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < NORM_TOLERANCE, "norm {norm}");
    assert_eq!(model.generate_embedding(PROMPT).await?, embedding);
    Ok(())
}

#[tokio::test]
async fn an_isq_embedder_and_a_reload_of_its_uqff_embed_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_embedding_checkpoint()?;
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_embedding_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let quantized_embedding = quantized.generate_embedding(PROMPT).await?;
    drop(quantized);

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
