//! Compute and compare text embeddings with cosine similarity.
//!
//! Run with: `cargo run --release --example embeddings -p inference-examples`

use anyhow::Result;
use inference::EmbeddingModelBuilder;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("Qwen/Qwen3-Embedding-0.6B")
        .with_logging()
        .build()
        .await?;

    let query = model.generate_embedding("What is graphene").await?;
    let related = model
        .generate_embedding("Graphene is a single layer of carbon atoms in a hexagonal lattice.")
        .await?;
    let unrelated = model
        .generate_embedding("The recipe calls for two cups of flour.")
        .await?;

    println!("Embedding dimension: {}", query.len());
    println!("related:   {:.4}", cosine(&query, &related));
    println!("unrelated: {:.4}", cosine(&query, &unrelated));

    Ok(())
}
