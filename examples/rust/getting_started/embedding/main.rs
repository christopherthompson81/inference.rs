//! Generate text embeddings using an embedding model.
//!
//! Run with: `cargo run --release --example embedding -p inference-examples`

use anyhow::Result;
use inference::{EmbeddingModelBuilder, EmbeddingRequestBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = EmbeddingModelBuilder::new("google/embeddinggemma-300m")
        .with_logging()
        .build()
        .await?;

    let embeddings = model
        .generate_embeddings(
            EmbeddingRequestBuilder::new()
                .add_prompt("task: search result | query: What is graphene?")
                .build()?,
        )
        .await?;
    println!("{:?}", embeddings.data.first().map(|data| &data.embedding));

    Ok(())
}
