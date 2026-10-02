//! Per-layer quantization control using a Topology.
//!
//! Run with: `cargo run --release --example topology -p inference-examples`

use anyhow::Result;
use inference::{IsqBits, ModelBuilder, PagedAttentionMetaBuilder, TextMessageRole, TextMessages};

// Layers 0-8 at Q3K, 8-16 at Q4K, 16-24 at Q6K and 24-32 at Q8_0; the rest take the auto ISQ.
const TOPOLOGY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/quantization/topology/topology.yml"
);

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Eight)
        .with_topology_from_path(TOPOLOGY)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
