//! Custom logits processor that modifies token probabilities during generation.
//!
//! Run with: `cargo run --release --example logits_processor -p inference-examples`

use anyhow::Result;
use inference::{
    IsqBits, ModelBuilder, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, in_place,
};
use rand::Rng;

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    let mut rng = rand::rng();
    let random_value: f32 = rng.random_range(0.0..=1.0);
    let threshold: f32 = rng.random_range(0.0..=0.5);

    let request = RequestBuilder::new()
        .add_logits_processor(in_place(move |logits, _context| {
            logits.iter_mut().for_each(|logit| *logit *= random_value);
            Ok(())
        }))
        // Zeroes every logit under the threshold.
        .add_logits_processor(in_place(move |logits, _context| {
            logits
                .iter_mut()
                .filter(|logit| **logit < threshold)
                .for_each(|logit| *logit = 0.0);
            Ok(())
        }))
        .add_message(
            TextMessageRole::User,
            "Please write a mathematical equation where a few numbers are added.",
        );

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
