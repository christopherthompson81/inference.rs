//! PaddleOCR-VL document OCR: recognize one image region and print the text.
//!
//! Task prompts: `OCR:` (text), `Table Recognition:` (OTSL markup), `Formula Recognition:` (LaTeX).
//!
//! Run with:
//! `cargo run --release --example paddleocr_vl_recognize -p inference -- <image> ["Table Recognition:"]`

use anyhow::{bail, Result};
use inference::{MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole};

// Real crops stop on EOS well under this; the cap only stops a pathological region running away.
const MAX_NEW_TOKENS: usize = 2048;

#[tokio::main]
async fn main() -> Result<()> {
    let Some(image_path) = std::env::args().nth(1) else {
        bail!("usage: paddleocr_vl_recognize <image> [prompt]");
    };
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "OCR:".to_string());

    let model = MultimodalModelBuilder::new("PaddlePaddle/PaddleOCR-VL-1.6")
        .with_logging()
        .build()
        .await?;

    let image = image::open(&image_path)?;
    let messages =
        MultimodalMessages::new().add_image_message(TextMessageRole::User, &prompt, vec![image]);
    let response = model
        .send_chat_request(RequestBuilder::from(messages).set_sampler_max_len(MAX_NEW_TOKENS))
        .await?;

    println!(
        "{}",
        response.choices[0].message.content.as_deref().unwrap_or("")
    );
    Ok(())
}
