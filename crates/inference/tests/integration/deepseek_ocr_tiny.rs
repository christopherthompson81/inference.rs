//! A tiny random-weight DeepSeek-OCR: a tiled image and a single-view one expand to their tokens and generate, and a
//! repeat served from the prefix and encoder caches decodes the same tokens.

use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

#[path = "../support/deepseek_ocr_tiny.rs"]
mod support;
use support::tiny_checkpoint;

const PROMPT: &str = "Free OCR.";
const MAX_LEN: usize = 6;
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));
// wider than one 640 tile, nearest a 2x1 grid of tiles
const TILED_SIZE: (u32, u32) = (700, 400);
const SINGLE_SIZE: (u32, u32) = (300, 200);
// the 256 global view's 4x4 grid and 640 tiles' 10x10, each row ending in a newline, then the separator
const GLOBAL_TOKENS: usize = 4 * 5 + 1;
const TILED_TOKENS: usize = 10 * (2 * 10 + 1) + GLOBAL_TOKENS;
// cached KV comes from a different prefill than a full recompute, so logprobs match only to rounding
const LOGPROB_TOLERANCE: f32 = 1e-3;

async fn build(dir: &std::path::Path) -> anyhow::Result<Model> {
    let mut builder =
        MultimodalModelBuilder::new(dir.to_string_lossy()).with_dtype(ModelDType::F32);
    if !ON_GPU {
        builder = builder.with_force_cpu();
    }
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        builder = builder.with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?);
    }
    Ok(builder.with_prefix_cache_n(Some(16)).build().await?)
}

fn image((width, height): (u32, u32)) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x + y) % 239) as u8])
    }))
}

fn request() -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image(TILED_SIZE), image(SINGLE_SIZE)],
    ))
    .set_sampler_max_len(MAX_LEN)
    .set_sampler_topk(1)
    .return_logprobs(true)
    .set_sampler_topn_logprobs(1)
}

fn trace(resp: &inference::ChatCompletionResponse) -> Vec<(u32, f32)> {
    resp.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect()
        })
        .unwrap_or_default()
}

#[tokio::test]
async fn tiled_and_single_view_images_expand_and_a_cached_repeat_decodes_the_same()
-> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text_only = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    let first = model.send_chat_request(request()).await?;
    // each image's `<image>\n` line adds its tokens and a newline the tokenizer may merge
    let added = first.usage.prompt_tokens - text_only.usage.prompt_tokens;
    let image_tokens = TILED_TOKENS + GLOBAL_TOKENS;
    assert!(
        (image_tokens..=image_tokens + 2).contains(&added),
        "the images added {added} prompt tokens, expected {image_tokens} and their newlines"
    );
    let first = trace(&first);
    assert!(!first.is_empty(), "the image request generated nothing");
    let second = trace(&model.send_chat_request(request()).await?);
    assert_eq!(first.len(), second.len());
    for ((a, la), (b, lb)) in first.iter().zip(&second) {
        assert_eq!(a, b);
        assert!((la - lb).abs() < LOGPROB_TOLERANCE, "{la} vs {lb}");
    }
    Ok(())
}
