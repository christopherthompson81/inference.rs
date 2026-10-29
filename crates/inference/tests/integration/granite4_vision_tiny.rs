//! A tiny random-weight Granite 4 Vision: an image request expands to its tiles' tokens and generates, and a repeat
//! served from the prefix cache decodes the same tokens.

use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

#[path = "../support/granite4_vision_tiny.rs"]
mod support;
use support::tiny_checkpoint;

const PROMPT: &str = "Describe it.";
const MAX_LEN: usize = 6;
// wider than tall, so the image takes two tiles and the unpadding trims rows
const IMAGE_SIZE: (u32, u32) = (50, 30);
// base tile (2x2 features) plus two tiles unpadded to 2 rows of 4 and a newline each
const IMAGE_TOKENS: usize = 4 + 2 * (4 + 1);
// cached KV comes from a different prefill than a full recompute, so logprobs match only to rounding
const LOGPROB_TOLERANCE: f32 = 1e-3;

async fn build(dir: &std::path::Path) -> anyhow::Result<Model> {
    Ok(MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_prefix_cache_n(Some(16))
        .build()
        .await?)
}

fn image() -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(
        IMAGE_SIZE.0,
        IMAGE_SIZE.1,
        |x, y| image::Rgb([(x * 5) as u8, (y * 8) as u8, ((x + y) * 3) as u8]),
    ))
}

fn request(prompt: &str) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        prompt,
        vec![image()],
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
async fn an_image_expands_to_its_tiles_and_a_cached_repeat_decodes_the_same() -> anyhow::Result<()>
{
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text_only = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    let first = model.send_chat_request(request(PROMPT)).await?;
    // the image adds its tokens and the `<image>\n` line's newline beyond the text-only prompt
    let added = first.usage.prompt_tokens - text_only.usage.prompt_tokens;
    assert!(
        (IMAGE_TOKENS..=IMAGE_TOKENS + 1).contains(&added),
        "the image added {added} prompt tokens, expected {IMAGE_TOKENS} and a newline"
    );
    let first = trace(&first);
    assert!(!first.is_empty(), "the image request generated nothing");
    let second = trace(&model.send_chat_request(request(PROMPT)).await?);
    assert_eq!(first.len(), second.len());
    for ((a, la), (b, lb)) in first.iter().zip(&second) {
        assert_eq!(a, b);
        assert!((la - lb).abs() < LOGPROB_TOLERANCE, "{la} vs {lb}");
    }
    Ok(())
}
