//! Gemma 3's text and image paths on a tiny random-weight checkpoint: its image tokens attend to each other both
//! ways, its sliding and global layers take different RoPE, and GPU builds take the paged media-prefix path.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};
use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

#[path = "../support/gemma3_tiny.rs"]
mod support;
use support::tiny_gemma3;

const PROMPT: &str = "describe";
// Longer than the sliding window and long enough that a shared prefix runs past the image into whole paged blocks.
const LONG_PROMPT: &str = "describe every part of this picture in order, from the top left corner to the bottom right one.";
const MAX_LEN: usize = 6;
const IMAGE_SIDE: u32 = 48;
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));
const PREFIX_CACHE_SEQS: usize = 16;
// Cached or chunked KV comes from a different prefill than a full recompute, so logprobs match only to rounding.
const LOGPROB_TOLERANCE: f32 = 1e-3;
// A CPU run repeats exactly, so the pins hold to well under the gap between the image and text traces.
#[cfg(not(any(feature = "cuda", feature = "metal")))]
const PIN_TOLERANCE: f32 = 1e-5;
const MIXED_BATCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

fn builder(dir: &Path) -> MultimodalModelBuilder {
    let builder = MultimodalModelBuilder::new(dir.to_string_lossy()).with_dtype(ModelDType::F32);
    // GPU builds take the paged path, so `--cuda` covers paged media prefill.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    let builder = builder.with_paged_attn(
        inference::PagedAttentionMetaBuilder::default()
            .build()
            .unwrap(),
    );
    if ON_GPU {
        builder
    } else {
        builder.with_force_cpu()
    }
}

// A deterministic gradient, so the pixels (and so the image tokens) are the same every run.
fn image(seed: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(IMAGE_SIDE, IMAGE_SIDE, |x, y| {
        Rgb([
            (x * 255 / IMAGE_SIDE) as u8 ^ seed,
            (y * 255 / IMAGE_SIDE) as u8,
            seed.wrapping_mul(37),
        ])
    }))
}

fn image_request(prompt: &str, seed: u8) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        prompt,
        vec![image(seed)],
    ))
}

fn text_request(prompt: &str) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_message(TextMessageRole::User, prompt))
}

// (token, logprob) per greedy step and the prompt tokens served from the prefix cache.
async fn trace(model: &Model, request: RequestBuilder) -> anyhow::Result<(Vec<(u32, f32)>, usize)> {
    let request = request
        .set_sampler_max_len(MAX_LEN)
        .set_sampler_topk(1)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(1);
    let response = model.send_chat_request(request).await?;
    let steps = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::ensure!(!steps.is_empty(), "the model generated nothing");
    let cached = response
        .usage
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |details| details.cached_tokens);
    Ok((steps, cached))
}

fn same_decode(a: &[(u32, f32)], b: &[(u32, f32)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < LOGPROB_TOLERANCE)
}

#[tokio::test]
async fn gemma3_prefix_cache_serves_only_the_same_image() -> anyhow::Result<()> {
    let checkpoint = tiny_gemma3()?;
    // The SDK turns the prefix cache on by default, so the reference model has it off.
    let fresh = builder(checkpoint.path())
        .with_prefix_cache_n(None)
        .build()
        .await?;
    let (fresh_a, _) = trace(&fresh, image_request(LONG_PROMPT, 1)).await?;
    let (fresh_b, _) = trace(&fresh, image_request(LONG_PROMPT, 200)).await?;

    let warm = builder(checkpoint.path())
        .with_prefix_cache_n(Some(PREFIX_CACHE_SEQS))
        .build()
        .await?;
    trace(&warm, image_request(LONG_PROMPT, 1)).await?;
    let (b, _) = trace(&warm, image_request(LONG_PROMPT, 200)).await?;
    anyhow::ensure!(
        same_decode(&b, &fresh_b),
        "image b was served image a's blocks: {b:?}"
    );
    let (a, cached) = trace(&warm, image_request(LONG_PROMPT, 1)).await?;
    // Only the paged prefix cache serves Gemma 3 image prompts; the sequence cacher has no media features to match
    anyhow::ensure!(
        (cached > 0) == ON_GPU,
        "image a hit {cached} cached tokens, expected a hit only with paged attention"
    );
    anyhow::ensure!(
        same_decode(&a, &fresh_a),
        "a prefix hit changed image a: {fresh_a:?} vs {a:?}"
    );
    Ok(())
}

// With the prefix cache on, the batched image request would reuse the first run's image blocks instead of prefilling.
#[tokio::test]
async fn gemma3_text_in_the_batch_leaves_the_image_unchanged() -> anyhow::Result<()> {
    let checkpoint = tiny_gemma3()?;
    let model = builder(checkpoint.path())
        .with_prefix_cache_n(None)
        .build()
        .await?;
    let (alone, _) = trace(&model, image_request(PROMPT, 7)).await?;
    let text = text_request(PROMPT).set_sampler_max_len(MAX_LEN);
    let (batched, text) = tokio::time::timeout(MIXED_BATCH_TIMEOUT, async {
        tokio::join!(
            trace(&model, image_request(PROMPT, 7)),
            model.send_chat_request(text)
        )
    })
    .await?;
    text?;
    let (batched, _) = batched?;
    anyhow::ensure!(
        same_decode(&batched, &alone),
        "a text-only request in the batch changed the image output: {alone:?} vs {batched:?}"
    );
    Ok(())
}

// Pins CPU decoding of an image and a text prompt: image tokens attend both ways, sliding layers use the local RoPE.
#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[tokio::test]
async fn gemma3_image_and_text_decode_as_recorded() -> anyhow::Result<()> {
    let checkpoint = tiny_gemma3()?;
    let model = builder(checkpoint.path())
        .with_prefix_cache_n(None)
        .build()
        .await?;
    let (image, _) = trace(&model, image_request(LONG_PROMPT, 3)).await?;
    let (text, _) = trace(&model, text_request(LONG_PROMPT)).await?;
    let expected_image: &[(u32, f32)] = &[
        (11, -3.6861477),
        (11, -3.6906662),
        (11, -3.7016187),
        (11, -3.70917),
        (11, -3.7074795),
        (11, -3.7125883),
    ];
    let expected_text: &[(u32, f32)] = &[
        (11, -3.6884634),
        (11, -3.6927276),
        (11, -3.703736),
        (11, -3.7107918),
        (11, -3.7083218),
        (11, -3.7133746),
    ];
    let pinned = |actual: &[(u32, f32)], expected: &[(u32, f32)]| {
        actual.len() == expected.len()
            && actual
                .iter()
                .zip(expected)
                .all(|(a, e)| a.0 == e.0 && (a.1 - e.1).abs() < PIN_TOLERANCE)
    };
    anyhow::ensure!(
        pinned(&image, expected_image),
        "image decode moved: {image:?}"
    );
    anyhow::ensure!(pinned(&text, expected_text), "text decode moved: {text:?}");
    Ok(())
}
