//! The image and video input paths of Qwen2-VL and Qwen3-VL on tiny random-weight checkpoints built at test time.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};
use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
    VideoInput,
};

#[path = "../support/qwen_vl_tiny.rs"]
mod support;
use support::{tiny_qwen2_vl, tiny_qwen3_vl};

const PROMPT: &str = "describe";
// Long enough that a shared prefix runs past the media into whole paged blocks; paged hits never end inside media.
const LONG_PROMPT: &str = "describe every part of this picture in order, from the top left corner to the bottom right one.";
const MAX_LEN: usize = 6;
// Side lengths that resize to different patch grids, so each image yields its own token count.
const IMAGE_SIDES: [(u32, u32); 2] = [(56, 56), (84, 56)];
const VIDEO_FRAMES: usize = 4;
const VIDEO_FPS: f64 = 2.0;

const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));
// Below one image's token count, so a chunk boundary falls inside every image and video.
const PREFILL_CHUNK: usize = 3;
const PREFIX_CACHE_SEQS: usize = 16;
// Cached or chunked KV comes from a different prefill than a full recompute, so logprobs match only to rounding.
const LOGPROB_TOLERANCE: f32 = 1e-3;
// A scheduler spin never completes either request, so the mixed-batch check fails on this instead of hanging.
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

async fn build(dir: &Path) -> anyhow::Result<Model> {
    Ok(builder(dir).build().await?)
}

// A deterministic gradient, so the pixels (and so the vision tokens) are the same every run.
fn image(width: u32, height: u32, seed: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
        Rgb([
            (x * 255 / width) as u8 ^ seed,
            (y * 255 / height) as u8,
            seed.wrapping_mul(37),
        ])
    }))
}

fn video() -> VideoInput {
    let frames = (0..VIDEO_FRAMES)
        .map(|i| image(56, 56, i as u8 * 50))
        .collect();
    VideoInput::from_frames(frames, VIDEO_FPS, None)
}

fn greedy(request: RequestBuilder) -> RequestBuilder {
    request
        .set_sampler_max_len(MAX_LEN)
        .set_sampler_topk(1)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(1)
}

async fn greedy_ids(model: &Model, request: RequestBuilder) -> anyhow::Result<(Vec<u32>, usize)> {
    let response = model.send_chat_request(greedy(request)).await?;
    let ids = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| t.top_logprobs[0].token)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::ensure!(!ids.is_empty(), "the model generated nothing");
    Ok((ids, response.usage.prompt_tokens))
}

fn images(sides: &[(u32, u32)]) -> RequestBuilder {
    let images = sides
        .iter()
        .enumerate()
        .map(|(i, &(w, h))| image(w, h, i as u8 * 90))
        .collect();
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        images,
    ))
}

fn videos() -> RequestBuilder {
    prompted_video(PROMPT)
}

fn prompted_video(prompt: &str) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_video_message(
        TextMessageRole::User,
        prompt,
        vec![video()],
    ))
}

/// Greedy ids and prompt length for one image, two images of different sizes, and one video.
async fn traces(model: &Model) -> anyhow::Result<Vec<(Vec<u32>, usize)>> {
    Ok(vec![
        greedy_ids(model, images(&IMAGE_SIDES[..1])).await?,
        greedy_ids(model, images(&IMAGE_SIDES)).await?,
        greedy_ids(model, videos()).await?,
    ])
}

#[tokio::test]
async fn qwen2_vl_images_and_video() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen2_vl()?;
    let model = build(checkpoint.path()).await?;
    let traces = traces(&model).await?;
    // 27 text tokens, a start/end pair per medium; a 56x56 image is 4 merged patches, the 84x56 one resizes to 28x56
    // (2) under max_pixels, the 4-frame video is 2 temporal by 2x2 (8)
    let expected = vec![
        (vec![237, 100, 34, 185, 26, 163], 33),
        (vec![257, 257, 187, 143, 256, 31], 37),
        (vec![5, 74, 256, 166, 32, 236], 37),
    ];
    assert_eq!(traces, expected);
    Ok(())
}

#[tokio::test]
async fn qwen3_vl_images_and_video() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_vl()?;
    let model = build(checkpoint.path()).await?;
    let traces = traces(&model).await?;
    // the video prompt adds a timestamp per temporal patch in front of each frame's pads
    let expected = vec![
        (vec![91, 91, 91, 91, 91, 91], 33),
        (vec![91, 142, 39, 91, 225, 39], 37),
        (vec![161, 213, 91, 161, 213, 91], 61),
    ];
    assert_eq!(traces, expected);
    Ok(())
}

// (token, logprob) per greedy step and the prompt tokens served from the prefix cache.
async fn trace(model: &Model, request: RequestBuilder) -> anyhow::Result<(Vec<(u32, f32)>, usize)> {
    let response = model.send_chat_request(greedy(request)).await?;
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

// Same-size images give identical prompts, so only the registered media span keeps their cached blocks apart.
fn same_size_image(seed: u8) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        LONG_PROMPT,
        vec![image(56, 56, seed)],
    ))
}

async fn prefix_cache_serves_only_the_same_media(dir: &Path) -> anyhow::Result<()> {
    let fresh = build(dir).await?;
    let (fresh_a, _) = trace(&fresh, same_size_image(1)).await?;
    let (fresh_b, _) = trace(&fresh, same_size_image(200)).await?;
    let (fresh_video, _) = trace(&fresh, prompted_video(LONG_PROMPT)).await?;

    let warm = builder(dir)
        .with_prefix_cache_n(Some(PREFIX_CACHE_SEQS))
        .build()
        .await?;
    trace(&warm, same_size_image(1)).await?;
    let (b, _) = trace(&warm, same_size_image(200)).await?;
    anyhow::ensure!(
        same_decode(&b, &fresh_b),
        "image b was served image a's blocks: {b:?}"
    );
    let (a, cached) = trace(&warm, same_size_image(1)).await?;
    anyhow::ensure!(cached > 0, "image a was not served from the prefix cache");
    anyhow::ensure!(
        same_decode(&a, &fresh_a),
        "a prefix hit changed image a: {fresh_a:?} vs {a:?}"
    );
    trace(&warm, prompted_video(LONG_PROMPT)).await?;
    let (video, cached) = trace(&warm, prompted_video(LONG_PROMPT)).await?;
    // The non-paged prefix cacher never serves sequences with video; paged attention caches their blocks.
    anyhow::ensure!(
        (cached > 0) == ON_GPU,
        "video prefix hit {cached} tokens, expected a hit only with paged attention"
    );
    anyhow::ensure!(
        same_decode(&video, &fresh_video),
        "a prefix hit changed the video: {video:?}"
    );
    Ok(())
}

async fn chunked_prefill_matches_one_prefill(dir: &Path) -> anyhow::Result<()> {
    let whole = build(dir).await?;
    let chunked = builder(dir)
        .with_max_prefill_chunk_tokens(PREFILL_CHUNK)
        .build()
        .await?;
    for (what, request) in [("two images", images(&IMAGE_SIDES)), ("video", videos())] {
        let (expected, _) = trace(&whole, request.clone()).await?;
        let (steps, _) = trace(&chunked, request).await?;
        anyhow::ensure!(
            same_decode(&steps, &expected),
            "chunked {what}: {expected:?} vs {steps:?}"
        );
    }
    Ok(())
}

async fn text_in_the_batch_leaves_media_unchanged(dir: &Path) -> anyhow::Result<()> {
    let model = build(dir).await?;
    let (alone, _) = trace(&model, images(&IMAGE_SIDES)).await?;
    let text = RequestBuilder::new()
        .add_message(TextMessageRole::User, PROMPT)
        .set_sampler_max_len(MAX_LEN);
    let (batched, text) = tokio::time::timeout(MIXED_BATCH_TIMEOUT, async {
        tokio::join!(
            trace(&model, images(&IMAGE_SIDES)),
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

#[tokio::test]
async fn qwen2_vl_prefix_cache_serves_only_the_same_media() -> anyhow::Result<()> {
    prefix_cache_serves_only_the_same_media(tiny_qwen2_vl()?.path()).await
}

#[tokio::test]
async fn qwen3_vl_prefix_cache_serves_only_the_same_media() -> anyhow::Result<()> {
    prefix_cache_serves_only_the_same_media(tiny_qwen3_vl()?.path()).await
}

#[tokio::test]
async fn qwen2_vl_chunked_prefill_matches_one_prefill() -> anyhow::Result<()> {
    chunked_prefill_matches_one_prefill(tiny_qwen2_vl()?.path()).await
}

#[tokio::test]
async fn qwen3_vl_chunked_prefill_matches_one_prefill() -> anyhow::Result<()> {
    chunked_prefill_matches_one_prefill(tiny_qwen3_vl()?.path()).await
}

#[tokio::test]
async fn qwen2_vl_text_in_the_batch_leaves_media_unchanged() -> anyhow::Result<()> {
    text_in_the_batch_leaves_media_unchanged(tiny_qwen2_vl()?.path()).await
}

#[tokio::test]
async fn qwen3_vl_text_in_the_batch_leaves_media_unchanged() -> anyhow::Result<()> {
    text_in_the_batch_leaves_media_unchanged(tiny_qwen3_vl()?.path()).await
}
