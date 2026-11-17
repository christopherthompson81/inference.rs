//! A tiny random-weight DeepSeek-OCR and OCR-2: a tiled image and a single-view one expand to their tokens and
//! generate, and a repeat served from the prefix and encoder caches decodes the same tokens.

use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

#[path = "../support/deepseek_ocr_tiny.rs"]
mod support;
use support::{tiny_checkpoint, tiny_checkpoint_with};

const PROMPT: &str = "Free OCR.";
const MAX_LEN: usize = 6;
const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));
// wider than one 640 tile, nearest a 2x1 grid of tiles
const TILED_SIZE: (u32, u32) = (700, 400);
const SINGLE_SIZE: (u32, u32) = (300, 200);
// the 256 global view's 4x4 grid and 640 tiles' 10x10, each row ending in a newline, then the separator
const GLOBAL_TOKENS: usize = 4 * 5 + 1;
const TILED_TOKENS: usize = 10 * (2 * 10 + 1) + GLOBAL_TOKENS;
// OCR-2 over the tiny checkpoint: a 768 tile's 12x12 queries, the 256 global view's 4x4, no newlines
const OCR2: &str = "DeepseekOCR2ForCausalLM";
const OCR2_WIDTH: usize = 32;
// wider than one 768 tile, nearest a 2x1 grid of tiles
const OCR2_TILED_SIZE: (u32, u32) = (1000, 500);
const OCR2_GLOBAL_TOKENS: usize = 4 * 4 + 1;
const OCR2_TILED_TOKENS: usize = 2 * 12 * 12 + OCR2_GLOBAL_TOKENS;
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

fn request(tiled: (u32, u32)) -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image(tiled), image(SINGLE_SIZE)],
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

// a tiled image of `tiled` pixels and a single-view one add `image_tokens`, and a cached repeat decodes the same
async fn images_expand_and_repeat(
    checkpoint: &std::path::Path,
    tiled: (u32, u32),
    image_tokens: usize,
) -> anyhow::Result<()> {
    let model = build(checkpoint).await?;
    let text_only = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    let first = model.send_chat_request(request(tiled)).await?;
    // each image's `<image>\n` line adds its tokens and a newline the tokenizer may merge
    let added = first.usage.prompt_tokens - text_only.usage.prompt_tokens;
    assert!(
        (image_tokens..=image_tokens + 2).contains(&added),
        "the images added {added} prompt tokens, expected {image_tokens} and their newlines"
    );
    let first = trace(&first);
    assert!(!first.is_empty(), "the image request generated nothing");
    let second = trace(&model.send_chat_request(request(tiled)).await?);
    assert_eq!(first.len(), second.len());
    for ((a, la), (b, lb)) in first.iter().zip(&second) {
        assert_eq!(a, b);
        assert!((la - lb).abs() < LOGPROB_TOLERANCE, "{la} vs {lb}");
    }
    Ok(())
}

#[tokio::test]
async fn tiled_and_single_view_images_expand_and_a_cached_repeat_decodes_the_same()
-> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    images_expand_and_repeat(checkpoint.path(), TILED_SIZE, TILED_TOKENS + GLOBAL_TOKENS).await
}

// the tiny checkpoint with OCR-2's query encoder in place of CLIP
fn make_ocr2(config: &mut serde_json::Value) {
    config["architectures"] = serde_json::json!([OCR2]);
    let width = &mut config["vision_config"]["width"];
    width
        .as_object_mut()
        .expect("the vision widths are an object")
        .remove("clip-l-14-224");
    width["qwen2-0-5b"] = serde_json::json!({
        "dim": OCR2_WIDTH,
        "layers": 2,
        "heads": 2,
        "kv_heads": 1,
        "intermediate_size": 48,
    });
    config["projector_config"]["input_dim"] = serde_json::json!(OCR2_WIDTH);
}

#[tokio::test]
async fn ocr2_tiled_and_single_view_images_expand_and_a_cached_repeat_decodes_the_same()
-> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint_with(make_ocr2)?;
    images_expand_and_repeat(
        checkpoint.path(),
        OCR2_TILED_SIZE,
        OCR2_TILED_TOKENS + OCR2_GLOBAL_TOKENS,
    )
    .await
}

// Unlimited-OCR's decode window over the tiny checkpoint: small, so a short reply runs past it
const UNLIMITED: &str = "UnlimitedOCRForCausalLM";
const UNLIMITED_WINDOW: usize = 4;
const UNLIMITED_MAX_LEN: usize = 12;
// bf16 decode over the window's kernel and the plain one agree to rounding while the window still covers the reply
const WINDOWED_TOLERANCE: f32 = 2e-2;

fn make_unlimited(dir: &std::path::Path) -> anyhow::Result<()> {
    let path = dir.join("config.json");
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    config["architectures"] = serde_json::json!([UNLIMITED]);
    config["language_config"]["sliding_window_size"] = serde_json::json!(UNLIMITED_WINDOW);
    std::fs::write(path, serde_json::to_string(&config)?)?;
    Ok(())
}

async fn build_bf16(dir: &std::path::Path) -> anyhow::Result<Model> {
    let builder = MultimodalModelBuilder::new(dir.to_string_lossy()).with_dtype(ModelDType::BF16);
    #[cfg(any(feature = "cuda", feature = "metal"))]
    let builder = builder.with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?);
    let builder = if ON_GPU {
        builder
    } else {
        builder.with_force_cpu()
    };
    Ok(builder.build().await?)
}

// one single-view image: Unlimited-OCR's 32 tiles would take the tiled one past the tiny context
fn long_request() -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image(SINGLE_SIZE)],
    ))
    .set_sampler_max_len(UNLIMITED_MAX_LEN)
    .set_sampler_ignore_eos(true)
    .set_sampler_topk(1)
    .return_logprobs(true)
    .set_sampler_topn_logprobs(1)
}

#[tokio::test]
async fn unlimited_ocr_decode_keeps_the_prompt_and_slides_over_its_reply() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    if !ON_GPU {
        make_unlimited(checkpoint.path())?;
        let error = build_bf16(checkpoint.path())
            .await
            .err()
            .expect("the window loaded without PagedAttention");
        assert!(format!("{error:#}").contains("PagedAttention"), "{error:#}");
        return Ok(());
    }
    let full = trace(
        &build_bf16(checkpoint.path())
            .await?
            .send_chat_request(long_request())
            .await?,
    );
    make_unlimited(checkpoint.path())?;
    let windowed = trace(
        &build_bf16(checkpoint.path())
            .await?
            .send_chat_request(long_request())
            .await?,
    );
    assert_eq!(full.len(), UNLIMITED_MAX_LEN);
    assert_eq!(windowed.len(), UNLIMITED_MAX_LEN);
    // decode step t sees all t generated keys while t <= the window, so those steps match full attention
    for (step, ((a, la), (b, lb))) in full
        .iter()
        .zip(&windowed)
        .enumerate()
        .take(UNLIMITED_WINDOW + 1)
    {
        assert_eq!(a, b, "step {step}");
        assert!(
            (la - lb).abs() < WINDOWED_TOLERANCE,
            "step {step}: {la} vs {lb}"
        );
    }
    let parted = full
        .iter()
        .zip(&windowed)
        .skip(UNLIMITED_WINDOW + 1)
        .any(|((a, la), (b, lb))| a != b || (la - lb).abs() > WINDOWED_TOLERANCE);
    assert!(
        parted,
        "the window dropped no reply keys: {full:?} vs {windowed:?}"
    );
    Ok(())
}
