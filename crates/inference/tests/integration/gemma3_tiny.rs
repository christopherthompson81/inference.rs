//! Gemma 3's text and image paths on a tiny random-weight checkpoint: its image tokens attend to each other both
//! ways, its sliding and global layers take different RoPE, and GPU builds take the paged media-prefix path.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};
use inference::{
    IsqType, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole,
};

#[path = "../support/decode_graphs.rs"]
mod decode_graphs;
#[path = "../support/gemma3_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::tiny_gemma3;
use traces::{close, uqff_files};

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
// Part of the engine's refusal to requantize a model with no tracked layers.
const NOT_AN_ISQ_LOAD: &str = "loaded with ISQ";
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

// A UQFF-writing load keeps the whole model on the host, so a GPU build cannot decode with it.
fn cpu_builder(dir: &Path) -> MultimodalModelBuilder {
    MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
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

async fn trace(model: &Model, request: RequestBuilder) -> anyhow::Result<(traces::Trace, usize)> {
    traces::greedy(model, request, MAX_LEN).await
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
        close(&b, &fresh_b, LOGPROB_TOLERANCE),
        "image b was served image a's blocks: {b:?}"
    );
    let (a, cached) = trace(&warm, image_request(LONG_PROMPT, 1)).await?;
    // Only the paged prefix cache serves Gemma 3 image prompts; the sequence cacher has no media features to match
    anyhow::ensure!(
        (cached > 0) == ON_GPU,
        "image a hit {cached} cached tokens, expected a hit only with paged attention"
    );
    anyhow::ensure!(
        close(&a, &fresh_a, LOGPROB_TOLERANCE),
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
        close(&batched, &alone, LOGPROB_TOLERANCE),
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
    anyhow::ensure!(
        close(&image, expected_image, PIN_TOLERANCE),
        "image decode moved: {image:?}"
    );
    anyhow::ensure!(
        close(&text, expected_text, PIN_TOLERANCE),
        "text decode moved: {text:?}"
    );
    Ok(())
}

// The second request for the same image takes its encoder output from the cache and decodes the same.
#[tokio::test]
async fn gemma3_repeated_image_hits_the_encoder_cache() -> anyhow::Result<()> {
    let checkpoint = tiny_gemma3()?;
    // with the prefix cache on, the repeat would reuse KV blocks and never reach the encoder
    let model = builder(checkpoint.path())
        .with_prefix_cache_n(None)
        .build()
        .await?;
    let encoder_cache = |model: &Model| -> anyhow::Result<(usize, usize)> {
        let stats = model.cache_stats()?;
        let cache = stats.data[0]
            .encoder_cache
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no encoder cache"))?;
        Ok((cache.hits, cache.misses))
    };
    let (first, _) = trace(&model, image_request(PROMPT, 9)).await?;
    let (hits, misses) = encoder_cache(&model)?;
    anyhow::ensure!(
        hits == 0 && misses > 0,
        "first image: {hits} hits, {misses} misses"
    );
    let (second, _) = trace(&model, image_request(PROMPT, 9)).await?;
    let (repeat_hits, repeat_misses) = encoder_cache(&model)?;
    anyhow::ensure!(
        repeat_hits > 0 && repeat_misses == misses,
        "repeated image: {repeat_hits} hits, {repeat_misses} misses"
    );
    anyhow::ensure!(
        close(&first, &second, LOGPROB_TOLERANCE),
        "a cached encoder output changed decoding: {first:?} vs {second:?}"
    );
    Ok(())
}

// An in-situ quantized load writes UQFF; reloading it decodes the same image the same way.
#[tokio::test]
async fn gemma3_isq_load_and_a_reload_of_its_uqff_decode_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_gemma3()?;
    let uqff_dir = tempfile::tempdir()?;
    let quantized = cpu_builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .write_uqff(uqff_dir.path().join("model.uqff"))
        .build()
        .await?;
    let (expected, _) = trace(&quantized, image_request(PROMPT, 4)).await?;
    drop(quantized);

    let written = uqff_files(uqff_dir.path())?;
    anyhow::ensure!(!written.is_empty(), "no UQFF written");
    let reloaded = cpu_builder(checkpoint.path())
        .from_uqff(vec![written[0].clone()])
        .build()
        .await?;
    let (actual, _) = trace(&reloaded, image_request(PROMPT, 4)).await?;
    anyhow::ensure!(
        close(&expected, &actual, LOGPROB_TOLERANCE),
        "the UQFF reload moved: {actual:?} vs {expected:?}"
    );
    Ok(())
}

// Requantizing an ISQ load in place works from the loaded weights, so Q8_0 then Q4_0 is pinned, not a Q4_0 load.
#[cfg(not(any(feature = "cuda", feature = "metal")))]
#[tokio::test]
async fn gemma3_re_isq_decodes_as_recorded() -> anyhow::Result<()> {
    let checkpoint = tiny_gemma3()?;
    let model = builder(checkpoint.path())
        .with_isq(IsqType::Q8_0)
        .build()
        .await?;
    let (before, _) = trace(&model, image_request(PROMPT, 6)).await?;
    model.re_isq_model(IsqType::Q4_0).await?;
    let (after, _) = trace(&model, image_request(PROMPT, 6)).await?;
    let expected_before: &[(u32, f32)] = &[
        (11, -3.6783395),
        (11, -3.6765847),
        (11, -3.6880455),
        (11, -3.7000399),
        (11, -3.69746),
        (11, -3.693663),
    ];
    let expected_after: &[(u32, f32)] = &[
        (11, -3.681465),
        (11, -3.680488),
        (11, -3.6885803),
        (11, -3.7020075),
        (11, -3.6990685),
        (11, -3.6988313),
    ];
    anyhow::ensure!(
        close(&before, expected_before, PIN_TOLERANCE),
        "Q8_0 decode moved: {before:?}"
    );
    anyhow::ensure!(
        close(&after, expected_after, PIN_TOLERANCE),
        "requantized decode moved: {after:?}"
    );
    Ok(())
}

// A model loaded without ISQ tracks no layers to requantize, and the call says so.
#[tokio::test]
async fn gemma3_re_isq_without_an_isq_load_fails() -> anyhow::Result<()> {
    let checkpoint = tiny_gemma3()?;
    let model = builder(checkpoint.path()).build().await?;
    let error = model.re_isq_model(IsqType::Q8_0).await.err();
    anyhow::ensure!(
        error
            .as_ref()
            .is_some_and(|e| e.to_string().contains(NOT_AN_ISQ_LOAD)),
        "re-ISQ of an unquantized load: {error:?}"
    );
    Ok(())
}

// Random weights settle on one token, so the pins mostly guard the graph path running at all.
const GRAPH_IMAGE_TRACE: &[(u32, f32)] = &[
    (11, -3.6849756),
    (11, -3.6883585),
    (11, -3.7020524),
    (11, -3.707307),
    (11, -3.7069917),
    (11, -3.7139778),
];
const GRAPH_TRACES: [&[(u32, f32)]; decode_graphs::GRAPH_PROMPTS.len()] = [
    &[
        (11, -3.6839383),
        (11, -3.6869273),
        (11, -3.6875715),
        (11, -3.6954334),
        (11, -3.6972377),
        (11, -3.700204),
    ],
    &[
        (11, -3.68789),
        (11, -3.687997),
        (11, -3.6993887),
        (11, -3.7050138),
        (11, -3.7048666),
        (11, -3.7056303),
    ],
    &[
        (11, -3.6786876),
        (11, -3.678305),
        (11, -3.6871538),
        (11, -3.6996465),
        (11, -3.6990786),
        (11, -3.7034485),
    ],
    &[
        (11, -3.6857686),
        (11, -3.6900928),
        (11, -3.7089043),
        (11, -3.722483),
        (11, -3.722449),
        (11, -3.7166421),
    ],
    &[
        (11, -3.6844528),
        (11, -3.687066),
        (11, -3.691625),
        (11, -3.7050107),
        (11, -3.70397),
        (11, -3.6996493),
    ],
];

// An image request, then rounds of concurrent text requests, on a bf16 paged build (graphs need both).
async fn image_then_text_rounds(
    snapshotter: &metrics_util::debugging::Snapshotter,
) -> anyhow::Result<(decode_graphs::Counters, decode_graphs::Counters)> {
    let checkpoint = tiny_gemma3()?;
    let model = MultimodalModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::BF16)
        .with_paged_attn(inference::PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;
    let (image, _) = trace(&model, image_request(LONG_PROMPT, 3)).await?;
    let image_counters = decode_graphs::Counters::take(snapshotter);
    decode_graphs::assert_trace(&image, GRAPH_IMAGE_TRACE, &image);
    let rounds = decode_graphs::rounds(&model, MAX_LEN).await?;
    decode_graphs::assert_traces(&rounds, &GRAPH_TRACES);
    Ok((image_counters, decode_graphs::Counters::take(snapshotter)))
}

// The multimodal pipeline's decode steps replay graphs for an image request and for the text requests after it.
#[tokio::test]
async fn gemma3_decode_after_an_image_replays_cuda_graphs() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    let snapshotter = decode_graphs::recorder();
    let (image, text) = image_then_text_rounds(&snapshotter).await?;
    decode_graphs::assert_replayed(&image);
    let replays = text.total(
        decode_graphs::EVENTS,
        &[("event", "replay"), ("outcome", "success")],
    );
    assert!(
        replays > 0,
        "text decode after the image never replayed: {text:?}"
    );
    decode_graphs::assert_no_fallback(&text);
    Ok(())
}

#[tokio::test]
async fn gemma3_decode_without_cuda_graphs_matches() -> anyhow::Result<()> {
    if !cfg!(feature = "cuda") {
        return Ok(());
    }
    decode_graphs::disable_graphs();
    let snapshotter = decode_graphs::recorder();
    let (image, text) = image_then_text_rounds(&snapshotter).await?;
    decode_graphs::assert_disabled(&image);
    decode_graphs::assert_disabled(&text);
    Ok(())
}
