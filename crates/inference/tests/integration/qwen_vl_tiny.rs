//! The image and video input paths of Qwen2-VL and Qwen3-VL on tiny random-weight checkpoints built at test time.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};
use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
    VideoInput,
};

#[path = "../support/qwen_vl_tiny.rs"]
mod support;
use support::{
    tiny_qwen2_vl, tiny_qwen3_5_moe, tiny_qwen3_5_moe_mtp, tiny_qwen3_5_mtp, tiny_qwen3_vl,
};

const PROMPT: &str = "describe";
// Long enough that a shared prefix runs past the media into whole paged blocks; paged hits never end inside media.
const LONG_PROMPT: &str = "describe every part of this picture in order, from the top left corner to the bottom right one.";
const MAX_LEN: usize = 6;
// Side lengths that resize to different patch grids, so each image yields its own token count.
const IMAGE_SIDES: [(u32, u32); 2] = [(56, 56), (84, 56)];
const VIDEO_FRAMES: usize = 4;
const VIDEO_FPS: f64 = 2.0;

const ON_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));
const PREFIX_CACHE_SEQS: usize = 16;
// Cached or chunked KV comes from a different prefill than a full recompute, so logprobs match only to rounding.
const LOGPROB_TOLERANCE: f32 = 1e-3;
// GPU and CPU BF16 runs round differently at each layer; on the tiny Qwen3.5-MoE they agree on every id within
// 0.17, where BF16 against F32 on CPU already moves ids and logprobs by over 1.
const BF16_LOGPROB_TOLERANCE: f32 = 0.25;
// Long enough for several draft-and-verify rounds.
const MTP_MAX_LEN: usize = 16;
const MTP_N_PREDICT: usize = 2;
// The verify and decode kernels move BF16 logprobs by up to 0.035 on the tiny Qwen3.5-MoE; a closer top two can swap.
const MTP_TIE_MARGIN: f32 = 0.1;
// So a run that parts at once fails instead of comparing nothing.
const MTP_MIN_AGREED: usize = 4;
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
    // The SDK turns the prefix cache on by default, so the reference model has it off.
    let fresh = builder(dir).with_prefix_cache_n(None).build().await?;
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

// With the prefix cache on, the batched image request would reuse the first run's media blocks instead of prefilling.
async fn text_in_the_batch_leaves_media_unchanged(dir: &Path) -> anyhow::Result<()> {
    let model = builder(dir).with_prefix_cache_n(None).build().await?;
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
async fn qwen2_vl_text_in_the_batch_leaves_media_unchanged() -> anyhow::Result<()> {
    text_in_the_batch_leaves_media_unchanged(tiny_qwen2_vl()?.path()).await
}

#[tokio::test]
async fn qwen3_vl_text_in_the_batch_leaves_media_unchanged() -> anyhow::Result<()> {
    text_in_the_batch_leaves_media_unchanged(tiny_qwen3_vl()?.path()).await
}

// Qwen3.5-MoE runs the Qwen3.5 text model with sparse feed-forwards. On CUDA its GDN layers need BF16 (the causal
// conv kernel takes F16/BF16), so a GPU build checks its paged, deferred-state decode against the same checkpoint on CPU.
async fn qwen3_5_moe_traces(model: &Model) -> anyhow::Result<[Vec<(u32, f32)>; 2]> {
    let text = RequestBuilder::new().add_message(TextMessageRole::User, PROMPT);
    // random weights favour one continuation for any prompt, so the logprobs carry what the prompt changes
    let (text, _) = trace(model, text).await?;
    let (image, _) = trace(model, images(&IMAGE_SIDES[..1])).await?;
    Ok([text, image])
}

#[tokio::test]
async fn qwen3_5_moe_text_and_image() -> anyhow::Result<()> {
    let checkpoint = tiny_qwen3_5_moe()?;
    if ON_GPU {
        let gpu = builder(checkpoint.path())
            .with_dtype(ModelDType::BF16)
            .build()
            .await?;
        let cpu = MultimodalModelBuilder::new(checkpoint.path().to_string_lossy())
            .with_dtype(ModelDType::BF16)
            .with_force_cpu()
            .build()
            .await?;
        let (gpu, cpu) = (
            qwen3_5_moe_traces(&gpu).await?,
            qwen3_5_moe_traces(&cpu).await?,
        );
        for (gpu, cpu) in gpu.iter().zip(&cpu) {
            let close = gpu.len() == cpu.len()
                && gpu
                    .iter()
                    .zip(cpu)
                    .all(|(g, c)| g.0 == c.0 && (g.1 - c.1).abs() < BF16_LOGPROB_TOLERANCE);
            anyhow::ensure!(close, "GPU decode {gpu:?} differs from CPU {cpu:?}");
        }
        return Ok(());
    }
    let model = build(checkpoint.path()).await?;
    let [text, image] = qwen3_5_moe_traces(&model).await?;
    let expected_text = [
        (260, -0.91792876),
        (174, -0.0019233831),
        (4, -0.18916462),
        (49, -0.6902862),
        (79, -0.08116475),
        (244, -0.87973154),
    ];
    let expected_image = [
        (260, -0.91860574),
        (174, -0.0017912925),
        (4, -0.16398197),
        (49, -0.80053025),
        (79, -0.063517146),
        (244, -0.8995498),
    ];
    anyhow::ensure!(
        same_decode(&text, &expected_text),
        "text decode moved: {text:?}"
    );
    anyhow::ensure!(
        same_decode(&image, &expected_image),
        "image decode moved: {image:?}"
    );
    Ok(())
}

// Steps, each the greedy id and logprob with the runner-up's logprob.
async fn mtp_trace(model: &Model, request: RequestBuilder) -> anyhow::Result<Vec<(u32, f32, f32)>> {
    let request = request
        .set_sampler_max_len(MTP_MAX_LEN)
        .set_sampler_topk(1)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(2);
    let response = model.send_chat_request(request).await?;
    let steps = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| {
                    let runner_up = t
                        .top_logprobs
                        .get(1)
                        .map_or(f32::NEG_INFINITY, |r| r.logprob);
                    (
                        t.top_logprobs[0].token,
                        t.top_logprobs[0].logprob,
                        runner_up,
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    anyhow::ensure!(!steps.is_empty(), "the model generated nothing");
    Ok(steps)
}

// Greedy verification keeps the target's own tokens, so drafting with the MTP head may only swap a token where the
// target's top two are within rounding (verification runs the multi-token kernels), and the runs part there.
// The head reads and writes paged KV, so only GPU builds run it.
async fn builtin_mtp_keeps_greedy_output(checkpoint: tempfile::TempDir) -> anyhow::Result<()> {
    let plain = builder(checkpoint.path())
        .with_dtype(ModelDType::BF16)
        .build()
        .await?;
    let mtp = builder(checkpoint.path())
        .with_dtype(ModelDType::BF16)
        .with_builtin_mtp(Some(MTP_N_PREDICT))
        .build()
        .await?;
    let requests = || {
        [
            RequestBuilder::new().add_message(TextMessageRole::User, PROMPT),
            images(&IMAGE_SIDES[..1]),
        ]
    };
    for (plain_request, mtp_request) in requests().into_iter().zip(requests()) {
        let expected = mtp_trace(&plain, plain_request).await?;
        let drafted = mtp_trace(&mtp, mtp_request).await?;
        let agreed = expected
            .iter()
            .zip(&drafted)
            .take_while(|(e, d)| e.0 == d.0)
            .count();
        let close = expected
            .iter()
            .zip(&drafted)
            .take(agreed)
            .all(|(e, d)| (e.1 - d.1).abs() < BF16_LOGPROB_TOLERANCE);
        // a swap is only allowed where the target's own top two were within rounding
        let swap_is_a_tie = match expected.get(agreed) {
            Some((_, top, runner_up)) => agreed < drafted.len() && top - runner_up < MTP_TIE_MARGIN,
            None => drafted.len() == expected.len(),
        };
        anyhow::ensure!(
            close && swap_is_a_tie && agreed >= MTP_MIN_AGREED.min(expected.len()),
            "MTP drafting changed the greedy output at step {agreed}: {drafted:?} vs {expected:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn qwen3_5_builtin_mtp_keeps_greedy_output() -> anyhow::Result<()> {
    if !ON_GPU {
        return Ok(());
    }
    builtin_mtp_keeps_greedy_output(tiny_qwen3_5_mtp()?).await
}

#[tokio::test]
async fn qwen3_5_moe_builtin_mtp_keeps_greedy_output() -> anyhow::Result<()> {
    if !ON_GPU {
        return Ok(());
    }
    builtin_mtp_keeps_greedy_output(tiny_qwen3_5_moe_mtp()?).await
}
