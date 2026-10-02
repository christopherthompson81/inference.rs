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
const MAX_LEN: usize = 6;
// Side lengths that resize to different patch grids, so each image yields its own token count.
const IMAGE_SIDES: [(u32, u32); 2] = [(56, 56), (84, 56)];
const VIDEO_FRAMES: usize = 4;
const VIDEO_FPS: f64 = 2.0;

async fn build(dir: &Path) -> anyhow::Result<Model> {
    Ok(MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?)
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
    RequestBuilder::from(MultimodalMessages::new().add_video_message(
        TextMessageRole::User,
        PROMPT,
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
        (vec![118, 257, 160, 66, 203, 203], 33),
        (vec![118, 223, 66, 203, 28, 223], 37),
        (vec![66, 203, 223, 172, 203, 253], 37),
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
        (vec![257, 205, 64, 111, 64, 111], 33),
        (vec![254, 64, 111, 28, 64, 111], 37),
        (vec![79, 79, 79, 79, 79, 79], 61),
    ];
    assert_eq!(traces, expected);
    Ok(())
}
