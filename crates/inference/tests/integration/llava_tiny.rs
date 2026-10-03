//! The text and image paths of LLaVA 1.5 (Llama text) and LLaVA-NeXT (Mistral text) on tiny random-weight checkpoints.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};
use inference::{
    Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};

#[path = "../support/llava_tiny.rs"]
mod support;
use support::{tiny_llava_next, tiny_llava15};

const PROMPT: &str = "describe";
const MAX_LEN: usize = 6;
// The fixtures' CLIP side; LLaVA-NeXT's anyres grid also takes the 2:1 image as two tiles.
const IMAGE_SIDE: u32 = 28;
const LOGPROB_TOLERANCE: f32 = 1e-4;

fn builder(dir: &Path) -> MultimodalModelBuilder {
    MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

fn image(width: u32, height: u32, seed: u8) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
        Rgb([
            (x * 255 / width) as u8 ^ seed,
            (y * 255 / height) as u8,
            seed.wrapping_mul(37),
        ])
    }))
}

// Greedy (token, logprob) per step and the prompt length.
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
    Ok((steps, response.usage.prompt_tokens))
}

async fn traces(model: &Model, side: (u32, u32)) -> anyhow::Result<Vec<(Vec<(u32, f32)>, usize)>> {
    let text =
        RequestBuilder::from(MultimodalMessages::new().add_message(TextMessageRole::User, PROMPT));
    let with_image = RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image(side.0, side.1, 90)],
    ));
    Ok(vec![
        trace(model, text).await?,
        trace(model, with_image).await?,
    ])
}

fn check(actual: &[(Vec<(u32, f32)>, usize)], expected: &[(Vec<(u32, f32)>, usize)]) {
    assert_eq!(actual.len(), expected.len());
    for ((steps, prompt), (want_steps, want_prompt)) in actual.iter().zip(expected) {
        assert_eq!(prompt, want_prompt, "prompt tokens: {actual:?}");
        let ids = steps.iter().map(|s| s.0).collect::<Vec<_>>();
        let want_ids = want_steps.iter().map(|s| s.0).collect::<Vec<_>>();
        assert_eq!(ids, want_ids, "greedy ids: {actual:?}");
        for ((_, got), (_, want)) in steps.iter().zip(want_steps) {
            assert!(
                (got - want).abs() <= LOGPROB_TOLERANCE,
                "logprobs: {actual:?}"
            );
        }
    }
}

#[tokio::test]
async fn llava15_text_and_image() -> anyhow::Result<()> {
    let checkpoint = tiny_llava15()?;
    let model = builder(checkpoint.path()).build().await?;
    let traces = traces(&model, (IMAGE_SIDE, IMAGE_SIDE)).await?;
    // 4 vision tokens: a 28-pixel image at patch 14
    let expected = [
        (
            vec![
                (149, -2.3602328),
                (189, -2.1962988),
                (99, -2.836937),
                (226, -2.4474363),
                (99, -2.557251),
                (175, -2.5741384),
            ],
            26,
        ),
        (
            vec![
                (33, -1.8135748),
                (93, -2.881034),
                (150, -1.8875551),
                (33, -1.0386267),
                (93, -1.7178918),
                (2, -2.1032774),
            ],
            30,
        ),
    ];
    check(&traces, &expected);
    Ok(())
}

#[tokio::test]
async fn llava_next_text_and_image() -> anyhow::Result<()> {
    let checkpoint = tiny_llava_next()?;
    let model = builder(checkpoint.path()).build().await?;
    let traces = traces(&model, (2 * IMAGE_SIDE, IMAGE_SIDE)).await?;
    // the text tower draws the same name-seeded weights as LLaVA 1.5's, and Mistral without a window is Llama
    let expected = [
        (
            vec![
                (149, -2.3602328),
                (189, -2.1962988),
                (99, -2.836937),
                (226, -2.4474363),
                (99, -2.557251),
                (175, -2.5741384),
            ],
            26,
        ),
        (
            vec![
                (141, -2.585176),
                (98, -3.0661695),
                (136, -2.8283815),
                (192, -1.8187708),
                (3, -2.5006533),
                (55, -2.5450103),
            ],
            40,
        ),
    ];
    check(&traces, &expected);
    Ok(())
}
