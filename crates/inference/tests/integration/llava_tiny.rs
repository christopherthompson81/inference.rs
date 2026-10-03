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
// Kernel reassociation that moves F32 logprobs by ~1e-6 grows to ~1e-3 through these large-weight layers.
const LOGPROB_TOLERANCE: f32 = 1e-4;
// The image prompt shares the text prompt's opening, so a warm model serves it partly from the prefix cache.
const MIN_CACHED_PREFIX: usize = 1;

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

// Greedy (token, logprob) per step, the prompt length, and the prompt tokens served from the prefix cache.
async fn trace(
    model: &Model,
    request: RequestBuilder,
) -> anyhow::Result<(Vec<(u32, f32)>, usize, usize)> {
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
    Ok((steps, response.usage.prompt_tokens, cached))
}

async fn traces(model: &Model, side: (u32, u32)) -> anyhow::Result<Vec<(Vec<(u32, f32)>, usize)>> {
    let text =
        RequestBuilder::from(MultimodalMessages::new().add_message(TextMessageRole::User, PROMPT));
    let with_image = RequestBuilder::from(MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        PROMPT,
        vec![image(side.0, side.1, 90)],
    ));
    let (text_steps, text_prompt, _) = trace(model, text).await?;
    // After a prefix hit the image must reach the model once, on the prompt step, and never again on decode.
    let (image_steps, image_prompt, cached) = trace(model, with_image).await?;
    anyhow::ensure!(
        cached >= MIN_CACHED_PREFIX,
        "the image prompt was not served from the prefix cache"
    );
    Ok(vec![(text_steps, text_prompt), (image_steps, image_prompt)])
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
    // 4 vision tokens: a 28-pixel image at patch 14; the image prompt reuses the text prompt's cached prefix
    let expected = [
        (
            vec![
                (149, -2.3598313),
                (189, -2.1979446),
                (99, -2.8372998),
                (226, -2.4462297),
                (99, -2.5567582),
                (175, -2.5719965),
            ],
            26,
        ),
        (
            vec![
                (33, -1.8118622),
                (93, -2.884374),
                (150, -1.8899802),
                (33, -1.039323),
                (93, -1.7181377),
                (2, -2.1036007),
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
                (149, -2.3598313),
                (189, -2.1979446),
                (99, -2.8372998),
                (226, -2.4462297),
                (99, -2.5567582),
                (175, -2.5719965),
            ],
            26,
        ),
        (
            vec![
                (141, -2.5872004),
                (98, -3.0652235),
                (136, -2.8261824),
                (192, -1.8183632),
                (3, -2.502676),
                (55, -2.5459356),
            ],
            40,
        ),
    ];
    check(&traces, &expected);
    Ok(())
}
