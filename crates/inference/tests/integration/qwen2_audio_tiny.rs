//! A tiny random-weight Qwen2-Audio: each clip takes its own numbered line before the text, its tokens counted from its
//! real frames, and a clip past the 30 s window is cut to it.

use inference::{
    AudioInput, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole,
};

#[path = "../support/qwen2_audio_tiny.rs"]
mod support;
use support::tiny_checkpoint;

const PROMPT: &str = "Hi";
const MAX_LEN: usize = 4;
const SAMPLE_RATE: u32 = 16_000;
// 3 s: 300 mel frames, 150 encoder positions, pooled by two
const SHORT_SAMPLES: usize = 48_000;
const SHORT_TOKENS: usize = 75;
// 40 s is cut to the 30 s window: 1500 positions, 750 tokens
const LONG_SAMPLES: usize = 640_000;
const LONG_TOKENS: usize = 750;
// `Audio 1: <|audio_bos|>` and `<|audio_eos|>\n`: the tiny vocabulary spells the text a byte at a time (a space is
// the 3-byte `▁`), the markers are one token each
const LINE_TOKENS: usize = 16;
const TONE_HZ: f32 = 440.0;

async fn build(dir: &std::path::Path) -> anyhow::Result<Model> {
    Ok(MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?)
}

fn clip(samples: usize) -> AudioInput {
    AudioInput {
        samples: (0..samples)
            .map(|i| {
                0.3 * (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect(),
        sample_rate: SAMPLE_RATE,
        channels: 1,
    }
}

async fn prompt_tokens(model: &Model, clips: Vec<AudioInput>) -> anyhow::Result<usize> {
    let request = if clips.is_empty() {
        RequestBuilder::new().add_message(TextMessageRole::User, PROMPT)
    } else {
        RequestBuilder::from(MultimodalMessages::new().add_audio_message(
            TextMessageRole::User,
            PROMPT,
            clips,
        ))
    };
    Ok(model
        .send_chat_request(request.set_sampler_max_len(MAX_LEN))
        .await?
        .usage
        .prompt_tokens)
}

#[tokio::test]
async fn a_clip_fills_its_real_frames_and_a_long_one_is_cut_to_the_window() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text = prompt_tokens(&model, Vec::new()).await?;
    let short = prompt_tokens(&model, vec![clip(SHORT_SAMPLES)]).await?;
    assert_eq!(short - text, SHORT_TOKENS + LINE_TOKENS);
    let long = prompt_tokens(&model, vec![clip(LONG_SAMPLES)]).await?;
    assert_eq!(long - text, LONG_TOKENS + LINE_TOKENS);
    Ok(())
}
