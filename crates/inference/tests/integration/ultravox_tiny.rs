//! A tiny random-weight Ultravox: a clip's placeholder run opens the user turn, one token per 8 encoder positions of
//! each 30 s chunk's real frames.

use inference::{
    AudioInput, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole,
};

#[path = "../support/ultravox_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::{tiny_checkpoint, tiny_repo_pair};

const PROMPT: &str = "Hi";
const MAX_LEN: usize = 4;
const SAMPLE_RATE: u32 = 16_000;
// 3 s: 300 mel frames, 150 encoder positions, stacked by 8 and rounded up
const SHORT_SAMPLES: usize = 48_000;
const SHORT_TOKENS: usize = 19;
// 35 s: a whole 3000-frame chunk (188 tokens) and a 500-frame one (32)
const LONG_SAMPLES: usize = 560_000;
const LONG_TOKENS: usize = 220;
const TONE_HZ: f32 = 440.0;
// a reload computes as the load that wrote it, so only rounding parts them
const UQFF_TOLERANCE: f32 = 1e-3;

fn builder(dir: &std::path::Path) -> MultimodalModelBuilder {
    MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

async fn build(dir: &std::path::Path) -> anyhow::Result<Model> {
    Ok(builder(dir).build().await?)
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
    let response = model
        .send_chat_request(request.set_sampler_max_len(MAX_LEN))
        .await?;
    assert!(response.usage.completion_tokens > 0);
    Ok(response.usage.prompt_tokens)
}

#[tokio::test]
async fn each_chunk_of_a_clip_fills_its_stacked_real_frames() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text = prompt_tokens(&model, Vec::new()).await?;
    let short = prompt_tokens(&model, vec![clip(SHORT_SAMPLES)]).await?;
    assert_eq!(short - text, SHORT_TOKENS);
    let long = prompt_tokens(&model, vec![clip(LONG_SAMPLES)]).await?;
    assert_eq!(long - text, LONG_TOKENS);
    Ok(())
}

// the UQFF an ISQ load writes reloads, its residual tensors under the names the model reads, and decodes alike
#[tokio::test]
async fn an_isq_load_writes_a_uqff_that_reloads_and_decodes_alike() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let request = || {
        RequestBuilder::from(MultimodalMessages::new().add_audio_message(
            TextMessageRole::User,
            PROMPT,
            vec![clip(SHORT_SAMPLES)],
        ))
    };
    let (written, read) =
        traces::isq_and_uqff_reload_traces(checkpoint.path(), builder, request, MAX_LEN).await?;
    anyhow::ensure!(
        traces::close(&written, &read, UQFF_TOLERANCE),
        "the UQFF reload moved: {read:?} vs {written:?}"
    );
    Ok(())
}

// a repo naming its text model's repository loads that model beside its own tower and decodes as the whole checkpoint
#[tokio::test]
async fn a_repo_naming_its_text_model_decodes_as_the_whole_checkpoint() -> anyhow::Result<()> {
    let request = || {
        RequestBuilder::from(MultimodalMessages::new().add_audio_message(
            TextMessageRole::User,
            PROMPT,
            vec![clip(SHORT_SAMPLES)],
        ))
    };
    let whole = tiny_checkpoint()?;
    let (expected, _) = traces::greedy(&build(whole.path()).await?, request(), MAX_LEN).await?;
    let (repo, _text) = tiny_repo_pair()?;
    let (actual, _) = traces::greedy(&build(repo.path()).await?, request(), MAX_LEN).await?;
    assert_eq!(actual, expected);
    Ok(())
}
