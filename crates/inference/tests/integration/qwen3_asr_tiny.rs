//! A tiny random-weight Qwen3-ASR: a clip expands to its encoder tokens between its start and end markers, the
//! encoder runs a full attention window and a shorter one, and a repeat served from the prefix cache decodes the same.

use inference::{
    AudioInput, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole,
};

#[path = "../support/qwen3_asr_tiny.rs"]
mod support;
use support::tiny_checkpoint;

const PROMPT: &str = "Transcribe it.";
const MAX_LEN: usize = 6;
const SAMPLE_RATE: u32 = 16_000;
// 3 s: 300 mel frames, 3 chunks of 13 tokens; the fixture's 2-chunk window splits them 26 + 13
const SAMPLES: usize = 48_000;
const AUDIO_TOKENS: usize = 39;
// the clip's start and end markers
const MARKERS: usize = 2;
const TONE_HZ: f32 = 440.0;
// cached KV comes from a different prefill than a full recompute, so logprobs match only to rounding
const LOGPROB_TOLERANCE: f32 = 1e-3;

async fn build(dir: &std::path::Path) -> anyhow::Result<Model> {
    Ok(MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_prefix_cache_n(Some(16))
        .build()
        .await?)
}

fn clip() -> AudioInput {
    AudioInput {
        samples: (0..SAMPLES)
            .map(|i| {
                0.3 * (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect(),
        sample_rate: SAMPLE_RATE,
        channels: 1,
    }
}

fn request() -> RequestBuilder {
    RequestBuilder::from(MultimodalMessages::new().add_audio_message(
        TextMessageRole::User,
        PROMPT,
        vec![clip()],
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

#[tokio::test]
async fn a_clip_expands_to_its_tokens_and_a_cached_repeat_decodes_the_same() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text_only = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    let first = model.send_chat_request(request()).await?;
    assert_eq!(
        first.usage.prompt_tokens - text_only.usage.prompt_tokens,
        AUDIO_TOKENS + MARKERS
    );
    let first = trace(&first);
    assert!(!first.is_empty(), "the audio request generated nothing");
    let second = trace(&model.send_chat_request(request()).await?);
    assert_eq!(first.len(), second.len());
    for ((a, la), (b, lb)) in first.iter().zip(&second) {
        assert_eq!(a, b);
        assert!((la - lb).abs() < LOGPROB_TOLERANCE, "{la} vs {lb}");
    }
    Ok(())
}
