//! A tiny random-weight VibeVoice-ASR: a clip fills one placeholder per hop and its template's duration, the two
//! encoders' chunks join as one pass, and a transcription request (its own diarizer included) answers.

use inference::{
    AudioInput, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole, TranscriptionRequest, TranscriptionResponseFormat,
    VerboseTranscriptionResponse,
};

use crate::audio_clips::wav;

#[path = "../support/vibevoice_asr_tiny.rs"]
mod support;
use support::{HOP, tiny_checkpoint};

const MAX_LEN: usize = 4;
const SAMPLE_RATE: u32 = 24_000;
// past the fixture's 2400-sample encoder chunk, so the clip's encoders carry their context across
const SHORT_SAMPLES: usize = 6_000;
// a hop and a half longer: two more placeholders (one partial), the same duration digits
const LONGER_SAMPLES: usize = SHORT_SAMPLES + HOP + HOP / 2;
const TONE_HZ: f32 = 330.0;
const DURATION_TOLERANCE: f64 = 1e-6;

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

async fn prompt_tokens(model: &Model, samples: usize) -> anyhow::Result<usize> {
    let request = RequestBuilder::from(MultimodalMessages::new().add_audio_message(
        TextMessageRole::User,
        "",
        vec![clip(samples)],
    ));
    Ok(model
        .send_chat_request(request.set_sampler_max_len(MAX_LEN))
        .await?
        .usage
        .prompt_tokens)
}

// the template writes the clip's placeholder; it fills one per hop, rounded up
#[tokio::test]
async fn a_clip_fills_one_placeholder_per_hop() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let short = prompt_tokens(&model, SHORT_SAMPLES).await?;
    let longer = prompt_tokens(&model, LONGER_SAMPLES).await?;
    assert_eq!(
        longer - short,
        LONGER_SAMPLES.div_ceil(HOP) - SHORT_SAMPLES.div_ceil(HOP)
    );
    Ok(())
}

// a transcription answers over the clip (random weights write no JSON, so its text stands as one segment)
#[tokio::test]
async fn a_transcription_request_answers() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let audio = wav(&clip(SHORT_SAMPLES));
    let request = TranscriptionRequest::new(TranscriptionResponseFormat::VerboseJson);
    let out = model.transcription(request, &audio).await?;
    let parsed: VerboseTranscriptionResponse = serde_json::from_str(&out.body)?;
    let seconds = SHORT_SAMPLES as f64 / f64::from(SAMPLE_RATE);
    assert!(
        (parsed.duration - seconds).abs() < DURATION_TOLERANCE,
        "{}",
        parsed.duration
    );
    assert!(parsed.segments.len() <= 1, "{:?}", parsed.segments);
    Ok(())
}
