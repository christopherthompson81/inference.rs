//! A tiny random-weight MERaLiON-2: a turn with audio takes the model card's instruction template, every 30 s chunk of
//! a clip 100 placeholders, and transcription goes through the same template.

use inference::{
    AudioInput, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole, TranscriptionRequest, TranscriptionResponseFormat,
};

use crate::audio_clips::wav;

#[path = "../support/meralion2_tiny.rs"]
mod support;
use support::tiny_checkpoint;

const PROMPT: &str = "Hi";
const MAX_LEN: usize = 4;
const SAMPLE_RATE: u32 = 16_000;
// 3 s is one chunk; 35 s is a whole chunk and a 5 s one
const SHORT_SAMPLES: usize = 48_000;
const LONG_SAMPLES: usize = 560_000;
const CHUNK_TOKENS: usize = 100;
// `Instruction: ` before the text and ` \nFollow the text instruction based on the following audio: ` after it: the
// tiny vocabulary spells them a byte at a time, a space as the 3-byte `▁`
const INSTRUCTION_TOKENS: usize = 95;
const TONE_HZ: f32 = 440.0;
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
async fn an_audio_turn_takes_the_instruction_template_and_100_tokens_a_chunk() -> anyhow::Result<()>
{
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text = prompt_tokens(&model, Vec::new()).await?;
    let short = prompt_tokens(&model, vec![clip(SHORT_SAMPLES)]).await?;
    assert_eq!(short - text, INSTRUCTION_TOKENS + CHUNK_TOKENS);
    let long = prompt_tokens(&model, vec![clip(LONG_SAMPLES)]).await?;
    assert_eq!(long - short, CHUNK_TOKENS);
    Ok(())
}

#[tokio::test]
async fn a_transcription_request_decodes_the_clip() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let request = TranscriptionRequest {
        model: "default".to_string(),
        language: None,
        prompt: None,
        response_format: TranscriptionResponseFormat::VerboseJson,
        temperature: None,
        timestamp_granularities: Vec::new(),
        beam_size: None,
        diarization_model: None,
        task: Default::default(),
    };
    let output = model
        .transcription(request, &wav(&clip(SHORT_SAMPLES)))
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let body: serde_json::Value = serde_json::from_str(&output.body)?;
    let duration = body["duration"].as_f64().unwrap_or_default();
    assert!((duration - SHORT_SAMPLES as f64 / f64::from(SAMPLE_RATE)).abs() < DURATION_TOLERANCE);
    Ok(())
}
