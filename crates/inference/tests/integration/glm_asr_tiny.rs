//! A tiny random-weight GLM-ASR: the repo's template writes each audio part, a clip takes the tokens of its 30 s
//! windows, and a transcription request decodes from the processor's prompt.

use inference::{
    AudioInput, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole, TranscriptionRequest, TranscriptionResponseFormat,
};

use crate::audio_clips::wav;

#[path = "../support/glm_asr_tiny.rs"]
mod support;
use support::tiny_checkpoint;

const PROMPT: &str = "Hi";
const MAX_LEN: usize = 4;
const SAMPLE_RATE: u32 = 16_000;
// 3 s: 300 frames, 150 encoder positions, merged by four
const SHORT_SAMPLES: usize = 48_000;
const SHORT_TOKENS: usize = 37;
// 40 s: a whole window (375) and 10 s (1000 frames, 500 positions, 125)
const LONG_SAMPLES: usize = 640_000;
const LONG_TOKENS: usize = 500;
// `<|begin_of_audio|>`, `<|end_of_audio|>`, `<|user|>` and the newline the template writes around each clip
const FRAME_TOKENS: usize = 4;
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

fn transcription() -> TranscriptionRequest {
    TranscriptionRequest {
        model: "default".to_string(),
        language: None,
        prompt: None,
        response_format: TranscriptionResponseFormat::VerboseJson,
        temperature: None,
        timestamp_granularities: Vec::new(),
        beam_size: None,
        diarization_model: None,
        task: Default::default(),
    }
}

#[tokio::test]
async fn a_clip_takes_its_windows_tokens_and_transcribes() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text = prompt_tokens(&model, Vec::new()).await?;
    let short = prompt_tokens(&model, vec![clip(SHORT_SAMPLES)]).await?;
    assert_eq!(short - text, SHORT_TOKENS + FRAME_TOKENS);
    let long = prompt_tokens(&model, vec![clip(LONG_SAMPLES)]).await?;
    assert_eq!(long - text, LONG_TOKENS + FRAME_TOKENS);
    let output = model
        .transcription(transcription(), &wav(&clip(SHORT_SAMPLES)))
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let body: serde_json::Value = serde_json::from_str(&output.body)?;
    assert!(body["text"].is_string());
    let mut translate = transcription();
    translate.task = inference::TranscriptionTask::Translate;
    assert!(
        model
            .transcription(translate, &wav(&clip(SHORT_SAMPLES)))
            .await
            .is_err()
    );
    Ok(())
}
