//! A tiny random-weight Voxtral: with no template in the checkpoint the loader's Mistral layout applies, and a clip
//! opens the user turn as `[BEGIN_AUDIO]` and one 30 s chunk's run of `[AUDIO]` tokens.

use inference::{
    AudioInput, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole, TranscriptionRequest, TranscriptionResponseFormat,
};

use crate::audio_clips::wav;

#[path = "../support/voxtral_chat_tiny.rs"]
mod support;
#[path = "../support/traces.rs"]
mod traces;
use support::tiny_checkpoint;

const PROMPT: &str = "Hi";
const SYSTEM: &str = "Brief.";
const MAX_LEN: usize = 4;
const SAMPLE_RATE: u32 = 16_000;
// 3 s pads to one 30 s chunk: 1500 encoder positions over the 4-position stacking
const SAMPLES: usize = 48_000;
const AUDIO_TOKENS: usize = 375;
// `[BEGIN_AUDIO]` before the run
const MARKERS: usize = 1;
// `<s>[INST]Hi[/INST]`: the tiny vocabulary spells "Hi" in two byte tokens
const TEXT_PROMPT_TOKENS: usize = 5;
// `[SYSTEM_PROMPT]` and `[/SYSTEM_PROMPT]` around the system prompt, one byte token per ASCII character
const SYSTEM_TOKENS: usize = 2 + SYSTEM.len();
const TONE_HZ: f32 = 440.0;
// a reload computes as the load that wrote it, so only rounding parts them
const UQFF_TOLERANCE: f32 = 1e-3;
const DURATION_TOLERANCE: f64 = 1e-6;

fn builder(dir: &std::path::Path) -> MultimodalModelBuilder {
    MultimodalModelBuilder::new(dir.to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
}

async fn build(dir: &std::path::Path) -> anyhow::Result<Model> {
    Ok(builder(dir).build().await?)
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

#[tokio::test]
async fn the_loader_template_frames_text_system_and_audio_turns() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let text = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    assert_eq!(text.usage.prompt_tokens, TEXT_PROMPT_TOKENS);
    let system = model
        .send_chat_request(
            RequestBuilder::new()
                .add_message(TextMessageRole::System, SYSTEM)
                .add_message(TextMessageRole::User, PROMPT)
                .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    assert_eq!(
        system.usage.prompt_tokens,
        TEXT_PROMPT_TOKENS + SYSTEM_TOKENS
    );
    let audio = model
        .send_chat_request(
            RequestBuilder::from(MultimodalMessages::new().add_audio_message(
                TextMessageRole::User,
                PROMPT,
                vec![clip()],
            ))
            .set_sampler_max_len(MAX_LEN),
        )
        .await?;
    assert_eq!(
        audio.usage.prompt_tokens - text.usage.prompt_tokens,
        AUDIO_TOKENS + MARKERS
    );
    assert!(audio.usage.completion_tokens > 0);
    Ok(())
}

fn transcription(language: Option<&str>) -> TranscriptionRequest {
    TranscriptionRequest {
        model: "default".to_string(),
        language: language.map(str::to_string),
        prompt: None,
        response_format: TranscriptionResponseFormat::VerboseJson,
        temperature: None,
        timestamp_granularities: Vec::new(),
        beam_size: None,
        diarization_model: None,
        task: Default::default(),
    }
}

// transcription mode decodes from `[TRANSCRIBE]`; a language off the model card and the translate task are refused
#[tokio::test]
async fn a_transcription_request_decodes_the_clip() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = build(checkpoint.path()).await?;
    let audio = wav(&clip());
    let output = model
        .transcription(transcription(Some("en")), &audio)
        .await
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let body: serde_json::Value = serde_json::from_str(&output.body)?;
    let duration = body["duration"].as_f64().unwrap_or_default();
    assert!((duration - SAMPLES as f64 / f64::from(SAMPLE_RATE)).abs() < DURATION_TOLERANCE);
    assert!(
        model
            .transcription(transcription(Some("ja")), &audio)
            .await
            .is_err()
    );
    let mut translate = transcription(None);
    translate.task = inference::TranscriptionTask::Translate;
    assert!(model.transcription(translate, &audio).await.is_err());
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
            vec![clip()],
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
