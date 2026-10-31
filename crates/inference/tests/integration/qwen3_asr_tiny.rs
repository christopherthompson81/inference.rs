//! A tiny random-weight Qwen3-ASR: a clip expands to its encoder tokens between its start and end markers, the
//! encoder runs a full attention window and a shorter one, and a repeat served from the prefix cache decodes the same.

use inference::{
    AudioInput, Model, ModelDType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder,
    TextMessageRole, TranscriptionRequest, TranscriptionResponseFormat,
};

#[path = "../support/qwen3_asr_tiny.rs"]
mod support;
use support::{tiny_checkpoint, tiny_transformers_checkpoint};

const PROMPT: &str = "Transcribe it.";
const MAX_LEN: usize = 6;
const SAMPLE_RATE: u32 = 16_000;
// 3 s: 300 mel frames, 3 chunks of 13 tokens; the fixture's 2-chunk window splits them 26 + 13
const SAMPLES: usize = 48_000;
const AUDIO_TOKENS: usize = 39;
// the clip's start and end markers
const MARKERS: usize = 2;
const TONE_HZ: f32 = 440.0;
const WAV_HEADER_BYTES: usize = 44;
const DURATION_TOLERANCE: f64 = 1e-6;
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

// 16-bit mono PCM in a RIFF/WAVE container
fn wav(clip: &AudioInput) -> Vec<u8> {
    let data = clip.samples.len() * 2;
    let mut bytes = Vec::with_capacity(WAV_HEADER_BYTES + data);
    let u32_le = |value: usize| {
        u32::try_from(value)
            .expect("a test clip fits a WAV")
            .to_le_bytes()
    };
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&u32_le(WAV_HEADER_BYTES - 8 + data));
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&clip.sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(clip.sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&u32_le(data));
    for sample in &clip.samples {
        bytes.extend_from_slice(&((sample * f32::from(i16::MAX)) as i16).to_le_bytes());
    }
    bytes
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
    }
}

// a transcription request decodes from the model's transcription prompt; a language it was not trained on is refused
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
    assert_eq!(body["language"], "en");
    assert!(
        model
            .transcription(transcription(Some("xx")), &audio)
            .await
            .is_err()
    );
    Ok(())
}

// transformers' `-hf` layout loads from its flat config and `model.` names, and transcribes as the original does
#[tokio::test]
async fn the_transformers_layout_transcribes_as_the_original() -> anyhow::Result<()> {
    let original = tiny_checkpoint()?;
    let transformers = tiny_transformers_checkpoint()?;
    let audio = wav(&clip());
    let mut bodies = Vec::new();
    for checkpoint in [&original, &transformers] {
        let model = build(checkpoint.path()).await?;
        let output = model
            .transcription(transcription(None), &audio)
            .await
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        bodies.push(output.body);
    }
    assert_eq!(bodies[0], bodies[1]);
    Ok(())
}

// as the `-hf` repos' template does: each audio part writes its placeholder, the user's text is dropped
const PARTS_TEMPLATE: &str = "{%- for m in messages -%}{%- if m.role == 'user' -%}<|im_start|>user\n{%- for c in m.content -%}{%- if c.type == 'audio' -%}<|audio_start|><|audio_pad|><|audio_end|>{%- endif -%}{%- endfor -%}<|im_end|>\n{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";

// a template that writes the audio placeholder itself gets the content parts, not the flattened text
#[tokio::test]
async fn a_template_writing_audio_placeholders_gets_the_parts() -> anyhow::Result<()> {
    let checkpoint = tiny_transformers_checkpoint()?;
    std::fs::write(
        checkpoint.path().join("chat_template.jinja"),
        PARTS_TEMPLATE,
    )?;
    let model = build(checkpoint.path()).await?;
    let response = model.send_chat_request(request()).await?;
    // flattened text would hold no placeholder for the template to keep, and the request would fail on the count
    assert!(response.usage.prompt_tokens > AUDIO_TOKENS + MARKERS);
    Ok(())
}
