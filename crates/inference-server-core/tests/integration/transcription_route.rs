//! /v1/audio/transcriptions end to end on a tiny random-weight Parakeet: the multipart form, formats and errors.

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use inference_server_core::inference_server_router_builder::InferenceRsServerRouterBuilder;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::{chat_route::body_text, diarization_support, parakeet_support, silero_support};

const BOUNDARY: &str = "inference-transcription-boundary";
const ROUTE: &str = "/v1/audio/transcriptions";
const VAD_ROUTE: &str = "/v1/audio/vad";
const DIARIZATION_ROUTE: &str = "/v1/audio/diarization";
const RATE: u32 = 16_000;
const SECONDS: usize = 1;
const TONE_HZ: f32 = 440.0;

async fn router(dir: &std::path::Path) -> anyhow::Result<axum::Router> {
    let spec = serde_json::from_value(json!({
        "model": {"Transcription": {"model_id": dir.to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await
}

fn tone_wav() -> Vec<u8> {
    let pcm: Vec<f32> = (0..RATE as usize * SECONDS)
        .map(|i| (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / RATE as f32).sin() * 0.3)
        .collect();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1).unwrap();
    wav
}

// a multipart form: text fields in order, then the audio as `file` when given
fn form(fields: &[(&str, &str)], audio: Option<&[u8]>) -> Request<Body> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    if let Some(audio) = audio {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(audio);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    Request::post(ROUTE)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_form_transcribes_with_word_timestamps() -> anyhow::Result<()> {
    let dir = parakeet_support::tiny_parakeet_checkpoint(parakeet_support::HEADS[0])?;
    let app = router(dir.path()).await?;
    let wav = tone_wav();

    let response = app
        .clone()
        .oneshot(form(
            &[
                ("model", "default"),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
                ("timestamp_granularities[]", "segment"),
                ("temperature", "0"),
            ],
            Some(&wav),
        ))
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
    let verbose: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(verbose["task"], "transcribe");
    assert!(verbose["words"].is_array(), "{verbose}");
    assert!(verbose["segments"].is_array(), "{verbose}");

    let response = app
        .clone()
        .oneshot(form(&[("response_format", "text")], Some(&wav)))
        .await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body_text(response).await?,
        verbose["text"].as_str().unwrap()
    );

    let models: Value = serde_json::from_str(
        &body_text(
            app.clone()
                .oneshot(Request::get("/v1/models").body(Body::empty())?)
                .await?,
        )
        .await?,
    )?;
    assert_eq!(models["data"][0]["category"], "transcription", "{models}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_form_without_audio_or_with_an_unknown_format_is_refused() -> anyhow::Result<()> {
    let dir = parakeet_support::tiny_parakeet_checkpoint(parakeet_support::HEADS[0])?;
    let app = router(dir.path()).await?;

    let response = app
        .clone()
        .oneshot(form(&[("model", "default")], None))
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(error["error"]["param"], "file", "{error}");

    let response = app
        .clone()
        .oneshot(form(&[("response_format", "mp3")], Some(&tone_wav())))
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let response = app
        .clone()
        .oneshot(form(&[("stream", "true")], Some(&tone_wav())))
        .await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(error["error"]["param"], "stream", "{error}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn the_vad_form_returns_segments_and_probabilities() -> anyhow::Result<()> {
    let dir = silero_support::tiny_silero_gguf(false)?;
    let spec = serde_json::from_value(json!({
        "model": {"VoiceActivity": {"model_id": dir.path().to_string_lossy()}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let app = InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await?;
    let mut request = form(
        &[
            ("threshold", "0"),
            ("neg_threshold", "-1"),
            ("return_probabilities", "true"),
        ],
        Some(&tone_wav()),
    );
    *request.uri_mut() = VAD_ROUTE.parse()?;
    let response = app.clone().oneshot(request).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let activity: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(
        activity["segments"].as_array().map(Vec::len),
        Some(1),
        "{activity}"
    );
    assert!(activity["probabilities"].is_array(), "{activity}");

    let mut bad = form(&[("threshold", "high")], Some(&tone_wav()));
    *bad.uri_mut() = VAD_ROUTE.parse()?;
    let response = app.clone().oneshot(bad).await?;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error: Value = serde_json::from_str(&body_text(response).await?)?;
    assert_eq!(error["error"]["param"], "threshold", "{error}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn the_diarization_form_returns_segments_or_rttm() -> anyhow::Result<()> {
    let dir = diarization_support::tiny_nemotron3_diarization()?;
    let spec = serde_json::from_value(json!({
        "model": {"Diarization": {"model_id": dir.path().to_string_lossy(), "dtype": "f32"}},
        "runtime": {"device": "cpu"},
    }))?;
    let engine = inference_api::Engine::load(spec).await?;
    let app = InferenceRsServerRouterBuilder::new()
        .with_engine(&engine)
        .build()
        .await?;
    let mut request = form(&[("threshold", "0")], Some(&tone_wav()));
    *request.uri_mut() = DIARIZATION_ROUTE.parse()?;
    let response = app.clone().oneshot(request).await?;
    assert_eq!(response.status(), StatusCode::OK);
    let diarization: Value = serde_json::from_str(&body_text(response).await?)?;
    let speakers = diarization["num_speakers"].as_u64().unwrap_or_default() as usize;
    assert_eq!(
        diarization["segments"].as_array().map(Vec::len),
        Some(speakers),
        "{diarization}"
    );

    let mut rttm = form(
        &[("threshold", "0"), ("response_format", "rttm")],
        Some(&tone_wav()),
    );
    *rttm.uri_mut() = DIARIZATION_ROUTE.parse()?;
    let response = app.clone().oneshot(rttm).await?;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_text(response).await?.lines().count(), speakers);
    Ok(())
}
