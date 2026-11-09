//! Nemotron-3 Diarization through the engine on a tiny random-weight checkpoint whose 6 s clips fill and compress
//! the speaker cache: frame counts, thresholds, RTTM, and repeats on the default device.

use inference::{
    ApiErrorKind, DiarizationModelBuilder, DiarizationRequest, DiarizationResponse,
    DiarizationResponseFormat, ModelDType,
};

#[path = "../support/nemotron3_diarization_tiny.rs"]
mod support;
use support::tiny_nemotron3_diarization;

const RATE: u32 = 16_000;
const SECONDS: f64 = 6.0;
const HOP: usize = 160;
const SPEAKERS: usize = 4;
const REPEATS: usize = 2;
const CHIRP_START_HZ: f64 = 150.0;
const CHIRP_RISE_HZ: f64 = 300.0;

fn chirp_wav() -> Vec<u8> {
    let pcm: Vec<f32> = (0..(SECONDS * f64::from(RATE)) as usize)
        .map(|i| {
            let t = i as f64 / f64::from(RATE);
            (0.3 * (2.0
                * std::f64::consts::PI
                * (CHIRP_START_HZ * t + CHIRP_RISE_HZ * t * t / 2.0))
                .sin()) as f32
        })
        .collect();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1).unwrap();
    wav
}

fn json(request: &DiarizationRequest, body: &str) -> DiarizationResponse {
    assert_eq!(request.response_format, DiarizationResponseFormat::Json);
    serde_json::from_str(body).unwrap()
}

#[tokio::test]
async fn the_cache_backed_model_scores_every_frame_and_thresholds_cut_it() -> anyhow::Result<()> {
    let dir = tiny_nemotron3_diarization()?;
    let model = DiarizationModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let wav = chirp_wav();
    let mut request = DiarizationRequest::new();
    request.return_probabilities = true;
    let out = model.diarization(request.clone(), &wav).await?;
    assert_eq!(out.content_type, "application/json");
    let diarization = json(&request, &out.body);
    let probabilities = diarization.probabilities.clone().unwrap_or_default();
    assert_eq!(diarization.num_speakers, SPEAKERS);
    assert_eq!(
        probabilities.len(),
        (SECONDS * f64::from(RATE)) as usize / HOP
    );
    assert!(
        probabilities
            .iter()
            .flatten()
            .all(|p| (0.0..=1.0).contains(p))
    );
    assert!(
        diarization
            .segments
            .windows(2)
            .all(|w| (w[0].start, w[0].speaker) <= (w[1].start, w[1].speaker))
    );

    // a zero threshold makes every speaker speak throughout, one segment each; 1 silences them all
    request.threshold = Some(0.0);
    request.return_probabilities = false;
    let all = json(
        &request,
        &model.diarization(request.clone(), &wav).await?.body,
    );
    assert_eq!(all.segments.len(), SPEAKERS, "{:?}", all.segments);
    assert!(
        all.segments
            .iter()
            .all(|s| s.start == 0.0 && (s.end - diarization.duration).abs() < 0.011)
    );
    request.threshold = Some(1.0);
    assert!(
        json(
            &request,
            &model.diarization(request.clone(), &wav).await?.body
        )
        .segments
        .is_empty()
    );
    request.threshold = Some(1.5);
    assert!(model.diarization(request.clone(), &wav).await.is_err());

    request.threshold = Some(0.0);
    request.response_format = DiarizationResponseFormat::Rttm;
    let rttm = model.diarization(request, &wav).await?;
    assert_eq!(
        rttm.body
            .lines()
            .filter(|l| l.starts_with("SPEAKER audio 1 "))
            .count(),
        SPEAKERS
    );

    let err = model
        .diarization(DiarizationRequest::new(), b"not audio")
        .await
        .expect_err("garbage decoded");
    assert_eq!(err.kind, ApiErrorKind::InvalidRequest, "{err}");
    Ok(())
}

// On the default device (CUDA under `--features cuda`) the cache's choices repeat exactly
#[tokio::test]
async fn diarization_repeats_exactly_on_the_default_device() -> anyhow::Result<()> {
    let dir = tiny_nemotron3_diarization()?;
    let model = DiarizationModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .build()
        .await?;
    let wav = chirp_wav();
    let mut request = DiarizationRequest::new();
    request.return_probabilities = true;
    let first = model.diarization(request.clone(), &wav).await?.body;
    for _ in 0..REPEATS {
        assert_eq!(model.diarization(request.clone(), &wav).await?.body, first);
    }
    Ok(())
}

// a recording diarized as a stream: `started` with its length, a `progress` per chunk rising to it, and last `done`
// with the one-shot response's segments; canceled at its first chunk it stops before the last
#[tokio::test]
async fn a_recording_diarizes_as_a_stream_of_chunk_progress() -> anyhow::Result<()> {
    use futures::StreamExt;
    let dir = tiny_nemotron3_diarization()?;
    let model = DiarizationModelBuilder::new(dir.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let wav = chirp_wav();
    let whole = model.diarization(DiarizationRequest::new(), &wav).await?;
    let whole: serde_json::Value = serde_json::from_str(&whole.body)?;
    let read = |stream: inference_api::speech_streams::SpeechStream, cancel: bool| async move {
        let mut events = stream.events;
        let mut out: Vec<serde_json::Value> = Vec::new();
        while let Some(event) = events.next().await {
            let event: serde_json::Value = serde_json::from_str(&event).unwrap();
            if cancel && event["event"] == "progress" {
                stream.cancellation.cancel();
            }
            out.push(event);
        }
        out
    };
    let events = read(
        model
            .diarization_stream_json(b"{}", Some(wav.clone()))
            .await?,
        false,
    )
    .await;
    assert_eq!(events[0]["event"], "started");
    assert!((events[0]["data"]["duration"].as_f64().unwrap() - SECONDS).abs() < 1e-6);
    let progress: Vec<f64> = events
        .iter()
        .filter(|e| e["event"] == "progress")
        .map(|e| e["data"]["seconds_done"].as_f64().unwrap())
        .collect();
    assert!(progress.len() > 1, "{} chunks", progress.len());
    assert!(progress.windows(2).all(|p| p[0] < p[1]));
    assert!((progress.last().unwrap() - SECONDS).abs() < 1e-6);
    let done = events.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["data"]["segments"], whole["segments"]);
    let canceled = read(model.diarization_stream_json(b"{}", Some(wav)).await?, true).await;
    assert_eq!(canceled.last().unwrap()["event"], "canceled");
    // the chunk running when the cancel lands still finishes and reports
    assert!(canceled.iter().filter(|e| e["event"] == "progress").count() < progress.len());
    Ok(())
}
