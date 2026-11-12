//! A tiny random-weight VibeVoice-ASR Streaming: a recording transcribes window by window through the engine, a
//! session fed in pieces writes the same transcript, and a pass past its cache budget starts afresh.

use inference::{
    ModelDType, TranscriptionModelBuilder, TranscriptionRequest, TranscriptionResponseFormat,
    TranscriptionSessionRequest, VerboseTranscriptionResponse,
};

#[path = "../support/vibevoice_streaming_tiny.rs"]
mod support;
use support::{CHUNK_SAMPLES, load, tiny_checkpoint};

const RATE: u32 = 24_000;
// three windows, the last one short
const SAMPLES: usize = 2 * CHUNK_SAMPLES + CHUNK_SAMPLES / 2;
const TONE_HZ: f32 = 330.0;
// pieces that do not line up with the windows
const PIECE: usize = 777;
const DURATION_TOLERANCE: f64 = 1e-6;
// a window's 30 frames between its two markers, 256 tokens of text at most and the closing marker
const WINDOW_POSITIONS: usize = 30 + 2 + 256 + 1;

fn pcm() -> Vec<f32> {
    (0..SAMPLES)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * TONE_HZ * i as f32 / RATE as f32).sin())
        .collect()
}

fn pcm16(pcm: &[f32]) -> Vec<u8> {
    pcm.iter()
        .flat_map(|s| ((s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16).to_le_bytes())
        .collect()
}

#[tokio::test]
async fn a_session_writes_the_recordings_transcript() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = TranscriptionModelBuilder::new(checkpoint.path().to_string_lossy())
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .build()
        .await?;
    let pcm = pcm();
    let mut wav = Vec::new();
    inference_models_speech::utils::write_pcm_as_wav(&mut wav, &pcm, RATE, 1)?;
    let whole = model
        .transcription(
            TranscriptionRequest::new(TranscriptionResponseFormat::VerboseJson),
            &wav,
        )
        .await?;
    let parsed: VerboseTranscriptionResponse = serde_json::from_str(&whole.body)?;
    let seconds = SAMPLES as f64 / f64::from(RATE);
    assert!(
        (parsed.duration - seconds).abs() < DURATION_TOLERANCE,
        "{}",
        parsed.duration
    );

    let mut session = model
        .transcription_session(TranscriptionSessionRequest {
            response_format: TranscriptionResponseFormat::VerboseJson,
            ..TranscriptionSessionRequest::new(RATE)
        })
        .await?;
    for chunk in pcm16(&pcm).chunks(2 * PIECE) {
        session.push_pcm16(chunk).await?;
    }
    let (_, streamed) = session.finish().await?;
    let streamed: VerboseTranscriptionResponse = serde_json::from_str(&streamed.body)?;
    // the upload is 16-bit PCM too, so both decode the same samples
    assert_eq!(streamed.text, parsed.text);
    assert_eq!(streamed.segments.len(), parsed.segments.len());
    assert!(
        (streamed.duration - seconds).abs() < DURATION_TOLERANCE,
        "{}",
        streamed.duration
    );
    Ok(())
}

#[test]
fn a_window_past_the_pass_budget_decodes_as_in_a_fresh_session() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = load(checkpoint.path())?;
    let pcm = pcm();
    let (window, chunk) = model.window_shape();
    let alone: Vec<String> = (0..SAMPLES)
        .step_by(chunk)
        .map(|at| {
            let mut session = model.start(None)?;
            Ok(model
                .window(&mut session, &pcm[at..(at + window).min(SAMPLES)])?
                .1)
        })
        .collect::<anyhow::Result<_>>()?;
    // room for the prompt and one window at its longest only, so each later window opens a pass
    let budget = model.start(None)?.offset() + WINDOW_POSITIONS;
    let model = model.with_pass_positions(budget);
    let mut session = model.start(None)?;
    let mut windows = model.push(&mut session, &pcm)?;
    windows.extend(model.finish(session)?.0);
    assert_eq!(
        windows.iter().map(|w| w.pass).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(
        windows.into_iter().map(|w| w.text).collect::<Vec<_>>(),
        alone
    );
    Ok(())
}

#[test]
fn a_context_joins_the_prompt() -> anyhow::Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let model = load(checkpoint.path())?;
    let plain = model.start(None)?.offset();
    assert_eq!(model.start(Some("  "))?.offset(), plain);
    assert!(model.start(Some("tone, hum"))?.offset() > plain);
    Ok(())
}
