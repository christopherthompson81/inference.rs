//! Streamed transcription: a session the transcription pipeline keeps, fed audio as it arrives.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use inference_core::{
    InferenceRs, RequestMessage, Response, TranscriptionOptions, TranscriptionStreamOp,
    TranscriptionStreamRequest, TranscriptionStreamUpdate,
};

use crate::{
    api_error::{ApiError, ApiErrorKind},
    generation::{render_transcript, run_unlogged, unexpected},
    lora_routing::DEFAULT_MODEL_ID,
    openai::{TranscriptionOutput, TranscriptionSessionRequest},
    types::SharedInferenceRsState,
};

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);
/// Bytes per little-endian PCM16 sample.
pub const PCM16_BYTES: usize = 2;
/// A PCM16 sample's full scale.
pub const PCM16_SCALE: f32 = 32768.0;
// telephone audio to studio audio; outside it a resampling ratio turns a few bytes into minutes of audio
const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 192_000;

/// An open streamed transcription. Each push returns the text the audio so far completes; `finish` returns the
/// rest and the whole transcript. Dropped unfinished, it closes its session.
///
/// A push is not cancel-safe: dropped before it completes, its audio may or may not have reached the session.
pub struct TranscriptionSession {
    state: SharedInferenceRsState,
    request: TranscriptionSessionRequest,
    // the model it opened on, not an alias or the default, which could later name another
    model: String,
    id: u64,
    open: bool,
    runtime: tokio::runtime::Handle,
}

pub(crate) async fn open(
    state: &SharedInferenceRsState,
    request: TranscriptionSessionRequest,
) -> Result<TranscriptionSession, ApiError> {
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&request.sample_rate) {
        return Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!("`sample_rate` must be {MIN_SAMPLE_RATE} to {MAX_SAMPLE_RATE}"),
            Some("invalid_sample_rate"),
            Some("sample_rate"),
        ));
    }
    let named = (request.model != DEFAULT_MODEL_ID).then_some(request.model.as_str());
    let model = state
        .resolve_alias_or_default(named)
        .map_err(|e| ApiError::from_error(&e, ApiErrorKind::NotFound))?;
    InferenceRs::maybe_log_request(
        state.clone(),
        serde_json::to_string(&request).unwrap_or_default(),
    );
    let id = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
    let op = TranscriptionStreamOp::Open {
        options: TranscriptionOptions {
            language: request.language.clone(),
            beam_size: None,
        },
        sample_rate: request.sample_rate,
    };
    step(state, &model, id, op).await?;
    Ok(TranscriptionSession {
        state: state.clone(),
        request,
        model,
        id,
        open: true,
        runtime: tokio::runtime::Handle::current(),
    })
}

async fn step(
    state: &SharedInferenceRsState,
    model: &str,
    session: u64,
    op: TranscriptionStreamOp,
) -> Result<TranscriptionStreamUpdate, ApiError> {
    let messages = RequestMessage::TranscriptionStream {
        request: TranscriptionStreamRequest { session, op },
    };
    match run_unlogged(state, model, messages).await? {
        Response::TranscriptionStream(update) => Ok(update),
        _ => Err(unexpected(state)),
    }
}

impl TranscriptionSession {
    /// Feeds mono samples at the session's rate; returns the text they add.
    pub async fn push(&mut self, pcm: &[f32]) -> Result<String, ApiError> {
        self.push_samples(pcm.to_vec()).await
    }

    /// As [`TranscriptionSession::push`], for little-endian 16-bit PCM bytes.
    pub async fn push_pcm16(&mut self, audio: &[u8]) -> Result<String, ApiError> {
        let (samples, rest) = audio.as_chunks::<PCM16_BYTES>();
        if !rest.is_empty() {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                "16-bit PCM takes an even number of bytes".to_string(),
                Some("invalid_audio"),
                Some("audio"),
            ));
        }
        let pcm = samples
            .iter()
            .map(|b| f32::from(i16::from_le_bytes(*b)) / PCM16_SCALE)
            .collect();
        self.push_samples(pcm).await
    }

    async fn push_samples(&mut self, pcm: Vec<f32>) -> Result<String, ApiError> {
        let op = TranscriptionStreamOp::Push(Arc::new(pcm));
        Ok(step(&self.state, &self.model, self.id, op).await?.delta)
    }

    /// Ends the session: the text its last audio adds, and the whole transcript in the requested format.
    pub async fn finish(mut self) -> Result<(String, TranscriptionOutput), ApiError> {
        self.open = false;
        let update = step(
            &self.state,
            &self.model,
            self.id,
            TranscriptionStreamOp::Finish,
        )
        .await?;
        let transcription = update
            .transcription
            .ok_or_else(|| unexpected(&self.state))?;
        let output = render_transcript(
            &self.request.as_transcription_request(),
            transcription,
            None,
        )?;
        Ok((update.delta, output))
    }
}

impl Drop for TranscriptionSession {
    fn drop(&mut self) {
        // a model unloaded since holds no sessions, and a close must not reload it
        if !self.open || !self.state.is_model_loaded(&self.model).unwrap_or(false) {
            return;
        }
        let (state, model, id) = (self.state.clone(), self.model.clone(), self.id);
        self.runtime.spawn(async move {
            let _ = step(&state, &model, id, TranscriptionStreamOp::Close).await;
        });
    }
}
