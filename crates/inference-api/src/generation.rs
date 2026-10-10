//! Image and speech generation as engine operations, free of HTTP.

use futures::future::BoxFuture;
use inference_core::{
    AudioInput, DiffusionGenerationParams, ImageChoice, ImageGenerationResponse,
    ImageGenerationResponseFormat, InferenceRs, NormalRequest, Request, RequestMessage, Response,
    SamplingParams, SegmentOptions, SpeechOptions, TimedText, Transcription, VoiceActivity,
    speech_utils::{self, Sample},
};

use inference_protocol::images::{encode_png, image_generation_response};

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::{base_process_non_streaming_response, create_response_channel, send_request},
    files::store_generated_image,
    lora_routing::DEFAULT_MODEL_ID,
    openai::{
        AudioResponseFormat, ImageGenerationRequest, SpeechGenerationRequest, SpeechSegment,
        TimestampGranularity, TranscriptionOutput, TranscriptionRequest, TranscriptionResponse,
        TranscriptionResponseFormat, TranscriptionSegment, TranscriptionWord,
        VerboseTranscriptionResponse, VoiceActivityRequest, VoiceActivityResponse, transcript_srt,
        transcript_vtt,
    },
    types::SharedInferenceRsState,
    util::validate_model_name,
};

const TRANSCRIBE_TASK: &str = "transcribe";
const SENTENCE_ENDS: [char; 4] = ['.', '?', '!', '\u{3002}'];
// a silence this long starts a new subtitle even mid-sentence
const SEGMENT_PAUSE_SECONDS: f64 = 1.5;
// unpunctuated transcripts (the English CTC and RNN-T checkpoints) still break into subtitle-sized segments
const MAX_SEGMENT_SECONDS: f64 = 30.0;

// Speech models emit f32 samples; PCM output is signed 16-bit little-endian.
const PCM_SAMPLE_FORMAT: &str = "s16le";
const UNEXPECTED_RESPONSE: &str =
    "the engine answered a generation request with another kind of response";

/// Encoded speech audio and its MIME type, which carries the sample rate and channel count.
pub struct SpeechAudio {
    pub bytes: Vec<u8>,
    pub content_type: String,
}

// Generation requests are one-shot and deterministic: no tools, streaming or sampling choices.
fn generation_request(
    state: &SharedInferenceRsState,
    model: &str,
    messages: RequestMessage,
    tx: tokio::sync::mpsc::Sender<Response>,
) -> Request {
    let mut request = NormalRequest::new_simple(
        messages,
        SamplingParams::deterministic(),
        tx,
        state.next_request_id(),
        None,
        None,
    );
    request.model_id = (model != DEFAULT_MODEL_ID).then(|| model.to_string());
    Request::Normal(Box::new(request))
}

// Sends a generation request and waits for its one final response.
async fn run(
    state: &SharedInferenceRsState,
    model: &str,
    repr: String,
    messages: RequestMessage,
) -> Result<Response, ApiError> {
    InferenceRs::maybe_log_request(state.clone(), repr);
    validate_model_name(model, state.clone())
        .map_err(|error| ApiError::from_error(&error, ApiErrorKind::InvalidRequest))?;
    let (tx, mut rx) = create_response_channel(None);
    send_request(state, generation_request(state, model, messages, tx))
        .await
        .map_err(|error| {
            InferenceRs::maybe_log_error(state.clone(), &error);
            ApiError::from_error(&error, ApiErrorKind::Internal)
        })?;
    base_process_non_streaming_response(
        &mut rx,
        state.clone(),
        |state, response| match response {
            Response::ValidationError(error) => Err(ApiError::from_error(
                error.as_ref(),
                ApiErrorKind::InvalidRequest,
            )),
            Response::InternalError(error) => {
                InferenceRs::maybe_log_error(state, &*error);
                Err(ApiError::from_error(error.as_ref(), ApiErrorKind::Internal))
            }
            Response::CompletionModelError(message, _) => {
                InferenceRs::maybe_log_error(state, &ModelErrorMessage(message));
                Err(ApiError::model_error())
            }
            response => Ok(response),
        },
        |state, error| {
            InferenceRs::maybe_log_error(state, error.as_ref());
            Err(ApiError::internal())
        },
    )
    .await
}

fn unexpected(state: &SharedInferenceRsState) -> ApiError {
    InferenceRs::maybe_log_error(
        state.clone(),
        &ModelErrorMessage(UNEXPECTED_RESPONSE.to_string()),
    );
    ApiError::internal()
}

/// Generates images from a prompt with a diffusion model.
pub(crate) fn generate_image<'a>(
    state: &'a SharedInferenceRsState,
    request: ImageGenerationRequest,
    owner: Option<&'a str>,
) -> BoxFuture<'a, Result<ImageGenerationResponse, ApiError>> {
    Box::pin(generate_image_inner(state, request, owner))
}

async fn generate_image_inner(
    state: &SharedInferenceRsState,
    request: ImageGenerationRequest,
    owner: Option<&str>,
) -> Result<ImageGenerationResponse, ApiError> {
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let messages = RequestMessage::ImageGeneration {
        prompt: request.prompt,
        generation_params: DiffusionGenerationParams {
            height: request.height,
            width: request.width,
        },
    };
    let Response::ImageGeneration(generated) = run(state, &request.model, repr, messages).await?
    else {
        return Err(unexpected(state));
    };
    let encode_error = |e: anyhow::Error| {
        InferenceRs::maybe_log_error(state.clone(), e.as_ref());
        ApiError::from_error(e.as_ref(), ApiErrorKind::Internal)
    };
    let response = match request.response_format {
        ImageGenerationResponseFormat::Url => {
            let model = (request.model != DEFAULT_MODEL_ID).then_some(request.model.as_str());
            let data = generated
                .images
                .iter()
                .map(|image| {
                    let url = store_generated_image(
                        state,
                        model,
                        encode_png(image).map_err(encode_error)?,
                        owner,
                    )?;
                    Ok(ImageChoice {
                        url: Some(url),
                        b64_json: None,
                    })
                })
                .collect::<Result<_, ApiError>>()?;
            ImageGenerationResponse {
                created: generated.created,
                data,
            }
        }
        format => image_generation_response(generated.created, &generated.images, format, None)
            .map_err(encode_error)?,
    };
    InferenceRs::maybe_log_response(state.clone(), &response);
    Ok(response)
}

/// Speaks `input` with a speech model, encoded as WAV or 16-bit PCM.
pub(crate) fn generate_speech<'a>(
    state: &'a SharedInferenceRsState,
    request: SpeechGenerationRequest,
) -> BoxFuture<'a, Result<SpeechAudio, ApiError>> {
    Box::pin(generate_speech_inner(state, request))
}

async fn generate_speech_inner(
    state: &SharedInferenceRsState,
    request: SpeechGenerationRequest,
) -> Result<SpeechAudio, ApiError> {
    let format = request.response_format;
    if !matches!(format, AudioResponseFormat::Wav | AudioResponseFormat::Pcm) {
        return Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            "Only wav and pcm response formats are supported.",
            Some("invalid_response_format"),
            Some("response_format"),
        ));
    }
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let messages = RequestMessage::SpeechGeneration {
        prompt: request.input,
        options: SpeechOptions {
            voice: request.voice,
            speed: request.speed,
            phonemes: request.phonemes,
            seed: request.seed,
        },
    };
    match run(state, &request.model, repr, messages).await? {
        Response::Speech {
            pcm,
            rate,
            channels,
        } => Ok(SpeechAudio {
            bytes: encode_speech(format, &pcm, rate, channels)?,
            content_type: format.audio_content_type(rate, channels, PCM_SAMPLE_FORMAT),
        }),
        _ => Err(unexpected(state)),
    }
}

/// Transcribes encoded `audio` (WAV, MP3, FLAC, ...) with a speech recognition model, in the request's format.
pub(crate) fn transcribe<'a>(
    state: &'a SharedInferenceRsState,
    request: TranscriptionRequest,
    audio: &'a [u8],
) -> BoxFuture<'a, Result<TranscriptionOutput, ApiError>> {
    Box::pin(transcribe_inner(state, request, audio))
}

async fn transcribe_inner(
    state: &SharedInferenceRsState,
    request: TranscriptionRequest,
    audio: &[u8],
) -> Result<TranscriptionOutput, ApiError> {
    let audio = decode_audio(audio)?;
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    match run(
        state,
        &request.model,
        repr,
        RequestMessage::Transcription { audio },
    )
    .await?
    {
        Response::Transcription(transcript) => render_transcript(&request, transcript),
        _ => Err(unexpected(state)),
    }
}

fn render_transcript(
    request: &TranscriptionRequest,
    transcript: Transcription,
) -> Result<TranscriptionOutput, ApiError> {
    let format = request.response_format;
    let segments = transcript_segments(&transcript.words);
    let body = match format {
        TranscriptionResponseFormat::Json => serde_json::to_string(&TranscriptionResponse {
            text: transcript.text,
        })
        .map_err(|_| ApiError::internal())?,
        TranscriptionResponseFormat::Text => transcript.text,
        TranscriptionResponseFormat::Srt => transcript_srt(&segments),
        TranscriptionResponseFormat::Vtt => transcript_vtt(&segments),
        TranscriptionResponseFormat::VerboseJson => {
            let words = request
                .timestamp_granularities
                .contains(&TimestampGranularity::Word)
                .then(|| {
                    transcript
                        .words
                        .iter()
                        .map(|w| TranscriptionWord {
                            word: w.text.clone(),
                            start: w.start,
                            end: w.end,
                        })
                        .collect()
                });
            serde_json::to_string(&VerboseTranscriptionResponse {
                task: TRANSCRIBE_TASK.to_string(),
                // the model reports none, so a language the client named is echoed back
                language: request.language.clone(),
                duration: transcript.duration,
                text: transcript.text,
                words,
                segments,
            })
            .map_err(|_| ApiError::internal())?
        }
    };
    Ok(TranscriptionOutput {
        body,
        content_type: format.content_type(),
    })
}

// a segment ends after a word closing a sentence, or before a pause long enough to be one
fn transcript_segments(words: &[TimedText]) -> Vec<TranscriptionSegment> {
    let mut segments: Vec<TranscriptionSegment> = Vec::new();
    let mut open = false;
    for word in words {
        let pause = segments.last().is_some_and(|s| {
            word.start - s.end >= SEGMENT_PAUSE_SECONDS || word.end - s.start > MAX_SEGMENT_SECONDS
        });
        match segments.last_mut() {
            Some(segment) if open && !pause => {
                segment.text.push(' ');
                segment.text.push_str(&word.text);
                segment.end = word.end;
            }
            _ => segments.push(TranscriptionSegment {
                id: segments.len(),
                start: word.start,
                end: word.end,
                text: word.text.clone(),
            }),
        }
        open = !word.text.ends_with(SENTENCE_ENDS);
    }
    segments
}

/// Speech segments of encoded `audio` from a voice activity model.
pub(crate) fn detect_voice_activity<'a>(
    state: &'a SharedInferenceRsState,
    request: VoiceActivityRequest,
    audio: &'a [u8],
) -> BoxFuture<'a, Result<VoiceActivityResponse, ApiError>> {
    Box::pin(detect_voice_activity_inner(state, request, audio))
}

async fn detect_voice_activity_inner(
    state: &SharedInferenceRsState,
    request: VoiceActivityRequest,
    audio: &[u8],
) -> Result<VoiceActivityResponse, ApiError> {
    validate_segment_request(&request)?;
    let audio = decode_audio(audio)?;
    let defaults = SegmentOptions::default();
    let options = SegmentOptions {
        threshold: request.threshold.unwrap_or(defaults.threshold),
        neg_threshold: request.neg_threshold.or(defaults.neg_threshold),
        min_speech_duration_ms: request
            .min_speech_duration_ms
            .unwrap_or(defaults.min_speech_duration_ms),
        max_speech_duration_s: request
            .max_speech_duration_s
            .or(defaults.max_speech_duration_s),
        min_silence_duration_ms: request
            .min_silence_duration_ms
            .unwrap_or(defaults.min_silence_duration_ms),
        speech_pad_ms: request.speech_pad_ms.unwrap_or(defaults.speech_pad_ms),
        ..defaults
    };
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let messages = RequestMessage::VoiceActivity { audio, options };
    match run(state, &request.model, repr, messages).await? {
        Response::VoiceActivity(activity) => Ok(voice_activity_response(
            activity,
            request.return_probabilities,
        )),
        _ => Err(unexpected(state)),
    }
}

// durations and pads are lengths of time; a negative one would invert segments the reference never produces
fn validate_segment_request(request: &VoiceActivityRequest) -> Result<(), ApiError> {
    let lengths = [
        ("min_speech_duration_ms", request.min_speech_duration_ms),
        ("min_silence_duration_ms", request.min_silence_duration_ms),
        ("speech_pad_ms", request.speech_pad_ms),
    ];
    for (param, value) in lengths {
        if value.is_some_and(|v| v < 0.0) {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                format!("`{param}` must be zero or more"),
                Some("invalid_segment_option"),
                Some(param),
            ));
        }
    }
    if request.max_speech_duration_s.is_some_and(|v| v <= 0.0) {
        return Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            "`max_speech_duration_s` must be more than zero",
            Some("invalid_segment_option"),
            Some("max_speech_duration_s"),
        ));
    }
    Ok(())
}

fn voice_activity_response(activity: VoiceActivity, probabilities: bool) -> VoiceActivityResponse {
    VoiceActivityResponse {
        duration: activity.duration,
        segments: activity
            .segments
            .iter()
            .map(|s| SpeechSegment {
                start: s.start,
                end: s.end,
            })
            .collect(),
        chunk_seconds: activity.chunk_seconds,
        probabilities: probabilities.then_some(activity.probabilities),
    }
}

fn decode_audio(audio: &[u8]) -> Result<AudioInput, ApiError> {
    AudioInput::from_bytes(audio).map_err(|e| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!("The audio could not be decoded: {e}"),
            Some("invalid_audio"),
            Some("file"),
        )
    })
}

fn encode_speech(
    format: AudioResponseFormat,
    pcm: &[f32],
    rate: usize,
    channels: usize,
) -> Result<Vec<u8>, ApiError> {
    Ok(match format {
        AudioResponseFormat::Pcm => pcm
            .iter()
            .flat_map(|sample| sample.to_i16().to_le_bytes())
            .collect(),
        AudioResponseFormat::Wav => {
            let mut buf = Vec::new();
            speech_utils::write_pcm_as_wav(&mut buf, pcm, rate as u32, channels as u16)
                .map_err(|_| ApiError::internal())?;
            buf
        }
        _ => unreachable!("only wav and pcm pass validation"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_is_little_endian_i16_and_wav_carries_a_riff_header() {
        let pcm = [0.0_f32, 1.0, -1.0];
        let bytes = encode_speech(AudioResponseFormat::Pcm, &pcm, 24_000, 1).unwrap();
        let samples: Vec<i16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| i16::from_le_bytes(*pair))
            .collect();
        assert_eq!(samples, [0, i16::MAX, -i16::MAX]);

        let wav = encode_speech(AudioResponseFormat::Wav, &pcm, 24_000, 1).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }

    fn word(text: &str, start: f64, end: f64) -> TimedText {
        TimedText {
            text: text.into(),
            start,
            end,
        }
    }

    // a sentence end closes a segment, and so does a pause of SEGMENT_PAUSE_SECONDS mid-sentence
    #[test]
    fn segments_close_at_sentence_ends_and_long_pauses() {
        let words = [
            word("Well,", 0.32, 0.56),
            word("hello.", 0.64, 1.0),
            word("Then", 1.1, 1.3),
            word("after", 3.0, 3.2),
            word("a", 3.2, 3.3),
            word("pause?", 3.3, 3.6),
        ];
        let segments = transcript_segments(&words);
        let spans: Vec<(&str, f64, f64)> = segments
            .iter()
            .map(|s| (s.text.as_str(), s.start, s.end))
            .collect();
        assert_eq!(
            spans,
            [
                ("Well, hello.", 0.32, 1.0),
                ("Then", 1.1, 1.3),
                ("after a pause?", 3.0, 3.6)
            ]
        );
        assert_eq!(segments.iter().map(|s| s.id).collect::<Vec<_>>(), [0, 1, 2]);
    }

    // unpunctuated speech with no pause still breaks into segments of at most MAX_SEGMENT_SECONDS
    #[test]
    fn unpunctuated_speech_is_cut_into_bounded_segments() {
        let words: Vec<TimedText> = (0..100)
            .map(|i| word("word", i as f64, i as f64 + 0.9))
            .collect();
        let segments = transcript_segments(&words);
        assert!(segments.len() > 1);
        assert!(
            segments
                .iter()
                .all(|s| s.end - s.start <= MAX_SEGMENT_SECONDS)
        );
    }
}
