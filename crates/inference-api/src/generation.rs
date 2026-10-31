//! Image and speech generation as engine operations, free of HTTP.

use base64::Engine as _;
use futures::future::BoxFuture;
use inference_core::{
    AudioInput, Diarization, DiarizationOptions, DiffusionGenerationParams, ImageChoice,
    ImageGenerationResponse, ImageGenerationResponseFormat, InferenceRs, MAX_REFERENCE_IMAGES,
    NormalRequest, ReferenceImages, Request, RequestMessage, Response, SamplingParams,
    SegmentOptions, SpeakerSegment, SpeechOptions, TimedText, Transcription, TranscriptionOptions,
    VoiceActivity,
    speech_utils::{self, Sample},
};

use inference_core::AudioTranscriber;
use inference_protocol::images::{encode_png, image_generation_response};

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::{base_process_non_streaming_response, create_response_channel, send_request},
    files::store_generated_image,
    lora_routing::DEFAULT_MODEL_ID,
    media_source::MediaAttachments,
    openai::{
        AudioResponseFormat, DiarizationOutput, DiarizationRequest, DiarizationResponse,
        DiarizationResponseFormat, DiarizationSegment, ImageEditRequest, ImageGenerationRequest,
        SpeechGenerationRequest, SpeechSegment, SpeechWithTimestamps, SpeechWord,
        TimestampGranularity, TranscriptionOutput, TranscriptionRequest, TranscriptionResponse,
        TranscriptionResponseFormat, TranscriptionSegment, TranscriptionWord,
        VerboseTranscriptionResponse, VoiceActivityRequest, VoiceActivityResponse,
        diarization_rttm, transcript_srt, transcript_vtt,
    },
    operations::TokenizeRequest,
    types::SharedInferenceRsState,
    util::{parse_image_url, validate_model_name},
};

// a transducer beam search costs a joint evaluation per hypothesis per frame; past this it buys nothing
const MAX_BEAM_SIZE: usize = 16;
// an audio-in decoder's transcript budget: its answer prefix, then tokens per second of audio
const TRANSCRIPT_BASE_TOKENS: usize = 64;
const TRANSCRIPT_TOKENS_PER_SECOND: f64 = 16.0;
// the training schedules count 1000 timesteps, the diffusion crates' `NUM_TRAIN_TIMESTEPS`
const MAX_DIFFUSION_STEPS: usize = 1000;
// OpenAI's own limit on images per request
const MAX_IMAGE_CHOICES: usize = 10;

const TRANSCRIBE_TASK: &str = "transcribe";
const SENTENCE_ENDS: [char; 4] = ['.', '?', '!', '\u{3002}'];
// a silence this long starts a new subtitle even mid-sentence
const SEGMENT_PAUSE_SECONDS: f64 = 1.5;
// unpunctuated transcripts (the English CTC and RNN-T checkpoints) still break into subtitle-sized segments
const MAX_SEGMENT_SECONDS: f64 = 30.0;

// Speech models emit f32 samples; PCM output is signed 16-bit little-endian.
const PCM_SAMPLE_FORMAT: &str = "s16le";
const JSON_CONTENT_TYPE: &str = "application/json";
const UNEXPECTED_RESPONSE: &str =
    "the engine answered a generation request with another kind of response";

/// Encoded speech audio and its MIME type, which carries the sample rate and channel count.
pub struct SpeechAudio {
    /// The audio, or with `timestamps` the `SpeechWithTimestamps` JSON carrying it.
    pub bytes: Vec<u8>,
    pub content_type: String,
    /// When each input word is spoken, from a model that can tell.
    pub words: Option<Vec<SpeechWord>>,
}

// Generation requests are one-shot and deterministic: no tools, streaming or sampling choices.
fn generation_request(
    state: &SharedInferenceRsState,
    model: &str,
    messages: RequestMessage,
    sampling: SamplingParams,
    tx: tokio::sync::mpsc::Sender<Response>,
) -> Request {
    let mut request =
        NormalRequest::new_simple(messages, sampling, tx, state.next_request_id(), None, None);
    request.model_id = (model != DEFAULT_MODEL_ID).then(|| model.to_string());
    Request::Normal(Box::new(request))
}

// Sends a generation request and waits for its one final response.
pub(crate) async fn run(
    state: &SharedInferenceRsState,
    model: &str,
    repr: String,
    messages: RequestMessage,
) -> Result<Response, ApiError> {
    run_choices(state, model, repr, messages, 1).await
}

// As `run`, for a request the engine answers with `n_choices` results in its one response.
async fn run_choices(
    state: &SharedInferenceRsState,
    model: &str,
    repr: String,
    messages: RequestMessage,
    n_choices: usize,
) -> Result<Response, ApiError> {
    InferenceRs::maybe_log_request(state.clone(), repr);
    let sampling = SamplingParams {
        n_choices,
        ..SamplingParams::deterministic()
    };
    dispatch(state, model, messages, sampling).await
}

// As `run`, for a request too frequent to log, as a streamed session's pushes are
pub(crate) async fn run_unlogged(
    state: &SharedInferenceRsState,
    model: &str,
    messages: RequestMessage,
) -> Result<Response, ApiError> {
    dispatch(state, model, messages, SamplingParams::deterministic()).await
}

async fn dispatch(
    state: &SharedInferenceRsState,
    model: &str,
    messages: RequestMessage,
    sampling: SamplingParams,
) -> Result<Response, ApiError> {
    validate_model_name(model, state.clone())
        .map_err(|error| ApiError::from_error(&error, ApiErrorKind::InvalidRequest))?;
    let (tx, mut rx) = create_response_channel(None);
    send_request(
        state,
        generation_request(state, model, messages, sampling, tx),
    )
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

pub(crate) fn unexpected(state: &SharedInferenceRsState) -> ApiError {
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
    let options = ImageOptions {
        model: &request.model,
        n_choices: request.n_choices,
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        response_format: request.response_format,
    };
    options.validate()?;
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let params = DiffusionGenerationParams {
        height: request.height,
        width: request.width,
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        seed: request.seed,
        ..Default::default()
    };
    run_image_request(state, &options, repr, request.prompt.clone(), params, owner).await
}

/// Edits an image, or places subject references in a new scene, with a diffusion model that takes references.
pub(crate) fn edit_image<'a>(
    state: &'a SharedInferenceRsState,
    request: ImageEditRequest,
    media: MediaAttachments,
    owner: Option<&'a str>,
) -> BoxFuture<'a, Result<ImageGenerationResponse, ApiError>> {
    Box::pin(edit_image_inner(state, request, media, owner))
}

async fn edit_image_inner(
    state: &SharedInferenceRsState,
    request: ImageEditRequest,
    media: MediaAttachments,
    owner: Option<&str>,
) -> Result<ImageGenerationResponse, ApiError> {
    let options = ImageOptions {
        model: &request.model,
        n_choices: request.n_choices,
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        response_format: request.response_format,
    };
    options.validate()?;
    // checked before any image is fetched
    if !(1..=MAX_REFERENCE_IMAGES).contains(&request.images.len()) {
        return Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!("`images` must hold 1 to {MAX_REFERENCE_IMAGES} images"),
            Some("invalid_images"),
            Some("images"),
        ));
    }
    let mut images = Vec::with_capacity(request.images.len());
    for (i, image) in request.images.iter().enumerate() {
        let image = parse_image_url(&image.image_url, &media)
            .await
            .map_err(|e| {
                ApiError::new(
                    ApiErrorKind::InvalidRequest,
                    format!("image {i}: {e}"),
                    Some("invalid_image"),
                    Some("images"),
                )
            })?;
        images.push(image);
    }
    // no size follows the first image: one image keeps its own size, several take its aspect
    let (width, height, match_reference) = match (request.width, request.height) {
        (Some(width), Some(height)) => (width, height, false),
        (None, None) => {
            let first = &images[0];
            (
                first.width() as usize,
                first.height() as usize,
                images.len() == 1,
            )
        }
        _ => {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                "give both `width` and `height`, or neither".to_string(),
                Some("invalid_size"),
                Some(if request.width.is_none() {
                    "width"
                } else {
                    "height"
                }),
            ));
        }
    };
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let params = DiffusionGenerationParams {
        height,
        width,
        references: ReferenceImages(std::sync::Arc::new(images)),
        match_reference,
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        seed: request.seed,
    };
    run_image_request(state, &options, repr, request.prompt.clone(), params, owner).await
}

/// What generation and edit requests share.
struct ImageOptions<'a> {
    model: &'a str,
    n_choices: usize,
    steps: Option<usize>,
    guidance_scale: Option<f64>,
    response_format: ImageGenerationResponseFormat,
}

impl ImageOptions<'_> {
    fn validate(&self) -> Result<(), ApiError> {
        if !(1..=MAX_IMAGE_CHOICES).contains(&self.n_choices) {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                format!("`n` must be 1 to {MAX_IMAGE_CHOICES}"),
                Some("invalid_n"),
                Some("n"),
            ));
        }
        if self
            .steps
            .is_some_and(|s| !(1..=MAX_DIFFUSION_STEPS).contains(&s))
        {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                format!("`steps` must be 1 to {MAX_DIFFUSION_STEPS}"),
                Some("invalid_steps"),
                Some("steps"),
            ));
        }
        if self
            .guidance_scale
            .is_some_and(|g| !g.is_finite() || g < 0.0)
        {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                "`guidance_scale` must be a finite number of at least 0".to_string(),
                Some("invalid_guidance_scale"),
                Some("guidance_scale"),
            ));
        }
        Ok(())
    }
}

async fn run_image_request(
    state: &SharedInferenceRsState,
    options: &ImageOptions<'_>,
    repr: String,
    prompt: String,
    generation_params: DiffusionGenerationParams,
    owner: Option<&str>,
) -> Result<ImageGenerationResponse, ApiError> {
    let messages = RequestMessage::ImageGeneration {
        prompt,
        generation_params,
    };
    let Response::ImageGeneration(generated) =
        run_choices(state, options.model, repr, messages, options.n_choices).await?
    else {
        return Err(unexpected(state));
    };
    let encode_error = |e: anyhow::Error| {
        InferenceRs::maybe_log_error(state.clone(), e.as_ref());
        ApiError::from_error(e.as_ref(), ApiErrorKind::Internal)
    };
    let response = match options.response_format {
        ImageGenerationResponseFormat::Url => {
            let model = (options.model != DEFAULT_MODEL_ID).then_some(options.model);
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
            words,
        } => {
            let audio = encode_speech(format, &pcm, rate, channels)?;
            let content_type = format.audio_content_type(rate, channels, PCM_SAMPLE_FORMAT);
            let words = words.map(|words| {
                words
                    .into_iter()
                    .map(|w| SpeechWord {
                        word: w.text,
                        ipa: w.ipa,
                        start: w.start,
                        end: w.end,
                    })
                    .collect::<Vec<_>>()
            });
            if !request.timestamps {
                return Ok(SpeechAudio {
                    bytes: audio,
                    content_type,
                    words,
                });
            }
            let body = SpeechWithTimestamps {
                audio: base64::engine::general_purpose::STANDARD.encode(&audio),
                content_type,
                duration: pcm.len() as f64 / (rate * channels.max(1)) as f64,
                words: words.clone().unwrap_or_default(),
            };
            Ok(SpeechAudio {
                bytes: serde_json::to_vec(&body).map_err(|_| ApiError::internal())?,
                content_type: JSON_CONTENT_TYPE.to_string(),
                words,
            })
        }
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
    if request
        .beam_size
        .is_some_and(|b| !(1..=MAX_BEAM_SIZE).contains(&b))
    {
        return Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!("`beam_size` must be 1 to {MAX_BEAM_SIZE}"),
            Some("invalid_beam_size"),
            Some("beam_size"),
        ));
    }
    let audio = decode_audio(audio)?;
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let model = (request.model != DEFAULT_MODEL_ID).then_some(request.model.as_str());
    let transcriber = state
        .audio_transcriber(model)
        .map_err(|error| ApiError::from_error(&error, ApiErrorKind::InvalidRequest))?;
    if let Some(transcriber) = transcriber {
        if request.diarization_model.is_some() {
            return Err(ApiError::new(
                ApiErrorKind::InvalidRequest,
                "`diarization_model` needs word timestamps, which this model does not give",
                Some("diarization_unsupported"),
                Some("diarization_model"),
            ));
        }
        let transcript = decode_transcript(state, &request, repr, audio, transcriber).await?;
        return render_transcript(&request, transcript, None);
    }
    let speech = |audio| RequestMessage::Transcription {
        audio,
        options: TranscriptionOptions {
            language: request.language.clone(),
            beam_size: request.beam_size,
        },
    };
    let Some(diarizer) = request.diarization_model.clone() else {
        return match run(state, &request.model, repr, speech(audio)).await? {
            Response::Transcription(transcript) => render_transcript(&request, transcript, None),
            _ => Err(unexpected(state)),
        };
    };
    // one upload, both models over it at once
    let diarization_request = DiarizationRequest {
        model: diarizer.clone(),
        threshold: None,
        response_format: Default::default(),
        return_probabilities: false,
    };
    let diarization_repr =
        serde_json::to_string(&diarization_request).map_err(|_| ApiError::internal())?;
    let turns = RequestMessage::Diarization {
        audio: audio.clone(),
        options: DiarizationOptions { threshold: None },
    };
    let (transcript, diarization) = futures::join!(
        run(state, &request.model, repr, speech(audio)),
        run(state, &diarizer, diarization_repr, turns),
    );
    match (transcript?, diarization?) {
        (Response::Transcription(transcript), Response::Diarization(diarization)) => {
            let speakers = word_speakers(&transcript.words, &diarization.segments);
            render_transcript(&request, transcript, Some(&speakers))
        }
        _ => Err(unexpected(state)),
    }
}

// An audio-in decoder's transcript: its transcription prompt around the clip, decoded greedily, the answer parsed.
async fn decode_transcript(
    state: &SharedInferenceRsState,
    request: &TranscriptionRequest,
    repr: String,
    audio: AudioInput,
    transcriber: AudioTranscriber,
) -> Result<Transcription, ApiError> {
    let prompt = (transcriber.prompt)(request.language.as_deref()).map_err(|error| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            error.to_string(),
            Some("unsupported_language"),
            Some("language"),
        )
    })?;
    let tokens = crate::operations::tokenize(
        state,
        TokenizeRequest {
            text: prompt,
            add_special_tokens: false,
            model: (request.model != DEFAULT_MODEL_ID).then(|| request.model.clone()),
        },
    )
    .await?
    .tokens;
    let duration = audio.samples.len() as f64
        / (f64::from(audio.sample_rate) * f64::from(audio.channels.max(1)));
    let sampling = SamplingParams {
        max_len: Some(transcript_token_budget(duration)),
        ..SamplingParams::deterministic()
    };
    let message = RequestMessage::MultimodalTokens {
        tokens,
        images: Vec::new(),
        audios: vec![audio],
    };
    InferenceRs::maybe_log_request(state.clone(), repr);
    let Response::CompletionDone(done) = dispatch(state, &request.model, message, sampling).await?
    else {
        return Err(unexpected(state));
    };
    let answer = done
        .choices
        .first()
        .map(|choice| choice.text.as_str())
        .unwrap_or_default();
    let parsed = (transcriber.parse)(answer);
    Ok(Transcription {
        text: parsed.text,
        tokens: Vec::new(),
        words: Vec::new(),
        duration,
        language: parsed.language,
    })
}

// generous for speech: a fast speaker's words run under four tokens a second
fn transcript_token_budget(seconds: f64) -> usize {
    TRANSCRIPT_BASE_TOKENS + (seconds.max(0.0) * TRANSCRIPT_TOKENS_PER_SECOND).ceil() as usize
}

// each word's speaker: the turn it overlaps most, else the turn nearest its middle; 0 without any turns
fn word_speakers(words: &[TimedText], turns: &[SpeakerSegment]) -> Vec<usize> {
    words
        .iter()
        .map(|word| {
            let middle = (word.start + word.end) / 2.0;
            let score = |turn: &SpeakerSegment| {
                let overlap = word.end.min(turn.end) - word.start.max(turn.start);
                let gap = (turn.start - middle).max(middle - turn.end).max(0.0);
                if overlap > 0.0 { overlap } else { -gap }
            };
            turns
                .iter()
                .max_by(|a, b| score(a).total_cmp(&score(b)))
                .map_or(0, |turn| turn.speaker)
        })
        .collect()
}

/// `speakers` gives each word's diarized speaker, when the request diarized.
pub(crate) fn render_transcript(
    request: &TranscriptionRequest,
    transcript: Transcription,
    speakers: Option<&[usize]>,
) -> Result<TranscriptionOutput, ApiError> {
    let format = request.response_format;
    // a model without word timings gives its whole transcript as one segment
    let segments = if transcript.words.is_empty() && !transcript.text.is_empty() {
        vec![TranscriptionSegment {
            id: 0,
            start: 0.0,
            end: transcript.duration,
            text: transcript.text.clone(),
            speaker: None,
        }]
    } else {
        transcript_segments(&transcript.words, speakers)
    };
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
                        .enumerate()
                        .map(|(i, w)| TranscriptionWord {
                            word: w.text.clone(),
                            start: w.start,
                            end: w.end,
                            speaker: speakers.map(|speakers| speakers[i]),
                        })
                        .collect()
                });
            serde_json::to_string(&VerboseTranscriptionResponse {
                task: TRANSCRIBE_TASK.to_string(),
                // the model's own, where it identifies one; otherwise a language the client named, echoed
                language: transcript
                    .language
                    .clone()
                    .or_else(|| request.language.clone()),
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

// a segment ends after a word closing a sentence, before a pause long enough to be one, and where the speaker changes
fn transcript_segments(
    words: &[TimedText],
    speakers: Option<&[usize]>,
) -> Vec<TranscriptionSegment> {
    let mut segments: Vec<TranscriptionSegment> = Vec::new();
    let mut open = false;
    for (i, word) in words.iter().enumerate() {
        let speaker = speakers.map(|speakers| speakers[i]);
        let pause = segments.last().is_some_and(|s| {
            word.start - s.end >= SEGMENT_PAUSE_SECONDS
                || word.end - s.start > MAX_SEGMENT_SECONDS
                || s.speaker != speaker
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
                speaker,
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

/// Who speaks when in encoded `audio`, from a diarization model, as JSON or RTTM.
pub(crate) fn diarize<'a>(
    state: &'a SharedInferenceRsState,
    request: DiarizationRequest,
    audio: &'a [u8],
) -> BoxFuture<'a, Result<DiarizationOutput, ApiError>> {
    Box::pin(diarize_inner(state, request, audio))
}

async fn diarize_inner(
    state: &SharedInferenceRsState,
    request: DiarizationRequest,
    audio: &[u8],
) -> Result<DiarizationOutput, ApiError> {
    if request.threshold.is_some_and(|t| !(0.0..=1.0).contains(&t)) {
        return Err(ApiError::new(
            ApiErrorKind::InvalidRequest,
            "`threshold` must be a probability, from 0 to 1",
            Some("invalid_threshold"),
            Some("threshold"),
        ));
    }
    let audio = decode_audio(audio)?;
    let repr = serde_json::to_string(&request).map_err(|_| ApiError::internal())?;
    let messages = RequestMessage::Diarization {
        audio,
        options: DiarizationOptions {
            threshold: request.threshold,
        },
    };
    match run(state, &request.model, repr, messages).await? {
        Response::Diarization(diarization) => render_diarization(&request, diarization),
        _ => Err(unexpected(state)),
    }
}

fn render_diarization(
    request: &DiarizationRequest,
    diarization: Diarization,
) -> Result<DiarizationOutput, ApiError> {
    let segments: Vec<DiarizationSegment> = diarization
        .segments
        .iter()
        .map(|s| DiarizationSegment {
            speaker: s.speaker,
            start: s.start,
            end: s.end,
        })
        .collect();
    let format = request.response_format;
    let body = match format {
        DiarizationResponseFormat::Rttm => diarization_rttm(&segments),
        DiarizationResponseFormat::Json => {
            let probabilities = request.return_probabilities.then(|| {
                diarization
                    .probabilities
                    .chunks(diarization.num_speakers)
                    .map(<[f32]>::to_vec)
                    .collect()
            });
            serde_json::to_string(&DiarizationResponse {
                duration: diarization.duration,
                num_speakers: diarization.num_speakers,
                segments,
                frame_seconds: diarization.frame_seconds,
                probabilities,
            })
            .map_err(|_| ApiError::internal())?
        }
    };
    Ok(DiarizationOutput {
        body,
        content_type: format.content_type(),
    })
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

    // an audio-in decoder's transcript has no word timings: its subtitles are the whole clip as one cue
    #[test]
    fn a_transcript_without_words_is_one_segment() -> anyhow::Result<()> {
        let request = TranscriptionRequest {
            model: DEFAULT_MODEL_ID.to_string(),
            language: None,
            prompt: None,
            response_format: TranscriptionResponseFormat::Srt,
            temperature: None,
            timestamp_granularities: Vec::new(),
            beam_size: None,
            diarization_model: None,
        };
        let transcript = Transcription {
            text: "The quick brown fox.".to_string(),
            tokens: Vec::new(),
            words: Vec::new(),
            duration: 2.5,
            language: Some("english".to_string()),
        };
        let srt =
            render_transcript(&request, transcript, None).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        assert_eq!(
            srt.body.trim(),
            "1\n00:00:00,000 --> 00:00:02,500\nThe quick brown fox."
        );
        Ok(())
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
        let segments = transcript_segments(&words, None);
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

    // a word goes to the turn it overlaps most, a word in no turn to the nearest, and segments break at the change
    #[test]
    fn diarized_words_split_segments_by_speaker() {
        let words = [
            word("Hello", 0.0, 0.4),
            word("there", 0.5, 0.9),
            word("Hi", 1.0, 1.2),
            word("back.", 1.25, 1.6),
            word("Gap", 5.0, 5.2),
        ];
        let turns = [
            SpeakerSegment {
                speaker: 0,
                start: 0.0,
                end: 0.95,
            },
            SpeakerSegment {
                speaker: 1,
                start: 0.85,
                end: 1.7,
            },
            SpeakerSegment {
                speaker: 0,
                start: 6.0,
                end: 7.0,
            },
        ];
        let speakers = word_speakers(&words, &turns);
        assert_eq!(speakers, [0, 0, 1, 1, 0]);
        let segments = transcript_segments(&words, Some(&speakers));
        let spans: Vec<(&str, Option<usize>)> = segments
            .iter()
            .map(|s| (s.text.as_str(), s.speaker))
            .collect();
        assert_eq!(
            spans,
            [
                ("Hello there", Some(0)),
                ("Hi back.", Some(1)),
                ("Gap", Some(0))
            ]
        );
        assert!(
            inference_protocol::openai::transcript_vtt(&segments).contains("<v Speaker 1>Hi back.")
        );
        assert!(
            inference_protocol::openai::transcript_srt(&segments)
                .contains("Speaker 0: Hello there")
        );
    }

    // unpunctuated speech with no pause still breaks into segments of at most MAX_SEGMENT_SECONDS
    #[test]
    fn unpunctuated_speech_is_cut_into_bounded_segments() {
        let words: Vec<TimedText> = (0..100)
            .map(|i| word("word", i as f64, i as f64 + 0.9))
            .collect();
        let segments = transcript_segments(&words, None);
        assert!(segments.len() > 1);
        assert!(
            segments
                .iter()
                .all(|s| s.end - s.start <= MAX_SEGMENT_SECONDS)
        );
    }
}
