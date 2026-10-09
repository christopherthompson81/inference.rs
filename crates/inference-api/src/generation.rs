//! Image and speech generation as engine operations, free of HTTP.

use futures::future::BoxFuture;
use inference_core::{
    DiffusionGenerationParams, ImageChoice, ImageGenerationResponse, ImageGenerationResponseFormat,
    InferenceRs, NormalRequest, Request, RequestMessage, Response, SamplingParams, SpeechOptions,
    speech_utils::{self, Sample},
};

use inference_protocol::images::{encode_png, image_generation_response};

use crate::{
    api_error::{ApiError, ApiErrorKind, ModelErrorMessage},
    dispatch::{base_process_non_streaming_response, create_response_channel, send_request},
    files::store_generated_image,
    lora_routing::DEFAULT_MODEL_ID,
    openai::{AudioResponseFormat, ImageGenerationRequest, SpeechGenerationRequest},
    types::SharedInferenceRsState,
    util::validate_model_name,
};

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
}
