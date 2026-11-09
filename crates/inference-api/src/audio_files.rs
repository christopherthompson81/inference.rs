//! Audio files probed and decoded without an engine: what a file holds, and a WAV copy of it (resampled, downmixed)
//! written as it decodes, for playback and waveforms.

use std::path::PathBuf;

use crate::{api_error::ApiError, request_body::parse_json};

// the decode's output rates, as the transcription session takes them
const MIN_SAMPLE_RATE: u32 = 8_000;
const MAX_SAMPLE_RATE: u32 = 192_000;

#[derive(Debug, inference_json::FromJson)]
#[serde(deny_unknown_fields)]
struct ProbeRequest {
    path: PathBuf,
}

#[derive(Debug, inference_json::FromJson)]
#[serde(deny_unknown_fields)]
struct DecodeRequest {
    path: PathBuf,
    out_path: PathBuf,
    #[serde(default)]
    sample_rate: Option<u32>,
    /// 1 downmixes; unset (or the file's count) keeps them.
    #[serde(default)]
    channels: Option<u16>,
    /// `s16` (default) or `f32`.
    #[serde(default)]
    sample_format: Option<String>,
}

/// `{"path"}`'s `{"duration", "sample_rate", "channels", "codec"}`, from its container's header; `duration` is null
/// where the container does not state it.
pub fn audio_probe_json(request: &[u8]) -> Result<String, ApiError> {
    let request: ProbeRequest = parse_json(request)?;
    let info = inference_audio::probe_file(&request.path).map_err(unreadable)?;
    Ok(serde_json::json!({
        "duration": info.duration,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "codec": info.codec,
    })
    .to_string())
}

/// Decodes `{"path"}` into a WAV at `"out_path"` as it reads, at `"sample_rate"`? (default the file's), `"channels"`?
/// (1 downmixes), `"sample_format"`? (`s16` or `f32`); `{"path", "duration", "sample_rate", "channels", "frames"}`
/// of the WAV written.
pub fn audio_decode_json(request: &[u8]) -> Result<String, ApiError> {
    let request: DecodeRequest = parse_json(request)?;
    if request
        .sample_rate
        .is_some_and(|rate| !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&rate))
    {
        return Err(ApiError::invalid_request(format!(
            "`sample_rate` must be {MIN_SAMPLE_RATE} to {MAX_SAMPLE_RATE}"
        )));
    }
    let float = match request.sample_format.as_deref() {
        None | Some("s16") => false,
        Some("f32") => true,
        Some(other) => {
            return Err(ApiError::invalid_request(format!(
                "`sample_format` `{other}` is not s16 or f32"
            )));
        }
    };
    let file_channels = inference_audio::probe_file(&request.path)
        .map_err(unreadable)?
        .channels;
    let mono = match request.channels {
        None => false,
        Some(1) => true,
        Some(n) if n == file_channels => false,
        Some(n) => {
            return Err(ApiError::invalid_request(format!(
                "`channels` {n}: a decode keeps the file's {file_channels} or downmixes to 1"
            )));
        }
    };
    let wav = inference_audio::decode_file_to_wav(
        &request.path,
        &request.out_path,
        request.sample_rate,
        mono,
        float,
    )
    .map_err(unreadable)?;
    Ok(serde_json::json!({
        "path": request.out_path,
        "duration": wav.frames as f64 / f64::from(wav.sample_rate),
        "sample_rate": wav.sample_rate,
        "channels": wav.channels,
        "frames": wav.frames,
    })
    .to_string())
}

fn unreadable(error: anyhow::Error) -> ApiError {
    ApiError::invalid_request(format!("The audio could not be read: {error:#}"))
}
