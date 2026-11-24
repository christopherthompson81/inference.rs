//! Control vectors derived on the loaded model from positive and negative texts, as llama.cpp's `cvector-generator`
//! `mean` method does, and kept in a cache keyed on the model and the texts.

use std::path::{Path, PathBuf};

use inference_core::{Request, RequestMessage, Response, SamplingParams};

use crate::api_error::ApiError;
use crate::operations::{DerivedControlVector, TokenizeRequest};
use crate::types::SharedInferenceRsState;

/// Where derived vectors are kept, in place of `~/.cache/inference-rs/control-vectors`.
pub const CONTROL_VECTOR_DIR_ENV: &str = "INFERENCE_RS_CONTROL_VECTOR_DIR";
const CACHE_SUBDIR: &str = "inference-rs/control-vectors";
// llama.cpp pads a pair's shorter prompt with the token for a space
const PAD_TEXT: &str = " ";
// a derivation prompt is scored in one step; longer ones would split across the scheduler's prompt chunks
const MAX_PROMPT_TOKENS: usize = 512;
const MEAN: &str = "mean";
// bumping this derives every cached vector again
const DERIVE_VERSION: u32 = 1;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn derive_error(message: impl std::fmt::Display) -> ApiError {
    ApiError::invalid_request(format!("deriving a control vector: {message}"))
}

fn cache_dir() -> Option<PathBuf> {
    std::env::var_os(CONTROL_VECTOR_DIR_ENV)
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
                .map(|cache| cache.join(CACHE_SUBDIR))
        })
}

fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, b| (h ^ u64::from(*b)).wrapping_mul(FNV_PRIME))
}

// the cached file of `derived` on the model named `model`: its texts, method and the model key it
fn cached_path(model: &str, derived: &DerivedControlVector) -> Option<PathBuf> {
    let mut hash = fnv(FNV_OFFSET, &DERIVE_VERSION.to_le_bytes());
    hash = fnv(hash, model.as_bytes());
    hash = fnv(hash, MEAN.as_bytes());
    for (side, texts) in [(b'+', &derived.positive), (b'-', &derived.negative)] {
        for text in texts {
            hash = fnv(fnv(hash, &[side]), text.as_bytes());
            hash = fnv(hash, &[0]);
        }
    }
    Some(cache_dir()?.join(format!("{hash:016x}.gguf")))
}

async fn tokens(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<u32>, ApiError> {
    Ok(crate::operations::tokenize(
        state,
        TokenizeRequest {
            model: model.map(str::to_string),
            text: text.to_string(),
            add_special_tokens,
        },
    )
    .await?
    .tokens)
}

// `tokens`' layer outputs, unsteered, from one scoring step of the model
async fn layer_outputs(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    tokens: Vec<u32>,
) -> Result<Vec<inference_tensor::Tensor>, ApiError> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let mut request = inference_core::NormalRequest::new_simple(
        RequestMessage::CompletionTokens(tokens),
        SamplingParams {
            max_len: Some(1),
            ..SamplingParams::deterministic()
        },
        tx,
        state.next_request_id(),
        None,
        None,
    );
    request.return_raw_logits = true;
    request.capture_layer_outputs = true;
    crate::dispatch::send_request_with_model(state, Request::Normal(Box::new(request)), model)
        .await
        .map_err(derive_error)?;
    match rx.recv().await {
        Some(Response::Raw { logits_chunks, .. }) if !logits_chunks.is_empty() => Ok(logits_chunks),
        Some(Response::Raw { .. }) => Err(derive_error("this model records no layer outputs")),
        Some(Response::ValidationError(error)) | Some(Response::InternalError(error)) => {
            Err(derive_error(error))
        }
        Some(_) | None => Err(ApiError::internal()),
    }
}

/// The file `derived` is kept in for the model named `model_name`, derived first when the cache lacks it.
pub(crate) async fn derived_file(
    state: &SharedInferenceRsState,
    model: Option<&str>,
    model_name: &str,
    derived: &DerivedControlVector,
) -> Result<PathBuf, ApiError> {
    if derived
        .method
        .as_deref()
        .is_some_and(|method| method != MEAN)
    {
        return Err(derive_error(format!(
            "method `{}`: only `{MEAN}` derives vectors",
            derived.method.as_deref().unwrap_or_default()
        )));
    }
    if derived.positive.is_empty() || derived.positive.len() != derived.negative.len() {
        return Err(derive_error(format!(
            "{} positive and {} negative texts; each positive text pairs with the negative one at its index",
            derived.positive.len(),
            derived.negative.len()
        )));
    }
    let path = cached_path(model_name, derived).ok_or_else(|| {
        derive_error(format!(
            "no cache directory: set `{CONTROL_VECTOR_DIR_ENV}`"
        ))
    })?;
    if path.is_file() {
        return Ok(path);
    }
    let pad = *tokens(state, model, PAD_TEXT, false)
        .await?
        .last()
        .ok_or_else(|| derive_error("the tokenizer gives no token for a space"))?;
    let mut pairs = Vec::with_capacity(derived.positive.len());
    for (positive, negative) in derived.positive.iter().zip(&derived.negative) {
        let mut positive = tokens(state, model, positive, true).await?;
        let mut negative = tokens(state, model, negative, true).await?;
        let len = positive.len().max(negative.len());
        if len > MAX_PROMPT_TOKENS {
            return Err(derive_error(format!(
                "a text of {len} tokens; each is at most {MAX_PROMPT_TOKENS}"
            )));
        }
        positive.resize(len, pad);
        negative.resize(len, pad);
        pairs.push((
            layer_outputs(state, model, positive).await?,
            layer_outputs(state, model, negative).await?,
        ));
    }
    let directions =
        inference_core::control_vectors::mean_directions(&pairs).map_err(derive_error)?;
    if directions.is_empty() {
        return Err(derive_error("the texts' layer outputs do not differ"));
    }
    write(&path, &directions, model_name)?;
    Ok(path)
}

fn write(path: &Path, directions: &[(usize, Vec<f32>)], model_name: &str) -> Result<(), ApiError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(derive_error)?;
    }
    inference_core::control_vectors::write_control_vector(path, directions, model_name)
        .map_err(derive_error)
}
