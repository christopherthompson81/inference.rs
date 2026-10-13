//! The models on this machine without an engine: inspect one, scan folders and the Hugging Face cache, delete from
//! the cache, and download from the Hub.

use std::path::PathBuf;
use std::time::Duration;

use futures::Stream;
use inference_core::{
    FileDownloadProgress, ModelDType, RequestCancellation, TokenSource, download_repo_files,
    list_model_files,
};
pub use inference_selection::inventory::{
    HfCacheDeletion, HfCacheListing, HfCacheRepo, ScanRequest, delete_hf_cache_repo,
    hf_cache_listing, inspect_model, scan_models,
};
use inference_selection::quant::{resolve_gguf_projector, resolve_gguf_quant};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;

use crate::api_error::ApiError;
use crate::request_body::{JsonRequest, parse_json};

const DEFAULT_REVISION: &str = "main";
const DOWNLOAD_THREAD: &str = "inference-download";
// a download's runtime: hf-hub fetches its chunks in parallel on whatever runtime it runs on
const DOWNLOAD_WORKERS: usize = 4;
// how often a running download looks for its cancellation
const CANCEL_POLL: Duration = Duration::from_millis(100);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInspectRequest {
    /// A model directory, GGUF (any split shard) or `.nemo` file.
    pub path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HfCacheDeleteRequest {
    /// `org/name`.
    pub repo_id: String,
    /// The cache to delete from instead of the environment's.
    #[serde(default)]
    pub hf_cache_path: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDownloadRequest {
    pub repo_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    /// Exactly these files. With no files and no `quant`, the whole repository.
    #[serde(default)]
    pub files: Vec<String>,
    /// A GGUF quantization the repository publishes (`Q4_K_M`, `4`): its files and the projector beside it.
    #[serde(default)]
    pub quant: Option<String>,
    /// `cache` (default), `none`, `literal:<token>` or `env:<variable>`.
    #[serde(default)]
    pub token_source: Option<String>,
    /// The cache to download into instead of the one loads use.
    #[serde(default)]
    pub hf_cache_path: Option<PathBuf>,
}

impl JsonRequest for ModelInspectRequest {
    fn from_json(body: &[u8]) -> Result<Self, ApiError> {
        parse_json(body)
    }
}

impl JsonRequest for ScanRequestBody {
    fn from_json(body: &[u8]) -> Result<Self, ApiError> {
        parse_json(body)
    }
}

impl JsonRequest for HfCacheDeleteRequest {
    fn from_json(body: &[u8]) -> Result<Self, ApiError> {
        parse_json(body)
    }
}

impl JsonRequest for ModelDownloadRequest {
    fn from_json(body: &[u8]) -> Result<Self, ApiError> {
        parse_json(body)
    }
}

#[derive(Debug, Deserialize)]
#[serde(transparent)]
pub struct ScanRequestBody(ScanRequest);

fn to_json(value: &impl Serialize) -> Result<String, ApiError> {
    serde_json::to_string(value).map_err(|_| ApiError::internal())
}

/// One model's facts: format, kind, architecture, quantization, size, base model and projectors.
pub fn model_inspect_json(request: &[u8]) -> Result<String, ApiError> {
    let request = ModelInspectRequest::from_json(request)?;
    let model = inspect_model(&request.path)
        .map_err(|error| ApiError::invalid_request(format!("{error:#}")))?;
    to_json(&model)
}

/// The models in the request's folders and, when asked, the Hugging Face cache.
pub fn models_scan_json(request: &[u8]) -> Result<String, ApiError> {
    let ScanRequestBody(request) = ScanRequestBody::from_json(request)?;
    to_json(&scan_models(&request))
}

/// Deletes a repository from the Hugging Face cache; the reply names it and the bytes freed.
pub fn hf_cache_delete_json(request: &[u8]) -> Result<String, ApiError> {
    let request = HfCacheDeleteRequest::from_json(request)?;
    let deleted = delete_hf_cache_repo(&request.repo_id, request.hf_cache_path.as_deref())
        .map_err(|error| ApiError::invalid_request(format!("{error:#}")))?;
    to_json(&deleted)
}

/// `org/name` (or a legacy bare `name`), as the Hub spells repository ids.
pub(crate) fn is_repo_id(source: &str) -> bool {
    let valid = |part: &str| {
        !part.is_empty()
            && !part.starts_with('.')
            && part
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    };
    let mut parts = source.split('/');
    let first = parts.next().is_some_and(valid);
    let second = parts.next().is_none_or(valid);
    first && second && parts.next().is_none()
}

// a path inside the repository: plain components only, no root, `.` or `..`
fn is_relative_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('\\')
        && std::path::Path::new(name)
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
}

fn event(name: &str, data: serde_json::Value) -> String {
    json!({ "event": name, "data": data }).to_string()
}

// the files `request` names: its list, its quant's GGUF and projector, or the whole repository
async fn download_files(
    request: &ModelDownloadRequest,
    revision: &str,
    token_source: &TokenSource,
) -> anyhow::Result<Vec<String>> {
    if request.quant.is_none() && !request.files.is_empty() {
        return Ok(request.files.clone());
    }
    let listed = list_model_files(&request.repo_id, revision, token_source, true).await?;
    let Some(quant) = request.quant.as_deref() else {
        return Ok(listed);
    };
    let mut files = resolve_gguf_quant(&listed, quant)?.files;
    if let Ok(Some(projector)) = resolve_gguf_projector(&listed, ModelDType::Auto) {
        files.extend(projector.files);
    }
    files.extend(request.files.iter().cloned());
    Ok(files)
}

/// Starts a download into the Hugging Face cache. Its events are `files` (data: the file names), `progress` (data:
/// `{"file", "done", "total"}` in bytes), then one of `done` (data: `{"repo_id", "revision", "files"}`, the local
/// paths), `canceled` or `error` (data: the error envelope).
/// The returned future does the download; the caller runs it on a runtime.
pub(crate) fn model_download(request: &[u8]) -> Result<ModelDownload, ApiError> {
    model_download_from(ModelDownloadRequest::from_json(request)?)
}

/// As [`model_download`], from a parsed request.
pub(crate) fn model_download_from(
    request: ModelDownloadRequest,
) -> Result<ModelDownload, ApiError> {
    // the cache's paths are built from these, so none may climb out of it
    if !is_repo_id(&request.repo_id) {
        return Err(ApiError::invalid_request(format!(
            "`{}` is not a Hugging Face repository id (`org/name`)",
            request.repo_id
        )));
    }
    let names = request.revision.iter().chain(&request.files);
    if let Some(bad) = names.into_iter().find(|name| !is_relative_name(name)) {
        return Err(ApiError::invalid_request(format!(
            "`{bad}` is not a plain revision or file name in the repository"
        )));
    }
    let token_source = match request.token_source.as_deref() {
        Some(value) => value
            .parse::<TokenSource>()
            .map_err(|error| ApiError::invalid_request(format!("invalid token_source: {error}")))?,
        None => TokenSource::CacheToken,
    };
    let (events, receiver) = mpsc::unbounded_channel();
    let cancellation = RequestCancellation::default();
    let canceled = cancellation.clone();
    let work = async move {
        let revision = request
            .revision
            .clone()
            .unwrap_or_else(|| DEFAULT_REVISION.to_string());
        let progress = events.clone();
        let work = async {
            let files = download_files(&request, &revision, &token_source).await?;
            let _ = events.send(event("files", json!(files)));
            download_repo_files(
                &request.repo_id,
                &revision,
                &files,
                &token_source,
                request.hf_cache_path.clone(),
                move |report: FileDownloadProgress| {
                    let _ = progress.send(event("progress", json!(report)));
                },
            )
            .await
        };
        let watch = async {
            while !canceled.is_canceled() {
                tokio::time::sleep(CANCEL_POLL).await;
            }
        };
        let last = tokio::select! {
            result = work => match result {
                Ok(paths) => event("done", json!({
                    "repo_id": request.repo_id,
                    "revision": revision,
                    "files": paths,
                })),
                Err(error) => event(
                    "error",
                    ApiError::invalid_request(format!("{error:#}")).to_openai_body(),
                ),
            },
            () = watch => event("canceled", serde_json::Value::Null),
        };
        let _ = events.send(last);
    };
    let events = futures::stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|event| (event, receiver))
    });
    Ok(ModelDownload {
        work: Box::pin(work),
        events: Box::pin(CancelOnDrop {
            events: Box::pin(events),
            cancellation: cancellation.clone(),
        }),
        cancellation,
    })
}

// a download's events; once nobody reads them, the download stops
struct CancelOnDrop {
    events: std::pin::Pin<Box<dyn Stream<Item = String> + Send>>,
    cancellation: RequestCancellation,
}

impl Stream for CancelOnDrop {
    type Item = String;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<String>> {
        self.events.as_mut().poll_next(cx)
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Runs a download's `work` to its end on a runtime of its own, on a thread of its own.
pub(crate) fn start_download(
    work: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>,
) -> Result<(), ApiError> {
    // hf-hub spawns each chunk as a task that outlives a dropped download; shutting down its own runtime ends them
    let downloader = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(DOWNLOAD_WORKERS)
        .enable_all()
        .thread_name(DOWNLOAD_THREAD)
        .build()
        .map_err(|_| ApiError::internal())?;
    std::thread::Builder::new()
        .name(DOWNLOAD_THREAD.to_string())
        .spawn(move || {
            downloader.block_on(work);
            downloader.shutdown_background();
        })
        .map_err(|_| ApiError::internal())?;
    Ok(())
}

/// A download not yet started: run `work`, and read `events` until they end.
pub(crate) struct ModelDownload {
    pub(crate) work: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>,
    pub(crate) events: std::pin::Pin<Box<dyn Stream<Item = String> + Send>>,
    pub(crate) cancellation: RequestCancellation,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn download(repo_id: &str, revision: Option<&str>, files: &[&str]) -> Result<(), ApiError> {
        model_download_from(ModelDownloadRequest {
            repo_id: repo_id.to_string(),
            revision: revision.map(str::to_string),
            files: files.iter().map(|file| file.to_string()).collect(),
            quant: None,
            token_source: None,
            hf_cache_path: None,
        })
        .map(drop)
    }

    #[test]
    fn a_download_names_nothing_outside_its_repository() {
        assert!(
            download(
                "org/model",
                Some("main"),
                &["config.json", "sub/model.gguf"]
            )
            .is_ok()
        );
        for (repo, revision, file) in [
            ("../org/model", None, "config.json"),
            ("org/model", Some("main/../../../../outside"), "config.json"),
            ("org/model", None, "../../outside"),
            ("org/model", None, "/etc/outside"),
            ("org/model", None, "./config.json"),
            ("org/model", None, "a\\..\\b"),
        ] {
            assert!(
                download(repo, revision, &[file]).is_err(),
                "{repo} {revision:?} {file}"
            );
        }
    }
}
