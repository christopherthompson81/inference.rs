//! The models a server loads and unloads on request: an inventory of its model folders and the Hugging Face cache,
//! loads from it into the engine (or into a document layout detector beside it) with their progress, downloads into
//! the cache and deletions.

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, RwLock};

use base64::Engine as _;
use inference_core::default_hf_cache_dir;
use inference_layout::pp_doclayout_v3::{DEFAULT_THRESHOLD, PPDocLayoutV3Detector};
use inference_selection::inventory::{InventoryKind, InventoryModel, ModelFormat, ModelScan};
use inference_tensor::Device;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api_error::{ApiError, ApiErrorKind};
use crate::engine::{Engine, ModelSpec, MtpSpec, ProgressSink, add_progress_sink, explicit_device};
use crate::model_inventory::{
    ModelDownloadRequest, ScanRequest, delete_hf_cache_repo, inspect_model, is_repo_id,
    model_download_from, scan_models, start_download,
};
use crate::models::{ModelOperationRequest, ModelStatus};
use crate::request_body::{JsonRequest, parse_json};
use inference_protocol::openai::{ModelCategory, ModelObject};

/// Largest image the layout detector takes, in pixels (a 600-dpi A3 page is ~70M); bigger ones are refused.
pub const MAX_LAYOUT_IMAGE_PIXELS: usize = 1 << 28;
const DATA_URL_BASE64: &str = ";base64,";
const SPLIT_FILE_SEPARATOR: &str = ";";
// the Hugging Face cache's repository dirs: `models--org--name`
const HF_REPO_DIR_PREFIX: &str = "models--";
const HF_REPO_SEPARATOR: &str = "--";
const LAYOUT_STATUS: ModelStatus = ModelStatus::Loaded;

/// A download's events, as JSON; dropping them stops it.
pub type DownloadEvents = std::pin::Pin<Box<dyn futures::Stream<Item = String> + Send>>;

/// Where a manager finds models and fetches them from.
#[derive(Clone, Debug, Default)]
pub struct ModelManagerConfig {
    /// Folders scanned for models; a load names a model in one of them, in the Hugging Face cache, or by repository.
    pub folders: Vec<PathBuf>,
    /// The Hugging Face cache downloads go into and loads read; unset takes the environment's.
    pub hf_cache_path: Option<PathBuf>,
    /// The token downloads use (`cache`, `env:VAR`, `literal:TOKEN`); unset uses the one `hf auth login` saved.
    pub token_source: Option<String>,
}

/// How a managed model serves requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ManagedKind {
    Text,
    Multimodal,
    Embedding,
    ImageGeneration,
    Speech,
    Transcription,
    VoiceActivity,
    Diarization,
    /// Document layout detection, through `POST /v1/layout/detect`.
    Layout,
}

impl From<ModelCategory> for ManagedKind {
    fn from(category: ModelCategory) -> Self {
        match category {
            ModelCategory::Text => Self::Text,
            ModelCategory::Multimodal => Self::Multimodal,
            ModelCategory::Diffusion => Self::ImageGeneration,
            ModelCategory::Transcription => Self::Transcription,
            ModelCategory::VoiceActivity => Self::VoiceActivity,
            ModelCategory::Diarization => Self::Diarization,
            ModelCategory::Speech => Self::Speech,
            ModelCategory::Embedding => Self::Embedding,
        }
    }
}

/// A load: what to load, the id requests will name it by, and how to run it.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ManagedLoadRequest {
    /// An inventory entry's `path`, or a Hugging Face repository id (`org/name`), downloaded when not cached; a layout
    /// model loads from the inventory.
    pub source: String,
    /// The id requests use; a model already holding it is unloaded first.
    pub model_id: String,
    /// Tokens of context the model's paged KV cache holds; unset uses the server's setting, and without paged attention
    /// it is ignored.
    #[serde(default)]
    pub paged_context_len: Option<usize>,
    /// Sequences the model serves at once; unset uses the server's `--max-seqs`.
    #[serde(default)]
    pub max_seqs: Option<usize>,
    /// Speculative decoding with the checkpoint's own MTP head (`n_predict`, `draft_sampling`); an assistant `model`
    /// is refused.
    #[serde(default)]
    pub mtp: Option<MtpSpec>,
    /// For a text GGUF, one of its entry's `projectors`; unset takes its only one, and none when there are several.
    #[serde(default)]
    pub mmproj: Option<String>,
    /// For a speech recognition model, a Silero VAD GGUF (an inventory `path`): recordings over 5 minutes are then cut
    /// at their silences, and run past the 24 minutes one pass takes.
    #[serde(default)]
    pub vad: Option<String>,
    /// For a layout detector, `cpu` or `cuda:N`; unset uses the engine's device. Other models run on the engine's.
    #[serde(default)]
    pub device: Option<String>,
}

/// A served model and where it came from.
#[derive(Clone, Debug, Serialize, utoipa::ToSchema)]
pub struct ManagedModel {
    pub model_id: String,
    pub kind: ManagedKind,
    /// The inventory path or repository it was loaded from; unset for a model the server started with.
    pub source: Option<String>,
    /// The projector GGUF loaded beside it.
    pub projector: Option<String>,
    /// The voice activity GGUF that cuts its long recordings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vad: Option<String>,
    pub status: ModelStatus,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ManagedModels {
    pub data: Vec<ManagedModel>,
}

/// A download into the server's Hugging Face cache.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ManagedDownloadRequest {
    pub repo_id: String,
    #[serde(default)]
    pub revision: Option<String>,
    /// Exactly these files. With no files and no `quant`, the whole repository.
    #[serde(default)]
    pub files: Vec<String>,
    /// A GGUF quantization the repository publishes (`Q4_K_M`, `4`): its files and the projector beside it.
    #[serde(default)]
    pub quant: Option<String>,
}

/// What to delete: a repository from the Hugging Face cache, or a GGUF in a model folder.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ManagedDeleteRequest {
    #[serde(default)]
    pub repo_id: Option<String>,
    /// A local GGUF entry's `path`; its split shards go with it.
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ManagedDeletion {
    /// The repository or path deleted.
    pub deleted: String,
    pub freed_bytes: u64,
}

/// A page to find the layout regions of.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct LayoutDetectRequest {
    /// The layout model's id.
    pub model: String,
    /// The page image (PNG, JPEG, ...) as base64 or a `data:` URL.
    pub image: String,
    /// Lowest score kept, 0 to 1.
    #[serde(default)]
    pub threshold: Option<f32>,
}

/// One region of a page, in its predicted reading order.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct LayoutRegion {
    /// The region's class (`text`, `table`, `formula`, `chart`, ...).
    pub label: String,
    pub score: f32,
    /// `[x1, y1, x2, y2]` in image pixels.
    pub bbox: [f32; 4],
    /// The region's outline in image pixels.
    pub polygon: Vec<[f32; 2]>,
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct LayoutDetectResponse {
    pub width: u32,
    pub height: u32,
    pub regions: Vec<LayoutRegion>,
}

macro_rules! json_requests {
    ($($request:ty),*) => {$(
        impl JsonRequest for $request {
            fn from_json(body: &[u8]) -> Result<Self, ApiError> {
                parse_json(body)
            }
        }
    )*};
}

json_requests!(
    ManagedLoadRequest,
    ManagedDownloadRequest,
    ManagedDeleteRequest,
    LayoutDetectRequest
);

struct Loaded {
    kind: ManagedKind,
    source: String,
    projector: Option<String>,
    vad: Option<String>,
}

struct LayoutModel {
    detector: Arc<PPDocLayoutV3Detector>,
    source: String,
}

// What a load names: a model the inventory lists, or a repository.
enum Source {
    Local(Box<InventoryModel>),
    Repo(String),
}

/// Loads, unloads, downloads and deletes the models of one engine, one load at a time.
pub struct ModelManager {
    engine: Engine,
    config: ModelManagerConfig,
    // one load at a time, so a load's progress events are its own
    gate: tokio::sync::Mutex<()>,
    loaded: RwLock<HashMap<String, Loaded>>,
    layouts: RwLock<HashMap<String, LayoutModel>>,
}

impl ModelManager {
    pub fn new(engine: Engine, config: ModelManagerConfig) -> Arc<Self> {
        Arc::new(Self {
            engine,
            config,
            gate: tokio::sync::Mutex::new(()),
            loaded: RwLock::new(HashMap::new()),
            layouts: RwLock::new(HashMap::new()),
        })
    }

    pub fn config(&self) -> &ModelManagerConfig {
        &self.config
    }

    /// The models in the folders and the Hugging Face cache.
    pub fn inventory(&self) -> ModelScan {
        scan_models(&ScanRequest {
            folders: self.config.folders.clone(),
            hf_cache: true,
            hf_cache_path: self.config.hf_cache_path.clone(),
        })
    }

    /// What is served: the engine's models, with where each was loaded from, and the layout models.
    pub fn models(&self) -> Result<ManagedModels, ApiError> {
        let loaded = self.loaded.read().map_err(|_| ApiError::internal())?;
        let mut data: Vec<ManagedModel> = self
            .engine
            .models()?
            .data
            .into_iter()
            .filter(is_model_card)
            .filter_map(|card| {
                // an unloaded model's card has no category; the manager knows the kind of those it loaded
                let entry = loaded.get(&card.id);
                let kind = entry
                    .map(|entry| entry.kind)
                    .or(card.category.map(Into::into))?;
                Some(ManagedModel {
                    kind,
                    source: entry.map(|entry| entry.source.clone()),
                    projector: entry.and_then(|entry| entry.projector.clone()),
                    vad: entry.and_then(|entry| entry.vad.clone()),
                    status: status_of(card.status.as_deref()),
                    model_id: card.id,
                })
            })
            .collect();
        let layouts = self.layouts.read().map_err(|_| ApiError::internal())?;
        data.extend(layouts.iter().map(|(id, layout)| ManagedModel {
            model_id: id.clone(),
            kind: ManagedKind::Layout,
            source: Some(layout.source.clone()),
            projector: None,
            vad: None,
            status: LAYOUT_STATUS,
        }));
        data.sort_by(|a, b| a.model_id.cmp(&b.model_id));
        Ok(ManagedModels { data })
    }

    /// Loads `request.source` as `request.model_id`, replacing what held that id; `progress` hears the load.
    pub async fn load(
        &self,
        request: ManagedLoadRequest,
        progress: Arc<ProgressSink>,
    ) -> Result<ManagedModel, ApiError> {
        let _gate = self.gate.lock().await;
        let id = request.model_id.trim().to_string();
        if id.is_empty() {
            return Err(ApiError::invalid_request("`model_id` is empty"));
        }
        if request.mtp.as_ref().is_some_and(|mtp| mtp.model.is_some()) {
            return Err(ApiError::invalid_request(
                "`mtp` drafts with the checkpoint's own head here; an assistant model is not loaded by the manager",
            ));
        }
        let source = self.resolve(&request.source)?;
        let kind = match &source {
            Source::Local(entry) => Some(loadable_kind(entry)?),
            Source::Repo(_) => None,
        };
        let projector = match (&source, &request.mmproj) {
            (Source::Local(entry), Some(mmproj)) if entry.projectors.contains(mmproj) => {
                Some(mmproj.clone())
            }
            (_, Some(mmproj)) => {
                return Err(ApiError::invalid_request(format!(
                    "`{mmproj}` is not a projector made for `{}`",
                    request.source
                )));
            }
            // the loader's rule: several candidates is a choice for the client, so the model loads text-only
            (Source::Local(entry), None) => match entry.projectors.as_slice() {
                [only] => Some(only.clone()),
                _ => None,
            },
            (Source::Repo(_), None) => None,
        };
        let vad = match &request.vad {
            None => None,
            Some(_) if kind != Some(ManagedKind::Transcription) => {
                return Err(ApiError::invalid_request(
                    "`vad` cuts a speech recognition model's long recordings; this source is not one",
                ));
            }
            Some(vad) => match self.resolve(vad)? {
                Source::Local(entry) if entry.kind == Some(InventoryKind::VoiceActivity) => {
                    Some(entry.path.clone())
                }
                _ => {
                    return Err(ApiError::invalid_request(format!(
                        "`{vad}` is not a voice activity model in the server's model folders"
                    )));
                }
            },
        };
        self.remove_id(&id).await?;
        let _progress = add_progress_sink(progress);
        let source_name = request.source.clone();
        if kind == Some(ManagedKind::Layout) {
            let Source::Local(entry) = source else {
                unreachable!("only inventory entries have a kind")
            };
            let device = match request.device.as_deref() {
                None | Some("auto") => self.engine.device().clone(),
                Some("cpu") => Device::Cpu,
                Some(device) => explicit_device(device, None)
                    .map_err(|error| ApiError::invalid_request(error.to_string()))?,
            };
            let path = PathBuf::from(&entry.path);
            let detector =
                tokio::task::spawn_blocking(move || PPDocLayoutV3Detector::load(&path, &device))
                    .await
                    .map_err(|_| ApiError::internal())?
                    .map_err(|error| {
                        ApiError::invalid_request(format!("loading the layout model: {error}"))
                    })?;
            self.layouts
                .write()
                .map_err(|_| ApiError::internal())?
                .insert(
                    id.clone(),
                    LayoutModel {
                        detector: Arc::new(detector),
                        source: source_name.clone(),
                    },
                );
            return Ok(ManagedModel {
                model_id: id,
                kind: ManagedKind::Layout,
                source: Some(source_name),
                projector: None,
                vad: None,
                status: LAYOUT_STATUS,
            });
        }
        if request.device.is_some() {
            return Err(ApiError::invalid_request(
                "`device` places a layout detector; the engine's models run on the engine's device",
            ));
        }
        let spec = ModelSpec {
            model: serde_json::from_value(self.model_selected(
                &source,
                projector.as_deref(),
                vad.as_deref(),
            ))
            .map_err(|error| ApiError::invalid_request(error.to_string()))?,
            model_id: Some(id.clone()),
            chat_template: None,
            jinja_explicit: None,
            max_model_len: None,
            hf_config_overrides: None,
            device_layers: None,
            isq: None,
            encoder_cache_memory_bytes: None,
            hf_revision: None,
            paged_context_len: request.paged_context_len,
            max_seqs: request.max_seqs,
            mtp: request.mtp,
        };
        self.engine.add_model(spec).await?;
        let kind = match kind.or_else(|| self.served_kind(&id)) {
            Some(kind) => kind,
            None => {
                // a served model the manager cannot describe is one it would leave untracked
                self.remove_id(&id).await?;
                return Err(ApiError::internal());
            }
        };
        self.loaded
            .write()
            .map_err(|_| ApiError::internal())?
            .insert(
                id.clone(),
                Loaded {
                    kind,
                    source: source_name.clone(),
                    projector: projector.clone(),
                    vad: vad.clone(),
                },
            );
        Ok(ManagedModel {
            model_id: id,
            kind,
            source: Some(source_name),
            projector,
            vad,
            status: ModelStatus::Loaded,
        })
    }

    /// Stops serving `model_id` and frees it.
    pub async fn unload(&self, request: ModelOperationRequest) -> Result<(), ApiError> {
        let _gate = self.gate.lock().await;
        if !self.remove_id(request.model_id.trim()).await? {
            return Err(ApiError::new(
                ApiErrorKind::NotFound,
                format!("model `{}` is not served", request.model_id),
                Some("model_not_found"),
                Some("model_id"),
            ));
        }
        Ok(())
    }

    /// Starts a download into the server's cache, with its token: its `files`, `progress`, then `done`, `canceled`
    /// or `error` events, as JSON. Dropping the stream stops the download.
    pub fn download(&self, request: ManagedDownloadRequest) -> Result<DownloadEvents, ApiError> {
        let download = model_download_from(ModelDownloadRequest {
            repo_id: request.repo_id,
            revision: request.revision,
            files: request.files,
            quant: request.quant,
            token_source: self.config.token_source.clone(),
            hf_cache_path: self.config.hf_cache_path.clone(),
        })?;
        start_download(download.work)?;
        Ok(download.events)
    }

    /// Deletes a cached repository or a folder's GGUF; never one that is loaded.
    pub fn delete(&self, request: ManagedDeleteRequest) -> Result<ManagedDeletion, ApiError> {
        // a load in progress has not recorded its source yet
        let _gate = self.gate.blocking_lock();
        let sources = self.sources()?;
        // a path holding a source, or one inside a source (a model started with `-f` is known by its folder)
        let in_use = |path: &Path| {
            sources
                .iter()
                .map(Path::new)
                .any(|source| source.starts_with(path) || path.starts_with(source))
        };
        match (request.repo_id, request.path) {
            (Some(repo_id), None) => {
                let hub = self.hub();
                let repo_dir = hub.as_deref().map(|hub| {
                    hub.join(format!(
                        "{HF_REPO_DIR_PREFIX}{}",
                        repo_id.replace('/', HF_REPO_SEPARATOR)
                    ))
                });
                if repo_dir.as_deref().is_some_and(in_use) || sources.contains(&repo_id) {
                    return Err(in_use_error(&repo_id));
                }
                let deleted = delete_hf_cache_repo(&repo_id, hub.as_deref())
                    .map_err(|error| ApiError::invalid_request(format!("{error:#}")))?;
                Ok(ManagedDeletion {
                    deleted: deleted.repo_id,
                    freed_bytes: deleted.freed_bytes,
                })
            }
            (None, Some(path)) => {
                let Source::Local(entry) = self.resolve(&path)? else {
                    return Err(ApiError::invalid_request(format!(
                        "`{path}` is not a local model"
                    )));
                };
                let in_folder = self
                    .config
                    .folders
                    .iter()
                    .any(|folder| Path::new(&path).starts_with(folder));
                if !in_folder || entry.format != ModelFormat::Gguf {
                    return Err(ApiError::invalid_request(
                        "only a model folder's GGUF files are deleted; delete a cached repository by `repo_id`",
                    ));
                }
                if entry.files.iter().any(|file| in_use(Path::new(file))) {
                    return Err(in_use_error(&path));
                }
                let mut freed_bytes = 0;
                for file in &entry.files {
                    freed_bytes += std::fs::metadata(file).map_or(0, |metadata| metadata.len());
                    std::fs::remove_file(file).map_err(|error| {
                        ApiError::invalid_request(format!("deleting `{file}`: {error}"))
                    })?;
                }
                Ok(ManagedDeletion {
                    deleted: path,
                    freed_bytes,
                })
            }
            _ => Err(ApiError::invalid_request(
                "give one of `repo_id` and `path`",
            )),
        }
    }

    /// The regions of a page, from a loaded layout model.
    pub async fn detect_layout(
        &self,
        request: LayoutDetectRequest,
    ) -> Result<LayoutDetectResponse, ApiError> {
        let detector = self
            .layouts
            .read()
            .map_err(|_| ApiError::internal())?
            .get(&request.model)
            .map(|layout| layout.detector.clone())
            .ok_or_else(|| {
                ApiError::new(
                    ApiErrorKind::NotFound,
                    format!("layout model `{}` is not loaded", request.model),
                    Some("model_not_found"),
                    Some("model"),
                )
            })?;
        let threshold = request.threshold.unwrap_or(DEFAULT_THRESHOLD);
        if !(0.0..=1.0).contains(&threshold) {
            return Err(ApiError::invalid_request(format!(
                "threshold {threshold} is outside [0, 1]"
            )));
        }
        let encoded = match request.image.split_once(DATA_URL_BASE64) {
            Some((_, data)) => data,
            None => request.image.as_str(),
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded.trim())
            .map_err(|error| {
                ApiError::invalid_request(format!("`image` is not base64: {error}"))
            })?;
        tokio::task::spawn_blocking(move || {
            let image = image::load_from_memory(&bytes)
                .map_err(|error| ApiError::invalid_request(format!("`image`: {error}")))?;
            let (width, height) = (image.width(), image.height());
            if width as usize * height as usize > MAX_LAYOUT_IMAGE_PIXELS {
                return Err(ApiError::invalid_request(format!(
                    "image {width}x{height} exceeds {MAX_LAYOUT_IMAGE_PIXELS} pixels"
                )));
            }
            let regions = detector
                .detect(&image.to_rgb8(), threshold)
                .map_err(|error| ApiError::invalid_request(format!("layout detection: {error}")))?
                .into_iter()
                .map(|detection| LayoutRegion {
                    label: detection.label,
                    score: detection.score,
                    bbox: detection.bbox,
                    polygon: detection.polygon,
                })
                .collect();
            Ok(LayoutDetectResponse {
                width,
                height,
                regions,
            })
        })
        .await
        .map_err(|_| ApiError::internal())?
    }

    fn hub(&self) -> Option<PathBuf> {
        self.config
            .hf_cache_path
            .clone()
            .or_else(default_hf_cache_dir)
    }

    // a path inside the folders or the cache, or a repository id; nothing else on the server is reachable
    fn resolve(&self, source: &str) -> Result<Source, ApiError> {
        let path = Path::new(source);
        if path.is_absolute() {
            // compared by components, so `.` and `..` would let one path pass for another
            let lexical = path
                .components()
                .all(|part| !matches!(part, Component::ParentDir | Component::CurDir));
            let roots = self.config.folders.iter().cloned().chain(self.hub());
            let inside = lexical && roots.into_iter().any(|root| path.starts_with(root));
            if !inside || !path.exists() {
                return Err(ApiError::invalid_request(format!(
                    "`{source}` is not a model in the server's model folders or Hugging Face cache"
                )));
            }
            let entry = inspect_model(path)
                .map_err(|error| ApiError::invalid_request(format!("{error:#}")))?;
            return Ok(Source::Local(Box::new(entry)));
        }
        // the loader reads a name that exists relative to the server's working directory as that path
        if is_repo_id(source) && !path.exists() {
            return Ok(Source::Repo(source.to_string()));
        }
        Err(ApiError::invalid_request(format!(
            "`{source}` is neither an inventory path nor a Hugging Face repository id"
        )))
    }

    fn model_selected(
        &self,
        source: &Source,
        projector: Option<&str>,
        vad: Option<&str>,
    ) -> serde_json::Value {
        let hf_cache = self
            .config
            .hf_cache_path
            .as_ref()
            .map(|path| path.to_string_lossy());
        match source {
            Source::Local(entry)
                if entry.format == ModelFormat::Gguf
                    && matches!(
                        entry.kind,
                        Some(InventoryKind::Text | InventoryKind::Multimodal)
                    ) =>
            {
                let first = Path::new(&entry.files[0]);
                let files: Vec<String> = entry
                    .files
                    .iter()
                    .filter_map(|file| Path::new(file).file_name())
                    .map(|name| name.to_string_lossy().into_owned())
                    .collect();
                let mut gguf = json!({
                    "quantized_model_id": first.parent().map(Path::to_string_lossy),
                    "quantized_filename": files.join(SPLIT_FILE_SEPARATOR),
                });
                // without a filename the default selection loads no projector
                if let Some(name) = projector.and_then(|projector| Path::new(projector).file_name())
                {
                    gguf["mmproj_filename"] = json!(name.to_string_lossy());
                }
                json!({ "GGUF": gguf })
            }
            // auto-detection routes speech recognition too, but only this selection takes a VAD
            Source::Local(entry) if vad.is_some() => json!({
                "Transcription": { "model_id": entry.path, "vad_model_id": vad }
            }),
            Source::Local(entry) => json!({ "Run": { "model_id": entry.path } }),
            Source::Repo(repo) => json!({ "Run": { "model_id": repo, "hf_cache_path": hf_cache } }),
        }
    }

    fn served_kind(&self, id: &str) -> Option<ManagedKind> {
        self.engine
            .models()
            .ok()?
            .data
            .into_iter()
            .find(|card| card.id == id)
            .and_then(|card| card.category)
            .map(Into::into)
    }

    // what serving needs on disk: each managed model's source, projector and VAD, and the id of every model the manager
    // did not load (started with `-m`, or added by `/v1/models/add`), which for a local model is its path
    fn sources(&self) -> Result<Vec<String>, ApiError> {
        let loaded = self.loaded.read().map_err(|_| ApiError::internal())?;
        let layouts = self.layouts.read().map_err(|_| ApiError::internal())?;
        let mut sources: Vec<String> = loaded
            .values()
            .flat_map(|entry| {
                [
                    Some(entry.source.clone()),
                    entry.projector.clone(),
                    entry.vad.clone(),
                ]
            })
            .flatten()
            .chain(layouts.values().map(|layout| layout.source.clone()))
            .collect();
        sources.extend(
            self.engine
                .models()?
                .data
                .into_iter()
                .filter(is_model_card)
                .filter(|card| !loaded.contains_key(&card.id))
                .map(|card| card.id),
        );
        Ok(sources)
    }

    // frees whatever serves `id`; whether anything did
    async fn remove_id(&self, id: &str) -> Result<bool, ApiError> {
        let layout = self
            .layouts
            .write()
            .map_err(|_| ApiError::internal())?
            .remove(id)
            .is_some();
        let served = self
            .engine
            .models()?
            .data
            .iter()
            .any(|card| card.id == id && is_model_card(card));
        if served {
            self.engine
                .remove_model(ModelOperationRequest {
                    model_id: id.to_string(),
                })
                .await?;
        }
        // forgotten only once the engine let it go, so a refused removal keeps its files protected
        self.loaded
            .write()
            .map_err(|_| ApiError::internal())?
            .remove(id);
        Ok(layout || served)
    }
}

// a served model's card, not the `default` alias or a LoRA adapter's
fn is_model_card(card: &ModelObject) -> bool {
    card.parent.is_none() && card.id != crate::lora_routing::DEFAULT_MODEL_ID
}

fn status_of(status: Option<&str>) -> ModelStatus {
    match status {
        Some("unloaded") => ModelStatus::Unloaded,
        Some("reloading") => ModelStatus::Reloading,
        _ => ModelStatus::Loaded,
    }
}

fn loadable_kind(entry: &InventoryModel) -> Result<ManagedKind, ApiError> {
    Ok(match entry.kind {
        Some(InventoryKind::Text) => ManagedKind::Text,
        Some(InventoryKind::Multimodal) => ManagedKind::Multimodal,
        Some(InventoryKind::Embedding) => ManagedKind::Embedding,
        Some(InventoryKind::ImageGeneration) => ManagedKind::ImageGeneration,
        Some(InventoryKind::Speech) => ManagedKind::Speech,
        Some(InventoryKind::Transcription) => ManagedKind::Transcription,
        Some(InventoryKind::VoiceActivity) => ManagedKind::VoiceActivity,
        Some(InventoryKind::Diarization) => ManagedKind::Diarization,
        Some(InventoryKind::Layout) => ManagedKind::Layout,
        Some(InventoryKind::Projector) => {
            return Err(ApiError::invalid_request(
                "a projector loads beside its text GGUF: load the GGUF with `mmproj`",
            ));
        }
        None => {
            return Err(ApiError::invalid_request(format!(
                "`{}` cannot be loaded: {}",
                entry.path,
                entry.error.as_deref().unwrap_or("its kind is unknown")
            )));
        }
    })
}

fn in_use_error(what: &str) -> ApiError {
    ApiError::new(
        ApiErrorKind::Conflict,
        format!("`{what}` is loaded; unload it first"),
        Some("model_in_use"),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_ids_are_told_from_other_sources() {
        for id in ["org/name", "gpt2", "Org.x/model-1_b"] {
            assert!(is_repo_id(id), "{id}");
        }
        for source in [
            "",
            "a/b/c",
            "../x",
            "org/.hidden",
            "relative/../path",
            "org/",
        ] {
            assert!(!is_repo_id(source), "{source}");
        }
    }
}
