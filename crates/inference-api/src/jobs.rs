//! Long work as jobs: world generation and reconstruction queue one at a time on a worker, hold their device to
//! themselves while they run, and record their stages, progress, preview images and outputs in a folder per job.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use inference_core::InferenceRs;
use inference_recon::progress::CANCELLED;
pub use inference_recon::progress::Progress;
use inference_tensor::Device;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use utoipa::ToSchema;

use crate::api_error::{ApiError, ApiErrorKind};
use crate::engine::Engine;
use crate::reconstruction::{
    ReconstructOptions, VideoOptions, WorldGenModels, WorldGenOptions, generate_world_on,
    reconstruct_on,
};
use crate::request_body::{JsonRequest, parse_json};

/// Every job id starts with this.
pub const JOB_ID_PREFIX: &str = "job_";
const RECORD: &str = "job.json";
const RECORD_PARTIAL: &str = "job.json.partial";
const INPUT_DIR: &str = "input";
// how long a job waits for the models on its device to finish their requests before it gives up
const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 120;
const EVENT_CAPACITY: usize = 256;
const INTERRUPTED: &str = "interrupted: the server stopped before the job finished";
// a world job's subfolders
const WORLD_DIR: &str = "world";
#[cfg(feature = "cuda")]
const SPLATS_DIR: &str = "splats";
const RECONSTRUCTION_DIR: &str = "reconstruction";
// files read as a video rather than images
const VIDEO_EXTENSIONS: [&str; 7] = ["mp4", "m4v", "mov", "avi", "mkv", "webm", "gif"];
// the checkpoints' places under a models root, as their repositories lay them out
const WORLDSTEREO_DIR: &str = "WorldStereo/worldstereo-memory-dmd";
const WAN_DIR: &str = "Wan2.1-I2V-14B-480P-Diffusers";
const MOGE_DIR: &str = "moge-2-vitl-normal";
const WORLDMIRROR_DIR: &str = "HY-World-2.0/HY-WorldMirror-2.0";
const LPIPS_FILE: &str = "lpips-vgg/lpips_vgg.safetensors";

/// Where jobs keep their folders and which weights they run.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(default)]
pub struct JobsConfig {
    /// One folder per job: its record, inputs, previews and outputs
    #[schema(value_type = String)]
    pub dir: PathBuf,
    /// The `worldstereo-memory-dmd` folder
    #[schema(value_type = Option<String>)]
    pub worldstereo: Option<PathBuf>,
    /// The Wan2.1-I2V-14B-480P diffusers folder
    #[schema(value_type = Option<String>)]
    pub wan: Option<PathBuf>,
    /// MoGe-2 ViT-L normal, converted to safetensors
    #[schema(value_type = Option<String>)]
    pub moge: Option<PathBuf>,
    /// WorldMirror 2.0
    #[schema(value_type = Option<String>)]
    pub worldmirror: Option<PathBuf>,
    /// VGG16 features and the LPIPS heads, for splat training
    #[schema(value_type = Option<String>)]
    pub lpips: Option<PathBuf>,
    /// The CUDA device jobs run on
    pub device_ordinal: usize,
    /// How long a job waits for the models on its device to go quiet before it fails, in seconds
    pub drain_timeout_secs: u64,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self {
            dir: default_jobs_dir(),
            worldstereo: None,
            wan: None,
            moge: None,
            worldmirror: None,
            lpips: None,
            device_ordinal: 0,
            drain_timeout_secs: DEFAULT_DRAIN_TIMEOUT_SECS,
        }
    }
}

impl JobsConfig {
    /// The checkpoints found under `root` in their repositories' layout: `WorldStereo/worldstereo-memory-dmd`,
    /// `Wan2.1-I2V-14B-480P-Diffusers`, `moge-2-vitl-normal`, `HY-World-2.0/HY-WorldMirror-2.0` and
    /// `lpips-vgg/lpips_vgg.safetensors`; a missing one stays unset.
    pub fn with_models_root(mut self, root: &Path) -> Self {
        let found = |relative: &str| {
            let path = root.join(relative);
            path.exists().then_some(path)
        };
        self.worldstereo = found(WORLDSTEREO_DIR);
        self.wan = found(WAN_DIR);
        self.moge = found(MOGE_DIR);
        self.worldmirror = found(WORLDMIRROR_DIR);
        self.lpips = found(LPIPS_FILE);
        self
    }
}

/// `$XDG_CACHE_HOME/inference-rs/engine-jobs`, or under `~/.cache`.
pub fn default_jobs_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("inference-rs")
        .join("engine-jobs")
}

/// Indoor scenes keep far distances; outdoor ones compress them.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SceneKind {
    Indoor,
    Outdoor,
}

/// A world from a panorama, the job's one input.
#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
pub struct WorldJob {
    /// The caption every clip is generated with
    #[serde(default)]
    pub prompt: String,
    /// Decided by the sky found in the panorama when unset
    #[serde(default)]
    pub scene: Option<SceneKind>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Train Gaussian splats on the result (on by default)
    #[serde(default)]
    pub splats: Option<bool>,
    /// Splat training steps
    #[serde(default)]
    pub steps: Option<usize>,
}

/// WorldMirror 2.0 on the job's inputs: images, or one video.
#[derive(Clone, Debug, Default, Serialize, Deserialize, ToSchema)]
pub struct ReconstructJob {
    /// The longest edge to infer at
    #[serde(default)]
    pub target_size: Option<usize>,
    /// Frames a second sampled from a video
    #[serde(default)]
    pub fps: Option<f64>,
    /// The most frames sampled from a video
    #[serde(default)]
    pub max_frames: Option<usize>,
}

/// What a job does.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobRequest {
    /// WorldStereo-2 keyframes along camera paths from a panorama, WorldMirror 2.0 on them, then Gaussian splats
    World(WorldJob),
    /// WorldMirror 2.0 cameras, depth, points and splats from images or a video
    Reconstruct(ReconstructJob),
}

impl JobRequest {
    fn kind(&self) -> &'static str {
        match self {
            Self::World(_) => "world",
            Self::Reconstruct(_) => "reconstruct",
        }
    }
}

/// A file a job takes: its bytes, or a path on this machine.
pub enum JobInput {
    Bytes { name: String, data: Vec<u8> },
    Path(PathBuf),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobStage {
    pub name: String,
    /// Steps in the stage; 0 when it has no count
    pub total: usize,
    pub done: usize,
    /// Unix seconds
    pub started_at: f64,
    #[serde(default)]
    pub finished_at: Option<f64>,
}

/// An image that shows how the job is going.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobPreview {
    pub stage: String,
    pub caption: String,
    /// Relative to the job's folder
    pub file: String,
}

/// A file a job made.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobOutput {
    pub name: String,
    /// Relative to the job's folder
    pub file: String,
}

/// A job's record: what it was asked, how far it got, and what it made.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobRecord {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub owner: Option<String>,
    pub status: JobStatus,
    /// Unix seconds
    pub created_at: f64,
    #[serde(default)]
    pub started_at: Option<f64>,
    #[serde(default)]
    pub finished_at: Option<f64>,
    #[schema(value_type = Object)]
    pub request: serde_json::Value,
    /// The input files' names, in `input/`
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub stages: Vec<JobStage>,
    #[serde(default)]
    pub previews: Vec<JobPreview>,
    #[serde(default)]
    pub outputs: Vec<JobOutput>,
    #[serde(default)]
    pub error: Option<String>,
    /// What the run reported (its `meta.json` fields, a training summary)
    #[serde(default)]
    #[schema(value_type = Object)]
    pub report: serde_json::Value,
}

/// A change to a running job.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    Status {
        status: JobStatus,
    },
    Stage {
        name: String,
        total: usize,
    },
    Progress {
        done: usize,
        total: usize,
    },
    Preview(JobPreview),
    Finished {
        status: JobStatus,
        #[serde(default)]
        error: Option<String>,
    },
}

/// A job and the files it takes, by their paths on this machine.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobSubmission {
    pub request: JobRequest,
    #[serde(default)]
    #[schema(value_type = Vec<String>)]
    pub inputs: Vec<PathBuf>,
}

/// This owner's jobs, newest first.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobList {
    pub data: Vec<JobRecord>,
}

/// A file in a job's folder, given relative to it.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobFileRequest {
    pub id: String,
    pub file: String,
}

/// Where a job's file is on this machine.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobFilePath {
    pub path: String,
}

/// A job removed with its folder.
#[derive(Clone, Debug, Serialize, Deserialize, ToSchema)]
pub struct JobDeleted {
    pub id: String,
    pub deleted: bool,
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

json_requests!(JobsConfig, JobSubmission, JobFileRequest);

/// What a job's work returns.
pub struct JobResult {
    pub outputs: Vec<JobOutput>,
    pub report: serde_json::Value,
}

type Work = Box<dyn FnOnce(&JobContext) -> anyhow::Result<JobResult> + Send>;

struct Job {
    record: Mutex<JobRecord>,
    events: broadcast::Sender<JobEvent>,
    cancel: AtomicBool,
    dir: PathBuf,
}

impl Job {
    fn new(record: JobRecord, dir: PathBuf) -> Self {
        Self {
            record: Mutex::new(record),
            events: broadcast::channel(EVENT_CAPACITY).0,
            cancel: AtomicBool::new(false),
            dir,
        }
    }

    fn record(&self) -> std::sync::MutexGuard<'_, JobRecord> {
        self.record
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn snapshot(&self) -> JobRecord {
        self.record().clone()
    }

    // writes the record beside a partial copy first, so a crash mid-write keeps the last whole one
    fn save(&self, record: &JobRecord) {
        let write = || -> std::io::Result<()> {
            let json = serde_json::to_vec_pretty(record).map_err(std::io::Error::other)?;
            std::fs::write(self.dir.join(RECORD_PARTIAL), json)?;
            std::fs::rename(self.dir.join(RECORD_PARTIAL), self.dir.join(RECORD))
        };
        if let Err(error) = write() {
            tracing::warn!("Could not save job {}: {error}", record.id);
        }
    }

    fn update(&self, persist: bool, event: Option<JobEvent>, change: impl FnOnce(&mut JobRecord)) {
        let mut record = self.record();
        change(&mut record);
        if persist {
            self.save(&record);
        }
        drop(record);
        if let Some(event) = event {
            let _ = self.events.send(event);
        }
    }

    fn finish(&self, outcome: anyhow::Result<JobResult>) {
        let (status, error, result) = match outcome {
            Ok(result) => (JobStatus::Succeeded, None, Some(result)),
            Err(_) if self.cancel.load(Ordering::Relaxed) => (JobStatus::Cancelled, None, None),
            Err(error) if format!("{error:#}").contains(CANCELLED) => {
                (JobStatus::Cancelled, None, None)
            }
            Err(error) => (JobStatus::Failed, Some(format!("{error:#}")), None),
        };
        let event = JobEvent::Finished {
            status,
            error: error.clone(),
        };
        self.update(true, Some(event), |record| {
            let now = now();
            if let Some(stage) = record.stages.last_mut()
                && stage.finished_at.is_none()
            {
                stage.finished_at = Some(now);
            }
            record.status = status;
            record.finished_at = Some(now);
            record.error = error;
            if let Some(result) = result {
                record.outputs = result.outputs;
                record.report = result.report;
            }
        });
    }
}

/// What a job's work sees: its folder, its device, and the progress hooks the pipelines report through.
pub struct JobContext {
    job: Arc<Job>,
    device: Option<Device>,
    runtime: tokio::runtime::Handle,
}

impl JobContext {
    /// The job's folder, where its work writes.
    pub fn dir(&self) -> &Path {
        &self.job.dir
    }

    /// The device the job holds, when it asked for one.
    pub fn device(&self) -> anyhow::Result<&Device> {
        self.device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("this job holds no device"))
    }

    /// The job's input files, in the order given.
    pub fn inputs(&self) -> Vec<PathBuf> {
        let input = self.job.dir.join(INPUT_DIR);
        self.job
            .record()
            .inputs
            .iter()
            .map(|name| input.join(name))
            .collect()
    }

    /// Runs `future` to completion on the job's worker.
    pub fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }
}

impl Progress for JobContext {
    fn stage(&self, name: &str, total: usize) {
        let event = JobEvent::Stage {
            name: name.to_string(),
            total,
        };
        self.job.update(true, Some(event), |record| {
            let now = now();
            if let Some(stage) = record.stages.last_mut()
                && stage.finished_at.is_none()
            {
                stage.finished_at = Some(now);
                stage.done = stage.total;
            }
            record.stages.push(JobStage {
                name: name.to_string(),
                total,
                done: 0,
                started_at: now,
                finished_at: None,
            });
        });
    }

    fn advance(&self, done: usize) {
        let mut total = 0;
        self.job.update(false, None, |record| {
            if let Some(stage) = record.stages.last_mut() {
                stage.done = done;
                total = stage.total;
            }
        });
        let _ = self.job.events.send(JobEvent::Progress { done, total });
    }

    fn preview(&self, caption: &str, path: &Path) {
        let Ok(file) = path.strip_prefix(&self.job.dir) else {
            return;
        };
        if !path.is_file() {
            return;
        }
        let file = file.to_string_lossy().replace('\\', "/");
        let mut preview = None;
        self.job.update(true, None, |record| {
            let entry = JobPreview {
                stage: record
                    .stages
                    .last()
                    .map(|s| s.name.clone())
                    .unwrap_or_default(),
                caption: caption.to_string(),
                file,
            };
            record.previews.push(entry.clone());
            preview = Some(entry);
        });
        if let Some(preview) = preview {
            let _ = self.job.events.send(JobEvent::Preview(preview));
        }
    }

    fn cancelled(&self) -> bool {
        self.job.cancel.load(Ordering::Relaxed)
    }
}

struct Queued {
    job: Arc<Job>,
    device: Option<Device>,
    work: Work,
}

/// The jobs of one engine: their records, the queue and the worker that runs it.
pub struct JobRunner {
    config: RwLock<JobsConfig>,
    jobs: RwLock<HashMap<String, Arc<Job>>>,
    // the records kept in the folder, read once on first use
    read: std::sync::Once,
    queue: Mutex<Option<std::sync::mpsc::Sender<Queued>>>,
}

impl Default for JobRunner {
    fn default() -> Self {
        Self {
            config: RwLock::new(JobsConfig::default()),
            jobs: RwLock::new(HashMap::new()),
            read: std::sync::Once::new(),
            queue: Mutex::new(None),
        }
    }
}

impl JobRunner {
    fn config(&self) -> JobsConfig {
        self.config
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn jobs(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Arc<Job>>> {
        self.read.call_once(|| self.read_records());
        self.jobs
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    // the folder's records; a job the server stopped in the middle of is failed
    fn read_records(&self) {
        let dir = self.config().dir;
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return;
        };
        let mut jobs = self
            .jobs
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(bytes) = std::fs::read(path.join(RECORD)) else {
                continue;
            };
            let Ok(mut record) = serde_json::from_slice::<JobRecord>(&bytes) else {
                continue;
            };
            let job = Job::new(record.clone(), path);
            if !record.status.is_finished() {
                record.status = JobStatus::Failed;
                record.error = Some(INTERRUPTED.to_string());
                record.finished_at = Some(now());
                job.save(&record);
                *job.record() = record.clone();
            }
            jobs.insert(record.id.clone(), Arc::new(job));
        }
    }

    fn sender(&self, state: &Arc<InferenceRs>) -> std::sync::mpsc::Sender<Queued> {
        let mut queue = self
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        queue
            .get_or_insert_with(|| {
                let (tx, rx) = std::sync::mpsc::channel::<Queued>();
                let state = Arc::downgrade(state);
                let drain = Duration::from_secs(self.config().drain_timeout_secs);
                std::thread::Builder::new()
                    .name("inference-jobs".into())
                    .spawn(move || run_queue(rx, state, drain))
                    .expect("spawning the job worker");
                tx
            })
            .clone()
    }
}

// one job at a time: wait for the device, run, record how it ended; holds the engine only while a job runs
fn run_queue(rx: std::sync::mpsc::Receiver<Queued>, state: Weak<InferenceRs>, drain: Duration) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!("The job worker could not start: {error}");
            return;
        }
    };
    for Queued { job, device, work } in rx {
        if job.cancel.load(Ordering::Relaxed) {
            job.finish(Err(anyhow::anyhow!(CANCELLED)));
            continue;
        }
        job.update(
            true,
            Some(JobEvent::Status {
                status: JobStatus::Running,
            }),
            |record| {
                record.status = JobStatus::Running;
                record.started_at = Some(now());
            },
        );
        let lease = match (&device, state.upgrade()) {
            (Some(device), Some(state)) => {
                let holder = {
                    let record = job.record();
                    format!("job {} ({})", record.id, record.kind)
                };
                match runtime.block_on(state.lease_device(device, &holder, drain)) {
                    Ok(lease) => Some(lease),
                    Err(error) => {
                        job.finish(Err(error.into()));
                        continue;
                    }
                }
            }
            (Some(_), None) => {
                job.finish(Err(anyhow::anyhow!("the engine has shut down")));
                continue;
            }
            (None, _) => None,
        };
        let context = JobContext {
            job: job.clone(),
            device,
            runtime: runtime.handle().clone(),
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&context)))
            .unwrap_or_else(|_| Err(anyhow::anyhow!("the job panicked")));
        drop(context);
        drop(lease);
        job.finish(outcome);
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

fn not_found(id: &str) -> ApiError {
    ApiError::new(
        ApiErrorKind::NotFound,
        format!("job `{id}` was not found"),
        Some("job_not_found"),
        Some("id"),
    )
}

fn not_configured(what: &str) -> ApiError {
    ApiError::new(
        ApiErrorKind::InvalidRequest,
        format!("{what} is not configured on this server"),
        Some("job_not_configured"),
        None,
    )
}

fn io_error(error: std::io::Error) -> ApiError {
    tracing::warn!("Job folder: {error}");
    ApiError::internal()
}

// a file name with no folders in it, or None
fn plain_name(name: &str) -> Option<String> {
    let name = Path::new(name).file_name()?.to_str()?;
    (!name.starts_with('.')).then(|| name.to_string())
}

fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| VIDEO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

fn output(dir: &Path, file: &str, name: &str) -> Option<JobOutput> {
    dir.join(file).is_file().then(|| JobOutput {
        name: name.to_string(),
        file: file.to_string(),
    })
}

fn world_work(config: &JobsConfig, job: WorldJob) -> Result<Work, ApiError> {
    let need =
        |path: &Option<PathBuf>, what: &str| path.clone().ok_or_else(|| not_configured(what));
    let models = WorldGenModels {
        worldstereo: need(&config.worldstereo, "the WorldStereo-2 checkpoint")?,
        wan: need(&config.wan, "the Wan 2.1 I2V checkpoint")?,
        moge: need(&config.moge, "the MoGe-2 checkpoint")?,
        worldmirror: Some(need(&config.worldmirror, "the WorldMirror 2.0 checkpoint")?),
    };
    let splats = job.splats.unwrap_or(true);
    let lpips = if splats {
        Some(need(&config.lpips, "the LPIPS weights for splat training")?)
    } else {
        None
    };
    if splats && !cfg!(feature = "cuda") {
        return Err(ApiError::invalid_request(
            "splat training needs a CUDA build",
        ));
    }
    let mut options = WorldGenOptions {
        prompt: job.prompt,
        outdoor: job.scene.map(|s| s == SceneKind::Outdoor),
        ..Default::default()
    };
    if let Some(seed) = job.seed {
        options.seed = seed;
    }
    let steps = job.steps;
    Ok(Box::new(move |ctx: &JobContext| {
        let device = ctx.device()?.clone();
        let panorama = ctx
            .inputs()
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("a world job takes a panorama"))?;
        let world = ctx.dir().join(WORLD_DIR);
        let world_report = generate_world_on(&models, &panorama, &world, &options, ctx, &device)?;
        let splats = match lpips {
            Some(lpips) => Some(train_world_splats(ctx, &world, lpips, steps, &device)?),
            None => None,
        };
        let report = serde_json::json!({ "world": world_report, "splats": splats });
        let dir = ctx.dir();
        let outputs = [
            ("splats", "splats/splats.ply"),
            ("compressed splats", "splats/splats.spz"),
            ("mesh", "splats/mesh_simplified.ply"),
            ("full mesh", "splats/mesh.ply"),
            (
                "reconstruction splats",
                "world/reconstruction/gaussians.ply",
            ),
            ("points", "world/reconstruction/points.ply"),
            ("world meta", "world/meta.json"),
        ]
        .into_iter()
        .filter_map(|(name, file)| output(dir, file, name))
        .collect();
        Ok(JobResult { outputs, report })
    }))
}

#[cfg(feature = "cuda")]
fn train_world_splats(
    ctx: &JobContext,
    world: &Path,
    lpips: PathBuf,
    steps: Option<usize>,
    device: &Device,
) -> anyhow::Result<serde_json::Value> {
    use crate::reconstruction::{SPLAT_DEFAULT_STEPS, TrainOptions, train_splats_on};
    let options = TrainOptions {
        steps: steps.unwrap_or(SPLAT_DEFAULT_STEPS),
        lpips,
    };
    let out = ctx.dir().join(SPLATS_DIR);
    let report = train_splats_on(&world.join("gs_data"), &out, &options, ctx, device)?;
    Ok(serde_json::to_value(report)?)
}

#[cfg(not(feature = "cuda"))]
fn train_world_splats(
    _: &JobContext,
    _: &Path,
    _: PathBuf,
    _: Option<usize>,
    _: &Device,
) -> anyhow::Result<serde_json::Value> {
    anyhow::bail!("splat training needs a CUDA build")
}

fn reconstruct_work(config: &JobsConfig, job: ReconstructJob) -> Result<Work, ApiError> {
    let checkpoint = config
        .worldmirror
        .clone()
        .ok_or_else(|| not_configured("the WorldMirror 2.0 checkpoint"))?;
    let mut options = ReconstructOptions::default();
    if let Some(size) = job.target_size {
        options.target_size = size;
    }
    let mut video = VideoOptions::default();
    if let Some(fps) = job.fps {
        video.fps = fps;
    }
    if let Some(max_frames) = job.max_frames {
        video.max_frames = max_frames;
    }
    Ok(Box::new(move |ctx: &JobContext| {
        let device = ctx.device()?.clone();
        let inputs = ctx.inputs();
        let input = match inputs.as_slice() {
            [] => anyhow::bail!("a reconstruction takes images or a video"),
            [one] if is_video(one) => one.clone(),
            _ => ctx.dir().join(INPUT_DIR),
        };
        ctx.stage("reconstruction", 0);
        let out = ctx.dir().join(RECONSTRUCTION_DIR);
        let run = ctx.block_on(reconstruct_on(
            &checkpoint,
            &input,
            &out,
            &options,
            video,
            &device,
        ))?;
        let dir = ctx.dir();
        let first_depth = "reconstruction/depth/depth_0000.png";
        if dir.join(first_depth).is_file() {
            ctx.preview("depth of the first frame", &dir.join(first_depth));
        }
        let outputs = [
            ("splats", "reconstruction/gaussians.ply"),
            ("points", "reconstruction/points.ply"),
            ("cameras", "reconstruction/camera_params.json"),
            ("meta", "reconstruction/meta.json"),
        ]
        .into_iter()
        .filter_map(|(name, file)| output(dir, file, name))
        .collect();
        let report = serde_json::json!({
            "frames": run.frames,
            "width": run.width,
            "height": run.height,
            "forward_seconds": run.forward_seconds,
            "total_seconds": run.total_seconds,
        });
        Ok(JobResult { outputs, report })
    }))
}

impl Engine {
    pub(crate) fn jobs_configure(&self, config: JobsConfig) {
        *self
            .job_runner()
            .config
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = config;
    }

    pub(crate) fn jobs_submit(
        &self,
        request: JobRequest,
        inputs: Vec<JobInput>,
    ) -> Result<JobRecord, ApiError> {
        let config = self.job_runner().config();
        let work = match request.clone() {
            JobRequest::World(job) => {
                if inputs.len() != 1 {
                    return Err(ApiError::invalid_request("a world job takes one panorama"));
                }
                world_work(&config, job)?
            }
            JobRequest::Reconstruct(job) => {
                if inputs.is_empty() {
                    return Err(ApiError::invalid_request(
                        "a reconstruction takes images or a video",
                    ));
                }
                reconstruct_work(&config, job)?
            }
        };
        let device = Device::new_cuda(config.device_ordinal).map_err(|error| {
            ApiError::new(
                ApiErrorKind::Unavailable,
                format!("jobs need CUDA device {}: {error}", config.device_ordinal),
                Some("device_unavailable"),
                None,
            )
        })?;
        let kind = request.kind();
        let request = serde_json::to_value(&request).map_err(|_| ApiError::internal())?;
        self.enqueue(&config, kind, request, inputs, Some(device), work)
    }

    pub(crate) fn jobs_submit_task(
        &self,
        kind: &str,
        device: Option<Device>,
        work: impl FnOnce(&JobContext) -> anyhow::Result<JobResult> + Send + 'static,
    ) -> Result<JobRecord, ApiError> {
        let config = self.job_runner().config();
        self.enqueue(
            &config,
            kind,
            serde_json::Value::Null,
            Vec::new(),
            device,
            Box::new(work),
        )
    }

    fn enqueue(
        &self,
        config: &JobsConfig,
        kind: &str,
        request: serde_json::Value,
        inputs: Vec<JobInput>,
        device: Option<Device>,
        work: Work,
    ) -> Result<JobRecord, ApiError> {
        let runner = self.job_runner();
        let id = format!("{JOB_ID_PREFIX}{}", uuid::Uuid::new_v4().simple());
        let dir = config.dir.join(&id);
        let input_dir = dir.join(INPUT_DIR);
        std::fs::create_dir_all(&input_dir).map_err(io_error)?;
        let mut names = Vec::with_capacity(inputs.len());
        let stored = (|| -> Result<(), ApiError> {
            for (i, input) in inputs.into_iter().enumerate() {
                let given = match &input {
                    JobInput::Bytes { name, .. } => name.clone(),
                    JobInput::Path(path) => path.to_string_lossy().to_string(),
                };
                let name = plain_name(&given).unwrap_or_else(|| format!("input_{i}"));
                if names.contains(&name) {
                    return Err(ApiError::invalid_request(format!(
                        "two inputs are named `{name}`"
                    )));
                }
                match input {
                    JobInput::Bytes { data, .. } => std::fs::write(input_dir.join(&name), data),
                    JobInput::Path(path) => std::fs::copy(&path, input_dir.join(&name)).map(|_| ()),
                }
                .map_err(|error| ApiError::invalid_request(format!("input `{given}`: {error}")))?;
                names.push(name);
            }
            Ok(())
        })();
        if let Err(error) = stored {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(error);
        }
        let record = JobRecord {
            id: id.clone(),
            kind: kind.to_string(),
            owner: self.owner().map(ToString::to_string),
            status: JobStatus::Queued,
            created_at: now(),
            started_at: None,
            finished_at: None,
            request,
            inputs: names,
            stages: Vec::new(),
            previews: Vec::new(),
            outputs: Vec::new(),
            error: None,
            report: serde_json::Value::Null,
        };
        let job = Arc::new(Job::new(record.clone(), dir));
        job.save(&record);
        // the folder's earlier records are read before this one joins them
        drop(runner.jobs());
        runner
            .jobs
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, job.clone());
        runner
            .sender(self.state())
            .send(Queued { job, device, work })
            .map_err(|_| ApiError::internal())?;
        Ok(record)
    }

    fn owned_job(&self, id: &str) -> Result<Arc<Job>, ApiError> {
        let job = self
            .job_runner()
            .jobs()
            .get(id)
            .cloned()
            .ok_or_else(|| not_found(id))?;
        if job.record().owner.as_deref() != self.owner() {
            return Err(not_found(id));
        }
        Ok(job)
    }

    pub(crate) fn jobs_get(&self, id: &str) -> Result<JobRecord, ApiError> {
        Ok(self.owned_job(id)?.snapshot())
    }

    pub(crate) fn jobs_list(&self) -> Vec<JobRecord> {
        let mut records = self
            .job_runner()
            .jobs()
            .values()
            .map(|job| job.snapshot())
            .filter(|record| record.owner.as_deref() == self.owner())
            .collect::<Vec<_>>();
        records.sort_by(|a, b| b.created_at.total_cmp(&a.created_at));
        records
    }

    pub(crate) fn jobs_cancel(&self, id: &str) -> Result<JobRecord, ApiError> {
        let job = self.owned_job(id)?;
        job.cancel.store(true, Ordering::Relaxed);
        Ok(job.snapshot())
    }

    pub(crate) fn jobs_delete(&self, id: &str) -> Result<(), ApiError> {
        let job = self.owned_job(id)?;
        if !job.record().status.is_finished() {
            return Err(ApiError::new(
                ApiErrorKind::Conflict,
                format!("job `{id}` has not finished"),
                Some("job_not_finished"),
                Some("id"),
            ));
        }
        self.job_runner()
            .jobs
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
        std::fs::remove_dir_all(&job.dir).map_err(io_error)
    }

    pub(crate) fn jobs_events(
        &self,
        id: &str,
    ) -> Result<(JobRecord, broadcast::Receiver<JobEvent>), ApiError> {
        let job = self.owned_job(id)?;
        let events = job.events.subscribe();
        Ok((job.snapshot(), events))
    }

    pub(crate) fn jobs_file(&self, id: &str, file: &str) -> Result<PathBuf, ApiError> {
        let job = self.owned_job(id)?;
        let relative = Path::new(file);
        let plain = relative
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)));
        let path = job.dir.join(relative);
        if !plain || file.is_empty() || !path.is_file() {
            return Err(ApiError::new(
                ApiErrorKind::NotFound,
                format!("job `{id}` has no file `{file}`"),
                Some("job_file_not_found"),
                Some("file"),
            ));
        }
        Ok(path)
    }
}
