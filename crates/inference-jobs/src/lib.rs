//! Long work as jobs: world generation and reconstruction queue one at a time on a worker, hold their device to
//! themselves while they run, and record their stages, progress, preview images and outputs in a folder per job.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub use inference_api::devices::JobNeed;
use inference_api::devices::{DeviceLease, Displace, Placed};
use inference_recon::progress::CANCELLED;
pub use inference_recon::progress::Progress;
use inference_tensor::Device;
use serde::Serialize;
use tokio::sync::broadcast;
use utoipa::ToSchema;

use inference_api::Engine;
use inference_api::api_error::{ApiError, ApiErrorKind};
use inference_api::request_body::{JsonRequest, parse_json};

use crate::reconstruction::{
    ReconstructOptions, VideoOptions, WorldGenModels, WorldGenOptions, generate_world_on,
    reconstruct_on,
};

pub mod image_to_3d;
pub mod reconstruction;
pub mod voxels;
pub mod worldgen_package;

/// Every job id starts with this.
pub const JOB_ID_PREFIX: &str = "job_";
const RECORD: &str = "job.json";
const RECORD_PARTIAL: &str = "job.json.partial";
const INPUT_DIR: &str = "input";
// how long a job waits for the models on its device to finish their requests before it gives up
const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 120;
const EVENT_CAPACITY: usize = 256;
const INTERRUPTED: &str = "interrupted: the server stopped before the job finished";
// a reconstruction's peak above idle is ~8.7 GB whatever its input: WorldMirror caps its tokens
const RECONSTRUCT_NEED_BYTES: u64 = 10 << 30;
// how often a job waiting for room looks again
const WAIT_POLL: Duration = Duration::from_secs(2);
// a world job's subfolders
const WORLD_DIR: &str = "world";
#[cfg(feature = "cuda")]
const SPLATS_DIR: &str = "splats";
const RECONSTRUCTION_DIR: &str = "reconstruction";
const VOXELS_DIR: &str = "voxels";
// a voxel job's peak above idle: the bf16 flows, f32 DINOv3 and BiRefNet, and the decoder's 1024^3 level
const VOXELS_NEED_BYTES: u64 = 12 << 30;
// an image-to-3D job's: one flow at a time, but two decoders' 1024^3 levels
const IMAGE_TO_3D_NEED_BYTES: u64 = 16 << 30;
const IMAGE_TO_3D_DIR: &str = "model";
// files read as a video rather than images
const VIDEO_EXTENSIONS: [&str; 7] = ["mp4", "m4v", "mov", "avi", "mkv", "webm", "gif"];
// the checkpoints' places under a models root, as their repositories lay them out
pub(crate) const WORLDSTEREO_DIR: &str = "WorldStereo/worldstereo-memory-dmd";
pub(crate) const WAN_DIR: &str = "Wan2.1-I2V-14B-480P-Diffusers";
pub(crate) const MOGE_DIR: &str = "moge-2-vitl-normal";
pub(crate) const WORLDMIRROR_DIR: &str = "HY-World-2.0/HY-WorldMirror-2.0";
pub(crate) const LPIPS_FILE: &str = "lpips-vgg/lpips_vgg.safetensors";
pub(crate) const JOVOX_DIR: &str = "JoVox-64";
pub(crate) const DINOV3_DIR: &str = "dinov3-vitl16-pretrain-lvd1689m";
pub(crate) const BIREFNET_DIR: &str = "BiRefNet";
pub(crate) const TRELLIS2_DIR: &str = "TRELLIS.2-4B";
pub(crate) const TRELLIS_LARGE_DIR: &str = "TRELLIS-image-large";
/// Where JoVox-64 is fetched from when no checkpoint is configured.
pub const JOVOX_REPO: &str = "xiangjuntang/JoVox-64";
/// Where DINOv3 ViT-L/16 is fetched from (gated: the token's account needs access).
pub const DINOV3_REPO: &str = "facebook/dinov3-vitl16-pretrain-lvd1689m";
/// Where TRELLIS.2 is fetched from.
pub const TRELLIS2_REPO: &str = "microsoft/TRELLIS.2-4B";
/// Where the sparse-structure decoder TRELLIS.2 reuses is fetched from.
pub const TRELLIS_LARGE_REPO: &str = "microsoft/TRELLIS-image-large";
/// Where BiRefNet is fetched from.
pub const BIREFNET_REPO: &str = "ZhengPeng7/BiRefNet";
/// Where WorldMirror 2.0 is fetched from when no checkpoint is configured.
pub const WORLDMIRROR_REPO: &str = "tencent/HY-World-2.0";
const WORLDMIRROR_FILES: [&str; 2] = ["config.json", "model.safetensors"];
// a models repo's files no job reads
const REPO_SKIPPED: [&str; 2] = [".gitattributes", "README.md"];
// a file every models repo holds, which finds its snapshot in the hub cache when the hub cannot be reached
const MODELS_REPO_MARKER: &str = "WorldStereo/worldstereo-memory-dmd/config.json";
pub(crate) const DOWNLOAD_STAGE: &str = "download";

/// How much a job may displace to run.
#[derive(
    Clone, Copy, Debug, Default, Serialize, inference_json::FromJson, ToSchema, PartialEq, Eq,
)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// Unload what the job needs, idle models first, waiting for busy ones to finish their requests
    #[default]
    Fastest,
    /// Unload only idle models; while one that is serving requests is in the way, wait
    WhenQuiet,
    /// Run only where the job fits beside the loaded models
    NoEviction,
}

impl Placement {
    fn displace(self) -> Displace {
        match self {
            Self::Fastest => Displace::Busy,
            Self::WhenQuiet => Displace::Idle,
            Self::NoEviction => Displace::Nothing,
        }
    }
}

/// Where jobs keep their folders and which weights they run.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
#[serde(default)]
pub struct JobsConfig {
    /// One folder per job: its record, inputs, previews and outputs; unset, the server's cache `inference-rs/engine-jobs`
    // the derived default would be the generating machine's path, so the document leaves it to the server
    #[schema(value_type = Option<String>, default = no_schema_default)]
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
    /// JoVox-64, for voxel jobs; fetched from its repository when unset
    #[schema(value_type = Option<String>)]
    pub jovox: Option<PathBuf>,
    /// DINOv3 ViT-L/16, JoVox's image encoder; fetched from its gated repository when unset
    #[schema(value_type = Option<String>)]
    pub dinov3: Option<PathBuf>,
    /// BiRefNet, which removes the background of a voxel job's opaque image; fetched from its repository when unset
    #[schema(value_type = Option<String>)]
    pub birefnet: Option<PathBuf>,
    /// TRELLIS.2-4B, whose second stage meshes a voxel job's structure and which runs image-to-3D jobs; fetched from
    /// its repository when unset
    #[schema(value_type = Option<String>)]
    pub trellis2: Option<PathBuf>,
    /// TRELLIS-image-large, whose sparse-structure decoder TRELLIS.2 reuses; fetched from its repository when unset
    #[schema(value_type = Option<String>)]
    pub trellis_large: Option<PathBuf>,
    /// A Hugging Face repo laid out as a models root, fetched into the hub cache before a job that needs a checkpoint
    /// the fields above leave unset
    pub models_repo: Option<String>,
    /// The repository WorldMirror 2.0 is fetched from when neither `worldmirror` nor the models repo provides it
    pub worldmirror_repo: Option<String>,
    /// The CUDA devices jobs may run on; every visible one when unset
    pub devices: Option<Vec<usize>>,
    /// How much a job may displace to run, unless its request says otherwise
    pub placement: Placement,
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
            jovox: None,
            dinov3: None,
            birefnet: None,
            trellis2: None,
            trellis_large: None,
            models_repo: None,
            worldmirror_repo: Some(WORLDMIRROR_REPO.to_string()),
            devices: None,
            placement: Placement::Fastest,
            drain_timeout_secs: DEFAULT_DRAIN_TIMEOUT_SECS,
        }
    }
}

impl JobsConfig {
    /// The checkpoints found under `root` in their repositories' layout: `WorldStereo/worldstereo-memory-dmd`,
    /// `Wan2.1-I2V-14B-480P-Diffusers`, `moge-2-vitl-normal`, `HY-World-2.0/HY-WorldMirror-2.0`,
    /// `lpips-vgg/lpips_vgg.safetensors`, `JoVox-64`, `dinov3-vitl16-pretrain-lvd1689m` and `BiRefNet`; a missing one
    /// stays unset.
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
        self.jovox = found(JOVOX_DIR);
        self.dinov3 = found(DINOV3_DIR);
        self.birefnet = found(BIREFNET_DIR);
        self.trellis2 = found(TRELLIS2_DIR);
        self.trellis_large = found(TRELLIS_LARGE_DIR);
        self
    }

    // this config with the voxel job's unset checkpoints fetched from their repositories into the hub cache
    fn fetched_voxels(
        &self,
        birefnet: bool,
        mesh: bool,
        progress: &dyn Progress,
    ) -> anyhow::Result<Self> {
        let models = voxels::fetch_models(
            self.jovox.clone(),
            self.dinov3.clone(),
            birefnet.then(|| self.birefnet.clone()),
            mesh.then(|| self.trellis2.clone()),
            progress,
        )?;
        Ok(Self {
            jovox: Some(models.jovox),
            dinov3: Some(models.dinov3),
            birefnet: models.birefnet.or_else(|| self.birefnet.clone()),
            trellis2: models.trellis2.or_else(|| self.trellis2.clone()),
            ..self.clone()
        })
    }

    // whether a checkpoint is set, or will be fetched from a repository before the job runs
    fn provides(&self, path: &Option<PathBuf>) -> bool {
        path.is_some() || self.models_repo.is_some()
    }

    // this config with the checkpoints it leaves to repositories fetched into the hub cache; a reconstruction
    // (`world` false) reads only WorldMirror, which the models repo does not hold
    fn fetched(&self, world: bool, progress: &dyn Progress) -> anyhow::Result<Self> {
        let mut config = self.clone();
        let world_unset = [&self.worldstereo, &self.wan, &self.moge, &self.lpips]
            .iter()
            .any(|path| path.is_none());
        if let Some(repo) = self.models_repo.as_deref().filter(|_| world && world_unset) {
            let found = Self::default().with_models_root(&fetch_models_repo(repo, progress)?);
            config.worldstereo = config.worldstereo.or(found.worldstereo);
            config.wan = config.wan.or(found.wan);
            config.moge = config.moge.or(found.moge);
            config.lpips = config.lpips.or(found.lpips);
            config.worldmirror = config.worldmirror.or(found.worldmirror);
        }
        if config.worldmirror.is_none()
            && let Some(repo) = &self.worldmirror_repo
        {
            let folder = Path::new(WORLDMIRROR_DIR)
                .file_name()
                .and_then(|f| f.to_str())
                .expect("a checkpoint folder name");
            let repo = hub_api()?.model(repo.clone());
            progress.stage(DOWNLOAD_STAGE, WORLDMIRROR_FILES.len());
            for (i, file) in WORLDMIRROR_FILES.iter().enumerate() {
                inference_recon::progress::check(progress)?;
                let path = repo.get(&format!("{folder}/{file}"))?;
                config.worldmirror = path.parent().map(Path::to_path_buf);
                progress.advance(i + 1);
            }
        }
        Ok(config)
    }
}

pub(crate) fn hub_api() -> anyhow::Result<hf_hub::api::sync::Api> {
    inference_api::build_api_with_cache(&inference_api::TokenSource::CacheToken, false, None)
}

// The models repo's files in the hub cache, downloading what is missing; returns its snapshot folder
fn fetch_models_repo(repo: &str, progress: &dyn Progress) -> anyhow::Result<PathBuf> {
    let api = hub_api()?.model(repo.to_string());
    let files = match api.info() {
        Ok(info) => info
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|f| !REPO_SKIPPED.contains(&f.as_str()))
            .collect::<Vec<_>>(),
        Err(error) => {
            // the hub is out of reach: a snapshot fetched before serves
            let cached = inference_api::system::hf_hub_cache_dir().and_then(|dir| {
                hf_hub::Cache::new(dir)
                    .model(repo.to_string())
                    .get(MODELS_REPO_MARKER)
            });
            return cached
                .and_then(|marker| {
                    marker
                        .ancestors()
                        .nth(Path::new(MODELS_REPO_MARKER).components().count())
                        .map(Path::to_path_buf)
                })
                .ok_or_else(|| anyhow::anyhow!("fetching `{repo}`: {error}"));
        }
    };
    progress.stage(DOWNLOAD_STAGE, files.len());
    let mut root = None;
    for (i, file) in files.iter().enumerate() {
        inference_recon::progress::check(progress)?;
        let path = api.get(file)?;
        root = path
            .ancestors()
            .nth(Path::new(file).components().count())
            .map(Path::to_path_buf);
        progress.advance(i + 1);
    }
    root.ok_or_else(|| anyhow::anyhow!("`{repo}` holds no files"))
}

// a schema default for a field whose real one the server computes from its own environment
fn no_schema_default() -> serde_json::Value {
    serde_json::Value::Null
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
#[derive(Clone, Copy, Debug, Serialize, inference_json::FromJson, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SceneKind {
    Indoor,
    Outdoor,
}

/// A world from a panorama, the job's one input.
#[derive(Clone, Debug, Default, Serialize, inference_json::FromJson, ToSchema)]
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
    /// How much it may displace to run, in place of the server's setting
    #[serde(default)]
    pub placement: Option<Placement>,
}

/// WorldMirror 2.0 on the job's inputs: images, or one video.
#[derive(Clone, Debug, Default, Serialize, inference_json::FromJson, ToSchema)]
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
    /// How much it may displace to run, in place of the server's setting
    #[serde(default)]
    pub placement: Option<Placement>,
}

/// JoVox's 64^3 voxel structure of the object in the job's one image.
#[derive(Clone, Debug, Default, Serialize, inference_json::FromJson, ToSchema)]
pub struct VoxelsJob {
    #[serde(default)]
    pub seed: Option<u64>,
    /// Sampling steps, 36 to 108 (72 by default)
    #[serde(default)]
    pub steps: Option<usize>,
    /// Classifier-free guidance, at least 1 (1.5 by default)
    #[serde(default)]
    pub cfg_strength: Option<f32>,
    /// 0 (the default) takes each slot's most likely value
    #[serde(default)]
    pub temperature: Option<f32>,
    /// The share of the prepared image the object's longest side fills (0.74 by default)
    #[serde(default)]
    pub fill: Option<f32>,
    /// Remove the background with BiRefNet even when the image has transparency of its own (an opaque image always
    /// has it removed)
    #[serde(default)]
    pub remove_background: Option<bool>,
    /// Mesh the structure with TRELLIS.2's second stage (on by default)
    #[serde(default)]
    pub mesh: Option<bool>,
    /// How much it may displace to run, in place of the server's setting
    #[serde(default)]
    pub placement: Option<Placement>,
}

/// Which resolutions an image-to-3D job runs TRELLIS.2 at.
#[derive(
    Clone, Copy, Debug, Default, Serialize, inference_json::FromJson, ToSchema, PartialEq, Eq,
)]
pub enum Trellis2Variant {
    /// Shape at 512, then again at 1024 on the upsampled cells; texture at 1024 (the reference's default)
    #[default]
    #[serde(rename = "1024_cascade")]
    Cascade1024,
    /// Everything at 512: faster, coarser
    #[serde(rename = "512")]
    Res512,
    /// Everything at 1024, on a 64^3 structure
    #[serde(rename = "1024")]
    Res1024,
    /// Shape at 512, then again at 1536 (lower when that is 49152 cells or more); texture at 1024
    #[serde(rename = "1536_cascade")]
    Cascade1536,
}

/// Changes to one TRELLIS.2 flow's sampler; unset fields keep the checkpoint's `pipeline.json` values.
#[derive(Clone, Copy, Debug, Default, Serialize, inference_json::FromJson, ToSchema)]
pub struct Trellis2Sampler {
    /// Sampling steps, 1 to 100 (12 by default)
    #[serde(default)]
    pub steps: Option<u32>,
    /// Classifier-free guidance strength, 1 for none (7.5 for structure and shape, 1 for texture)
    #[serde(default)]
    pub guidance_strength: Option<f64>,
    /// How far guidance is rescaled toward the conditioned prediction's spread, 0 to 1
    #[serde(default)]
    pub guidance_rescale: Option<f64>,
    /// The timestep schedule's shift, above 0 (1: uniform)
    #[serde(default)]
    pub rescale_t: Option<f64>,
}

impl Trellis2Sampler {
    fn choice(self) -> crate::image_to_3d::SamplerChoice {
        crate::image_to_3d::SamplerChoice {
            steps: self.steps.map(|s| s as usize),
            guidance_strength: self.guidance_strength,
            guidance_rescale: self.guidance_rescale,
            rescale_t: self.rescale_t,
        }
    }
}

/// TRELLIS.2's textured mesh of the object in the job's one image.
#[derive(Clone, Debug, Default, Serialize, inference_json::FromJson, ToSchema)]
pub struct ImageTo3dJob {
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub variant: Option<Trellis2Variant>,
    /// Remove the background with BiRefNet even when the image has transparency of its own (an opaque image always
    /// has it removed)
    #[serde(default)]
    pub remove_background: Option<bool>,
    /// Per-vertex color and material instead of baked textures (faster: no unwrapping or baking)
    #[serde(default)]
    pub vertex_colors: Option<bool>,
    /// The baked textures' width and height (2048 by default)
    #[serde(default)]
    pub texture_size: Option<u32>,
    /// Rebuild the surface as a closed shell (narrow-band dual contouring) before baking
    #[serde(default)]
    pub remesh: Option<bool>,
    /// Faces the exported mesh is simplified to (1,000,000 by default)
    #[serde(default)]
    pub faces: Option<u32>,
    /// The sparse structure flow's sampler
    #[serde(default)]
    pub structure_sampler: Option<Trellis2Sampler>,
    /// The shape flows' sampler
    #[serde(default)]
    pub shape_sampler: Option<Trellis2Sampler>,
    /// The texture flow's sampler
    #[serde(default)]
    pub texture_sampler: Option<Trellis2Sampler>,
    /// How much it may displace to run, in place of the server's setting
    #[serde(default)]
    pub placement: Option<Placement>,
}

/// What a job does.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobRequest {
    /// WorldStereo-2 keyframes along camera paths from a panorama, WorldMirror 2.0 on them, then Gaussian splats
    World(WorldJob),
    /// WorldMirror 2.0 cameras, depth, points and splats from images or a video
    Reconstruct(ReconstructJob),
    /// JoVox's voxel structure of the object in one image
    Voxels(VoxelsJob),
    /// TRELLIS.2's textured mesh of the object in one image
    #[serde(rename = "image_to_3d")]
    ImageTo3d(ImageTo3dJob),
}

impl JobRequest {
    fn kind(&self) -> &'static str {
        match self {
            Self::World(_) => "world",
            Self::Reconstruct(_) => "reconstruct",
            Self::Voxels(_) => "voxels",
            Self::ImageTo3d(_) => "image_to_3d",
        }
    }
}

/// A file a job takes: its bytes, or a path on this machine.
pub enum JobInput {
    Bytes { name: String, data: Vec<u8> },
    Path(PathBuf),
}

#[derive(Clone, Copy, Debug, Serialize, inference_json::FromJson, ToSchema, PartialEq, Eq)]
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

#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
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
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
pub struct JobPreview {
    pub stage: String,
    pub caption: String,
    /// Relative to the job's folder
    pub file: String,
}

/// A file a job made.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
pub struct JobOutput {
    pub name: String,
    /// Relative to the job's folder
    pub file: String,
}

/// A job's record: what it was asked, how far it got, and what it made.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
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
    /// Why a queued job has not started: what it waits for
    #[serde(default)]
    pub waiting: Option<String>,
    /// The CUDA device it ran on
    #[serde(default)]
    pub device: Option<usize>,
    /// What the run reported (its `meta.json` fields, a training summary)
    #[serde(default)]
    #[schema(value_type = Object)]
    pub report: serde_json::Value,
}

/// A change to a running job.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    Status {
        status: JobStatus,
    },
    /// A queued job can't start yet, and why.
    Waiting {
        reason: String,
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
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
pub struct JobSubmission {
    pub request: JobRequest,
    #[serde(default)]
    #[schema(value_type = Vec<String>)]
    pub inputs: Vec<PathBuf>,
}

/// This owner's jobs, newest first.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
pub struct JobList {
    pub data: Vec<JobRecord>,
}

/// A file in a job's folder, given relative to it.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
pub struct JobFileRequest {
    pub id: String,
    pub file: String,
}

/// Where a job's file is on this machine.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
pub struct JobFilePath {
    pub path: String,
}

/// A job removed with its folder.
#[derive(Clone, Debug, Serialize, inference_json::FromJson, ToSchema)]
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
// what a job does before it takes a device, such as fetching checkpoints
type Prepare = Box<dyn FnOnce(&JobContext) -> anyhow::Result<()> + Send>;

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

// what a job takes, and where it may look for it
struct Target {
    need: JobNeed,
    devices: Vec<usize>,
    displace: Displace,
}

// a job's device need, its steps before taking the device and its work
struct Task {
    target: Option<Target>,
    prepare: Option<Prepare>,
    work: Work,
}

struct Queued {
    job: Arc<Job>,
    task: Task,
    // the submitter's engine, held only while its job waits or runs, so an idle queue keeps no engine alive
    engine: Engine,
}

/// The jobs beside an engine: their records, the queue and the worker that runs it. Cloning shares it; a server holds
/// one and passes each call the engine scoped to the caller, whose jobs are theirs alone.
#[derive(Clone, Default)]
pub struct JobQueue(Arc<JobRunner>);

struct JobRunner {
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

    fn sender(&self) -> std::sync::mpsc::Sender<Queued> {
        let mut queue = self
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        queue
            .get_or_insert_with(|| {
                let (tx, rx) = std::sync::mpsc::channel::<Queued>();
                let drain = Duration::from_secs(self.config().drain_timeout_secs);
                std::thread::Builder::new()
                    .name("inference-jobs".into())
                    .spawn(move || run_queue(rx, drain))
                    .expect("spawning the job worker");
                tx
            })
            .clone()
    }
}

// one job at a time: wait for the device, run, record how it ended; holds an engine only while a job does
fn run_queue(rx: std::sync::mpsc::Receiver<Queued>, drain: Duration) {
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
    for queued in rx {
        let (job, outcome) = run_job(queued, &runtime, drain);
        job.finish(outcome);
    }
}

// one job, its engine and closures dropped by the time it returns: once its outcome is out, a caller may end the
// process, and the engine's last reference must not be torn down on this thread as it exits
fn run_job(
    Queued {
        job,
        engine,
        task: Task {
            target,
            prepare,
            work,
        },
    }: Queued,
    runtime: &tokio::runtime::Runtime,
    drain: Duration,
) -> (Arc<Job>, anyhow::Result<JobResult>) {
    if job.cancel.load(Ordering::Relaxed) {
        return (job, Err(anyhow::anyhow!(CANCELLED)));
    }
    if let Some(prepare) = prepare {
        let context = JobContext {
            job: job.clone(),
            device: None,
            runtime: runtime.handle().clone(),
        };
        let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepare(&context)))
            .unwrap_or_else(|_| Err(anyhow::anyhow!("the job panicked")));
        if let Err(error) = prepared {
            return (job, Err(error));
        }
    }
    let (device, lease) = match &target {
        Some(target) => match runtime.block_on(take_device(&engine, &job, target, drain)) {
            Ok((device, lease)) => (Some(device), Some(lease)),
            Err(error) => return (job, Err(error)),
        },
        None => (None, None),
    };
    job.update(
        true,
        Some(JobEvent::Status {
            status: JobStatus::Running,
        }),
        |record| {
            record.status = JobStatus::Running;
            record.started_at = Some(now());
            record.waiting = None;
        },
    );
    let context = JobContext {
        job: job.clone(),
        device,
        runtime: runtime.handle().clone(),
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&context)))
        .unwrap_or_else(|_| Err(anyhow::anyhow!("the job panicked")));
    drop(context);
    drop(lease);
    (job, outcome)
}

// waits until the engine's placement takes a device for the job, then leases it; fails when none ever could
async fn take_device(
    engine: &Engine,
    job: &Job,
    target: &Target,
    drain: Duration,
) -> anyhow::Result<(Device, DeviceLease)> {
    let holder = {
        let record = job.record();
        format!("job {} ({})", record.id, record.kind)
    };
    loop {
        if job.cancel.load(Ordering::Relaxed) {
            anyhow::bail!(CANCELLED);
        }
        let placed = engine
            .place_need(Some(&target.devices), target.need, target.displace)
            .map_err(anyhow::Error::msg)?;
        match placed {
            Placed::Run { ordinal, evict } => {
                let unload = (target.need != JobNeed::Device).then_some(evict.as_slice());
                let lease = engine
                    .lease_device(ordinal, &holder, unload, drain)
                    .await
                    .map_err(anyhow::Error::msg)?;
                job.update(true, None, |record| record.device = Some(ordinal));
                return Ok((Device::new_cuda(ordinal)?, lease));
            }
            Placed::Wait(reason) => {
                let changed = job.record().waiting.as_deref() != Some(reason.as_str());
                if changed {
                    let event = JobEvent::Waiting {
                        reason: reason.clone(),
                    };
                    job.update(true, Some(event), |record| record.waiting = Some(reason));
                }
                tokio::time::sleep(WAIT_POLL).await;
            }
            Placed::Never(reason) => anyhow::bail!("no device can take this job: {reason}"),
        }
    }
}

// the CUDA devices jobs may use: the configured ones, or every visible one
fn candidate_devices(config: &JobsConfig) -> Result<Vec<usize>, ApiError> {
    inference_api::devices::candidate_devices(config.devices.as_deref())
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

// The job's config with its repositories' checkpoints, which its prepare step fetches before the work runs
fn fetching(config: &JobsConfig, world: bool) -> (Prepare, Arc<std::sync::OnceLock<JobsConfig>>) {
    let fetched = Arc::new(std::sync::OnceLock::new());
    let (config, slot) = (config.clone(), fetched.clone());
    let prepare: Prepare = Box::new(move |ctx: &JobContext| {
        let _ = slot.set(config.fetched(world, ctx)?);
        Ok(())
    });
    (prepare, fetched)
}

// a checkpoint a fetched config should hold
fn fetched_path(path: &Option<PathBuf>, what: &str) -> anyhow::Result<PathBuf> {
    path.clone()
        .ok_or_else(|| anyhow::anyhow!("the models repo holds no {what}"))
}

fn world_work(config: &JobsConfig, job: WorldJob) -> Result<(Prepare, Work), ApiError> {
    let need = |path: &Option<PathBuf>, what: &str| {
        config
            .provides(path)
            .then_some(())
            .ok_or_else(|| not_configured(what))
    };
    need(&config.worldstereo, "the WorldStereo-2 checkpoint")?;
    need(&config.wan, "the Wan 2.1 I2V checkpoint")?;
    need(&config.moge, "the MoGe-2 checkpoint")?;
    if !config.provides(&config.worldmirror) && config.worldmirror_repo.is_none() {
        return Err(not_configured("the WorldMirror 2.0 checkpoint"));
    }
    let splats = job.splats.unwrap_or(true);
    if splats {
        need(&config.lpips, "the LPIPS weights for splat training")?;
    }
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
    let (prepare, fetched) = fetching(config, true);
    let work: Work = Box::new(move |ctx: &JobContext| {
        let config = fetched.get().expect("prepared before the work runs");
        let models = WorldGenModels {
            worldstereo: fetched_path(&config.worldstereo, "WorldStereo-2 checkpoint")?,
            wan: fetched_path(&config.wan, "Wan 2.1 I2V checkpoint")?,
            moge: fetched_path(&config.moge, "MoGe-2 checkpoint")?,
            worldmirror: Some(fetched_path(
                &config.worldmirror,
                "WorldMirror 2.0 checkpoint",
            )?),
        };
        let lpips = splats
            .then(|| fetched_path(&config.lpips, "LPIPS weights"))
            .transpose()?;
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
    });
    Ok((prepare, work))
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

fn reconstruct_work(config: &JobsConfig, job: ReconstructJob) -> Result<(Prepare, Work), ApiError> {
    if config.worldmirror.is_none() && config.worldmirror_repo.is_none() {
        return Err(not_configured("the WorldMirror 2.0 checkpoint"));
    }
    let (prepare, fetched) = fetching(config, false);
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
    let work: Work = Box::new(move |ctx: &JobContext| {
        let config = fetched.get().expect("prepared before the work runs");
        let checkpoint = fetched_path(&config.worldmirror, "WorldMirror 2.0 checkpoint")?;
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
    });
    Ok((prepare, work))
}

fn voxels_work(config: &JobsConfig, job: VoxelsJob) -> Result<(Prepare, Work), ApiError> {
    use crate::voxels::{VoxelModels, VoxelOptions};
    let mut options = VoxelOptions::default();
    let s = &mut options.sample;
    s.seed = job.seed.unwrap_or(s.seed);
    s.steps = job.steps.unwrap_or(s.steps);
    s.cfg_strength = job.cfg_strength.unwrap_or(s.cfg_strength);
    s.temperature = job.temperature.unwrap_or(s.temperature);
    options.fill = job.fill.unwrap_or(options.fill);
    options.remove_background = job.remove_background.unwrap_or(false);
    options.mesh = job.mesh.unwrap_or(true);
    if !crate::voxels::STEPS_RANGE.contains(&s.steps) {
        return Err(ApiError::invalid_request(format!(
            "steps must be in {}..={}",
            crate::voxels::STEPS_RANGE.start(),
            crate::voxels::STEPS_RANGE.end()
        )));
    }
    if s.cfg_strength < 1.0 || s.temperature < 0.0 || !(options.fill > 0.0 && options.fill <= 1.0) {
        return Err(ApiError::invalid_request(
            "cfg_strength must be >= 1, temperature >= 0 and fill in (0, 1]",
        ));
    }
    let fetched = Arc::new(std::sync::OnceLock::new());
    let (base, slot) = (config.clone(), fetched.clone());
    let (remove, mesh) = (options.remove_background, options.mesh);
    let prepare: Prepare = Box::new(move |ctx: &JobContext| {
        let image = ctx.inputs().into_iter().next();
        let birefnet = match &image {
            Some(image) => crate::voxels::needs_background_removal(image, remove)?,
            None => false,
        };
        let _ = slot.set(base.fetched_voxels(birefnet, mesh, ctx)?);
        Ok(())
    });
    let work: Work = Box::new(move |ctx: &JobContext| {
        let config: &JobsConfig = fetched.get().expect("prepared before the work runs");
        let models = VoxelModels {
            jovox: fetched_path(&config.jovox, "JoVox-64 checkpoint")?,
            dinov3: fetched_path(&config.dinov3, "DINOv3 checkpoint")?,
            birefnet: config.birefnet.clone(),
            trellis2: config.trellis2.clone(),
        };
        let device = ctx.device()?.clone();
        let image = ctx
            .inputs()
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("a voxel job takes one image"))?;
        let out = ctx.dir().join(VOXELS_DIR);
        let report =
            crate::voxels::generate_voxels_on(&models, &image, &out, &options, ctx, &device)?;
        let dir = ctx.dir();
        let outputs = [
            ("mesh", "voxels/mesh.ply"),
            ("voxels", "voxels/voxels.ply"),
            ("prepared image", "voxels/prepared.png"),
            ("meta", "voxels/meta.json"),
        ]
        .into_iter()
        .filter_map(|(name, file)| output(dir, file, name))
        .collect();
        Ok(JobResult {
            outputs,
            report: serde_json::to_value(report)?,
        })
    });
    Ok((prepare, work))
}

fn image_to_3d_work(config: &JobsConfig, job: ImageTo3dJob) -> Result<(Prepare, Work), ApiError> {
    use crate::image_to_3d::{
        BakeOptions, DECIMATION_TARGET, FACES_RANGE, ImageTo3dOptions, Settings, Variant,
        fetch_models, generate_3d_on,
    };
    let variant = match job.variant.unwrap_or_default() {
        Trellis2Variant::Cascade1024 => Variant::Cascade1024,
        Trellis2Variant::Res512 => Variant::Res512,
        Trellis2Variant::Res1024 => Variant::Res1024,
        Trellis2Variant::Cascade1536 => Variant::Cascade1536,
    };
    let sampler = |s: Option<Trellis2Sampler>| s.unwrap_or_default().choice();
    let settings = Settings {
        structure: sampler(job.structure_sampler),
        shape: sampler(job.shape_sampler),
        texture: sampler(job.texture_sampler),
        ..Settings::new(variant, job.seed.unwrap_or(0))
    };
    for (stage, choice) in [
        ("structure_sampler", settings.structure),
        ("shape_sampler", settings.shape),
        ("texture_sampler", settings.texture),
    ] {
        if let Some(problem) = choice.problem() {
            return Err(ApiError::invalid_request(format!("{stage}: {problem}")));
        }
    }
    let faces = job.faces.map_or(DECIMATION_TARGET, |f| f as usize);
    if !FACES_RANGE.contains(&faces) {
        return Err(ApiError::invalid_request(format!(
            "faces must be in {}..={}",
            FACES_RANGE.start(),
            FACES_RANGE.end()
        )));
    }
    let options = ImageTo3dOptions {
        settings,
        faces,
        remove_background: job.remove_background.unwrap_or(false),
        bake: (!job.vertex_colors.unwrap_or(false)).then(|| BakeOptions {
            texture_size: job
                .texture_size
                .map_or(BakeOptions::default().texture_size, |s| s as usize),
            remesh: job.remesh.unwrap_or(false),
            ..BakeOptions::default()
        }),
    };
    let fetched = Arc::new(std::sync::OnceLock::new());
    let (base, slot) = (config.clone(), fetched.clone());
    let prepare: Prepare = Box::new(move |ctx: &JobContext| {
        let birefnet = match ctx.inputs().first() {
            Some(image) => {
                crate::voxels::needs_background_removal(image, options.remove_background)?
            }
            None => false,
        };
        let models = fetch_models(
            base.trellis2.clone(),
            base.trellis_large.clone(),
            base.dinov3.clone(),
            birefnet.then(|| base.birefnet.clone()),
            ctx,
        )?;
        let _ = slot.set(models);
        Ok(())
    });
    let work: Work = Box::new(move |ctx: &JobContext| {
        let models = fetched.get().expect("prepared before the work runs");
        let device = ctx.device()?.clone();
        let image = ctx
            .inputs()
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("an image-to-3D job takes one image"))?;
        let out = ctx.dir().join(IMAGE_TO_3D_DIR);
        let report = generate_3d_on(models, &image, &out, &options, ctx, &device)?;
        let dir = ctx.dir();
        let outputs = [
            ("model", "model/model.glb"),
            ("mesh", "model/mesh.ply"),
            ("prepared image", "model/prepared.png"),
            ("meta", "model/meta.json"),
        ]
        .into_iter()
        .filter_map(|(name, file)| output(dir, file, name))
        .collect();
        Ok(JobResult {
            outputs,
            report: serde_json::to_value(report)?,
        })
    });
    Ok((prepare, work))
}

impl JobQueue {
    /// Sets where jobs keep their folders and which weights they run; takes effect for jobs submitted afterwards.
    pub fn configure(&self, config: JobsConfig) {
        *self
            .0
            .config
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = config;
    }

    /// Queues a world generation or reconstruction with its input files for `engine`'s owner; its record as queued.
    pub fn submit(
        &self,
        engine: &Engine,
        request: JobRequest,
        inputs: Vec<JobInput>,
    ) -> Result<JobRecord, ApiError> {
        let config = self.0.config();
        let (need, placement, (prepare, work)) = match request.clone() {
            JobRequest::World(job) => {
                if inputs.len() != 1 {
                    return Err(ApiError::invalid_request("a world job takes one panorama"));
                }
                (JobNeed::Device, job.placement, world_work(&config, job)?)
            }
            JobRequest::Reconstruct(job) => {
                if inputs.is_empty() {
                    return Err(ApiError::invalid_request(
                        "a reconstruction takes images or a video",
                    ));
                }
                let placement = job.placement;
                let work = reconstruct_work(&config, job)?;
                (JobNeed::Bytes(RECONSTRUCT_NEED_BYTES), placement, work)
            }
            JobRequest::Voxels(job) => {
                if inputs.len() != 1 {
                    return Err(ApiError::invalid_request("a voxel job takes one image"));
                }
                let placement = job.placement;
                let work = voxels_work(&config, job)?;
                (JobNeed::Bytes(VOXELS_NEED_BYTES), placement, work)
            }
            JobRequest::ImageTo3d(job) => {
                if inputs.len() != 1 {
                    return Err(ApiError::invalid_request(
                        "an image-to-3D job takes one image",
                    ));
                }
                let placement = job.placement;
                let work = image_to_3d_work(&config, job)?;
                (JobNeed::Bytes(IMAGE_TO_3D_NEED_BYTES), placement, work)
            }
        };
        let target = Target {
            need,
            devices: candidate_devices(&config)?,
            displace: placement.unwrap_or(config.placement).displace(),
        };
        let kind = request.kind();
        let request = serde_json::to_value(&request).map_err(|_| ApiError::internal())?;
        let task = Task {
            target: Some(target),
            prepare: Some(prepare),
            work,
        };
        self.enqueue(engine, &config, kind, request, inputs, task)
    }

    /// Queues `work` as a job of `kind`, placed on a device with room for `need` by the engine's placement (none for
    /// work off the GPU); its record as queued. Its context reports progress through [`Progress`] and checks for
    /// cancellation.
    pub fn submit_task(
        &self,
        engine: &Engine,
        kind: &str,
        need: Option<JobNeed>,
        work: impl FnOnce(&JobContext) -> anyhow::Result<JobResult> + Send + 'static,
    ) -> Result<JobRecord, ApiError> {
        let config = self.0.config();
        let target = match need {
            Some(need) => Some(Target {
                need,
                devices: candidate_devices(&config)?,
                displace: config.placement.displace(),
            }),
            None => None,
        };
        let task = Task {
            target,
            prepare: None,
            work: Box::new(work),
        };
        self.enqueue(
            engine,
            &config,
            kind,
            serde_json::Value::Null,
            Vec::new(),
            task,
        )
    }

    fn enqueue(
        &self,
        engine: &Engine,
        config: &JobsConfig,
        kind: &str,
        request: serde_json::Value,
        inputs: Vec<JobInput>,
        task: Task,
    ) -> Result<JobRecord, ApiError> {
        let runner = &self.0;
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
            owner: engine.owner().map(ToString::to_string),
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
            waiting: None,
            device: None,
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
            .sender()
            .send(Queued {
                job,
                task,
                engine: engine.clone(),
            })
            .map_err(|_| ApiError::internal())?;
        Ok(record)
    }

    fn owned_job(&self, engine: &Engine, id: &str) -> Result<Arc<Job>, ApiError> {
        let job = self
            .0
            .jobs()
            .get(id)
            .cloned()
            .ok_or_else(|| not_found(id))?;
        if job.record().owner.as_deref() != engine.owner() {
            return Err(not_found(id));
        }
        Ok(job)
    }

    /// A job's record.
    pub fn get(&self, engine: &Engine, id: &str) -> Result<JobRecord, ApiError> {
        Ok(self.owned_job(engine, id)?.snapshot())
    }

    /// `engine`'s owner's jobs, newest first.
    pub fn list(&self, engine: &Engine) -> JobList {
        let mut records = self
            .0
            .jobs()
            .values()
            .map(|job| job.snapshot())
            .filter(|record| record.owner.as_deref() == engine.owner())
            .collect::<Vec<_>>();
        records.sort_by(|a, b| b.created_at.total_cmp(&a.created_at));
        JobList { data: records }
    }

    /// Asks a job to stop: a queued one never starts, a running one stops at its next step.
    pub fn cancel(&self, engine: &Engine, id: &str) -> Result<JobRecord, ApiError> {
        let job = self.owned_job(engine, id)?;
        job.cancel.store(true, Ordering::Relaxed);
        Ok(job.snapshot())
    }

    /// Removes a finished job and its folder.
    pub fn delete(&self, engine: &Engine, id: &str) -> Result<JobDeleted, ApiError> {
        let job = self.owned_job(engine, id)?;
        if !job.record().status.is_finished() {
            return Err(ApiError::new(
                ApiErrorKind::Conflict,
                format!("job `{id}` has not finished"),
                Some("job_not_finished"),
                Some("id"),
            ));
        }
        self.0
            .jobs
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
        std::fs::remove_dir_all(&job.dir).map_err(io_error)?;
        Ok(JobDeleted {
            id: id.to_string(),
            deleted: true,
        })
    }

    /// A job's changes from now on, after its record as it stands.
    pub fn events(
        &self,
        engine: &Engine,
        id: &str,
    ) -> Result<(JobRecord, broadcast::Receiver<JobEvent>), ApiError> {
        let job = self.owned_job(engine, id)?;
        let events = job.events.subscribe();
        Ok((job.snapshot(), events))
    }

    /// The path of a file in a job's folder, given relative to it.
    pub fn file(&self, engine: &Engine, id: &str, file: &str) -> Result<PathBuf, ApiError> {
        let job = self.owned_job(engine, id)?;
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
