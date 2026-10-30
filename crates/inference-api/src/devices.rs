//! The engine's GPUs as work outside the engine sees them: what each holds, where a need would go, and a lease that
//! keeps models off a device while that work runs.

use std::time::Duration;

pub use inference_core::DeviceLease;
pub use inference_core::placement::{DeviceSnapshot, Displace, JobNeed, Placed, ResidentModel};
use inference_tensor::Device;
use serde::Serialize;

use crate::api_error::{ApiError, ApiErrorKind};

/// How long a lease waits for busy models to finish their requests unless the request says otherwise.
pub const DEFAULT_DRAIN_TIMEOUT_SECS: u64 = 120;

/// Which devices a call looks at; every visible CUDA device when `devices` is absent.
#[derive(Debug, Default, inference_json::FromJson)]
pub struct DevicesRequest {
    #[serde(default)]
    pub devices: Option<Vec<usize>>,
}

/// How much of a device a need takes.
#[derive(Debug, Clone, Copy, inference_json::FromJson, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NeedJson {
    /// The whole device: every model there unloads.
    Device,
    /// This many bytes; models that leave room for them stay.
    Bytes(u64),
}

impl From<NeedJson> for JobNeed {
    fn from(need: NeedJson) -> Self {
        match need {
            NeedJson::Device => JobNeed::Device,
            NeedJson::Bytes(bytes) => JobNeed::Bytes(bytes),
        }
    }
}

/// Which resident models a need may unload.
#[derive(Debug, Clone, Copy, Default, inference_json::FromJson)]
#[serde(rename_all = "snake_case")]
pub enum DisplaceJson {
    #[default]
    Busy,
    Idle,
    Nothing,
}

impl From<DisplaceJson> for Displace {
    fn from(displace: DisplaceJson) -> Self {
        match displace {
            DisplaceJson::Busy => Displace::Busy,
            DisplaceJson::Idle => Displace::Idle,
            DisplaceJson::Nothing => Displace::Nothing,
        }
    }
}

/// Where `need` would go among `devices` now.
#[derive(Debug, inference_json::FromJson)]
pub struct PlaceRequest {
    #[serde(default)]
    pub devices: Option<Vec<usize>>,
    pub need: NeedJson,
    #[serde(default)]
    pub displace: DisplaceJson,
}

/// Holds `device` for `holder`, unloading `unload` (every model there when absent) once each goes quiet.
#[derive(Debug, inference_json::FromJson)]
pub struct LeaseRequest {
    pub device: usize,
    pub holder: String,
    #[serde(default)]
    pub unload: Option<Vec<String>>,
    #[serde(default)]
    pub drain_timeout_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ResidentModelJson {
    pub id: String,
    pub bytes: Option<u64>,
    pub pinned: bool,
    pub quiet: bool,
    pub last_used: u64,
}

#[derive(Debug, Serialize)]
pub struct DeviceSnapshotJson {
    pub device: usize,
    pub free: u64,
    pub total: u64,
    pub held_by: Option<String>,
    pub models: Vec<ResidentModelJson>,
}

impl From<DeviceSnapshot> for DeviceSnapshotJson {
    fn from(snapshot: DeviceSnapshot) -> Self {
        Self {
            device: snapshot.ordinal,
            free: snapshot.free,
            total: snapshot.total,
            held_by: snapshot.held_by,
            models: snapshot
                .models
                .into_iter()
                .map(|model| ResidentModelJson {
                    id: model.id,
                    bytes: model.bytes,
                    pinned: model.pinned,
                    quiet: model.quiet,
                    last_used: model.last_used,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DeviceSnapshotList {
    pub data: Vec<DeviceSnapshotJson>,
}

/// A placement: take `device` unloading `evict`, wait (why), or never (why).
#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PlacedJson {
    Run { device: usize, evict: Vec<String> },
    Wait { reason: String },
    Never { reason: String },
}

impl From<Placed> for PlacedJson {
    fn from(placed: Placed) -> Self {
        match placed {
            Placed::Run { ordinal, evict } => Self::Run {
                device: ordinal,
                evict,
            },
            Placed::Wait(reason) => Self::Wait { reason },
            Placed::Never(reason) => Self::Never { reason },
        }
    }
}

/// The CUDA devices `devices` names, or every visible one; an error when there are none or one is not visible.
pub fn candidate_devices(devices: Option<&[usize]>) -> Result<Vec<usize>, ApiError> {
    #[cfg(feature = "cuda")]
    let visible = inference_tensor::cuda_backend::device_count().unwrap_or(0);
    #[cfg(not(feature = "cuda"))]
    let visible = 0;
    let devices = match devices {
        Some(devices) => devices.to_vec(),
        None => (0..visible).collect(),
    };
    if devices.is_empty() || devices.iter().any(|&d| d >= visible) {
        return Err(ApiError::new(
            ApiErrorKind::Unavailable,
            format!("this needs a CUDA device; {visible} visible, {devices:?} asked for"),
            Some("device_unavailable"),
            None,
        ));
    }
    Ok(devices)
}

pub(crate) fn cuda_device(ordinal: usize) -> Result<Device, ApiError> {
    Device::new_cuda(ordinal).map_err(|error| {
        ApiError::new(
            ApiErrorKind::Unavailable,
            format!("CUDA device {ordinal}: {error}"),
            Some("device_unavailable"),
            None,
        )
    })
}

pub(crate) fn drain_timeout(secs: Option<u64>) -> Duration {
    Duration::from_secs(secs.unwrap_or(DEFAULT_DRAIN_TIMEOUT_SECS))
}
