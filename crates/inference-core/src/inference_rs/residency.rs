//! Which models stay resident: least-recently-used eviction of idle, unpinned models to make room for a load.

use std::sync::atomic::Ordering;
use std::time::Duration;

use inference_tensor::DeviceLocation;

use crate::*;

use super::{EngineInstance, InferenceRs, InferenceRsError, UnloadedModelState};

// a model that took a request this recently is not idle yet, though its engine has not seen it
const IDLE_GRACE_MS: u64 = 1_000;
// kept free beyond a model's measured footprint, for what its first requests allocate
const ROOM_MARGIN_BYTES: u64 = 512 << 20;
const RELOAD_WAIT_POLL: Duration = Duration::from_millis(100);
const LEASE_DRAIN_POLL: Duration = Duration::from_millis(100);

pub(super) type Leases = Arc<std::sync::Mutex<HashMap<DeviceLocation, String>>>;

/// A device held for one job: models are evicted from it and none load there until this is dropped.
pub struct DeviceLease {
    leases: Leases,
    location: DeviceLocation,
}

impl Drop for DeviceLease {
    fn drop(&mut self) {
        self.leases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.location);
    }
}

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Whether a load failed for want of device memory (an allocation, or a cache that could not get its need).
pub fn is_memory_error(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("out of memory")
        || m.contains("out_of_memory")
        || m.contains("best-effort minimum")
        || m.contains("safely available")
        || m.contains("of usable capacity")
        || m.contains("not enough memory")
}

fn same_device(a: &Device, b: &Device) -> bool {
    a.location() == b.location()
}

fn on_device(engine: &EngineInstance, device: &Device) -> bool {
    engine
        .reboot_state
        .loader_config
        .as_ref()
        .is_some_and(|config| same_device(&config.device, device))
}

// nothing running, waiting or queued, and no request reached it within the grace
fn is_quiet(engine: &EngineInstance, now: u64) -> bool {
    engine.logger.num_running() == 0
        && engine.logger.num_waiting() == 0
        && engine.sender.capacity() == engine.sender.max_capacity()
        && now.saturating_sub(engine.last_used.load(Ordering::Relaxed)) > IDLE_GRACE_MS
}

#[cfg(feature = "cuda")]
fn free_bytes(device: &Device) -> Option<u64> {
    if !device.is_cuda() {
        return None;
    }
    crate::utils::memory_usage::MemoryUsage
        .query(device)
        .ok()
        .map(|m| m.available() as u64)
}

#[cfg(not(feature = "cuda"))]
fn free_bytes(_device: &Device) -> Option<u64> {
    None
}

impl InferenceRs {
    // resident models on `device` other than `exclude` that are unpinned and idle, least recently used first
    fn eviction_candidates(
        &self,
        exclude: &str,
        device: &Device,
    ) -> Result<Vec<String>, InferenceRsError> {
        let engines = self
            .engines
            .read()
            .map_err(|_| InferenceRsError::EnginePoisoned)?;
        let now = now_ms();
        let mut idle = engines
            .iter()
            .filter(|(id, engine)| {
                id.as_str() != exclude
                    && on_device(engine, device)
                    && !engine.pinned
                    && is_quiet(engine, now)
            })
            .map(|(id, engine)| (engine.last_used.load(Ordering::Relaxed), id.clone()))
            .collect::<Vec<_>>();
        idle.sort();
        Ok(idle.into_iter().map(|(_, id)| id).collect())
    }

    /// Unloads the least recently used idle, unpinned model on `device` other than `exclude`; whether one was.
    pub fn evict_one(&self, exclude: &str, device: &Device) -> Result<bool, InferenceRsError> {
        for id in self.eviction_candidates(exclude, device)? {
            match self.unload_model(&id) {
                Ok(()) => {
                    tracing::info!("Evicted {id} to make room for {exclude}");
                    return Ok(true);
                }
                // taken by a request or another unload meanwhile
                Err(_) => continue,
            }
        }
        Ok(false)
    }

    // evicts until `bytes` plus a margin are free on `device`, or nothing more can go
    fn make_room(
        &self,
        exclude: &str,
        device: &Device,
        bytes: u64,
    ) -> Result<(), InferenceRsError> {
        while let Some(free) = free_bytes(device) {
            if free >= bytes + ROOM_MARGIN_BYTES || !self.evict_one(exclude, device)? {
                break;
            }
        }
        Ok(())
    }

    // Loads an unloaded or declared model, making room for it: evictions sized from its last footprint, then on a
    // memory failure one more eviction and a retry; once nothing can go, a default cache may take less than its need.
    pub(super) async fn load_with_room(
        &self,
        model_id: &str,
        state: UnloadedModelState,
    ) -> Result<(), InferenceRsError> {
        let device = state.loader_config.device.clone();
        self.check_device_free(&device)?;
        if let Some(footprint) = state.footprint {
            self.make_room(model_id, &device, footprint.bytes)?;
        }
        let mut need_floor = true;
        loop {
            let mut attempt = state.clone();
            if let Some(cache) = attempt.loader_config.paged_attn_config.as_mut() {
                cache.need_floor = need_floor;
            }
            match self.do_reload_model(model_id, attempt).await {
                Err(InferenceRsError::ReloadFailed(message)) if is_memory_error(&message) => {
                    if self.evict_one(model_id, &device)? {
                        continue;
                    }
                    if need_floor {
                        need_floor = false;
                        continue;
                    }
                    return Err(InferenceRsError::ReloadFailed(message));
                }
                result => return result,
            }
        }
    }

    /// Waits for `model_id` to be resident, loading it if it is unloaded and waiting out another caller's load.
    pub(super) async fn ensure_resident(&self, model_id: &str) -> Result<(), InferenceRsError> {
        loop {
            if self
                .engines
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?
                .contains_key(model_id)
            {
                return Ok(());
            }
            let unloaded = self
                .unloaded_models
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?
                .contains_key(model_id);
            let reloading = self
                .reloading_models
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?
                .contains(model_id);
            if !unloaded && !reloading {
                return Err(InferenceRsError::ModelNotFound(model_id.to_string()));
            }
            if unloaded && !reloading {
                match self.reload_model(model_id).await {
                    Ok(()) | Err(InferenceRsError::ModelAlreadyLoaded(_)) => return Ok(()),
                    Err(InferenceRsError::ModelReloading(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            tokio::time::sleep(RELOAD_WAIT_POLL).await;
        }
    }

    /// What holds `device`, when a job does.
    pub fn device_holder(&self, device: &Device) -> Option<String> {
        self.leases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&device.location())
            .cloned()
    }

    /// Fails with [`InferenceRsError::DeviceBusy`] while a job holds `device`.
    pub fn check_device_free(&self, device: &Device) -> Result<(), InferenceRsError> {
        match self.device_holder(device) {
            Some(holder) => Err(InferenceRsError::DeviceBusy(format!("held by {holder}"))),
            None => Ok(()),
        }
    }

    /// Takes `device` for `holder`: from then on nothing loads there, and each model resident there is unloaded once
    /// it goes quiet. Fails if another job holds it, a pinned model is resident there, or a model stays busy past
    /// `drain_timeout`.
    pub async fn lease_device(
        &self,
        device: &Device,
        holder: &str,
        drain_timeout: Duration,
    ) -> Result<DeviceLease, InferenceRsError> {
        let location = device.location();
        {
            let mut leases = self
                .leases
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(other) = leases.get(&location) {
                return Err(InferenceRsError::DeviceBusy(format!("held by {other}")));
            }
            leases.insert(location, holder.to_string());
        }
        let lease = DeviceLease {
            leases: self.leases.clone(),
            location,
        };
        let deadline = std::time::Instant::now() + drain_timeout;
        loop {
            let now = now_ms();
            let resident = self
                .engines
                .read()
                .map_err(|_| InferenceRsError::EnginePoisoned)?
                .iter()
                .filter(|(_, engine)| on_device(engine, device))
                .map(|(id, engine)| (id.clone(), engine.pinned, is_quiet(engine, now)))
                .collect::<Vec<_>>();
            if let Some((id, ..)) = resident.iter().find(|(_, pinned, _)| *pinned) {
                return Err(InferenceRsError::DeviceBusy(format!(
                    "model `{id}` is pinned there"
                )));
            }
            let loading_here = {
                let reloading = self
                    .reloading_models
                    .read()
                    .map_err(|_| InferenceRsError::EnginePoisoned)?
                    .clone();
                let unloaded = self
                    .unloaded_models
                    .read()
                    .map_err(|_| InferenceRsError::EnginePoisoned)?;
                reloading.iter().any(|id| {
                    unloaded
                        .get(id)
                        .is_some_and(|state| same_device(&state.loader_config.device, device))
                })
            };
            if resident.is_empty() && !loading_here {
                return Ok(lease);
            }
            for (id, _, quiet) in &resident {
                if !quiet {
                    continue;
                }
                match self.unload_model(id) {
                    Ok(()) => tracing::info!("Unloaded {id} for {holder}"),
                    Err(
                        InferenceRsError::ModelAlreadyUnloaded(_)
                        | InferenceRsError::ModelNotFound(_),
                    ) => {}
                    Err(error) => return Err(error),
                }
            }
            if std::time::Instant::now() >= deadline {
                let busy = resident.iter().find(|(_, _, quiet)| !quiet).map_or_else(
                    || "a model is loading".to_string(),
                    |(id, ..)| format!("model `{id}` stayed busy"),
                );
                return Err(InferenceRsError::DeviceBusy(busy));
            }
            tokio::time::sleep(LEASE_DRAIN_POLL).await;
        }
    }
}
