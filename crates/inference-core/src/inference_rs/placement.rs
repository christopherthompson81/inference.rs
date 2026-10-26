//! Where a job runs: the device it takes and the models it unloads there, from a snapshot of each candidate device.

/// Kept free beyond a sized job's need, for what its first allocations and the CUDA pool's slack take.
pub const NEED_MARGIN_BYTES: u64 = 512 << 20;

/// What a job takes of a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobNeed {
    /// The whole device: every model there unloads.
    Device,
    /// This much memory; models that leave room for it stay.
    Bytes(u64),
}

/// Which resident models a job may unload to make room.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Displace {
    /// Any unpinned model, idle ones first, waiting for busy ones to finish their requests.
    #[default]
    Busy,
    /// Only models with nothing running or waiting; while a busy one is in the way the job waits.
    Idle,
    /// None: the job runs only where it fits beside what is resident.
    Nothing,
}

/// A model resident on a device, as a placement sees it.
#[derive(Clone, Debug)]
pub struct ResidentModel {
    pub id: String,
    /// Its measured footprint; unknown for a model loaded before footprints, or loading now.
    pub bytes: Option<u64>,
    pub pinned: bool,
    /// Nothing running or waiting, and no request just in.
    pub quiet: bool,
    /// When a request last reached it (unix ms), for least-recently-used order.
    pub last_used: u64,
}

/// One candidate device as it stands.
#[derive(Clone, Debug)]
pub struct DeviceSnapshot {
    pub ordinal: usize,
    pub free: u64,
    pub total: u64,
    /// The job holding it, if one does.
    pub held_by: Option<String>,
    pub models: Vec<ResidentModel>,
}

/// What a job should do now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placed {
    /// Take device `ordinal`, unloading `evict` (every model there for a whole-device need).
    Run { ordinal: usize, evict: Vec<String> },
    /// Nothing fits yet; why, to show while it waits.
    Wait(String),
    /// No candidate can ever take it as things are configured.
    Never(String),
}

const GIB: u64 = 1 << 30;

// to a tenth of a GiB, rounded
fn gib(bytes: u64) -> String {
    let tenths = (bytes * 10 + GIB / 2) / GIB;
    format!("{}.{} GiB", tenths / 10, tenths % 10)
}

// why one device can't take the job: not yet, or not as things are configured
enum Blocked {
    Wait(String),
    Never(String),
}

// (serving models displaced, models displaced, bytes displaced): lower is cheaper
type Cost = (usize, usize, u64);

fn whole(device: &DeviceSnapshot, displace: Displace) -> Result<(Cost, Vec<String>), Blocked> {
    let o = device.ordinal;
    if let Some(model) = device.models.iter().find(|m| m.pinned) {
        return Err(Blocked::Never(format!(
            "GPU {o}: model `{}` is pinned there",
            model.id
        )));
    }
    let busy = device
        .models
        .iter()
        .filter(|m| !m.quiet)
        .collect::<Vec<_>>();
    if displace == Displace::Nothing
        && let Some(model) = device.models.first()
    {
        return Err(Blocked::Wait(format!(
            "GPU {o}: model `{}` is loaded there",
            model.id
        )));
    }
    if displace == Displace::Idle
        && let Some(model) = busy.first()
    {
        return Err(Blocked::Wait(format!(
            "GPU {o}: model `{}` is serving requests",
            model.id
        )));
    }
    let bytes = device.models.iter().filter_map(|m| m.bytes).sum();
    let evict = device.models.iter().map(|m| m.id.clone()).collect();
    Ok(((busy.len(), device.models.len(), bytes), evict))
}

fn sized(
    device: &DeviceSnapshot,
    need: u64,
    displace: Displace,
) -> Result<(Cost, Vec<String>), Blocked> {
    let o = device.ordinal;
    let need = need + NEED_MARGIN_BYTES;
    if need > device.total {
        return Err(Blocked::Never(format!(
            "GPU {o}: {} needed, the card has {}",
            gib(need),
            gib(device.total)
        )));
    }
    if device.free >= need {
        return Ok(((0, 0, 0), Vec::new()));
    }
    // idle models first, least recently used first, then (if allowed) the busy ones
    let mut order = device
        .models
        .iter()
        .filter(|m| !m.pinned)
        .collect::<Vec<_>>();
    order.sort_by_key(|m| (!m.quiet, m.last_used));
    let short = |freed: u64| device.free + freed < need;
    let (mut freed, mut evict, mut busy) = (0, Vec::new(), 0);
    for model in &order {
        if !short(freed) {
            break;
        }
        let allowed = match displace {
            Displace::Busy => true,
            Displace::Idle => model.quiet,
            Displace::Nothing => false,
        };
        if !allowed {
            let why = if displace == Displace::Nothing {
                format!(
                    "GPU {o}: {} needed, {} free beside the loaded models",
                    gib(need),
                    gib(device.free)
                )
            } else {
                format!(
                    "GPU {o}: making room would unload `{}`, which is serving requests",
                    model.id
                )
            };
            return Err(Blocked::Wait(why));
        }
        freed += model.bytes.unwrap_or(0);
        busy += usize::from(!model.quiet);
        evict.push(model.id.clone());
    }
    if short(freed) {
        let unpinned_known = order.iter().all(|m| m.bytes.is_some());
        let pinned = device.models.iter().any(|m| m.pinned);
        let why = format!(
            "GPU {o}: {} needed, {} free once every unpinned model unloads",
            gib(need),
            gib(device.free + freed)
        );
        // only pinned models (or memory outside the engine) stand in the way, and they won't leave on their own
        return Err(if unpinned_known && pinned {
            Blocked::Never(why)
        } else {
            Blocked::Wait(why)
        });
    }
    Ok(((busy, evict.len(), freed), evict))
}

/// Where a job with `need` runs now among `devices`, unloading only what `displace` allows: a device it fits on as it
/// is wins, then the one that displaces the fewest serving models, then the fewest models, then the least memory.
pub fn place(devices: &[DeviceSnapshot], need: JobNeed, displace: Displace) -> Placed {
    let mut best: Option<(Cost, u64, usize, Vec<String>)> = None;
    let (mut waits, mut refusals) = (Vec::new(), Vec::new());
    for device in devices {
        if let Some(holder) = &device.held_by {
            waits.push(format!("GPU {} is held by {holder}", device.ordinal));
            continue;
        }
        let planned = match need {
            JobNeed::Device => whole(device, displace),
            JobNeed::Bytes(bytes) => sized(device, bytes, displace),
        };
        match planned {
            Ok((cost, evict)) => {
                // ties go to the device with more free memory
                let better = best.as_ref().is_none_or(|(c, free, ..)| {
                    (cost, std::cmp::Reverse(device.free)) < (*c, std::cmp::Reverse(*free))
                });
                if better {
                    best = Some((cost, device.free, device.ordinal, evict));
                }
            }
            Err(Blocked::Wait(why)) => waits.push(why),
            Err(Blocked::Never(why)) => refusals.push(why),
        }
    }
    match best {
        Some((_, _, ordinal, evict)) => Placed::Run { ordinal, evict },
        None if !waits.is_empty() => Placed::Wait(waits.join("; ")),
        None if !refusals.is_empty() => Placed::Never(refusals.join("; ")),
        None => Placed::Never("no CUDA device to run on".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn model(id: &str, gb: u64, quiet: bool, last_used: u64) -> ResidentModel {
        ResidentModel {
            id: id.to_string(),
            bytes: Some(gb * GB),
            pinned: false,
            quiet,
            last_used,
        }
    }

    fn device(ordinal: usize, free_gb: u64, models: Vec<ResidentModel>) -> DeviceSnapshot {
        DeviceSnapshot {
            ordinal,
            free: free_gb * GB,
            total: 24 * GB,
            held_by: None,
            models,
        }
    }

    fn run(ordinal: usize, evict: &[&str]) -> Placed {
        Placed::Run {
            ordinal,
            evict: evict.iter().map(ToString::to_string).collect(),
        }
    }

    #[test]
    fn a_sized_job_that_fits_unloads_nothing() {
        let devices = [device(0, 14, vec![model("chat", 8, false, 1)])];
        for displace in [Displace::Busy, Displace::Idle, Displace::Nothing] {
            assert_eq!(
                place(&devices, JobNeed::Bytes(10 * GB), displace),
                run(0, &[])
            );
        }
    }

    #[test]
    fn a_sized_job_unloads_the_least_recently_used_idle_model_first() {
        let devices = [device(
            0,
            4,
            vec![
                model("busy", 6, false, 1),
                model("new", 6, true, 9),
                model("old", 6, true, 2),
            ],
        )];
        assert_eq!(
            place(&devices, JobNeed::Bytes(9 * GB), Displace::Busy),
            run(0, &["old"])
        );
        assert_eq!(
            place(&devices, JobNeed::Bytes(9 * GB), Displace::Idle),
            run(0, &["old"])
        );
        assert!(matches!(
            place(&devices, JobNeed::Bytes(9 * GB), Displace::Nothing),
            Placed::Wait(_)
        ));
    }

    #[test]
    fn only_the_fastest_placement_displaces_a_serving_model() {
        let devices = [device(
            0,
            2,
            vec![model("idle", 4, true, 1), model("busy", 8, false, 5)],
        )];
        assert_eq!(
            place(&devices, JobNeed::Bytes(10 * GB), Displace::Busy),
            run(0, &["idle", "busy"])
        );
        let waiting = place(&devices, JobNeed::Bytes(10 * GB), Displace::Idle);
        assert!(
            matches!(&waiting, Placed::Wait(why) if why.contains("`busy`")),
            "{waiting:?}"
        );
    }

    #[test]
    fn a_job_goes_where_it_displaces_least() {
        let devices = [
            device(0, 4, vec![model("busy", 12, false, 1)]),
            device(1, 4, vec![model("idle", 12, true, 1)]),
            device(2, 16, vec![model("small", 4, false, 1)]),
        ];
        assert_eq!(
            place(&devices, JobNeed::Bytes(10 * GB), Displace::Busy),
            run(2, &[])
        );
        assert_eq!(
            place(&devices[..2], JobNeed::Bytes(10 * GB), Displace::Busy),
            run(1, &["idle"])
        );
        assert_eq!(
            place(&devices[..2], JobNeed::Device, Displace::Busy),
            run(1, &["idle"])
        );
    }

    #[test]
    fn a_whole_device_job_waits_for_serving_models_unless_it_may_displace_them() {
        let devices = [device(0, 10, vec![model("chat", 8, false, 1)])];
        assert_eq!(
            place(&devices, JobNeed::Device, Displace::Busy),
            run(0, &["chat"])
        );
        assert!(matches!(
            place(&devices, JobNeed::Device, Displace::Idle),
            Placed::Wait(_)
        ));
        let empty = [device(0, 23, Vec::new())];
        assert_eq!(
            place(&empty, JobNeed::Device, Displace::Nothing),
            run(0, &[])
        );
    }

    #[test]
    fn pinned_models_and_small_cards_refuse_what_waiting_cannot_fix() {
        let mut pinned = model("kept", 16, true, 1);
        pinned.pinned = true;
        let devices = [device(0, 4, vec![pinned, model("idle", 2, true, 1)])];
        assert!(matches!(
            place(&devices, JobNeed::Device, Displace::Busy),
            Placed::Never(_)
        ));
        assert!(matches!(
            place(&devices, JobNeed::Bytes(10 * GB), Displace::Busy),
            Placed::Never(_)
        ));
        let small = [device(0, 23, Vec::new())];
        assert!(matches!(
            place(&small, JobNeed::Bytes(30 * GB), Displace::Busy),
            Placed::Never(_)
        ));
    }

    #[test]
    fn a_held_device_makes_the_job_wait_for_it() {
        let mut held = device(0, 23, Vec::new());
        held.held_by = Some("job one".into());
        let waiting = place(&[held.clone()], JobNeed::Bytes(GB), Displace::Busy);
        assert!(
            matches!(&waiting, Placed::Wait(why) if why.contains("job one")),
            "{waiting:?}"
        );
        assert_eq!(
            place(
                &[held, device(1, 20, Vec::new())],
                JobNeed::Bytes(GB),
                Displace::Busy
            ),
            run(1, &[])
        );
    }
}
