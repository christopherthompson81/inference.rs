//! Opt-in per-op timing for the long-running tools (`INFERENCE_PROFILE=1`): named spans, each ended by a sync of its
//! device so a span's time is the op's whole cost (its queued kernels included), summed per name and thread with
//! their counts and written out by [`report`]. Unset, a span is one cached check. The syncs serialize what would
//! overlap, so a profiled run's total is not the plain run's; the per-op costs are what it measures.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::Device;

const PROFILE_ENV: &str = "INFERENCE_PROFILE";

// (span name, thread name) -> (time, calls)
type Totals = BTreeMap<(&'static str, String), (Duration, u64)>;

fn totals() -> &'static Mutex<Totals> {
    static TOTALS: OnceLock<Mutex<Totals>> = OnceLock::new();
    TOTALS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Whether `INFERENCE_PROFILE` is set.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os(PROFILE_ENV).is_some_and(|v| !v.is_empty() && v != "0"))
}

/// A running span: its time (to `device`'s sync, when one is given) is added to `name`'s total when it drops.
pub struct Span {
    name: &'static str,
    device: Option<Device>,
    start: Instant,
}

/// A span over an op on `device` (`None` for host work), or nothing when profiling is off.
pub fn span(name: &'static str, device: Option<&Device>) -> Option<Span> {
    enabled().then(|| Span {
        name,
        device: device.cloned(),
        start: Instant::now(),
    })
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(device) = &self.device {
            // an unwinding span still records its time; a failed sync leaves the time as measured
            let _ = device.synchronize();
        }
        let elapsed = self.start.elapsed();
        let thread = std::thread::current().name().map_or_else(
            || format!("{:?}", std::thread::current().id()),
            str::to_string,
        );
        let mut totals = totals().lock().unwrap_or_else(|e| e.into_inner());
        let entry = totals.entry((self.name, thread)).or_default();
        entry.0 += elapsed;
        entry.1 += 1;
    }
}

/// The spans so far, longest first: total seconds, calls, mean milliseconds, name and thread; empty when off.
pub fn report() -> String {
    if !enabled() {
        return String::new();
    }
    let totals = totals().lock().unwrap_or_else(|e| e.into_inner());
    let mut rows = totals.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| std::cmp::Reverse(row.1.0));
    let mut out = format!(
        "{:>10} {:>8} {:>10}  span (thread)\n",
        "seconds", "calls", "mean ms"
    );
    for ((name, thread), (time, calls)) in rows {
        let mean = time.as_secs_f64() * 1e3 / (*calls).max(1) as f64;
        out += &format!(
            "{:>10.2} {:>8} {:>10.2}  {name} ({thread})\n",
            time.as_secs_f64(),
            calls,
            mean
        );
    }
    out
}
