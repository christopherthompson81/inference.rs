use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

const CUDA_GRAPHS_ENV: &str = "INFERENCE_RS_CUDA_GRAPHS";

static CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
// set by a loaded model whose forward waits on the host, which a captured graph cannot replay
static CUDA_GRAPHS_RULED_OUT: AtomicBool = AtomicBool::new(false);

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on") {
                true
            } else if matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off") {
                false
            } else {
                default
            }
        })
        .unwrap_or(default)
}

pub fn cuda_graphs_enabled() -> bool {
    !CUDA_GRAPHS_RULED_OUT.load(Ordering::Relaxed)
        && *CUDA_GRAPHS_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_ENV, true))
}

/// Turns CUDA graphs off for the rest of the process: a layer of the loaded model syncs with the host mid-forward.
pub fn rule_out_cuda_graphs() {
    CUDA_GRAPHS_RULED_OUT.store(true, Ordering::Relaxed);
}
