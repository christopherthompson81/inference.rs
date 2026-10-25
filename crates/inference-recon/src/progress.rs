//! How a long pipeline reports its stages and previews, and learns that it should stop.

use std::path::Path;

use inference_tensor::Result;

/// The error a pipeline stops with when its caller cancels it.
pub const CANCELLED: &str = "cancelled";

/// A run's stage, progress and preview hooks, and its cancellation flag; each method defaults to doing nothing.
pub trait Progress: Sync {
    /// A stage starts, `total` steps long (0 when it has no count).
    fn stage(&self, _name: &str, _total: usize) {}
    /// `done` of the current stage's steps are finished.
    fn advance(&self, _done: usize) {}
    /// An image the run wrote under its output folder that shows how it is going.
    fn preview(&self, _caption: &str, _path: &Path) {}
    /// Whether the caller asked the run to stop; checked between steps.
    fn cancelled(&self) -> bool {
        false
    }
}

/// Reports nothing and never stops a run.
pub struct NoProgress;

impl Progress for NoProgress {}

/// Fails with [`CANCELLED`] once the caller has asked the run to stop.
pub fn check(progress: &dyn Progress) -> Result<()> {
    if progress.cancelled() {
        inference_tensor::bail!("{CANCELLED}");
    }
    Ok(())
}
