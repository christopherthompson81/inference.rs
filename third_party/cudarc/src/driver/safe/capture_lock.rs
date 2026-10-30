//! CUDA rejects some calls while any stream of a context captures a graph, and they invalidate that capture: a
//! context synchronize, a module unload, a memory pool trim or attribute change, peer access setup. Threads sharing
//! a context (several models in one process) hold this lock shared across a capture and exclusively across those
//! calls. A module unload fails silently in `Drop`, so [`super::CudaModule`] takes it there.

use std::cell::Cell;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

static CONTEXT_WIDE: RwLock<()> = RwLock::new(());

thread_local! {
    // captures this thread has open, which an exclusive scope on the same thread must not wait for
    static OPEN_CAPTURES: Cell<usize> = const { Cell::new(0) };
}

/// Held from before a stream's `begin_capture` until its capture has ended; captures on other streams run alongside.
pub struct CaptureScope {
    // none for a capture nested in one this thread already holds: a second shared lock could wait on a queued writer
    _shared: Option<RwLockReadGuard<'static, ()>>,
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        OPEN_CAPTURES.with(|open| open.set(open.get() - 1));
    }
}

pub fn capture_scope() -> CaptureScope {
    let shared = (OPEN_CAPTURES.with(Cell::get) == 0)
        .then(|| CONTEXT_WIDE.read().unwrap_or_else(PoisonError::into_inner));
    OPEN_CAPTURES.with(|open| open.set(open.get() + 1));
    CaptureScope { _shared: shared }
}

/// Held across a call CUDA rejects during any capture, so it waits for open captures to end. A thread inside its own
/// capture gets no lock: waiting would deadlock, and CUDA reports the call's failure as before.
pub fn context_wide_scope() -> Option<RwLockWriteGuard<'static, ()>> {
    if OPEN_CAPTURES.with(Cell::get) > 0 {
        return None;
    }
    Some(CONTEXT_WIDE.write().unwrap_or_else(PoisonError::into_inner))
}
