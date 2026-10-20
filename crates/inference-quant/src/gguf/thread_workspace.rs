//! Device scratch for the GGUF matmul launchers, one buffer per thread, stream and size class. A buffer keeps its
//! address for its thread's life, which CUDA graph replay needs, and is freed when the thread ends: an engine's
//! threads end with its runtime, when its model is unloaded.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use inference_tensor::cuda::cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut, SyncOnDrop};
use inference_tensor::{CudaDevice, Result};

pub(crate) struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    // what the buffer holds, for a caller that can reuse it; any other use clears it
    holds: Option<WorkspaceContent>,
}

/// What a scratch buffer was filled from, within one shared-input scope: the input's device address, its shape and
/// dtype.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WorkspaceContent {
    pub(crate) scope: u64,
    pub(crate) input: u64,
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    pub(crate) dtype: inference_tensor::DType,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: inference_tensor::cuda::DeviceId,
    stream: usize,
    thread: ThreadId,
    capacity: usize,
}

/// One kind of scratch buffer, kept per thread.
pub(crate) struct ThreadWorkspaces {
    slots: OnceLock<Mutex<HashMap<WorkspaceKey, &'static Mutex<WorkspaceSlot>>>>,
}

pub(crate) struct WorkspaceGuard<'a> {
    slot: MutexGuard<'static, WorkspaceSlot>,
    stream: &'a CudaStream,
}

impl WorkspaceGuard<'_> {
    pub(crate) fn ptr_mut(&mut self) -> (u64, SyncOnDrop<'_>) {
        self.slot.slice.device_ptr_mut(self.stream)
    }

    /// Whether the buffer already holds `content`; it is recorded as holding it from here on, as the caller fills it
    /// when this returns false.
    pub(crate) fn claim(&mut self, content: WorkspaceContent) -> bool {
        let held = self.slot.holds == Some(content);
        self.slot.holds = Some(content);
        held
    }
}

// the workspaces this thread made, freed as it ends
struct ThreadExit {
    thread: ThreadId,
    kinds: RefCell<Vec<&'static ThreadWorkspaces>>,
}

impl Drop for ThreadExit {
    fn drop(&mut self) {
        for kind in self.kinds.borrow().iter() {
            kind.release(self.thread);
        }
    }
}

thread_local! {
    static THREAD_EXIT: ThreadExit = ThreadExit {
        thread: std::thread::current().id(),
        kinds: RefCell::new(Vec::new()),
    };
}

impl ThreadWorkspaces {
    pub(crate) const fn new() -> Self {
        Self {
            slots: OnceLock::new(),
        }
    }

    fn slots(&self) -> &Mutex<HashMap<WorkspaceKey, &'static Mutex<WorkspaceSlot>>> {
        self.slots.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// This thread's buffer of at least `bytes` on `stream`, rounded up to a power of two; its contents count as
    /// unknown from here on.
    pub(crate) fn ensure<'a>(
        &'static self,
        dev: &CudaDevice,
        bytes: usize,
        stream: &'a CudaStream,
    ) -> Result<WorkspaceGuard<'a>> {
        let mut guard = self.ensure_claimable(dev, bytes, stream)?;
        guard.slot.holds = None;
        Ok(guard)
    }

    /// `ensure` for a caller that says what it fills the buffer with (`WorkspaceGuard::claim`), so a later claim of
    /// the same contents finds them.
    pub(crate) fn ensure_claimable<'a>(
        &'static self,
        dev: &CudaDevice,
        bytes: usize,
        stream: &'a CudaStream,
    ) -> Result<WorkspaceGuard<'a>> {
        let capacity = bytes.max(1).next_power_of_two();
        let thread = std::thread::current().id();
        let key = WorkspaceKey {
            device: dev.id(),
            stream: stream.cu_stream() as usize,
            thread,
            capacity,
        };
        let slot: &'static Mutex<WorkspaceSlot> = {
            let mut slots = self.slots().lock().unwrap();
            match slots.get(&key).copied() {
                Some(slot) => slot,
                None => {
                    let slice = unsafe { dev.alloc::<u8>(capacity)? };
                    let slot =
                        Box::leak(Box::new(Mutex::new(WorkspaceSlot { slice, holds: None })));
                    slots.insert(key, slot);
                    THREAD_EXIT.with(|exit| {
                        let mut kinds = exit.kinds.borrow_mut();
                        if !kinds.iter().any(|kind| std::ptr::eq(*kind, self)) {
                            kinds.push(self);
                        }
                    });
                    slot
                }
            }
        };
        Ok(WorkspaceGuard {
            slot: slot.lock().unwrap(),
            stream,
        })
    }

    fn release(&self, thread: ThreadId) {
        let mut slots = self.slots().lock().unwrap();
        let ended: Vec<_> = slots
            .keys()
            .filter(|key| key.thread == thread)
            .copied()
            .collect();
        for key in ended {
            if let Some(slot) = slots.remove(&key) {
                // SAFETY: the slot was leaked from a Box in `ensure`; only its own thread, now ending, could lock it
                drop(unsafe { Box::from_raw(std::ptr::from_ref(slot).cast_mut()) });
            }
        }
    }

    #[cfg(test)]
    fn held_by(&self, thread: ThreadId) -> usize {
        self.slots()
            .lock()
            .unwrap()
            .keys()
            .filter(|key| key.thread == thread)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::ThreadWorkspaces;

    static TEST_WORKSPACES: ThreadWorkspaces = ThreadWorkspaces::new();

    #[test]
    fn a_threads_workspaces_are_freed_when_it_ends() -> inference_tensor::Result<()> {
        let Ok(device) = inference_tensor::Device::new_cuda(0) else {
            return Ok(());
        };
        let dev = device.as_cuda_device()?.clone();
        let worker = std::thread::spawn(move || -> inference_tensor::Result<_> {
            let stream = dev.cuda_stream();
            TEST_WORKSPACES.ensure(&dev, 1 << 20, &stream)?;
            TEST_WORKSPACES.ensure(&dev, 1 << 10, &stream)?;
            let thread = std::thread::current().id();
            Ok((thread, TEST_WORKSPACES.held_by(thread)))
        });
        let (thread, held) = worker.join().unwrap()?;
        assert_eq!(held, 2);
        assert_eq!(TEST_WORKSPACES.held_by(thread), 0);
        Ok(())
    }
}
