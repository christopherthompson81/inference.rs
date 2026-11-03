use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;

use crate::Result;

/// Dispatches one command buffer takes before it is submitted (without waiting) and a new one opens, so the GPU starts
/// on a long run of work while the CPU still records the rest.
pub(crate) const DISPATCHES_PER_SUBMIT: usize = 64;

/// A resource a recorded command uses, kept alive until the command finishes.
pub type Retained = Arc<dyn Any + Send + Sync>;

/// Command buffers in order: the one recording, those submitted and not known finished, and spare ones. A submitted
/// buffer keeps every resource its commands use alive until its fence signals.
pub(crate) struct Stream {
    pool: vk::CommandPool,
    open: Option<Batch>,
    in_flight: VecDeque<Batch>,
    spare: Vec<(vk::CommandBuffer, vk::Fence)>,
}

pub(crate) struct Batch {
    pub(crate) buffer: vk::CommandBuffer,
    fence: vk::Fence,
    pub(crate) dispatches: usize,
    pub(crate) retained: Vec<Retained>,
}

impl Stream {
    pub(crate) fn new(device: &ash::Device, family: u32) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the create info outlives the call
        let pool = unsafe { device.create_command_pool(&info, None)? };
        Ok(Self {
            pool,
            open: None,
            in_flight: VecDeque::new(),
            spare: Vec::new(),
        })
    }

    /// The batch recording now, opened if none is.
    pub(crate) fn open(&mut self, device: &ash::Device) -> Result<&mut Batch> {
        if self.open.is_none() {
            let (buffer, fence) = match self.spare.pop() {
                Some(spare) => spare,
                None => {
                    let alloc = vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1);
                    // SAFETY: the pool belongs to this device, and the infos outlive their calls
                    unsafe {
                        (
                            device.allocate_command_buffers(&alloc)?[0],
                            device.create_fence(&vk::FenceCreateInfo::default(), None)?,
                        )
                    }
                }
            };
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: a spare buffer finished (its fence was waited on and reset) and was reset below in `reap`
            unsafe { device.begin_command_buffer(buffer, &begin)? };
            self.open = Some(Batch {
                buffer,
                fence,
                dispatches: 0,
                retained: Vec::new(),
            });
        }
        Ok(self.open.as_mut().expect("opened above"))
    }

    /// Submits the open batch, if any, without waiting for it.
    pub(crate) fn flush(&mut self, device: &ash::Device, queue: vk::Queue) -> Result<()> {
        let Some(batch) = self.open.take() else {
            return Ok(());
        };
        let buffers = [batch.buffer];
        // SAFETY: the buffer was begun in `open` and is ended once here; its fence is unsignaled
        unsafe {
            device.end_command_buffer(batch.buffer)?;
            device.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(&buffers)],
                batch.fence,
            )?;
        }
        self.in_flight.push_back(batch);
        self.reap(device, false)
    }

    /// Submits the open batch and waits for everything submitted.
    pub(crate) fn synchronize(&mut self, device: &ash::Device, queue: vk::Queue) -> Result<()> {
        self.flush(device, queue)?;
        self.reap(device, true)
    }

    // releases the batches that finished (all of them, waiting, when `wait`), keeping their buffers as spares
    fn reap(&mut self, device: &ash::Device, wait: bool) -> Result<()> {
        while let Some(batch) = self.in_flight.front() {
            // SAFETY: the fence belongs to this device and was submitted with its batch
            let done = unsafe {
                if wait {
                    device.wait_for_fences(&[batch.fence], true, u64::MAX)?;
                    true
                } else {
                    device.get_fence_status(batch.fence)?
                }
            };
            if !done {
                break;
            }
            let batch = self.in_flight.pop_front().expect("checked above");
            // SAFETY: the batch finished, so its buffer and fence are idle
            unsafe {
                device.reset_fences(&[batch.fence])?;
                device.reset_command_buffer(batch.buffer, vk::CommandBufferResetFlags::empty())?;
            }
            self.spare.push((batch.buffer, batch.fence));
        }
        Ok(())
    }

    /// # Safety
    /// Every batch must have finished (call [`Self::synchronize`] first).
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: the caller's contract; the pool frees its buffers
        unsafe {
            for (_, fence) in self.spare.drain(..) {
                device.destroy_fence(fence, None);
            }
            device.destroy_command_pool(self.pool, None);
        }
    }
}
