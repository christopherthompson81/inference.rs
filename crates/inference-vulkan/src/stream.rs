use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;

use crate::Result;

/// Dispatches one command buffer takes before it is submitted (without waiting) and a new one opens, so the GPU starts
/// on a long run of work while the CPU still records the rest.
pub(crate) const DISPATCHES_PER_SUBMIT: usize = 64;
/// Dispatches a batch submits after while the GPU has nothing else to run, so it starts soon after a synchronize.
pub(crate) const DISPATCHES_WHEN_IDLE: usize = 8;
/// The multiply-adds a batch submits after, well inside the driver's job timeout (~2 s on amdgpu) at the slowest
/// kernels' rate.
pub(crate) const WORK_PER_SUBMIT: u64 = 4_000_000_000;
/// Submitted batches a discrete GPU's stream lets run unfinished before a submit waits for the oldest: the CPU records
/// a whole forward ahead of the GPU otherwise, and the temporaries its batches hold (a long prompt's, many GB) would
/// fill device memory.
pub(crate) const DISCRETE_MAX_IN_FLIGHT: usize = 3;

/// A resource a recorded command uses, kept alive until the command finishes.
pub type Retained = Arc<dyn Any + Send + Sync>;

/// Command buffers in order: the one recording, those submitted and not known finished, and spare ones. A submitted
/// buffer keeps every resource its commands use alive until its fence signals.
pub(crate) struct Stream {
    pool: vk::CommandPool,
    pub(crate) profile: Option<Box<Profile>>,
    // the most submitted batches left unfinished, if bounded
    pub(crate) max_in_flight: Option<usize>,
    open: Option<Batch>,
    in_flight: VecDeque<Batch>,
    spare: Vec<(vk::CommandBuffer, vk::Fence)>,
}

pub(crate) struct Batch {
    pub(crate) buffer: vk::CommandBuffer,
    fence: vk::Fence,
    pub(crate) dispatches: usize,
    // the multiply-adds its heavy dispatches declared
    pub(crate) work: u64,
    pub(crate) retained: Vec<Retained>,
    // the kernels whose dispatches wrote timestamps 1.. of the profiler's pool, while profiling
    pub(crate) profiled: Vec<&'static str>,
    pub(crate) queries: Option<vk::QueryPool>,
}

/// Set to profile the GPU time of each kernel, printed every `PROFILE_PRINT_SECS`.
pub const PROFILE_ENV: &str = "INFERENCE_VULKAN_PROFILE";
// timestamps one batch's query pool holds: its opening one and a dispatch's each
pub(crate) const PROFILE_QUERIES: u32 = 8192;
const PROFILE_PRINT_SECS: u64 = 5;

/// Each kernel's dispatch count and GPU time since the last print.
#[derive(Default)]
pub(crate) struct Profile {
    kernels: std::collections::HashMap<&'static str, (u64, f64)>,
    since: Option<std::time::Instant>,
    pub(crate) ns_per_tick: f64,
}

impl Profile {
    pub(crate) fn new(ns_per_tick: f64) -> Self {
        Self {
            ns_per_tick,
            ..Default::default()
        }
    }

    fn add(&mut self, names: &[&'static str], ticks: &[u64]) {
        for (i, name) in names.iter().enumerate() {
            let entry = self.kernels.entry(name).or_default();
            entry.0 += 1;
            entry.1 += ticks[i + 1].saturating_sub(ticks[i]) as f64 * self.ns_per_tick;
        }
        let since = *self.since.get_or_insert_with(std::time::Instant::now);
        if since.elapsed().as_secs() >= PROFILE_PRINT_SECS {
            let mut rows: Vec<_> = self.kernels.drain().collect();
            rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
            let total: f64 = rows.iter().map(|r| r.1.1).sum();
            eprintln!(
                "vulkan profile over {:.1} s: {:.1} ms on the GPU",
                since.elapsed().as_secs_f64(),
                total / 1e6
            );
            for (name, (count, ns)) in rows {
                eprintln!(
                    "  {name:<20} {count:>8} dispatches {:>9.1} ms {:>7.1} us each",
                    ns / 1e6,
                    ns / 1e3 / count as f64
                );
            }
            self.since = Some(std::time::Instant::now());
        }
    }
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
            profile: None,
            max_in_flight: None,
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
            let queries = profile_pool(device, buffer)?;
            self.open = Some(Batch {
                buffer,
                fence,
                dispatches: 0,
                work: 0,
                retained: Vec::new(),
                profiled: Vec::new(),
                queries,
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
        if let Some(max) = self.max_in_flight {
            while self.in_flight.len() > max {
                self.wait_oldest(device)?;
            }
        }
        self.reap(device, false)
    }

    /// Whether every submitted batch has finished.
    pub(crate) fn idle(&self, device: &ash::Device) -> Result<bool> {
        let Some(last) = self.in_flight.back() else {
            return Ok(true);
        };
        // SAFETY: the fence belongs to this device and was submitted with its batch
        Ok(unsafe { device.get_fence_status(last.fence)? })
    }

    /// Waits for the oldest submitted batch, if any, and releases what finished; whether there was one.
    pub(crate) fn wait_oldest(&mut self, device: &ash::Device) -> Result<bool> {
        let Some(oldest) = self.in_flight.front() else {
            return Ok(false);
        };
        // SAFETY: the fence belongs to this device and was submitted with its batch
        unsafe { device.wait_for_fences(&[oldest.fence], true, u64::MAX)? };
        self.reap(device, false)?;
        Ok(true)
    }

    /// Submits the open batch and waits for everything submitted.
    pub(crate) fn synchronize(&mut self, device: &ash::Device, queue: vk::Queue) -> Result<()> {
        self.flush(device, queue)?;
        self.reap(device, true)
    }

    // releases the batches that finished (all of them, waiting, when `wait`), keeping their buffers as spares
    fn reap(&mut self, device: &ash::Device, wait: bool) -> Result<()> {
        let mut profile = self.profile.as_mut();
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
            if let (Some(pool), Some(profile)) = (batch.queries, profile.as_deref_mut()) {
                let mut ticks = vec![0u64; batch.profiled.len() + 1];
                // SAFETY: the batch finished, so every timestamp it wrote is available
                unsafe {
                    device.get_query_pool_results(
                        pool,
                        0,
                        &mut ticks,
                        vk::QueryResultFlags::TYPE_64,
                    )?;
                    device.destroy_query_pool(pool, None);
                }
                profile.add(&batch.profiled, &ticks);
            }
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

// a fresh query pool for a batch, its first timestamp written, while profiling
fn profile_pool(device: &ash::Device, buffer: vk::CommandBuffer) -> Result<Option<vk::QueryPool>> {
    if std::env::var_os(PROFILE_ENV).is_none() {
        return Ok(None);
    }
    let info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(PROFILE_QUERIES);
    // SAFETY: the buffer is recording, and the pool is reset before its first write
    unsafe {
        let pool = device.create_query_pool(&info, None)?;
        device.cmd_reset_query_pool(buffer, pool, 0, PROFILE_QUERIES);
        device.cmd_write_timestamp(buffer, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 0);
        Ok(Some(pool))
    }
}
