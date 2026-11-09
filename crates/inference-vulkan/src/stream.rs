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
/// The buffer bytes allocated since the last submit that submit the batch recording: a batch keeps its temporaries
/// alive until it finishes, so with [`DISCRETE_MAX_IN_FLIGHT`] in flight this bounds a forward's live temporaries
/// (on an iGPU, host RAM: Parakeet's subsampling over 10 minutes had held 4 GB, many small kernels a batch).
pub(crate) const BYTES_PER_SUBMIT: u64 = 256 << 20;
/// Submitted batches a stream lets run unfinished before a submit waits for the oldest: the CPU records a whole forward
/// ahead of the GPU otherwise, and the temporaries its batches hold (a long prompt's, many GB) would fill device memory,
/// or an iGPU's host RAM.
pub(crate) const DISCRETE_MAX_IN_FLIGHT: usize = 3;
/// The buffer bytes the batches in flight may have allocated before a submit waits for the oldest (one batch may
/// pass it): a long speech request's batches each allocate a whole BYTES_PER_SUBMIT, two in flight rather than three;
/// at half this, Qwen3.8-27B's batches waited too (prefill 342 tok/s, was ~500).
pub(crate) const MAX_IN_FLIGHT_BYTES: u64 = 512 << 20;

/// A barrier's scope on one side, its stages and accesses. Work owes the next command its barrier rather than ending on
/// one: NVIDIA's driver (615.71) stalls matrix-core kernels ~1 s some seconds into work whose submits end on barriers.
#[derive(Clone, Copy)]
pub(crate) struct Owed {
    pub(crate) stages: vk::PipelineStageFlags,
    pub(crate) access: vk::AccessFlags,
}

impl Owed {
    /// A dispatch's writes.
    pub(crate) const WRITES: Self = Self {
        stages: vk::PipelineStageFlags::COMPUTE_SHADER,
        access: vk::AccessFlags::SHADER_WRITE,
    };
    /// A dispatch's reads and writes.
    pub(crate) const COMPUTE: Self = Self {
        stages: vk::PipelineStageFlags::COMPUTE_SHADER,
        access: vk::AccessFlags::from_raw(
            vk::AccessFlags::SHADER_READ.as_raw() | vk::AccessFlags::SHADER_WRITE.as_raw(),
        ),
    };
    /// Compute and transfer work alike, every access.
    pub(crate) const ALL: Self = Self {
        stages: vk::PipelineStageFlags::from_raw(
            vk::PipelineStageFlags::COMPUTE_SHADER.as_raw()
                | vk::PipelineStageFlags::TRANSFER.as_raw(),
        ),
        access: vk::AccessFlags::from_raw(
            vk::AccessFlags::SHADER_READ.as_raw()
                | vk::AccessFlags::SHADER_WRITE.as_raw()
                | vk::AccessFlags::TRANSFER_READ.as_raw()
                | vk::AccessFlags::TRANSFER_WRITE.as_raw(),
        ),
    };
}

/// A resource a recorded command uses, kept alive until the command finishes.
pub type Retained = Arc<dyn Any + Send + Sync>;

/// Command buffers in order on one queue: the one recording, those submitted and not known finished, and spare ones. A
/// submitted buffer keeps every resource its commands use alive until its fence signals. Batches are numbered from 1;
/// the stream's timeline semaphore takes each one's number as it finishes, so another stream's batch can wait on it.
pub(crate) struct Stream {
    pool: vk::CommandPool,
    queue: vk::Queue,
    pub(crate) timeline: vk::Semaphore,
    // the number of the last batch submitted
    pub(crate) submitted: u64,
    pub(crate) profile: Option<Box<Profile>>,
    // the most submitted batches left unfinished, if bounded
    pub(crate) max_in_flight: Option<usize>,
    open: Option<Batch>,
    // the barrier the last submitted batch owed, recorded at the start of the next
    owed: Option<Owed>,
    in_flight: VecDeque<Batch>,
    spare: Vec<(vk::CommandBuffer, vk::Fence)>,
}

pub(crate) struct Batch {
    pub(crate) buffer: vk::CommandBuffer,
    fence: vk::Fence,
    pub(crate) serial: u64,
    // other streams' batches it waits for, by stream: their numbers, unfinished when its commands were recorded
    pub(crate) waits: Vec<(usize, u64)>,
    pub(crate) dispatches: usize,
    // the multiply-adds its heavy dispatches declared
    pub(crate) work: u64,
    pub(crate) retained: Vec<Retained>,
    // the dispatches that wrote timestamps 1.. of the profiler's pool, while profiling
    pub(crate) profiled: Vec<Profiled>,
    pub(crate) queries: Option<vk::QueryPool>,
    // the buffer bytes allocated while it recorded, which its retained temporaries hold until it finishes
    pub(crate) bytes: u64,
    pub(crate) owed: Option<Owed>,
}

/// Set to profile the GPU time of each kernel, printed every `PROFILE_PRINT_SECS`.
pub const PROFILE_ENV: &str = "INFERENCE_VULKAN_PROFILE";
/// Set to label every dispatch with its kernel's name (VK_EXT_debug_utils), for an external GPU profiler's ranges.
pub const LABELS_ENV: &str = "INFERENCE_VULKAN_LABELS";
// timestamps one batch's query pool holds: its opening one and a dispatch's each
pub(crate) const PROFILE_QUERIES: u32 = 8192;
const PROFILE_PRINT_SECS: u64 = 5;

/// A profiled dispatch: its kernel, and the shape and multiply-adds its op declared, if it did.
pub(crate) struct Profiled {
    pub(crate) kernel: &'static str,
    pub(crate) shape: Option<(String, u64)>,
}

/// Each kernel and shape's dispatch count, GPU time and multiply-adds since the last print.
#[derive(Default)]
pub(crate) struct Profile {
    kernels: std::collections::HashMap<(&'static str, String), (u64, f64, u64)>,
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

    fn add(&mut self, dispatches: &[Profiled], ticks: &[u64]) {
        for (i, d) in dispatches.iter().enumerate() {
            let (shape, macs) = d.shape.clone().unwrap_or_default();
            let entry = self.kernels.entry((d.kernel, shape)).or_default();
            entry.0 += 1;
            entry.1 += ticks[i + 1].saturating_sub(ticks[i]) as f64 * self.ns_per_tick;
            entry.2 += macs;
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
            for ((name, shape), (count, ns, macs)) in rows {
                // two operations a multiply-add
                let rate = if macs > 0 {
                    format!(" {:>6.1} TFLOPS", 2.0 * macs as f64 / ns / 1e3)
                } else {
                    String::new()
                };
                eprintln!(
                    "  {name:<22} {shape:<36} {count:>8} dispatches {:>9.1} ms {:>8.1} us each{rate}",
                    ns / 1e6,
                    ns / 1e3 / count as f64
                );
            }
            self.since = Some(std::time::Instant::now());
        }
    }
}

impl Stream {
    pub(crate) fn new(device: &ash::Device, family: u32, queue: vk::Queue) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        // SAFETY: the create infos outlive the calls
        let (pool, timeline) = unsafe {
            (
                device.create_command_pool(&info, None)?,
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut timeline),
                    None,
                )?,
            )
        };
        Ok(Self {
            pool,
            queue,
            timeline,
            submitted: 0,
            profile: None,
            max_in_flight: None,
            open: None,
            owed: None,
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
                serial: self.submitted + 1,
                waits: Vec::new(),
                dispatches: 0,
                work: 0,
                retained: Vec::new(),
                profiled: Vec::new(),
                queries,
                bytes: 0,
                owed: self.owed.take(),
            });
        }
        Ok(self.open.as_mut().expect("opened above"))
    }

    /// The other streams' batches the open batch waits for.
    pub(crate) fn open_waits(&self) -> &[(usize, u64)] {
        self.open.as_ref().map_or(&[], |b| &b.waits)
    }

    /// Submits the open batch, if any, without waiting for it, after `waits` (other streams' timelines reaching the
    /// values given); it allocated `bytes` while recording.
    pub(crate) fn flush(
        &mut self,
        device: &ash::Device,
        bytes: u64,
        waits: &[(vk::Semaphore, u64)],
    ) -> Result<()> {
        let Some(mut batch) = self.open.take() else {
            return Ok(());
        };
        batch.bytes = bytes;
        self.owed = batch.owed;
        let buffers = [batch.buffer];
        let (semaphores, values): (Vec<_>, Vec<_>) = waits.iter().copied().unzip();
        let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let signal = [self.timeline];
        let signal_values = [batch.serial];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&buffers)
            .wait_semaphores(&semaphores)
            .wait_dst_stage_mask(&stages)
            .signal_semaphores(&signal)
            .push_next(&mut timeline);
        // SAFETY: the buffer was begun in `open` and is ended once here; its fence is unsignaled, and the timeline's
        // value rises with each batch
        unsafe {
            device.end_command_buffer(batch.buffer)?;
            device.queue_submit(self.queue, &[submit], batch.fence)?;
        }
        self.submitted = batch.serial;
        self.in_flight.push_back(batch);
        if let Some(max) = self.max_in_flight {
            let held = |s: &Self| s.in_flight.iter().map(|b| b.bytes).sum::<u64>();
            while self.in_flight.len() > max
                || (self.in_flight.len() > 1 && held(self) > MAX_IN_FLIGHT_BYTES)
            {
                self.wait_oldest(device)?;
            }
        }
        self.reap(device, false)
    }

    /// Whether nothing is recorded and unsubmitted, and every submitted batch has finished.
    pub(crate) fn quiet(&self, device: &ash::Device) -> Result<bool> {
        Ok(self.open.is_none() && self.idle(device)?)
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

    // releases the batches that finished (all of them, waiting, when `wait`), keeping their buffers as spares
    pub(crate) fn reap(&mut self, device: &ash::Device, wait: bool) -> Result<()> {
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
            device.destroy_semaphore(self.timeline, None);
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
