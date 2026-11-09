use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::kernels::Kernels;
use crate::memory::BufferPool;
use crate::stream::{
    BYTES_PER_SUBMIT, DISPATCHES_PER_SUBMIT, DISPATCHES_WHEN_IDLE, Retained, Stream,
    WORK_PER_SUBMIT,
};
use crate::{Error, Result};

const REQUIRED_EXTENSIONS: [&CStr; 3] = [
    ash::ext::external_memory_host::NAME,
    ash::khr::push_descriptor::NAME,
    ash::khr::shader_integer_dot_product::NAME,
];

/// One Vulkan device and its compute queue, with the kernels' pipelines built on it.
pub struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub(crate) device: ash::Device,
    pub(crate) host_memory: ash::ext::external_memory_host::Device,
    pub(crate) push_descriptor: ash::khr::push_descriptor::Device,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) import_alignment: u64,
    // the device's memory is the host's, so buffers live in mapped host RAM
    pub(crate) unified: bool,
    // where the device's free memory is asked for: its physical device, and whether it reports a live budget
    // (VK_EXT_memory_budget), else its device-local heap's size, less what the context holds
    physical: vk::PhysicalDevice,
    has_budget: bool,
    heap_size: u64,
    // device-local bytes every live and pooled buffer holds
    pub(crate) device_bytes: std::sync::atomic::AtomicU64,
    name: String,
    queue: vk::Queue,
    stream: Mutex<Stream>,
    pub(crate) pool: Mutex<BufferPool>,
    // buffer bytes handed out, and the count at the last submit
    pub(crate) allocated: AtomicU64,
    flushed_at: AtomicU64,
    pub(crate) kernels: Kernels,
    // the matrix-core kernels, on a device that has them
    pub(crate) coop: Option<crate::kernels::CoopKernels>,
    // the prompt matmuls take them too (prompt attention always does)
    coop_matmuls: bool,
    // labels each dispatch records, under LABELS_ENV
    pub(crate) labels: Option<ash::ext::debug_utils::Device>,
    // decode activations already quantized, until something writes their buffer or their blocks'
    quantized: Mutex<Vec<Quantized>>,
    // set by `shut_down`: the device is gone, so what drops after leaks and what records fails
    pub(crate) shut: std::sync::atomic::AtomicBool,
}

/// A decode activation's int8 blocks (quantize_q8_1's, at `s_off` in `blocks`) for `values` contiguous floats of `x`
/// from `x_off`: the same blocks whatever rows they make, rows being whole blocks; both buffers held, so the pool
/// cannot hand either out again while it is cached.
pub(crate) struct Quantized {
    x: Arc<crate::Buffer>,
    x_off: usize,
    values: usize,
    pub(crate) blocks: Arc<crate::Buffer>,
    pub(crate) s_off: usize,
}

// decode activations the cache keeps (a layer's input serves at most a few matvecs)
const QUANTIZED_KEPT: usize = 4;

/// Appends commands to the stream's open command buffer; see [`Context::record`]. Each dispatch is ordered after the
/// ones before it.
pub struct Recorder<'a> {
    pub(crate) context: &'a Context,
    pub(crate) buffer: vk::CommandBuffer,
    retained: RefCell<Vec<Retained>>,
    pub(crate) dispatches: Cell<usize>,
    pub(crate) work: Cell<u64>,
    queries: Option<vk::QueryPool>,
    profiled: RefCell<Vec<crate::stream::Profiled>>,
    // the shape and multiply-adds the next profiled dispatch reports, set by its op
    shape: RefCell<Option<(String, u64)>>,
    // the barrier the commands so far owe the next one
    owed: Cell<Option<crate::stream::Owed>>,
}

impl Context {
    /// The first integrated GPU (else any GPU) that takes compute work, imports host memory and pushes descriptors.
    pub fn new() -> Result<Arc<Self>> {
        Self::with_matrix_cores(MatrixCores::Default)
    }

    /// [`Self::new`], with prompt matmuls on the device's matrix cores (VK_KHR_cooperative_matrix) where `mode` allows
    /// and the device has them.
    pub fn with_matrix_cores(mode: MatrixCores) -> Result<Arc<Self>> {
        // SAFETY: the loader is opened once here and kept alive in `_entry` for the context's life
        let entry = unsafe { ash::Entry::load() }.map_err(|err| Error::Load(err.to_string()))?;
        // 1.3 where the device has it: NV_cooperative_matrix2's shaders are SPIR-V 1.6
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let labels = std::env::var_os(crate::stream::LABELS_ENV).is_some();
        let instance_extensions = [ash::ext::debug_utils::NAME.as_ptr()];
        let instance_extensions: &[_] = if labels { &instance_extensions } else { &[] };
        // SAFETY: the create info and what it points to outlive the call
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(instance_extensions),
                None,
            )?
        };
        match Self::with_instance(entry, instance.clone(), mode) {
            Ok(context) => Ok(context),
            Err(err) => {
                // SAFETY: nothing was created from the instance that is still alive
                unsafe { instance.destroy_instance(None) };
                Err(err)
            }
        }
    }

    fn with_instance(
        entry: ash::Entry,
        instance: ash::Instance,
        matrix_cores: MatrixCores,
    ) -> Result<Arc<Self>> {
        let (physical, family) = pick_device(&instance)?;
        // SAFETY: `physical` came from this instance
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = properties
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host_props);
        // SAFETY: as above, and the chained struct outlives the call
        unsafe { instance.get_physical_device_properties2(physical, &mut props2) };
        let import_alignment = host_props.min_imported_host_pointer_alignment;
        let unified = properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU;
        let (has_budget, heap_size) = memory_budget(&instance, physical);

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let pin = coop_subgroup(&instance, physical);
        let has_coop = match matrix_cores {
            MatrixCores::Off => false,
            MatrixCores::Default | MatrixCores::Any => true,
        } && pin.is_some()
            && coop_matrix_f16(&entry, &instance, physical);
        let pin_subgroup = has_coop && pin == Some(SubgroupPin::Required);
        let has_cm2 = has_coop
            && properties.api_version >= vk::API_VERSION_1_3
            && !crate::shaders::MOE_MM_CM2_Q4_0_N128.is_empty()
            && std::env::var_os(NO_CM2_ENV).is_none()
            && coop_matrix2(&instance, physical);
        let mut extensions = REQUIRED_EXTENSIONS.map(CStr::as_ptr).to_vec();
        if has_coop {
            extensions.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }
        if has_cm2 {
            extensions.push(COOP_MATRIX2_NAME.as_ptr());
            extensions.push(DECODE_VECTOR_NAME.as_ptr());
        }
        if pin_subgroup {
            extensions.push(ash::ext::subgroup_size_control::NAME.as_ptr());
        }
        let mut size_control =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default().subgroup_size_control(true);
        let mut features11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(has_coop)
            .vulkan_memory_model(has_coop)
            .buffer_device_address(has_cm2);
        let mut cm2_features = CoopMatrix2Features::all();
        let mut decode_vector = DecodeVectorFeatures::with(vk::TRUE);
        let mut dot_product = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(true);
        let mut coop_features =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features11)
            .push_next(&mut features12)
            .push_next(&mut dot_product);
        if has_coop {
            create = create.push_next(&mut coop_features);
        }
        if has_cm2 {
            create = create
                .push_next(&mut cm2_features)
                .push_next(&mut decode_vector);
        }
        if pin_subgroup {
            create = create.push_next(&mut size_control);
        }
        // SAFETY: the create info and everything it points to outlive the call
        let device = unsafe { instance.create_device(physical, &create, None)? };
        // SAFETY: the device was created from this instance
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let host_memory = ash::ext::external_memory_host::Device::new(&instance, &device);
        let push_descriptor = ash::khr::push_descriptor::Device::new(&instance, &device);
        // SAFETY: the family was checked to exist and queue 0 was asked for
        let queue = unsafe { device.get_device_queue(family, 0) };
        let mut stream = Stream::new(&device, family)?;
        stream.max_in_flight = Some(crate::stream::DISCRETE_MAX_IN_FLIGHT);
        if std::env::var_os(crate::stream::PROFILE_ENV).is_some() {
            stream.profile = Some(Box::new(crate::stream::Profile::new(f64::from(
                properties.limits.timestamp_period,
            ))));
        }
        let disk_cache = crate::pipeline::DiskCache::open(&device, &properties)?;
        let kernels = Kernels::new(&device, disk_cache.cache)?;
        let coop = if has_coop {
            Some(crate::kernels::CoopKernels::new(
                &device,
                disk_cache.cache,
                pin_subgroup.then_some(COOP_SUBGROUP),
                has_cm2,
            )?)
        } else {
            None
        };
        // SAFETY: every pipeline is built
        unsafe { disk_cache.finish(&device) };
        let labels = std::env::var_os(crate::stream::LABELS_ENV)
            .is_some()
            .then(|| ash::ext::debug_utils::Device::new(&instance, &device));
        let coop_matmuls = matrix_cores == MatrixCores::Any || !unified;
        Ok(Arc::new(Self {
            coop_matmuls,
            labels,
            _entry: entry,
            instance,
            device,
            host_memory,
            push_descriptor,
            memory_properties,
            import_alignment,
            unified,
            physical,
            has_budget,
            heap_size,
            device_bytes: std::sync::atomic::AtomicU64::new(0),
            name,
            queue,
            stream: Mutex::new(stream),
            pool: Mutex::default(),
            allocated: AtomicU64::new(0),
            flushed_at: AtomicU64::new(0),
            kernels,
            coop,
            quantized: Mutex::default(),
            shut: std::sync::atomic::AtomicBool::new(false),
        }))
    }

    /// The cached int8 blocks of `values` floats of `x` from `x_off`, if nothing has written either since.
    pub(crate) fn find_quantized(
        &self,
        x: &Arc<crate::Buffer>,
        x_off: usize,
        values: usize,
    ) -> Option<(Arc<crate::Buffer>, usize)> {
        let cache = self.quantized.lock().expect("quantized cache poisoned");
        cache
            .iter()
            .find(|q| Arc::ptr_eq(&q.x, x) && q.x_off == x_off && q.values == values)
            .map(|q| (q.blocks.clone(), q.s_off))
    }

    pub(crate) fn remember_quantized(
        &self,
        x: &Arc<crate::Buffer>,
        x_off: usize,
        values: usize,
        blocks: &Arc<crate::Buffer>,
        s_off: usize,
    ) {
        let mut cache = self.quantized.lock().expect("quantized cache poisoned");
        if cache.len() >= QUANTIZED_KEPT {
            cache.remove(0);
        }
        cache.push(Quantized {
            x: x.clone(),
            x_off,
            values,
            blocks: blocks.clone(),
            s_off,
        });
    }

    /// Drops cached quantizations that `buffer` holds the floats or the blocks of: something is about to write it.
    pub(crate) fn forget_quantized(&self, buffer: vk::Buffer) {
        let mut cache = self.quantized.lock().expect("quantized cache poisoned");
        if !cache.is_empty() {
            cache.retain(|q| q.x.buffer != buffer && q.blocks.buffer != buffer);
        }
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// The device-local memory free now: the heap's budget less its usage, which other processes' and this context's
    /// allocations both count in (VK_EXT_memory_budget), else the heap less what this context holds.
    pub(crate) fn device_free(&self) -> u64 {
        if !self.has_budget {
            return self
                .heap_size
                .saturating_sub(self.device_bytes.load(std::sync::atomic::Ordering::Relaxed));
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: the physical device came from this context's instance, and the chained struct outlives the call
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical, &mut props)
        };
        let memory = props.memory_properties;
        (0..memory.memory_heap_count as usize)
            .filter(|&i| {
                memory.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|i| budget.heap_budget[i].saturating_sub(budget.heap_usage[i]))
            .max()
            .unwrap_or(0)
    }

    /// A discrete device's memory as `(total, free)`: its device-local heap, and what is free in it counting the
    /// buffer pool's idle device memory, which an allocation frees first; `None` on a unified device.
    pub fn device_memory(&self) -> Option<(u64, u64)> {
        if self.unified {
            return None;
        }
        let idle = self
            .pool
            .lock()
            .expect("vulkan pool poisoned")
            .device_bytes();
        Some((self.heap_size, self.device_free() + idle))
    }

    /// Whether the device's memory is the host's (an iGPU, or a CPU device), so its buffers are mapped host RAM.
    pub fn is_unified(&self) -> bool {
        self.unified
    }

    /// Whether f32 matmuls run on the device's matrix cores, multiplying their inputs rounded to f16.
    pub fn has_matrix_cores(&self) -> bool {
        self.coop.is_some()
    }

    /// Appends `record`'s commands to the stream without waiting for them; a batch that has grown long is submitted.
    /// What the commands read or write must be [`Recorder::retain`]ed, or outlive the next [`Self::synchronize`].
    pub fn record<T>(&self, record: impl FnOnce(&Recorder<'_>) -> Result<T>) -> Result<T> {
        let mut stream = self.stream.lock().expect("vulkan stream poisoned");
        self.live()?;
        let batch = stream.open(&self.device)?;
        let recorder = Recorder {
            context: self,
            buffer: batch.buffer,
            retained: RefCell::new(Vec::new()),
            dispatches: Cell::new(0),
            work: Cell::new(0),
            queries: batch.queries,
            profiled: RefCell::new(std::mem::take(&mut batch.profiled)),
            shape: RefCell::new(None),
            owed: Cell::new(batch.owed),
        };
        let out = record(&recorder);
        batch.owed = recorder.owed.get();
        batch.profiled = recorder.profiled.into_inner();
        batch.retained.append(&mut recorder.retained.into_inner());
        batch.dispatches += recorder.dispatches.get();
        batch.work += recorder.work.get();
        let dispatches = batch.dispatches;
        let allocated = self.allocated.load(Ordering::Relaxed);
        let since = allocated - self.flushed_at.load(Ordering::Relaxed);
        if dispatches >= DISPATCHES_PER_SUBMIT
            || batch.work >= WORK_PER_SUBMIT
            || since >= BYTES_PER_SUBMIT
            || (dispatches >= DISPATCHES_WHEN_IDLE && stream.idle(&self.device)?)
        {
            stream.flush(&self.device, self.queue, since)?;
            self.flushed_at.store(allocated, Ordering::Relaxed);
        }
        out
    }

    /// Submits what is recorded without waiting for it.
    // the matrix-core kernels where the prompt matmuls take them
    pub(crate) fn matmul_coop(&self) -> Option<&crate::kernels::CoopKernels> {
        self.coop.as_ref().filter(|_| self.coop_matmuls)
    }

    pub fn flush(&self) -> Result<()> {
        let mut stream = self.stream.lock().expect("vulkan stream poisoned");
        self.live()?;
        let allocated = self.allocated.load(Ordering::Relaxed);
        let since = allocated - self.flushed_at.swap(allocated, Ordering::Relaxed);
        stream.flush(&self.device, self.queue, since)
    }

    /// Submits what is recorded and waits for the GPU to finish everything submitted.
    pub fn synchronize(&self) -> Result<()> {
        let mut stream = self.stream.lock().expect("vulkan stream poisoned");
        self.live()?;
        stream.synchronize(&self.device, self.queue)
    }

    /// [`Self::synchronize`] unless the stream is in use (a buffer allocated while recording), so it never waits on
    /// itself; whether it waited.
    pub(crate) fn try_synchronize(&self) -> Result<bool> {
        let Ok(mut stream) = self.stream.try_lock() else {
            return Ok(false);
        };
        self.live()?;
        stream.synchronize(&self.device, self.queue)?;
        Ok(true)
    }

    /// Waits for the oldest submitted batch, returning its buffers to the pool, unless the stream is in use or nothing is
    /// in flight; whether it waited.
    pub(crate) fn try_wait_oldest(&self) -> Result<bool> {
        let Ok(mut stream) = self.stream.try_lock() else {
            return Ok(false);
        };
        self.live()?;
        stream.wait_oldest(&self.device)
    }

    /// [`Self::record`], then [`Self::synchronize`].
    pub fn submit<T>(&self, record: impl FnOnce(&Recorder<'_>) -> Result<T>) -> Result<T> {
        let out = self.record(record)?;
        self.synchronize()?;
        Ok(out)
    }

    pub(crate) fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        let types = &self.memory_properties.memory_types
            [..self.memory_properties.memory_type_count as usize];
        (0..types.len() as u32)
            .find(|&i| bits & (1 << i) != 0 && types[i as usize].property_flags.contains(want))
    }
}

impl Recorder<'_> {
    // a timestamp after `kernel`'s dispatch, while profiling and the batch's pool has room
    /// While profiling, the shape and multiply-adds the next dispatch reports (its op's main one).
    pub(crate) fn profile_shape(&self, shape: impl FnOnce() -> String, macs: usize) {
        if self.queries.is_some() {
            *self.shape.borrow_mut() = Some((shape(), macs as u64));
        }
    }

    pub(crate) fn profile(&self, kernel: &'static str) {
        let Some(pool) = self.queries else {
            return;
        };
        let shape = self.shape.borrow_mut().take();
        let mut profiled = self.profiled.borrow_mut();
        if profiled.len() + 1 >= crate::stream::PROFILE_QUERIES as usize {
            return;
        }
        profiled.push(crate::stream::Profiled { kernel, shape });
        // SAFETY: the buffer is recording and the query index is in the pool
        unsafe {
            self.context.device.cmd_write_timestamp(
                self.buffer,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                pool,
                profiled.len() as u32,
            );
        }
    }

    /// Copies `size` bytes between buffers, ordered after and before the commands recorded around it.
    pub(crate) fn copy(
        &self,
        src: vk::Buffer,
        src_at: u64,
        dst: vk::Buffer,
        dst_at: u64,
        size: u64,
    ) {
        let region = vk::BufferCopy::default()
            .src_offset(src_at)
            .dst_offset(dst_at)
            .size(size);
        self.context.forget_quantized(dst);
        self.transfer(|device, buffer| {
            // SAFETY: the command buffer is recording, and the caller retains both buffers
            unsafe { device.cmd_copy_buffer(buffer, src, dst, &[region]) }
        });
    }

    /// Fills a buffer's first `size` bytes (a multiple of 4) with `word`, ordered as [`Self::copy`].
    pub(crate) fn fill(&self, dst: vk::Buffer, size: u64, word: u32) {
        self.fill_range(dst, 0, size, word);
    }

    /// [`Self::fill`] from byte `at` (a multiple of 4).
    pub(crate) fn fill_range(&self, dst: vk::Buffer, at: u64, size: u64, word: u32) {
        self.context.forget_quantized(dst);
        self.transfer(|device, buffer| {
            // SAFETY: as above
            unsafe { device.cmd_fill_buffer(buffer, dst, at, size, word) }
        });
    }

    // a transfer command ordered with the compute work on either side
    fn transfer(&self, command: impl FnOnce(&ash::Device, vk::CommandBuffer)) {
        // after everything before it, whatever the commands before owe
        self.owe(crate::stream::Owed::ALL);
        self.settle(crate::stream::Owed::ALL);
        command(&self.context.device, self.buffer);
        self.owe(crate::stream::Owed::ALL);
        self.dispatches.set(self.dispatches.get() + 1);
    }

    /// Records the barrier the commands before owe, if any, ahead of a command in `next`'s stages and accesses.
    pub(crate) fn settle(&self, next: crate::stream::Owed) {
        let Some(owed) = self.owed.take() else {
            return;
        };
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(owed.access)
            .dst_access_mask(next.access);
        // SAFETY: the command buffer is recording
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.buffer,
                owed.stages,
                next.stages,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Orders the next command after what was just recorded (the barrier goes in ahead of it, see [`Self::settle`]).
    pub(crate) fn owe(&self, owed: crate::stream::Owed) {
        self.owed.set(Some(owed));
    }

    /// Keeps `resource` alive until the commands recorded here have finished: a pooled buffer's allocation rather than
    /// the buffer, which goes back to the pool for later-recorded work when its last user drops it.
    pub fn retain(&self, resource: Retained) {
        let resource = match resource.downcast::<crate::Buffer>() {
            Ok(buffer) => buffer.retained(),
            Err(other) => other,
        };
        self.retained.borrow_mut().push(resource);
    }
}

impl Context {
    pub(crate) fn is_shut(&self) -> bool {
        self.shut.load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn live(&self) -> Result<()> {
        if self.is_shut() {
            return Err(Error::Invalid("the Vulkan device was shut down".into()));
        }
        Ok(())
    }

    /// Waits for the GPU, then destroys the device and instance while buffers and clones of the context may remain:
    /// those leak when they drop and recording fails. For process exit, which otherwise tears the driver down under
    /// its own still-running threads (NVIDIA's crash there).
    pub fn shut_down(&self) {
        let mut stream = self
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.shut.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: the stream's lock keeps every recording out, the GPU is idle after the wait, and nothing uses the
        // device after `shut` (buffers leak, recording fails)
        unsafe { self.destroy(&mut stream, &mut pool) };
    }

    // SAFETY: no submit may still run, nor anything use the device after
    unsafe fn destroy(&self, stream: &mut Stream, pool: &mut BufferPool) {
        let _ = stream.synchronize(&self.device, self.queue);
        // SAFETY: the caller's contract; every object here was created from this device
        unsafe {
            let _ = self.device.device_wait_idle();
            self.kernels.destroy(&self.device);
            if let Some(coop) = &self.coop {
                coop.destroy(&self.device);
            }
            stream.destroy(&self.device);
            pool.destroy(&self.device);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.is_shut() {
            return;
        }
        let mut stream = self
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: the last reference is dropping, so nothing else records or uses the device
        unsafe { self.destroy(&mut stream, &mut pool) };
    }
}

// the matrix-core kernels' tiling assumes subgroups this wide
const COOP_SUBGROUP: u32 = 32;
/// Where prompt matmuls and attention may run on matrix cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixCores {
    Off,
    /// Everything on a discrete GPU, an iGPU's attention only: its matmuls lose to dp4a (a 780M's 2048-token prompt
    /// ran 10% slower on them) where its attention over a long cache gains 3.7x.
    Default,
    Any,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SubgroupPin {
    // compute subgroups are already COOP_SUBGROUP wide
    Default,
    // they can be made so per pipeline (VK_EXT_subgroup_size_control)
    Required,
}

// how the device's compute subgroups come to be COOP_SUBGROUP wide, if they can
fn coop_subgroup(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Option<SubgroupPin> {
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut props = vk::PhysicalDeviceProperties2::default()
        .push_next(&mut subgroup)
        .push_next(&mut control);
    // SAFETY: `physical` came from this instance, and the chained structs outlive the call
    unsafe { instance.get_physical_device_properties2(physical, &mut props) };
    if subgroup.subgroup_size == COOP_SUBGROUP {
        return Some(SubgroupPin::Default);
    }
    // SAFETY: as above
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let has_ext = extensions
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(ash::ext::subgroup_size_control::NAME));
    let mut features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features);
    // SAFETY: as above
    unsafe { instance.get_physical_device_features2(physical, &mut features2) };
    (has_ext
        && features.subgroup_size_control == vk::TRUE
        && (control.min_subgroup_size..=control.max_subgroup_size).contains(&COOP_SUBGROUP)
        && control
            .required_subgroup_size_stages
            .contains(vk::ShaderStageFlags::COMPUTE))
    .then_some(SubgroupPin::Required)
}

// whether the device multiplies 16 x 16 x 16 f16 matrices into f32 on its matrix cores (VK_KHR_cooperative_matrix),
// with the 16-bit arithmetic and memory model the kernels use
fn coop_matrix_f16(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> bool {
    // SAFETY: `physical` came from this instance
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    if !extensions
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(ash::khr::cooperative_matrix::NAME))
    {
        return false;
    }
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features12);
    // SAFETY: as above, and the chained struct outlives the call
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    if features12.shader_float16 == vk::FALSE || features12.vulkan_memory_model == vk::FALSE {
        return false;
    }
    let coop = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    // SAFETY: as above
    let shapes = unsafe { coop.get_physical_device_cooperative_matrix_properties(physical) }
        .unwrap_or_default();
    shapes.iter().any(|p| {
        (p.m_size, p.n_size, p.k_size) == (16, 16, 16)
            && p.a_type == vk::ComponentTypeKHR::FLOAT16
            && p.b_type == vk::ComponentTypeKHR::FLOAT16
            && p.c_type == vk::ComponentTypeKHR::FLOAT32
            && p.result_type == vk::ComponentTypeKHR::FLOAT32
            && p.scope == vk::ScopeKHR::SUBGROUP
    })
}

const COOP_MATRIX2_NAME: &CStr = c"VK_NV_cooperative_matrix2";
// VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV, past ash's headers
const COOP_MATRIX2_FEATURES: vk::StructureType = vk::StructureType::from_raw(1_000_593_000);
/// Set to keep NVIDIA's cooperative matrix 2 kernels off (for A/B runs against the coopmat1 ones).
pub const NO_CM2_ENV: &str = "INFERENCE_VULKAN_NO_CM2";

// VkPhysicalDeviceCooperativeMatrix2FeaturesNV
#[repr(C)]
struct CoopMatrix2Features {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    workgroup_scope: vk::Bool32,
    flexible_dimensions: vk::Bool32,
    reductions: vk::Bool32,
    conversions: vk::Bool32,
    per_element_operations: vk::Bool32,
    tensor_addressing: vk::Bool32,
    block_loads: vk::Bool32,
}

impl CoopMatrix2Features {
    fn with(value: vk::Bool32) -> Self {
        Self {
            s_type: COOP_MATRIX2_FEATURES,
            p_next: std::ptr::null_mut(),
            workgroup_scope: value,
            flexible_dimensions: value,
            reductions: value,
            conversions: value,
            per_element_operations: value,
            tensor_addressing: value,
            block_loads: value,
        }
    }

    fn all() -> Self {
        Self::with(vk::TRUE)
    }

    fn has_all(&self) -> bool {
        [
            self.workgroup_scope,
            self.flexible_dimensions,
            self.reductions,
            self.conversions,
            self.per_element_operations,
            self.tensor_addressing,
            self.block_loads,
        ]
        .iter()
        .all(|&f| f == vk::TRUE)
    }
}

const DECODE_VECTOR_NAME: &CStr = c"VK_NV_cooperative_matrix_decode_vector";
const DECODE_VECTOR_FEATURES: vk::StructureType = vk::StructureType::from_raw(1_000_689_000);

// VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV: the tensor loads' four-value decode functions
#[repr(C)]
struct DecodeVectorFeatures {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    decode_vector: vk::Bool32,
}

impl DecodeVectorFeatures {
    fn with(value: vk::Bool32) -> Self {
        Self {
            s_type: DECODE_VECTOR_FEATURES,
            p_next: std::ptr::null_mut(),
            decode_vector: value,
        }
    }
}

// SAFETY: a repr(C) Vulkan structure led by its sType and pNext
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for DecodeVectorFeatures {}
// SAFETY: as above
unsafe impl vk::ExtendsDeviceCreateInfo for DecodeVectorFeatures {}

// SAFETY: a repr(C) Vulkan structure led by its sType and pNext
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for CoopMatrix2Features {}
// SAFETY: as above
unsafe impl vk::ExtendsDeviceCreateInfo for CoopMatrix2Features {}

// whether the device has NVIDIA's cooperative matrix 2 with every feature its kernels use (the decode vector too),
// and device addresses
fn coop_matrix2(instance: &ash::Instance, physical: vk::PhysicalDevice) -> bool {
    // SAFETY: `physical` came from this instance
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let has = |name| {
        extensions
            .iter()
            .any(|ext| ext.extension_name_as_c_str() == Ok(name))
    };
    if !has(COOP_MATRIX2_NAME) || !has(DECODE_VECTOR_NAME) {
        return false;
    }
    let mut cm2 = CoopMatrix2Features::with(vk::FALSE);
    let mut decode = DecodeVectorFeatures::with(vk::FALSE);
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut features12)
        .push_next(&mut cm2)
        .push_next(&mut decode);
    // SAFETY: as above, and the chained structs outlive the call
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    cm2.has_all()
        && decode.decode_vector == vk::TRUE
        && features12.buffer_device_address == vk::TRUE
}

// whether the device reports a live memory budget (VK_EXT_memory_budget), and its largest device-local heap's size
fn memory_budget(instance: &ash::Instance, physical: vk::PhysicalDevice) -> (bool, u64) {
    // SAFETY: `physical` came from this instance
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let has_budget = extensions
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
    // SAFETY: as above
    let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
    let heap = memory.memory_heaps[..memory.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0);
    (has_budget, heap)
}

// an integrated GPU first: its memory is the host's, which the crate's zero-copy imports rely on for speed
fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
    // SAFETY: the instance is alive
    let devices = unsafe { instance.enumerate_physical_devices()? };
    let mut candidates = Vec::new();
    for physical in devices {
        // SAFETY: `physical` came from this instance
        let (properties, families, extensions) = unsafe {
            (
                instance.get_physical_device_properties(physical),
                instance.get_physical_device_queue_family_properties(physical),
                instance.enumerate_device_extension_properties(physical)?,
            )
        };
        let has = |name: &CStr| {
            extensions
                .iter()
                .any(|ext| ext.extension_name_as_c_str() == Ok(name))
        };
        if !REQUIRED_EXTENSIONS.iter().all(|name| has(name)) {
            continue;
        }
        // a compute-only family where there is one: on a graphics queue NVIDIA's driver stalls matrix-core kernels ~1 s
        let compute =
            |f: &&vk::QueueFamilyProperties| f.queue_flags.contains(vk::QueueFlags::COMPUTE);
        let Some(family) = families
            .iter()
            .position(|f| compute(&f) && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .or_else(|| families.iter().position(|f| compute(&f)))
        else {
            continue;
        };
        let rank = match properties.device_type {
            vk::PhysicalDeviceType::INTEGRATED_GPU => 0,
            vk::PhysicalDeviceType::DISCRETE_GPU => 1,
            _ => 2,
        };
        candidates.push((rank, physical, family as u32));
    }
    candidates.sort_by_key(|&(rank, ..)| rank);
    candidates
        .first()
        .map(|&(_, physical, family)| (physical, family))
        .ok_or(Error::NoDevice)
}
