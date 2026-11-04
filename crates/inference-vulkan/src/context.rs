use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::kernels::Kernels;
use crate::memory::BufferPool;
use crate::stream::{
    DISPATCHES_PER_SUBMIT, DISPATCHES_WHEN_IDLE, Retained, Stream, WORK_PER_SUBMIT,
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
    // device-local bytes free when the context was made, and those the long-lived (exact) buffers hold now
    pub(crate) device_budget: u64,
    pub(crate) exact_device_bytes: std::sync::atomic::AtomicU64,
    name: String,
    queue: vk::Queue,
    stream: Mutex<Stream>,
    pub(crate) pool: Mutex<BufferPool>,
    pub(crate) kernels: Kernels,
}

/// Appends commands to the stream's open command buffer; see [`Context::record`]. Each dispatch is ordered after the
/// ones before it.
pub struct Recorder<'a> {
    pub(crate) context: &'a Context,
    pub(crate) buffer: vk::CommandBuffer,
    retained: RefCell<Vec<Retained>>,
    pub(crate) dispatches: Cell<usize>,
    pub(crate) work: Cell<u64>,
    queries: Option<vk::QueryPool>,
    profiled: RefCell<Vec<&'static str>>,
}

impl Context {
    /// The first integrated GPU (else any GPU) that takes compute work, imports host memory and pushes descriptors.
    pub fn new() -> Result<Arc<Self>> {
        // SAFETY: the loader is opened once here and kept alive in `_entry` for the context's life
        let entry = unsafe { ash::Entry::load() }.map_err(|err| Error::Load(err.to_string()))?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        // SAFETY: the create info and the application info it points to outlive the call
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )?
        };
        match Self::with_instance(entry, instance.clone()) {
            Ok(context) => Ok(context),
            Err(err) => {
                // SAFETY: nothing was created from the instance that is still alive
                unsafe { instance.destroy_instance(None) };
                Err(err)
            }
        }
    }

    fn with_instance(entry: ash::Entry, instance: ash::Instance) -> Result<Arc<Self>> {
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
        let device_budget = device_budget(&instance, physical);

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let extensions = REQUIRED_EXTENSIONS.map(CStr::as_ptr);
        let mut features11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut dot_product = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(true);
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features11)
            .push_next(&mut dot_product);
        // SAFETY: the create info and everything it points to outlive the call
        let device = unsafe { instance.create_device(physical, &create, None)? };
        // SAFETY: the device was created from this instance
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let host_memory = ash::ext::external_memory_host::Device::new(&instance, &device);
        let push_descriptor = ash::khr::push_descriptor::Device::new(&instance, &device);
        // SAFETY: the family was checked to exist and queue 0 was asked for
        let queue = unsafe { device.get_device_queue(family, 0) };
        let mut stream = Stream::new(&device, family)?;
        if std::env::var_os(crate::stream::PROFILE_ENV).is_some() {
            stream.profile = Some(Box::new(crate::stream::Profile::new(f64::from(
                properties.limits.timestamp_period,
            ))));
        }
        let kernels = Kernels::new(&device)?;
        Ok(Arc::new(Self {
            _entry: entry,
            instance,
            device,
            host_memory,
            push_descriptor,
            memory_properties,
            import_alignment,
            unified,
            device_budget,
            exact_device_bytes: std::sync::atomic::AtomicU64::new(0),
            name,
            queue,
            stream: Mutex::new(stream),
            pool: Mutex::default(),
            kernels,
        }))
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Whether the device's memory is the host's (an iGPU, or a CPU device), so its buffers are mapped host RAM.
    pub fn is_unified(&self) -> bool {
        self.unified
    }

    /// Appends `record`'s commands to the stream without waiting for them; a batch that has grown long is submitted.
    /// What the commands read or write must be [`Recorder::retain`]ed, or outlive the next [`Self::synchronize`].
    pub fn record<T>(&self, record: impl FnOnce(&Recorder<'_>) -> Result<T>) -> Result<T> {
        let mut stream = self.stream.lock().expect("vulkan stream poisoned");
        let batch = stream.open(&self.device)?;
        let recorder = Recorder {
            context: self,
            buffer: batch.buffer,
            retained: RefCell::new(Vec::new()),
            dispatches: Cell::new(0),
            work: Cell::new(0),
            queries: batch.queries,
            profiled: RefCell::new(std::mem::take(&mut batch.profiled)),
        };
        let out = record(&recorder);
        batch.profiled = recorder.profiled.into_inner();
        batch.retained.append(&mut recorder.retained.into_inner());
        batch.dispatches += recorder.dispatches.get();
        batch.work += recorder.work.get();
        let dispatches = batch.dispatches;
        if dispatches >= DISPATCHES_PER_SUBMIT
            || batch.work >= WORK_PER_SUBMIT
            || (dispatches >= DISPATCHES_WHEN_IDLE && stream.idle(&self.device)?)
        {
            stream.flush(&self.device, self.queue)?;
        }
        out
    }

    /// Submits what is recorded without waiting for it.
    pub fn flush(&self) -> Result<()> {
        self.stream
            .lock()
            .expect("vulkan stream poisoned")
            .flush(&self.device, self.queue)
    }

    /// Submits what is recorded and waits for the GPU to finish everything submitted.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .lock()
            .expect("vulkan stream poisoned")
            .synchronize(&self.device, self.queue)
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
    pub(crate) fn profile(&self, kernel: &'static str) {
        let Some(pool) = self.queries else {
            return;
        };
        let mut profiled = self.profiled.borrow_mut();
        if profiled.len() + 1 >= crate::stream::PROFILE_QUERIES as usize {
            return;
        }
        profiled.push(kernel);
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
        self.transfer(|device, buffer| {
            // SAFETY: the command buffer is recording, and the caller retains both buffers
            unsafe { device.cmd_copy_buffer(buffer, src, dst, &[region]) }
        });
    }

    /// Fills a buffer's first `size` bytes (a multiple of 4) with `word`, ordered as [`Self::copy`].
    pub(crate) fn fill(&self, dst: vk::Buffer, size: u64, word: u32) {
        self.transfer(|device, buffer| {
            // SAFETY: as above
            unsafe { device.cmd_fill_buffer(buffer, dst, 0, size, word) }
        });
    }

    // a transfer command between barriers that order it with the compute work on either side
    fn transfer(&self, command: impl FnOnce(&ash::Device, vk::CommandBuffer)) {
        let device = &self.context.device;
        let all = vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::TRANSFER_READ
            | vk::AccessFlags::TRANSFER_WRITE;
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(all)
            .dst_access_mask(all);
        // SAFETY: the command buffer is recording
        unsafe {
            device.cmd_pipeline_barrier(
                self.buffer,
                stages,
                stages,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            command(device, self.buffer);
            device.cmd_pipeline_barrier(
                self.buffer,
                stages,
                stages,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
        self.dispatches.set(self.dispatches.get() + 1);
    }

    /// Keeps `resource` alive until the commands recorded here have finished.
    pub fn retain(&self, resource: Retained) {
        self.retained.borrow_mut().push(resource);
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let stream = self.stream.get_mut().expect("vulkan stream poisoned");
        let _ = stream.synchronize(&self.device, self.queue);
        // SAFETY: everything submitted finished above, and every object here was created from this device
        unsafe {
            let _ = self.device.device_wait_idle();
            self.kernels.destroy(&self.device);
            stream.destroy(&self.device);
            self.pool
                .get_mut()
                .expect("vulkan pool poisoned")
                .destroy(&self.device);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// the device-local heaps' free bytes: their budget less what is in use (VK_EXT_memory_budget), else their size
fn device_budget(instance: &ash::Instance, physical: vk::PhysicalDevice) -> u64 {
    // SAFETY: `physical` came from this instance
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
    let has_budget = extensions
        .iter()
        .any(|ext| ext.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::default();
    if has_budget {
        props = props.push_next(&mut budget);
    }
    // SAFETY: as above, and the chained struct outlives the call
    unsafe { instance.get_physical_device_memory_properties2(physical, &mut props) };
    let memory = props.memory_properties;
    (0..memory.memory_heap_count as usize)
        .filter(|&i| {
            memory.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .map(|i| {
            if has_budget {
                budget.heap_budget[i].saturating_sub(budget.heap_usage[i])
            } else {
                memory.memory_heaps[i].size
            }
        })
        .max()
        .unwrap_or(0)
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
        let Some(family) = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
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
