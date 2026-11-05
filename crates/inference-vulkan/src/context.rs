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
    pub(crate) kernels: Kernels,
    // the matrix-core kernels, on a device that has them
    pub(crate) coop: Option<crate::kernels::CoopKernels>,
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
        Self::with_matrix_cores(MatrixCores::Discrete)
    }

    /// [`Self::new`], with prompt matmuls on the device's matrix cores (VK_KHR_cooperative_matrix) where `mode` allows
    /// and the device has them.
    pub fn with_matrix_cores(mode: MatrixCores) -> Result<Arc<Self>> {
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
            MatrixCores::Discrete => !unified,
            MatrixCores::Any => true,
        } && pin.is_some()
            && coop_matrix_f16(&entry, &instance, physical);
        let pin_subgroup = has_coop && pin == Some(SubgroupPin::Required);
        let mut extensions = REQUIRED_EXTENSIONS.map(CStr::as_ptr).to_vec();
        if has_coop {
            extensions.push(ash::khr::cooperative_matrix::NAME.as_ptr());
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
            .vulkan_memory_model(has_coop);
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
        if !unified {
            stream.max_in_flight = Some(crate::stream::DISCRETE_MAX_IN_FLIGHT);
        }
        if std::env::var_os(crate::stream::PROFILE_ENV).is_some() {
            stream.profile = Some(Box::new(crate::stream::Profile::new(f64::from(
                properties.limits.timestamp_period,
            ))));
        }
        let kernels = Kernels::new(&device)?;
        let coop = if has_coop {
            Some(crate::kernels::CoopKernels::new(
                &device,
                pin_subgroup.then_some(COOP_SUBGROUP),
            )?)
        } else {
            None
        };
        Ok(Arc::new(Self {
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
            kernels,
            coop,
        }))
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

    /// [`Self::synchronize`] unless the stream is in use (a buffer allocated while recording), so it never waits on
    /// itself; whether it waited.
    pub(crate) fn try_synchronize(&self) -> Result<bool> {
        let Ok(mut stream) = self.stream.try_lock() else {
            return Ok(false);
        };
        stream.synchronize(&self.device, self.queue)?;
        Ok(true)
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
        self.fill_range(dst, 0, size, word);
    }

    /// [`Self::fill`] from byte `at` (a multiple of 4).
    pub(crate) fn fill_range(&self, dst: vk::Buffer, at: u64, size: u64, word: u32) {
        self.transfer(|device, buffer| {
            // SAFETY: as above
            unsafe { device.cmd_fill_buffer(buffer, dst, at, size, word) }
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
            if let Some(coop) = &self.coop {
                coop.destroy(&self.device);
            }
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

// the matrix-core kernels' tiling assumes subgroups this wide
const COOP_SUBGROUP: u32 = 32;
/// Where prompt matmuls and attention may run on matrix cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixCores {
    Off,
    /// A discrete GPU's only: an iGPU's lose to its dp4a kernels (a 780M's prompts run 4-8% slower on them).
    Discrete,
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
