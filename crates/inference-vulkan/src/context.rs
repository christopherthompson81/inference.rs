use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::kernels::Kernels;
use crate::{Error, Result};

const REQUIRED_EXTENSIONS: [&CStr; 2] = [
    ash::ext::external_memory_host::NAME,
    ash::khr::push_descriptor::NAME,
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
    name: String,
    queue: vk::Queue,
    submit: Mutex<Submit>,
    pub(crate) kernels: Kernels,
}

// one command buffer recorded and waited on at a time
struct Submit {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

/// Records one submission's commands; see [`Context::submit`].
pub struct Recorder<'a> {
    pub(crate) context: &'a Context,
    pub(crate) buffer: vk::CommandBuffer,
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

        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let extensions = REQUIRED_EXTENSIONS.map(CStr::as_ptr);
        let mut features11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut features11);
        // SAFETY: the create info and everything it points to outlive the call
        let device = unsafe { instance.create_device(physical, &create, None)? };
        // SAFETY: the device was created from this instance
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let host_memory = ash::ext::external_memory_host::Device::new(&instance, &device);
        let push_descriptor = ash::khr::push_descriptor::Device::new(&instance, &device);
        // SAFETY: the family was checked to exist and queue 0 was asked for
        let queue = unsafe { device.get_device_queue(family, 0) };
        let submit = Submit::new(&device, family)?;
        let kernels = Kernels::new(&device)?;
        Ok(Arc::new(Self {
            _entry: entry,
            instance,
            device,
            host_memory,
            push_descriptor,
            memory_properties,
            import_alignment,
            name,
            queue,
            submit: Mutex::new(submit),
            kernels,
        }))
    }

    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// Records `record`'s commands into one command buffer, submits it and waits for the GPU to finish it.
    pub fn submit<T>(&self, record: impl FnOnce(&Recorder<'_>) -> Result<T>) -> Result<T> {
        let submit = self.submit.lock().expect("vulkan submit poisoned");
        // SAFETY: the buffer is idle (the last submit was waited on) and owned by this lock's holder
        unsafe {
            self.device
                .reset_command_buffer(submit.buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                submit.buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let out = record(&Recorder {
            context: self,
            buffer: submit.buffer,
        });
        // SAFETY: as above; the buffer was begun above, and the fence is unsignaled between submits
        unsafe {
            self.device.end_command_buffer(submit.buffer)?;
            let out = out?;
            let buffers = [submit.buffer];
            let info = vk::SubmitInfo::default().command_buffers(&buffers);
            self.device
                .queue_submit(self.queue, &[info], submit.fence)?;
            self.device
                .wait_for_fences(&[submit.fence], true, u64::MAX)?;
            self.device.reset_fences(&[submit.fence])?;
            Ok(out)
        }
    }

    pub(crate) fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        let types = &self.memory_properties.memory_types
            [..self.memory_properties.memory_type_count as usize];
        (0..types.len() as u32)
            .find(|&i| bits & (1 << i) != 0 && types[i as usize].property_flags.contains(want))
    }
}

impl Recorder<'_> {
    /// Orders the commands recorded so far before the ones recorded after, for compute reading compute's writes.
    pub fn barrier(&self) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: the buffer is recording
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }
}

impl Submit {
    fn new(device: &ash::Device, family: u32) -> Result<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the create infos outlive their calls, and the pool outlives its buffer (both freed in Drop)
        unsafe {
            let pool = device.create_command_pool(&pool_info, None)?;
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let buffer = device.allocate_command_buffers(&alloc)?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            Ok(Self {
                pool,
                buffer,
                fence,
            })
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let submit = self.submit.get_mut().expect("vulkan submit poisoned");
        // SAFETY: no submit is in flight (each is waited on), and every object here was created from this device
        unsafe {
            let _ = self.device.device_wait_idle();
            self.kernels.destroy(&self.device);
            self.device.destroy_fence(submit.fence, None);
            self.device.destroy_command_pool(submit.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
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
