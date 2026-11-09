use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ash::vk;

use crate::{Error, Recorder, Result};

/// A compute pipeline over `bindings` storage buffers (bindings 0..n of set 0, pushed per dispatch) and a block of
/// push constants.
pub(crate) struct Pipeline {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bindings: usize,
    push_bytes: u32,
    name: &'static str,
    // bit b: binding b's buffer may be written (its block not decorated NonWritable)
    pub(crate) writes: u64,
}

// where compiled pipelines persist, under the user's cache directory, a file per device and driver build
const DISK_CACHE_SUBDIR: &str = "inference-rs/vulkan";
// VkPipelineCacheHeaderVersionOne: header length, version, vendor, device (u32 each), then the cache UUID
const CACHE_HEADER_BYTES: usize = 16 + vk::UUID_SIZE;

// SPIR-V opcodes and decorations the binding scan reads
const OP_TYPE_POINTER: u32 = 32;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_BINDING: u32 = 33;

// which bindings a module may write: a binding's variables, each writable unless it or every member of its block is
// NonWritable
fn writable_bindings(code: &[u32]) -> u64 {
    use std::collections::{HashMap, HashSet};
    let mut binding = HashMap::new();
    let mut readonly_vars = HashSet::new();
    // block -> (members, NonWritable members)
    let mut members: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut pointee = HashMap::new();
    let mut var_type = HashMap::new();
    let mut at = 5;
    while at < code.len() {
        let count = (code[at] >> 16) as usize;
        let op = code[at] & 0xffff;
        let w = &code[at..(at + count.max(1)).min(code.len())];
        match op {
            OP_DECORATE if w.len() >= 3 => match w[2] {
                DECORATION_BINDING if w.len() >= 4 => {
                    binding.insert(w[1], w[3]);
                }
                DECORATION_NON_WRITABLE => {
                    readonly_vars.insert(w[1]);
                }
                _ => {}
            },
            OP_MEMBER_DECORATE if w.len() >= 4 => {
                let entry = members.entry(w[1]).or_default();
                entry.0 = entry.0.max(w[2] + 1);
                if w[3] == DECORATION_NON_WRITABLE {
                    entry.1 += 1;
                }
            }
            OP_TYPE_POINTER if w.len() >= 4 => {
                pointee.insert(w[1], w[3]);
            }
            OP_VARIABLE if w.len() >= 3 => {
                var_type.insert(w[2], w[1]);
            }
            _ => {}
        }
        at += count.max(1);
    }
    let mut writes = 0u64;
    for (var, &b) in &binding {
        let block = var_type.get(var).and_then(|t| pointee.get(t));
        // Offset decorates every member of a block, so its count is the block's members
        let readonly = readonly_vars.contains(var)
            || block
                .and_then(|s| members.get(s))
                .is_some_and(|&(n, ro)| n > 0 && ro >= n);
        if !readonly && b < 64 {
            writes |= 1 << b;
        }
    }
    writes
}

/// A pipeline cache read from and written back to the user's cache directory, so a context whose shaders were
/// compiled before skips the driver's compile.
pub(crate) struct DiskCache {
    pub(crate) cache: vk::PipelineCache,
    path: Option<PathBuf>,
    loaded: usize,
}

fn user_cache() -> Option<PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
}

// whether `data` is a cache this device and driver wrote: drivers need not check a foreign one
fn header_matches(data: &[u8], properties: &vk::PhysicalDeviceProperties) -> bool {
    let word = |at: usize| u32::from_le_bytes(data[at..at + 4].try_into().expect("4 bytes"));
    data.len() >= CACHE_HEADER_BYTES
        && word(0) as usize >= CACHE_HEADER_BYTES
        && word(4) == vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32
        && word(8) == properties.vendor_id
        && word(12) == properties.device_id
        && data[16..CACHE_HEADER_BYTES] == properties.pipeline_cache_uuid
}

impl DiskCache {
    pub(crate) fn open(
        device: &ash::Device,
        properties: &vk::PhysicalDeviceProperties,
    ) -> Result<Self> {
        let uuid: String = properties
            .pipeline_cache_uuid
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let path = user_cache().map(|dir| {
            dir.join(DISK_CACHE_SUBDIR).join(format!(
                "{:04x}-{:04x}-{uuid}.bin",
                properties.vendor_id, properties.device_id
            ))
        });
        let data = path
            .as_ref()
            .and_then(|path| std::fs::read(path).ok())
            .filter(|data| header_matches(data, properties))
            .unwrap_or_default();
        // SAFETY: the create info and its data outlive the call
        let cache = unsafe {
            device.create_pipeline_cache(
                &vk::PipelineCacheCreateInfo::default().initial_data(&data),
                None,
            )?
        };
        Ok(Self {
            cache,
            path,
            loaded: data.len(),
        })
    }

    /// Writes the cache back where it grew (best effort: a read-only cache only costs the next run its compile),
    /// then destroys it.
    ///
    /// # Safety
    /// No pipeline creation may still be using the cache.
    pub(crate) unsafe fn finish(self, device: &ash::Device) {
        // SAFETY: the caller's contract
        let data = unsafe { device.get_pipeline_cache_data(self.cache) }.unwrap_or_default();
        if let Some(path) = &self.path
            && data.len() != self.loaded
            && let Some(dir) = path.parent()
            && std::fs::create_dir_all(dir).is_ok()
        {
            // a temporary beside it, renamed into place, so another process never reads half a file
            let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
            if std::fs::write(&temporary, &data).is_ok()
                && std::fs::rename(&temporary, path).is_err()
            {
                let _ = std::fs::remove_file(&temporary);
            }
        }
        // SAFETY: the caller's contract
        unsafe { device.destroy_pipeline_cache(self.cache, None) };
    }
}

/// What [`Pipeline::build_all`] builds one pipeline from.
#[derive(Clone, Copy)]
pub(crate) struct PipelineSpec {
    pub(crate) spirv: &'static [u8],
    pub(crate) bindings: usize,
    pub(crate) push_bytes: u32,
    pub(crate) name: &'static str,
    pub(crate) subgroup: Option<u32>,
    // the shader's specialization constant 0 (a 16-bit float kernel's `HALF_F16`), where it is given
    pub(crate) constant: Option<u32>,
}

impl PipelineSpec {
    pub(crate) const fn new(
        spirv: &'static [u8],
        bindings: usize,
        push_bytes: u32,
        name: &'static str,
    ) -> Self {
        Self {
            spirv,
            bindings,
            push_bytes,
            name,
            subgroup: None,
            constant: None,
        }
    }

    pub(crate) const fn constant(self, value: u32) -> Self {
        Self {
            constant: Some(value),
            ..self
        }
    }

    pub(crate) const fn subgroup(self, subgroup: Option<u32>) -> Self {
        Self { subgroup, ..self }
    }
}

impl Pipeline {
    /// Every spec's pipeline, in order, built across the machine's cores through `cache`: the driver compiles each
    /// one on the calling thread.
    pub(crate) fn build_all(
        device: &ash::Device,
        cache: vk::PipelineCache,
        specs: &[PipelineSpec],
    ) -> Result<Vec<Self>> {
        let next = AtomicUsize::new(0);
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(specs.len());
        let mut built: Vec<(usize, Result<Self>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(spec) = specs.get(index) else {
                                break out;
                            };
                            out.push((index, Self::build(device, cache, spec)));
                        }
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect()
        });
        built.sort_by_key(|(index, _)| *index);
        built.into_iter().map(|(_, pipeline)| pipeline).collect()
    }

    // its subgroups `subgroup` invocations wide when given (VK_EXT_subgroup_size_control); vkCreateComputePipelines
    // synchronizes the cache itself
    fn build(device: &ash::Device, cache: vk::PipelineCache, spec: &PipelineSpec) -> Result<Self> {
        let PipelineSpec {
            spirv,
            bindings,
            push_bytes,
            name,
            subgroup,
            constant,
        } = *spec;
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv))
            .map_err(|err| Error::Invalid(format!("embedded SPIR-V: {err}")))?;
        let writes = writable_bindings(&code);
        let layout_bindings: Vec<_> = (0..bindings as u32)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_bytes)];
        // SAFETY: each create info and what it points to outlives its call; the module is only needed until the
        // pipeline exists
        unsafe {
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&layout_bindings),
                None,
            )?;
            let set_layouts = [set_layout];
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )?;
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(subgroup.unwrap_or_default());
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            if subgroup.is_some() {
                stage = stage.push_next(&mut required);
            }
            let entries = [vk::SpecializationMapEntry::default()
                .constant_id(0)
                .offset(0)
                .size(4)];
            let data = constant.unwrap_or_default().to_ne_bytes();
            let specialization = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&data);
            if constant.is_some() {
                stage = stage.specialization_info(&specialization);
            }
            let created = device.create_compute_pipelines(
                cache,
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            );
            device.destroy_shader_module(module, None);
            let pipeline = created.map_err(|(_, err)| err)?[0];
            Ok(Self {
                set_layout,
                layout,
                pipeline,
                bindings,
                push_bytes,
                name,
                writes,
            })
        }
    }

    /// Records one dispatch over `groups` workgroups with `buffers` bound in order and `push` as its constants, ordered
    /// before whatever is recorded after it.
    pub(crate) fn dispatch(
        &self,
        recorder: &Recorder<'_>,
        buffers: &[vk::Buffer],
        push: &[u8],
        groups: [u32; 3],
    ) {
        self.record(recorder, buffers, push, groups, true);
    }

    /// As [`Self::dispatch`], but free to run alongside the next dispatch, which must neither read what this one
    /// writes nor write what it reads.
    pub(crate) fn dispatch_beside_next(
        &self,
        recorder: &Recorder<'_>,
        buffers: &[vk::Buffer],
        push: &[u8],
        groups: [u32; 3],
    ) {
        self.record(recorder, buffers, push, groups, false);
    }

    fn record(
        &self,
        recorder: &Recorder<'_>,
        buffers: &[vk::Buffer],
        push: &[u8],
        groups: [u32; 3],
        ordered: bool,
    ) {
        assert_eq!(
            buffers.len(),
            self.bindings,
            "a dispatch binds every buffer its pipeline takes"
        );
        assert_eq!(
            push.len(),
            self.push_bytes as usize,
            "a dispatch pushes its pipeline's constants whole"
        );
        let infos: Vec<_> = buffers
            .iter()
            .map(|&buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        let context = recorder.context;
        for (b, &buffer) in buffers.iter().enumerate() {
            if self.writes & (1 << b) != 0 {
                context.forget_quantized(buffer);
            }
        }
        let label = context.labels.as_ref().map(|labels| {
            (
                labels,
                std::ffi::CString::new(self.name).unwrap_or_default(),
            )
        });
        recorder.settle(crate::stream::Owed::COMPUTE);
        // SAFETY: the command buffer is recording, and the descriptors' buffers outlive the submit (its caller waits)
        unsafe {
            if let Some((labels, name)) = &label {
                labels.cmd_begin_debug_utils_label(
                    recorder.buffer,
                    &vk::DebugUtilsLabelEXT::default().label_name(name),
                );
            }
            context.device.cmd_bind_pipeline(
                recorder.buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            context.push_descriptor.cmd_push_descriptor_set(
                recorder.buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &writes,
            );
            context.device.cmd_push_constants(
                recorder.buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            );
            context
                .device
                .cmd_dispatch(recorder.buffer, groups[0], groups[1], groups[2]);
            if let Some((labels, _)) = &label {
                labels.cmd_end_debug_utils_label(recorder.buffer);
            }
        }
        if ordered {
            // the next command reads what this one wrote
            recorder.owe(crate::stream::Owed::WRITES);
        }
        recorder.dispatches.set(recorder.dispatches.get() + 1);
        recorder.profile(self.name);
    }

    /// # Safety
    /// No submit may still use the pipeline.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        // SAFETY: the caller's contract
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
