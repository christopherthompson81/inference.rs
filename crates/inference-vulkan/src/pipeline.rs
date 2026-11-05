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

impl Pipeline {
    pub(crate) fn new(
        device: &ash::Device,
        spirv: &[u8],
        bindings: usize,
        push_bytes: u32,
        name: &'static str,
    ) -> Result<Self> {
        Self::with_subgroup(device, spirv, bindings, push_bytes, name, None)
    }

    /// [`Self::new`], its subgroups `subgroup` invocations wide when given (VK_EXT_subgroup_size_control).
    pub(crate) fn with_subgroup(
        device: &ash::Device,
        spirv: &[u8],
        bindings: usize,
        push_bytes: u32,
        name: &'static str,
        subgroup: Option<u32>,
    ) -> Result<Self> {
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
            let created = device.create_compute_pipelines(
                vk::PipelineCache::null(),
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
            if ordered {
                // the next dispatch reads what this one wrote
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                context.device.cmd_pipeline_barrier(
                    recorder.buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
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
