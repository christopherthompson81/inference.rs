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
}

impl Pipeline {
    pub(crate) fn new(
        device: &ash::Device,
        spirv: &[u8],
        bindings: usize,
        push_bytes: u32,
    ) -> Result<Self> {
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv))
            .map_err(|err| Error::Invalid(format!("embedded SPIR-V: {err}")))?;
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
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
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
            })
        }
    }

    /// Records one dispatch over `groups` workgroups with `buffers` bound in order and `push` as its constants.
    pub(crate) fn dispatch(
        &self,
        recorder: &Recorder<'_>,
        buffers: &[vk::Buffer],
        push: &[u8],
        groups: [u32; 3],
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
        // SAFETY: the command buffer is recording, and the descriptors' buffers outlive the submit (its caller waits)
        unsafe {
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
        }
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
