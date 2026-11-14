//! A standard decoder stack's layers built one at a time, on any device, from the checkpoint: for a converter that
//! streams a model larger than its device through it (exllamav3's way of writing EXL3), the model itself loaded on
//! the host.

use std::collections::HashMap;
use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder, TrackedModule};
use inference_tensor::{DType, Device, Result, Tensor};

use super::{
    AttentionBlock, CausalLm, DecoderLayer, DecoderSpec, Ffn, LayerAttention, LayerFfn, LayerLoad,
    StandardLayers, build_layer,
};
use crate::attention::FlashParams;
use crate::device_map::{DeviceMappedMask, DeviceMapper, DummyDeviceMapper};
use crate::layers::masker::NotACache;
use crate::layers::{CausalMaskConfig, CausalMasker};
use crate::model::{LayerStream, ModelForwardContext, StreamedLayer};
use crate::paged_attention::AttentionImplementation;

/// What rebuilds one layer of a stack [`super::DecoderStack::new`] made.
pub(super) struct LayerRebuild {
    pub spec: DecoderSpec,
    pub vb_m: ShardedVarBuilder,
    pub is_gptx: bool,
    pub attention: AttentionImplementation,
    pub dtype: DType,
}

impl LayerRebuild {
    // layer `layer_idx` on `device`, with the modules it added to the load's tracker taken back out of it
    fn layer(
        &self,
        layer_idx: usize,
        device: &Device,
    ) -> Result<(DecoderLayer<AttentionBlock, Ffn>, Vec<TrackedModule>)> {
        let mapper = DummyDeviceMapper::new(device.clone());
        let vb_m = self.vb_m.clone().set_device(device.clone());
        let rope = |kind| -> Result<_> {
            let rope = self
                .spec
                .rope(kind, &vb_m, device, self.is_gptx, self.dtype)?;
            Ok(HashMap::from([(device.location(), Arc::new(rope))]))
        };
        let builder = StandardLayers {
            spec: &self.spec,
            ropes: rope(&self.spec.rope)?,
            local_ropes: self.spec.local_rope.as_ref().map(rope).transpose()?,
        };
        let comm = mapper.get_comm_for(layer_idx)?;
        let load = LayerLoad {
            mapper: &mapper,
            layer_idx,
            loading_isq: false,
            comm: &comm,
            device,
            attention: &self.attention,
        };
        let shape = self.spec.shape();
        let before = vb_m.tracker().get().len();
        let layer = build_layer(
            &builder,
            &shape,
            &load,
            vb_m.pp(shape.module_names.layers).pp(layer_idx),
        )?;
        let modules = vb_m.tracker().get().split_off(before);
        Ok((layer, modules))
    }
}

struct Layer {
    layer: DecoderLayer<AttentionBlock, Ffn>,
    layer_idx: usize,
    window: Option<usize>,
    modules: Vec<TrackedModule>,
}

impl StreamedLayer for Layer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let tokens = xs.dim(1)?;
        let ids = Tensor::zeros((1, tokens), DType::U32, xs.device())?;
        let mask = CausalMasker.make_causal_mask(
            &ids,
            &NotACache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window: self.window,
                ..Default::default()
            },
        )?;
        let mask = DeviceMappedMask::from_single(mask).get(xs.device());
        let (offsets, context_lens, position_ids) = ([0], [(0, tokens)], [tokens]);
        let flash = FlashParams::empty(true);
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash);
        self.layer
            .forward(xs, &mask, None, &mut ctx, self.layer_idx, None)
    }

    fn modules(&self) -> &[TrackedModule] {
        &self.modules
    }
}

impl<A: LayerAttention, F: LayerFfn> LayerStream for CausalLm<A, F> {
    fn num_layers(&self) -> usize {
        self.stack
            .rebuild
            .as_ref()
            .map_or(self.stack.layers.len(), |rebuild| rebuild.spec.num_layers())
    }

    fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        self.stack.embed(ids)
    }

    fn build_layer(&self, layer: usize, device: &Device) -> Result<Box<dyn StreamedLayer>> {
        let Some(rebuild) = &self.stack.rebuild else {
            inference_tensor::bail!("this decoder stack does not rebuild its layers");
        };
        let (built, modules) = rebuild.layer(layer, device)?;
        let window = built.self_attn.sliding_window();
        Ok(Box::new(Layer {
            layer: built,
            layer_idx: layer,
            window,
            modules,
        }))
    }

    fn final_norm(&self, xs: &Tensor) -> Result<Tensor> {
        self.stack.norm.forward(&xs.to_device(&self.stack.device)?)
    }

    fn head(&self) -> Arc<dyn QuantMethod> {
        self.lm_head.clone()
    }
}
