//! The stack RWKV-6 and RWKV-7 share: embeddings, layers that each read the previous token through two token shifts
//! and carry per-head WKV matrices, a final LayerNorm and the head. No attention: each layer's state is the hybrid
//! cache's conv slot (its two shifted inputs, `2 * hidden` wide and one step deep) and its WKV matrices (f32).

use std::sync::{Arc, Mutex};

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::nn::LayerNorm;
use inference_tensor::{DType, Device, Module, Result, Tensor};

use crate::gdn::RecurrentBatchKind;
use crate::kv_cache::{
    EitherCache, HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType,
    RecurrentLayerConfig, RecurrentStateSpec,
};
use crate::layers::embedding_with_legacy_tied_uqff;
use crate::model::{IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel};
use crate::paged_attention::ModelConfigMetadata;
use crate::utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder};

/// The context the planner may offer a sequence: RWKV has no positions.
pub const MAX_POSITIONS: usize = 1 << 20;

/// One RWKV layer: its time-mix and channel-mix with their norms.
pub(crate) trait RwkvBlock: Send + Sync {
    /// `x` advanced through the layer, with the time-mix's and channel-mix's last inputs (the next step's shifts);
    /// `shifts` the previous ones, `state` the WKV matrices (advanced in place), `v_first` RWKV-7's first value.
    fn forward(
        &self,
        x: &Tensor,
        shifts: [&Tensor; 2],
        state: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<[Tensor; 3]>;

    /// The tensors ISQ does not take, under the layer's prefix.
    fn residual_tensors(&self, uvb: &UnVarBuilder);
}

/// Where a checkpoint keeps the stack's ends and layers.
pub(crate) struct RwkvNames {
    pub root: &'static str,
    pub embeddings: &'static str,
    pub norm: &'static str,
    pub layers: &'static str,
    pub head: &'static str,
}

/// The stack's widths and its norms.
pub(crate) struct RwkvShape<'a> {
    pub vocab: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub tie_word_embeddings: bool,
    pub norm_eps: f64,
    pub quantization_config: &'a Option<QuantizedConfig>,
    /// llama.cpp's RWKV-6 GGUFs divide each layer's output weights by `2^(layer / n)` and halve the hidden state
    /// after every `n`th layer to match.
    pub rescaled_every: Option<usize>,
}

impl RwkvShape<'_> {
    /// A sequence's state: per layer the two shifted inputs and the heads' WKV matrices in f32.
    pub fn recurrent_state_bytes(&self, dtype: DType) -> usize {
        self.layers
            * (2 * self.hidden * dtype.size_in_bytes()
                + self.heads * self.head_dim * self.head_dim * DType::F32.size_in_bytes())
    }

    pub fn layer_norm(&self, vb: ShardedVarBuilder) -> Result<LayerNorm> {
        let config = inference_tensor::nn::LayerNormConfig {
            eps: self.norm_eps,
            ..Default::default()
        };
        crate::layers::layer_norm(self.hidden, config, vb)
    }
}

pub struct RwkvModel {
    embeddings: Arc<dyn QuantMethod>,
    // layer 0's LayerNorm on the embeddings
    pre_norm: LayerNorm,
    blocks: Vec<Box<dyn RwkvBlock>>,
    norm: LayerNorm,
    lm_head: Arc<dyn QuantMethod>,
    names: RwkvNames,
    pre_norm_name: &'static str,
    dtype: DType,
    hidden: usize,
    rescaled_every: Option<usize>,
    cache: EitherCache,
    device: Device,
    mapper: Box<dyn crate::device_map::DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl RwkvModel {
    /// The stack over `block` (each layer's builder, from its device-placed `VarBuilder`); `pre_norm` names layer 0's
    /// LayerNorm on the embeddings inside its layer.
    pub(crate) fn new(
        shape: &RwkvShape<'_>,
        names: RwkvNames,
        pre_norm: &'static str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        block: impl Fn(usize, ShardedVarBuilder) -> Result<Box<dyn RwkvBlock>> + Sync,
    ) -> Result<Self> {
        if shape.hidden != shape.heads * shape.head_dim {
            inference_tensor::bail!(
                "RWKV hidden size {} is not {} heads of {}",
                shape.hidden,
                shape.heads,
                shape.head_dim
            );
        }
        let vb_m = vb.pp(names.root);
        let vb_head = vb.pp(names.head);
        let mapper = normal_loading_metadata.mapper;
        let loading_isq = normal_loading_metadata.loading_isq;
        let dtype = vb_m.dtype();
        let embeddings = embedding_with_legacy_tied_uqff(
            shape.vocab,
            shape.hidden,
            mapper.set_nm_device(vb_m.pp(names.embeddings), loading_isq),
            shape
                .tie_word_embeddings
                .then(|| mapper.set_nm_device(vb_head.clone(), loading_isq)),
            shape.quantization_config,
        )?;
        let norm = shape.layer_norm(mapper.set_nm_device(vb_m.pp(names.norm), false))?;
        let lm_head = if shape.tie_word_embeddings {
            embeddings.clone()
        } else {
            ReplicatedLayer::new(
                shape.hidden,
                shape.vocab,
                shape.quantization_config,
                false,
                mapper.set_nm_device(vb_head, loading_isq),
            )?
        };
        let vb_l = vb_m.pp(names.layers);
        let pre_norm_layer =
            shape.layer_norm(mapper.set_device(0, vb_l.pp(0), loading_isq).pp(pre_norm))?;
        let blocks = NiceProgressBar::<_, 'b'>(
            0..shape.layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer| {
            block(layer, mapper.set_device(layer, vb_l.pp(layer), loading_isq))
        })?;
        let layer_devices = (0..shape.layers)
            .map(|layer| {
                mapper
                    .device_for(layer, false)
                    .unwrap_or(&normal_loading_metadata.real_device)
                    .clone()
            })
            .collect::<Vec<_>>();
        let cache = HybridCache::new(
            HybridCacheConfig {
                layer_types: vec![HybridLayerType::Recurrent; shape.layers],
                max_seq_len: MAX_POSITIONS,
                recurrent: RecurrentLayerConfig {
                    conv_dim: 2 * shape.hidden,
                    conv_width: 1,
                    state: RecurrentStateSpec::Opaque {
                        dims: vec![shape.heads, shape.head_dim, shape.head_dim],
                    },
                    recurrent_dtype: Some(DType::F32),
                },
                auxiliary: Vec::new(),
            },
            dtype,
            &layer_devices,
        )?;
        Ok(Self {
            embeddings,
            pre_norm: pre_norm_layer,
            blocks,
            norm,
            lm_head,
            names,
            pre_norm_name: pre_norm,
            dtype,
            hidden: shape.hidden,
            rescaled_every: shape.rescaled_every.filter(|n| *n > 0),
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(cache))),
            device: normal_loading_metadata.real_device,
            mapper,
            cfg: ModelConfigMetadata {
                max_seq_len: MAX_POSITIONS,
                num_layers: shape.layers,
                hidden_size: shape.hidden,
                num_kv_heads: 1,
                num_attn_heads: 1,
                sliding_window: None,
                k_head_dim: shape.head_dim,
                v_head_dim: shape.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
        })
    }

    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let x = self.embeddings.embedding_forward(input_ids, self.dtype)?;
        let mut x = self.pre_norm.forward(&self.mapper.map(x, 0)?)?;
        let metadata = ctx.recurrent_metadata().cloned().ok_or_else(|| {
            inference_tensor::Error::msg("RWKV needs the hybrid cache's recurrent metadata")
        })?;
        let fresh =
            metadata.batch_kind() == RecurrentBatchKind::Prefill && ctx.is_first_prompt_chunk();
        let mut cache = self.cache.hybrid();
        let mut v_first = None;
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;
            let indices = cache.state_indices_for_layer(layer_idx)?.ok_or_else(|| {
                inference_tensor::Error::msg(format!(
                    "RWKV layer {layer_idx} has no recurrent state indices"
                ))
            })?;
            let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(layer_idx) else {
                inference_tensor::bail!("RWKV layer {layer_idx} is not recurrent in the cache");
            };
            // the shifts `(b, 2h, 1)` and the WKV matrices, copied so the update never writes a snapshot's rows
            let (shifts, state) = if fresh {
                let rows = indices.dim(0)?;
                (
                    Tensor::zeros((rows, 2 * self.hidden, 1), x.dtype(), x.device())?,
                    Tensor::zeros(
                        pool.recurrent_state.dims()[1..].to_vec(),
                        DType::F32,
                        x.device(),
                    )?
                    .unsqueeze(0)?
                    .repeat((rows, 1, 1, 1))?,
                )
            } else {
                (
                    pool.gather_conv_state(&indices)?,
                    pool.gather_recurrent_state(&indices)?.copy()?,
                )
            };
            let shift = |at: usize| shifts.narrow(1, at, self.hidden)?.squeeze(2);
            let [out, attn_last, ffn_last] =
                block.forward(&x, [&shift(0)?, &shift(self.hidden)?], &state, &mut v_first)?;
            x = out;
            if self
                .rescaled_every
                .is_some_and(|n| (layer_idx + 1).is_multiple_of(n))
            {
                x = (x * 0.5)?;
            }
            let shifts = Tensor::cat(&[attn_last, ffn_last], 1)?
                .unsqueeze(2)?
                .to_dtype(shifts.dtype())?;
            pool.scatter_conv_state_with_host_indices(
                &indices,
                metadata.state_indices_host(),
                &shifts,
            )?;
            pool.scatter_recurrent_state_with_host_indices(
                &indices,
                metadata.state_indices_host(),
                &state,
            )?;
        }
        let x = self.mapper.cast_nm_device(&x, false)?;
        let x = self.norm.forward(&x)?;
        let x = ctx.logits(&x)?;
        ctx.lm_head(&*self.lm_head, &x)
    }
}

impl IsqModel for RwkvModel {
    // the norms, the token-shift mixes, the LoRAs and the small vectors stay as the checkpoint has them
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp(self.names.root);
        uvb_m.pp(self.names.embeddings).add(&self.embeddings);
        uvb_m.pp(self.names.norm).add(&self.norm);
        let layers = uvb_m.pp(self.names.layers);
        layers.pp(0).pp(self.pre_norm_name).add(&self.pre_norm);
        for (i, block) in self.blocks.iter().enumerate() {
            block.residual_tensors(&layers.pp(i));
        }
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for RwkvModel {}

impl NormalModel for RwkvModel {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }

    fn max_seq_len(&self) -> usize {
        MAX_POSITIONS
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for RwkvModel {}
