//! The stack RWKV-6, RWKV-7 and their Qwen2 hybrids share: embeddings, layers that each read the previous token
//! through their token shifts (two in RWKV, one in the hybrids) and carry per-head WKV matrices, a final norm and the
//! head. No attention: each layer's state is the hybrid cache's conv slot (its shifted inputs, one step deep) and its
//! WKV matrices (f32).

use std::sync::{Arc, Mutex};

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::nn::LayerNorm;
use inference_tensor::{DType, Device, Module, Result, Tensor};

use crate::layers::RmsNorm;

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

/// One RWKV layer: its time-mix and channel-mix (or MLP) with their norms.
pub(crate) trait RwkvBlock: Send + Sync {
    /// `x` advanced through the layer, with each token shift's last input (the next step's shifts); `shifts` the
    /// previous ones `(b, hidden)` each, `state` the WKV matrices (advanced in place), `v_first` RWKV-7's first value.
    fn forward(
        &self,
        x: &Tensor,
        shifts: &[Tensor],
        state: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)>;

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

/// RWKV's LayerNorms or the hybrids' RMSNorms.
pub(crate) enum RwkvNorm {
    Layer(LayerNorm),
    Rms(RmsNorm),
}

impl Module for RwkvNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Layer(norm) => norm.forward(x),
            Self::Rms(norm) => norm.forward(x),
        }
    }
}

impl RwkvNorm {
    pub fn residual_tensors(&self, uvb: &UnVarBuilder) {
        match self {
            Self::Layer(norm) => uvb.add(norm),
            Self::Rms(norm) => uvb.add(norm),
        }
    }
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
    /// RMSNorms (the Qwen2 hybrids) in place of LayerNorms
    pub rms_norm: bool,
    /// the token shifts per layer: the time-mix's and the channel-mix's, or only the time-mix's in the hybrids
    pub shifts: usize,
    pub quantization_config: &'a Option<QuantizedConfig>,
    /// llama.cpp's RWKV-6 GGUFs divide each layer's output weights by `2^(layer / n)` and halve the hidden state
    /// after every `n`th layer to match.
    pub rescaled_every: Option<usize>,
}

impl RwkvShape<'_> {
    /// A sequence's state: per layer the shifted inputs and the heads' WKV matrices in f32.
    pub fn recurrent_state_bytes(&self, dtype: DType) -> usize {
        self.layers
            * (self.shifts * self.hidden * dtype.size_in_bytes()
                + self.heads * self.head_dim * self.head_dim * DType::F32.size_in_bytes())
    }

    pub fn layer_norm(&self, vb: ShardedVarBuilder) -> Result<LayerNorm> {
        let config = inference_tensor::nn::LayerNormConfig {
            eps: self.norm_eps,
            ..Default::default()
        };
        crate::layers::layer_norm(self.hidden, config, vb)
    }

    pub fn norm(&self, vb: ShardedVarBuilder) -> Result<RwkvNorm> {
        Ok(if self.rms_norm {
            RwkvNorm::Rms(RmsNorm::new(self.hidden, self.norm_eps, vb)?)
        } else {
            RwkvNorm::Layer(self.layer_norm(vb)?)
        })
    }
}

pub struct RwkvModel {
    embeddings: Arc<dyn QuantMethod>,
    // layer 0's LayerNorm on the embeddings (RWKV's, not the hybrids'), with its name in the layer
    pre_norm: Option<(LayerNorm, &'static str)>,
    blocks: Vec<Box<dyn RwkvBlock>>,
    norm: RwkvNorm,
    lm_head: Arc<dyn QuantMethod>,
    names: RwkvNames,
    dtype: DType,
    hidden: usize,
    shifts: usize,
    rescaled_every: Option<usize>,
    cache: EitherCache,
    device: Device,
    mapper: Box<dyn crate::device_map::DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl RwkvModel {
    /// The stack over `block` (each layer's builder, from its device-placed `VarBuilder` and its comm); `pre_norm`
    /// names layer 0's LayerNorm on the embeddings inside its layer, when it has one.
    pub(crate) fn new(
        shape: &RwkvShape<'_>,
        names: RwkvNames,
        pre_norm: Option<&'static str>,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        block: impl Fn(
            usize,
            ShardedVarBuilder,
            &Arc<inference_quant::Comm>,
        ) -> Result<Box<dyn RwkvBlock>>
        + Sync,
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
        let norm = shape.norm(mapper.set_nm_device(vb_m.pp(names.norm), false))?;
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
        let pre_norm = pre_norm
            .map(|name| -> Result<(LayerNorm, &'static str)> {
                let vb = mapper.set_device(0, vb_l.pp(0), loading_isq).pp(name);
                Ok((shape.layer_norm(vb)?, name))
            })
            .transpose()?;
        let blocks = NiceProgressBar::<_, 'b'>(
            0..shape.layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer| {
            let comm = mapper.get_comm_for(layer)?;
            block(
                layer,
                mapper.set_device(layer, vb_l.pp(layer), loading_isq),
                &comm,
            )
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
                    conv_dim: shape.shifts * shape.hidden,
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
            pre_norm,
            blocks,
            norm,
            lm_head,
            names,
            dtype,
            hidden: shape.hidden,
            shifts: shape.shifts,
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
        let mut x = self.mapper.map(x, 0)?;
        if let Some((pre_norm, _)) = &self.pre_norm {
            x = pre_norm.forward(&x)?;
        }
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
            // the shifts `(b, n h, 1)` and the WKV matrices, copied so the update never writes a snapshot's rows
            let (shifts, state) = if fresh {
                let rows = indices.dim(0)?;
                (
                    Tensor::zeros((rows, self.shifts * self.hidden, 1), x.dtype(), x.device())?,
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
            let previous = (0..self.shifts)
                .map(|i| shifts.narrow(1, i * self.hidden, self.hidden)?.squeeze(2))
                .collect::<Result<Vec<_>>>()?;
            let (out, lasts) = block.forward(&x, &previous, &state, &mut v_first)?;
            x = out;
            if self
                .rescaled_every
                .is_some_and(|n| (layer_idx + 1).is_multiple_of(n))
            {
                x = (x * 0.5)?;
            }
            let shifts = Tensor::cat(&lasts, 1)?
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
        self.norm.residual_tensors(&uvb_m.pp(self.names.norm));
        let layers = uvb_m.pp(self.names.layers);
        if let Some((pre_norm, name)) = &self.pre_norm {
            layers.pp(0).pp(name).add(pre_norm);
        }
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
