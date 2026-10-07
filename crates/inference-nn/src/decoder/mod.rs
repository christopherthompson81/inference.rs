//! The pre-norm attention + gated-MLP decoder most text models share, built from a [`DecoderSpec`].
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use inference_quant::{
    ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer, RowParallelLayer,
    ShardedVarBuilder,
};
use inference_tensor::{DType, Device, Module, Result, Tensor};

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeLoraTarget, MlpLayer},
    attention::{AttentionDispatch, AttentionMask, Sdpa, SdpaParams},
    device_map::{DeviceMappedMask, DeviceMapper},
    kv_cache::{EitherCache, KvCache, NormalCache, NormalCacheType},
    layers::{
        Activation, CausalMasker, Llama3RopeConfig, Llama3RopeSpec, Llama3RotaryEmbedding, Mlp,
        RmsNorm, RotaryEmbedding, YarnRopeConfig, embedding, embedding_with_legacy_tied_uqff,
        masker::CausalMaskConfig, masker::PastKvLenCache,
    },
    model::{IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel},
    paged_attention::{
        AttentionImplementation, KvCacheLayout, ModelConfigMetadata, PagedAttention,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

// Llama 3 checkpoints may carry per-frequency rope factors under this name
const ROPE_FREQS: &str = "rope_freqs.weight";
const QK_NORM_BEFORE_ROPE: (&str, &str) = ("q_norm", "k_norm");

const AMOE_LORA_TARGETS: &[AnyMoeLoraTarget] = &[
    AnyMoeLoraTarget::up("gate_proj"),
    AnyMoeLoraTarget::up("up_proj"),
    AnyMoeLoraTarget::down("down_proj"),
];

/// How the stack's RoPE tables are built.
#[derive(Clone, Debug)]
pub enum RopeKind {
    Default {
        theta: f32,
    },
    /// Llama 3 or linear scaling, with the checkpoint's `rope_freqs.weight` factors when it has them.
    Llama3 {
        theta: f32,
        scaling: Option<Llama3RopeConfig>,
    },
    Yarn(YarnRopeConfig),
}

/// Per-head q/k RMS norm: fused into RoPE as `q_norm`/`k_norm`, or applied after it under the given names.
#[derive(Clone, Copy, Debug)]
pub enum QkNorm {
    BeforeRope,
    AfterRope { q: &'static str, k: &'static str },
}

/// Mistral's position-dependent query scaling, `1 + scale * ln(1 + floor(pos / floor_scale))`.
#[derive(Clone, Copy, Debug)]
pub struct AttentionTemperature {
    pub scale: f32,
    pub floor_scale: usize,
}

/// The shape and switches of one decoder stack; a model's config builds it.
#[derive(Clone, Debug)]
pub struct DecoderSpec {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub rms_norm_eps: f64,
    pub rope: RopeKind,
    pub max_position_embeddings: usize,
    pub qkv_bias: bool,
    pub qk_norm: Option<QkNorm>,
    /// Layers that skip RoPE (NoPE).
    pub no_rope_layers: Vec<usize>,
    pub attention_temperature: Option<AttentionTemperature>,
    /// One entry per layer: the sliding window it attends over, or `None` for full attention.
    pub layer_windows: Vec<Option<usize>>,
    pub tie_word_embeddings: bool,
    pub quantization_config: Option<QuantizedConfig>,
}

impl DecoderSpec {
    pub fn num_layers(&self) -> usize {
        self.layer_windows.len()
    }

    /// The window the stack's sliding layers share; the mask is built once for all of them.
    pub fn sliding_window(&self) -> Option<usize> {
        self.layer_windows.iter().flatten().next().copied()
    }

    fn rope(
        &self,
        vb_m: &ShardedVarBuilder,
        device: &Device,
        is_gptx: bool,
        dtype: DType,
    ) -> Result<RotaryEmbedding> {
        match &self.rope {
            RopeKind::Default { theta } => RotaryEmbedding::new(
                *theta,
                self.head_dim,
                self.max_position_embeddings,
                device,
                is_gptx,
                dtype,
            ),
            RopeKind::Llama3 { theta, scaling } => {
                let freq_factors = vb_m
                    .contains_tensor(ROPE_FREQS)
                    .then(|| {
                        vb_m.clone()
                            .set_device(device.clone())
                            .get_unchecked_dtype(ROPE_FREQS, DType::F32)
                    })
                    .transpose()?;
                let spec = Llama3RopeSpec {
                    rope_theta: *theta,
                    head_dim: self.head_dim,
                    max_position_embeddings: self.max_position_embeddings,
                    scaling: scaling.as_ref(),
                };
                Ok(
                    Llama3RotaryEmbedding::new(
                        dtype,
                        spec,
                        device,
                        is_gptx,
                        freq_factors.as_ref(),
                    )?
                    .into_inner(),
                )
            }
            RopeKind::Yarn(yarn) => RotaryEmbedding::new_yarn(yarn, device, is_gptx, dtype),
        }
    }

    fn cache_types(&self) -> Vec<NormalCacheType> {
        self.layer_windows
            .iter()
            .map(|window| match window {
                Some(window) => NormalCacheType::SlidingWindow { window: *window },
                None => NormalCacheType::Normal {
                    max_seq_len: self.max_position_embeddings,
                },
            })
            .collect()
    }
}

/// Separate q/k/v/o projections, optional per-head q/k RMS norm, RoPE, and the engine's attention dispatch.
pub struct AttentionBlock {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    qk_norm: Option<(QkNorm, RmsNorm, RmsNorm)>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    attention_temperature: Option<AttentionTemperature>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

/// Where a layer's weights come from and how they are placed.
struct LayerLoad<'a> {
    spec: &'a DecoderSpec,
    mapper: &'a dyn DeviceMapper,
    layer_idx: usize,
    loading_isq: bool,
    comm: &'a Arc<inference_quant::Comm>,
}

impl AttentionBlock {
    fn new(
        load: &LayerLoad<'_>,
        vb: ShardedVarBuilder,
        rotary_emb: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let LayerLoad {
            spec,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = *load;
        let (hidden, head_dim) = (spec.hidden_size, spec.head_dim);
        let qc = &spec.quantization_config;
        let q_proj = ColumnParallelLayer::new(
            hidden,
            spec.num_heads * head_dim,
            qc,
            spec.qkv_bias,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = inference_quant::compute_kv_shard(spec.num_kv_heads, head_dim, comm)?;
        let kv = |name| {
            ColumnParallelLayer::new_with_shard(
                hidden,
                spec.num_kv_heads * head_dim,
                qc,
                spec.qkv_bias,
                comm,
                kv_shard,
                mapper.set_device(layer_idx, vb.pp(name), loading_isq),
            )
        };
        let (k_proj, v_proj) = (kv("k_proj")?, kv("v_proj")?);
        let o_proj = RowParallelLayer::new(
            spec.num_heads * head_dim,
            hidden,
            qc,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let qk_norm = spec
            .qk_norm
            .map(|placement| -> Result<_> {
                let (q, k) = match placement {
                    QkNorm::BeforeRope => QK_NORM_BEFORE_ROPE,
                    QkNorm::AfterRope { q, k } => (q, k),
                };
                let norm = |name| {
                    RmsNorm::new(
                        head_dim,
                        spec.rms_norm_eps,
                        mapper.set_device(layer_idx, vb.pp(name), false),
                    )
                };
                Ok((placement, norm(q)?, norm(k)?))
            })
            .transpose()?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qk_norm,
            num_heads: spec.num_heads / comm.world_size(),
            num_kv_heads: (spec.num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb: (!spec.no_rope_layers.contains(&layer_idx)).then_some(rotary_emb),
            attention_temperature: spec.attention_temperature,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: inference_quant::compute_n_kv_groups(
                    spec.num_kv_heads,
                    spec.num_heads,
                    comm,
                )?,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: spec.layer_windows[layer_idx],
                sinks: None,
                chunk: None,
            },
        })
    }

    /// Without a `kv_cache` the block attends over this call's tokens only, as an encoder does.
    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: Option<&mut KvCache>,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let (q, k, v) =
            crate::ops::qkv_projections(xs, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        let heads = |t: Tensor, n: usize| -> Result<Tensor> {
            if q_len != 1 {
                t.reshape((b_sz, q_len, n, self.head_dim))?.transpose(1, 2)
            } else {
                t.reshape((b_sz, n, q_len, self.head_dim))
            }
        };
        let (q, k, v) = (
            heads(q, self.num_heads)?,
            heads(k, self.num_kv_heads)?,
            heads(v, self.num_kv_heads)?,
        );

        let (q, k) = self.rope_and_norm(q, k, ctx)?;

        let attn_output = match kv_cache {
            Some(kv_cache) => AttentionDispatch {
                paged_attn: self.paged_attn.as_ref(),
                paged_layer: ctx.paged_layer(layer_idx),
                kv_cache,
                sdpa_params: &self.sdpa_params,
                flash_params: ctx.flash_params(),
            }
            .run(&q, &k, &v, attention_mask)?,
            None => Sdpa.run_attention(
                &q,
                &k,
                &v,
                attention_mask,
                Some(ctx.flash_params()),
                &self.sdpa_params,
            )?,
        };
        let attn_output = if !matches!(attention_mask, AttentionMask::None) {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn_output)
    }

    fn rope_and_norm(
        &self,
        q: Tensor,
        k: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<(Tensor, Tensor)> {
        let Some(rotary_emb) = &self.rotary_emb else {
            return Ok((q, k));
        };
        let positions = ctx
            .text_positions(q.device(), q.dim(2)?)?
            .ok_or_else(|| inference_tensor::Error::msg("missing RoPE positions"))?;
        let (q, k) = match &self.qk_norm {
            Some((QkNorm::BeforeRope, q_norm, k_norm)) => rotary_emb.forward_qk_norm(
                &q,
                &k,
                q_norm.weight(),
                k_norm.weight(),
                q_norm.eps(),
                k_norm.eps(),
                positions,
            )?,
            Some((QkNorm::AfterRope { .. }, q_norm, k_norm)) => {
                let (q, k) = rotary_emb.forward(&q, &k, positions)?;
                (q_norm.forward(&q)?, k_norm.forward(&k)?)
            }
            None => rotary_emb.forward(&q, &k, positions)?,
        };
        let Some(AttentionTemperature { scale, floor_scale }) = self.attention_temperature else {
            return Ok((q, k));
        };
        let (b_sz, _, q_len, _) = q.dims4()?;
        let floor = (positions.to_dtype(DType::F32)? / floor_scale as f64)?.floor()?;
        let scales =
            ((((floor + 1.)?.log()? * f64::from(scale))? + 1.)?).reshape((b_sz, 1, q_len, 1))?;
        let q = q
            .to_dtype(DType::F32)?
            .broadcast_mul(&scales)?
            .to_dtype(q.dtype())?;
        Ok((q, k))
    }

    fn add_residual(&self, uvb: &UnVarBuilder) {
        if let Some((placement, q_norm, k_norm)) = &self.qk_norm {
            let (q, k) = match placement {
                QkNorm::BeforeRope => QK_NORM_BEFORE_ROPE,
                QkNorm::AfterRope { q, k } => (*q, *k),
            };
            uvb.pp(q).add(q_norm);
            uvb.pp(k).add(k_norm);
        }
    }
}

/// Pre-norm residual attention then MLP.
pub struct DecoderLayer {
    pub self_attn: AttentionBlock,
    pub mlp: Box<dyn MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        load: &LayerLoad<'_>,
        vb: ShardedVarBuilder,
        rotary_emb: Arc<RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let LayerLoad {
            spec,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = *load;
        let self_attn = AttentionBlock::new(
            load,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            rotary_emb,
            paged_attn,
        )?;
        let mlp = Mlp::new(
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            spec.hidden_size,
            spec.intermediate_size,
            &spec.quantization_config,
            spec.hidden_act,
            comm,
        )?;
        let norm = |name| {
            RmsNorm::new(
                spec.hidden_size,
                spec.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp(name), false),
            )
        };
        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: Option<&mut KvCache>,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

/// Each layer attends through `full`, or through `sliding` when it has a window; either is built only if used.
pub struct LayerMasks {
    full: Option<DeviceMappedMask>,
    sliding: Option<DeviceMappedMask>,
}

/// Embeddings, layers and final norm: the part a causal LM and an embedder share.
pub struct DecoderStack {
    pub embed_tokens: Arc<dyn QuantMethod>,
    pub layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    dtype: DType,
    sliding_window: Option<usize>,
    has_full_layers: bool,
    pub device: Device,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl DecoderStack {
    /// `tied_lm_head` names the head an untied-in-name, tied-in-fact UQFF may carry; `None` for an embedder.
    pub fn new(
        spec: &DecoderSpec,
        vb_m: ShardedVarBuilder,
        tied_lm_head: Option<ShardedVarBuilder>,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: &AttentionImplementation,
    ) -> Result<Self> {
        // a NoPE layer skips the RoPE step that carries q/k norm and the temperature
        if !spec.no_rope_layers.is_empty()
            && (spec.qk_norm.is_some() || spec.attention_temperature.is_some())
        {
            inference_tensor::bail!(
                "NoPE layers with q/k norm or attention temperature are not supported"
            );
        }
        // one sliding mask serves every sliding layer
        if spec
            .layer_windows
            .iter()
            .flatten()
            .any(|window| Some(*window) != spec.sliding_window())
        {
            inference_tensor::bail!(
                "decoder layers slide over different windows: {:?}",
                spec.layer_windows
            );
        }
        if let Some(quant_cfg) = &spec.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;
        let loading_isq = normal_loading_metadata.loading_isq;
        let dtype = vb_m.dtype();
        let embed_vb = mapper.set_nm_device(vb_m.pp("embed_tokens"), loading_isq);
        let embed_tokens = match tied_lm_head {
            Some(head) => embedding_with_legacy_tied_uqff(
                spec.vocab_size,
                spec.hidden_size,
                embed_vb,
                spec.tie_word_embeddings
                    .then(|| mapper.set_nm_device(head, loading_isq)),
                &spec.quantization_config,
            )?,
            None => embedding(
                spec.vocab_size,
                spec.hidden_size,
                embed_vb,
                &spec.quantization_config,
            )?,
        };

        let real_device = &normal_loading_metadata.real_device;
        let ropes = crate::device_map::per_layer_device(
            &*mapper,
            spec.num_layers(),
            real_device,
            |device| spec.rope(&vb_m, device, is_gptx, dtype),
        )?;
        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..spec.num_layers(),
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| -> Result<DecoderLayer> {
            let device = mapper.device_for(layer_idx, false).unwrap_or(real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(spec.head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            let load = LayerLoad {
                spec,
                mapper: &*mapper,
                layer_idx,
                loading_isq,
                comm: &comm,
            };
            DecoderLayer::new(&load, vb_l.pp(layer_idx), rotary_emb, paged_attn)
        })?;
        let norm = RmsNorm::new(
            spec.hidden_size,
            spec.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            dtype,
            sliding_window: spec.sliding_window(),
            has_full_layers: spec.layer_windows.iter().any(Option::is_none),
            device: real_device.clone(),
            mapper,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)
    }

    /// The full mask, and the sliding one when a layer slides, for this call's tokens after `past`.
    pub fn masks(
        &self,
        input_ids: &Tensor,
        dtype: DType,
        past: &dyn PastKvLenCache,
        is_first_prompt_chunk: bool,
    ) -> Result<LayerMasks> {
        let mask = |sliding_window| -> Result<DeviceMappedMask> {
            let mask = CausalMasker.make_causal_mask(
                input_ids,
                past,
                dtype,
                &CausalMaskConfig {
                    sliding_window,
                    ..Default::default()
                },
            )?;
            // PagedAttention prompt chunking
            let mask = if is_first_prompt_chunk {
                mask
            } else {
                AttentionMask::None
            };
            DeviceMappedMask::new(mask, &*self.mapper)
        };
        Ok(LayerMasks {
            full: self.has_full_layers.then(|| mask(None)).transpose()?,
            sliding: self
                .sliding_window
                .map(|window| mask(Some(window)))
                .transpose()?,
        })
    }

    /// The normed hidden states; `cache` is `None` for an encoder pass over the call's tokens alone.
    pub fn forward(
        &self,
        mut xs: Tensor,
        masks: &LayerMasks,
        mut cache: Option<&mut [KvCache]>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let layer_mask = match (&masks.sliding, &masks.full) {
                (Some(sliding), _) if layer.self_attn.sdpa_params.sliding_window.is_some() => {
                    sliding
                }
                (_, Some(full)) => full,
                (Some(sliding), None) => sliding,
                (None, None) => unreachable!("a stack has a full or a sliding layer"),
            };
            let kv_cache = cache.as_deref_mut().map(|cache| &mut cache[i]);
            xs = layer.forward(&xs, &layer_mask.get(xs.device()), kv_cache, ctx, i)?;
        }
        xs.to_device(&self.device)?.apply(&self.norm)
    }

    /// The tensors ISQ leaves alone, under the stack's own prefix.
    pub fn residual_uvb(&self, uvb_m: &UnVarBuilder) {
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            layer.self_attn.add_residual(&uvb_l.pp("self_attn"));
        }
    }
}

/// A [`DecoderStack`] under `model.` with an `lm_head`, as the engine runs a text model.
pub struct CausalLm {
    stack: DecoderStack,
    lm_head: Arc<dyn QuantMethod>,
    cache: EitherCache,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
}

impl CausalLm {
    pub fn new(
        spec: &DecoderSpec,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        Self::new_inner(
            spec,
            vb.pp("model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    /// As [`CausalLm::new`] with the stack and head under the prefixes a multimodal checkpoint gives them.
    pub fn new_inner(
        spec: &DecoderSpec,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let loading_isq = normal_loading_metadata.loading_isq;
        let stack = DecoderStack::new(
            spec,
            vb_m,
            Some(vb_lm_head.clone()),
            is_gptx,
            normal_loading_metadata,
            &attention_mechanism,
        )?;
        let lm_head = if spec.tie_word_embeddings {
            stack.embed_tokens.clone()
        } else {
            ReplicatedLayer::new(
                spec.hidden_size,
                spec.vocab_size,
                &spec.quantization_config,
                false,
                stack.mapper.set_nm_device(vb_lm_head, loading_isq),
            )?
        };
        let world_size = stack.mapper.get_comm_for(0)?.world_size();
        Ok(Self {
            lm_head,
            cache: EitherCache::Normal(NormalCache::from_types(spec.cache_types())),
            max_seq_len: spec.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: spec.max_position_embeddings,
                num_layers: spec.num_layers(),
                hidden_size: spec.hidden_size,
                num_kv_heads: (spec.num_kv_heads / world_size).max(1),
                num_attn_heads: spec.num_heads / world_size,
                sliding_window: spec.sliding_window(),
                k_head_dim: spec.head_dim,
                v_head_dim: spec.head_dim,
                kv_cache_layout: KvCacheLayout::Standard,
            },
            stack,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.stack.embed(input_ids)
    }

    pub fn embed_dtype(&self) -> DType {
        self.stack.dtype
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward_embeds(input_ids, self.stack.embed(input_ids)?, ctx)
    }

    /// Runs the stack over `xs`, embeddings a multimodal model may have spliced media into.
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        xs: Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let cache = &mut self.cache.normal().0;
        let masks = self.stack.masks(
            input_ids,
            xs.dtype(),
            &ctx.mask_cache(cache),
            ctx.is_first_prompt_chunk(),
        )?;
        let xs = self.stack.forward(xs, &masks, Some(cache), ctx)?;
        let xs = ctx.logits(&xs)?;
        ctx.lm_head(&*self.lm_head, &xs)
    }
}

impl CausalLm {
    /// The tensors ISQ leaves alone, under `uvb_m`, the stack's prefix.
    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        self.stack.residual_uvb(&uvb_m);
        uvb_m.to_safetensors()
    }
}

impl IsqModel for CausalLm {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.stack.residual_uvb(&uvb.pp("model"));
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for CausalLm {}

impl NormalModel for CausalLm {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.stack.device
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
    fn supports_packed_prefill(&self) -> bool {
        true
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        true
    }
}

impl AnyMoeBaseModelMixin for CausalLm {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.stack.layers.iter().map(|layer| &*layer.mlp).collect()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.stack
            .layers
            .iter_mut()
            .map(|layer| &mut layer.mlp)
            .collect()
    }
    fn amoe_lora_targets(&self) -> &'static [AnyMoeLoraTarget] {
        AMOE_LORA_TARGETS
    }
    fn amoe_fine_tuned_expert(
        &self,
        layer: usize,
        base: &dyn MlpLayer,
        vb: ShardedVarBuilder,
    ) -> Result<Box<dyn MlpLayer>> {
        let (dtype, device) = base.dtype_device();
        Ok(Box::new(Mlp::replicate(
            base.get_params(),
            vb.set_dtype(dtype).set_device(device),
            base.hidden_act(),
            &self.stack.mapper.get_comm_for(layer)?,
        )?))
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_SEQ: usize = 4096;
    const WINDOW: usize = 128;

    #[test]
    fn eager_cache_layout_follows_each_layers_window() {
        let spec = DecoderSpec {
            vocab_size: 8,
            hidden_size: 8,
            intermediate_size: 8,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 8,
            hidden_act: Activation::Silu,
            rms_norm_eps: 1e-6,
            rope: RopeKind::Default { theta: 1e4 },
            max_position_embeddings: MAX_SEQ,
            qkv_bias: false,
            qk_norm: None,
            no_rope_layers: Vec::new(),
            attention_temperature: None,
            layer_windows: vec![None, Some(WINDOW), None],
            tie_word_embeddings: false,
            quantization_config: None,
        };
        let types = spec.cache_types();
        assert!(matches!(
            types[0],
            NormalCacheType::Normal {
                max_seq_len: MAX_SEQ
            }
        ));
        assert!(matches!(
            types[1],
            NormalCacheType::SlidingWindow { window: WINDOW }
        ));
        assert!(matches!(
            types[2],
            NormalCacheType::Normal {
                max_seq_len: MAX_SEQ
            }
        ));
        assert_eq!(spec.sliding_window(), Some(WINDOW));
    }
}
