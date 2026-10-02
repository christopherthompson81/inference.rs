//! The decoder DeepSeek-V2, DeepSeek-V3, GLM4-MoE and GLM4-MoE-Lite share: MoE block, layers and model, over an attention.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod mla;

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};

pub use mla::{MlaAttention, MlaConfig, MlaKvLayout, mla_softmax_scale};

use crate::attention::{AttentionMask, FlashParams};
use crate::kv_cache::{EitherCache, KvCache, NormalCache};
use crate::layers::masker::CausalMaskConfig;
use crate::model::{IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel};
use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{Activation, CausalMasker, Mlp, RmsNorm, embedding_with_legacy_tied_uqff},
    moe::{GroupedRouter, GroupedRouterConfig, MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

/// The normalised config of one family model; `A` is its attention's own config.
#[derive(Clone, Debug)]
pub struct FamilyConfig<A> {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub hidden_act: Activation,
    pub quantization_config: Option<QuantizedConfig>,
    pub moe: Option<MoeSpec>,
    pub attn: A,
}

#[derive(Clone, Debug)]
pub struct MoeSpec {
    pub n_routed_experts: usize,
    // DeepSeek configs leave it optional and only unwrap it on a MoE layer
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: Option<usize>,
    pub shared_expert: Option<SharedExpert>,
    pub router: GroupedRouterConfig,
}

/// The shared expert of a MoE layer and its intermediate size.
#[derive(Clone, Copy, Debug)]
pub enum SharedExpert {
    /// A tensor-parallel `Mlp` (DeepSeek-V2/V3).
    Sharded(usize),
    /// A replicated gate/up/down `Expert` (GLM4-MoE, GLM4-MoE-Lite).
    Replicated(usize),
}

impl<A> FamilyConfig<A> {
    fn moe_spec(&self) -> &MoeSpec {
        self.moe.as_ref().expect("MoE layer without a MoE spec")
    }

    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.moe.as_ref().is_some_and(|moe| {
            layer_idx >= moe.first_k_dense_replace
                && moe
                    .moe_layer_freq
                    .is_none_or(|freq| layer_idx.is_multiple_of(freq))
        })
    }
}

/// What a decoder layer needs to build its attention.
pub struct LayerCtx<'a, A> {
    pub cfg: &'a FamilyConfig<A>,
    pub mapper: &'a dyn DeviceMapper,
    pub layer_idx: usize,
    pub loading_isq: bool,
    pub comm: &'a Arc<inference_quant::Comm>,
}

/// The attention half of a family decoder layer.
pub trait FamilyAttention: Sized + Send + Sync {
    type Config: Send + Sync;
    type Rope: Send + Sync;

    const CUDA_DECODE_GRAPHS: bool = false;

    fn rope(
        cfg: &FamilyConfig<Self::Config>,
        dtype: DType,
        device: &Device,
        is_gptx: bool,
    ) -> Result<Self::Rope>;

    fn paged_head_dim(cfg: &FamilyConfig<Self::Config>) -> usize;

    fn new(
        ctx: &LayerCtx<'_, Self::Config>,
        rotary_emb: Arc<Self::Rope>,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self>;

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor>;

    /// The tensors ISQ leaves alone, under `self_attn`.
    fn add_residual(&self, uvb: &UnVarBuilder);

    /// The projections, under `self_attn`, that only a MoE-experts-only ISQ keeps unquantized.
    fn add_projections(&self, uvb: &UnVarBuilder);

    fn model_metadata(
        cfg: &FamilyConfig<Self::Config>,
        attention_mechanism: &AttentionImplementation,
        real_device: &Device,
        world_size: usize,
    ) -> ModelConfigMetadata;
}

struct Expert {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Expert {
    fn new<A>(
        cfg: &FamilyConfig<A>,
        vb: ShardedVarBuilder,
        intermediate_size: usize,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        Ok(Self {
            gate: ReplicatedLayer::new(
                hidden_size,
                intermediate_size,
                &cfg.quantization_config,
                false,
                vb.pp("gate_proj"),
            )?,
            up: ReplicatedLayer::new(
                hidden_size,
                intermediate_size,
                &cfg.quantization_config,
                false,
                vb.pp("up_proj"),
            )?,
            down: ReplicatedLayer::new(
                intermediate_size,
                hidden_size,
                &cfg.quantization_config,
                false,
                vb.pp("down_proj"),
            )?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?;
        let rhs = self.up.forward(xs)?;
        self.down
            .forward(&crate::ops::mul_and_act(&lhs, &rhs, self.act)?)
    }
}

enum SharedMlp {
    Sharded(Mlp),
    Replicated(Expert),
}

impl SharedMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Sharded(mlp) => mlp.forward(xs),
            Self::Replicated(expert) => expert.forward(xs),
        }
    }
}

pub struct MoeGate {
    weight: Tensor,
    lora_site: Option<Arc<inference_quant::LoraSiteHandle>>,
    router: GroupedRouter,
}

impl MoeGate {
    pub fn new<A>(
        cfg: &FamilyConfig<A>,
        vb: ShardedVarBuilder,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let spec = cfg.moe_spec();
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
        let lora_site = inference_quant::register_dynamic_lora_site(
            &vb.clone().set_dtype(DType::F32),
            inference_quant::LoraLinearSpec::replicated(cfg.hidden_size, n_routed_experts),
        )?;
        let router = GroupedRouter::load(
            spec.router,
            spec.num_experts_per_tok.unwrap(),
            &vb,
            n_routed_experts,
        )?;
        Ok(Self {
            weight,
            lora_site,
            router,
        })
    }

    /// (topk_idx, topk_weight)
    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?.to_dtype(DType::F32)?;
        let logits = xs.broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        let logits = match &self.lora_site {
            Some(site) => inference_quant::apply_dynamic_lora_delta(site, &xs, logits)?,
            None => logits,
        };
        self.router.route(&logits)
    }
}

pub fn add_moe_gate_residual_tensors(
    uvb: &UnVarBuilder,
    weight: &Tensor,
    correction_bias: Option<&Tensor>,
) {
    uvb.add_tensor("weight", weight.clone());
    if let Some(bias) = correction_bias {
        uvb.add_tensor("e_score_correction_bias", bias.clone());
    }
}

struct Moe {
    experts: MoEExperts,
    shared_experts: Option<SharedMlp>,
    gate: MoeGate,
}

impl Moe {
    fn new<A>(ctx: &LayerCtx<'_, A>, vb: ShardedVarBuilder, real_device: Device) -> Result<Self> {
        let LayerCtx {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = *ctx;
        let spec = cfg.moe_spec();
        let n_routed_experts = spec.n_routed_experts;
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);

        let moe_cfg = MoEExpertsConfig {
            num_experts: n_routed_experts,
            num_experts_per_tok: spec.num_experts_per_tok.unwrap(),
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: spec.moe_intermediate_size,
            expert_proj_names: crate::moe::ExpertProjNames::DEFAULT,
        };

        let experts = MoEExperts::new(
            &moe_cfg,
            mapper.set_device(layer_idx, vb.clone(), loading_isq),
            layer_device,
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;

        let shared_vb = || mapper.set_device(layer_idx, vb.pp("shared_experts"), loading_isq);
        let shared_experts = match spec.shared_expert {
            Some(SharedExpert::Sharded(intermediate_size)) => Some(SharedMlp::Sharded(Mlp::new(
                shared_vb(),
                cfg.hidden_size,
                intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)),
            Some(SharedExpert::Replicated(intermediate_size)) => Some(SharedMlp::Replicated(
                Expert::new(cfg, shared_vb(), intermediate_size)?,
            )),
            None => None,
        };
        let gate = MoeGate::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("gate"), false),
            n_routed_experts,
        )?;
        Ok(Self {
            experts,
            shared_experts,
            gate,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;

        let (topk_idx, topk_weight) = self.gate.forward(xs)?;

        let mut y = self.experts.forward(xs, topk_weight, &topk_idx)?;
        y = y.reshape((b_size, seq_len, hidden_dim))?;

        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }

        Ok(y)
    }
}

enum MoeOrMlp {
    Moe(Box<Moe>),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct DecoderLayer<T> {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: T,
    moe_or_mlp: MoeOrMlp,
}

impl<T: FamilyAttention> DecoderLayer<T> {
    fn new(
        ctx: &LayerCtx<'_, T::Config>,
        rotary_emb: Arc<T::Rope>,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        real_device: Device,
    ) -> Result<Self> {
        let LayerCtx {
            cfg,
            mapper,
            layer_idx,
            loading_isq,
            comm,
        } = *ctx;
        let attn = T::new(ctx, rotary_emb, vb.pp("self_attn"), paged_attn)?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let moe_or_mlp = if cfg.is_moe_layer(layer_idx) {
            MoeOrMlp::Moe(Box::new(Moe::new(ctx, vb.pp("mlp"), real_device)?))
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?)
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: &AttentionMask,
        kv_cache: &mut KvCache,
        ctx: &mut ModelForwardContext<'_>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .attn
            .forward(&xs, attention_mask, kv_cache, ctx, layer_idx)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    fn add_residual(&self, uvb_l: &UnVarBuilder) {
        uvb_l.pp("input_layernorm").add(&self.input_layernorm);
        uvb_l
            .pp("post_attention_layernorm")
            .add(&self.post_attention_layernorm);
        self.attn.add_residual(&uvb_l.pp("self_attn"));
        if let MoeOrMlp::Moe(moe) = &self.moe_or_mlp {
            add_moe_gate_residual_tensors(
                &uvb_l.pp("mlp").pp("gate"),
                &moe.gate.weight,
                moe.gate.router.e_score_correction_bias(),
            );
        }
    }
}

pub struct FamilyModel<T> {
    lm_head: Arc<dyn QuantMethod>,
    embed_tokens: Arc<dyn QuantMethod>,
    dtype: DType,
    norm: RmsNorm,
    layers: Vec<DecoderLayer<T>>,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    cfg: ModelConfigMetadata,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl<T: FamilyAttention> FamilyModel<T> {
    pub fn new(
        cfg: &FamilyConfig<T::Config>,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let mapper = normal_loading_metadata.mapper;
        let dtype = vb_m.dtype();

        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            cfg.tie_word_embeddings.then(|| {
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;

        let ropes = crate::device_map::per_layer_device(
            &*mapper,
            cfg.num_hidden_layers,
            &normal_loading_metadata.real_device,
            |device| T::rope(cfg, vb.dtype(), device, is_gptx),
        )?;

        let paged_head_dim = T::paged_head_dim(cfg);
        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer<T>> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(paged_head_dim, device, None)
                        .expect("Failed to create PagedAttention"),
                ),
            };
            let comm = mapper.get_comm_for(layer_idx)?;
            let ctx = LayerCtx {
                cfg,
                mapper: &*mapper,
                layer_idx,
                loading_isq: normal_loading_metadata.loading_isq,
                comm: &comm,
            };
            DecoderLayer::new(
                &ctx,
                rotary_emb,
                vb_l.pp(layer_idx),
                paged_attn,
                normal_loading_metadata.real_device.clone(),
            )
        })?;

        let world_size = mapper.get_comm_for(0)?.world_size();
        Ok(Self {
            lm_head,
            embed_tokens,
            dtype,
            norm,
            layers,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            device: normal_loading_metadata.real_device.clone(),
            max_seq_len: cfg.max_position_embeddings,
            cfg: T::model_metadata(
                cfg,
                &attention_mechanism,
                &normal_loading_metadata.real_device,
                world_size,
            ),
            mapper,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let mut xs = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
        let cache = &mut self.cache.normal().0;
        let mask_cache = ctx.mask_cache(cache);
        let attention_mask = CausalMasker.make_causal_mask(
            input_ids,
            &mask_cache,
            xs.dtype(),
            &CausalMaskConfig::default(),
        )?;
        // PagedAttention prompt chunking
        let attention_mask = if ctx.is_first_prompt_chunk() {
            attention_mask
        } else {
            AttentionMask::None
        };
        let attention_mask = DeviceMappedMask::new(attention_mask, &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(&xs, &attention_mask.get(xs.device()), &mut cache[i], ctx, i)?;
        }
        let xs = xs.to_device(&self.device)?;
        let xs = xs.apply(&self.norm)?;
        let xs = ctx.logits(&xs)?;
        ctx.lm_head(&*self.lm_head, &xs)
    }

    fn residual_uvb(&self, with_projections: bool) -> UnVarBuilder {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            layer.add_residual(&uvb_l);
            if with_projections {
                layer.attn.add_projections(&uvb_l.pp("self_attn"));
            }
        }
        uvb
    }
}

impl<T: FamilyAttention> IsqModel for FamilyModel<T> {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.residual_uvb(false).to_safetensors()
    }

    fn residual_tensors_moe_experts_only(&self) -> Option<Vec<(String, Tensor)>> {
        Some(self.residual_uvb(true).to_safetensors())
    }
}

impl<T: FamilyAttention> crate::speculative::SpeculativeTargetMixin for FamilyModel<T> {}

impl<T: FamilyAttention> NormalModel for FamilyModel<T> {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::model::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
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
        T::CUDA_DECODE_GRAPHS
    }
}

impl<T: FamilyAttention> AnyMoeBaseModelMixin for FamilyModel<T> {}
