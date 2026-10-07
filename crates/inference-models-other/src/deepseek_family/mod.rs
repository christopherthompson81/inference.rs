//! The decoder DeepSeek-V2, DeepSeek-V3, GLM4-MoE and GLM4-MoE-Lite share: MoE block, layers and model, over an attention.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod mla;

use std::{collections::HashMap, sync::Arc};

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::{DType, Device, DeviceLocation, Result, Tensor};

pub use mla::{MlaAttention, MlaConfig, MlaKvLayout, mla_softmax_scale};

use crate::model::NormalLoadingMetadata;
use crate::{
    decoder::{
        CausalLm, DecoderStack, LayerAttention, LayerBuilder, LayerFfn, LayerLoad, MlpKind,
        NormKind, NormNames, StackShape,
    },
    device_map::DeviceMapper,
    layers::{Activation, Mlp},
    moe::{GroupedRouter, GroupedRouterConfig, MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    utils::unvarbuilder::UnVarBuilder,
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

/// How a family's attention is built; it runs as the shared decoder's [`LayerAttention`].
pub trait FamilyAttention: LayerAttention + Sized {
    type Config: Send + Sync;
    type Rope: Send + Sync;

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

pub(crate) struct MoeGate {
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

pub(crate) fn add_moe_gate_residual_tensors(
    uvb: &UnVarBuilder,
    weight: &Tensor,
    correction_bias: Option<&Tensor>,
) {
    uvb.add_tensor("weight", weight.clone());
    if let Some(bias) = correction_bias {
        uvb.add_tensor("e_score_correction_bias", bias.clone());
    }
}

pub struct Moe {
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

/// A family layer's feed-forward: a routed MoE block, or a dense MLP on the first dense layers.
pub enum MoeOrMlp {
    Moe(Box<Moe>),
    Mlp(Mlp),
}

impl LayerFfn for MoeOrMlp {
    const MOE_EXPERTS_ONLY_ISQ: bool = true;

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }

    fn add_residual(&self, uvb: &UnVarBuilder) {
        if let Self::Moe(moe) = self {
            add_moe_gate_residual_tensors(
                &uvb.pp("gate"),
                &moe.gate.weight,
                moe.gate.router.e_score_correction_bias(),
            );
        }
    }
}

/// The family attention `T` and its MoE or MLP, layer by layer.
struct FamilyLayers<'a, T: FamilyAttention> {
    cfg: &'a FamilyConfig<T::Config>,
    ropes: HashMap<DeviceLocation, Arc<T::Rope>>,
    real_device: Device,
}

impl<T: FamilyAttention> LayerBuilder for FamilyLayers<'_, T> {
    type Attention = T;
    type Ffn = MoeOrMlp;

    fn build(&self, load: &LayerLoad<'_>, vb: ShardedVarBuilder) -> Result<(T, MoeOrMlp)> {
        let cfg = self.cfg;
        let ctx = LayerCtx {
            cfg,
            mapper: load.mapper,
            layer_idx: load.layer_idx,
            loading_isq: load.loading_isq,
            comm: load.comm,
        };
        let rotary_emb = self
            .ropes
            .get(&load.device.location())
            .expect("No RoPE for device location!")
            .clone();
        let paged_attn = match load.attention {
            AttentionImplementation::Eager => None,
            AttentionImplementation::PagedAttention => Some(PagedAttention::new(
                T::paged_head_dim(cfg),
                load.device,
                None,
            )?),
        };
        let attn = T::new(&ctx, rotary_emb, vb.pp("self_attn"), paged_attn)?;
        let ffn = if cfg.is_moe_layer(load.layer_idx) {
            MoeOrMlp::Moe(Box::new(Moe::new(
                &ctx,
                vb.pp("mlp"),
                self.real_device.clone(),
            )?))
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                load.mapper
                    .set_device(load.layer_idx, vb.pp("mlp"), load.loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                cfg.hidden_act,
                load.comm,
            )?)
        };
        Ok((attn, ffn))
    }
}

pub type FamilyModel<T> = CausalLm<T, MoeOrMlp>;

/// A family model over the attention `T`: the shared decoder with a MoE feed-forward past the dense layers.
pub fn new_family_model<T: FamilyAttention>(
    cfg: &FamilyConfig<T::Config>,
    vb: ShardedVarBuilder,
    is_gptx: bool,
    normal_loading_metadata: NormalLoadingMetadata,
    attention_mechanism: AttentionImplementation,
) -> Result<FamilyModel<T>> {
    let layer_windows = vec![None; cfg.num_hidden_layers];
    let shape = StackShape {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        rms_norm_eps: cfg.rms_norm_eps,
        layer_windows: &layer_windows,
        tie_word_embeddings: cfg.tie_word_embeddings,
        quantization_config: &cfg.quantization_config,
        norm: NormKind::Rms,
        norm_names: NormNames::PRE,
        embed_scale: None,
        final_logit_softcap: None,
        mlp: MlpKind::Gated,
        lm_head_bias: false,
        unquantized_lm_head: false,
    };
    let loading_isq = normal_loading_metadata.loading_isq;
    let dtype = vb.dtype();
    let stack = DecoderStack::new_with(
        shape,
        vb.pp("model"),
        Some(vb.pp("lm_head")),
        normal_loading_metadata,
        &attention_mechanism,
        |mapper, real_device| {
            let ropes = crate::device_map::per_layer_device(
                mapper,
                cfg.num_hidden_layers,
                real_device,
                |device| T::rope(cfg, dtype, device, is_gptx),
            )?;
            Ok(FamilyLayers::<T> {
                cfg,
                ropes,
                real_device: real_device.clone(),
            })
        },
    )?;
    let world_size = stack.mapper.get_comm_for(0)?.world_size();
    let metadata = T::model_metadata(cfg, &attention_mechanism, &stack.device, world_size);
    CausalLm::with_stack(
        stack,
        shape,
        vb.pp("lm_head"),
        loading_isq,
        cfg.max_position_embeddings,
        metadata,
    )
}

/// A `moe_layer_freq` of 0 would make no layer past the first a MoE layer and divide by zero in the sizing.
pub(crate) fn nonzero_moe_layer_freq<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<usize, D::Error> {
    let freq = <usize as serde::Deserialize>::deserialize(deserializer)?;
    if freq == 0 {
        return Err(serde::de::Error::custom(
            "moe_layer_freq must be at least 1",
        ));
    }
    Ok(freq)
}
