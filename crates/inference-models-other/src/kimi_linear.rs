//! Kimi-Linear (`KimiLinearForCausalLM`): mostly Kimi Delta Attention layers (a gated delta rule with a decay per key
//! channel) with one MLA layer in four, whose rope part of the key carries no positions, and DeepSeek-V3's MoE (a dense
//! first layer). The KDA layers keep their state in the hybrid cache's recurrent pool, the MLA layers a paged latent
//! cache.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::{DType, Device, Module, Result, Tensor};
use serde::Serialize;

use crate::attention::AttentionMask;
use crate::decoder::{LayerAttention, LayerFfn};
use crate::deepseek_family::{
    FamilyConfig, LayerCtx, MlaAttention, MlaConfig, Moe, MoeOrMlp, MoeSpec, SharedExpert,
    mla_softmax_scale,
};
use crate::device_map::DeviceMappedMask;
use crate::gdn::{GdnLayerCache, KdaConfig, KimiDeltaAttention};
use crate::kv_cache::{
    EitherCache, HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType,
    RecurrentLayerConfig, RecurrentStateSpec,
};
use crate::layers::masker::{CausalMaskConfig, PastKvLenCache};
use crate::layers::{Activation, CausalMasker, Mlp, RmsNorm, embedding_with_legacy_tied_uqff};
use crate::model::{
    ForwardMaskCache, IsqModel, ModelForwardContext, NormalLoadingMetadata, NormalModel,
};
use crate::moe::{GroupedRouterConfig, RouterMethod, RouterRenorm, RouterScoring};
use crate::paged_attention::{
    AttentionImplementation, HybridPagedKvCacheConfig, ModelConfigLike, ModelConfigMetadata,
    PagedAttention,
};
use crate::serde_default_fn;
use crate::utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder};

// the MoE layers' module, where the dense first layer has `mlp`
const MOE: &str = "block_sparse_moe";
const MLP: &str = "mlp";
// KDA's recurrence advances its state in f32
const STATE_DTYPE: DType = DType::F32;

serde_default_fn!(f64, default_rms_norm_eps, 1e-5);
serde_default_fn!(usize, default_one, 1);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(usize, default_max_positions, 1 << 20);

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct LinearAttnConfig {
    /// One-based, as the checkpoint's config numbers them.
    pub full_attn_layers: Vec<usize>,
    pub head_dim: usize,
    pub num_heads: usize,
    pub short_conv_kernel_size: usize,
}

#[derive(Debug, Clone, inference_json::FromJson, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    #[serde(default)]
    pub mla_use_nope: bool,
    #[serde(default)]
    pub rope_theta: f32,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub num_shared_experts: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_one")]
    pub moe_layer_freq: usize,
    #[serde(default = "default_true")]
    pub moe_renormalize: bool,
    pub routed_scaling_factor: f64,
    #[serde(default = "default_one")]
    pub num_expert_group: usize,
    #[serde(default = "default_one")]
    pub topk_group: usize,
    #[serde(default = "default_max_positions")]
    pub model_max_length: usize,
    pub linear_attn_config: LinearAttnConfig,
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn is_full_attention(&self, layer: usize) -> bool {
        self.linear_attn_config
            .full_attn_layers
            .contains(&(layer + 1))
    }

    pub fn kda(&self) -> KdaConfig {
        KdaConfig {
            hidden_size: self.hidden_size,
            heads: self.linear_attn_config.num_heads,
            head_dim: self.linear_attn_config.head_dim,
            conv_kernel: self.linear_attn_config.short_conv_kernel_size,
            norm_eps: self.rms_norm_eps,
            quantization_config: self.quantization_config.clone(),
        }
    }

    fn kda_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|layer| !self.is_full_attention(*layer))
            .count()
    }

    /// A sequence's recurrent state: per KDA layer its conv window over q|k|v and its heads' matrices.
    pub fn recurrent_state_bytes(&self, dtype: DType) -> usize {
        let kda = self.kda();
        let matrices = kda.heads * kda.head_dim * kda.head_dim * STATE_DTYPE.size_in_bytes();
        self.kda_layers() * (kda.conv_dim() * kda.conv_kernel * dtype.size_in_bytes() + matrices)
    }

    pub fn family(&self) -> FamilyConfig<MlaConfig> {
        let q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        FamilyConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            max_position_embeddings: self.model_max_length,
            rms_norm_eps: self.rms_norm_eps,
            tie_word_embeddings: self.tie_word_embeddings,
            hidden_act: Activation::Silu,
            quantization_config: self.quantization_config.clone(),
            moe: Some(MoeSpec {
                n_routed_experts: self.num_experts,
                num_experts_per_tok: Some(self.num_experts_per_token),
                moe_intermediate_size: self.moe_intermediate_size,
                first_k_dense_replace: self.first_k_dense_replace,
                moe_layer_freq: Some(self.moe_layer_freq),
                shared_expert: (self.num_shared_experts > 0).then_some(SharedExpert::Sharded(
                    self.moe_intermediate_size * self.num_shared_experts,
                )),
                router: GroupedRouterConfig {
                    scoring: RouterScoring::Sigmoid,
                    method: RouterMethod::NoAuxTc,
                    n_group: self.num_expert_group,
                    topk_group: self.topk_group,
                    routed_scaling_factor: self.routed_scaling_factor,
                    renorm: RouterRenorm::TopkProb {
                        norm_topk_prob: self.moe_renormalize,
                    },
                },
                expert_proj_names: crate::moe::ExpertProjNames::MIXTRAL,
            }),
            attn: MlaConfig {
                q_lora_rank: self.q_lora_rank,
                kv_lora_rank: self.kv_lora_rank,
                qk_nope_head_dim: self.qk_nope_head_dim,
                qk_rope_head_dim: self.qk_rope_head_dim,
                v_head_dim: self.v_head_dim,
                attention_bias: false,
                softmax_scale: mla_softmax_scale(q_head_dim, None),
                rope_theta: self.rope_theta,
                rope_scaling: None,
                label: "Kimi-Linear",
            },
        }
    }

    fn is_moe_layer(&self, layer: usize) -> bool {
        layer >= self.first_k_dense_replace && layer.is_multiple_of(self.moe_layer_freq)
    }
}

enum Mixer {
    Kda(KimiDeltaAttention),
    Mla(MlaAttention),
}

struct Layer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    mixer: Mixer,
    ffn: MoeOrMlp,
    ffn_name: &'static str,
}

pub struct Model {
    embed_tokens: Arc<dyn QuantMethod>,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    cache: EitherCache,
    mapper: Box<dyn crate::device_map::DeviceMapper + Send + Sync>,
    device: Device,
    dtype: DType,
    cfg: ModelConfigMetadata,
    paged_layers: Vec<bool>,
    recurrent_state_bytes: usize,
    max_seq_len: usize,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if cfg.q_lora_rank.is_none() && !cfg.mla_use_nope {
            inference_tensor::bail!("Kimi-Linear's MLA carries no positions (`mla_use_nope`)");
        }
        let family = cfg.family();
        let vb_m = vb.pp("model");
        let mapper = normal_loading_metadata.mapper;
        let loading_isq = normal_loading_metadata.loading_isq;
        let real_device = normal_loading_metadata.real_device.clone();
        let dtype = vb.dtype();
        let embed_tokens = embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            cfg.tie_word_embeddings
                .then(|| mapper.set_nm_device(vb.pp("lm_head"), false)),
            &cfg.quantization_config,
        )?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), loading_isq),
            )?
        };
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let kda = cfg.kda();
        let vb_l = vb_m.pp("layers");
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let vb = vb_l.pp(layer_idx);
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&real_device)
                .clone();
            let comm = mapper.get_comm_for(layer_idx)?;
            let ctx = LayerCtx {
                cfg: &family,
                mapper: &*mapper,
                layer_idx,
                loading_isq,
                comm: &comm,
            };
            let norm = |name: &str| {
                RmsNorm::new(
                    cfg.hidden_size,
                    cfg.rms_norm_eps,
                    mapper.set_device(layer_idx, vb.pp(name), false),
                )
            };
            let mixer = if cfg.is_full_attention(layer_idx) {
                let paged_attn = match attention_mechanism {
                    AttentionImplementation::Eager => None,
                    AttentionImplementation::PagedAttention => {
                        Some(PagedAttention::new(cfg.v_head_dim, &device, None)?)
                    }
                };
                Mixer::Mla(MlaAttention::load(
                    &ctx,
                    None,
                    vb.pp("self_attn"),
                    paged_attn,
                )?)
            } else {
                Mixer::Kda(KimiDeltaAttention::load(
                    &kda,
                    mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
                )?)
            };
            let (ffn, ffn_name) = if cfg.is_moe_layer(layer_idx) {
                (
                    MoeOrMlp::Moe(Box::new(Moe::new(&ctx, vb.pp(MOE), real_device.clone())?)),
                    MOE,
                )
            } else {
                (
                    MoeOrMlp::Mlp(Mlp::new(
                        mapper.set_device(layer_idx, vb.pp(MLP), loading_isq),
                        cfg.hidden_size,
                        cfg.intermediate_size,
                        &cfg.quantization_config,
                        Activation::Silu,
                        &comm,
                    )?),
                    MLP,
                )
            };
            Ok(Layer {
                input_layernorm: norm("input_layernorm")?,
                post_attention_layernorm: norm("post_attention_layernorm")?,
                mixer,
                ffn,
                ffn_name,
            })
        })?;
        let paged_layers = (0..cfg.num_hidden_layers)
            .map(|layer| cfg.is_full_attention(layer))
            .collect::<Vec<_>>();
        let layer_devices = (0..cfg.num_hidden_layers)
            .map(|layer| {
                mapper
                    .device_for(layer, false)
                    .unwrap_or(&real_device)
                    .clone()
            })
            .collect::<Vec<_>>();
        let cache = HybridCache::new(
            HybridCacheConfig {
                layer_types: paged_layers
                    .iter()
                    .map(|&full| {
                        if full {
                            HybridLayerType::Attention
                        } else {
                            HybridLayerType::Recurrent
                        }
                    })
                    .collect(),
                max_seq_len: cfg.model_max_length,
                recurrent: RecurrentLayerConfig {
                    conv_dim: kda.conv_dim(),
                    conv_width: kda.conv_kernel,
                    state: RecurrentStateSpec::Opaque {
                        dims: vec![kda.heads, kda.head_dim, kda.head_dim],
                    },
                    recurrent_dtype: Some(STATE_DTYPE),
                },
                auxiliary: Vec::new(),
            },
            dtype,
            &layer_devices,
        )?;
        let world_size = mapper.get_comm_for(0)?.world_size();
        let metadata =
            MlaAttention::metadata(&family, &attention_mechanism, &real_device, world_size);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(cache))),
            mapper,
            device: real_device,
            dtype,
            cfg: metadata,
            paged_layers,
            recurrent_state_bytes: cfg.recurrent_state_bytes(dtype),
            max_seq_len: cfg.model_max_length,
        })
    }

    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        let mut x = self.embed_tokens.embedding_forward(input_ids, self.dtype)?;
        let metadata = ctx.recurrent_metadata().cloned().ok_or_else(|| {
            inference_tensor::Error::msg("Kimi-Linear needs the hybrid cache's recurrent metadata")
        })?;
        let mut cache = self.cache.hybrid();
        // later chunks attend through the paged cache, so only the first needs a mask
        let mask = if ctx.is_first_prompt_chunk() {
            if ctx.is_paged() {
                CausalMasker.make_causal_mask(
                    input_ids,
                    &ForwardMaskCache::Paged(ctx.seqlen_offsets()),
                    x.dtype(),
                    &CausalMaskConfig::default(),
                )?
            } else {
                CausalMasker.make_causal_mask(
                    input_ids,
                    &*cache as &dyn PastKvLenCache,
                    x.dtype(),
                    &CausalMaskConfig::default(),
                )?
            }
        } else {
            AttentionMask::None
        };
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = self.mapper.map(x, layer_idx)?;
            let xn = layer.input_layernorm.forward(&x)?;
            let out = match &layer.mixer {
                Mixer::Mla(attn) => {
                    let Some(HybridLayerCache::Attention(kv_cache)) = cache.get_mut(layer_idx)
                    else {
                        inference_tensor::bail!("Kimi-Linear layer {layer_idx} has no KV cache");
                    };
                    attn.forward(
                        &xn,
                        &mask.get(xn.device()),
                        Some(kv_cache),
                        ctx,
                        layer_idx,
                        None,
                    )?
                }
                Mixer::Kda(kda) => {
                    let indices = cache.state_indices_for_layer(layer_idx)?.ok_or_else(|| {
                        inference_tensor::Error::msg(format!(
                            "Kimi-Linear layer {layer_idx} has no recurrent state indices"
                        ))
                    })?;
                    let Some(HybridLayerCache::Recurrent(pool)) = cache.get_mut(layer_idx) else {
                        inference_tensor::bail!("Kimi-Linear layer {layer_idx} is not recurrent");
                    };
                    let mut state = GdnLayerCache::checkout(pool, &indices)?;
                    let out = kda.forward(&xn, &mut state, metadata.batch_kind())?;
                    state.commit(pool, &indices, metadata.state_indices_host())?;
                    out
                }
            };
            let x_attn = (x + out)?;
            let ffn = layer
                .ffn
                .forward(&layer.post_attention_layernorm.forward(&x_attn)?)?;
            x = (x_attn + ffn)?;
        }
        let x = self.mapper.cast_nm_device(&x, false)?;
        let x = self.norm.forward(&x)?;
        let x = ctx.logits(&x)?;
        ctx.lm_head(&*self.lm_head, &x)
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            let attn = uvb_l.pp("self_attn");
            match &layer.mixer {
                Mixer::Mla(mla) => mla.add_residual(&attn),
                Mixer::Kda(kda) => kda.add_residual(&attn),
            }
            layer.ffn.add_residual(&uvb_l.pp(layer.ffn_name));
        }
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl NormalModel for Model {
    fn forward(&self, input_ids: &Tensor, ctx: &mut ModelForwardContext<'_>) -> Result<Tensor> {
        self.forward(input_ids, ctx)
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }

    // the MLA layers page their latent cache, the KDA layers keep theirs in the recurrent pool
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(
            HybridPagedKvCacheConfig::new(self.cfg.clone(), self.paged_layers.clone())
                .with_recurrent_state_bytes(self.recurrent_state_bytes),
        )
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for Model {}
