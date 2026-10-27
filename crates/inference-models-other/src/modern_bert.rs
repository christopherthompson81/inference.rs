//! ModernBERT: a pre-norm encoder with rotary positions, GeGLU, bias-free norms, and sliding-window layers between
//! every `global_attn_every_n_layers`-th global one.

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Embedding, LayerNorm, LayerNormConfig, Module};
use inference_tensor::{D, Device, Result, Tensor};
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams},
    device_map::DeviceMapper,
    layers::{Activation, dense_embedding, layer_norm, masker::BidirectionalMasker},
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    rope_encoder::{QkvLayout, RopeAttentionShape, RopeSelfAttention, RopeStyle, rotary},
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
};

const TOK_EMBEDDINGS: &str = "embeddings.tok_embeddings.weight";
// `*ForMaskedLM` / `*ForSequenceClassification` checkpoints keep the encoder under `model.`
const BASE_PREFIX: &str = "model";
const SLIDING_LAYER: &str = "sliding_attention";

serde_default_fn!(f64, norm_eps, 1e-5);
serde_default_fn!(usize, global_attn_every_n_layers, 3);
serde_default_fn!(usize, local_attention, 128);
serde_default_fn!(f32, global_rope_theta, 160_000.0);
serde_default_fn!(f32, local_rope_theta, 10_000.0);
serde_default_fn!(Activation, hidden_activation, Activation::Gelu);

#[derive(Debug, Clone, Deserialize)]
struct RopeTheta {
    rope_theta: f32,
}

/// transformers 5 writes the two thetas as `rope_parameters`; older configs as `global_rope_theta` / `local_rope_theta`.
#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    full_attention: RopeTheta,
    sliding_attention: RopeTheta,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "hidden_activation")]
    pub hidden_activation: Activation,
    #[serde(default = "global_attn_every_n_layers")]
    pub global_attn_every_n_layers: usize,
    /// The whole window: a sliding layer attends `local_attention / 2` tokens each way.
    #[serde(default = "local_attention")]
    pub local_attention: usize,
    #[serde(default = "global_rope_theta")]
    pub global_rope_theta: f32,
    #[serde(default = "local_rope_theta")]
    pub local_rope_theta: f32,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

impl Config {
    pub fn is_sliding(&self, layer: usize) -> bool {
        match &self.layer_types {
            Some(types) => types.get(layer).is_some_and(|t| t == SLIDING_LAYER),
            None => !layer.is_multiple_of(self.global_attn_every_n_layers),
        }
    }

    fn rope_thetas(&self) -> (f32, f32) {
        match &self.rope_parameters {
            Some(p) => (p.full_attention.rope_theta, p.sliding_attention.rope_theta),
            None => (self.global_rope_theta, self.local_rope_theta),
        }
    }

    fn norm_config(&self) -> LayerNormConfig {
        LayerNormConfig {
            eps: self.norm_eps,
            remove_mean: true,
            affine: self.norm_bias,
        }
    }
}

struct Layer {
    // layer 0 attends its embeddings unnormalized
    attn_norm: Option<LayerNorm>,
    attn: RopeSelfAttention,
    mlp_norm: LayerNorm,
    wi: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
    act: Activation,
    sliding: bool,
}

impl Layer {
    fn forward(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let normed = match &self.attn_norm {
            Some(norm) => norm.forward(xs)?,
            None => xs.clone(),
        };
        let xs = (xs + self.attn.forward(&normed, mask)?)?;
        let gated = self.wi.forward(&self.mlp_norm.forward(&xs)?)?;
        let half = gated.dim(D::Minus1)? / 2;
        let (input, gate) = (
            gated.narrow(D::Minus1, 0, half)?,
            gated.narrow(D::Minus1, half, half)?,
        );
        xs + self.wo.forward(&(self.act.forward(&input)? * gate)?)?
    }
}

pub struct Model {
    tok_embeddings: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    // distance < window: `local_attention / 2` each way, inclusive
    window: usize,
    prefix: Option<&'static str>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    device: Device,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            inference_tensor::bail!("Expected AttentionImplementation::Eager");
        }
        let prefix = (!vb.contains_tensor(TOK_EMBEDDINGS)).then_some(BASE_PREFIX);
        let vb = match prefix {
            Some(prefix) => vb.pp(prefix),
            None => vb,
        };
        let NormalLoadingMetadata {
            mapper,
            loading_isq,
            real_device,
            ..
        } = normal_loading_metadata;
        let hidden = cfg.hidden_size;
        let head_dim = hidden / cfg.num_attention_heads;
        let (global_theta, local_theta) = cfg.rope_thetas();
        let tables = |device: &Device| -> Result<_> {
            Ok((
                rotary(
                    global_theta,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    vb.dtype(),
                )?,
                rotary(
                    local_theta,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    vb.dtype(),
                )?,
            ))
        };
        let base_tables = tables(&real_device)?;
        let vb_l = vb.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb = vb_l.pp(i);
                let place = |vb: ShardedVarBuilder| mapper.set_device(i, vb, loading_isq);
                let norm = |name| {
                    layer_norm(
                        hidden,
                        cfg.norm_config(),
                        mapper.set_device(i, vb.pp(name), false),
                    )
                };
                let (global, local) = match mapper.device_for(i, false) {
                    Some(device) if !device.same_device(&real_device) => tables(device)?,
                    _ => base_tables.clone(),
                };
                let sliding = cfg.is_sliding(i);
                let vb_attn = vb.pp("attn");
                let vb_mlp = vb.pp("mlp");
                Ok(Layer {
                    attn_norm: (i > 0).then(|| norm("attn_norm")).transpose()?,
                    attn: RopeSelfAttention::new(
                        &RopeAttentionShape {
                            hidden,
                            num_heads: cfg.num_attention_heads,
                            bias: cfg.attention_bias,
                            layout: QkvLayout::Stacked,
                            style: RopeStyle::NeoX,
                        },
                        if sliding { local } else { global },
                        place(vb_attn.pp("Wqkv")),
                        place(vb_attn.pp("Wo")),
                    )?,
                    mlp_norm: norm("mlp_norm")?,
                    wi: inference_quant::linear_b(
                        hidden,
                        2 * cfg.intermediate_size,
                        cfg.mlp_bias,
                        &None,
                        place(vb_mlp.pp("Wi")),
                    )?,
                    wo: inference_quant::linear_b(
                        cfg.intermediate_size,
                        hidden,
                        cfg.mlp_bias,
                        &None,
                        place(vb_mlp.pp("Wo")),
                    )?,
                    act: cfg.hidden_activation,
                    sliding,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let vb_e = mapper.set_nm_device(vb.pp("embeddings"), false);
        Ok(Self {
            tok_embeddings: dense_embedding(
                cfg.vocab_size,
                hidden,
                vb_e.pp("tok_embeddings"),
                &None,
            )?,
            emb_norm: layer_norm(hidden, cfg.norm_config(), vb_e.pp("norm"))?,
            layers,
            final_norm: layer_norm(
                hidden,
                cfg.norm_config(),
                mapper.set_nm_device(vb.pp("final_norm"), false),
            )?,
            window: cfg.local_attention / 2 + 1,
            prefix,
            mapper,
            device: real_device,
        })
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let root = UnVarBuilder::new();
        let prefixed = self.prefix.map(|prefix| root.pp(prefix));
        let uvb = prefixed.as_ref().unwrap_or(&root);
        uvb.pp("embeddings")
            .pp("tok_embeddings")
            .add(&self.tok_embeddings);
        uvb.pp("embeddings").pp("norm").add(&self.emb_norm);
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("layers").pp(i);
            if let Some(norm) = &layer.attn_norm {
                uvb_l.pp("attn_norm").add(norm);
            }
            uvb_l.pp("mlp_norm").add(&layer.mlp_norm);
        }
        uvb.pp("final_norm").add(&self.final_norm);
        root.to_safetensors()
    }
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        _token_type_ids: Option<&Tensor>,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self
            .emb_norm
            .forward(&self.tok_embeddings.forward(input_ids)?)?;
        let (_, seq_len) = input_ids.dims2()?;
        // a prompt inside one window needs no mask on the sliding layers
        let sliding_mask = (seq_len >= self.window)
            .then(|| {
                BidirectionalMasker
                    .make_sliding_mask(input_ids, xs.dtype(), self.window)
                    .map(AttentionMask::Custom)
            })
            .transpose()?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            let mask = match (&sliding_mask, layer.sliding) {
                (Some(AttentionMask::Custom(mask)), true) => {
                    AttentionMask::Custom(mask.to_device(xs.device())?)
                }
                _ => AttentionMask::None,
            };
            xs = layer.forward(&xs, &mask)?;
        }
        self.final_norm.forward(&xs.to_device(&self.device)?)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}
