//! NeoBERT: a pre-norm RMSNorm encoder with Llama-1 (interleaved) rotary positions, a per-head fused QKV and SwiGLU.

use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::{Embedding, Module};
use inference_tensor::{D, Device, Result, Tensor};
use serde::Deserialize;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams},
    device_map::DeviceMapper,
    layers::{RmsNorm, dense_embedding},
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    rope_encoder::{QkvLayout, RopeAttentionShape, RopeSelfAttention, RopeStyle, rotary},
    serde_default_fn,
    utils::unvarbuilder::UnVarBuilder,
};

const TOKEN_EMBEDDINGS: &str = "encoder.weight";
// `NeoBERTLMHead` checkpoints keep the encoder under `model.`
const BASE_PREFIX: &str = "model";
const ROPE_THETA: f32 = 10_000.0;
// NeoBERT sizes its SwiGLU at 2/3 of `intermediate_size`, rounded up to this multiple
const FFN_MULTIPLE: usize = 8;

serde_default_fn!(f64, norm_eps, 1e-6);
serde_default_fn!(usize, max_length, 1024);

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "max_length")]
    pub max_length: usize,
}

impl Config {
    pub fn ffn_size(&self) -> usize {
        (2 * self.intermediate_size / 3).div_ceil(FFN_MULTIPLE) * FFN_MULTIPLE
    }
}

struct Layer {
    attention_norm: RmsNorm,
    attn: RopeSelfAttention,
    ffn_norm: RmsNorm,
    w12: Arc<dyn QuantMethod>,
    w3: Arc<dyn QuantMethod>,
}

impl Layer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs
            + self
                .attn
                .forward(&self.attention_norm.forward(xs)?, &AttentionMask::None)?)?;
        let gated = self.w12.forward(&self.ffn_norm.forward(&xs)?)?;
        let half = gated.dim(D::Minus1)? / 2;
        let (x1, x2) = (
            gated.narrow(D::Minus1, 0, half)?,
            gated.narrow(D::Minus1, half, half)?,
        );
        xs + self.w3.forward(&(x1.silu()? * x2)?)?
    }
}

pub struct Model {
    encoder: Embedding,
    layers: Vec<Layer>,
    layer_norm: RmsNorm,
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
        let prefix = (!vb.contains_tensor(TOKEN_EMBEDDINGS)).then_some(BASE_PREFIX);
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
        let (hidden, eps) = (cfg.hidden_size, cfg.norm_eps);
        let head_dim = hidden / cfg.num_attention_heads;
        let base_rotary = rotary(
            ROPE_THETA,
            head_dim,
            cfg.max_length,
            &real_device,
            vb.dtype(),
        )?;
        let shape = RopeAttentionShape {
            hidden,
            num_heads: cfg.num_attention_heads,
            bias: false,
            layout: QkvLayout::PerHead,
            style: RopeStyle::Interleaved,
        };
        let vb_l = vb.pp("transformer_encoder");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb = vb_l.pp(i);
                let place = |vb: ShardedVarBuilder| mapper.set_device(i, vb, loading_isq);
                let norm =
                    |name| RmsNorm::new(hidden, eps, mapper.set_device(i, vb.pp(name), false));
                let layer_rotary = match mapper.device_for(i, false) {
                    Some(device) if !device.same_device(&real_device) => {
                        rotary(ROPE_THETA, head_dim, cfg.max_length, device, vb.dtype())?
                    }
                    _ => base_rotary.clone(),
                };
                let vb_ffn = vb.pp("ffn");
                Ok(Layer {
                    attention_norm: norm("attention_norm")?,
                    attn: RopeSelfAttention::new(
                        &shape,
                        layer_rotary,
                        place(vb.pp("qkv")),
                        place(vb.pp("wo")),
                    )?,
                    ffn_norm: norm("ffn_norm")?,
                    w12: inference_quant::linear_b(
                        hidden,
                        2 * cfg.ffn_size(),
                        false,
                        &None,
                        place(vb_ffn.pp("w12")),
                    )?,
                    w3: inference_quant::linear_b(
                        cfg.ffn_size(),
                        hidden,
                        false,
                        &None,
                        place(vb_ffn.pp("w3")),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            encoder: dense_embedding(
                cfg.vocab_size,
                hidden,
                mapper.set_nm_device(vb.pp("encoder"), false),
                &None,
            )?,
            layers,
            layer_norm: RmsNorm::new(
                hidden,
                eps,
                mapper.set_nm_device(vb.pp("layer_norm"), false),
            )?,
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
        uvb.pp("encoder").add(&self.encoder);
        uvb.pp("layer_norm").add(&self.layer_norm);
        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("transformer_encoder").pp(i);
            uvb_l.pp("attention_norm").add(&layer.attention_norm);
            uvb_l.pp("ffn_norm").add(&layer.ffn_norm);
        }
        root.to_safetensors()
    }
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        _extras: &crate::embedding::EmbeddingExtras,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = self.encoder.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&self.mapper.map(xs, i)?)?;
        }
        self.layer_norm.forward(&xs.to_device(&self.device)?)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}
