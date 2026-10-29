//! BLIP-2's Q-Former over learned queries, as Granite 4 Vision's and Granite Speech's projectors use it.
#![allow(clippy::cast_precision_loss)]

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{Result, Tensor};

use crate::layers::{layer_norm, linear_b};
use crate::utils::unvarbuilder::UnVarBuilder;

struct QFormerAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    norm: LayerNorm,
    heads: usize,
    head_dim: usize,
}

impl QFormerAttention {
    fn new(
        hidden: usize,
        context: usize,
        heads: usize,
        eps: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let attention = vb.pp("attention");
        Ok(Self {
            query: linear_b(hidden, hidden, true, attention.pp("query"))?,
            key: linear_b(context, hidden, true, attention.pp("key"))?,
            value: linear_b(context, hidden, true, attention.pp("value"))?,
            dense: linear_b(hidden, hidden, true, vb.pp("output").pp("dense"))?,
            norm: layer_norm(hidden, eps, vb.pp("output").pp("LayerNorm"))?,
            heads,
            head_dim: hidden / heads,
        })
    }

    fn forward(&self, xs: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (batch, len, hidden) = xs.dims3()?;
        let split = |t: Tensor| -> Result<Tensor> {
            let rows = t.dim(1)?;
            t.reshape((batch, rows, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = split(self.query.forward(xs)?)?;
        let k = split(self.key.forward(context)?)?;
        let v = split(self.value.forward(context)?)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let weights = (q.matmul(&k.t()?)? * scale)?;
        let weights = inference_tensor::nn::ops::softmax_last_dim(&weights)?;
        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, len, hidden))?;
        self.norm.forward(&(self.dense.forward(&out)? + xs)?)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let attention = uvb.pp("attention");
        attention.pp("query").add(&self.query);
        attention.pp("key").add(&self.key);
        attention.pp("value").add(&self.value);
        uvb.pp("output").pp("dense").add(&self.dense);
        uvb.pp("output").pp("LayerNorm").add(&self.norm);
        uvb.to_safetensors()
    }
}

struct QFormerLayer {
    attention: QFormerAttention,
    crossattention: QFormerAttention,
    intermediate: Linear,
    output: Linear,
    output_norm: LayerNorm,
}

impl QFormerLayer {
    fn forward(&self, xs: &Tensor, encoder: &Tensor) -> Result<Tensor> {
        let xs = self.attention.forward(xs, xs)?;
        let xs = self.crossattention.forward(&xs, encoder)?;
        let hidden = self.intermediate.forward(&xs)?.gelu_erf()?;
        self.output_norm
            .forward(&(self.output.forward(&hidden)? + xs)?)
    }
}

/// A BLIP-2 Q-Former's sizes.
pub struct QFormerDims {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub layer_norm_eps: f64,
    /// The width of the states the queries cross-attend to.
    pub encoder_hidden: usize,
}

/// transformers' `Blip2QFormerModel` over query embeddings only (no text input): post-norm self-attention, then
/// cross-attention to the encoder states, then a GELU feed-forward.
pub struct QFormer {
    layernorm: LayerNorm,
    layers: Vec<QFormerLayer>,
}

impl QFormer {
    pub fn new(dims: &QFormerDims, vb: ShardedVarBuilder) -> Result<Self> {
        let (hidden, heads, eps) = (dims.hidden, dims.heads, dims.layer_norm_eps);
        let layers_vb = vb.pp("encoder").pp("layer");
        let layers = (0..dims.layers)
            .map(|i| {
                let vb = layers_vb.pp(i);
                Ok(QFormerLayer {
                    attention: QFormerAttention::new(
                        hidden,
                        hidden,
                        heads,
                        eps,
                        vb.pp("attention"),
                    )?,
                    crossattention: QFormerAttention::new(
                        hidden,
                        dims.encoder_hidden,
                        heads,
                        eps,
                        vb.pp("crossattention"),
                    )?,
                    intermediate: linear_b(
                        hidden,
                        dims.intermediate,
                        true,
                        vb.pp("intermediate_query").pp("dense"),
                    )?,
                    output: linear_b(
                        dims.intermediate,
                        hidden,
                        true,
                        vb.pp("output_query").pp("dense"),
                    )?,
                    output_norm: layer_norm(hidden, eps, vb.pp("output_query").pp("LayerNorm"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layernorm: layer_norm(hidden, eps, vb.pp("layernorm"))?,
            layers,
        })
    }

    pub fn forward(&self, queries: &Tensor, encoder: &Tensor) -> Result<Tensor> {
        let mut xs = self.layernorm.forward(queries)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, encoder)?;
        }
        Ok(xs)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("layernorm").add(&self.layernorm);
        for (i, layer) in self.layers.iter().enumerate() {
            let l = uvb.pp("encoder").pp("layer").pp(i);
            l.pp("attention").extend(layer.attention.residual_tensors());
            l.pp("crossattention")
                .extend(layer.crossattention.residual_tensors());
            l.pp("intermediate_query")
                .pp("dense")
                .add(&layer.intermediate);
            l.pp("output_query").pp("dense").add(&layer.output);
            l.pp("output_query").pp("LayerNorm").add(&layer.output_norm);
        }
        uvb.to_safetensors()
    }
}
