//! Granite 4 Vision's Window Q-Former projector: a BLIP-2 Q-Former whose queries each read one window of the tile.
#![allow(clippy::cast_precision_loss)]

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{Result, Tensor};

use super::config::Config;
use crate::layers::{layer_norm, linear_b};
use crate::utils::unvarbuilder::UnVarBuilder;

// the projector's input LayerNorm, separate from the Q-Former's
const PROJECTOR_NORM_EPS: f64 = 1e-6;
// spatial projectors read one corner of every 2x2 block
const SPATIAL_BLOCK: usize = 2;

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

struct QFormer {
    layernorm: LayerNorm,
    layers: Vec<QFormerLayer>,
}

impl QFormer {
    fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let vision = config.vision_config.hidden_size;
        let q = &config.qformer_config;
        let (hidden, heads, eps) = (q.hidden(vision), q.heads(vision), q.layer_norm_eps);
        let layers_vb = vb.pp("encoder").pp("layer");
        let layers = (0..q.num_hidden_layers)
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
                        q.encoder_hidden(vision),
                        heads,
                        eps,
                        vb.pp("crossattention"),
                    )?,
                    intermediate: linear_b(
                        hidden,
                        q.intermediate_size,
                        true,
                        vb.pp("intermediate_query").pp("dense"),
                    )?,
                    output: linear_b(
                        q.intermediate_size,
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

    fn forward(&self, queries: &Tensor, encoder: &Tensor) -> Result<Tensor> {
        let mut xs = self.layernorm.forward(queries)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, encoder)?;
        }
        Ok(xs)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
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

/// transformers' `Granite4VisionWindowQFormerDownsampler`: `(tiles, patches, vision)` to `(tiles, features, text)`.
pub struct WindowQFormerDownsampler {
    norm: LayerNorm,
    query: Tensor,
    image_positions: Tensor,
    qformer: QFormer,
    out_linear: Linear,
    spatial_offset: Option<usize>,
    image_side: usize,
    query_side: usize,
    window_side: usize,
}

/// `(B, side*side, C)` raster to `(B*windows^2, window^2, C)`, windows in raster order.
fn windowed(xs: &Tensor, side: usize, window: usize) -> Result<Tensor> {
    let (batch, _, channels) = xs.dims3()?;
    let windows = side / window;
    xs.reshape((batch, windows, window, windows, window, channels))?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((batch * windows * windows, window * window, channels))
}

fn unwindowed(xs: &Tensor, windows: usize, window: usize) -> Result<Tensor> {
    let (batch_windows, _, channels) = xs.dims3()?;
    let batch = batch_windows / (windows * windows);
    xs.reshape((batch, windows, windows, window, window, channels))?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((batch, windows * window * windows * window, channels))
}

impl WindowQFormerDownsampler {
    pub fn new(
        config: &Config,
        spatial_offset: Option<usize>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let vision = config.vision_config.hidden_size;
        let (query_side, window_side) = config.downsample_sides()?;
        let image_side = config.patches_per_side();
        if !image_side.is_multiple_of(window_side) {
            inference_tensor::bail!(
                "Granite 4 Vision's {image_side}-patch tile does not split into {window_side}-patch windows"
            );
        }
        Ok(Self {
            norm: layer_norm(vision, PROJECTOR_NORM_EPS, vb.pp("norm"))?,
            query: vb.get((1, query_side * query_side, vision), "query")?,
            image_positions: vb.get((1, window_side * window_side, vision), "image_positions")?,
            qformer: QFormer::new(config, vb.pp("qformer"))?,
            out_linear: linear_b(
                config.qformer_config.hidden(vision),
                config.text_config.hidden_size,
                true,
                vb.pp("out_linear"),
            )?,
            spatial_offset,
            image_side,
            query_side,
            window_side,
        })
    }

    fn downsample(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, _, channels) = xs.dims3()?;
        let side = self.image_side;
        match self.spatial_offset {
            Some(offset) => {
                let (row, col) = (offset / SPATIAL_BLOCK, offset % SPATIAL_BLOCK);
                let half = side / SPATIAL_BLOCK;
                xs.reshape((batch, half, SPATIAL_BLOCK, half, SPATIAL_BLOCK, channels))?
                    .narrow(2, row, 1)?
                    .narrow(4, col, 1)?
                    .reshape((batch, half * half, channels))
            }
            None => {
                // area interpolation to an integer fraction of the side is an average pool
                let out_side = side * self.query_side / self.window_side;
                let kernel = side / out_side;
                xs.reshape((batch, side, side, channels))?
                    .permute((0, 3, 1, 2))?
                    .avg_pool2d(kernel)?
                    .permute((0, 2, 3, 1))?
                    .reshape((batch, out_side * out_side, channels))
            }
        }
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let windows = self.image_side / self.window_side;
        let features = self.norm.forward(features)?;
        let encoder = windowed(&features, self.image_side, self.window_side)?
            .broadcast_add(&self.image_positions)?;
        let downsampled = self.downsample(&features)?;
        let queries = windowed(&downsampled, windows * self.query_side, self.query_side)?
            .broadcast_add(&self.query)?;
        let out = self.qformer.forward(&queries, &encoder)?;
        self.out_linear
            .forward(&unwindowed(&out, windows, self.query_side)?)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("norm").add(&self.norm);
        uvb.add_tensor("query", self.query.clone());
        uvb.add_tensor("image_positions", self.image_positions.clone());
        uvb.pp("qformer").extend(self.qformer.residual_tensors());
        uvb.pp("out_linear").add(&self.out_linear);
        uvb.to_safetensors()
    }
}
