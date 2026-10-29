//! Granite 4 Vision's Window Q-Former projector: a BLIP-2 Q-Former whose queries each read one window of the tile.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{Result, Tensor};

use super::config::Config;
use crate::blip2_qformer::{QFormer, QFormerDims};
use crate::layers::{layer_norm, linear_b};
use crate::utils::unvarbuilder::UnVarBuilder;

// the projector's input LayerNorm, separate from the Q-Former's
const PROJECTOR_NORM_EPS: f64 = 1e-6;
// spatial projectors read one corner of every 2x2 block
const SPATIAL_BLOCK: usize = 2;

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
            qformer: QFormer::new(
                &QFormerDims {
                    hidden: config.qformer_config.hidden(vision),
                    heads: config.qformer_config.heads(vision),
                    layers: config.qformer_config.num_hidden_layers,
                    intermediate: config.qformer_config.intermediate_size,
                    layer_norm_eps: config.qformer_config.layer_norm_eps,
                    encoder_hidden: config.qformer_config.encoder_hidden(vision),
                },
                vb.pp("qformer"),
            )?,
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
