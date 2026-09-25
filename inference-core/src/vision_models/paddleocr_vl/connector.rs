//! `mlp_AR` connector (transformers `Projector`, `vision_return_embed_list=True` branch).

use crate::layers::{layer_norm, linear};
use crate::utils::unvarbuilder::UnVarBuilder;
use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module};
use inference_quant::ShardedVarBuilder;

pub struct Connector {
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
    merge_size: usize,
    vision_hidden: usize,
}

impl Connector {
    pub fn load(
        vb: ShardedVarBuilder,
        vision_hidden: usize,
        merge_size: usize,
        text_hidden: usize,
    ) -> Result<Self> {
        let merged = vision_hidden * merge_size * merge_size;
        Ok(Self {
            // eps 1e-5, deliberately different from the vision tower's 1e-6.
            pre_norm: layer_norm(vision_hidden, 1e-5, vb.pp("pre_norm"))?,
            linear_1: linear(merged, merged, vb.pp("linear_1"))?,
            linear_2: linear(merged, text_hidden, vb.pp("linear_2"))?,
            merge_size,
            vision_hidden,
        })
    }

    pub fn forward(&self, x: &Tensor, t: usize, h: usize, w: usize) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?; // per-patch, before the merge
        let x = self.merge(&x, t, h, w)?;
        let x = self.linear_1.forward(&x)?.gelu_erf()?; // exact erf GELU, unlike the tower's tanh GELU
        self.linear_2.forward(&x)
    }

    // einops `(t h p1 w p2) d -> (t h w) (p1 p2 d)`; each merged row is [TL|TR|BL|BR].
    fn merge(&self, x: &Tensor, t: usize, h: usize, w: usize) -> Result<Tensor> {
        let m = self.merge_size;
        let d = self.vision_hidden;
        x.reshape((t, h / m, m, w / m, m, d))? // (t, hb, p1, wb, p2, d)
            .permute((0, 1, 3, 2, 4, 5))? // (t, hb, wb, p1, p2, d)
            .contiguous()?
            .reshape((t * (h / m) * (w / m), m * m * d))
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_c = uvb.pp("mlp_AR");
        uvb_c.pp("pre_norm").add(&self.pre_norm);
        uvb_c.pp("linear_1").add(&self.linear_1);
        uvb_c.pp("linear_2").add(&self.linear_2);
        uvb.to_safetensors()
    }
}
