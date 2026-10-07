use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::Linear;
use inference_tensor::{Result, Tensor};

use crate::layers;

use super::config::Idefics3Config;

pub struct Idefics3SimpleMLP {
    pub proj: Linear,
}

impl Idefics3SimpleMLP {
    pub fn new(cfg: &Idefics3Config, vb: ShardedVarBuilder) -> Result<Self> {
        let in_dim = cfg.vision_config.hidden_size * cfg.scale_factor.pow(2);
        let out_dim = cfg.text_config.hidden_size;
        Ok(Self {
            proj: layers::linear_no_bias(in_dim, out_dim, vb.pp("proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.proj)
    }
}

pub struct Idefics3Connector {
    scale_factor: usize,
    pub modality_projection: Idefics3SimpleMLP,
}

impl Idefics3Connector {
    pub fn new(cfg: &Idefics3Config, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            scale_factor: cfg.scale_factor,
            modality_projection: Idefics3SimpleMLP::new(cfg, vb.pp("modality_projection"))?,
        })
    }

    pub fn pixel_shuffle(&self, x: &Tensor, scale_factor: usize) -> Result<Tensor> {
        let (bs, seq, embed_dim) = x.dims3()?;
        let height = (seq as f32).sqrt() as usize;
        let width = height;
        let mut x = x.reshape((bs, height, width, embed_dim))?;
        x = x.reshape((bs, height, width / scale_factor, embed_dim * scale_factor))?;
        x = x.permute((0, 2, 1, 3))?;
        x = x.reshape((
            bs,
            width / scale_factor,
            height / scale_factor,
            embed_dim * scale_factor.pow(2),
        ))?;
        x = x.permute((0, 2, 1, 3))?;
        x.reshape((
            bs,
            (seq as f32 / scale_factor.pow(2) as f32) as usize,
            embed_dim * scale_factor.pow(2),
        ))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let image_hidden_states = self.pixel_shuffle(x, self.scale_factor)?;
        self.modality_projection.forward(&image_hidden_states)
    }
}
