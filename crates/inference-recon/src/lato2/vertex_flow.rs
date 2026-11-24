//! V-Flow (`VertexSLatFlowModel`): a sparse DiT over the scaffold's 64^3 cells that denoises 32-channel vertex
//! latents under DINOv2 tokens. RoPE on the cells' coordinates rotates q and k before their RMS norms, and a second
//! timestep embedder adds the requested vertex count (`density`) to the timestep's.

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use super::structure::schedule;
use crate::trellis2::dit::{BLOCK_EPS, Block, CrossKv, DitSpec, Rope, TimestepMlp};

const OUT_EPS: f64 = 1e-5;
const TIME_SCALE: f64 = 1000.0;
/// The latent's channels
pub const LATENT_CHANNELS: usize = 32;
/// The vertex count `density` is a share of (`max_vertex_num`), scaled to timestep range
pub const MAX_VERTICES: f64 = 10000.0;

const FLOW_DIT: DitSpec = DitSpec {
    channels: 768,
    cond_channels: 1024,
    blocks: 12,
    heads: 12,
    mlp: 3072,
    share_mod: false,
    qk_norm: true,
    cross_qk_norm: false,
    rope_first: true,
};

/// `VertFlowEulerCfgSampler` as the image script runs it: guidance at every step.
#[derive(Clone, Copy, Debug)]
pub struct VertexSampler {
    pub steps: usize,
    pub cfg_strength: f64,
    pub rescale_t: f64,
}

pub const VERTEX_SAMPLER: VertexSampler = VertexSampler {
    steps: 24,
    cfg_strength: 3.0,
    rescale_t: 1.0,
};

pub struct VertexFlow {
    input: Linear,
    t_embed: TimestepMlp,
    density_embed: TimestepMlp,
    blocks: Vec<Block>,
    out_norm: LayerNorm,
    out: Linear,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl VertexFlow {
    /// The blocks run in `dtype` (bf16, as the reference's autocast); the input, embedders and output layer in f32.
    pub fn load(dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = FLOW_DIT.channels;
        let device = vb.device().clone();
        let ones = Tensor::ones(c, DType::F32, &device)?;
        let zeros = Tensor::zeros(c, DType::F32, &device)?;
        let plain = LayerNorm::new(ones.clone(), zeros.clone(), BLOCK_EPS);
        Ok(Self {
            input: inference_tensor::nn::linear(LATENT_CHANNELS, c, vb.pp("input_layer"))?,
            t_embed: TimestepMlp::load(c, vb.pp("t_embedder"))?,
            density_embed: TimestepMlp::load(c, vb.pp("density_embedder"))?,
            blocks: (0..FLOW_DIT.blocks)
                .map(|i| Block::load(FLOW_DIT, plain.clone(), dtype, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            out_norm: LayerNorm::new(ones, zeros, OUT_EPS),
            out: inference_tensor::nn::linear(c, LATENT_CHANNELS, vb.pp("out_layer"))?,
            head_dim: c / FLOW_DIT.heads,
            dtype,
            device,
        })
    }

    /// Each block's cross-attention keys and values over `cond` `[batch, tokens, 1024]`.
    pub fn condition(&self, cond: &Tensor) -> Result<Vec<CrossKv>> {
        let cond = cond.to_dtype(self.dtype)?;
        self.blocks.iter().map(|b| b.cross_kv(&cond)).collect()
    }

    /// The rotations for the scaffold's cells.
    pub fn rope(&self, cells: &[[i32; 3]]) -> Result<Rope> {
        let positions: Vec<[f32; 3]> = cells.iter().map(|c| c.map(|v| v as f32)).collect();
        Rope::new(&positions, self.head_dim, &self.device)
    }

    /// The velocity `[batch, cells, 32]` (f32) at model time `t` and `density` for latents `x` `[batch, cells, 32]`.
    pub fn forward(
        &self,
        x: &Tensor,
        t: f32,
        density: f32,
        rope: &Rope,
        kv: &[CrossKv],
    ) -> Result<Tensor> {
        let h = self.input.forward(x)?.to_dtype(self.dtype)?;
        let t_emb = (self.t_embed.forward(t, &self.device)?
            + self.density_embed.forward(density, &self.device)?)?
        .to_dtype(self.dtype)?;
        let mut h = h;
        for (block, kv) in self.blocks.iter().zip(kv) {
            h = block.forward(&h, &t_emb, Some(rope), kv)?;
        }
        let h = self.out_norm.forward(&h.to_dtype(DType::F32)?)?;
        self.out.forward(&h)
    }

    /// Euler sampling from `noise` `[1, cells, 32]` toward `vertices` vertices: `kv` holds the condition then the zero
    /// condition, mixed as `(1 + s) cond - s uncond` at every step.
    pub fn sample(
        &self,
        noise: &Tensor,
        vertices: usize,
        rope: &Rope,
        kv: &[CrossKv],
        params: &VertexSampler,
        progress: &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<Tensor> {
        let density = (vertices as f64 / MAX_VERTICES * TIME_SCALE) as f32;
        let ts = schedule(params.steps, params.rescale_t);
        let mut x = noise.clone();
        for (i, pair) in ts.windows(2).enumerate() {
            let (t, t_prev) = (pair[0], pair[1]);
            let both = self.forward(
                &x.repeat((2, 1, 1))?,
                (TIME_SCALE * t) as f32,
                density,
                rope,
                kv,
            )?;
            let s = params.cfg_strength;
            let v = ((both.narrow(0, 0, 1)? * (1.0 + s))? - (both.narrow(0, 1, 1)? * s)?)?;
            x = (&x - (v * (t - t_prev))?)?;
            progress(i + 1)?;
        }
        Ok(x)
    }
}
