//! TRELLIS-image-large's sparse-structure flow (`ss_flow_img_dit_L_16l8_fp16`): a dense DiT over the 8-channel 16^3
//! latent with absolute sinusoidal positions, each block its own AdaLN, under DINOv2 tokens; its Euler sampler guides
//! only inside an interval of the schedule. TRELLIS's structure decoder turns the latent into 64^3 occupancy.

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use super::ape;
use crate::trellis2::dit::{BLOCK_EPS, Block, CrossKv, DitSpec, TimestepMlp};

// the final F.layer_norm's default
const OUT_EPS: f64 = 1e-5;
const TIME_SCALE: f64 = 1000.0;
/// The latent's side and channels
pub const LATENT_SIDE: usize = 16;
pub const LATENT_CHANNELS: usize = 8;

const FLOW_DIT: DitSpec = DitSpec {
    channels: 1024,
    cond_channels: 1024,
    blocks: 24,
    heads: 16,
    mlp: 4096,
    share_mod: false,
    qk_norm: true,
    cross_qk_norm: false,
    rope_first: false,
};

/// `SSFlowEulerSampler` (TRELLIS's `FlowEulerGuidanceIntervalSampler`) as LATO.2's image script runs it.
#[derive(Clone, Copy, Debug)]
pub struct StructureSampler {
    pub steps: usize,
    pub cfg_strength: f64,
    pub cfg_interval: (f64, f64),
    pub rescale_t: f64,
}

pub const STRUCTURE_SAMPLER: StructureSampler = StructureSampler {
    steps: 25,
    cfg_strength: 5.0,
    cfg_interval: (0.5, 1.0),
    rescale_t: 3.0,
};

/// The schedule `rescale * t / (1 + (rescale - 1) t)` over `steps` uniform steps from 1 to 0.
pub fn schedule(steps: usize, rescale: f64) -> Vec<f64> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f64 / steps as f64;
            rescale * t / (1.0 + (rescale - 1.0) * t)
        })
        .collect()
}

pub struct StructureFlow {
    input: Linear,
    pos: Tensor,
    t_embed: TimestepMlp,
    blocks: Vec<Block>,
    out_norm: LayerNorm,
    out: Linear,
    dtype: DType,
    device: Device,
}

impl StructureFlow {
    /// The blocks run in `dtype` (f16, as the checkpoint's `use_fp16`); the input, timestep and output layers in f32.
    pub fn load(dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = FLOW_DIT.channels;
        let device = vb.device().clone();
        let ones = Tensor::ones(c, DType::F32, &device)?;
        let zeros = Tensor::zeros(c, DType::F32, &device)?;
        let plain = LayerNorm::new(ones.clone(), zeros.clone(), BLOCK_EPS);
        let cells = crate::trellis2::structure::grid_cells(LATENT_SIDE);
        let coords: Vec<[f32; 3]> = cells.iter().map(|c| c.map(|v| v as f32)).collect();
        Ok(Self {
            input: inference_tensor::nn::linear(LATENT_CHANNELS, c, vb.pp("input_layer"))?,
            pos: ape(&coords, c, &device)?.unsqueeze(0)?,
            t_embed: TimestepMlp::load(c, vb.pp("t_embedder"))?,
            blocks: (0..FLOW_DIT.blocks)
                .map(|i| Block::load(FLOW_DIT, plain.clone(), dtype, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            out_norm: LayerNorm::new(ones, zeros, OUT_EPS),
            out: inference_tensor::nn::linear(c, LATENT_CHANNELS, vb.pp("out_layer"))?,
            dtype,
            device,
        })
    }

    /// Each block's cross-attention keys and values over `cond` `[batch, tokens, 1024]`.
    pub fn condition(&self, cond: &Tensor) -> Result<Vec<CrossKv>> {
        let cond = cond.to_dtype(self.dtype)?;
        self.blocks.iter().map(|b| b.cross_kv(&cond)).collect()
    }

    /// The velocity `[batch, 4096, 8]` (f32) at model time `t` for latents `x` `[batch, 4096, 8]` (cells x-major).
    pub fn forward(&self, x: &Tensor, t: f32, kv: &[CrossKv]) -> Result<Tensor> {
        let h = self.input.forward(x)?.broadcast_add(&self.pos)?;
        let t_emb = self
            .t_embed
            .forward(t, &self.device)?
            .to_dtype(self.dtype)?;
        let mut h = h.to_dtype(self.dtype)?;
        for (block, kv) in self.blocks.iter().zip(kv) {
            h = block.forward(&h, &t_emb, None, kv)?;
        }
        let h = self.out_norm.forward(&h.to_dtype(DType::F32)?)?;
        self.out.forward(&h)
    }

    /// Euler sampling from `noise` `[1, 4096, 8]`: `kv` holds the condition then the zero condition; steps inside the
    /// guidance interval mix them as `(1 + s) cond - s uncond`, the others run the condition alone.
    pub fn sample(
        &self,
        noise: &Tensor,
        kv: &[CrossKv],
        params: &StructureSampler,
        progress: &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<Tensor> {
        let cond_only: Vec<CrossKv> = kv.iter().map(|k| k.first(1)).collect::<Result<_>>()?;
        let ts = schedule(params.steps, params.rescale_t);
        let mut x = noise.clone();
        for (i, pair) in ts.windows(2).enumerate() {
            let (t, t_prev) = (pair[0], pair[1]);
            let model_t = (TIME_SCALE * t) as f32;
            let (lo, hi) = params.cfg_interval;
            let v = if lo <= t && t <= hi {
                let both = self.forward(&x.repeat((2, 1, 1))?, model_t, kv)?;
                let s = params.cfg_strength;
                ((both.narrow(0, 0, 1)? * (1.0 + s))? - (both.narrow(0, 1, 1)? * s)?)?
            } else {
                self.forward(&x, model_t, &cond_only)?
            };
            x = (&x - (v * (t - t_prev))?)?;
            progress(i + 1)?;
        }
        Ok(x)
    }
}

/// The occupied cells of TRELLIS's Z-up `n`^3 occupancy logits (x-major) as LATO.2's Y-up voxels: `(x, y, z)` to
/// `(x, z, n - 1 - y)`, sorted x-major.
pub fn y_up_voxels(logits: &[f32], n: usize) -> Vec<[i32; 3]> {
    let mut cells: Vec<[i32; 3]> = logits
        .iter()
        .enumerate()
        .filter(|&(_, &l)| l > 0.0)
        .map(|(i, _)| {
            let (x, y, z) = (i / (n * n), i / n % n, i % n);
            [x as i32, z as i32, (n - 1 - y) as i32]
        })
        .collect();
    cells.sort_unstable();
    cells
}
