//! TRELLIS.2's flows (`SLatFlowModel` for the shape and texture latents, `SparseStructureFlowModel` for the dense
//! structure latent: the same DiT, here over a full grid's cells) and their Euler sampler: latents on a structure's
//! cells, denoised under a DINOv3 condition with classifier-free guidance over an interval of the schedule.

use inference_tensor::nn::{LayerNorm, Linear, Module, VarBuilder};
use inference_tensor::{DType, Device, Result, Tensor};

use super::dit::{BLOCK_EPS, Block, CrossKv, DitSpec, Rope, TimestepMlp};

// the final F.layer_norm's default
const OUT_EPS: f64 = 1e-5;
// the sampler hands the model `1000 * t`
const TIME_SCALE: f64 = 1000.0;

// every TRELLIS.2-4B flow's DiT
const DIT_1_3B: DitSpec = DitSpec {
    channels: 1536,
    cond_channels: 1024,
    blocks: 30,
    heads: 12,
    mlp: 8192,
    share_mod: true,
};

/// `slat_flow_img2shape_dit_1_3B_{512,1024}_bf16.json`
pub const SHAPE_FLOW_1024: ShapeFlowSpec = ShapeFlowSpec {
    dit: DIT_1_3B,
    in_channels: 32,
    out_channels: 32,
};
/// `ss_flow_img_dit_1_3B_64_bf16.json`: 8 channels on a 16^3 grid
pub const STRUCTURE_FLOW: ShapeFlowSpec = ShapeFlowSpec {
    dit: DIT_1_3B,
    in_channels: 8,
    out_channels: 8,
};
/// `slat_flow_imgshape2tex_dit_1_3B_{512,1024}_bf16.json`: noise beside the normalized shape latents in
pub const TEXTURE_FLOW: ShapeFlowSpec = ShapeFlowSpec {
    dit: DIT_1_3B,
    in_channels: 64,
    out_channels: 32,
};

#[derive(Clone, Copy, Debug)]
pub struct ShapeFlowSpec {
    pub dit: DitSpec,
    pub in_channels: usize,
    pub out_channels: usize,
}

/// `FlowEulerGuidanceIntervalSampler` as `pipeline.json` sets it for the shape latents.
#[derive(Clone, Copy, Debug)]
pub struct SamplerParams {
    pub steps: usize,
    pub guidance_strength: f64,
    pub guidance_rescale: f64,
    pub guidance_interval: (f64, f64),
    pub rescale_t: f64,
    pub sigma_min: f64,
    /// The guidance rescale's spread: torch's `std` (Bessel-corrected) on a dense latent, the sparse tensors' own
    /// population spread on cells
    pub dense: bool,
}

pub const SHAPE_SAMPLER: SamplerParams = SamplerParams {
    steps: 12,
    guidance_strength: 7.5,
    guidance_rescale: 0.5,
    guidance_interval: (0.6, 1.0),
    rescale_t: 3.0,
    sigma_min: 1e-5,
    dense: false,
};
pub const STRUCTURE_SAMPLER: SamplerParams = SamplerParams {
    steps: 12,
    guidance_strength: 7.5,
    guidance_rescale: 0.7,
    guidance_interval: (0.6, 1.0),
    rescale_t: 5.0,
    sigma_min: 1e-5,
    dense: true,
};
pub const TEXTURE_SAMPLER: SamplerParams = SamplerParams {
    steps: 12,
    guidance_strength: 1.0,
    guidance_rescale: 0.0,
    guidance_interval: (0.6, 0.9),
    rescale_t: 3.0,
    sigma_min: 1e-5,
    dense: false,
};

pub struct ShapeFlow {
    input: Linear,
    t_embed: TimestepMlp,
    shared_mod: Linear,
    blocks: Vec<Block>,
    out_norm: LayerNorm,
    out: Linear,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl ShapeFlow {
    /// The blocks run in `dtype` (bf16, as the reference); the input, timestep, shared modulation and output layers in
    /// f32, as the reference keeps them.
    pub fn load(spec: ShapeFlowSpec, dtype: DType, vb: VarBuilder) -> Result<Self> {
        let c = spec.dit.channels;
        let device = vb.device().clone();
        let plain = LayerNorm::new(
            Tensor::ones(c, DType::F32, &device)?,
            Tensor::zeros(c, DType::F32, &device)?,
            BLOCK_EPS,
        );
        Ok(Self {
            input: inference_tensor::nn::linear(spec.in_channels, c, vb.pp("input_layer"))?,
            t_embed: TimestepMlp::load(c, vb.pp("t_embedder"))?,
            shared_mod: inference_tensor::nn::linear(c, 6 * c, vb.pp("adaLN_modulation").pp(1))?,
            blocks: (0..spec.dit.blocks)
                .map(|i| Block::load(spec.dit, plain.clone(), dtype, vb.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            out_norm: LayerNorm::new(
                Tensor::ones(c, DType::F32, &device)?,
                Tensor::zeros(c, DType::F32, &device)?,
                OUT_EPS,
            ),
            out: inference_tensor::nn::linear(c, spec.out_channels, vb.pp("out_layer"))?,
            head_dim: c / spec.dit.heads,
            dtype,
            device,
        })
    }

    /// Each block's cross-attention keys and values over `cond` `[batch, tokens, 1024]`.
    pub fn condition(&self, cond: &Tensor) -> Result<Vec<CrossKv>> {
        let cond = cond.to_dtype(self.dtype)?;
        self.blocks.iter().map(|b| b.cross_kv(&cond)).collect()
    }

    /// The velocity `[batch, cells, latent]` (f32) at timestep `t` (already scaled by 1000) for latents `x` `[batch,
    /// cells, latent]` on the cells whose rotations `rope` holds, each batch row under its own condition in `kv`.
    pub fn forward(&self, x: &Tensor, t: f32, rope: &Rope, kv: &[CrossKv]) -> Result<Tensor> {
        self.forward_inspect(x, t, rope, kv, &mut |_, _| {})
    }

    // `forward`, handing `inspect` the input layer's output, the shared modulation and each block's output
    pub(crate) fn forward_inspect(
        &self,
        x: &Tensor,
        t: f32,
        rope: &Rope,
        kv: &[CrossKv],
        inspect: &mut dyn FnMut(&str, &Tensor),
    ) -> Result<Tensor> {
        let h = self.input.forward(x)?;
        inspect("input_layer", &h);
        let h = h.to_dtype(self.dtype)?;
        let t_emb = self.t_embed.forward(t, &self.device)?;
        let shared = self.shared_mod.forward(&t_emb.silu()?)?;
        inspect("adaLN_modulation", &shared);
        let shared = shared.to_dtype(self.dtype)?;
        let mut h = h;
        for (i, (block, kv)) in self.blocks.iter().zip(kv).enumerate() {
            h = if i == 0 {
                block.forward_inspect(&h, &shared, rope, kv, &mut |name, t| {
                    inspect(&format!("b0.{name}"), t)
                })?
            } else {
                block.forward(&h, &shared, rope, kv)?
            };
            inspect(&format!("block{i}"), &h);
        }
        let h = self.out_norm.forward(&h.to_dtype(DType::F32)?)?;
        self.out.forward(&h)
    }

    /// The rotations for cells at integer `coords`.
    pub fn rope(&self, coords: &[[usize; 3]]) -> Result<Rope> {
        let positions: Vec<[f32; 3]> = coords.iter().map(|c| c.map(|v| v as f32)).collect();
        Rope::new(&positions, self.head_dim, &self.device)
    }

    /// Euler sampling from `noise` `[1, cells, out_channels]` (f32): `kv` holds the condition then the zero condition.
    /// Steps in the guidance interval run both and mix them (`cfg = s * pos + (1 - s) * neg`), rescaled toward the
    /// positive prediction's x0 spread; the others run the condition alone. `concat` `[1, cells, channels]` is appended
    /// to the latents' channels at every step (the texture flow's shape latents).
    pub fn sample(
        &self,
        noise: &Tensor,
        rope: &Rope,
        kv: &[CrossKv],
        concat: Option<&Tensor>,
        params: &SamplerParams,
        progress: &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<Tensor> {
        let input = |x: &Tensor| -> Result<Tensor> {
            match concat {
                Some(c) => Tensor::cat(&[x, c], 2),
                None => Ok(x.clone()),
            }
        };
        let cond_only: Vec<CrossKv> = kv.iter().map(|k| k.first(1)).collect::<Result<_>>()?;
        let n = params.steps;
        let r = params.rescale_t;
        let ts: Vec<f64> = (0..=n)
            .map(|i| {
                let t = 1.0 - i as f64 / n as f64;
                r * t / (1.0 + (r - 1.0) * t)
            })
            .collect();
        let s_min = params.sigma_min;
        let mut x = noise.clone();
        for (i, pair) in ts.windows(2).enumerate() {
            let (t, t_prev) = (pair[0], pair[1]);
            let model_t = (TIME_SCALE * t) as f32;
            let (lo, hi) = params.guidance_interval;
            let guided = lo <= t && t <= hi && params.guidance_strength != 1.0;
            let v = if guided {
                let both = self.forward(&input(&x)?.repeat((2, 1, 1))?, model_t, rope, kv)?;
                let (pos, neg) = (both.narrow(0, 0, 1)?, both.narrow(0, 1, 1)?);
                let s = params.guidance_strength;
                let cfg = ((&pos * s)? + (&neg * (1.0 - s))?)?;
                if params.guidance_rescale > 0.0 {
                    // x0 = (1 - s_min) x_t - (s_min + (1 - s_min) t) v
                    let k = s_min + (1.0 - s_min) * t;
                    let x0 = |v: &Tensor| (&x * (1.0 - s_min))? - (v * k)?;
                    let (x0_pos, x0_cfg) = (x0(&pos)?, x0(&cfg)?);
                    let spread = |x: &Tensor| spread(x, params.dense);
                    let ratio = (spread(&x0_pos)? / spread(&x0_cfg)?)?;
                    let rescaled = x0_cfg.broadcast_mul(&ratio)?;
                    let g = params.guidance_rescale;
                    let x0 = ((rescaled * g)? + (x0_cfg * (1.0 - g))?)?;
                    // back to a velocity: ((1 - s_min) x_t - x0) / k
                    (((&x * (1.0 - s_min))? - x0)? / k)?
                } else {
                    cfg
                }
            } else {
                self.forward(&input(&x)?, model_t, rope, &cond_only)?
            };
            x = (&x - (v * (t - t_prev))?)?;
            progress(i + 1)?;
        }
        Ok(x)
    }
}

// A sample's spread over all its cells and channels, `[1, 1, 1]`: sqrt(E[x^2] - E[x]^2) for sparse latents, torch's
// Bessel-corrected std for a dense one
fn spread(x: &Tensor, dense: bool) -> Result<Tensor> {
    if dense {
        let n = x.elem_count() as f64;
        let centered = x.broadcast_sub(&x.mean_all()?)?;
        return (centered.sqr()?.sum_all()? / (n - 1.0))?
            .sqrt()?
            .reshape((1, 1, 1));
    }
    let mean = x.mean_all()?;
    let mean2 = x.sqr()?.mean_all()?;
    (mean2 - mean.sqr()?)?.sqrt()?.reshape((1, 1, 1))
}

/// Latents normalized by `pipeline.json`'s statistics: `(x - mean) / std` per channel.
pub fn normalize(x: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
    let device = x.device();
    let c = mean.len();
    let mean = Tensor::from_slice(mean, (1, 1, c), device)?;
    let std = Tensor::from_slice(std, (1, 1, c), device)?;
    x.broadcast_sub(&mean)?.broadcast_div(&std)
}

/// Latents de-normalized by `pipeline.json`'s statistics: `x * std + mean` per channel.
pub fn denormalize(x: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor> {
    let device = x.device();
    let c = mean.len();
    let mean = Tensor::from_slice(mean, (1, 1, c), device)?;
    let std = Tensor::from_slice(std, (1, 1, c), device)?;
    x.broadcast_mul(&std)?.broadcast_add(&mean)
}
