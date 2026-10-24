//! Wan 2.1's 3D-causal VAE, run one frame at a time as the keyframe pipelines do. A lone frame only ever meets the
//! causal convs' last temporal tap (the two before it are zero padding) and skips the temporal resamplers, so every
//! 3D conv becomes the 2D conv of its last kernel slice.

use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Result, Tensor};
use serde::Deserialize;

// Frames run this many at a time, which bounds the full-resolution im2col buffers
const FRAMES_PER_CHUNK: usize = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    pub base_dim: usize,
    pub z_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub temperal_downsample: Vec<bool>,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

// A causal conv's weight `[o, i, kt, kh, kw]` as the 2D conv a single frame sees: its last temporal slice
struct FrameConv {
    weight: Tensor,
    bias: Tensor,
    padding: usize,
    stride: usize,
}

impl FrameConv {
    fn load(vb: &ShardedVarBuilder, padding: usize, dtype: DType) -> Result<Self> {
        let weight = vb.get_unchecked("weight")?;
        let weight = match weight.rank() {
            5 => {
                let kt = weight.dim(2)?;
                weight.narrow(2, kt - 1, 1)?.squeeze(2)?
            }
            _ => weight,
        };
        Ok(Self {
            weight: weight.to_dtype(dtype)?.contiguous()?,
            bias: vb.get_unchecked("bias")?.to_dtype(dtype)?,
            padding,
            stride: 1,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.conv2d(&self.weight, self.padding, self.stride, 1, 1)?
            .broadcast_add(&self.bias.reshape((1, (), 1, 1))?)
    }
}

// L2 normalization over channels, times sqrt(channels) and a per-channel gain
struct RmsNorm {
    gamma: Tensor,
    scale: f64,
}

impl RmsNorm {
    fn load(vb: &ShardedVarBuilder, dim: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            gamma: vb
                .get_unchecked("gamma")?
                .flatten_all()?
                .reshape((1, dim, 1, 1))?
                .to_dtype(dtype)?,
            scale: (dim as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let norm = xs
            .to_dtype(DType::F32)?
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?;
        (xs.to_dtype(DType::F32)?.broadcast_div(&norm)? * self.scale)?
            .to_dtype(xs.dtype())?
            .broadcast_mul(&self.gamma)
    }
}

struct ResBlock {
    norm1: RmsNorm,
    conv1: FrameConv,
    norm2: RmsNorm,
    conv2: FrameConv,
    shortcut: Option<FrameConv>,
}

impl ResBlock {
    fn load(vb: ShardedVarBuilder, in_dim: usize, out_dim: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::load(&vb.pp("norm1"), in_dim, dtype)?,
            conv1: FrameConv::load(&vb.pp("conv1"), 1, dtype)?,
            norm2: RmsNorm::load(&vb.pp("norm2"), out_dim, dtype)?,
            conv2: FrameConv::load(&vb.pp("conv2"), 1, dtype)?,
            shortcut: if in_dim != out_dim {
                Some(FrameConv::load(&vb.pp("conv_shortcut"), 0, dtype)?)
            } else {
                None
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = match &self.shortcut {
            Some(conv) => conv.forward(xs)?,
            None => xs.clone(),
        };
        let x = self.conv1.forward(&self.norm1.forward(xs)?.silu()?)?;
        let x = self.conv2.forward(&self.norm2.forward(&x)?.silu()?)?;
        x + h
    }
}

// Single-head attention over a frame's pixels
struct AttnBlock {
    norm: RmsNorm,
    to_qkv: FrameConv,
    proj: FrameConv,
}

impl AttnBlock {
    fn load(vb: ShardedVarBuilder, dim: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::load(&vb.pp("norm"), dim, dtype)?,
            to_qkv: FrameConv::load(&vb.pp("to_qkv"), 0, dtype)?,
            proj: FrameConv::load(&vb.pp("proj"), 0, dtype)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (n, c, h, w) = xs.dims4()?;
        let qkv = self
            .to_qkv
            .forward(&self.norm.forward(xs)?)?
            .reshape((n, 3 * c, h * w))?
            .transpose(1, 2)?;
        let q = qkv.narrow(2, 0, c)?.contiguous()?;
        let k = qkv.narrow(2, c, c)?.contiguous()?;
        let v = qkv.narrow(2, 2 * c, c)?.contiguous()?;
        let scores = (q
            .to_dtype(DType::F32)?
            .matmul(&k.to_dtype(DType::F32)?.t()?)?
            / (c as f64).sqrt())?;
        let probs = inference_tensor::nn::ops::softmax_last_dim(&scores)?;
        let out = probs
            .matmul(&v.to_dtype(DType::F32)?)?
            .to_dtype(xs.dtype())?
            .transpose(1, 2)?
            .reshape((n, c, h, w))?;
        self.proj.forward(&out)? + xs
    }
}

struct MidBlock {
    res0: ResBlock,
    attn: AttnBlock,
    res1: ResBlock,
}

impl MidBlock {
    fn load(vb: ShardedVarBuilder, dim: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            res0: ResBlock::load(vb.pp("resnets").pp(0), dim, dim, dtype)?,
            attn: AttnBlock::load(vb.pp("attentions").pp(0), dim, dtype)?,
            res1: ResBlock::load(vb.pp("resnets").pp(1), dim, dim, dtype)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.res1
            .forward(&self.attn.forward(&self.res0.forward(xs)?)?)
    }
}

enum EncoderLayer {
    Res(ResBlock),
    // zero-pad right and bottom by one, then a stride-2 3x3 conv
    Down(FrameConv),
}

/// The encoder half: `[n, 3, H, W]` frames in [-1, 1] to latent means `[n, z, H/8, W/8]`.
pub struct WanEncoder {
    conv_in: FrameConv,
    layers: Vec<EncoderLayer>,
    mid: MidBlock,
    norm_out: RmsNorm,
    conv_out: FrameConv,
    quant_conv: FrameConv,
    z_dim: usize,
}

impl WanEncoder {
    fn load(vb: ShardedVarBuilder, cfg: &VaeConfig, dtype: DType) -> Result<Self> {
        let enc = vb.pp("encoder");
        let dims = std::iter::once(cfg.base_dim)
            .chain(cfg.dim_mult.iter().map(|m| cfg.base_dim * m))
            .collect::<Vec<_>>();
        let mut layers = Vec::new();
        let mut index = 0;
        for (i, pair) in dims.windows(2).enumerate() {
            let mut in_dim = pair[0];
            for _ in 0..cfg.num_res_blocks {
                layers.push(EncoderLayer::Res(ResBlock::load(
                    enc.pp("down_blocks").pp(index),
                    in_dim,
                    pair[1],
                    dtype,
                )?));
                index += 1;
                in_dim = pair[1];
            }
            if i != cfg.dim_mult.len() - 1 {
                let mut conv = FrameConv::load(
                    &enc.pp("down_blocks").pp(index).pp("resample").pp(1),
                    0,
                    dtype,
                )?;
                conv.stride = 2;
                layers.push(EncoderLayer::Down(conv));
                index += 1;
            }
        }
        let top = *dims.last().expect("dim_mult is not empty");
        Ok(Self {
            conv_in: FrameConv::load(&enc.pp("conv_in"), 1, dtype)?,
            layers,
            mid: MidBlock::load(enc.pp("mid_block"), top, dtype)?,
            norm_out: RmsNorm::load(&enc.pp("norm_out"), top, dtype)?,
            conv_out: FrameConv::load(&enc.pp("conv_out"), 1, dtype)?,
            quant_conv: FrameConv::load(&vb.pp("quant_conv"), 0, dtype)?,
            z_dim: cfg.z_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_in.forward(xs)?;
        for layer in &self.layers {
            x = match layer {
                EncoderLayer::Res(block) => block.forward(&x)?,
                EncoderLayer::Down(conv) => {
                    conv.forward(&x.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?)?
                }
            };
        }
        let x = self.mid.forward(&x)?;
        let x = self.conv_out.forward(&self.norm_out.forward(&x)?.silu()?)?;
        // the posterior's mode: the mean half of (mean, log variance)
        self.quant_conv.forward(&x)?.narrow(1, 0, self.z_dim)
    }
}

struct UpBlock {
    resnets: Vec<ResBlock>,
    // nearest 2x, then a 3x3 conv that halves the channels
    upsample: Option<FrameConv>,
}

/// The decoder half: latents `[n, z, h, w]` to frames `[n, 3, 8h, 8w]`, unclipped.
pub struct WanDecoder {
    post_quant_conv: FrameConv,
    conv_in: FrameConv,
    mid: MidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: RmsNorm,
    conv_out: FrameConv,
}

impl WanDecoder {
    fn load(vb: ShardedVarBuilder, cfg: &VaeConfig, dtype: DType) -> Result<Self> {
        let dec = vb.pp("decoder");
        let top = cfg.base_dim * cfg.dim_mult.last().expect("dim_mult is not empty");
        let dims = std::iter::once(top)
            .chain(cfg.dim_mult.iter().rev().map(|m| cfg.base_dim * m))
            .collect::<Vec<_>>();
        let mut up_blocks = Vec::new();
        for (i, pair) in dims.windows(2).enumerate() {
            let vb = dec.pp("up_blocks").pp(i);
            let mut in_dim = if i > 0 { pair[0] / 2 } else { pair[0] };
            let mut resnets = Vec::new();
            for r in 0..=cfg.num_res_blocks {
                resnets.push(ResBlock::load(
                    vb.pp("resnets").pp(r),
                    in_dim,
                    pair[1],
                    dtype,
                )?);
                in_dim = pair[1];
            }
            let upsample = if i != cfg.dim_mult.len() - 1 {
                Some(FrameConv::load(
                    &vb.pp("upsamplers").pp(0).pp("resample").pp(1),
                    1,
                    dtype,
                )?)
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets, upsample });
        }
        let bottom = *dims.last().expect("dim_mult is not empty");
        Ok(Self {
            post_quant_conv: FrameConv::load(&vb.pp("post_quant_conv"), 0, dtype)?,
            conv_in: FrameConv::load(&dec.pp("conv_in"), 1, dtype)?,
            mid: MidBlock::load(dec.pp("mid_block"), top, dtype)?,
            up_blocks,
            norm_out: RmsNorm::load(&dec.pp("norm_out"), bottom, dtype)?,
            conv_out: FrameConv::load(&dec.pp("conv_out"), 1, dtype)?,
        })
    }

    fn forward(&self, zs: &Tensor) -> Result<Tensor> {
        let x = self.post_quant_conv.forward(zs)?;
        let mut x = self.mid.forward(&self.conv_in.forward(&x)?)?;
        for block in &self.up_blocks {
            for res in &block.resnets {
                x = res.forward(&x)?;
            }
            if let Some(conv) = &block.upsample {
                let (_, _, h, w) = x.dims4()?;
                let up = x
                    .to_dtype(DType::F32)?
                    .upsample_nearest2d(2 * h, 2 * w)?
                    .to_dtype(x.dtype())?;
                x = conv.forward(&up)?;
            }
        }
        self.conv_out.forward(&self.norm_out.forward(&x)?.silu()?)
    }
}

/// Wan 2.1's VAE for single frames, with the latent normalization the Wan pipelines apply.
pub struct WanVae {
    encoder: WanEncoder,
    decoder: WanDecoder,
    mean: Tensor,
    std: Tensor,
    dtype: DType,
}

impl WanVae {
    pub fn new(vb: ShardedVarBuilder, cfg: &VaeConfig, dtype: DType) -> Result<Self> {
        let z = cfg.z_dim;
        let device = vb.device().clone();
        Ok(Self {
            encoder: WanEncoder::load(vb.clone(), cfg, dtype)?,
            decoder: WanDecoder::load(vb, cfg, dtype)?,
            mean: Tensor::from_slice(&cfg.latents_mean, (1, z, 1, 1), &device)?,
            std: Tensor::from_slice(&cfg.latents_std, (1, z, 1, 1), &device)?,
            dtype,
        })
    }

    /// Frames `[n, 3, H, W]` in [-1, 1] to normalized latents `[n, z, H/8, W/8]` in f32: `(mean - mu) / sigma` per
    /// channel, each frame encoded on its own.
    pub fn encode(&self, frames: &Tensor) -> Result<Tensor> {
        let n = frames.dim(0)?;
        let chunks = (0..n)
            .step_by(FRAMES_PER_CHUNK)
            .map(|start| {
                let len = FRAMES_PER_CHUNK.min(n - start);
                let z = self
                    .encoder
                    .forward(&frames.narrow(0, start, len)?.to_dtype(self.dtype)?)?
                    .to_dtype(DType::F32)?;
                z.broadcast_sub(&self.mean)?.broadcast_div(&self.std)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&chunks, 0)
    }

    /// Normalized latents `[n, z, h, w]` to frames `[n, 3, 8h, 8w]` in [-1, 1] (f32), each frame decoded on its own.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let n = latents.dim(0)?;
        let chunks = (0..n)
            .step_by(FRAMES_PER_CHUNK)
            .map(|start| {
                let len = FRAMES_PER_CHUNK.min(n - start);
                let z = latents
                    .narrow(0, start, len)?
                    .to_dtype(DType::F32)?
                    .broadcast_mul(&self.std)?
                    .broadcast_add(&self.mean)?;
                self.decoder
                    .forward(&z.to_dtype(self.dtype)?)?
                    .to_dtype(DType::F32)?
                    .clamp(-1.0, 1.0)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&chunks, 0)
    }
}
