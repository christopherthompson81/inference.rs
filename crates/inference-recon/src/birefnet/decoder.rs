//! BiRefNet's squeeze block and decoder, inference path only: deformable-ASPP decoder blocks, gradient-attention gates,
//! lateral skips, and the image fed back at every scale as patches stacked into channels.

use inference_tensor::nn::VarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};

const BN_EPS: f64 = 1e-5;
// the decoder blocks' and ASPP's widths
const INTER: usize = 64;
const ASPP: usize = 256;
const GDT: usize = 16;
// the ASPP's deformable branches beside its 1x1
const DEFORM_SIZES: [usize; 3] = [1, 3, 7];

/// A convolution with its bias and an eval-mode BatchNorm folded to a per-channel affine.
struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    affine: Option<(Tensor, Tensor)>,
    pad: usize,
}

fn batch_norm(c: usize, vb: VarBuilder) -> Result<(Tensor, Tensor)> {
    let scale = (vb.get(c, "weight")? / (vb.get(c, "running_var")? + BN_EPS)?.sqrt()?)?;
    let shift = (vb.get(c, "bias")? - (vb.get(c, "running_mean")? * &scale)?)?;
    Ok((scale.reshape((1, c, 1, 1))?, shift.reshape((1, c, 1, 1))?))
}

impl Conv {
    fn load(
        (cin, cout, k): (usize, usize, usize),
        bias: bool,
        bn: Option<VarBuilder>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            weight: vb.get((cout, cin, k, k), "weight")?,
            bias: if bias {
                Some(vb.get(cout, "bias")?.reshape((1, cout, 1, 1))?)
            } else {
                None
            },
            affine: bn.map(|vb| batch_norm(cout, vb)).transpose()?,
            pad: k / 2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.conv2d(&self.weight, self.pad, 1, 1, 1)?;
        if let Some(b) = &self.bias {
            xs = xs.broadcast_add(b)?;
        }
        self.affine(xs)
    }

    fn affine(&self, xs: Tensor) -> Result<Tensor> {
        match &self.affine {
            Some((scale, shift)) => xs.broadcast_mul(scale)?.broadcast_add(shift),
            None => Ok(xs),
        }
    }
}

/// Modulated deformable convolution (torchvision's `deform_conv2d` with a mask, stride 1, one offset group): each tap
/// samples the input bilinearly at its offset, zero outside, scaled by `2 * sigmoid` of the modulator.
struct DeformConv {
    offset: Conv,
    modulator: Conv,
    // `[taps, out, in]`
    weight: Tensor,
    k: usize,
}

impl DeformConv {
    fn load(cin: usize, cout: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        let taps = k * k;
        let weight = vb
            .pp("regular_conv")
            .get((cout, cin, k, k), "weight")?
            .reshape((cout, cin, taps))?
            .permute((2, 0, 1))?
            .contiguous()?;
        Ok(Self {
            offset: Conv::load((cin, 2 * taps, k), true, None, vb.pp("offset_conv"))?,
            modulator: Conv::load((cin, taps, k), true, None, vb.pp("modulator_conv"))?,
            weight,
            k,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let hw = h * w;
        let device = xs.device();
        let offset = self.offset.forward(xs)?.reshape((b, (), hw))?;
        let mask = (inference_tensor::nn::ops::sigmoid(&self.modulator.forward(xs)?)? * 2.0)?
            .reshape((b, (), hw))?;
        let flat = xs.reshape((b, c, hw))?;
        let (ys, xs_) = grid(h, w, device)?;
        let (hf, wf) = (h as f64, w as f64);
        let pad = (self.k / 2) as f64;
        let mut out: Option<Tensor> = None;
        for tap in 0..self.k * self.k {
            let (ki, kj) = ((tap / self.k) as f64, (tap % self.k) as f64);
            let y = ys
                .broadcast_add(&offset.narrow(1, 2 * tap, 1)?)?
                .affine(1.0, ki - pad)?;
            let x = xs_
                .broadcast_add(&offset.narrow(1, 2 * tap + 1, 1)?)?
                .affine(1.0, kj - pad)?;
            let inside = in_range(&y, -1.0, hf)?.mul(&in_range(&x, -1.0, wf)?)?;
            let (y0, x0) = (y.floor()?, x.floor()?);
            let (ly, lx) = ((&y - &y0)?, (&x - &x0)?);
            let (hy, hx) = (ly.affine(-1.0, 1.0)?, lx.affine(-1.0, 1.0)?);
            let gate = inside.mul(&mask.narrow(1, tap, 1)?)?;
            let mut sampled: Option<Tensor> = None;
            for (dy, dx, wy, wx) in [
                (0.0, 0.0, &hy, &hx),
                (0.0, 1.0, &hy, &lx),
                (1.0, 0.0, &ly, &hx),
                (1.0, 1.0, &ly, &lx),
            ] {
                let (cy, cx) = (y0.affine(1.0, dy)?, x0.affine(1.0, dx)?);
                let valid = in_range(&cy, -0.5, hf - 0.5)?.mul(&in_range(&cx, -0.5, wf - 0.5)?)?;
                let index = (cy.clamp(0.0, hf - 1.0)? * wf)?
                    .add(&cx.clamp(0.0, wf - 1.0)?)?
                    .to_dtype(DType::U32)?
                    .broadcast_as((b, c, hw))?
                    .contiguous()?;
                let weight = wy.mul(wx)?.mul(&valid)?.mul(&gate)?;
                let v = flat.gather(&index, 2)?.broadcast_mul(&weight)?;
                sampled = Some(match sampled {
                    Some(s) => (s + v)?,
                    None => v,
                });
            }
            let term = self
                .weight
                .get(tap)?
                .unsqueeze(0)?
                .broadcast_matmul(&sampled.expect("four corners"))?;
            out = Some(match out {
                Some(o) => (o + term)?,
                None => term,
            });
        }
        out.expect("at least one tap").reshape((b, (), h, w))
    }
}

// each output pixel's row and column, `[1, 1, h * w]` (f32)
fn grid(h: usize, w: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let ys: Vec<f32> = (0..h * w).map(|i| (i / w) as f32).collect();
    let xs: Vec<f32> = (0..h * w).map(|i| (i % w) as f32).collect();
    Ok((
        Tensor::from_vec(ys, (1, 1, h * w), device)?,
        Tensor::from_vec(xs, (1, 1, h * w), device)?,
    ))
}

// 1 where `lo < v < hi`, else 0 (f32)
fn in_range(v: &Tensor, lo: f64, hi: f64) -> Result<Tensor> {
    v.gt(lo)?.mul(&v.lt(hi)?)?.to_dtype(DType::F32)
}

struct AsppBranch {
    conv: DeformConv,
    bn: (Tensor, Tensor),
}

impl AsppBranch {
    fn load(cin: usize, k: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: DeformConv::load(cin, ASPP, k, vb.pp("atrous_conv"))?,
            bn: batch_norm(ASPP, vb.pp("bn"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv
            .forward(xs)?
            .broadcast_mul(&self.bn.0)?
            .broadcast_add(&self.bn.1)?
            .relu()
    }
}

struct AsppDeformable {
    branches: Vec<AsppBranch>,
    pool: Conv,
    out: Conv,
}

impl AsppDeformable {
    fn load(cin: usize, vb: VarBuilder) -> Result<Self> {
        let mut branches = vec![AsppBranch::load(cin, 1, vb.pp("aspp1"))?];
        for (i, &k) in DEFORM_SIZES.iter().enumerate() {
            branches.push(AsppBranch::load(cin, k, vb.pp("aspp_deforms").pp(i))?);
        }
        let gap = vb.pp("global_avg_pool");
        Ok(Self {
            branches,
            pool: Conv::load((cin, ASPP, 1), false, Some(gap.pp(2)), gap.pp(1))?,
            out: Conv::load(
                (ASPP * (2 + DEFORM_SIZES.len()), cin, 1),
                false,
                Some(vb.pp("bn1")),
                vb.pp("conv1"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = xs.dims4()?;
        let mut parts = self
            .branches
            .iter()
            .map(|br| br.forward(xs))
            .collect::<Result<Vec<_>>>()?;
        let pooled = self
            .pool
            .forward(&xs.mean_keepdim(D::Minus1)?.mean_keepdim(D::Minus2)?)?
            .relu()?;
        parts.push(pooled.broadcast_as((b, ASPP, h, w))?.contiguous()?);
        self.out.forward(&Tensor::cat(&parts, 1)?)?.relu()
    }
}

/// conv, BN, ReLU, deformable ASPP, conv, BN.
pub(crate) struct DecBlock {
    conv_in: Conv,
    att: AsppDeformable,
    conv_out: Conv,
}

impl DecBlock {
    pub(crate) fn load(cin: usize, cout: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv_in: Conv::load(
                (cin, INTER, 3),
                true,
                Some(vb.pp("bn_in")),
                vb.pp("conv_in"),
            )?,
            att: AsppDeformable::load(INTER, vb.pp("dec_att"))?,
            conv_out: Conv::load(
                (INTER, cout, 3),
                true,
                Some(vb.pp("bn_out")),
                vb.pp("conv_out"),
            )?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.conv_in.forward(xs)?.relu()?;
        self.conv_out.forward(&self.att.forward(&x)?)
    }
}

struct SimpleConvs {
    conv1: Conv,
    conv_out: Conv,
}

impl SimpleConvs {
    fn load(cin: usize, cout: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1: Conv::load((cin, INTER, 3), true, None, vb.pp("conv1"))?,
            conv_out: Conv::load((INTER, cout, 3), true, None, vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv_out.forward(&self.conv1.forward(xs)?)
    }
}

// The gradient-attention gate: conv, BN, ReLU, then a sigmoid map the features are scaled by
struct Gate {
    conv: Conv,
    attn: Conv,
}

impl Gate {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let g = self.conv.forward(xs)?.relu()?;
        xs.broadcast_mul(&inference_tensor::nn::ops::sigmoid(
            &self.attn.forward(&g)?,
        )?)
    }
}

// The image as `grid x grid` patches of the reference's size, stacked into channels: `b c (hg h) (wg w) -> b (c hg
// wg) h w`
fn patches(image: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (b, c, ih, iw) = image.dims4()?;
    let (gh, gw) = (ih / h, iw / w);
    image
        .reshape((b, c, gh, h, gw, w))?
        .permute((0, 1, 2, 4, 3, 5))?
        .reshape((b, c * gh * gw, h, w))
}

fn resize(xs: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (_, _, ih, iw) = xs.dims4()?;
    if (ih, iw) == (h, w) {
        return Ok(xs.clone());
    }
    xs.upsample_bilinear2d(h, w, true)
}

pub(crate) struct Decoder {
    // the image-patch blocks for the 1/1, 1/4, 1/8, 1/16 and 1/32 scales
    ipt: Vec<SimpleConvs>,
    // decoder blocks 4 (deepest) to 1
    blocks: Vec<DecBlock>,
    laterals: Vec<Conv>,
    gates: Vec<Gate>,
    out: Conv,
}

impl Decoder {
    /// `channels`: the encoder's four widths, deepest first.
    pub(crate) fn load(channels: [usize; 4], vb: VarBuilder) -> Result<Self> {
        let ch = channels;
        let ipt_out = [ch[3] / 8, ch[2] / 8, ch[1] / 8, ch[0] / 8, ch[0] / 8];
        let ipt = (0..5)
            .map(|i| {
                let grid = [1, 4, 8, 16, 32][i];
                SimpleConvs::load(
                    3 * grid * grid,
                    ipt_out[i],
                    vb.pp(format!("ipt_blk{}", i + 1)),
                )
            })
            .collect::<Result<_>>()?;
        let block_io = [
            (ch[0] + ch[0] / 8, ch[1]),
            (ch[1] + ch[0] / 8, ch[2]),
            (ch[2] + ch[1] / 8, ch[3]),
            (ch[3] + ch[2] / 8, ch[3] / 2),
        ];
        let blocks = block_io
            .iter()
            .enumerate()
            .map(|(i, &(cin, cout))| {
                DecBlock::load(cin, cout, vb.pp(format!("decoder_block{}", 4 - i)))
            })
            .collect::<Result<_>>()?;
        let laterals = (0..3)
            .map(|i| {
                Conv::load(
                    (ch[i + 1], ch[i + 1], 1),
                    true,
                    None,
                    vb.pp(format!("lateral_block{}", 4 - i)).pp("conv"),
                )
            })
            .collect::<Result<_>>()?;
        let gates = (0..3)
            .map(|i| {
                let n = 4 - i;
                let conv = vb.pp(format!("gdt_convs_{n}"));
                Ok(Gate {
                    conv: Conv::load((ch[i + 1], GDT, 3), true, Some(conv.pp(1)), conv.pp(0))?,
                    attn: Conv::load(
                        (GDT, 1, 1),
                        true,
                        None,
                        vb.pp(format!("gdt_convs_attn_{n}")).pp(0),
                    )?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            ipt,
            blocks,
            laterals,
            gates,
            out: Conv::load(
                (ch[3] / 2 + ch[3] / 8, 1, 1),
                true,
                None,
                vb.pp("conv_out1").pp(0),
            )?,
        })
    }

    /// The mask logits `[B, 1, H, W]` from the image and the encoder's features, finest first.
    pub(crate) fn forward(&self, image: &Tensor, feats: [&Tensor; 4]) -> Result<Tensor> {
        let [x1, x2, x3, x4] = feats;
        let with_patches = |xs: &Tensor, ipt: &SimpleConvs| -> Result<Tensor> {
            let (_, _, h, w) = xs.dims4()?;
            Tensor::cat(&[xs, &ipt.forward(&patches(image, h, w)?)?], 1)
        };
        let mut p = with_patches(x4, &self.ipt[4])?;
        let skips = [x3, x2, x1];
        for i in 0..3 {
            p = self.gates[i].forward(&self.blocks[i].forward(&p)?)?;
            let skip = skips[i];
            let (_, _, h, w) = skip.dims4()?;
            p = (resize(&p, h, w)? + self.laterals[i].forward(skip)?)?;
            p = with_patches(&p, &self.ipt[3 - i])?;
        }
        let p = self.blocks[3].forward(&p)?;
        let (_, _, h, w) = image.dims4()?;
        let p = with_patches(&resize(&p, h, w)?, &self.ipt[0])?;
        self.out.forward(&p)
    }
}
