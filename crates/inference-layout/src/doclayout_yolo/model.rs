use inference_tensor::nn::{Activation, VarBuilder};
use inference_tensor::{Module, Result, Tensor};

use crate::layers::{ConvNorm, ConvNormSpec};

const BN_EPS: f64 = 1e-3;
const NAMES: (&str, &str) = ("conv", "bn");
const ACT: Activation = Activation::Silu;
/// Dilation rates of the P3 and P4 G2L_CRM stages: module attributes in the checkpoint, not tensors.
const P3_DILATIONS: [usize; 3] = [1, 2, 3];
const P4_DILATIONS: [usize; 3] = [1, 3, 5];
/// PSA runs one head per 64 channels, keys and queries at half the head width.
const PSA_HEAD_CHANNELS: usize = 64;
const POOL_KERNEL: usize = 5;
const POOL_REPEATS: usize = 3;
pub const REG_MAX: usize = 16;
pub const STRIDES: [usize; 3] = [8, 16, 32];

fn conv_shape(vb: &VarBuilder) -> Result<(usize, usize, usize)> {
    let (out_c, in_per_group, k, _) = vb.pp(NAMES.0).get_unchecked("weight")?.dims4()?;
    Ok((out_c, in_per_group, k))
}

/// `Conv`: conv, BatchNorm, then SiLU unless `act` is None. Shapes come from the checkpoint; `in_c` sets the groups.
fn conv(vb: VarBuilder, in_c: usize, stride: usize, act: Option<Activation>) -> Result<ConvNorm> {
    let (out_c, in_per_group, k) = conv_shape(&vb)?;
    let mut spec = ConvNormSpec::new(in_c, out_c, k)
        .stride(stride)
        .groups(in_c / in_per_group)
        .names(NAMES)
        .eps(BN_EPS);
    if let Some(act) = act {
        spec = spec.act(act);
    }
    spec.load(vb)
}

fn out_channels(vb: &VarBuilder) -> Result<usize> {
    Ok(conv_shape(vb)?.0)
}

fn count_blocks(vb: &VarBuilder, leaf: &str) -> usize {
    (0..)
        .take_while(|i| vb.contains_tensor(&format!("m.{i}.{leaf}")))
        .count()
}

/// The shared-weight dilated conv of G2L_CRM: one kernel run at each rate, gated, then fused per channel.
struct DilatedBlock {
    /// BatchNorm-folded `dcv` weight `(c, c, k, k)` and bias, applied at every rate.
    dcv_w: Tensor,
    dcv_b: Tensor,
    kernel: usize,
    dilations: [usize; 3],
    cv2: ConvNorm,
    gating: ConvNorm,
    /// The grouped `(c, 3, 1, 1)` fusion as a dense 1x1 over the concatenated branches.
    fuse: ConvNorm,
}

impl DilatedBlock {
    fn new(vb: VarBuilder, c: usize, dilations: [usize; 3]) -> Result<Self> {
        let (_, _, kernel) = conv_shape(&vb.pp("dcv"))?;
        let (dcv_w, dcv_b) = ConvNormSpec::new(c, c, kernel)
            .names(NAMES)
            .eps(BN_EPS)
            .folded(vb.pp("dcv"))?;
        let branches = c * dilations.len();
        let gating = conv(vb.pp("conv_gating"), branches, 1, Some(ACT))?;
        let fuse_vb = vb.pp("conv1x1");
        let (w, b) = ConvNormSpec::new(branches, c, 1)
            .groups(c)
            .names(NAMES)
            .eps(BN_EPS)
            .folded(fuse_vb)?;
        let per = dilations.len();
        // output channel o reads inputs o*per..o*per+per; spread its taps into a block-diagonal dense weight
        let eye = Tensor::eye(c, w.dtype(), w.device())?;
        let dense = eye
            .unsqueeze(2)?
            .broadcast_mul(&w.reshape((c, 1, per))?)?
            .reshape((c, branches, 1, 1))?;
        let fuse = ConvNormSpec::new(branches, c, 1)
            .names(NAMES)
            .act(ACT)
            .build(dense, b)?;
        Ok(Self {
            dcv_w,
            dcv_b: dcv_b.reshape((1, c, 1, 1))?,
            kernel,
            dilations,
            cv2: conv(vb.pp("cv2"), c, 1, Some(ACT))?,
            gating,
            fuse,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let branches = self
            .dilations
            .iter()
            .map(|&d| {
                let y = xs
                    .conv2d(&self.dcv_w, d * (self.kernel / 2), 1, d, 1)?
                    .broadcast_add(&self.dcv_b)?
                    .silu()?;
                self.cv2.forward(&y)
            })
            .collect::<Result<Vec<_>>>()?;
        let cat = Tensor::cat(&branches, 1)?;
        let gate = inference_tensor::nn::ops::sigmoid(&self.gating.forward(&cat)?)?;
        xs + self.fuse.forward(&(cat * gate)?)?
    }
}

enum Block {
    /// Two 3x3 convs.
    Bottleneck {
        cv1: ConvNorm,
        cv2: ConvNorm,
        add: bool,
    },
    /// A bottleneck with the dilated block between its convs (G2L_CRM).
    Dilated {
        cv1: ConvNorm,
        block: Box<DilatedBlock>,
        cv2: ConvNorm,
    },
    /// Compact inverted block: depthwise and pointwise convs.
    Cib { convs: Vec<ConvNorm>, add: bool },
}

impl Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Bottleneck { cv1, cv2, add } => {
                let y = cv2.forward(&cv1.forward(xs)?)?;
                if *add { xs + y } else { Ok(y) }
            }
            Self::Dilated { cv1, block, cv2 } => {
                xs + cv2.forward(&block.forward(&cv1.forward(xs)?)?)?
            }
            Self::Cib { convs, add } => {
                let y = convs.iter().try_fold(xs.clone(), |y, c| c.forward(&y))?;
                if *add { xs + y } else { Ok(y) }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum BlockKind {
    Bottleneck { add: bool },
    Dilated([usize; 3]),
    Cib { add: bool },
}

/// C2f and its variants (G2L_CRM, C2fCIB): split, a chain of blocks each keeping its output, concatenated, fused.
struct C2f {
    cv1: ConvNorm,
    blocks: Vec<Block>,
    cv2: ConvNorm,
    half: usize,
}

impl C2f {
    fn new(vb: VarBuilder, in_c: usize, kind: BlockKind) -> Result<Self> {
        let cv1 = conv(vb.pp("cv1"), in_c, 1, Some(ACT))?;
        let half = out_channels(&vb.pp("cv1"))? / 2;
        let leaf = match kind {
            BlockKind::Cib { .. } => "cv1.0.conv.weight",
            _ => "cv1.conv.weight",
        };
        let blocks = (0..count_blocks(&vb, leaf))
            .map(|i| {
                let b = vb.pp("m").pp(i);
                Ok(match kind {
                    BlockKind::Bottleneck { add } => Block::Bottleneck {
                        cv1: conv(b.pp("cv1"), half, 1, Some(ACT))?,
                        cv2: conv(b.pp("cv2"), half, 1, Some(ACT))?,
                        add,
                    },
                    BlockKind::Dilated(dilations) => Block::Dilated {
                        cv1: conv(b.pp("cv1"), half, 1, Some(ACT))?,
                        block: Box::new(DilatedBlock::new(b.pp("dilated_block"), half, dilations)?),
                        cv2: conv(b.pp("cv2"), half, 1, Some(ACT))?,
                    },
                    BlockKind::Cib { add } => {
                        let mut c = half;
                        let convs = (0..5)
                            .map(|j| {
                                let vb = b.pp("cv1").pp(j);
                                let layer = conv(vb.clone(), c, 1, Some(ACT))?;
                                c = out_channels(&vb)?;
                                Ok(layer)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Block::Cib { convs, add }
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let cv2 = conv(vb.pp("cv2"), half * (2 + blocks.len()), 1, Some(ACT))?;
        Ok(Self {
            cv1,
            blocks,
            cv2,
            half,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(xs)?;
        let mut parts = vec![
            y.narrow(1, 0, self.half)?,
            y.narrow(1, self.half, self.half)?,
        ];
        for block in &self.blocks {
            let next = block.forward(parts.last().expect("two halves"))?;
            parts.push(next);
        }
        self.cv2.forward(&Tensor::cat(&parts, 1)?)
    }
}

/// Spatial-channel decoupled downsampling: pointwise, then a depthwise stride-2 conv.
struct SCDown {
    cv1: ConvNorm,
    cv2: ConvNorm,
}

impl SCDown {
    fn new(vb: VarBuilder, in_c: usize) -> Result<Self> {
        let c = out_channels(&vb.pp("cv1"))?;
        Ok(Self {
            cv1: conv(vb.pp("cv1"), in_c, 1, Some(ACT))?,
            cv2: conv(vb.pp("cv2"), c, 2, None)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.cv2.forward(&self.cv1.forward(xs)?)
    }
}

fn max_pool_same(xs: &Tensor) -> Result<Tensor> {
    let pad = POOL_KERNEL / 2;
    let (b, c, h, w) = xs.dims4()?;
    let neg = Tensor::full(f32::NEG_INFINITY, (b, c, h, pad), xs.device())?.to_dtype(xs.dtype())?;
    let xs = Tensor::cat(&[&neg, xs, &neg], 3)?;
    let neg = Tensor::full(f32::NEG_INFINITY, (b, c, pad, w + 2 * pad), xs.device())?
        .to_dtype(xs.dtype())?;
    Tensor::cat(&[&neg, &xs, &neg], 2)?.max_pool2d_with_stride(POOL_KERNEL, 1)
}

struct Sppf {
    cv1: ConvNorm,
    cv2: ConvNorm,
}

impl Sppf {
    fn new(vb: VarBuilder, in_c: usize) -> Result<Self> {
        let c = out_channels(&vb.pp("cv1"))?;
        Ok(Self {
            cv1: conv(vb.pp("cv1"), in_c, 1, Some(ACT))?,
            cv2: conv(vb.pp("cv2"), c * (POOL_REPEATS + 1), 1, Some(ACT))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut parts = vec![self.cv1.forward(xs)?];
        for _ in 0..POOL_REPEATS {
            let next = max_pool_same(parts.last().expect("cv1 output"))?;
            parts.push(next);
        }
        self.cv2.forward(&Tensor::cat(&parts, 1)?)
    }
}

/// Partial self-attention: half the channels get global attention and a feed-forward, the other half pass through.
struct Psa {
    cv1: ConvNorm,
    cv2: ConvNorm,
    qkv: ConvNorm,
    proj: ConvNorm,
    pe: ConvNorm,
    ffn: [ConvNorm; 2],
    half: usize,
    heads: usize,
    key_dim: usize,
}

impl Psa {
    fn new(vb: VarBuilder, in_c: usize) -> Result<Self> {
        let c = out_channels(&vb.pp("cv1"))?;
        let half = c / 2;
        let heads = half / PSA_HEAD_CHANNELS;
        let attn = vb.pp("attn");
        let ffn_c = out_channels(&vb.pp("ffn").pp(0))?;
        Ok(Self {
            cv1: conv(vb.pp("cv1"), in_c, 1, Some(ACT))?,
            cv2: conv(vb.pp("cv2"), c, 1, Some(ACT))?,
            qkv: conv(attn.pp("qkv"), half, 1, None)?,
            proj: conv(attn.pp("proj"), half, 1, None)?,
            pe: conv(attn.pp("pe"), half, 1, None)?,
            ffn: [
                conv(vb.pp("ffn").pp(0), half, 1, Some(ACT))?,
                conv(vb.pp("ffn").pp(1), ffn_c, 1, None)?,
            ],
            half,
            heads,
            key_dim: half / heads / 2,
        })
    }

    fn attention(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let n = h * w;
        let head_dim = c / self.heads;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, self.heads, 2 * self.key_dim + head_dim, n))?;
        let q = qkv.narrow(2, 0, self.key_dim)?.contiguous()?;
        let k = qkv.narrow(2, self.key_dim, self.key_dim)?.contiguous()?;
        let v = qkv.narrow(2, 2 * self.key_dim, head_dim)?.contiguous()?;
        let scale = (self.key_dim as f64).powf(-0.5);
        let attn = (q.transpose(2, 3)?.contiguous()?.matmul(&k)? * scale)?;
        let attn = inference_tensor::nn::ops::softmax_last_dim(&attn)?;
        let out = v
            .matmul(&attn.transpose(2, 3)?.contiguous()?)?
            .reshape((b, c, h, w))?;
        let out = (out + self.pe.forward(&v.reshape((b, c, h, w))?)?)?;
        self.proj.forward(&out)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(xs)?;
        let a = y.narrow(1, 0, self.half)?;
        let b = y.narrow(1, self.half, self.half)?.contiguous()?;
        let b = (&b + self.attention(&b)?)?;
        let b = (&b + self.ffn[1].forward(&self.ffn[0].forward(&b)?)?)?;
        self.cv2.forward(&Tensor::cat(&[a, b], 1)?)
    }
}

/// The one-to-one (NMS-free) branch of the v10 head for one level.
struct HeadLevel {
    reg: [ConvNorm; 3],
    cls: [ConvNorm; 5],
}

impl HeadLevel {
    fn new(vb: VarBuilder, level: usize, in_c: usize) -> Result<Self> {
        let reg = vb.pp("one2one_cv2").pp(level);
        let reg_c = out_channels(&reg.pp(0))?;
        let cls = vb.pp("one2one_cv3").pp(level);
        let cls_mid = out_channels(&cls.pp(0).pp(1))?;
        let cls_c = out_channels(&cls.pp(1).pp(1))?;
        let (nc, _, _, _) = cls.pp(2).get_unchecked("weight")?.dims4()?;
        Ok(Self {
            reg: [
                conv(reg.pp(0), in_c, 1, Some(ACT))?,
                conv(reg.pp(1), reg_c, 1, Some(ACT))?,
                ConvNormSpec::new(reg_c, 4 * REG_MAX, 1).load_plain(reg.pp(2))?,
            ],
            cls: [
                conv(cls.pp(0).pp(0), in_c, 1, Some(ACT))?,
                conv(cls.pp(0).pp(1), in_c, 1, Some(ACT))?,
                conv(cls.pp(1).pp(0), cls_mid, 1, Some(ACT))?,
                conv(cls.pp(1).pp(1), cls_mid, 1, Some(ACT))?,
                ConvNormSpec::new(cls_c, nc, 1).load_plain(cls.pp(2))?,
            ],
        })
    }

    /// `(b, 4 * REG_MAX, h*w)` box distributions and `(b, nc, h*w)` class logits.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, _, h, w) = xs.dims4()?;
        let reg = self.reg.iter().try_fold(xs.clone(), |y, c| c.forward(&y))?;
        let cls = self.cls.iter().try_fold(xs.clone(), |y, c| c.forward(&y))?;
        let nc = cls.dim(1)?;
        Ok((
            reg.reshape((b, 4 * REG_MAX, h * w))?,
            cls.reshape((b, nc, h * w))?,
        ))
    }
}

/// Head outputs over every anchor of every level: boxes `(b, 4, n)` as `[x1, y1, x2, y2]` in input pixels, and class
/// probabilities `(b, nc, n)`.
pub struct RawOutputs {
    pub boxes: Tensor,
    pub scores: Tensor,
}

/// YOLOv10-m with G2L_CRM backbone stages, as DocLayout-YOLO ships it (`model.<i>` layer indices).
pub struct DocLayoutYolo {
    stem: [ConvNorm; 2],
    c2f2: C2f,
    down3: ConvNorm,
    crm4: C2f,
    down5: SCDown,
    crm6: C2f,
    down7: SCDown,
    crm8: C2f,
    sppf9: Sppf,
    psa10: Psa,
    c2f13: C2f,
    c2f16: C2f,
    down17: ConvNorm,
    cib19: C2f,
    down20: SCDown,
    cib22: C2f,
    head: Vec<HeadLevel>,
    classes: usize,
}

impl DocLayoutYolo {
    /// `vb` holds the module's state dict (`model.0.conv.weight`, ...).
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let m = vb.pp("model");
        let l = |i: usize| m.pp(i);
        let oc = |i: usize, sub: &str| out_channels(&l(i).pp(sub));
        let c0 = out_channels(&l(0))?;
        let c1 = out_channels(&l(1))?;
        let c2 = oc(2, "cv2")?;
        let c3 = out_channels(&l(3))?;
        let c4 = oc(4, "cv2")?;
        let c5 = oc(5, "cv2")?;
        let c6 = oc(6, "cv2")?;
        let c7 = oc(7, "cv2")?;
        let c8 = oc(8, "cv2")?;
        let c9 = oc(9, "cv2")?;
        let c10 = oc(10, "cv2")?;
        let c13 = oc(13, "cv2")?;
        let c16 = oc(16, "cv2")?;
        let c17 = out_channels(&l(17))?;
        let c19 = oc(19, "cv2")?;
        let c20 = oc(20, "cv2")?;
        let c22 = oc(22, "cv2")?;
        let head = [c16, c19, c22]
            .iter()
            .enumerate()
            .map(|(level, &c)| HeadLevel::new(l(23), level, c))
            .collect::<Result<Vec<_>>>()?;
        let classes = l(23)
            .pp("one2one_cv3")
            .pp(0)
            .pp(2)
            .get_unchecked("weight")?
            .dim(0)?;
        Ok(Self {
            stem: [conv(l(0), 3, 2, Some(ACT))?, conv(l(1), c0, 2, Some(ACT))?],
            c2f2: C2f::new(l(2), c1, BlockKind::Bottleneck { add: true })?,
            down3: conv(l(3), c2, 2, Some(ACT))?,
            crm4: C2f::new(l(4), c3, BlockKind::Dilated(P3_DILATIONS))?,
            down5: SCDown::new(l(5), c4)?,
            crm6: C2f::new(l(6), c5, BlockKind::Dilated(P4_DILATIONS))?,
            down7: SCDown::new(l(7), c6)?,
            crm8: C2f::new(l(8), c7, BlockKind::Cib { add: true })?,
            sppf9: Sppf::new(l(9), c8)?,
            psa10: Psa::new(l(10), c9)?,
            c2f13: C2f::new(l(13), c10 + c6, BlockKind::Bottleneck { add: false })?,
            c2f16: C2f::new(l(16), c13 + c4, BlockKind::Bottleneck { add: false })?,
            down17: conv(l(17), c16, 2, Some(ACT))?,
            cib19: C2f::new(l(19), c17 + c13, BlockKind::Cib { add: true })?,
            down20: SCDown::new(l(20), c19)?,
            cib22: C2f::new(l(22), c20 + c10, BlockKind::Cib { add: true })?,
            head,
            classes,
        })
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    /// The P3, P4 and P5 features the head reads, from `(b, 3, h, w)` pixels in `[0, 1]` (h, w multiples of 32).
    pub fn features(&self, xs: &Tensor) -> Result<[Tensor; 3]> {
        let up = |t: &Tensor| -> Result<Tensor> {
            let (_, _, h, w) = t.dims4()?;
            t.upsample_nearest2d(2 * h, 2 * w)
        };
        let x = self.stem[1].forward(&self.stem[0].forward(xs)?)?;
        let x = self.down3.forward(&self.c2f2.forward(&x)?)?;
        let p3 = self.crm4.forward(&x)?;
        let p4 = self.crm6.forward(&self.down5.forward(&p3)?)?;
        let x = self.crm8.forward(&self.down7.forward(&p4)?)?;
        let p5 = self.psa10.forward(&self.sppf9.forward(&x)?)?;
        let n4 = self.c2f13.forward(&Tensor::cat(&[up(&p5)?, p4], 1)?)?;
        let o3 = self.c2f16.forward(&Tensor::cat(&[up(&n4)?, p3], 1)?)?;
        let o4 = self
            .cib19
            .forward(&Tensor::cat(&[self.down17.forward(&o3)?, n4], 1)?)?;
        let o5 = self
            .cib22
            .forward(&Tensor::cat(&[self.down20.forward(&o4)?, p5], 1)?)?;
        Ok([o3, o4, o5])
    }

    pub fn forward(&self, xs: &Tensor) -> Result<RawOutputs> {
        let feats = self.features(xs)?;
        let mut boxes = Vec::with_capacity(STRIDES.len());
        let mut scores = Vec::with_capacity(STRIDES.len());
        for ((level, feat), stride) in self.head.iter().zip(&feats).zip(STRIDES) {
            let (_, _, h, w) = feat.dims4()?;
            let (reg, cls) = level.forward(feat)?;
            boxes.push(decode(&reg, h, w, stride)?);
            scores.push(inference_tensor::nn::ops::sigmoid(&cls)?);
        }
        Ok(RawOutputs {
            boxes: Tensor::cat(&boxes, 2)?,
            scores: Tensor::cat(&scores, 2)?,
        })
    }
}

/// Distribution focal loss decode: the expected distance over `REG_MAX` bins on each side of each anchor centre.
fn decode(reg: &Tensor, h: usize, w: usize, stride: usize) -> Result<Tensor> {
    let (b, _, n) = reg.dims3()?;
    let dev = reg.device();
    let bins = Tensor::arange(0f32, REG_MAX as f32, dev)?
        .to_dtype(reg.dtype())?
        .reshape((1, 1, REG_MAX, 1))?;
    let probs = inference_tensor::nn::ops::softmax(&reg.reshape((b, 4, REG_MAX, n))?, 2)?;
    let dist = probs.broadcast_mul(&bins)?.sum(2)?;
    let xs = Tensor::arange(0f32, w as f32, dev)?.affine(1., 0.5)?;
    let ys = Tensor::arange(0f32, h as f32, dev)?.affine(1., 0.5)?;
    let ax = xs.unsqueeze(0)?.broadcast_as((h, w))?.flatten_all()?;
    let ay = ys.unsqueeze(1)?.broadcast_as((h, w))?.flatten_all()?;
    let anchors = Tensor::stack(&[&ax, &ay, &ax, &ay], 0)?
        .unsqueeze(0)?
        .to_dtype(reg.dtype())?;
    let sign = Tensor::new(&[-1f32, -1., 1., 1.], dev)?
        .to_dtype(reg.dtype())?
        .reshape((1, 4, 1))?;
    anchors.broadcast_add(&dist.broadcast_mul(&sign)?)? * stride as f64
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use inference_tensor::{DType, Device, Result, Tensor};

    use super::*;

    const CLASSES: usize = 3;

    struct Ckpt(HashMap<String, Tensor>);

    impl Ckpt {
        fn put(&mut self, name: String, shape: &[usize], lo: f64, hi: f64) -> Result<()> {
            let t = Tensor::rand(lo as f32, hi as f32, shape, &Device::Cpu)?;
            self.0.insert(name, t);
            Ok(())
        }

        /// `Conv` with BatchNorm: weight `(out, in_per_group, k, k)`.
        fn conv(&mut self, name: &str, out: usize, in_per_group: usize, k: usize) -> Result<()> {
            let bound = (1. / (in_per_group * k * k) as f64).sqrt();
            self.put(
                format!("{name}.conv.weight"),
                &[out, in_per_group, k, k],
                -bound,
                bound,
            )?;
            self.put(format!("{name}.bn.weight"), &[out], 0.5, 1.5)?;
            self.put(format!("{name}.bn.bias"), &[out], -0.1, 0.1)?;
            self.put(format!("{name}.bn.running_mean"), &[out], -0.1, 0.1)?;
            self.put(format!("{name}.bn.running_var"), &[out], 0.5, 1.5)
        }

        fn plain(&mut self, name: &str, out: usize, inp: usize) -> Result<()> {
            let bound = (1. / inp as f64).sqrt();
            self.put(format!("{name}.weight"), &[out, inp, 1, 1], -bound, bound)?;
            self.put(format!("{name}.bias"), &[out], -0.1, 0.1)
        }

        fn c2f(&mut self, name: &str, inp: usize, out: usize, block: &str) -> Result<()> {
            let half = out / 2;
            self.conv(&format!("{name}.cv1"), out, inp, 1)?;
            let b = format!("{name}.m.0");
            match block {
                "bottleneck" => {
                    self.conv(&format!("{b}.cv1"), half, half, 3)?;
                    self.conv(&format!("{b}.cv2"), half, half, 3)?;
                }
                "dilated" => {
                    self.conv(&format!("{b}.cv1"), half, half, 3)?;
                    self.conv(&format!("{b}.cv2"), half, half, 3)?;
                    let d = format!("{b}.dilated_block");
                    self.conv(&format!("{d}.dcv"), half, half, 3)?;
                    self.conv(&format!("{d}.cv2"), half, half, 1)?;
                    self.conv(&format!("{d}.conv_gating"), 3 * half, 1, 1)?;
                    self.conv(&format!("{d}.conv1x1"), half, 3, 1)?;
                }
                _ => {
                    let c = format!("{b}.cv1");
                    self.conv(&format!("{c}.0"), half, 1, 3)?;
                    self.conv(&format!("{c}.1"), 2 * half, half, 1)?;
                    self.conv(&format!("{c}.2"), 2 * half, 1, 3)?;
                    self.conv(&format!("{c}.3"), half, 2 * half, 1)?;
                    self.conv(&format!("{c}.4"), half, 1, 3)?;
                }
            }
            self.conv(&format!("{name}.cv2"), out, 3 * half, 1)
        }

        fn scdown(&mut self, name: &str, inp: usize, out: usize) -> Result<()> {
            self.conv(&format!("{name}.cv1"), out, inp, 1)?;
            self.conv(&format!("{name}.cv2"), out, 1, 3)
        }
    }

    /// A random DocLayout-YOLO state dict with every layer one block deep and narrow channels.
    pub(crate) fn tiny_checkpoint() -> Result<HashMap<String, Tensor>> {
        let mut c = Ckpt(HashMap::new());
        let m = |i: usize| format!("model.{i}");
        c.conv(&m(0), 8, 3, 3)?;
        c.conv(&m(1), 16, 8, 3)?;
        c.c2f(&m(2), 16, 16, "bottleneck")?;
        c.conv(&m(3), 32, 16, 3)?;
        c.c2f(&m(4), 32, 32, "dilated")?;
        c.scdown(&m(5), 32, 64)?;
        c.c2f(&m(6), 64, 64, "dilated")?;
        c.scdown(&m(7), 64, 128)?;
        c.c2f(&m(8), 128, 128, "cib")?;
        c.conv(&format!("{}.cv1", m(9)), 64, 128, 1)?;
        c.conv(&format!("{}.cv2", m(9)), 128, 256, 1)?;
        let psa = m(10);
        c.conv(&format!("{psa}.cv1"), 128, 128, 1)?;
        c.conv(&format!("{psa}.cv2"), 128, 128, 1)?;
        c.conv(&format!("{psa}.attn.qkv"), 128, 64, 1)?;
        c.conv(&format!("{psa}.attn.proj"), 64, 64, 1)?;
        c.conv(&format!("{psa}.attn.pe"), 64, 1, 3)?;
        c.conv(&format!("{psa}.ffn.0"), 128, 64, 1)?;
        c.conv(&format!("{psa}.ffn.1"), 64, 128, 1)?;
        c.c2f(&m(13), 128 + 64, 64, "bottleneck")?;
        c.c2f(&m(16), 64 + 32, 32, "bottleneck")?;
        c.conv(&m(17), 32, 32, 3)?;
        c.c2f(&m(19), 32 + 64, 64, "cib")?;
        c.scdown(&m(20), 64, 64)?;
        c.c2f(&m(22), 64 + 128, 128, "cib")?;
        for (level, inp) in [32, 64, 128].into_iter().enumerate() {
            let reg = format!("model.23.one2one_cv2.{level}");
            c.conv(&format!("{reg}.0"), 16, inp, 3)?;
            c.conv(&format!("{reg}.1"), 16, 16, 3)?;
            c.plain(&format!("{reg}.2"), 4 * REG_MAX, 16)?;
            let cls = format!("model.23.one2one_cv3.{level}");
            c.conv(&format!("{cls}.0.0"), inp, 1, 3)?;
            c.conv(&format!("{cls}.0.1"), 16, inp, 1)?;
            c.conv(&format!("{cls}.1.0"), 16, 1, 3)?;
            c.conv(&format!("{cls}.1.1"), 16, 16, 1)?;
            c.plain(&format!("{cls}.2"), CLASSES, 16)?;
        }
        Ok(c.0)
    }

    #[test]
    fn a_tiny_checkpoint_runs_alike_on_every_device() -> Result<()> {
        let ckpt = tiny_checkpoint()?;
        let input = Tensor::rand(0f32, 1., (1, 3, 96, 64), &Device::Cpu)?;
        let mut first: Option<RawOutputs> = None;
        for dev in crate::test_util::devices()? {
            let vb = VarBuilder::from_tensors(ckpt.clone(), DType::F32, &dev);
            let model = DocLayoutYolo::new(vb)?;
            assert_eq!(model.classes(), CLASSES);
            let out = model.forward(&input.to_device(&dev)?)?;
            // 12x8 + 6x4 + 3x2 anchors
            assert_eq!(out.boxes.dims(), [1, 4, 126]);
            assert_eq!(out.scores.dims(), [1, CLASSES, 126]);
            match &first {
                None => first = Some(out),
                Some(cpu) => {
                    let boxes = crate::test_util::rel_err(&out.boxes, &cpu.boxes)?;
                    let scores = crate::test_util::max_abs(&out.scores, &cpu.scores)?;
                    assert!(
                        boxes < 1e-4 && scores < 1e-4,
                        "{dev:?}: boxes {boxes}, scores {scores}"
                    );
                }
            }
        }
        Ok(())
    }
}
