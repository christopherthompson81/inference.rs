use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, Conv2d, Conv2dConfig, Linear, VarBuilder};

const BN_EPS: f64 = 1e-5;
/// The AVX2 direct conv tiles output channels in groups of this size.
const DIRECT_OC_MULTIPLE: usize = 4;
/// Below this many input channels per-tap CPU GEMMs are too thin and im2col + one GEMM wins (stem conv).
const IMPLICIT_MIN_IN_C: usize = 16;

pub const HF_CONV: (&str, &str) = ("convolution", "normalization");
pub const RTDETR_CONV: (&str, &str) = ("conv", "norm");

/// Depthwise conv; candle runs `groups=C` as C separate convolutions.
#[derive(Debug, Clone)]
struct Depthwise {
    /// `(C, 1, k, k)`
    w: Tensor,
    /// `(C,)`
    b: Tensor,
    /// `(1, C, 1, 1)` weight per `(ky, kx)`, row-major, for the backends without a custom kernel.
    taps: Vec<Tensor>,
    bias: Tensor,
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl Depthwise {
    fn new(
        w: &Tensor,
        bias: &Tensor,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let c = w.dim(0)?;
        let mut taps = Vec::with_capacity(kernel * kernel);
        for ky in 0..kernel {
            for kx in 0..kernel {
                taps.push(
                    w.narrow(2, ky, 1)?
                        .narrow(3, kx, 1)?
                        .reshape((1, c, 1, 1))?
                        .contiguous()?,
                );
            }
        }
        Ok(Self {
            w: w.contiguous()?,
            b: bias.contiguous()?,
            taps,
            bias: bias.reshape((1, c, 1, 1))?,
            kernel,
            stride,
            padding,
        })
    }
}

impl Module for Depthwise {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.device().is_metal() {
            return self.forward_taps(xs);
        }
        crate::depthwise::depthwise_conv2d(xs, &self.w, &self.b, self.stride, self.padding)
    }
}

impl Depthwise {
    /// Sum of k*k shifted taps; strided taps are a phase select after padding to a stride multiple.
    fn forward_taps(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let (k, s, p) = (self.kernel, self.stride, self.padding);
        let ho = (h + 2 * p - k) / s + 1;
        let wo = (w + 2 * p - k) / s + 1;
        let hp = (h + 2 * p).div_ceil(s) * s;
        let wp = (w + 2 * p).div_ceil(s) * s;
        let xp = xs
            .pad_with_zeros(2, p, hp - h - p)?
            .pad_with_zeros(3, p, wp - w - p)?
            .reshape((b, c, hp / s, s, wp / s, s))?;
        let mut acc = self.bias.clone();
        for ky in 0..k {
            for kx in 0..k {
                let v = xp
                    .narrow(3, ky % s, 1)?
                    .narrow(5, kx % s, 1)?
                    .squeeze(5)?
                    .squeeze(3)?
                    .narrow(2, ky / s, ho)?
                    .narrow(3, kx / s, wo)?;
                acc = v
                    .broadcast_mul(&self.taps[ky * k + kx])?
                    .broadcast_add(&acc)?;
            }
        }
        Ok(acc)
    }
}

/// 1x1 stride-1 conv as a matmul; candle's conv2d would im2col (copy) the whole input first.
#[derive(Debug, Clone)]
struct Pointwise {
    /// `(out_c, in_c)`
    w: Tensor,
    /// `cpu_direct::pack_weights` layout, when the AVX2 CPU kernel can run this conv.
    cpu_packed: Option<Tensor>,
    /// `(1, out_c, 1)`
    b: Tensor,
}

impl Module for Pointwise {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let out = self
            .w
            .broadcast_matmul(&xs.contiguous()?.reshape((b, c, h * w))?)?
            .broadcast_add(&self.b)?;
        out.reshape((b, self.w.dim(0)?, h, w))
    }
}

/// Dense k*k conv as cols-last im2col + matmul, avoiding candle's uncoalesced im2col and output transpose.
#[derive(Debug, Clone)]
struct Dense {
    conv: Conv2d,
    /// `(out_c, in_c*k*k + 1)`: the last column is the bias, matched by im2col's ones row.
    w2d: Tensor,
    /// `cpu_direct::pack_weights` layout, when the AVX2 CPU kernel can run this conv.
    cpu_packed: Option<Tensor>,
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl Dense {
    fn new(w: Tensor, b: Tensor, kernel: usize, stride: usize, padding: usize) -> Result<Self> {
        let (o, i, _, _) = w.dims4()?;
        let w2d = Tensor::cat(
            &[w.reshape((o, i * kernel * kernel))?, b.reshape((o, 1))?],
            1,
        )?;
        let cfg = Conv2dConfig {
            padding,
            stride,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let cpu_packed = if w.device().is_cpu()
            && crate::cpu_direct::available()
            && o.is_multiple_of(DIRECT_OC_MULTIPLE)
        {
            Some(crate::cpu_direct::pack_weights(&w)?)
        } else {
            None
        };
        Ok(Self {
            conv: Conv2d::new(w, Some(b), cfg),
            w2d,
            cpu_packed,
            kernel,
            stride,
            padding,
        })
    }
}

impl Module for Dense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.device().is_metal() {
            return self.conv.forward(xs);
        }
        if xs.device().is_cpu() && xs.dim(1)? >= IMPLICIT_MIN_IN_C {
            let bias = self
                .conv
                .bias()
                .expect("dense conv is built with a folded bias");
            return crate::cpu_conv::conv2d(
                xs,
                self.conv.weight(),
                bias,
                self.stride,
                self.padding,
            );
        }
        let (b, _, h, w) = xs.dims4()?;
        let ho = (h + 2 * self.padding - self.kernel) / self.stride + 1;
        let wo = (w + 2 * self.padding - self.kernel) / self.stride + 1;
        let cols = crate::im2col::im2col(xs, self.kernel, self.stride, self.padding)?;
        self.w2d
            .broadcast_matmul(&cols)?
            .reshape((b, self.w2d.dim(0)?, ho, wo))
    }
}

#[derive(Debug, Clone)]
enum Conv {
    Dense(Dense),
    Depthwise(Depthwise),
    Pointwise(Pointwise),
}

/// Conv2d with an inference-mode BatchNorm folded into its weight and bias at load time.
#[derive(Debug, Clone)]
pub struct ConvNorm {
    conv: Conv,
    act: Option<Activation>,
}

impl ConvNorm {
    /// The AVX2 CPU kernels fuse bias + activation; returns `None` when this conv/device has no fused path.
    fn forward_fused_cpu(&self, xs: &Tensor) -> Result<Option<Tensor>> {
        use crate::cpu_direct::{self, Act};
        if !xs.device().is_cpu() || !cpu_direct::available() {
            return Ok(None);
        }
        let Some(act) = Act::from_candle(self.act) else {
            return Ok(None);
        };
        match &self.conv {
            Conv::Dense(Dense {
                cpu_packed: Some(packed),
                conv,
                stride,
                padding,
                ..
            }) => {
                let bias = conv.bias().expect("dense conv is built with a folded bias");
                cpu_direct::conv2d(xs, packed, bias, *stride, *padding, act).map(Some)
            }
            Conv::Depthwise(c) => {
                cpu_direct::depthwise(xs, &c.w, &c.b, c.stride, c.padding, act).map(Some)
            }
            Conv::Pointwise(Pointwise {
                cpu_packed: Some(packed),
                b,
                ..
            }) => cpu_direct::pointwise(xs, packed, &b.flatten_all()?, act).map(Some),
            _ => Ok(None),
        }
    }
}

impl Module for ConvNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let Some(y) = self.forward_fused_cpu(xs)? {
            return Ok(y);
        }
        let xs = match &self.conv {
            Conv::Dense(c) => c.forward(xs)?,
            Conv::Depthwise(c) => c.forward(xs)?,
            Conv::Pointwise(c) => c.forward(xs)?,
        };
        match self.act {
            Some(act) => act.forward(&xs),
            None => Ok(xs),
        }
    }
}

/// Defaults to stride 1, groups 1, `(k-1)/2` padding, no activation, BN eps 1e-5 and `convolution`/`normalization` keys.
pub struct ConvNormSpec {
    in_c: usize,
    out_c: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    act: Option<Activation>,
    names: (&'static str, &'static str),
    eps: f64,
}

impl ConvNormSpec {
    pub fn new(in_c: usize, out_c: usize, kernel: usize) -> Self {
        Self {
            in_c,
            out_c,
            kernel,
            stride: 1,
            padding: (kernel - 1) / 2,
            groups: 1,
            act: None,
            names: HF_CONV,
            eps: BN_EPS,
        }
    }

    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    pub fn groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    pub fn act(mut self, act: Activation) -> Self {
        self.act = Some(act);
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn names(mut self, names: (&'static str, &'static str)) -> Self {
        self.names = names;
        self
    }

    pub fn load(self, vb: VarBuilder) -> Result<ConvNorm> {
        let (w, b) = self.folded(vb)?;
        self.build(w, b)
    }

    /// RepVGG re-parameterization: `self(x) + one_by_one(x)` as a single conv (1x1 added into the centre tap).
    pub fn load_with_1x1(
        self,
        one_by_one: ConvNormSpec,
        vb: VarBuilder,
        vb_1x1: VarBuilder,
    ) -> Result<ConvNorm> {
        if one_by_one.kernel != 1
            || self.groups != 1
            || self.kernel.is_multiple_of(2)
            || self.stride != one_by_one.stride
        {
            candle_core::bail!(
                "RepVGG merge needs an odd k*k conv and a 1x1 conv with the same stride"
            );
        }
        let (w, b) = self.folded(vb)?;
        let (w1, b1) = one_by_one.folded(vb_1x1)?;
        let pad = (self.kernel - 1) / 2;
        let w1 = w1
            .pad_with_zeros(2, pad, pad)?
            .pad_with_zeros(3, pad, pad)?;
        self.build((w + w1)?, (b + b1)?)
    }

    /// Conv weight and bias with the BatchNorm folded in.
    fn folded(&self, vb: VarBuilder) -> Result<(Tensor, Tensor)> {
        let w = vb.pp(self.names.0).get(
            (
                self.out_c,
                self.in_c / self.groups,
                self.kernel,
                self.kernel,
            ),
            "weight",
        )?;
        let bn = vb.pp(self.names.1);
        let gamma = bn.get(self.out_c, "weight")?;
        let beta = bn.get(self.out_c, "bias")?;
        let mean = bn.get(self.out_c, "running_mean")?;
        let var = bn.get(self.out_c, "running_var")?;
        let scale = gamma.div(&(var + self.eps)?.sqrt()?)?;
        let w = w.broadcast_mul(&scale.reshape((self.out_c, 1, 1, 1))?)?;
        let b = (beta - mean.mul(&scale)?)?;
        Ok((w, b))
    }

    fn build(self, w: Tensor, b: Tensor) -> Result<ConvNorm> {
        let conv = if self.groups > 1 && self.groups == self.in_c && self.in_c == self.out_c {
            Conv::Depthwise(Depthwise::new(
                &w,
                &b,
                self.kernel,
                self.stride,
                self.padding,
            )?)
        } else if self.kernel == 1 && self.stride == 1 && self.groups == 1 && self.padding == 0 {
            let cpu_packed = if w.device().is_cpu()
                && crate::cpu_direct::available()
                && self.out_c.is_multiple_of(DIRECT_OC_MULTIPLE)
            {
                Some(crate::cpu_direct::pack_weights(&w)?)
            } else {
                None
            };
            Conv::Pointwise(Pointwise {
                w: w.reshape((self.out_c, self.in_c))?,
                b: b.reshape((1, self.out_c, 1))?,
                cpu_packed,
            })
        } else if self.groups == 1 {
            Conv::Dense(Dense::new(w, b, self.kernel, self.stride, self.padding)?)
        } else {
            candle_core::bail!("grouped conv with groups={} is not supported", self.groups);
        };
        Ok(ConvNorm {
            conv,
            act: self.act,
        })
    }
}

/// HF `MLPPredictionHead`: `layers.{i}` linears with ReLU between.
#[derive(Debug, Clone)]
pub struct MlpHead {
    layers: Vec<Linear>,
}

impl MlpHead {
    pub fn new(
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        n: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("layers");
        let layers = (0..n)
            .map(|i| {
                let i_d = if i == 0 { in_dim } else { hidden };
                let o_d = if i == n - 1 { out_dim } else { hidden };
                candle_nn::linear(i_d, o_d, vb.pp(i))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
}

impl Module for MlpHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for (i, l) in self.layers.iter().enumerate() {
            xs = l.forward(&xs)?;
            if i + 1 < self.layers.len() {
                xs = xs.relu()?;
            }
        }
        Ok(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    #[test]
    fn dense_matches_conv() -> Result<()> {
        let dev = Device::Cpu;
        for (k, s, p, h, w) in [(3, 1, 1, 9, 7), (3, 2, 1, 10, 11), (2, 1, 0, 6, 5)] {
            let x = Tensor::randn(0f32, 1., (2, 4, h, w), &dev)?;
            let wt = Tensor::randn(0f32, 1., (3, 4, k, k), &dev)?;
            let b = Tensor::randn(0f32, 1., 3, &dev)?;
            let want = x
                .conv2d(&wt, p, s, 1, 1)?
                .broadcast_add(&b.reshape((1, 3, 1, 1))?)?;
            let mk = |d: &Device| Dense::new(wt.to_device(d)?, b.to_device(d)?, k, s, p);
            let got = mk(&dev)?.forward(&x)?;
            let err = max_abs(&got, &want)?;
            assert!(err < 1e-4, "k={k} s={s} err={err}");
            #[cfg(feature = "cuda")]
            {
                let cuda = Device::new_cuda(0)?;
                let got = mk(&cuda)?.forward(&x.to_device(&cuda)?)?.to_device(&dev)?;
                let err = max_abs(&got, &want)?;
                assert!(err < 1e-4, "cuda k={k} s={s} err={err}");
            }
        }
        Ok(())
    }

    #[test]
    fn pointwise_matches_conv() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1., (2, 5, 7, 9), &dev)?;
        let wt = Tensor::randn(0f32, 1., (3, 5, 1, 1), &dev)?;
        let b = Tensor::randn(0f32, 1., 3, &dev)?;
        let want = x
            .conv2d(&wt, 0, 1, 1, 1)?
            .broadcast_add(&b.reshape((1, 3, 1, 1))?)?;
        let got = Pointwise {
            w: wt.reshape((3, 5))?,
            b: b.reshape((1, 3, 1))?,
            cpu_packed: None,
        }
        .forward(&x)?;
        let err = (got - want)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        assert!(err < 1e-4, "err={err}");
        Ok(())
    }

    #[test]
    fn depthwise_matches_grouped_conv() -> Result<()> {
        let dev = Device::Cpu;
        for (k, s, h, w) in [
            (3, 1, 9, 7),
            (3, 2, 10, 10),
            (3, 2, 9, 11),
            (5, 1, 8, 13),
            (5, 2, 7, 7),
        ] {
            let c = 6;
            let x = Tensor::randn(0f32, 1., (2, c, h, w), &dev)?;
            let wt = Tensor::randn(0f32, 1., (c, 1, k, k), &dev)?;
            let b = Tensor::randn(0f32, 1., c, &dev)?;
            let p = (k - 1) / 2;
            let want = x
                .conv2d(&wt, p, s, 1, c)?
                .broadcast_add(&b.reshape((1, c, 1, 1))?)?;
            let dw = Depthwise::new(&wt, &b, k, s, p)?;
            for (name, got) in [("op", dw.forward(&x)?), ("taps", dw.forward_taps(&x)?)] {
                assert_eq!(got.dims(), want.dims(), "{name} k={k} s={s} {h}x{w}");
                let err = max_abs(&got, &want)?;
                assert!(err < 1e-4, "{name} k={k} s={s} {h}x{w} err={err}");
            }
            #[cfg(feature = "cuda")]
            {
                let cuda = Device::new_cuda(0)?;
                let dw = Depthwise::new(&wt.to_device(&cuda)?, &b.to_device(&cuda)?, k, s, p)?;
                let got = dw.forward(&x.to_device(&cuda)?)?.to_device(&dev)?;
                let err = max_abs(&got, &want)?;
                assert!(err < 1e-4, "cuda k={k} s={s} {h}x{w} err={err}");
            }
        }
        Ok(())
    }

    fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()
    }
}
