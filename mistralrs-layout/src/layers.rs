use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, Conv2d, Conv2dConfig, Linear, VarBuilder};

const BN_EPS: f64 = 1e-5;
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
    act: Option<Activation>,
}

impl Depthwise {
    fn new(
        w: &Tensor,
        bias: &Tensor,
        geom: (usize, usize, usize),
        act: Option<Activation>,
    ) -> Result<Self> {
        let (kernel, stride, padding) = geom;
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
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let Some(act) = cpu_fused(xs.device(), self.act) {
            return crate::cpu_direct::depthwise(
                xs,
                &self.w,
                &self.b,
                self.stride,
                self.padding,
                act,
            );
        }
        let y = if xs.device().is_metal() {
            self.forward_taps(xs)?
        } else {
            crate::depthwise::depthwise_conv2d(xs, &self.w, &self.b, self.stride, self.padding)?
        };
        activate(y, self.act)
    }

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

/// Applies `act` unless the kernel already fused it.
fn activate(xs: Tensor, act: Option<Activation>) -> Result<Tensor> {
    match act {
        Some(act) => act.forward(&xs),
        None => Ok(xs),
    }
}

/// `Some(act)` when the AVX2 CPU kernels can fuse this activation on this device.
fn cpu_fused(dev: &candle_core::Device, act: Option<Activation>) -> Option<crate::cpu_direct::Act> {
    if dev.is_cpu() && crate::cpu_direct::available() {
        crate::cpu_direct::Act::from_candle(act)
    } else {
        None
    }
}

/// 1x1 stride-1 conv as a matmul; candle's conv2d would im2col (copy) the whole input first.
#[derive(Debug, Clone)]
struct Pointwise {
    /// `(out_c, in_c)` for the GEMM path; `None` when the AVX2 kernel owns this conv.
    w: Option<Tensor>,
    /// `cpu_direct::pack_weights` layout for the AVX2 kernel.
    cpu_packed: Option<Tensor>,
    /// `(out_c,)`
    b: Tensor,
    act: Option<Activation>,
}

impl Pointwise {
    fn new(w: &Tensor, b: &Tensor, act: Option<Activation>) -> Result<Self> {
        let (o, c, _, _) = w.dims4()?;
        let fused =
            cpu_fused(w.device(), act).is_some() && o.is_multiple_of(crate::cpu_direct::OC_T);
        Ok(Self {
            w: (!fused).then(|| w.reshape((o, c))).transpose()?,
            cpu_packed: fused
                .then(|| crate::cpu_direct::pack_weights(w))
                .transpose()?,
            b: b.clone(),
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let (Some(packed), Some(act)) = (&self.cpu_packed, cpu_fused(xs.device(), self.act)) {
            return crate::cpu_direct::pointwise(xs, packed, &self.b, act);
        }
        let w = self
            .w
            .as_ref()
            .expect("pointwise keeps GEMM weights unless the AVX2 kernel owns it");
        let (b, c, h, hw) = xs.dims4()?;
        let o = w.dim(0)?;
        let out = w
            .broadcast_matmul(&xs.contiguous()?.reshape((b, c, h * hw))?)?
            .broadcast_add(&self.b.reshape((1, o, 1))?)?;
        activate(out.reshape((b, o, h, hw))?, self.act)
    }
}

/// Dense k*k conv. Each device keeps only the weight layout its backend uses.
#[derive(Debug, Clone)]
struct Dense {
    /// Metal, and CPU without the AVX2 kernel (implicit GEMM).
    conv: Option<Conv2d>,
    /// `(out_c, in_c*k*k + 1)`, the last column being the bias matched by im2col's ones row: CUDA, and thin CPU inputs.
    w2d: Option<Tensor>,
    /// `cpu_direct::pack_weights` layout for the AVX2 kernel.
    cpu_packed: Option<Tensor>,
    b: Tensor,
    kernel: usize,
    stride: usize,
    padding: usize,
    act: Option<Activation>,
}

impl Dense {
    fn new(
        w: Tensor,
        b: Tensor,
        geom: (usize, usize, usize),
        act: Option<Activation>,
    ) -> Result<Self> {
        let (kernel, stride, padding) = geom;
        let (o, i, _, _) = w.dims4()?;
        let dev = w.device().clone();
        let fused = cpu_fused(&dev, act).is_some() && o.is_multiple_of(crate::cpu_direct::OC_T);
        let cpu_packed = fused
            .then(|| crate::cpu_direct::pack_weights(&w))
            .transpose()?;
        let needs_w2d = !fused && (dev.is_cuda() || (dev.is_cpu() && i < IMPLICIT_MIN_IN_C));
        let w2d = needs_w2d
            .then(|| {
                Tensor::cat(
                    &[w.reshape((o, i * kernel * kernel))?, b.reshape((o, 1))?],
                    1,
                )
            })
            .transpose()?;
        let needs_conv = !fused && (dev.is_metal() || (dev.is_cpu() && i >= IMPLICIT_MIN_IN_C));
        let cfg = Conv2dConfig {
            padding,
            stride,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        Ok(Self {
            conv: needs_conv.then(|| Conv2d::new(w, Some(b.clone()), cfg)),
            w2d,
            cpu_packed,
            b,
            kernel,
            stride,
            padding,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let (Some(packed), Some(act)) = (&self.cpu_packed, cpu_fused(xs.device(), self.act)) {
            return crate::cpu_direct::conv2d(xs, packed, &self.b, self.stride, self.padding, act);
        }
        if let Some(conv) = &self.conv {
            let y = if xs.device().is_cpu() {
                crate::cpu_conv::conv2d(xs, conv.weight(), &self.b, self.stride, self.padding)?
            } else {
                conv.forward(xs)?
            };
            return activate(y, self.act);
        }
        let w2d = self
            .w2d
            .as_ref()
            .expect("dense conv keeps a weight layout for its device");
        let (b, _, h, w) = xs.dims4()?;
        let ho = (h + 2 * self.padding - self.kernel) / self.stride + 1;
        let wo = (w + 2 * self.padding - self.kernel) / self.stride + 1;
        let cols = crate::im2col::im2col(xs, self.kernel, self.stride, self.padding)?;
        activate(
            w2d.broadcast_matmul(&cols)?
                .reshape((b, w2d.dim(0)?, ho, wo))?,
            self.act,
        )
    }
}

#[derive(Debug, Clone)]
enum Conv {
    Dense(Dense),
    Depthwise(Depthwise),
    Pointwise(Pointwise),
}

/// Conv2d with an inference-mode BatchNorm folded into its weight and bias at load time; the activation lives in the
/// conv variant so each backend can fuse it.
#[derive(Debug, Clone)]
pub struct ConvNorm {
    conv: Conv,
}

impl Module for ConvNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.conv {
            Conv::Dense(c) => c.forward(xs),
            Conv::Depthwise(c) => c.forward(xs),
            Conv::Pointwise(c) => c.forward(xs),
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
        let geom = (self.kernel, self.stride, self.padding);
        let conv = if self.groups > 1 && self.groups == self.in_c && self.in_c == self.out_c {
            Conv::Depthwise(Depthwise::new(&w, &b, geom, self.act)?)
        } else if self.kernel == 1 && self.stride == 1 && self.groups == 1 && self.padding == 0 {
            Conv::Pointwise(Pointwise::new(&w, &b, self.act)?)
        } else if self.groups == 1 {
            Conv::Dense(Dense::new(w, b, geom, self.act)?)
        } else {
            candle_core::bail!("grouped conv with groups={} is not supported", self.groups);
        };
        Ok(ConvNorm { conv })
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

    // GELU has no fused kernel, so it exercises each variant's unfused path next to the fused ones
    const ACTS: [Option<Activation>; 4] = [
        None,
        Some(Activation::Relu),
        Some(Activation::Silu),
        Some(Activation::Gelu),
    ];

    fn devices() -> Result<Vec<Device>> {
        std::iter::once(Ok(Device::Cpu))
            .chain(cfg!(feature = "cuda").then(|| Device::new_cuda(0)))
            .collect()
    }

    fn reference(
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        conv: (usize, usize, usize),
        act: Option<Activation>,
    ) -> Result<Tensor> {
        let (s, p, groups) = conv;
        let o = w.dim(0)?;
        activate(
            x.conv2d(w, p, s, 1, groups)?
                .broadcast_add(&b.reshape((1, o, 1, 1))?)?,
            act,
        )
    }

    fn rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_device(&Device::Cpu)?;
        let scale = b
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?
            .max(1e-6);
        Ok((a - b)?
            .abs()?
            .flatten_all()?
            .max(D::Minus1)?
            .to_scalar::<f32>()?
            / scale)
    }

    #[test]
    fn dense_dispatch_matches_conv() -> Result<()> {
        let cpu = Device::Cpu;
        // (out, in): 8 outputs take the fused AVX2 kernel; 3 fall back to implicit GEMM (in >= 16) or im2col (in < 16)
        for (o, i) in [(8, 4), (8, 20), (3, 4), (3, 20)] {
            for (k, s, p, h, w) in [(3, 1, 1, 9, 7), (3, 2, 1, 10, 11), (2, 1, 0, 6, 5)] {
                let x = Tensor::randn(0f32, 1., (2, i, h, w), &cpu)?;
                let wt = Tensor::randn(0f32, 1., (o, i, k, k), &cpu)?;
                let b = Tensor::randn(0f32, 1., o, &cpu)?;
                for act in ACTS {
                    let want = reference(&x, &wt, &b, (s, p, 1), act)?;
                    for dev in devices()? {
                        let conv =
                            Dense::new(wt.to_device(&dev)?, b.to_device(&dev)?, (k, s, p), act)?;
                        let err = rel_err(&conv.forward(&x.to_device(&dev)?)?, &want)?;
                        assert!(
                            err < 1e-5,
                            "{dev:?} o={o} i={i} k={k} s={s} {act:?} rel err={err}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn pointwise_dispatch_matches_conv() -> Result<()> {
        let cpu = Device::Cpu;
        for o in [8, 3] {
            let x = Tensor::randn(0f32, 1., (2, 5, 7, 9), &cpu)?;
            let wt = Tensor::randn(0f32, 1., (o, 5, 1, 1), &cpu)?;
            let b = Tensor::randn(0f32, 1., o, &cpu)?;
            for act in ACTS {
                let want = reference(&x, &wt, &b, (1, 0, 1), act)?;
                for dev in devices()? {
                    let conv = Pointwise::new(&wt.to_device(&dev)?, &b.to_device(&dev)?, act)?;
                    let err = rel_err(&conv.forward(&x.to_device(&dev)?)?, &want)?;
                    assert!(err < 1e-5, "{dev:?} o={o} {act:?} rel err={err}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn depthwise_dispatch_matches_grouped_conv() -> Result<()> {
        let cpu = Device::Cpu;
        let c = 6;
        for (k, s, h, w) in [
            (3, 1, 9, 7),
            (3, 2, 10, 10),
            (3, 2, 9, 11),
            (5, 1, 8, 13),
            (5, 2, 7, 7),
        ] {
            let p = (k - 1) / 2;
            let x = Tensor::randn(0f32, 1., (2, c, h, w), &cpu)?;
            let wt = Tensor::randn(0f32, 1., (c, 1, k, k), &cpu)?;
            let b = Tensor::randn(0f32, 1., c, &cpu)?;
            for act in ACTS {
                let want = reference(&x, &wt, &b, (s, p, c), act)?;
                for dev in devices()? {
                    let conv =
                        Depthwise::new(&wt.to_device(&dev)?, &b.to_device(&dev)?, (k, s, p), act)?;
                    let err = rel_err(&conv.forward(&x.to_device(&dev)?)?, &want)?;
                    assert!(
                        err < 1e-5,
                        "{dev:?} k={k} s={s} {h}x{w} {act:?} rel err={err}"
                    );
                }
            }
            // the Metal fallback, checked on CPU
            let conv = Depthwise::new(&wt, &b, (k, s, p), None)?;
            let err = rel_err(
                &conv.forward_taps(&x)?,
                &reference(&x, &wt, &b, (s, p, c), None)?,
            )?;
            assert!(err < 1e-5, "taps k={k} s={s} {h}x{w} rel err={err}");
        }
        Ok(())
    }
}
