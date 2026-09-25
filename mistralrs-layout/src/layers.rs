use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, Conv2d, Conv2dConfig, Linear, VarBuilder};

const BN_EPS: f64 = 1e-5;

pub const HF_CONV: (&str, &str) = ("convolution", "normalization");
pub const RTDETR_CONV: (&str, &str) = ("conv", "norm");

/// Depthwise conv as a sum of shifted taps; candle runs `groups=C` as C separate convolutions.
#[derive(Debug, Clone)]
struct Depthwise {
    /// `(1, C, 1, 1)` weight per `(ky, kx)`, row-major.
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
        let (b, c, h, w) = xs.dims4()?;
        let (k, s, p) = (self.kernel, self.stride, self.padding);
        let ho = (h + 2 * p - k) / s + 1;
        let wo = (w + 2 * p - k) / s + 1;
        // pad each spatial dim to a multiple of the stride so strided taps become a phase select + narrow
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

#[derive(Debug, Clone)]
enum Conv {
    Dense(Conv2d),
    Depthwise(Depthwise),
}

/// Conv2d with an inference-mode BatchNorm folded into its weight and bias at load time.
#[derive(Debug, Clone)]
pub struct ConvNorm {
    conv: Conv,
    act: Option<Activation>,
}

impl Module for ConvNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = match &self.conv {
            Conv::Dense(c) => c.forward(xs)?,
            Conv::Depthwise(c) => c.forward(xs)?,
        };
        match self.act {
            Some(act) => act.forward(&xs),
            None => Ok(xs),
        }
    }
}

/// Defaults to stride 1, groups 1, `(k-1)/2` padding, no activation and `convolution`/`normalization` keys.
pub struct ConvNormSpec {
    in_c: usize,
    out_c: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    act: Option<Activation>,
    names: (&'static str, &'static str),
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

    pub fn names(mut self, names: (&'static str, &'static str)) -> Self {
        self.names = names;
        self
    }

    pub fn load(self, vb: VarBuilder) -> Result<ConvNorm> {
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
        let scale = gamma.div(&(var + BN_EPS)?.sqrt()?)?;
        let w = w.broadcast_mul(&scale.reshape((self.out_c, 1, 1, 1))?)?;
        let b = (beta - mean.mul(&scale)?)?;
        let conv = if self.groups > 1 && self.groups == self.in_c && self.in_c == self.out_c {
            Conv::Depthwise(Depthwise::new(
                &w,
                &b,
                self.kernel,
                self.stride,
                self.padding,
            )?)
        } else {
            let cfg = Conv2dConfig {
                padding: self.padding,
                stride: self.stride,
                dilation: 1,
                groups: self.groups,
                cudnn_fwd_algo: None,
            };
            Conv::Dense(Conv2d::new(w, Some(b), cfg))
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
            let got = Depthwise::new(&wt, &b, k, s, p)?.forward(&x)?;
            assert_eq!(got.dims(), want.dims(), "k={k} s={s} {h}x{w}");
            let err = (got - want)?
                .abs()?
                .flatten_all()?
                .max(D::Minus1)?
                .to_scalar::<f32>()?;
            assert!(err < 1e-4, "k={k} s={s} {h}x{w} err={err}");
        }
        Ok(())
    }
}
