use inference_tensor::nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder,
    linear, ops,
};
use inference_tensor::{
    CpuStorage, CustomOp1, D, DType, Device, IndexOp, Layout, Result, Shape, Tensor,
};
use rayon::prelude::*;

use super::config::IstftNetConfig;
use super::dsp::{self, SourceNoise};
use crate::weight_norm::{Snake1d, conv_transpose1d_weight_norm, conv1d, conv1d_weight_norm};

const NORM_EPS: f64 = 1e-5;
const RES_SLOPE: f32 = 0.2;
const GENERATOR_SLOPE: f64 = 0.1;
// F.leaky_relu's default, before conv_post
const POST_SLOPE: f64 = 0.01;
// the reference hardcodes the decoder widths rather than reading them from the config
const DECODER_DIM: usize = 1024;
const ASR_RES_DIM: usize = 64;
const DECODE_BLOCKS: usize = 4;
const NOISE_RES_KERNEL: usize = 7;
const LAST_NOISE_RES_KERNEL: usize = 11;
const NOISE_RES_DILATIONS: [usize; 3] = [1, 3, 5];

/// Instance norm without affine (the release stores no InstanceNorm1d weights), then a style-driven affine.
#[derive(Debug, Clone)]
pub struct AdaIn1d {
    fc: Linear,
    channels: usize,
}

impl AdaIn1d {
    pub fn new(style_dim: usize, channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc: linear(style_dim, 2 * channels, vb.pp("fc"))?,
            channels,
        })
    }

    /// AdaIN then `act`; on the CPU one pass per channel row instead of a dozen tensor ops.
    pub fn forward_act(&self, xs: &Tensor, s: &Tensor, act: &Activation) -> Result<Tensor> {
        // the fused pass needs one style row per batch item; anything else keeps the broadcasting tensor path
        if !xs.device().is_cpu() || xs.dtype() != DType::F32 || s.dim(0)? != xs.dim(0)? {
            return act.forward(&self.forward(xs, s)?);
        }
        let h = self.fc.forward(s)?.to_dtype(DType::F32)?;
        let c = self.channels;
        let scale = (h.narrow(1, 0, c)? + 1.)?.flatten_all()?.to_vec1::<f32>()?;
        let shift = h.narrow(1, c, c)?.flatten_all()?.to_vec1::<f32>()?;
        xs.contiguous()?
            .apply_op1_no_bwd(&FusedAdaIn { scale, shift, act })
    }

    pub fn forward(&self, xs: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?.unsqueeze(2)?;
        let gamma = (h.narrow(1, 0, self.channels)? + 1.)?;
        let beta = h.narrow(1, self.channels, self.channels)?;
        let mean = xs.mean_keepdim(D::Minus1)?;
        let centered = xs.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&(var + NORM_EPS)?.sqrt()?)?;
        normed.broadcast_mul(&gamma)?.broadcast_add(&beta)
    }
}

/// What follows an AdaIN: Snake (per-channel alpha) or a leaky ReLU.
#[derive(Debug, Clone)]
pub enum Activation {
    Snake {
        module: Snake1d,
        alpha: Vec<f32>,
        inv_alpha: Vec<f32>,
    },
    LeakyRelu(f32),
}

impl Activation {
    fn snake(alpha: Tensor) -> Result<Self> {
        let module = Snake1d::new(alpha.clone(), 0.)?;
        let alpha = alpha
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let inv_alpha = alpha.iter().map(|a| a.recip()).collect();
        Ok(Self::Snake {
            module,
            alpha,
            inv_alpha,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Snake { module, .. } => module.forward(xs),
            Self::LeakyRelu(slope) => ops::leaky_relu(xs, f64::from(*slope)),
        }
    }
}

/// Instance norm, the per-(batch, channel) affine and the activation over contiguous F32 rows.
struct FusedAdaIn<'a> {
    scale: Vec<f32>,
    shift: Vec<f32>,
    act: &'a Activation,
}

impl CustomOp1 for FusedAdaIn<'_> {
    fn name(&self) -> &'static str {
        "kokoro-fused-adain"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(data) = storage else {
            inference_tensor::bail!("fused AdaIN takes F32")
        };
        let (_, c, t) = layout.shape().dims3()?;
        let start = layout.start_offset();
        let src = &data[start..start + layout.shape().elem_count()];
        let mut out = vec![0f32; src.len()];
        out.par_chunks_mut(t)
            .zip(src.par_chunks(t))
            .enumerate()
            .for_each(|(row, (dst, x))| {
                let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / t as f64;
                let var = x
                    .iter()
                    .map(|&v| (f64::from(v) - mean).powi(2))
                    .sum::<f64>()
                    / t as f64;
                let inv_std = (1. / (var + NORM_EPS).sqrt()) as f32;
                let (mean, scale, shift) = (mean as f32, self.scale[row], self.shift[row]);
                for (d, &v) in dst.iter_mut().zip(x) {
                    let y = (v - mean) * inv_std * scale + shift;
                    *d = match self.act {
                        Activation::Snake {
                            alpha, inv_alpha, ..
                        } => {
                            let s = (alpha[row % c] * y).sin();
                            y + inv_alpha[row % c] * s * s
                        }
                        Activation::LeakyRelu(slope) => {
                            if y < 0. {
                                y * slope
                            } else {
                                y
                            }
                        }
                    };
                }
            });
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }
}

/// The depthwise transposed conv (kernel 3, stride 2, padding 1, output padding 1) as two interleaved phases.
#[derive(Debug, Clone)]
struct DepthwiseUp2 {
    taps: [Tensor; 3],
    bias: Tensor,
}

impl DepthwiseUp2 {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 1,
            output_padding: 1,
            stride: 2,
            dilation: 1,
            groups: channels,
        };
        let conv = conv_transpose1d_weight_norm(channels, channels, 3, true, cfg, vb)?;
        let w = conv.weight();
        let tap = |k: usize| w.i((.., 0, k))?.reshape((1, channels, 1));
        Ok(Self {
            taps: [tap(0)?, tap(1)?, tap(2)?],
            bias: conv.bias().expect("bias").reshape((1, channels, 1))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, t) = xs.dims3()?;
        // even outputs see tap 1 of x[m]; odd ones tap 2 of x[m] and tap 0 of x[m + 1]
        let even = xs.broadcast_mul(&self.taps[1])?;
        let next = Tensor::cat(
            &[xs.narrow(2, 1, t - 1)?, xs.zeros_like()?.narrow(2, 0, 1)?],
            2,
        )?;
        let odd = (xs.broadcast_mul(&self.taps[2])? + next.broadcast_mul(&self.taps[0])?)?;
        Tensor::stack(&[even, odd], 3)?
            .reshape((b, c, 2 * t))?
            .broadcast_add(&self.bias)
    }
}

fn upsample_nearest2(xs: &Tensor) -> Result<Tensor> {
    let (b, c, t) = xs.dims3()?;
    xs.unsqueeze(3)?
        .broadcast_as((b, c, t, 2))?
        .contiguous()?
        .reshape((b, c, 2 * t))
}

#[derive(Debug, Clone)]
pub struct AdainResBlk1d {
    norm1: AdaIn1d,
    norm2: AdaIn1d,
    conv1: Conv1d,
    conv2: Conv1d,
    conv1x1: Option<Conv1d>,
    pool: Option<DepthwiseUp2>,
}

impl AdainResBlk1d {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let k3 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            norm1: AdaIn1d::new(style_dim, dim_in, vb.pp("norm1"))?,
            norm2: AdaIn1d::new(style_dim, dim_out, vb.pp("norm2"))?,
            conv1: conv1d_weight_norm(dim_in, dim_out, 3, true, k3, vb.pp("conv1"))?,
            conv2: conv1d_weight_norm(dim_out, dim_out, 3, true, k3, vb.pp("conv2"))?,
            conv1x1: (dim_in != dim_out)
                .then(|| {
                    conv1d_weight_norm(
                        dim_in,
                        dim_out,
                        1,
                        false,
                        Default::default(),
                        vb.pp("conv1x1"),
                    )
                })
                .transpose()?,
            pool: upsample
                .then(|| DepthwiseUp2::new(dim_in, vb.pp("pool")))
                .transpose()?,
        })
    }

    pub fn forward(&self, xs: &Tensor, s: &Tensor) -> Result<Tensor> {
        let act = Activation::LeakyRelu(RES_SLOPE);
        let mut res = self.norm1.forward_act(xs, s, &act)?;
        if let Some(pool) = &self.pool {
            res = pool.forward(&res)?;
        }
        let res = self
            .norm2
            .forward_act(&self.conv1.forward(&res)?, s, &act)?;
        let res = self.conv2.forward(&res)?;
        let mut short = if self.pool.is_some() {
            upsample_nearest2(xs)?
        } else {
            xs.clone()
        };
        if let Some(conv) = &self.conv1x1 {
            short = conv.forward(&short)?;
        }
        (res + short)? * std::f64::consts::FRAC_1_SQRT_2
    }
}

/// HiFi-GAN's ResBlock1 with AdaIN and Snake.
#[derive(Debug, Clone)]
struct AdaInResBlock1 {
    layers: Vec<(AdaIn1d, Activation, Conv1d, AdaIn1d, Activation, Conv1d)>,
}

impl AdaInResBlock1 {
    fn new(
        channels: usize,
        kernel: usize,
        dilations: &[usize],
        style_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let snake = |name: &str, j: usize| -> Result<Activation> {
            Activation::snake(vb.pp(name).get((1, channels, 1), &j.to_string())?)
        };
        let layers = dilations
            .iter()
            .enumerate()
            .map(|(j, &d)| {
                let dilated = Conv1dConfig {
                    padding: (kernel * d - d) / 2,
                    dilation: d,
                    ..Default::default()
                };
                let plain = Conv1dConfig {
                    padding: (kernel - 1) / 2,
                    ..Default::default()
                };
                Ok((
                    AdaIn1d::new(style_dim, channels, vb.pp("adain1").pp(j))?,
                    snake("alpha1", j)?,
                    conv1d_weight_norm(
                        channels,
                        channels,
                        kernel,
                        true,
                        dilated,
                        vb.pp("convs1").pp(j),
                    )?,
                    AdaIn1d::new(style_dim, channels, vb.pp("adain2").pp(j))?,
                    snake("alpha2", j)?,
                    conv1d_weight_norm(
                        channels,
                        channels,
                        kernel,
                        true,
                        plain,
                        vb.pp("convs2").pp(j),
                    )?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for (n1, a1, c1, n2, a2, c2) in &self.layers {
            let xt = c1.forward(&n1.forward_act(&xs, s, a1)?)?;
            let xt = c2.forward(&n2.forward_act(&xt, s, a2)?)?;
            xs = (xt + xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Generator {
    source_weights: Vec<f32>,
    source_bias: f32,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<AdaInResBlock1>,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AdaInResBlock1>,
    conv_post: Conv1d,
    kernels: usize,
    n_fft: usize,
    hop: usize,
    upsample_scale: usize,
}

impl Generator {
    fn new(cfg: &IstftNetConfig, style_dim: usize, vb: VarBuilder) -> Result<Self> {
        let source = linear(dsp::HARMONICS, 1, vb.pp("m_source.l_linear"))?;
        let source_weights = source.weight().flatten_all()?.to_vec1::<f32>()?;
        let source_bias = source.bias().expect("bias").to_vec1::<f32>()?[0];
        let stages = cfg.upsample_rates.len();
        let spec_channels = cfg.gen_istft_n_fft + 2;
        let (mut noise_convs, mut noise_res, mut ups, mut resblocks) =
            (vec![], vec![], vec![], vec![]);
        for (i, (&u, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(&cfg.upsample_kernel_sizes)
            .enumerate()
        {
            let (c_in, c_out) = (
                cfg.upsample_initial_channel >> i,
                cfg.upsample_initial_channel >> (i + 1),
            );
            let up = ConvTranspose1dConfig {
                padding: (k - u) / 2,
                stride: u,
                ..Default::default()
            };
            ups.push(conv_transpose1d_weight_norm(
                c_in,
                c_out,
                k,
                true,
                up,
                vb.pp("ups").pp(i),
            )?);
            for (j, (&rk, rd)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(&cfg.resblock_dilation_sizes)
                .enumerate()
            {
                let index = i * cfg.resblock_kernel_sizes.len() + j;
                resblocks.push(AdaInResBlock1::new(
                    c_out,
                    rk,
                    rd,
                    style_dim,
                    vb.pp("resblocks").pp(index),
                )?);
            }
            let noise_vb = vb.pp("noise_convs").pp(i);
            noise_convs.push(if i + 1 < stages {
                let stride: usize = cfg.upsample_rates[i + 1..].iter().product();
                let conv = Conv1dConfig {
                    padding: stride.div_ceil(2),
                    stride,
                    ..Default::default()
                };
                conv1d(spec_channels, c_out, 2 * stride, conv, noise_vb)?
            } else {
                conv1d(spec_channels, c_out, 1, Default::default(), noise_vb)?
            });
            let kernel = if i + 1 < stages {
                NOISE_RES_KERNEL
            } else {
                LAST_NOISE_RES_KERNEL
            };
            noise_res.push(AdaInResBlock1::new(
                c_out,
                kernel,
                &NOISE_RES_DILATIONS,
                style_dim,
                vb.pp("noise_res").pp(i),
            )?);
        }
        let last = cfg.upsample_initial_channel >> stages;
        let post = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        Ok(Self {
            source_weights,
            source_bias,
            noise_convs,
            noise_res,
            ups,
            resblocks,
            conv_post: conv1d_weight_norm(last, spec_channels, 7, true, post, vb.pp("conv_post"))?,
            kernels: cfg.resblock_kernel_sizes.len(),
            n_fft: cfg.gen_istft_n_fft,
            hop: cfg.gen_istft_hop_size,
            upsample_scale: cfg.upsample_rates.iter().product::<usize>() * cfg.gen_istft_hop_size,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        s: &Tensor,
        f0: &[f32],
        noise: &mut SourceNoise,
    ) -> Result<Vec<f32>> {
        let device = xs.device();
        let source = dsp::harmonic_source(
            f0,
            self.upsample_scale,
            &self.source_weights,
            self.source_bias,
            noise,
        )?;
        let (har, frames) = dsp::stft(&source, self.n_fft, self.hop);
        let har = Tensor::from_vec(har, (1, self.n_fft + 2, frames), device)?;
        let mut xs = xs.clone();
        for (i, up) in self.ups.iter().enumerate() {
            xs = ops::leaky_relu(&xs, GENERATOR_SLOPE)?;
            let x_source = self.noise_res[i].forward(&self.noise_convs[i].forward(&har)?, s)?;
            xs = up.forward(&xs)?;
            if i + 1 == self.ups.len() {
                // ReflectionPad1d((1, 0))
                xs = Tensor::cat(&[xs.narrow(2, 1, 1)?, xs], 2)?;
            }
            xs = (xs + x_source)?;
            let blocks = &self.resblocks[i * self.kernels..(i + 1) * self.kernels];
            let mut sum = blocks[0].forward(&xs, s)?;
            for block in &blocks[1..] {
                sum = (sum + block.forward(&xs, s)?)?;
            }
            xs = (sum / self.kernels as f64)?;
        }
        let xs = self.conv_post.forward(&ops::leaky_relu(&xs, POST_SLOPE)?)?;
        let spec = xs.i(0)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        Ok(dsp::istft(&spec, self.n_fft, self.hop))
    }
}

#[derive(Debug, Clone)]
pub struct Decoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res: Conv1d,
    generator: Generator,
}

impl Decoder {
    pub fn new(
        dim_in: usize,
        style_dim: usize,
        cfg: &IstftNetConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cat_dim = DECODER_DIM + 2 + ASR_RES_DIM;
        let decode = (0..DECODE_BLOCKS)
            .map(|i| {
                let last = i + 1 == DECODE_BLOCKS;
                let out = if last {
                    cfg.upsample_initial_channel
                } else {
                    DECODER_DIM
                };
                AdainResBlk1d::new(cat_dim, out, style_dim, last, vb.pp("decode").pp(i))
            })
            .collect::<Result<_>>()?;
        let half = Conv1dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            encode: AdainResBlk1d::new(dim_in + 2, DECODER_DIM, style_dim, false, vb.pp("encode"))?,
            decode,
            f0_conv: conv1d_weight_norm(1, 1, 3, true, half, vb.pp("F0_conv"))?,
            n_conv: conv1d_weight_norm(1, 1, 3, true, half, vb.pp("N_conv"))?,
            asr_res: conv1d_weight_norm(
                dim_in,
                ASR_RES_DIM,
                1,
                true,
                Default::default(),
                vb.pp("asr_res.0"),
            )?,
            generator: Generator::new(cfg, style_dim, vb.pp("generator"))?,
        })
    }

    /// `asr` is (1, hidden, frames), `f0` and `n` (1, 2 * frames), `s` the decoder half of the voice style.
    pub fn forward(
        &self,
        asr: &Tensor,
        f0: &Tensor,
        n: &Tensor,
        s: &Tensor,
        noise: &mut SourceNoise,
    ) -> Result<Vec<f32>> {
        let f0_down = self.f0_conv.forward(&f0.unsqueeze(1)?)?;
        let n_down = self.n_conv.forward(&n.unsqueeze(1)?)?;
        let mut xs = self
            .encode
            .forward(&Tensor::cat(&[asr, &f0_down, &n_down], 1)?, s)?;
        let asr_res = self.asr_res.forward(asr)?;
        for block in &self.decode {
            xs = block.forward(&Tensor::cat(&[&xs, &asr_res, &f0_down, &n_down], 1)?, s)?;
        }
        let f0 = f0
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        self.generator.forward(&xs, s, &f0, noise)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_tensor::{DType, Device, Tensor};

    use super::*;

    // The CPU's one-pass AdaIN against the tensor ops it stands in for, with both activations and a batch of two
    #[test]
    fn fused_adain_matches_the_tensor_path() -> Result<()> {
        let (batch, channels, len, style) = (2, 4, 9, 3);
        let dev = Device::Cpu;
        let weights = HashMap::from([
            (
                "fc.weight".to_string(),
                Tensor::randn(0f32, 1., (2 * channels, style), &dev)?,
            ),
            (
                "fc.bias".to_string(),
                Tensor::randn(0f32, 1., 2 * channels, &dev)?,
            ),
        ]);
        let ada = AdaIn1d::new(
            style,
            channels,
            VarBuilder::from_tensors(weights, DType::F32, &dev),
        )?;
        // narrowed from a larger tensor, so the fused op reads from a non-zero offset
        let x = Tensor::randn(0f32, 1., (batch + 1, channels, len), &dev)?.narrow(0, 1, batch)?;
        let s = Tensor::randn(0f32, 1., (batch, style), &dev)?;
        let alpha = (Tensor::rand(0.5f32, 1.5, (1, channels, 1), &dev)?).contiguous()?;
        for act in [Activation::snake(alpha)?, Activation::LeakyRelu(RES_SLOPE)] {
            let want = act.forward(&ada.forward(&x, &s)?)?;
            let got = ada.forward_act(&x, &s, &act)?;
            let diff = (got - want)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-5, "{act:?}: {diff}");
        }
        Ok(())
    }

    // The two-phase form against the grouped transposed conv it replaces
    #[test]
    fn depthwise_up2_matches_the_grouped_transposed_conv() -> Result<()> {
        let (channels, len) = (3, 7);
        let dev = Device::Cpu;
        let v = Tensor::randn(0f32, 1., (channels, 1, 3), &dev)?;
        let weights = HashMap::from([
            ("weight_v".to_string(), v.clone()),
            (
                "weight_g".to_string(),
                Tensor::ones((channels, 1, 1), DType::F32, &dev)?,
            ),
            ("bias".to_string(), Tensor::randn(0f32, 1., channels, &dev)?),
        ]);
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let up = DepthwiseUp2::new(channels, vb)?;
        let x = Tensor::randn(0f32, 1., (2, channels, len), &dev)?;
        let w = v.broadcast_div(&v.sqr()?.sum_keepdim((1, 2))?.sqrt()?)?;
        let want = x
            .conv_transpose1d(&w, 1, 1, 2, 1, channels)?
            .broadcast_add(&weights["bias"].reshape((1, channels, 1))?)?;
        let got = up.forward(&x)?;
        assert_eq!(got.dims(), want.dims());
        let diff = (got - want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "{diff}");
        Ok(())
    }
}
