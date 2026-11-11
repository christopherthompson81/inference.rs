//! ECAPA-TDNN (SE-Res2Net blocks, multi-layer feature aggregation, attentive statistics pooling) as SpeechBrain
//! writes it: the Qwen3-TTS speaker encoder and the VoxLingua107 language identifier.

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::ops::{sigmoid, softmax_last_dim};
use inference_tensor::{D, DType, Device, Result, Tensor};

const STATS_EPS: f64 = 1e-12;
const BATCH_NORM_EPS: f64 = 1e-5;

/// How a checkpoint names and normalizes the blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcapaLayout {
    /// SpeechBrain's own: each conv wrapped as `<name>.conv`, a BatchNorm after every TDNN block's ReLU and one before
    /// `fc`.
    SpeechBrain,
    /// Qwen3-TTS's port: bare convs, no BatchNorm.
    Bare,
}

#[derive(Clone, Debug)]
pub struct EcapaConfig {
    pub input_dim: usize,
    pub channels: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub dilations: Vec<usize>,
    pub attention_channels: usize,
    pub res2net_scale: usize,
    pub se_channels: usize,
    pub out_dim: usize,
    pub layout: EcapaLayout,
}

/// An eval-mode BatchNorm as a per-channel scale and shift, `(1, c, 1)` each.
pub struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

impl BatchNorm {
    /// `vb` holds the norm's `weight`, `bias`, `running_mean` and `running_var`.
    pub fn new(channels: usize, vb: &ShardedVarBuilder) -> Result<Self> {
        let var = vb.get(channels, "running_var")?;
        let scale = vb
            .get(channels, "weight")?
            .div(&(var + BATCH_NORM_EPS)?.sqrt()?)?;
        let shift = (vb.get(channels, "bias")? - vb.get(channels, "running_mean")?.mul(&scale)?)?;
        Ok(Self {
            scale: scale.reshape((1, (), 1))?,
            shift: shift.reshape((1, (), 1))?,
        })
    }

    /// `xs` is `(b, c, t)`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.scale.to_dtype(xs.dtype())?)?
            .broadcast_add(&self.shift.to_dtype(xs.dtype())?)
    }

    /// `xs` is `(b, c)`.
    pub fn forward_2d(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(&xs.unsqueeze(2)?)?.squeeze(2)
    }
}

// a 'same' conv with reflect padding, as SpeechBrain's `padding="same", padding_mode="reflect"`
struct Conv {
    weight: Tensor,
    bias: Tensor,
    dilation: usize,
    pad: usize,
}

impl Conv {
    // `vb` is the conv module's path; SpeechBrain's wrapper keeps the torch conv one level down
    fn new(
        (in_ch, out_ch, kernel, dilation): (usize, usize, usize, usize),
        layout: EcapaLayout,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let vb = match layout {
            EcapaLayout::SpeechBrain => vb.pp("conv"),
            EcapaLayout::Bare => vb.clone(),
        };
        Ok(Self {
            weight: vb.get((out_ch, in_ch, kernel), "weight")?,
            bias: vb.get(out_ch, "bias")?,
            dilation,
            pad: dilation * (kernel - 1) / 2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.pad > 0 {
            let len = xs.dim(D::Minus1)?;
            let last = len as isize - 1;
            let index: Vec<u32> = (0..len + 2 * self.pad)
                .map(|i| {
                    let i = i as isize - self.pad as isize;
                    let i = if i < 0 {
                        -i
                    } else if i > last {
                        2 * last - i
                    } else {
                        i
                    };
                    u32::try_from(i).unwrap_or(0)
                })
                .collect();
            let index = Tensor::from_vec(index, len + 2 * self.pad, xs.device())?;
            xs.index_select(&index, 2)?
        } else {
            xs.clone()
        };
        xs.conv1d(&self.weight, 0, 1, self.dilation, 1)?
            .broadcast_add(&self.bias.reshape((1, (), 1))?)
    }
}

// conv, ReLU, then (SpeechBrain) BatchNorm
struct Tdnn {
    conv: Conv,
    norm: Option<BatchNorm>,
}

impl Tdnn {
    fn new(
        shape: (usize, usize, usize, usize),
        layout: EcapaLayout,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let norm = match layout {
            EcapaLayout::SpeechBrain => Some(BatchNorm::new(shape.1, &vb.pp("norm").pp("norm"))?),
            EcapaLayout::Bare => None,
        };
        Ok(Self {
            conv: Conv::new(shape, layout, &vb.pp("conv"))?,
            norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv.forward(xs)?.relu()?;
        match &self.norm {
            Some(norm) => norm.forward(&xs),
            None => Ok(xs),
        }
    }
}

struct SeRes2Net {
    tdnn1: Tdnn,
    res2net: Vec<Tdnn>,
    tdnn2: Tdnn,
    se1: Conv,
    se2: Conv,
    scale: usize,
}

impl SeRes2Net {
    fn new(cfg: &EcapaConfig, i: usize, vb: &ShardedVarBuilder) -> Result<Self> {
        let (ch_in, ch) = (cfg.channels[i - 1], cfg.channels[i]);
        let width = ch / cfg.res2net_scale;
        let layout = cfg.layout;
        let se = vb.pp("se_block");
        Ok(Self {
            tdnn1: Tdnn::new((ch_in, ch, 1, 1), layout, &vb.pp("tdnn1"))?,
            res2net: (0..cfg.res2net_scale - 1)
                .map(|j| {
                    let shape = (width, width, cfg.kernel_sizes[i], cfg.dilations[i]);
                    Tdnn::new(shape, layout, &vb.pp("res2net_block").pp("blocks").pp(j))
                })
                .collect::<Result<_>>()?,
            tdnn2: Tdnn::new((ch, ch, 1, 1), layout, &vb.pp("tdnn2"))?,
            se1: Conv::new((ch, cfg.se_channels, 1, 1), layout, &se.pp("conv1"))?,
            se2: Conv::new((cfg.se_channels, ch, 1, 1), layout, &se.pp("conv2"))?,
            scale: cfg.res2net_scale,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.tdnn1.forward(xs)?;
        let parts = h.chunk(self.scale, 1)?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);
        for (i, part) in parts.iter().enumerate() {
            let out = match i {
                0 => part.clone(),
                1 => self.res2net[0].forward(part)?,
                _ => self.res2net[i - 1]
                    .forward(&(part + outputs.last().expect("an earlier part"))?)?,
            };
            outputs.push(out);
        }
        let h = self.tdnn2.forward(&Tensor::cat(&outputs, 1)?)?;
        let gate = self.se1.forward(&h.mean_keepdim(2)?)?.relu()?;
        let gate = sigmoid(&self.se2.forward(&gate)?)?;
        h.broadcast_mul(&gate)? + xs
    }
}

// mean and std over time of `xs` `(1, c, t)` under weights `w` that sum to 1 per channel
fn statistics(xs: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
    let mean = xs.broadcast_mul(w)?.sum_keepdim(2)?;
    let var = xs
        .broadcast_sub(&mean)?
        .sqr()?
        .broadcast_mul(w)?
        .sum_keepdim(2)?;
    Ok((mean, var.clamp(STATS_EPS, f64::INFINITY)?.sqrt()?))
}

pub struct Ecapa {
    first: Tdnn,
    blocks: Vec<SeRes2Net>,
    mfa: Tdnn,
    asp_tdnn: Tdnn,
    asp_conv: Conv,
    asp_norm: Option<BatchNorm>,
    fc: Conv,
}

impl Ecapa {
    pub fn new(cfg: &EcapaConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let ch = &cfg.channels;
        let last = ch.len() - 1;
        let layout = cfg.layout;
        let blocks = (1..last)
            .map(|i| SeRes2Net::new(cfg, i, &vb.pp("blocks").pp(i)))
            .collect::<Result<_>>()?;
        let aggregated = ch[1..last].iter().sum();
        let asp = vb.pp("asp");
        let asp_norm = match layout {
            EcapaLayout::SpeechBrain => {
                Some(BatchNorm::new(2 * ch[last], &vb.pp("asp_bn").pp("norm"))?)
            }
            EcapaLayout::Bare => None,
        };
        Ok(Self {
            first: Tdnn::new(
                (cfg.input_dim, ch[0], cfg.kernel_sizes[0], cfg.dilations[0]),
                layout,
                &vb.pp("blocks").pp(0),
            )?,
            blocks,
            mfa: Tdnn::new(
                (
                    aggregated,
                    ch[last],
                    cfg.kernel_sizes[last],
                    cfg.dilations[last],
                ),
                layout,
                &vb.pp("mfa"),
            )?,
            asp_tdnn: Tdnn::new(
                (3 * ch[last], cfg.attention_channels, 1, 1),
                layout,
                &asp.pp("tdnn"),
            )?,
            asp_conv: Conv::new(
                (cfg.attention_channels, ch[last], 1, 1),
                layout,
                &asp.pp("conv"),
            )?,
            asp_norm,
            fc: Conv::new((2 * ch[last], cfg.out_dim, 1, 1), layout, &vb.pp("fc"))?,
        })
    }

    /// The embedding `(out_dim,)` of `[frames][input_dim]` features.
    pub fn embed(&self, features: &[Vec<f32>], device: &Device, dtype: DType) -> Result<Tensor> {
        let frames = features.len();
        if frames == 0 {
            inference_tensor::bail!("the audio is too short");
        }
        let dim = features[0].len();
        let flat: Vec<f32> = features.iter().flatten().copied().collect();
        let xs = Tensor::from_vec(flat, (1, frames, dim), device)?
            .to_dtype(dtype)?
            .transpose(1, 2)?
            .contiguous()?;
        let mut h = self.first.forward(&xs)?;
        let mut outputs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            h = block.forward(&h)?;
            outputs.push(h.clone());
        }
        let h = self.mfa.forward(&Tensor::cat(&outputs, 1)?)?;
        #[allow(clippy::cast_precision_loss)]
        let uniform = (Tensor::ones((1, 1, frames), h.dtype(), device)? / frames as f64)?;
        let (mean, std) = statistics(&h, &uniform)?;
        let context = Tensor::cat(
            &[
                h.clone(),
                mean.repeat((1, 1, frames))?,
                std.repeat((1, 1, frames))?,
            ],
            1,
        )?;
        let attention = self
            .asp_conv
            .forward(&self.asp_tdnn.forward(&context)?.tanh()?)?;
        let attention = softmax_last_dim(&attention.contiguous()?)?;
        let (mean, std) = statistics(&h, &attention)?;
        let pooled = Tensor::cat(&[mean, std], 1)?;
        let pooled = match &self.asp_norm {
            Some(norm) => norm.forward(&pooled)?,
            None => pooled,
        };
        self.fc.forward(&pooled)?.flatten_all()
    }
}
