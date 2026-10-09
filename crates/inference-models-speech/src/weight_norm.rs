//! Convolutions (weight-normed ones folded at load) and the Snake activation, shared by the speech decoders.

use inference_tensor::nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder,
};
use inference_tensor::{Result, Tensor};

// https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html, dim 0: one norm per leading channel
fn folded(vb: &VarBuilder, lead: usize, rest: (usize, usize)) -> Result<Tensor> {
    let g = vb.get((lead, 1, 1), "weight_g")?;
    let v = vb.get((lead, rest.0, rest.1), "weight_v")?;
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    v.broadcast_mul(&g)?.broadcast_div(&norm)
}

pub fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = folded(&vb, out_c, (in_c / config.groups, kernel_size))?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}

pub fn conv1d(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_c, in_c / config.groups, kernel_size), "weight")?;
    Ok(Conv1d::new(weight, Some(vb.get(out_c, "bias")?), config))
}

pub fn conv_transpose1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    bias: bool,
    config: ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight = folded(&vb, in_c, (out_c / config.groups, kernel_size))?;
    let bias = if bias {
        Some(vb.get(out_c, "bias")?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(weight, bias, config))
}

/// `x + sin(alpha x)^2 / (alpha + eps)` with a per-channel `alpha` of shape (1, channels, 1).
#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor,
    inv_alpha: Tensor,
}

impl Snake1d {
    pub fn new(alpha: Tensor, eps: f64) -> Result<Self> {
        let inv_alpha = (&alpha + eps)?.recip()?;
        Ok(Self { alpha, inv_alpha })
    }
}

impl inference_tensor::Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_shape = xs.shape();
        let xs = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs)?.sin()?;
        (xs + self.inv_alpha.broadcast_mul(&sin.sqr()?)?)?.reshape(xs_shape)
    }
}
