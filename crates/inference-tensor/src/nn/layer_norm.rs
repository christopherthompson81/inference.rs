//! Layer Normalization.
//!
//! This layer applies Layer Normalization over a mini-batch of inputs as described in [`Layer
//! Normalization`]. The input is expected to have three dimensions: a batch dimension, a length,
//! and a hidden size, the normalization is applied over the last dimension.
//!
//! # Example
//!
//! ```rust
//! use inference_tensor::{Tensor, Device::Cpu};
//! use inference_tensor::nn::{LayerNorm, Module};
//! # fn main() -> inference_tensor::Result<()> {
//!
//! let w = Tensor::new(&[1f32, 1f32, 1f32], &Cpu)?;
//! let b = Tensor::new(&[0f32, 0f32, 0f32], &Cpu)?;
//! let layer = LayerNorm::new(w, b, 1e-5);
//!
//! let xs = Tensor::new(
//!     &[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]],
//!     &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! let expected = Tensor::new(
//!     &[[[-1.2247f32, 0.0,  1.2247],
//!        [-1.2247, 0.0,  1.2247],
//!        [ 1.2247, 0.0, -1.2247]]],
//!     &Cpu)?;
//! assert!((ys - expected)?.abs()?.max_all()?.to_scalar::<f32>()? < 1e-4);
//! # Ok(()) }
//! ```
//!
//! [`Layer Normalization`]: https://arxiv.org/abs/1607.06450
use crate::{DType, Error, Module, Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNormConfig {
    pub eps: f64,
    /// Whether to remove the mean or not, the default is true and when set to false, this turns
    /// this layer into RmsNorm.
    pub remove_mean: bool,
    pub affine: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        }
    }
}

impl From<f64> for LayerNormConfig {
    fn from(eps: f64) -> Self {
        Self {
            eps,
            remove_mean: true,
            affine: true,
        }
    }
}

// This layer norm version handles both weight and bias so removes the mean.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: Some(bias),
            remove_mean: true,
            eps,
        }
    }

    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            remove_mean: true,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.is_contiguous() && self.remove_mean {
            if let Some(bias) = self.bias.as_ref() {
                return crate::nn::ops::layer_norm(x, &self.weight, bias, self.eps as f32);
            }
        }
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let x = if self.remove_mean {
            let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn layer_norm<C: Into<LayerNormConfig>>(
    size: usize,
    config: C,
    vb: crate::nn::VarBuilder,
) -> Result<LayerNorm> {
    let config = config.into();

    // Convert old format to new format if needed from a PyTorch state_dict
    // Safetensors not always in new weight/bias format
    // https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L575
    let weight_tensor_name = ["weight", "gamma"]
        .iter()
        .find(|&name| vb.contains_tensor(name))
        .ok_or_else(|| Error::Msg("Failed to find weight tensor".into()))?;

    let weight = vb.get_with_hints(size, weight_tensor_name, crate::nn::Init::Const(1.))?;

    let bias_tensor_name = ["bias", "beta"]
        .iter()
        .find(|&name| vb.contains_tensor(name))
        .ok_or_else(|| Error::Msg("Failed to find weight tensor".into()))?;

    let bias = if config.affine {
        Some(vb.get_with_hints(size, bias_tensor_name, crate::nn::Init::Const(0.))?)
    } else {
        None
    };

    Ok(LayerNorm {
        weight,
        bias,
        remove_mean: config.remove_mean,
        eps: config.eps,
    })
}
