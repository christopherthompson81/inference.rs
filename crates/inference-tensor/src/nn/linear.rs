//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use inference_tensor::{Tensor, Device::Cpu};
//! use inference_tensor::nn::{Linear, Module};
//! # fn main() -> inference_tensor::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use crate::{Result, Tensor};

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// This layer with its f32 weight kept in half precision where its large products round it so anyway: f16 on a
    /// Vulkan GPU with matrix cores, bf16 on a CPU whose bf16 scope may take bf16 inputs. Half the weight's memory,
    /// those products unchanged (the bias stays f32); smaller ones read a widened copy. Otherwise itself.
    pub fn to_half(&self) -> Result<Self> {
        let device = self.weight.device();
        let half = if crate::vulkan::has_matrix_cores(device) {
            crate::DType::F16
        } else if device.is_cpu() && crate::cpu::bf16_allowed() {
            crate::DType::BF16
        } else {
            return Ok(self.clone());
        };
        if self.weight.dtype() != crate::DType::F32 || self.weight.rank() != 2 {
            return Ok(self.clone());
        }
        Ok(Self::new(self.weight.to_dtype(half)?, self.bias.clone()))
    }
}

impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> crate::Result<Tensor> {
        // a weight kept in half precision (`Linear::to_half`): the matrix cores or the CPU's bf16 products read it as
        // it is, any other product a widened copy
        if self.weight.dtype() != x.dtype() && (x.device().is_vulkan() || x.device().is_cpu()) {
            if x.device().is_cpu() {
                if let Some(y) =
                    crate::cpu::linear_bf16_weight(x, &self.weight, self.bias.as_ref())?
                {
                    return Ok(y);
                }
            } else if let Some(y) = crate::vulkan::linear_half_weight(x, &self.weight)? {
                return match &self.bias {
                    None => Ok(y),
                    Some(bias) => y.broadcast_add(bias),
                };
            }
            let widened = Self::new(self.weight.to_dtype(x.dtype())?, self.bias.clone());
            return widened.forward(x);
        }
        // When possible, we avoid using a broadcasted matmul as it is much slower
        // than the standard matmul for the cuda and cpu backends.
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Create or initialize a new linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vb: crate::nn::VarBuilder) -> Result<Linear> {
    let init_ws = crate::nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = crate::nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use crate::nn::Module;
    use crate::{DType, Device, Tensor};

    // on a CPU with bf16 pairs, a halved layer keeps a bf16 weight: inside the bf16 scope its large product is the f32
    // layer's there (both round the weight), a small one or one outside the f32 layer's over the rounded weight
    #[test]
    fn a_bf16_weight_linear_matches_its_f32_layer_on_the_cpu() -> crate::Result<()> {
        if !crate::cpu::bf16_allowed() {
            return Ok(());
        }
        let (n, k) = (300, 512);
        let weight = Tensor::randn(0f32, 1., (n, k), &Device::Cpu)?;
        let bias = Tensor::randn(0f32, 1., n, &Device::Cpu)?;
        let full = Linear::new(weight.clone(), Some(bias.clone()));
        let halved = full.to_half()?;
        assert_eq!(halved.weight().dtype(), DType::BF16);
        let rounded = Linear::new(
            weight.to_dtype(DType::BF16)?.to_dtype(DType::F32)?,
            Some(bias),
        );
        let off = |a: &Tensor, b: &Tensor| -> crate::Result<f32> {
            (a - b)?.abs()?.max_all()?.to_scalar::<f32>()
        };
        // 1, 3 rows below the scope's size: the rounded weight's f32 product; 512 past it: the scope's bf16 one
        for (rows, scoped) in [(1, false), (3, false), (512, true)] {
            let x = Tensor::randn(0f32, 1., (1, rows, k), &Device::Cpu)?;
            let inside = crate::cpu::with_bf16_matmuls(|| halved.forward(&x))?;
            assert_eq!(inside.dims(), [1, rows, n]);
            let want = if scoped {
                crate::cpu::with_bf16_matmuls(|| full.forward(&x))?
            } else {
                rounded.forward(&x)?
            };
            assert!(off(&inside, &want)? < 1e-3, "{rows} rows inside the scope");
            assert!(
                off(&halved.forward(&x)?, &rounded.forward(&x)?)? < 1e-3,
                "{rows} rows outside"
            );
        }
        Ok(())
    }
}
