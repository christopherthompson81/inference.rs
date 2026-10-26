//! A linear that is dense for safetensors and quantized when the var builder has a quantized source (a GGUF file).

use std::borrow::Cow;
use std::sync::Arc;

use inference_quant::{QuantMethod, ShardedVarBuilder};
use inference_tensor::nn::Linear;
use inference_tensor::quantized::{QStorage, QTensor};
use inference_tensor::{Device, Module, Result, Tensor};

use crate::layers;

#[derive(Debug, Clone)]
pub enum MaybeQuantLinear {
    Dense(Linear),
    Quantized(Arc<dyn QuantMethod>),
    /// GGUF blocks dequantized at each use, so the activations stay at full precision
    Dequantized(Arc<QTensor>),
}

impl MaybeQuantLinear {
    pub fn new(in_dim: usize, out_dim: usize, bias: bool, vb: ShardedVarBuilder) -> Result<Self> {
        if vb.weight_source().is_none() {
            return Ok(Self::Dense(layers::linear_b(in_dim, out_dim, bias, vb)?));
        }
        let layer = if bias {
            inference_quant::linear(in_dim, out_dim, &None, vb)?
        } else {
            inference_quant::linear_no_bias(in_dim, out_dim, &None, vb)?
        };
        Ok(Self::Quantized(layer))
    }

    /// Like `new` without a bias, but a quantized weight is dequantized for each matmul instead of quantizing the
    /// activations: T5's outlier activations lose whole tokens to 8-bit activation blocks.
    pub fn new_dequantizing(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let blocks = vb
            .weight_source()
            .and_then(|source| source.mapped_blocks(&vb.prefix()))
            .and_then(|blocks| Some((blocks.ty.candle()?, blocks)))
            .filter(|(dtype, _)| dtype.block_size() > 1);
        let Some((dtype, blocks)) = blocks else {
            return Self::new(in_dim, out_dim, false, vb);
        };
        let storage = QStorage::from_data(Cow::Borrowed(blocks.bytes()), vb.device(), dtype)?;
        Ok(Self::Dequantized(Arc::new(QTensor::new(
            storage,
            blocks.dims.clone(),
        )?)))
    }

    /// The same layer on `device`; offloading moves dense layers only.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        match self {
            Self::Dense(linear) => Ok(Self::Dense(Linear::new(
                linear.weight().to_device(device)?,
                linear
                    .bias()
                    .map(|bias| bias.to_device(device))
                    .transpose()?,
            ))),
            Self::Quantized(_) | Self::Dequantized(_) => {
                inference_tensor::bail!(
                    "quantized diffusion layers cannot be offloaded between devices"
                )
            }
        }
    }
}

impl Module for MaybeQuantLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(xs),
            Self::Quantized(layer) => layer.forward(xs),
            Self::Dequantized(weight) => {
                let weight = weight.dequantize(xs.device())?.to_dtype(xs.dtype())?;
                Linear::new(weight, None).forward(xs)
            }
        }
    }
}
