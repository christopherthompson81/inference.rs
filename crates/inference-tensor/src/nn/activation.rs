//! Activation Functions
//!
use crate::{Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, serde::Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    Mish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}

impl super::Module for Activation {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu_erf(),
            // https://github.com/huggingface/transformers/blob/12f043eaeaabfef6f6efea411d98e6f6d3c094b7/src/transformers/activations.py#L49-L78
            Self::NewGelu => xs.gelu(),
            Self::Relu => xs.relu(),
            Self::Relu2 => xs.relu()?.sqr(),
            Self::Relu6 => xs.clamp(0f32, 6f32),
            Self::Silu => xs.silu(),
            Self::Sigmoid => crate::nn::ops::sigmoid(xs),
            Self::HardSigmoid => crate::nn::ops::hard_sigmoid(xs),
            Self::Swiglu => crate::nn::ops::swiglu(xs),
            Self::Swish => xs * crate::nn::ops::sigmoid(xs)?,
            Self::HardSwish => xs * crate::nn::ops::hard_sigmoid(xs)?,
            Self::Mish => crate::nn::ops::mish(xs),
            &Self::Elu(alpha) => xs.elu(alpha),
            &Self::LeakyRelu(negative_slope) => crate::nn::ops::leaky_relu(xs, negative_slope),
            Self::GeluPytorchTanh => xs.gelu(),
        }
    }
}
