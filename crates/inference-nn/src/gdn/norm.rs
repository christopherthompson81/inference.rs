#[cfg(feature = "cuda")]
use inference_quant::QuantizedActivation;
use inference_quant::ShardedVarBuilder;
use inference_tensor::{D, DType, Device, Result, Tensor};

use super::GdnOutputGate;

#[cfg(feature = "cuda")]
use crate::cuda::gdn::GdnFp8OutputSpec;

pub struct RmsNormGated {
    pub weight: Tensor,
    eps: f64,
    gate: GdnOutputGate,
}

impl RmsNormGated {
    #[cfg(test)]
    pub fn from_parts(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            eps,
            gate: GdnOutputGate::Silu,
        }
    }

    pub fn new(
        size: usize,
        eps: f64,
        gate: GdnOutputGate,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let mut weight = vb.get(size, "weight")?;
        if let Some(target_dev) = isq_target_device {
            weight = weight.to_device(target_dev)?;
        }
        Ok(Self { weight, eps, gate })
    }

    pub fn gate(&self) -> GdnOutputGate {
        self.gate
    }

    #[cfg(feature = "cuda")]
    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda()
            && (2..=4).contains(&x.rank())
            && (2..=4).contains(&gate.rank())
            && gate.elem_count() == x.elem_count()
            && x.dim(D::Minus1)? == self.weight.elem_count()
            && gate.dtype() == x.dtype()
            && self.weight.dtype() == x.dtype()
            && matches!(x.dtype(), DType::F16 | DType::BF16)
        {
            return crate::cuda::gdn::rmsnorm_gated_cuda(
                x,
                gate,
                &self.weight,
                self.eps,
                self.gate,
            );
        }

        // one pass on Vulkan: the norm over a head (the weight's width, a group as wide as the row), then the gate
        if x.device().is_vulkan() && x.dim(D::Minus1)? == self.weight.elem_count() {
            let order = match self.gate {
                GdnOutputGate::Silu => inference_tensor::vulkan::GatedNorm::NormThenSilu,
                GdnOutputGate::Sigmoid => inference_tensor::vulkan::GatedNorm::NormThenSigmoid,
            };
            return inference_tensor::vulkan::gated_rmsnorm_in(
                &x.to_dtype(DType::F32)?,
                &gate.reshape(x.shape().clone())?.to_dtype(DType::F32)?,
                &self.weight.to_dtype(DType::F32)?,
                self.eps as f32,
                self.weight.elem_count(),
                order,
            )?
            .to_dtype(x.dtype());
        }
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = gate.reshape(x.shape().clone())?.to_dtype(DType::F32)?;
        let gate = match self.gate {
            GdnOutputGate::Silu => inference_tensor::nn::ops::silu(&gate)?,
            GdnOutputGate::Sigmoid => inference_tensor::nn::ops::sigmoid(&gate)?,
        };
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_mul(&gate)?;
        out.to_dtype(dtype)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_quantized(
        &self,
        x: &Tensor,
        gate: &Tensor,
        spec: &GdnFp8OutputSpec,
        num_v_heads: usize,
        head_v_dim: usize,
    ) -> Result<QuantizedActivation> {
        crate::cuda::gdn::rmsnorm_gated_quantized_cuda(x, gate, self, spec, num_v_heads, head_v_dim)
    }
}

#[cfg(test)]
mod vulkan_tests {
    use inference_tensor::{Device, Result, Tensor};

    use super::{GdnOutputGate, RmsNormGated};

    // the one-pass Vulkan norm against the composite on the CPU: both gates, a decode row (whose norm also writes int8
    // blocks for the next matvec), a prompt's rows, and a long prompt's head rows past one grid dimension; 16 heads of 128
    #[test]
    fn vulkan_gated_gdn_norm_matches_the_cpu() -> Result<()> {
        let Ok(vk) = Device::new_vulkan(0) else {
            return Ok(());
        };
        const TOLERANCE: f32 = 1e-5;
        let cpu = Device::Cpu;
        let weight = Tensor::randn(1f32, 0.1, 128, &cpu)?;
        for gate in [GdnOutputGate::Silu, GdnOutputGate::Sigmoid] {
            let norm = RmsNormGated {
                weight: weight.clone(),
                eps: 1e-6,
                gate,
            };
            let on_vk = RmsNormGated {
                weight: weight.to_device(&vk)?,
                eps: 1e-6,
                gate,
            };
            // 4200 tokens: 67,200 head rows
            for rows in [1usize, 37, 4200] {
                let x = Tensor::randn(0f32, 1.0, (rows, 16, 128), &cpu)?;
                let g = Tensor::randn(0f32, 2.0, (rows, 16, 128), &cpu)?;
                let want = norm.forward(&x, &g)?;
                let got = on_vk.forward(&x.to_device(&vk)?, &g.to_device(&vk)?)?;
                let diff = (got.to_device(&cpu)? - &want)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(diff < TOLERANCE, "{gate:?} {rows} rows: {diff}");
            }
        }
        Ok(())
    }
}
