//! Qwen3.8-Flash-Next's hyper-connection elementwise work, each step one launch: the grouped norm a block reads its
//! streams through, the low-rank mix's activation, the mean over streams, the write-back weights and the write-back.

use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use inference_tensor::{
    CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor,
    backend::BackendStorage,
};

#[derive(Clone, Copy)]
enum HyperOp {
    GroupNorm { group: usize, eps: f32 },
    Mix { streams: usize, hidden: usize },
    ScaledSilu { scale: f32 },
    InjectWeights { scale: f32 },
    InjectAdd { streams: usize, hidden: usize },
}

fn int(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(inference_tensor::Error::wrap)
}

impl HyperOp {
    fn name(&self) -> &'static str {
        match self {
            Self::GroupNorm { .. } => "hc-group-norm",
            Self::Mix { .. } => "hc-mix",
            Self::ScaledSilu { .. } => "hc-scaled-silu",
            Self::InjectWeights { .. } => "hc-inject-weights",
            Self::InjectAdd { .. } => "hc-inject-add",
        }
    }

    // the output's shape, from the first input's
    fn out_shape(&self, first: &Shape) -> Result<Shape> {
        Ok(match *self {
            Self::Mix { streams, .. } => {
                let mut dims = first.dims().to_vec();
                let last = dims
                    .last_mut()
                    .expect("a mixed tensor has a last dimension");
                *last /= streams;
                Shape::from(dims)
            }
            _ => first.clone(),
        })
    }

    fn launch(&self, inputs: &[(&CudaStorage, &Layout)]) -> Result<(CudaStorage, Shape)> {
        let dtype = inputs[0].0.dtype();
        for (storage, layout) in inputs {
            if storage.dtype() != dtype || !layout.is_contiguous() {
                inference_tensor::bail!("{} takes contiguous inputs of one dtype", self.name());
            }
        }
        let shape = self.out_shape(inputs[0].1.shape())?;
        let device = inputs[0].0.device();
        macro_rules! run {
            ($ty:ty, $norm:ident, $mix:ident, $silu:ident, $weights:ident, $add:ident) => {{
                let stream = device.cuda_stream();
                let views = inputs
                    .iter()
                    .map(|(s, l)| Ok(s.as_cuda_slice::<$ty>()?.slice(l.start_offset()..)))
                    .collect::<Result<Vec<_>>>()?;
                let ptrs = views
                    .iter()
                    .map(|v| v.device_ptr(&stream))
                    .collect::<Vec<_>>();
                let input = |i: usize| ptrs[i].0 as *const core::ffi::c_void;
                let mut out = unsafe { device.alloc::<$ty>(shape.elem_count())? };
                let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
                let out_ptr = out_ptr as *mut core::ffi::c_void;
                let cu = stream.cu_stream() as i64;
                let n = i64::try_from(shape.elem_count()).map_err(inference_tensor::Error::wrap)?;
                let width = *inputs[0].1.shape().dims().last().unwrap_or(&1);
                let rows = inputs[0].1.shape().elem_count() / width.max(1);
                // SAFETY: every input is a contiguous device buffer of the shape the op checked
                let status = unsafe {
                    match *self {
                        Self::GroupNorm { group, eps } => super::ffi::$norm(
                            input(0),
                            input(1),
                            out_ptr,
                            int(rows)?,
                            int(width)?,
                            int(group)?,
                            eps,
                            cu,
                        ),
                        Self::Mix { streams, hidden } => super::ffi::$mix(
                            input(0),
                            input(1),
                            out_ptr,
                            int(rows)?,
                            int(streams)?,
                            int(hidden)?,
                            cu,
                        ),
                        Self::ScaledSilu { scale } => {
                            super::ffi::$silu(input(0), out_ptr, n, scale, cu)
                        }
                        Self::InjectWeights { scale } => {
                            super::ffi::$weights(input(0), out_ptr, n, scale, cu)
                        }
                        Self::InjectAdd { streams, hidden } => super::ffi::$add(
                            input(0),
                            input(1),
                            input(2),
                            out_ptr,
                            int(rows)?,
                            int(streams)?,
                            int(hidden)?,
                            cu,
                        ),
                    }
                };
                drop(out_guard);
                drop(ptrs);
                if status != 0 {
                    inference_tensor::bail!("{} failed with status {status}", self.name());
                }
                CudaStorage::wrap_cuda_slice(out, device.clone())
            }};
        }
        let storage = match dtype {
            DType::BF16 => run!(
                half::bf16,
                hc_group_norm_bf16,
                hc_mix_bf16,
                hc_scaled_silu_bf16,
                hc_inject_weights_bf16,
                hc_inject_add_bf16
            ),
            DType::F32 => run!(
                f32,
                hc_group_norm_f32,
                hc_mix_f32,
                hc_scaled_silu_f32,
                hc_inject_weights_f32,
                hc_inject_add_f32
            ),
            dtype => inference_tensor::bail!("{} does not take {dtype:?}", self.name()),
        };
        Ok((storage, shape))
    }
}

impl CustomOp1 for HyperOp {
    fn name(&self) -> &'static str {
        HyperOp::name(self)
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        inference_tensor::bail!("{} runs on CUDA only", HyperOp::name(self))
    }

    fn cuda_fwd(&self, a: &CudaStorage, a_l: &Layout) -> Result<(CudaStorage, Shape)> {
        self.launch(&[(a, a_l)])
    }
}

impl CustomOp2 for HyperOp {
    fn name(&self) -> &'static str {
        HyperOp::name(self)
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        inference_tensor::bail!("{} runs on CUDA only", HyperOp::name(self))
    }

    fn cuda_fwd(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(&[(a, a_l), (b, b_l)])
    }
}

impl CustomOp3 for HyperOp {
    fn name(&self) -> &'static str {
        HyperOp::name(self)
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        inference_tensor::bail!("{} runs on CUDA only", HyperOp::name(self))
    }

    fn cuda_fwd(
        &self,
        a: &CudaStorage,
        a_l: &Layout,
        b: &CudaStorage,
        b_l: &Layout,
        c: &CudaStorage,
        c_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch(&[(a, a_l), (b, b_l), (c, c_l)])
    }
}

/// Whether the fused hyper-connection ops take `x`: a BF16 or F32 tensor on a CUDA device.
pub fn supports(x: &Tensor) -> bool {
    x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F32)
}

/// Each `group`-wide slice of `x`'s last dimension RMS-normed, times `scale` (one per element of that dimension).
pub fn group_norm(x: &Tensor, scale: &Tensor, group: usize, eps: f32) -> Result<Tensor> {
    let scale = scale.to_dtype(x.dtype())?;
    x.contiguous()?
        .apply_op2_no_bwd(&scale.contiguous()?, &HyperOp::GroupNorm { group, eps })
}

/// The mean over `streams` of `sigmoid(mix) * normed`, both `[.., streams * hidden]`, as `[.., hidden]`.
pub fn mix(mix: &Tensor, normed: &Tensor, streams: usize, hidden: usize) -> Result<Tensor> {
    mix.contiguous()?
        .apply_op2_no_bwd(&normed.contiguous()?, &HyperOp::Mix { streams, hidden })
}

/// `silu(x * scale)`.
pub fn scaled_silu(x: &Tensor, scale: f32) -> Result<Tensor> {
    x.contiguous()?
        .apply_op1_no_bwd(&HyperOp::ScaledSilu { scale })
}

/// `2 sigmoid(x * scale)`.
pub fn inject_weights(x: &Tensor, scale: f32) -> Result<Tensor> {
    x.contiguous()?
        .apply_op1_no_bwd(&HyperOp::InjectWeights { scale })
}

/// `streams + output * inject[s]` into every stream: `streams` `[.., streams * hidden]`, `output` `[.., hidden]`,
/// `inject` `[.., streams]`.
pub fn inject_add(streams_in: &Tensor, output: &Tensor, inject: &Tensor) -> Result<Tensor> {
    let streams = *inject.dims().last().unwrap_or(&1);
    let hidden = *output.dims().last().unwrap_or(&1);
    let output = output.to_dtype(streams_in.dtype())?;
    let inject = inject.to_dtype(streams_in.dtype())?;
    streams_in.contiguous()?.apply_op3_no_bwd(
        &output.contiguous()?,
        &inject.contiguous()?,
        &HyperOp::InjectAdd { streams, hidden },
    )
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use inference_tensor::{D, Device};

    const ROWS: usize = 3;
    const STREAMS: usize = 4;
    const HIDDEN: usize = 96;
    const LOWRANK: usize = 8;
    const EPS: f32 = 1e-6;
    const TOLERANCE: f32 = 1e-5;

    fn patterned(shape: (usize, usize), salt: usize) -> Result<Tensor> {
        let values = (0..shape.0 * shape.1)
            .map(|i| ((i * 37 + salt * 19) % 211) as f32 / 105.0 - 1.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu)
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_device(&Device::Cpu)? - b.to_device(&Device::Cpu)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()
    }

    // each fused op against the tensor ops the model runs without it, in F32
    #[test]
    fn fused_ops_match_the_tensor_ops() -> Result<()> {
        let Ok(cuda) = Device::new_cuda(0) else {
            return Ok(());
        };
        let width = STREAMS * HIDDEN;
        let x = patterned((ROWS, width), 1)?;
        let scale = (patterned((1, width), 2)?.squeeze(0)? + 1.0)?;
        let on = |t: &Tensor| t.to_device(&cuda);

        let grouped = x.reshape((ROWS, STREAMS, HIDDEN))?;
        let rms = (grouped.sqr()?.mean_keepdim(D::Minus1)? + f64::from(EPS))?.sqrt()?;
        let expected = grouped
            .broadcast_div(&rms)?
            .reshape((ROWS, width))?
            .broadcast_mul(&scale)?;
        assert!(
            max_diff(&group_norm(&on(&x)?, &on(&scale)?, HIDDEN, EPS)?, &expected)? < TOLERANCE
        );

        let mix_in = patterned((ROWS, width), 3)?;
        let expected = (inference_tensor::nn::ops::sigmoid(&mix_in)? * &x)?
            .reshape((ROWS, STREAMS, HIDDEN))?
            .mean(1)?;
        assert!(max_diff(&mix(&on(&mix_in)?, &on(&x)?, STREAMS, HIDDEN)?, &expected)? < TOLERANCE);

        let low = patterned((ROWS, LOWRANK), 4)?;
        let expected = inference_tensor::nn::ops::silu(&(&low * 0.25)?)?;
        assert!(max_diff(&scaled_silu(&on(&low)?, 0.25)?, &expected)? < TOLERANCE);

        let raw = patterned((ROWS, STREAMS), 5)?;
        let expected = (inference_tensor::nn::ops::sigmoid(&(&raw * 0.25)?)? * 2.0)?;
        assert!(max_diff(&inject_weights(&on(&raw)?, 0.25)?, &expected)? < TOLERANCE);

        let output = patterned((ROWS, HIDDEN), 6)?;
        let weights = patterned((ROWS, STREAMS), 7)?;
        let expected = (&x
            + output
                .unsqueeze(1)?
                .broadcast_mul(&weights.unsqueeze(2)?)?
                .reshape((ROWS, width))?)?;
        assert!(
            max_diff(
                &inject_add(&on(&x)?, &on(&output)?, &on(&weights)?)?,
                &expected
            )? < TOLERANCE
        );
        Ok(())
    }
}
