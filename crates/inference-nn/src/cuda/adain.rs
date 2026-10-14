use inference_tensor::{
    CpuStorage, CudaStorage, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor,
    backend::BackendStorage,
};

/// What follows the instance norm and the style's affine.
#[derive(Clone, Copy, Debug)]
pub enum AdainActivation {
    /// Snake with a per-channel alpha, passed as a (channels,) tensor.
    Snake,
    LeakyRelu(f32),
}

struct AdainAct {
    eps: f32,
    slope: f32,
}

fn view<'a>(
    (storage, layout): (&'a CudaStorage, &Layout),
) -> Result<inference_tensor::cuda_backend::cudarc::driver::CudaView<'a, f32>> {
    Ok(storage
        .as_cuda_slice::<f32>()?
        .slice(layout.start_offset()..))
}

impl AdainAct {
    fn launch(
        &self,
        xs: (&CudaStorage, &Layout),
        h: (&CudaStorage, &Layout),
        alpha: Option<(&CudaStorage, &Layout)>,
    ) -> Result<(CudaStorage, Shape)> {
        use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        let (batch, channels, len) = xs.1.shape().dims3()?;
        if h.1.shape().dims2()? != (batch, 2 * channels)
            || alpha.is_some_and(|(_, l)| l.shape().elem_count() != channels)
        {
            inference_tensor::bail!(
                "AdaIN shape mismatch: xs={:?}, h={:?}",
                xs.1.shape(),
                h.1.shape()
            );
        }
        let inputs = [Some(xs), Some(h), alpha];
        for (storage, layout) in inputs.iter().flatten() {
            if storage.dtype() != DType::F32 || !layout.is_contiguous() {
                inference_tensor::bail!("the fused AdaIN takes contiguous F32 inputs");
            }
        }
        let device = xs.0.device();
        let stream = device.cuda_stream();
        let (xs_view, h_view) = (view(xs)?, view(h)?);
        let alpha_view = alpha.map(view).transpose()?;
        let mut out = unsafe { device.alloc::<f32>(batch * channels * len)? };
        let (xs_ptr, xs_guard) = xs_view.device_ptr(&stream);
        let (h_ptr, h_guard) = h_view.device_ptr(&stream);
        let alpha_ptr = alpha_view.as_ref().map(|view| view.device_ptr(&stream));
        let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
        let status = unsafe {
            crate::cuda::ffi::adain_act_f32(
                xs_ptr as *const core::ffi::c_void,
                h_ptr as *const core::ffi::c_void,
                alpha_ptr.as_ref().map_or(std::ptr::null(), |(ptr, _)| {
                    *ptr as *const core::ffi::c_void
                }),
                out_ptr as *mut core::ffi::c_void,
                i32::try_from(batch * channels).map_err(inference_tensor::Error::wrap)?,
                i32::try_from(channels).map_err(inference_tensor::Error::wrap)?,
                i32::try_from(len).map_err(inference_tensor::Error::wrap)?,
                self.eps,
                self.slope,
                stream.cu_stream() as i64,
            )
        };
        drop(out_guard);
        drop(alpha_ptr);
        drop(h_guard);
        drop(xs_guard);
        if status != 0 {
            inference_tensor::bail!("adain_act_f32 failed with status {status}");
        }
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            xs.1.shape().clone(),
        ))
    }
}

impl CustomOp2 for AdainAct {
    fn name(&self) -> &'static str {
        "adain-leaky-relu"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        inference_tensor::bail!("the fused AdaIN kernel requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        xs: &CudaStorage,
        xs_l: &Layout,
        h: &CudaStorage,
        h_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch((xs, xs_l), (h, h_l), None)
    }
}

impl CustomOp3 for AdainAct {
    fn name(&self) -> &'static str {
        "adain-snake"
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
        inference_tensor::bail!("the fused AdaIN kernel requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        xs: &CudaStorage,
        xs_l: &Layout,
        h: &CudaStorage,
        h_l: &Layout,
        alpha: &CudaStorage,
        alpha_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        self.launch((xs, xs_l), (h, h_l), Some((alpha, alpha_l)))
    }
}

/// Instance norm over `xs` (batch, channels, len), the affine `h` (batch, 2 * channels) holds as scale - 1 then shift,
/// and `act` (Snake taking `alpha`, one per channel), in one launch.
pub fn adain_act(
    xs: &Tensor,
    h: &Tensor,
    act: AdainActivation,
    alpha: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let (xs, h) = (xs.contiguous()?, h.contiguous()?);
    match (act, alpha) {
        (AdainActivation::Snake, Some(alpha)) => {
            xs.apply_op3_no_bwd(&h, &alpha.contiguous()?, &AdainAct { eps, slope: 0. })
        }
        (AdainActivation::LeakyRelu(slope), None) => {
            xs.apply_op2_no_bwd(&h, &AdainAct { eps, slope })
        }
        _ => inference_tensor::bail!("Snake takes an alpha and a leaky ReLU none"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::{D, Device};

    fn reference(
        xs: &Tensor,
        h: &Tensor,
        act: AdainActivation,
        alpha: Option<&Tensor>,
    ) -> Result<Tensor> {
        let c = xs.dim(1)?;
        let h = h.unsqueeze(2)?;
        let mean = xs.mean_keepdim(D::Minus1)?;
        let centered = xs.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let y = centered
            .broadcast_div(&(var + 1e-5)?.sqrt()?)?
            .broadcast_mul(&(h.narrow(1, 0, c)? + 1.)?)?
            .broadcast_add(&h.narrow(1, c, c)?)?;
        match (act, alpha) {
            (AdainActivation::Snake, Some(alpha)) => {
                let alpha = alpha.reshape((1, c, 1))?;
                let s = y.broadcast_mul(&alpha)?.sin()?;
                y + s.sqr()?.broadcast_div(&alpha)?
            }
            (AdainActivation::LeakyRelu(slope), _) => {
                let negative = (y.minimum(0f32)? * f64::from(slope))?;
                y.maximum(0f32)? + negative
            }
            _ => unreachable!(),
        }
    }

    // rows shorter and longer than a block, and a length that leaves partial warps
    #[test]
    fn matches_the_tensor_ops_for_both_activations() -> Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        for (batch, channels, len) in [(1, 3, 7), (2, 5, 300), (1, 4, 5000)] {
            let xs = Tensor::randn(1f32, 2., (batch, channels, len), &Device::Cpu)?;
            let h = Tensor::randn(0f32, 0.5, (batch, 2 * channels), &Device::Cpu)?;
            let alpha = Tensor::rand(0.2f32, 2., channels, &Device::Cpu)?;
            for (act, alpha) in [
                (AdainActivation::Snake, Some(&alpha)),
                (AdainActivation::LeakyRelu(0.2), None),
            ] {
                let want = reference(&xs, &h, act, alpha)?;
                let alpha_gpu = alpha.map(|a| a.to_device(&device)).transpose()?;
                let got = adain_act(
                    &xs.to_device(&device)?,
                    &h.to_device(&device)?,
                    act,
                    alpha_gpu.as_ref(),
                    1e-5,
                )?;
                let diff = (got.to_device(&Device::Cpu)? - &want)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(diff < 1e-4, "{act:?} {batch}x{channels}x{len}: {diff}");
            }
        }
        Ok(())
    }
}
