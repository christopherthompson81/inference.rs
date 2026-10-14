use inference_tensor::{
    CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor,
    backend::BackendStorage,
};

struct BidirectionalLstm;

impl CustomOp2 for BidirectionalLstm {
    fn name(&self) -> &'static str {
        "lstm-bidirectional"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        inference_tensor::bail!("the fused LSTM recurrence requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        gx: &CudaStorage,
        gx_layout: &Layout,
        w: &CudaStorage,
        w_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        let (steps, gates2) = gx_layout.shape().dims2()?;
        let (directions, hidden, gates) = w_layout.shape().dims3()?;
        if directions != 2 || gates != 4 * hidden || gates2 != 2 * gates {
            inference_tensor::bail!(
                "LSTM shape mismatch: gx={:?}, w_hh_t={:?}",
                gx_layout.shape(),
                w_layout.shape()
            );
        }
        if gx.dtype() != DType::F32 || w.dtype() != DType::F32 {
            inference_tensor::bail!("the fused LSTM recurrence runs in F32");
        }
        if !gx_layout.is_contiguous() || !w_layout.is_contiguous() {
            inference_tensor::bail!("the fused LSTM recurrence needs contiguous inputs");
        }
        let device = gx.device();
        let stream = device.cuda_stream();
        let gx_slice = gx.as_cuda_slice::<f32>()?;
        let w_slice = w.as_cuda_slice::<f32>()?;
        let gx_view = gx_slice.slice(gx_layout.start_offset()..);
        let w_view = w_slice.slice(w_layout.start_offset()..);
        let mut out = unsafe { device.alloc::<f32>(steps * 2 * hidden)? };
        let (gx_ptr, gx_guard) = gx_view.device_ptr(&stream);
        let (w_ptr, w_guard) = w_view.device_ptr(&stream);
        let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
        let status = unsafe {
            crate::cuda::ffi::lstm_bidirectional_f32(
                gx_ptr as *const core::ffi::c_void,
                w_ptr as *const core::ffi::c_void,
                out_ptr as *mut core::ffi::c_void,
                i32::try_from(steps).map_err(inference_tensor::Error::wrap)?,
                i32::try_from(hidden).map_err(inference_tensor::Error::wrap)?,
                stream.cu_stream() as i64,
            )
        };
        drop(out_guard);
        drop(w_guard);
        drop(gx_guard);
        if status != 0 {
            inference_tensor::bail!("lstm_bidirectional_f32 failed with status {status}");
        }
        Ok((
            CudaStorage::wrap_cuda_slice(out, device.clone()),
            Shape::from((steps, 2 * hidden)),
        ))
    }
}

/// Both directions of a one-layer PyTorch LSTM in one launch; the layouts are documented on the kernel in lstm.cu.
pub fn lstm_bidirectional(gx: &Tensor, w_hh_t: &Tensor) -> Result<Tensor> {
    gx.contiguous()?
        .apply_op2_no_bwd(&w_hh_t.contiguous()?, &BidirectionalLstm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::Device;

    // a direct host loop of PyTorch's LSTM cell, one direction
    fn reference(gx: &[f32], w: &[f32], steps: usize, hidden: usize, reverse: bool) -> Vec<f32> {
        let gates = 4 * hidden;
        let sigmoid = |x: f32| 1. / (1. + (-x).exp());
        let (mut h, mut c) = (vec![0f32; hidden], vec![0f32; hidden]);
        let mut out = vec![0f32; steps * hidden];
        for step in 0..steps {
            let t = if reverse { steps - 1 - step } else { step };
            let mut g = gx[t * gates..(t + 1) * gates].to_vec();
            for (k, hk) in h.iter().enumerate() {
                for (j, gj) in g.iter_mut().enumerate() {
                    *gj += hk * w[k * gates + j];
                }
            }
            for j in 0..hidden {
                c[j] = sigmoid(g[hidden + j]) * c[j] + sigmoid(g[j]) * g[2 * hidden + j].tanh();
                h[j] = sigmoid(g[3 * hidden + j]) * c[j].tanh();
            }
            out[t * hidden..(t + 1) * hidden].copy_from_slice(&h);
        }
        out
    }

    // hidden sizes below, at and past one gate per thread, and one that leaves the unrolled loop a tail
    #[test]
    fn matches_a_host_loop_in_both_directions() -> Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        for (steps, hidden) in [(1, 3), (9, 64), (17, 256), (5, 300), (4, 7u16)] {
            let scale = f32::from(hidden).sqrt().recip();
            let hidden = usize::from(hidden);
            let gates = 4 * hidden;
            let gx = Tensor::randn(0f32, 1., (steps, 2 * gates), &Device::Cpu)?;
            // PyTorch's 1 / sqrt(hidden) init scale; larger weights compound float rounding across the steps
            let w = Tensor::randn(0f32, scale, (2, hidden, gates), &Device::Cpu)?;
            let got = lstm_bidirectional(&gx.to_device(&device)?, &w.to_device(&device)?)?
                .to_vec2::<f32>()?;
            let gx_rows = gx.to_vec2::<f32>()?;
            for direction in 0..2 {
                let gx_dir: Vec<f32> = gx_rows
                    .iter()
                    .flat_map(|row| row[direction * gates..(direction + 1) * gates].to_vec())
                    .collect();
                let w_dir = w.get(direction)?.flatten_all()?.to_vec1::<f32>()?;
                let want = reference(&gx_dir, &w_dir, steps, hidden, direction == 1);
                for t in 0..steps {
                    for j in 0..hidden {
                        let (g, e) = (got[t][direction * hidden + j], want[t * hidden + j]);
                        assert!(
                            (g - e).abs() < 1e-4,
                            "steps {steps} hidden {hidden} direction {direction} t {t} j {j}: {g} vs {e}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
