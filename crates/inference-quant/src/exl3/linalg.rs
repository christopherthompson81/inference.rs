//! The dense factorizations EXL3's encoder needs (a Hessian's Cholesky factor, a symmetric solve), through cuSOLVER.
//! Matrices are row-major and symmetric, so cuSOLVER's column-major view of one is the same matrix; an upper
//! factorization there leaves L (`a = L L^T`) in the lower triangle as we read it. The other triangle keeps its
//! input values, so a caller reads the lower one only.

use inference_tensor::cuda_backend::cudarc::cusolver::sys as cs;
use inference_tensor::{DType, Device, Result, Storage, Tensor};

use crate::utils::slice_ptr;

struct Handle(cs::cusolverDnHandle_t);

impl Handle {
    fn new(dev: &inference_tensor::CudaDevice) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        // a handle's creation and destruction reach the whole context, which no other thread's capture may see
        let _scope = inference_tensor::cuda_backend::context_wide_scope();
        // SAFETY: an out pointer for the new handle
        check(unsafe { cs::cusolverDnCreate(&mut handle) }, "create")?;
        let handle = Self(handle);
        // SAFETY: a live handle and this device's stream
        check(
            unsafe {
                cs::cusolverDnSetStream(handle.0, dev.cuda_stream().cu_stream() as cs::cudaStream_t)
            },
            "set its stream",
        )?;
        Ok(handle)
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        let _scope = inference_tensor::cuda_backend::context_wide_scope();
        // SAFETY: the handle this owns, destroyed once
        unsafe { cs::cusolverDnDestroy(self.0) };
    }
}

fn check(status: cs::cusolverStatus_t, what: &str) -> Result<()> {
    if status != cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
        inference_tensor::bail!("cuSOLVER could not {what}: {status:?}");
    }
    Ok(())
}

// the factorization's info word: 0, or the order of the leading minor that is not positive definite
fn info(dev_info: &Tensor) -> Result<i32> {
    Ok(dev_info.to_device(&Device::Cpu)?.to_vec1::<i32>()?[0])
}

/// The Cholesky factor of the symmetric positive definite `a` `(n, n)`: L in the lower triangle (the upper one
/// holds `a`'s values), or None when `a` is not positive definite.
pub(crate) fn cholesky(a: &Tensor) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = a.device() else {
        inference_tensor::bail!("the EXL3 encoder factors on CUDA");
    };
    let n = a.dim(0)?;
    let factor = a.to_dtype(DType::F32)?.contiguous()?.copy()?;
    let dev_info = Tensor::zeros(1, DType::I32, a.device())?;
    let handle = Handle::new(dev)?;
    {
        let (f_s, f_l) = factor.storage_and_layout();
        let (i_s, i_l) = dev_info.storage_and_layout();
        let (Storage::Cuda(f_c), Storage::Cuda(i_c)) = (&*f_s, &*i_s) else {
            inference_tensor::bail!("the factorization's tensors live on CUDA");
        };
        let (ptr, _f) = slice_ptr(f_c.as_cuda_slice::<f32>()?, f_l.start_offset());
        let (info_ptr, _i) = slice_ptr(i_c.as_cuda_slice::<i32>()?, i_l.start_offset());
        let mut lwork = 0;
        let upper = cs::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
        // SAFETY: `factor` is this call's own n x n copy, factored in place, and `dev_info` its own info word
        check(
            unsafe {
                cs::cusolverDnSpotrf_bufferSize(
                    handle.0,
                    upper,
                    n as i32,
                    ptr as *mut f32,
                    n as i32,
                    &mut lwork,
                )
            },
            "size a factorization",
        )?;
        // SAFETY: cuSOLVER writes its workspace before reading it
        let work = unsafe { dev.alloc::<f32>(lwork.max(1) as usize)? };
        let (work_ptr, _w) = slice_ptr(&work, 0);
        // SAFETY: as above, with a workspace of the size it asked for
        check(
            unsafe {
                cs::cusolverDnSpotrf(
                    handle.0,
                    upper,
                    n as i32,
                    ptr as *mut f32,
                    n as i32,
                    work_ptr as *mut f32,
                    lwork,
                    info_ptr as *mut i32,
                )
            },
            "factor a Hessian",
        )?;
    }
    Ok((info(&dev_info)? == 0).then_some(factor))
}

/// `x` with `a x = b` for the symmetric positive definite `a` `(n, n)` and `b` `(n)`, or None when `a` is not
/// positive definite.
pub(crate) fn solve_spd(a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
    let Device::Cuda(dev) = a.device() else {
        inference_tensor::bail!("the EXL3 encoder solves on CUDA");
    };
    let n = a.dim(0)?;
    let Some(factor) = cholesky(a)? else {
        return Ok(None);
    };
    let x = b.to_dtype(DType::F32)?.contiguous()?.copy()?;
    let dev_info = Tensor::zeros(1, DType::I32, a.device())?;
    let handle = Handle::new(dev)?;
    {
        let (f_s, f_l) = factor.storage_and_layout();
        let (x_s, x_l) = x.storage_and_layout();
        let (i_s, i_l) = dev_info.storage_and_layout();
        let (Storage::Cuda(f_c), Storage::Cuda(x_c), Storage::Cuda(i_c)) = (&*f_s, &*x_s, &*i_s)
        else {
            inference_tensor::bail!("the solve's tensors live on CUDA");
        };
        let (f_ptr, _f) = slice_ptr(f_c.as_cuda_slice::<f32>()?, f_l.start_offset());
        let (x_ptr, _x) = slice_ptr(x_c.as_cuda_slice::<f32>()?, x_l.start_offset());
        let (info_ptr, _i) = slice_ptr(i_c.as_cuda_slice::<i32>()?, i_l.start_offset());
        // SAFETY: the factor cholesky made, and `x` this call's own copy of `b`, solved in place
        check(
            unsafe {
                cs::cusolverDnSpotrs(
                    handle.0,
                    cs::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                    n as i32,
                    1,
                    f_ptr as *const f32,
                    n as i32,
                    x_ptr as *mut f32,
                    n as i32,
                    info_ptr as *mut i32,
                )
            },
            "solve",
        )?;
    }
    Ok((info(&dev_info)? == 0).then_some(x))
}

#[cfg(test)]
mod tests {
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::{cholesky, solve_spd};

    // L L^T rebuilds the matrix from the lower triangle cholesky leaves, and solve_spd's x satisfies a x = b
    #[test]
    fn the_factor_rebuilds_its_matrix_and_the_solve_holds() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let n = 48;
        let m = Tensor::randn(0f32, 1f32, (n, n), &Device::Cpu)?;
        let a = (m.matmul(&m.t()?)? + (Tensor::eye(n, DType::F32, &Device::Cpu)? * n as f64)?)?;
        let factor = cholesky(&a.to_device(&device)?)?.expect("positive definite");
        let f = factor.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let lower = (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| if c <= r { f[r][c] } else { 0.0 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let l = Tensor::new(lower, &Device::Cpu)?;
        let rebuilt = l.matmul(&l.t()?)?;
        let err = (&rebuilt - &a)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-3, "L L^T off by {err}");
        let b = Tensor::randn(0f32, 1f32, n, &Device::Cpu)?;
        let x =
            solve_spd(&a.to_device(&device)?, &b.to_device(&device)?)?.expect("positive definite");
        let ax = a
            .matmul(&x.to_device(&Device::Cpu)?.unsqueeze(1)?)?
            .squeeze(1)?;
        let err = (&ax - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-3, "a x off b by {err}");
        let not_pd = (Tensor::eye(n, DType::F32, &Device::Cpu)? * -1.0)?;
        assert!(cholesky(&not_pd.to_device(&device)?)?.is_none());
        Ok(())
    }
}
