//! Fused row ops of diffusion transformers over a bf16 residual stream (see `kernels/cuda/dit.cu`), each one launch in
//! place of the composed f32 tensor ops.

use std::ffi::c_void;

use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use inference_tensor::cuda_backend::{CudaStorage, CudaStorageSlice};
use inference_tensor::{DType, Result, Shape, Storage, Tensor};

// Holds the inputs' storage locks for the launch and hands out their device pointers, each at its layout's offset.
fn launch(
    inputs: &[Option<&Tensor>],
    out_shape: Shape,
    run: impl FnOnce(&[*const c_void], *mut c_void, i64) -> i32,
    name: &str,
) -> Result<Tensor> {
    let storages = inputs
        .iter()
        .map(|t| t.map(Tensor::storage_and_layout))
        .collect::<Vec<_>>();
    let device = storages
        .iter()
        .flatten()
        .find_map(|(storage, _)| match &**storage {
            Storage::Cuda(s) => Some(s.device.clone()),
            _ => None,
        });
    let Some(device) = device else {
        inference_tensor::bail!("{name} takes CUDA tensors");
    };
    let stream = device.cuda_stream();
    let mut guards = Vec::new();
    let mut ptrs = Vec::with_capacity(inputs.len());
    for entry in &storages {
        let Some((storage, layout)) = entry else {
            ptrs.push(std::ptr::null());
            continue;
        };
        let Storage::Cuda(s) = &**storage else {
            inference_tensor::bail!("{name} takes CUDA tensors");
        };
        let ptr = match &s.slice {
            CudaStorageSlice::BF16(v) => {
                let (p, g) = v.device_ptr(&stream);
                guards.push(g);
                (p as *const half::bf16)
                    .wrapping_add(layout.start_offset())
                    .cast::<c_void>()
            }
            CudaStorageSlice::F32(v) => {
                let (p, g) = v.device_ptr(&stream);
                guards.push(g);
                (p as *const f32)
                    .wrapping_add(layout.start_offset())
                    .cast::<c_void>()
            }
            _ => inference_tensor::bail!("{name} takes bf16 and f32 tensors"),
        };
        ptrs.push(ptr);
    }
    // SAFETY: the kernel writes every element
    let mut out = unsafe { device.alloc::<half::bf16>(out_shape.elem_count())? };
    let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
    let status = run(&ptrs, out_ptr as *mut c_void, stream.cu_stream() as i64);
    drop(out_guard);
    drop(guards);
    if status != 0 {
        inference_tensor::bail!("{name} failed with status {status}");
    }
    let storage = CudaStorage {
        slice: CudaStorageSlice::BF16(out),
        device,
    };
    Ok(Tensor::from((Storage::Cuda(storage), out_shape)))
}

fn check(t: &Tensor, dtype: DType, what: &str) -> Result<()> {
    if t.dtype() != dtype || !t.is_contiguous() {
        inference_tensor::bail!("{what} must be contiguous {dtype:?}, got {:?}", t.dtype());
    }
    Ok(())
}

fn check_vec(t: Option<&Tensor>, cols: usize, what: &str) -> Result<()> {
    if let Some(t) = t {
        check(t, DType::F32, what)?;
        if t.elem_count() != cols {
            inference_tensor::bail!("{what} has {} values for {cols} columns", t.elem_count());
        }
    }
    Ok(())
}

fn i32_of(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(inference_tensor::Error::wrap)
}

/// LayerNorm over the last dim of bf16 `xs`, then `* weight + bias` and the adaLN `* (1 + scale) + shift` (each an
/// optional f32 vector of the row width), in f32, returned in bf16.
pub fn layer_norm(
    xs: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    scale: Option<&Tensor>,
    shift: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    check(&xs, DType::BF16, "layer norm input")?;
    let cols = xs.dim(inference_tensor::D::Minus1)?;
    let rows = xs.elem_count() / cols;
    for (t, what) in [
        (weight, "weight"),
        (bias, "bias"),
        (scale, "scale"),
        (shift, "shift"),
    ] {
        check_vec(t, cols, what)?;
    }
    let (rows, cols) = (i32_of(rows)?, i32_of(cols)?);
    launch(
        &[Some(&xs), weight, bias, scale, shift],
        xs.shape().clone(),
        |p, out, stream| unsafe {
            super::ffi::dit_layer_norm_bf16(
                p[0], p[1], p[2], p[3], p[4], out, rows, cols, eps, stream,
            )
        },
        "dit layer norm",
    )
}

/// `xs + ys * gate` for bf16 `xs` and `ys` and an f32 `gate` of the row width, in f32, returned in bf16.
pub fn gated_residual(xs: &Tensor, ys: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let (xs, ys) = (xs.contiguous()?, ys.contiguous()?);
    check(&xs, DType::BF16, "residual")?;
    check(&ys, DType::BF16, "update")?;
    if xs.shape() != ys.shape() {
        inference_tensor::bail!(
            "gated residual shapes {:?} and {:?}",
            xs.shape(),
            ys.shape()
        );
    }
    let cols = xs.dim(inference_tensor::D::Minus1)?;
    let gate = gate.contiguous()?;
    check_vec(Some(&gate), cols, "gate")?;
    let (n, cols) = (xs.elem_count() as i64, i32_of(cols)?);
    launch(
        &[Some(&xs), Some(&ys), Some(&gate)],
        xs.shape().clone(),
        |p, out, stream| unsafe {
            super::ffi::dit_gated_residual_bf16(p[0], p[1], p[2], out, n, cols, stream)
        },
        "dit gated residual",
    )
}

/// RMSNorm over the whole row of bf16 `xs` `[b, tokens, heads * head_dim]` times the f32 `weight`, rounded to bf16,
/// then with `rope` (f32 `cos`, `sin`, each `[tokens, head_dim / 2]`) adjacent pairs rotated; returns
/// `[b, heads, tokens, head_dim]` bf16.
pub fn rms_norm_rope(
    xs: &Tensor,
    weight: &Tensor,
    rope: Option<(&Tensor, &Tensor)>,
    heads: usize,
    eps: f32,
) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    check(&xs, DType::BF16, "rms norm input")?;
    let (b, tokens, cols) = xs.dims3()?;
    if !cols.is_multiple_of(heads) || !(cols / heads).is_multiple_of(2) {
        inference_tensor::bail!("{cols} columns do not split into {heads} even heads");
    }
    let head_dim = cols / heads;
    check_vec(Some(weight), cols, "rms norm weight")?;
    if let Some((cos, sin)) = rope {
        for (t, what) in [(cos, "cos"), (sin, "sin")] {
            check(t, DType::F32, what)?;
            if t.dims2()? != (tokens, head_dim / 2) {
                inference_tensor::bail!(
                    "{what} is {:?} for {tokens} tokens of {head_dim}",
                    t.shape()
                );
            }
        }
    }
    let (bi, ti, hi, di) = (
        i32_of(b)?,
        i32_of(tokens)?,
        i32_of(heads)?,
        i32_of(head_dim)?,
    );
    launch(
        &[
            Some(&xs),
            Some(weight),
            rope.map(|r| r.0),
            rope.map(|r| r.1),
        ],
        Shape::from_dims(&[b, heads, tokens, head_dim]),
        |p, out, stream| unsafe {
            super::ffi::dit_rms_norm_rope_bf16(
                p[0], p[1], p[2], p[3], out, bi, ti, hi, di, eps, stream,
            )
        },
        "dit rms norm rope",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::{D, Device};

    const BF16_ULP: f32 = 1.0 / 64.0;

    fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()
    }

    // the composed f32 ops the kernels replace, on rows wider and narrower than a block
    #[test]
    fn matches_the_composed_ops() -> Result<()> {
        skip_without_cuda!();
        let dev = Device::new_cuda(0)?;
        for (tokens, heads, head_dim) in [(5, 2, 8), (37, 40, 128)] {
            let cols = heads * head_dim;
            let xs =
                (Tensor::randn(0f32, 1., (1, tokens, cols), &dev)? * 3.0)?.to_dtype(DType::BF16)?;
            let ys = Tensor::randn(0f32, 1., (1, tokens, cols), &dev)?.to_dtype(DType::BF16)?;
            let vec = |m: f32| -> Result<Tensor> { Tensor::randn(m, 0.3, cols, &dev) };
            let (w, b, scale, shift, gate) = (vec(1.)?, vec(0.)?, vec(0.)?, vec(0.)?, vec(0.)?);

            let x = xs.to_dtype(DType::F32)?;
            let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
            let normed = centered
                .broadcast_div(&(centered.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)?;
            let want = normed
                .broadcast_mul(&w)?
                .broadcast_add(&b)?
                .broadcast_mul(&(&scale + 1.0)?)?
                .broadcast_add(&shift)?;
            let got = layer_norm(&xs, Some(&w), Some(&b), Some(&scale), Some(&shift), 1e-6)?;
            let tol = BF16_ULP * want.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(max_diff(&got, &want)? <= tol, "layer norm {tokens}x{cols}");

            let want = (&x + ys.to_dtype(DType::F32)?.broadcast_mul(&gate)?)?;
            let got = gated_residual(&xs, &ys, &gate)?;
            let tol = BF16_ULP * want.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                max_diff(&got, &want)? <= tol,
                "gated residual {tokens}x{cols}"
            );

            let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
            let normed = x
                .broadcast_div(&rms)?
                .broadcast_mul(&w)?
                .to_dtype(DType::BF16)?;
            let angles = Tensor::rand(0f32, std::f32::consts::TAU, (tokens, head_dim / 2), &dev)?;
            let (cos, sin) = (angles.cos()?, angles.sin()?);
            let pairs = normed
                .to_dtype(DType::F32)?
                .reshape((1, tokens, heads, head_dim / 2, 2))?
                .transpose(1, 2)?;
            let x1 = pairs.narrow(4, 0, 1)?.squeeze(4)?;
            let x2 = pairs.narrow(4, 1, 1)?.squeeze(4)?;
            let (c, s) = (
                cos.unsqueeze(0)?.unsqueeze(0)?,
                sin.unsqueeze(0)?.unsqueeze(0)?,
            );
            let o1 = (x1.broadcast_mul(&c)? - x2.broadcast_mul(&s)?)?;
            let o2 = (x1.broadcast_mul(&s)? + x2.broadcast_mul(&c)?)?;
            let want = Tensor::stack(&[o1, o2], 4)?.reshape((1, heads, tokens, head_dim))?;
            let got = rms_norm_rope(&xs, &w, Some((&cos, &sin)), heads, 1e-6)?;
            let tol = BF16_ULP * want.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                max_diff(&got, &want)? <= tol,
                "rms norm rope {tokens}x{cols}"
            );
            let want = normed
                .reshape((1, tokens, heads, head_dim))?
                .transpose(1, 2)?;
            let got = rms_norm_rope(&xs, &w, None, heads, 1e-6)?;
            assert!(
                max_diff(&got, &want)? <= tol,
                "rms norm heads {tokens}x{cols}"
            );
        }
        Ok(())
    }
}
