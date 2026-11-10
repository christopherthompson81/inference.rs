//! The DiT blocks' fused bf16 passes on CUDA (`kernels/cuda/dit.cu`): LayerNorm with AdaLN modulation, q/k
//! normalization with the rotations (written straight in attention's `[batch, heads, tokens, head_dim]` layout), and
//! the gated residual. Each keeps the unfused path's arithmetic.

use half::bf16;
use inference_tensor::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
use inference_tensor::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use inference_tensor::{DType, Result, Storage, Tensor};

const MODULE: &str = "inference_recon_dit";
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dit.fatbin"));
// threads per `ln_mod_bf16` row; `LN_BLOCK` in the kernel source
const LN_BLOCK: u32 = 256;
const BLOCK: u32 = 256;
const WARP: u32 = 32;
// a null device pointer, for an input a kernel goes without
const NULL: u64 = 0;

/// Whether the fused passes apply: bf16 on a CUDA device.
pub fn applies(x: &Tensor) -> bool {
    x.device().is_cuda() && x.dtype() == DType::BF16
}

// launches `$name` with `$cfg` and the arguments in its signature's order
macro_rules! launch {
    ($dev:expr, $name:expr, $cfg:expr, $($arg:expr),* $(,)?) => {{
        let func = $dev.get_or_load_custom_image($name, MODULE, IMAGE)?;
        let mut builder = func.builder();
        $( builder.arg($arg); )*
        // SAFETY: the arguments follow the kernel's signature, with buffers sized for its counts
        unsafe { builder.launch($cfg) }.w()?;
    }};
}

// a contiguous tensor's storage, held while its view is in use
macro_rules! view {
    ($t:expr, $ty:ty, $guard:ident, $view:ident) => {
        let $guard = $t.storage_and_layout();
        let start = $guard.1.start_offset();
        let Storage::Cuda(storage) = &*$guard.0 else {
            inference_tensor::bail!("a fused DiT pass got a tensor off CUDA");
        };
        let $view: CudaView<$ty> = storage.as_cuda_slice::<$ty>()?.slice(start..);
    };
}

fn output(
    dev: &inference_tensor::cuda_backend::CudaDevice,
    out: inference_tensor::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    shape: impl Into<inference_tensor::Shape>,
) -> Tensor {
    Tensor::from((
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::BF16(out),
            device: dev.clone(),
        }),
        shape.into(),
    ))
}

// the modulation's row stride: zero when one row serves the whole batch, as a broadcast does
fn mod_stride(m: &Tensor, c: usize) -> Result<i32> {
    Ok(if m.dim(0)? == 1 { 0 } else { (6 * c) as i32 })
}

fn cuda_device(t: &Tensor) -> Result<inference_tensor::cuda_backend::CudaDevice> {
    match t.device() {
        inference_tensor::Device::Cuda(dev) => Ok(dev.clone()),
        _ => inference_tensor::bail!("a fused DiT pass got a tensor off CUDA"),
    }
}

/// `x` `[batch, tokens, c]` LayerNorm'd (with `affine`'s f32 weight and bias when given) and rounded, then when
/// `modulation` `(m [batch, 6c], scale chunk, shift chunk)` is given `h * (1 + scale) + shift`.
pub fn ln_mod(
    x: &Tensor,
    affine: Option<(&Tensor, &Tensor)>,
    modulation: Option<(&Tensor, usize, usize)>,
    eps: f32,
) -> Result<Tensor> {
    let (batch, tokens, c) = x.dims3()?;
    let dev = cuda_device(x)?;
    let x = x.contiguous()?;
    view!(x, bf16, xg, xv);
    let mut out = unsafe { dev.alloc::<bf16>(batch * tokens * c) }?;
    let cfg = LaunchConfig {
        grid_dim: ((batch * tokens) as u32, 1, 1),
        block_dim: (LN_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (c32, rows) = (c as i32, tokens as i32);
    match (affine, modulation) {
        (Some((w, bias)), None) => {
            view!(w, f32, wg, wv);
            view!(bias, f32, bg, bv);
            launch!(
                dev,
                "ln_mod_bf16",
                cfg,
                &xv,
                &wv,
                &bv,
                &NULL,
                &0i32,
                &0i32,
                &0i32,
                &mut out,
                &c32,
                &rows,
                &eps
            );
            drop((wg, bg));
        }
        (None, Some((m, scale, shift))) => {
            let m = m.contiguous()?;
            view!(m, bf16, mg, mv);
            let (stride, scale_at, shift_at) =
                (mod_stride(&m, c)?, (scale * c) as i32, (shift * c) as i32);
            launch!(
                dev,
                "ln_mod_bf16",
                cfg,
                &xv,
                &NULL,
                &NULL,
                &mv,
                &stride,
                &scale_at,
                &shift_at,
                &mut out,
                &c32,
                &rows,
                &eps
            );
            drop(mg);
        }
        _ => inference_tensor::bail!("ln_mod takes an affine norm or a modulation"),
    }
    drop(xg);
    Ok(output(&dev, out, (batch, tokens, c)))
}

/// Part `part` of `src` `[batch, tokens, parts, heads, d]` as `[batch, heads, tokens, d]`: F.normalize'd times
/// `gain` `[heads, d]` and rounded when given, then rotated by `rope` `(cos, sin)` `[tokens, d/2]` when given.
pub fn heads(
    src: &Tensor,
    part: usize,
    gain: Option<&Tensor>,
    rope: Option<(&Tensor, &Tensor)>,
    eps: f32,
) -> Result<Tensor> {
    let (batch, tokens, parts, heads, d) = src.dims5()?;
    let dev = cuda_device(src)?;
    let src = src.contiguous()?;
    view!(src, bf16, sg, sv);
    let mut out = unsafe { dev.alloc::<bf16>(batch * heads * tokens * d) }?;
    let rows = (batch * tokens * heads) as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows * WARP).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (token_stride, offset) = ((parts * heads * d) as i64, (part * heads * d) as i64);
    let dims = [batch as i32, tokens as i32, heads as i32, d as i32];
    let gain = gain.map(Tensor::contiguous).transpose()?;
    let rope = rope
        .map(|(c, s)| Ok::<_, inference_tensor::Error>((c.contiguous()?, s.contiguous()?)))
        .transpose()?;
    match (&gain, &rope) {
        (Some(g), Some((c, s))) => {
            view!(g, f32, gg, gv);
            view!(c, f32, cg, cv);
            view!(s, f32, s2, s2v);
            launch!(
                dev,
                "head_norm_rope_bf16",
                cfg,
                &sv,
                &token_stride,
                &offset,
                &gv,
                &cv,
                &s2v,
                &mut out,
                &dims[0],
                &dims[1],
                &dims[2],
                &dims[3],
                &eps
            );
            drop((gg, cg, s2));
        }
        (Some(g), None) => {
            view!(g, f32, gg, gv);
            launch!(
                dev,
                "head_norm_rope_bf16",
                cfg,
                &sv,
                &token_stride,
                &offset,
                &gv,
                &NULL,
                &NULL,
                &mut out,
                &dims[0],
                &dims[1],
                &dims[2],
                &dims[3],
                &eps
            );
            drop(gg);
        }
        (None, None) => {
            launch!(
                dev,
                "head_norm_rope_bf16",
                cfg,
                &sv,
                &token_stride,
                &offset,
                &NULL,
                &NULL,
                &NULL,
                &mut out,
                &dims[0],
                &dims[1],
                &dims[2],
                &dims[3],
                &eps
            );
        }
        (None, Some(_)) => inference_tensor::bail!("heads rotates only normalized heads"),
    }
    drop(sg);
    Ok(output(&dev, out, (batch, heads, tokens, d)))
}

/// `x + y * gate` (`gate` chunk `chunk` of `m` `[batch, 6c]`), or `x + y` without one, in bf16.
pub fn residual(x: &Tensor, y: &Tensor, gate: Option<(&Tensor, usize)>) -> Result<Tensor> {
    let (batch, tokens, c) = x.dims3()?;
    let dev = cuda_device(x)?;
    let (x, y) = (x.contiguous()?, y.contiguous()?);
    view!(x, bf16, xg, xv);
    view!(y, bf16, yg, yv);
    let n = (batch * tokens * c) as i64;
    let mut out = unsafe { dev.alloc::<bf16>(n as usize) }?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (c32, rows) = (c as i32, tokens as i32);
    match gate {
        Some((m, chunk)) => {
            let m = m.contiguous()?;
            view!(m, bf16, mg, mv);
            let (stride, at) = (mod_stride(&m, c)?, (chunk * c) as i32);
            launch!(
                dev,
                "gate_residual_bf16",
                cfg,
                &xv,
                &yv,
                &mv,
                &stride,
                &at,
                &mut out,
                &n,
                &c32,
                &rows
            );
            drop(mg);
        }
        None => {
            launch!(
                dev,
                "gate_residual_bf16",
                cfg,
                &xv,
                &yv,
                &NULL,
                &0i32,
                &0i32,
                &mut out,
                &n,
                &c32,
                &rows
            );
        }
    }
    drop((xg, yg));
    Ok(output(&dev, out, (batch, tokens, c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::{D, Device};

    inference_tensor::gpu_cases!(
        cuda_cases,
        [
            ln_mod_matches_the_ops,
            heads_match_the_ops,
            residual_matches_the_ops
        ]
    );

    const EPS: f32 = 1e-6;
    // one bf16 rounding of a value of order one
    const TOLERANCE: f32 = 1.6e-2;

    fn device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn worst(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()
    }

    fn ln_mod_matches_the_ops() -> Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let (b, n, c) = (2, 37, 192);
        let x = Tensor::randn(0f32, 1.0, (b, n, c), &dev)?.to_dtype(DType::BF16)?;
        let m = Tensor::randn(0f32, 0.5, (b, 6 * c), &dev)?.to_dtype(DType::BF16)?;
        let norm = inference_tensor::nn::LayerNorm::new(
            Tensor::randn(1f32, 0.1, c, &dev)?,
            Tensor::randn(0f32, 0.1, c, &dev)?,
            f64::from(EPS),
        );
        let plain = |x: &Tensor| -> Result<Tensor> {
            let x = x.to_dtype(DType::F32)?;
            let x = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
            let var = x.sqr()?.mean_keepdim(D::Minus1)?;
            x.broadcast_div(&(var + f64::from(EPS))?.sqrt()?)
        };
        let part = |i: usize| m.narrow(1, i * c, c)?.unsqueeze(1);
        let want = plain(&x)?
            .to_dtype(DType::BF16)?
            .broadcast_mul(&(part(1)? + 1.0)?)?
            .broadcast_add(&part(0)?)?;
        let got = ln_mod(&x, None, Some((&m, 1, 0)), EPS)?;
        assert!(
            worst(&got, &want)? <= TOLERANCE,
            "modulated: {}",
            worst(&got, &want)?
        );
        // one modulation row broadcast over the batch, as the guided pass runs it
        let m1 = m.narrow(0, 0, 1)?;
        let part1 = |i: usize| m1.narrow(1, i * c, c)?.unsqueeze(1);
        let want = plain(&x)?
            .to_dtype(DType::BF16)?
            .broadcast_mul(&(part1(1)? + 1.0)?)?
            .broadcast_add(&part1(0)?)?;
        let got = ln_mod(&x, None, Some((&m1, 1, 0)), EPS)?;
        assert!(
            worst(&got, &want)? <= TOLERANCE,
            "broadcast: {}",
            worst(&got, &want)?
        );
        let want = inference_tensor::nn::Module::forward(&norm, &x.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16)?;
        let got = ln_mod(
            &x,
            Some((norm.weight(), norm.bias().expect("a bias"))),
            None,
            EPS,
        )?;
        assert!(
            worst(&got, &want)? <= TOLERANCE,
            "affine: {}",
            worst(&got, &want)?
        );
        Ok(())
    }

    fn heads_match_the_ops() -> Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let (b, n, h, d) = (2, 29, 3, 128);
        let qkv = Tensor::randn(0f32, 1.0, (b, n, 3, h, d), &dev)?.to_dtype(DType::BF16)?;
        let gain = Tensor::randn(1f32, 0.2, (h, d), &dev)?;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| [i as f32, (i % 5) as f32, (i % 3) as f32])
            .collect();
        let rope = crate::trellis2::dit::Rope::new(&positions, d, &dev)?;
        let (cos, sin) = rope.tables();
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let normed = {
            let x = q.to_dtype(DType::F32)?;
            let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-12)?;
            x.broadcast_div(&norm)?
                .broadcast_mul(&gain)?
                .to_dtype(DType::BF16)?
        };
        let want = rope.apply(&normed)?.transpose(1, 2)?;
        let got = heads(&qkv, 0, Some(&gain), Some((cos, sin)), 1e-12)?;
        assert!(
            worst(&got, &want)? <= TOLERANCE,
            "rotated: {}",
            worst(&got, &want)?
        );
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;
        assert_eq!(
            worst(&heads(&qkv, 2, None, None, 1e-12)?, &v)?,
            0.0,
            "v is copied"
        );
        Ok(())
    }

    fn residual_matches_the_ops() -> Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let (b, n, c) = (2, 41, 96);
        let x = Tensor::randn(0f32, 1.0, (b, n, c), &dev)?.to_dtype(DType::BF16)?;
        let y = Tensor::randn(0f32, 1.0, (b, n, c), &dev)?.to_dtype(DType::BF16)?;
        let m = Tensor::randn(0f32, 1.0, (b, 6 * c), &dev)?.to_dtype(DType::BF16)?;
        let gate = m.narrow(1, 2 * c, c)?.unsqueeze(1)?;
        let want = (&x + y.broadcast_mul(&gate)?)?;
        assert_eq!(
            worst(&residual(&x, &y, Some((&m, 2)))?, &want)?,
            0.0,
            "gated"
        );
        let m1 = m.narrow(0, 0, 1)?;
        let want = (&x + y.broadcast_mul(&m1.narrow(1, 2 * c, c)?.unsqueeze(1)?)?)?;
        assert_eq!(
            worst(&residual(&x, &y, Some((&m1, 2)))?, &want)?,
            0.0,
            "broadcast gate"
        );
        assert_eq!(worst(&residual(&x, &y, None)?, &(&x + &y)?)?, 0.0, "plain");
        Ok(())
    }
}
