//! The DiT blocks' fused passes on CUDA (`kernels/cuda/dit.cu`), in bf16 (TRELLIS.2) or f16 (Idol): LayerNorm with
//! AdaLN modulation, q/k normalization with the rotations (written straight in attention's `[batch, heads, tokens,
//! head_dim]` layout), and the gated residual. Each keeps the unfused path's arithmetic.

use half::{bf16, f16};
use inference_tensor::cuda_backend::cudarc::driver::{
    CudaSlice, CudaView, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use inference_tensor::cuda_backend::{CudaDType, CudaDevice, WrapErr};
use inference_tensor::{DType, Result, Shape, Storage, Tensor};

const MODULE: &str = "inference_recon_dit";
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dit.fatbin"));
// threads per `ln_mod` row; `LN_BLOCK` in the kernel source
const LN_BLOCK: u32 = 256;
const BLOCK: u32 = 256;
const WARP: u32 = 32;
// a null device pointer, for an input a kernel goes without
const NULL: u64 = 0;

/// Whether the fused passes apply: bf16 or f16 on a CUDA device.
pub fn applies(x: &Tensor) -> bool {
    x.device().is_cuda() && matches!(x.dtype(), DType::BF16 | DType::F16)
}

// an element type the kernels are built for, named by its kernels' suffix
trait Elem: CudaDType + DeviceRepr + ValidAsZeroBits + Copy {
    const SUFFIX: &'static str;
}

impl Elem for bf16 {
    const SUFFIX: &'static str = "bf16";
}

impl Elem for f16 {
    const SUFFIX: &'static str = "f16";
}

// runs `$body` with `$t` the element type of `$x`
macro_rules! by_dtype {
    ($x:expr, $t:ident => $body:expr) => {
        match $x.dtype() {
            DType::BF16 => {
                type $t = bf16;
                $body
            }
            DType::F16 => {
                type $t = f16;
                $body
            }
            other => {
                inference_tensor::bail!("the fused DiT passes take bf16 or f16, got {other:?}")
            }
        }
    };
}

// launches `$name` (suffixed with the element type) with `$cfg` and the arguments in its signature's order
macro_rules! launch {
    ($dev:expr, $t:ty, $name:expr, $cfg:expr, $($arg:expr),* $(,)?) => {{
        let name = format!("{}_{}", $name, <$t as Elem>::SUFFIX);
        let func = $dev.get_or_load_custom_image(&name, MODULE, IMAGE)?;
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

// the view of a held tensor's storage
fn cuda_view<'a, T: CudaDType>(
    held: &'a (
        inference_tensor::StorageRef<'a>,
        &'a inference_tensor::Layout,
    ),
) -> Result<CudaView<'a, T>> {
    let Storage::Cuda(storage) = &*held.0 else {
        inference_tensor::bail!("a fused DiT pass got a tensor off CUDA");
    };
    Ok(storage.as_cuda_slice::<T>()?.slice(held.1.start_offset()..))
}

fn output<T: Elem>(dev: &CudaDevice, out: CudaSlice<T>, shape: impl Into<Shape>) -> Tensor {
    Tensor::from((
        Storage::Cuda(T::wrap_cuda_slice(out, dev.clone())),
        shape.into(),
    ))
}

// the modulation's row stride: zero when one row serves the whole batch, as a broadcast does
fn mod_stride(m: &Tensor, c: usize) -> Result<i32> {
    Ok(if m.dim(0)? == 1 { 0 } else { (6 * c) as i32 })
}

fn cuda_device(t: &Tensor) -> Result<CudaDevice> {
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
    by_dtype!(x, T => ln_mod_as::<T>(x, affine, modulation, eps))
}

fn ln_mod_as<T: Elem>(
    x: &Tensor,
    affine: Option<(&Tensor, &Tensor)>,
    modulation: Option<(&Tensor, usize, usize)>,
    eps: f32,
) -> Result<Tensor> {
    let (batch, tokens, c) = x.dims3()?;
    let dev = cuda_device(x)?;
    let x = x.contiguous()?;
    view!(x, T, xg, xv);
    let mut out = unsafe { dev.alloc::<T>(batch * tokens * c) }?;
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
                dev, T, "ln_mod", cfg, &xv, &wv, &bv, &NULL, &0i32, &0i32, &0i32, &mut out, &c32,
                &rows, &eps
            );
            drop((wg, bg));
        }
        (None, Some((m, scale, shift))) => {
            let m = m.contiguous()?;
            view!(m, T, mg, mv);
            let (stride, scale_at, shift_at) =
                (mod_stride(&m, c)?, (scale * c) as i32, (shift * c) as i32);
            launch!(
                dev, T, "ln_mod", cfg, &xv, &NULL, &NULL, &mv, &stride, &scale_at, &shift_at,
                &mut out, &c32, &rows, &eps
            );
            drop(mg);
        }
        _ => inference_tensor::bail!("ln_mod takes an affine norm or a modulation"),
    }
    drop(xg);
    Ok(output(&dev, out, (batch, tokens, c)))
}

/// The rotations `heads` applies: f32 `(cos, sin)` over adjacent channel pairs.
#[derive(Clone, Copy)]
pub struct Rotation<'a> {
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    /// `[batch, tokens, heads, d/2]`, each head its own; otherwise `[tokens, d/2]`, shared by the batch and heads
    pub per_head: bool,
    /// rotate, round, then normalize (Idol); otherwise normalize, round, then rotate (TRELLIS.2)
    pub first: bool,
}

/// Part `part` of `src` `[batch, tokens, parts, heads, d]` plus its slice of `bias` as `[batch, heads, tokens, d]`:
/// F.normalize'd times `gain` `[heads, d]` and rounded when given, and rotated by `rope` when given.
pub fn heads(
    src: &Tensor,
    bias: &Tensor,
    part: usize,
    gain: Option<&Tensor>,
    rope: Option<Rotation>,
    eps: f32,
) -> Result<Tensor> {
    by_dtype!(src, T => heads_as::<T>(src, bias, part, gain, rope, eps))
}

fn heads_as<T: Elem>(
    src: &Tensor,
    bias: &Tensor,
    part: usize,
    gain: Option<&Tensor>,
    rope: Option<Rotation>,
    eps: f32,
) -> Result<Tensor> {
    let (batch, tokens, parts, heads, d) = src.dims5()?;
    let dev = cuda_device(src)?;
    let src = src.contiguous()?;
    view!(src, T, sg, sv);
    view!(bias, T, bg, bv);
    let mut out = unsafe { dev.alloc::<T>(batch * heads * tokens * d) }?;
    let rows = (batch * tokens * heads) as u32;
    let cfg = LaunchConfig {
        grid_dim: ((rows * WARP).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (token_stride, offset) = ((parts * heads * d) as i64, (part * heads * d) as i64);
    let dims = [batch as i32, tokens as i32, heads as i32, d as i32];
    let gain = gain.map(Tensor::contiguous).transpose()?;
    let pairs = (d / 2) as i64;
    let (rope, strides, first) = match rope {
        Some(r) => {
            let strides = if r.per_head {
                (
                    tokens as i64 * heads as i64 * pairs,
                    heads as i64 * pairs,
                    pairs,
                )
            } else {
                (0, pairs, 0)
            };
            (
                Some((r.cos.contiguous()?, r.sin.contiguous()?)),
                strides,
                i32::from(r.first),
            )
        }
        None => (None, (0, 0, 0), 0),
    };
    if rope.is_some() && gain.is_none() {
        inference_tensor::bail!("heads rotates only normalized heads");
    }
    let name = format!("head_norm_rope_{}", T::SUFFIX);
    let func = dev.get_or_load_custom_image(&name, MODULE, IMAGE)?;
    let mut builder = func.builder();
    builder.arg(&sv).arg(&bv).arg(&token_stride).arg(&offset);
    let gain_guard = gain.as_ref().map(|g| g.storage_and_layout());
    let gain_view = gain_guard.as_ref().map(cuda_view::<f32>).transpose()?;
    match &gain_view {
        Some(v) => builder.arg(v),
        None => builder.arg(&NULL),
    };
    let rope_guards = rope
        .as_ref()
        .map(|(c, s)| (c.storage_and_layout(), s.storage_and_layout()));
    let rope_views = rope_guards
        .as_ref()
        .map(|(c, s)| {
            Ok::<_, inference_tensor::Error>((cuda_view::<f32>(c)?, cuda_view::<f32>(s)?))
        })
        .transpose()?;
    match &rope_views {
        Some((c, s)) => builder.arg(c).arg(s),
        None => builder.arg(&NULL).arg(&NULL),
    };
    builder
        .arg(&strides.0)
        .arg(&strides.1)
        .arg(&strides.2)
        .arg(&first)
        .arg(&mut out);
    builder
        .arg(&dims[0])
        .arg(&dims[1])
        .arg(&dims[2])
        .arg(&dims[3])
        .arg(&eps);
    // SAFETY: the arguments follow the kernel's signature, with buffers sized for its counts
    unsafe { builder.launch(cfg) }.w()?;
    drop((gain_guard, rope_guards, sg, bg));
    Ok(output(&dev, out, (batch, heads, tokens, d)))
}

/// `x + (y + bias) * gate` (`gate` chunk `chunk` of `m` `[batch, 6c]`), or `x + (y + bias)` without one.
pub fn residual(
    x: &Tensor,
    y: &Tensor,
    bias: &Tensor,
    gate: Option<(&Tensor, usize)>,
) -> Result<Tensor> {
    by_dtype!(x, T => residual_as::<T>(x, y, bias, gate))
}

fn residual_as<T: Elem>(
    x: &Tensor,
    y: &Tensor,
    bias: &Tensor,
    gate: Option<(&Tensor, usize)>,
) -> Result<Tensor> {
    let (batch, tokens, c) = x.dims3()?;
    let dev = cuda_device(x)?;
    let (x, y) = (x.contiguous()?, y.contiguous()?);
    view!(x, T, xg, xv);
    view!(y, T, yg, yv);
    view!(bias, T, bg, bv);
    let n = (batch * tokens * c) as i64;
    let mut out = unsafe { dev.alloc::<T>(n as usize) }?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (c32, rows) = (c as i32, tokens as i32);
    match gate {
        Some((m, chunk)) => {
            let m = m.contiguous()?;
            view!(m, T, mg, mv);
            let (stride, at) = (mod_stride(&m, c)?, (chunk * c) as i32);
            launch!(
                dev,
                T,
                "gate_residual",
                cfg,
                &xv,
                &yv,
                &bv,
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
                T,
                "gate_residual",
                cfg,
                &xv,
                &yv,
                &bv,
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
    drop((xg, yg, bg));
    Ok(output(&dev, out, (batch, tokens, c)))
}

/// `gelu(x + bias)` (tanh form) for `x` `[batch, tokens, c]`, as the ops round it.
pub fn bias_gelu(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    by_dtype!(x, T => bias_gelu_as::<T>(x, bias))
}

fn bias_gelu_as<T: Elem>(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (batch, tokens, c) = x.dims3()?;
    let dev = cuda_device(x)?;
    let x = x.contiguous()?;
    view!(x, T, xg, xv);
    view!(bias, T, bg, bv);
    let n = (batch * tokens * c) as i64;
    let mut out = unsafe { dev.alloc::<T>(n as usize) }?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let c32 = c as i32;
    launch!(dev, T, "bias_gelu", cfg, &xv, &bv, &n, &c32, &mut out);
    drop((xg, bg));
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
            per_head_rotation_first_matches_the_ops,
            residual_matches_the_ops,
            bias_gelu_matches_the_ops
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
        let bias = Tensor::randn(0f32, 0.5, 3 * h * d, &dev)?.to_dtype(DType::BF16)?;
        let qkv_b = qkv.broadcast_add(&bias.reshape((3, h, d))?)?;
        let q = qkv_b.narrow(2, 0, 1)?.squeeze(2)?;
        let normed = {
            let x = q.to_dtype(DType::F32)?;
            let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-12)?;
            x.broadcast_div(&norm)?
                .broadcast_mul(&gain)?
                .to_dtype(DType::BF16)?
        };
        let want = rope.apply(&normed)?.transpose(1, 2)?;
        let rot = Rotation {
            cos,
            sin,
            per_head: false,
            first: false,
        };
        let got = heads(&qkv, &bias, 0, Some(&gain), Some(rot), 1e-12)?;
        assert!(
            worst(&got, &want)? <= TOLERANCE,
            "rotated: {}",
            worst(&got, &want)?
        );
        let v = qkv_b.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;
        assert_eq!(
            worst(&heads(&qkv, &bias, 2, None, None, 1e-12)?, &v)?,
            0.0,
            "v is copied with its bias"
        );
        Ok(())
    }

    // Idol's order in f16: each head rotated by its own angles in f32 and rounded, then F.normalize'd times its gain
    fn per_head_rotation_first_matches_the_ops() -> Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let (b, n, h, d) = (2, 23, 4, 64);
        let qkv = Tensor::randn(0f32, 1.0, (b, n, 3, h, d), &dev)?.to_dtype(DType::F16)?;
        let gain = Tensor::randn(8f32, 1.0, (h, d), &dev)?;
        let angles = Tensor::randn(0f32, 3.0, (b, n, h, d / 2), &dev)?;
        let (cos, sin) = (angles.cos()?, angles.sin()?);
        let bias = Tensor::randn(0f32, 0.5, 3 * h * d, &dev)?.to_dtype(DType::F16)?;
        let k = qkv
            .broadcast_add(&bias.reshape((3, h, d))?)?
            .narrow(2, 1, 1)?
            .squeeze(2)?;
        let pairs = k.to_dtype(DType::F32)?.reshape((b, n, h, d / 2, 2))?;
        let (x0, x1) = (
            pairs.narrow(4, 0, 1)?.squeeze(4)?,
            pairs.narrow(4, 1, 1)?.squeeze(4)?,
        );
        let rotated = Tensor::stack(
            &[
                ((&x0 * &cos)? - (&x1 * &sin)?)?,
                ((&x0 * &sin)? + (&x1 * &cos)?)?,
            ],
            4,
        )?
        .reshape((b, n, h, d))?
        .to_dtype(DType::F16)?
        .to_dtype(DType::F32)?;
        let norm = rotated
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .maximum(1e-12)?;
        let want = rotated
            .broadcast_div(&norm)?
            .broadcast_mul(&gain)?
            .to_dtype(DType::F16)?
            .transpose(1, 2)?;
        let rot = Rotation {
            cos: &cos,
            sin: &sin,
            per_head: true,
            first: true,
        };
        let got = heads(&qkv, &bias, 1, Some(&gain), Some(rot), 1e-12)?;
        // one f16 rounding at the gain's magnitude
        let tolerance = 8.0 * 2f32.powi(-10) * 2.0;
        assert!(
            worst(&got, &want)? <= tolerance,
            "per head: {}",
            worst(&got, &want)?
        );
        Ok(())
    }

    fn residual_matches_the_ops() -> Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let (b, n, c) = (2, 41, 96);
        let x = Tensor::randn(0f32, 1.0, (b, n, c), &dev)?.to_dtype(DType::BF16)?;
        let y_raw = Tensor::randn(0f32, 1.0, (b, n, c), &dev)?.to_dtype(DType::BF16)?;
        let bias = Tensor::randn(0f32, 0.5, c, &dev)?.to_dtype(DType::BF16)?;
        let y = y_raw.broadcast_add(&bias)?;
        let m = Tensor::randn(0f32, 1.0, (b, 6 * c), &dev)?.to_dtype(DType::BF16)?;
        let gate = m.narrow(1, 2 * c, c)?.unsqueeze(1)?;
        let want = (&x + y.broadcast_mul(&gate)?)?;
        assert_eq!(
            worst(&residual(&x, &y_raw, &bias, Some((&m, 2)))?, &want)?,
            0.0,
            "gated"
        );
        let m1 = m.narrow(0, 0, 1)?;
        let want = (&x + y.broadcast_mul(&m1.narrow(1, 2 * c, c)?.unsqueeze(1)?)?)?;
        assert_eq!(
            worst(&residual(&x, &y_raw, &bias, Some((&m1, 2)))?, &want)?,
            0.0,
            "broadcast gate"
        );
        assert_eq!(
            worst(&residual(&x, &y_raw, &bias, None)?, &(&x + &y)?)?,
            0.0,
            "plain"
        );
        Ok(())
    }

    fn bias_gelu_matches_the_ops() -> Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let (b, n, c) = (2, 33, 160);
        let x = Tensor::randn(0f32, 2.0, (b, n, c), &dev)?.to_dtype(DType::BF16)?;
        let bias = Tensor::randn(0f32, 0.5, c, &dev)?.to_dtype(DType::BF16)?;
        let want = x.broadcast_add(&bias)?.gelu()?;
        assert_eq!(worst(&bias_gelu(&x, &bias)?, &want)?, 0.0);
        Ok(())
    }
}
