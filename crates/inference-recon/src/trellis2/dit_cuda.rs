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
// tokens one thread of `value_abs_sums_bf16` sums before adding its partial sum to the column's
const SUM_TOKENS: u32 = 256;
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

// a device f32 1.0, for a scale left out
fn unit(dev: &inference_tensor::cuda_backend::CudaDevice) -> Result<Tensor> {
    Tensor::ones(1, DType::F32, &inference_tensor::Device::Cuda(dev.clone()))
}

/// The power of two part `part`'s values (as [`heads`] makes them from `src` `[batch, tokens, parts, heads, d]` and
/// `bias`) are divided by before attention: f32 `[scale, 1 / scale]` on the device, no sync. fattn sums the
/// softmax-weighted values in f16 before normalizing, so a row spread over many keys can pass f16's 65504; the scale is
/// the largest per-(batch, head, channel) sum of |v| over the tokens, over `limit`, rounded up to a power of two (1 when
/// it fits), which divides and multiplies back exactly.
pub fn value_scale(src: &Tensor, bias: &Tensor, part: usize, limit: f32) -> Result<Tensor> {
    let (batch, tokens, parts, heads, d) = src.dims5()?;
    let dev = cuda_device(src)?;
    let src = src.contiguous()?;
    view!(src, bf16, sg, sv);
    view!(bias, bf16, bg, bv);
    let width = heads * d;
    let mut sums = dev.alloc_zeros::<f32>(batch * width)?;
    let (token_stride, offset) = ((parts * width) as i64, (part * width) as i64);
    let cfg = LaunchConfig {
        grid_dim: (
            (width as u32).div_ceil(BLOCK),
            (tokens as u32).div_ceil(SUM_TOKENS),
            batch as u32,
        ),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (tokens32, width32, per) = (tokens as i32, width as i32, SUM_TOKENS as i32);
    launch!(
        dev,
        "value_abs_sums_bf16",
        cfg,
        &sv,
        &bv,
        &token_stride,
        &offset,
        &tokens32,
        &width32,
        &per,
        &mut sums
    );
    drop((sg, bg));
    let mut out = unsafe { dev.alloc::<f32>(2) }?;
    let one_block = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let n = (batch * width) as i32;
    launch!(
        dev,
        "value_scale_finish",
        one_block,
        &sums,
        &n,
        &limit,
        &mut out
    );
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::F32(out),
            device: dev,
        }),
        inference_tensor::Shape::from(2),
    )))
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

/// Part `part` of `src` `[batch, tokens, parts, heads, d]` (plus `bias`) as `[batch, heads, tokens, d]`: F.normalize'd
/// times `gain` `[heads, d]` and rounded when given, then rotated by `rope` `(cos, sin)` `[tokens, d/2]` when given,
/// and times `inv_scale` (f32 `[1]`, a power of two: [`value_scale`]'s second value) when given.
pub fn heads(
    src: &Tensor,
    bias: &Tensor,
    part: usize,
    gain: Option<&Tensor>,
    rope: Option<(&Tensor, &Tensor)>,
    inv_scale: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let (batch, tokens, parts, heads, d) = src.dims5()?;
    let dev = cuda_device(src)?;
    let src = src.contiguous()?;
    view!(src, bf16, sg, sv);
    view!(bias, bf16, bg, bv);
    let one = unit(&dev)?;
    let inv_scale = inv_scale.unwrap_or(&one);
    view!(inv_scale, f32, ig, iv);
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
                &bv,
                &iv,
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
                &bv,
                &iv,
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
                &bv,
                &iv,
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
    drop((sg, bg, ig));
    Ok(output(&dev, out, (batch, heads, tokens, d)))
}

/// `x + (y * scale + bias) * gate` (`gate` chunk `chunk` of `m` `[batch, 6c]`), or without the gate, in bf16; `scale`
/// (f32 `[1]`, a power of two) undoes [`value_scale`] on an attention output, and is 1 when not given.
pub fn residual(
    x: &Tensor,
    y: &Tensor,
    bias: &Tensor,
    scale: Option<&Tensor>,
    gate: Option<(&Tensor, usize)>,
) -> Result<Tensor> {
    let (batch, tokens, c) = x.dims3()?;
    let dev = cuda_device(x)?;
    let (x, y) = (x.contiguous()?, y.contiguous()?);
    view!(x, bf16, xg, xv);
    view!(y, bf16, yg, yv);
    view!(bias, bf16, bg, bv);
    let one = unit(&dev)?;
    let scale = scale.unwrap_or(&one);
    view!(scale, f32, scg, scv);
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
                &bv,
                &scv,
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
                &bv,
                &scv,
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
    drop((xg, yg, bg, scg));
    Ok(output(&dev, out, (batch, tokens, c)))
}

/// `gelu(x + bias)` (tanh form) for `x` `[batch, tokens, c]`, as the ops round it.
pub fn bias_gelu(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (batch, tokens, c) = x.dims3()?;
    let dev = cuda_device(x)?;
    let x = x.contiguous()?;
    view!(x, bf16, xg, xv);
    view!(bias, bf16, bg, bv);
    let n = (batch * tokens * c) as i64;
    let mut out = unsafe { dev.alloc::<bf16>(n as usize) }?;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let c32 = c as i32;
    launch!(dev, "bias_gelu_bf16", cfg, &xv, &bv, &n, &c32, &mut out);
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
            residual_matches_the_ops,
            value_scale_is_the_power_of_two_the_sums_need,
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
        let got = heads(&qkv, &bias, 0, Some(&gain), Some((cos, sin)), None, 1e-12)?;
        assert!(
            worst(&got, &want)? <= TOLERANCE,
            "rotated: {}",
            worst(&got, &want)?
        );
        let v = qkv_b.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;
        assert_eq!(
            worst(&heads(&qkv, &bias, 2, None, None, None, 1e-12)?, &v)?,
            0.0,
            "v is copied with its bias"
        );
        let eighth = Tensor::new(&[0.125f32], &dev)?;
        assert_eq!(
            worst(
                &heads(&qkv, &bias, 2, None, None, Some(&eighth), 1e-12)?,
                &(&v / 8.0)?
            )?,
            0.0,
            "v divided by a power of two, exactly"
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
            worst(&residual(&x, &y_raw, &bias, None, Some((&m, 2)))?, &want)?,
            0.0,
            "gated"
        );
        let m1 = m.narrow(0, 0, 1)?;
        let want = (&x + y.broadcast_mul(&m1.narrow(1, 2 * c, c)?.unsqueeze(1)?)?)?;
        assert_eq!(
            worst(&residual(&x, &y_raw, &bias, None, Some((&m1, 2)))?, &want)?,
            0.0,
            "broadcast gate"
        );
        assert_eq!(
            worst(&residual(&x, &y_raw, &bias, None, None)?, &(&x + &y)?)?,
            0.0,
            "plain"
        );
        // an attention output over values divided by 8 comes back multiplied by 8
        let eight = Tensor::new(&[8f32], &dev)?;
        let scaled = (&y_raw / 8.0)?;
        assert_eq!(
            worst(
                &residual(&x, &scaled, &bias, Some(&eight), None)?,
                &(&x + &y)?
            )?,
            0.0,
            "scaled"
        );
        Ok(())
    }

    fn value_scale_is_the_power_of_two_the_sums_need() -> Result<()> {
        let Some(dev) = device() else { return Ok(()) };
        let (b, n, h, d) = (2, 300, 3, 64);
        // part 2 of each token: |v + bias| = 20 everywhere but one channel of batch 1, head 2 at 50: a column sum of
        // 15000, which over 2048 needs 8 (7.3 rounded up)
        let mut values = vec![1f32; b * n * 3 * h * d];
        for bi in 0..b {
            for t in 0..n {
                for k in 0..h * d {
                    let at = ((bi * n + t) * 3 + 2) * h * d + k;
                    values[at] = if bi == 1 && k == 2 * d + 5 {
                        49.0
                    } else {
                        -21.0
                    };
                }
            }
        }
        let qkv = Tensor::from_vec(values, (b, n, 3, h, d), &dev)?.to_dtype(DType::BF16)?;
        let bias = Tensor::ones(3 * h * d, DType::BF16, &dev)?;
        let scales = value_scale(&qkv, &bias, 2, 2048.0)?.to_vec1::<f32>()?;
        assert_eq!(scales, [8.0, 0.125]);
        let fits = value_scale(&qkv, &bias, 2, 1.0e6)?.to_vec1::<f32>()?;
        assert_eq!(
            fits,
            [1.0, 1.0],
            "sums under the limit keep the values as they are"
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
