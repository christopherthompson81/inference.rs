//! The sparse convolution's im2col gather on CUDA (`kernels/cuda/sparse.cu`): whole rows in 16-byte vectors, where the
//! generic `index_select` moves one element a thread.

use half::f16;
use inference_tensor::cuda_backend::cudarc::driver::{
    CudaSlice, CudaView, DeviceRepr, LaunchConfig, PushKernelArg,
};
use inference_tensor::cuda_backend::{CudaDevice, CudaStorage, CudaStorageSlice, WrapErr};
use inference_tensor::{DType, Device, Result, Storage, Tensor};

use super::sparse::TAPS;

const MODULE: &str = "inference_recon_sparse";
static IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sparse.fatbin"));
const VECTOR_BYTES: usize = 16;
const BLOCK: u32 = 256;
// blocks a launch caps at; the kernel strides over the rest
const MAX_BLOCKS: u64 = 1 << 16;
const WARP: u32 = 32;
// rows this wide or wider take a block of this many threads in the tensor layernorm kernel, narrower a warp
const LN_WIDE: usize = 1024;
// a null device pointer, for an input a kernel goes without
const NULL: u64 = 0;

/// Whether [`gather`] takes rows of `feats` `[cells, c]`: on CUDA, each row a whole number of vectors.
pub fn applies(feats: &Tensor) -> Result<bool> {
    Ok(feats.device().is_cuda()
        && (feats.dim(1)? * feats.dtype().size_in_bytes()).is_multiple_of(VECTOR_BYTES))
}

/// Rows `index[start..start + rows]` (u32) of `feats` `[cells, c]`, zeros for an index of `missing`: `[rows, c]`.
pub fn gather(
    feats: &Tensor,
    index: &Tensor,
    start: usize,
    rows: usize,
    missing: u32,
) -> Result<Tensor> {
    match feats.dtype() {
        DType::F16 => {
            gather_as::<half::f16>(feats, index, start, rows, missing, CudaStorageSlice::F16)
        }
        DType::BF16 => {
            gather_as::<half::bf16>(feats, index, start, rows, missing, CudaStorageSlice::BF16)
        }
        DType::F32 => gather_as::<f32>(feats, index, start, rows, missing, CudaStorageSlice::F32),
        dtype => inference_tensor::bail!("no sparse gather for {dtype:?}"),
    }
}

fn gather_as<T: DeviceRepr + inference_tensor::cuda_backend::CudaDType>(
    feats: &Tensor,
    index: &Tensor,
    start: usize,
    rows: usize,
    missing: u32,
    wrap: fn(CudaSlice<T>) -> CudaStorageSlice,
) -> Result<Tensor> {
    let Device::Cuda(dev) = feats.device() else {
        inference_tensor::bail!("the sparse gather needs CUDA");
    };
    let dev: CudaDevice = dev.clone();
    let c = feats.dim(1)?;
    let feats = feats.contiguous()?;
    let (fs, fl) = feats.storage_and_layout();
    let (is, il) = index.storage_and_layout();
    let (Storage::Cuda(fs), Storage::Cuda(is)) = (&*fs, &*is) else {
        inference_tensor::bail!("the sparse gather needs CUDA");
    };
    let src: CudaView<T> = fs.as_cuda_slice::<T>()?.slice(fl.start_offset()..);
    let ids: CudaView<u32> = is
        .as_cuda_slice::<u32>()?
        .slice(il.start_offset() + start..);
    let row_vecs = (c * std::mem::size_of::<T>() / VECTOR_BYTES) as u32;
    // SAFETY: the kernel writes every element
    let mut out = unsafe { dev.alloc::<T>((rows * c).max(1)) }?;
    let total = rows as u64 * row_vecs as u64;
    if total > 0 {
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(BLOCK as u64).min(MAX_BLOCKS) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_custom_image("gather_rows", MODULE, IMAGE)?;
        let mut builder = func.builder();
        let rows = rows as u64;
        builder.arg(&src);
        builder.arg(&ids);
        builder.arg(&rows);
        builder.arg(&row_vecs);
        builder.arg(&missing);
        builder.arg(&mut out);
        // SAFETY: the arguments follow the kernel's signature; `ids` holds `rows` indices below `feats`' rows or `missing`
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage {
            slice: wrap(out),
            device: dev,
        }),
        inference_tensor::Shape::from((rows, c)),
    )))
}

fn u32_view<'a>(
    guard: &'a (
        inference_tensor::StorageRef<'a>,
        &'a inference_tensor::Layout,
    ),
) -> Result<CudaView<'a, u32>> {
    let Storage::Cuda(s) = &*guard.0 else {
        inference_tensor::bail!("the sparse kernels need CUDA");
    };
    Ok(s.as_cuda_slice::<u32>()?.slice(guard.1.start_offset()..))
}

/// The neighbour table of a subdivided level from its parents' (`parent` `[parents * 27]`, `parent_missing` for an
/// inactive tap): `rows` `[children]` the parent row * 8 + child slot each child came from, `slots` `[parents * 8]` the
/// child each slot became or `children` when it was not kept. `[children * 27]`, `children` for an inactive tap.
pub fn child_neighbors(
    parent: &Tensor,
    parent_missing: u32,
    rows: &Tensor,
    slots: &Tensor,
    children: usize,
) -> Result<Tensor> {
    let Device::Cuda(dev) = parent.device() else {
        inference_tensor::bail!("the sparse kernels need CUDA");
    };
    let dev: CudaDevice = dev.clone();
    let (pg, rg, sg) = (
        parent.storage_and_layout(),
        rows.storage_and_layout(),
        slots.storage_and_layout(),
    );
    let (pv, rv, sv) = (u32_view(&pg)?, u32_view(&rg)?, u32_view(&sg)?);
    let total = (children * TAPS) as u64;
    // SAFETY: the kernel writes every element
    let mut out = unsafe { dev.alloc::<u32>((total as usize).max(1)) }?;
    if total > 0 {
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(BLOCK as u64).min(MAX_BLOCKS) as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_custom_image("child_neighbors", MODULE, IMAGE)?;
        let mut builder = func.builder();
        let n = children as u32;
        builder.arg(&pv);
        builder.arg(&parent_missing);
        builder.arg(&rv);
        builder.arg(&sv);
        builder.arg(&n);
        builder.arg(&mut out);
        // SAFETY: the arguments follow the kernel's signature; the tables hold indices below their sentinels
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::U32(out),
            device: dev,
        }),
        inference_tensor::Shape::from(children * TAPS),
    )))
}

/// Whether the decoder's fused passes apply: f16 on CUDA.
pub fn fuses(x: &Tensor) -> bool {
    x.device().is_cuda() && x.dtype() == DType::F16
}

// a contiguous tensor's storage, held while its view is in use
macro_rules! view {
    ($t:expr, $ty:ty, $guard:ident, $view:ident) => {
        let $guard = $t.storage_and_layout();
        let Storage::Cuda(storage) = &*$guard.0 else {
            inference_tensor::bail!("the sparse kernels need CUDA");
        };
        let $view: CudaView<$ty> = storage
            .as_cuda_slice::<$ty>()?
            .slice($guard.1.start_offset()..);
    };
}

fn cuda(t: &Tensor) -> Result<CudaDevice> {
    match t.device() {
        Device::Cuda(dev) => Ok(dev.clone()),
        _ => inference_tensor::bail!("the sparse kernels need CUDA"),
    }
}

fn f16_tensor(dev: CudaDevice, out: CudaSlice<f16>, shape: (usize, usize)) -> Tensor {
    Tensor::from((
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::F16(out),
            device: dev,
        }),
        inference_tensor::Shape::from(shape),
    ))
}

fn elementwise(n: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK as u64).clamp(1, MAX_BLOCKS) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `x` `[rows, c]` (f16, plus `bias` first when given) LayerNorm'd in f32 with `alpha` and `beta` (f32), back in f16,
/// then silu'd when `act`: the ops' result, rounding for rounding.
pub fn layer_norm(
    x: &Tensor,
    bias: Option<&Tensor>,
    alpha: &Tensor,
    beta: &Tensor,
    eps: f32,
    act: bool,
) -> Result<Tensor> {
    let dev = cuda(x)?;
    let (rows, c) = x.dims2()?;
    let x = x.contiguous()?;
    view!(x, f16, xg, xv);
    view!(alpha, f32, ag, av);
    view!(beta, f32, bg, bv);
    // SAFETY: the kernel writes every element
    let mut out = unsafe { dev.alloc::<f16>((rows * c).max(1)) }?;
    if rows > 0 {
        // the block the tensor kernel takes, so the sums run in its order
        let block = if c < LN_WIDE { WARP } else { LN_WIDE as u32 };
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_custom_image("ln_f16", MODULE, IMAGE)?;
        let mut builder = func.builder();
        let (c32, block32, act) = (c as i32, block as i32, act as i32);
        builder.arg(&xv);
        let guard;
        let bias_view;
        match bias {
            Some(b) => {
                guard = b.storage_and_layout();
                let Storage::Cuda(s) = &*guard.0 else {
                    inference_tensor::bail!("the sparse kernels need CUDA");
                };
                bias_view = s.as_cuda_slice::<f16>()?.slice(guard.1.start_offset()..);
                builder.arg(&bias_view);
            }
            None => {
                builder.arg(&NULL);
            }
        }
        builder.arg(&av);
        builder.arg(&bv);
        builder.arg(&c32);
        builder.arg(&block32);
        builder.arg(&eps);
        builder.arg(&act);
        builder.arg(&mut out);
        // SAFETY: the arguments follow the kernel's signature, each row `c` values
        unsafe { builder.launch(cfg) }.w()?;
    }
    drop((xg, ag, bg));
    Ok(f16_tensor(dev, out, (rows, c)))
}

/// `silu(x + bias)` (f16 `[rows, c]`, bias `[c]`), each rounded.
pub fn bias_silu(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let dev = cuda(x)?;
    let (rows, c) = x.dims2()?;
    let x = x.contiguous()?;
    view!(x, f16, xg, xv);
    view!(bias, f16, bg, bv);
    let n = (rows * c) as u64;
    // SAFETY: the kernel writes every element
    let mut out = unsafe { dev.alloc::<f16>((n as usize).max(1)) }?;
    if n > 0 {
        let func = dev.get_or_load_custom_image("bias_silu_f16", MODULE, IMAGE)?;
        let mut builder = func.builder();
        let c32 = c as i32;
        builder.arg(&xv);
        builder.arg(&bv);
        builder.arg(&n);
        builder.arg(&c32);
        builder.arg(&mut out);
        // SAFETY: the arguments follow the kernel's signature
        unsafe { builder.launch(elementwise(n)) }.w()?;
    }
    drop((xg, bg));
    Ok(f16_tensor(dev, out, (rows, c)))
}

/// `residual + (x + bias)` (f16 `[rows, c]`, bias `[c]`), each rounded.
pub fn bias_residual(x: &Tensor, bias: &Tensor, residual: &Tensor) -> Result<Tensor> {
    let dev = cuda(x)?;
    let (rows, c) = x.dims2()?;
    let (x, residual) = (x.contiguous()?, residual.contiguous()?);
    view!(x, f16, xg, xv);
    view!(bias, f16, bg, bv);
    view!(residual, f16, rg, rv);
    let n = (rows * c) as u64;
    // SAFETY: the kernel writes every element
    let mut out = unsafe { dev.alloc::<f16>((n as usize).max(1)) }?;
    if n > 0 {
        let func = dev.get_or_load_custom_image("bias_residual_f16", MODULE, IMAGE)?;
        let mut builder = func.builder();
        let c32 = c as i32;
        builder.arg(&xv);
        builder.arg(&bv);
        builder.arg(&rv);
        builder.arg(&n);
        builder.arg(&c32);
        builder.arg(&mut out);
        // SAFETY: the arguments follow the kernel's signature
        unsafe { builder.launch(elementwise(n)) }.w()?;
    }
    drop((xg, bg, rg));
    Ok(f16_tensor(dev, out, (rows, c)))
}
