//! The Vulkan backend: storage in host-visible memory the iGPU and the CPU share, on `inference_vulkan`'s stream. An op
//! with no Vulkan kernel yet waits for the GPU, runs the CPU backend's on a copy and uploads the result, so a model
//! runs whole while its kernels are ported one at a time.

use std::sync::{Arc, Mutex, OnceLock};

use inference_vulkan::{
    BlockType, Bound, Context, HostBuffer, MAX_DIMS, Map, MapOp, MatMul, MatOperand, MatVec,
    Operand, RowOp, Rows, Select,
};

use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, DeviceLocation, Layout, Result, Shape, Tensor, WithDType};

// a zero-element tensor still gets a buffer
const MIN_BUFFER_BYTES: usize = 16;

static CONTEXT: OnceLock<std::result::Result<Arc<Context>, String>> = OnceLock::new();

fn vk_err(err: inference_vulkan::Error) -> crate::Error {
    crate::Error::wrap(err)
}

/// The process's one Vulkan device, shared by every `VulkanDevice` so tensors made through any of them combine.
#[derive(Clone)]
pub struct VulkanDevice {
    context: Arc<Context>,
    seed: Arc<Mutex<u64>>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanDevice({})", self.context.device_name())
    }
}

impl VulkanDevice {
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    fn buffer(&self, bytes: usize) -> Result<HostBuffer> {
        HostBuffer::new(&self.context, bytes.max(MIN_BUFFER_BYTES)).map_err(vk_err)
    }

    /// A buffer of `bytes`, which the device reads without an import's per-submit cost.
    pub(crate) fn host_copy(&self, bytes: &[u8]) -> Result<Arc<HostBuffer>> {
        let mut buffer = self.buffer(bytes.len())?;
        buffer.as_bytes_mut()[..bytes.len()].copy_from_slice(bytes);
        Ok(Arc::new(buffer))
    }

    fn storage_from_bytes(
        &self,
        bytes: &[u8],
        count: usize,
        dtype: DType,
    ) -> Result<VulkanStorage> {
        let mut buffer = self.buffer(bytes.len())?;
        buffer.as_bytes_mut()[..bytes.len()].copy_from_slice(bytes);
        Ok(VulkanStorage {
            buffer: Arc::new(buffer),
            device: self.clone(),
            count,
            dtype,
        })
    }
}

/// A tensor's elements in a host-visible Vulkan buffer.
pub struct VulkanStorage {
    buffer: Arc<HostBuffer>,
    device: VulkanDevice,
    count: usize,
    dtype: DType,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanStorage({} x {:?})", self.count, self.dtype)
    }
}

fn bytes_of<T: WithDType>(values: &[T]) -> &[u8] {
    // SAFETY: every `WithDType` is plain old data, so its bytes are initialized and any alignment of u8 holds
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn values_of<T: WithDType>(bytes: &[u8], count: usize) -> Vec<T> {
    let mut out = Vec::<T>::with_capacity(count);
    // SAFETY: `bytes` holds at least `count` values' bytes, which any bit pattern of a `WithDType` makes valid
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr().cast::<u8>(),
            count * size_of::<T>(),
        );
        out.set_len(count);
    }
    out
}

macro_rules! cpu_bytes {
    ($storage:expr, |$values:ident| $body:expr) => {
        match $storage {
            CpuStorage::U8($values) => $body,
            CpuStorage::U32($values) => $body,
            CpuStorage::I16($values) => $body,
            CpuStorage::I32($values) => $body,
            CpuStorage::I64($values) => $body,
            CpuStorage::BF16($values) => $body,
            CpuStorage::F16($values) => $body,
            CpuStorage::F32($values) => $body,
            CpuStorage::F64($values) => $body,
            CpuStorage::F8E4M3($values) => $body,
            CpuStorage::F6E2M3($values)
            | CpuStorage::F6E3M2($values)
            | CpuStorage::F4($values)
            | CpuStorage::F8E8M0($values) => $body,
        }
    };
}

impl VulkanStorage {
    pub fn buffer(&self) -> &Arc<HostBuffer> {
        &self.buffer
    }

    pub fn count(&self) -> usize {
        self.count
    }

    // the CPU backend's copy, after the GPU has finished every write to it
    fn cpu(&self) -> Result<CpuStorage> {
        self.device.synchronize()?;
        let bytes = self.buffer.as_bytes();
        Ok(match self.dtype {
            DType::U8 => CpuStorage::U8(values_of(bytes, self.count)),
            DType::U32 => CpuStorage::U32(values_of(bytes, self.count)),
            DType::I16 => CpuStorage::I16(values_of(bytes, self.count)),
            DType::I32 => CpuStorage::I32(values_of(bytes, self.count)),
            DType::I64 => CpuStorage::I64(values_of(bytes, self.count)),
            DType::BF16 => CpuStorage::BF16(values_of(bytes, self.count)),
            DType::F16 => CpuStorage::F16(values_of(bytes, self.count)),
            DType::F32 => CpuStorage::F32(values_of(bytes, self.count)),
            DType::F64 => CpuStorage::F64(values_of(bytes, self.count)),
            DType::F8E4M3 => CpuStorage::F8E4M3(values_of(bytes, self.count)),
            DType::F6E2M3 => CpuStorage::F6E2M3(values_of(bytes, self.count)),
            DType::F6E3M2 => CpuStorage::F6E3M2(values_of(bytes, self.count)),
            DType::F4 => CpuStorage::F4(values_of(bytes, self.count)),
            DType::F8E8M0 => CpuStorage::F8E8M0(values_of(bytes, self.count)),
        })
    }

    fn upload(&self, cpu: &CpuStorage) -> Result<Self> {
        self.device.storage_from_cpu_storage(cpu)
    }

    // an in-place op's result: the storage takes a buffer of `cpu`'s values
    fn replace(&mut self, cpu: &CpuStorage) -> Result<()> {
        *self = self.upload(cpu)?;
        Ok(())
    }
}

fn unary_op(name: &str) -> Option<MapOp> {
    Some(match name {
        "exp" => MapOp::Exp,
        "log" => MapOp::Log,
        "sin" => MapOp::Sin,
        "cos" => MapOp::Cos,
        "tanh" => MapOp::Tanh,
        "neg" => MapOp::Neg,
        "recip" => MapOp::Recip,
        "sqr" => MapOp::Sqr,
        "sqrt" => MapOp::Sqrt,
        "gelu" => MapOp::Gelu,
        "silu" => MapOp::Silu,
        "abs" => MapOp::Abs,
        "ceil" => MapOp::Ceil,
        "floor" => MapOp::Floor,
        "round" => MapOp::Round,
        "relu" => MapOp::Relu,
        "sign" => MapOp::Sign,
        _ => return None,
    })
}

fn binary_op(name: &str) -> Option<MapOp> {
    Some(match name {
        "add" => MapOp::Add,
        "sub" => MapOp::Sub,
        "mul" => MapOp::Mul,
        "div" => MapOp::Div,
        "maximum" => MapOp::Maximum,
        "minimum" => MapOp::Minimum,
        _ => return None,
    })
}

fn contiguous_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for d in (0..dims.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * dims[d + 1];
    }
    strides
}

// the dtypes whose elements the GPU copies as 32-bit words
fn word_sized(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::U32 | DType::I32)
}

impl VulkanStorage {
    fn empty(&self, count: usize, dtype: DType) -> Result<Self> {
        Ok(Self {
            buffer: Arc::new(self.device.buffer(count * dtype.size_in_bytes())?),
            device: self.device.clone(),
            count,
            dtype,
        })
    }

    fn operand<'a>(&'a self, l: &'a Layout) -> Operand<'a> {
        Operand {
            buffer: &self.buffer,
            offset: l.start_offset(),
            strides: l.stride(),
        }
    }

    // `op` over `l`'s elements (and `rhs`'s, laid out as `rhs_l`) into a new contiguous f32 storage
    fn map_f32(
        &self,
        op: MapOp,
        l: &Layout,
        rhs: Option<(&Self, &Layout)>,
        params: [f32; 2],
    ) -> Result<Self> {
        let dims = l.shape().dims();
        let out = self.empty(l.shape().elem_count(), DType::F32)?;
        let strides = contiguous_strides(dims);
        self.device
            .context
            .record(|r| {
                r.map_f32(&Map {
                    op,
                    dims,
                    a: self.operand(l),
                    b: rhs.map(|(rhs, rhs_l)| rhs.operand(rhs_l)),
                    out: Operand {
                        buffer: &out.buffer,
                        offset: 0,
                        strides: &strides,
                    },
                    params,
                })
            })
            .map_err(vk_err)?;
        Ok(out)
    }

    // copies `src_l`'s elements of `self` into `dst` from `dst_offset`, at `dst_strides`, as 32-bit words
    fn copy_words(
        &self,
        src_l: &Layout,
        dst: &Self,
        dst_offset: usize,
        dst_strides: &[usize],
    ) -> Result<()> {
        self.device
            .context
            .record(|r| {
                r.copy_b32(
                    &Map {
                        op: MapOp::Copy,
                        dims: src_l.shape().dims(),
                        a: self.operand(src_l),
                        b: None,
                        out: Operand {
                            buffer: &dst.buffer,
                            offset: dst_offset,
                            strides: dst_strides,
                        },
                        params: [0.0; 2],
                    },
                    0,
                )
            })
            .map_err(vk_err)
    }

    // index_select (`gather` false) or gather of contiguous 32-bit `self` by contiguous u32 `ids` along `dim`
    fn select(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        dim: usize,
        gather: bool,
    ) -> Option<Result<Self>> {
        if !word_sized(self.dtype)
            || ids.dtype != DType::U32
            || !l.is_contiguous()
            || !ids_l.is_contiguous()
        {
            return None;
        }
        let dims = l.shape().dims();
        let left: usize = dims[..dim].iter().product();
        let right: usize = dims[dim + 1..].iter().product();
        let n = if gather {
            ids_l.shape().dims()[dim]
        } else {
            ids_l.shape().elem_count()
        };
        let out = match self.empty(left * n * right, self.dtype) {
            Ok(out) => out,
            Err(err) => return Some(Err(err)),
        };
        let recorded = self.device.context.record(|r| {
            r.select_b32(&Select {
                left,
                src_dim: dims[dim],
                right,
                n,
                gather,
                src: (&self.buffer, l.start_offset()),
                ids: (&ids.buffer, ids_l.start_offset()),
                out: &out.buffer,
            })
        });
        Some(recorded.map_err(vk_err).map(|()| out))
    }

    fn native(&self, l: &Layout) -> bool {
        self.dtype == DType::F32 && l.shape().rank() <= MAX_DIMS
    }
}

/// `op` over `storage`'s contiguous last-dim rows (with `alpha` for RMS norm), or `None` where the kernel does not take it.
pub(crate) fn rows(
    storage: &VulkanStorage,
    l: &Layout,
    alpha: Option<(&VulkanStorage, &Layout)>,
    op: RowOp,
    eps: f32,
) -> Result<Option<(VulkanStorage, Shape)>> {
    let alpha_ok = alpha.is_none_or(|(a, al)| a.dtype == DType::F32 && al.is_contiguous());
    if storage.dtype != DType::F32 || !l.is_contiguous() || l.shape().rank() == 0 || !alpha_ok {
        return Ok(None);
    }
    let dims = l.shape().dims();
    let cols = dims[dims.len() - 1];
    let rows = l.shape().elem_count() / cols.max(1);
    let (out_count, out_dtype, out_dims) = if op.reduces() {
        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().expect("rank checked") = 1;
        let dtype = if matches!(op, RowOp::ArgMax | RowOp::ArgMin) {
            DType::U32
        } else {
            DType::F32
        };
        (rows, dtype, out_dims)
    } else {
        (rows * cols, DType::F32, dims.to_vec())
    };
    let out = storage.empty(out_count, out_dtype)?;
    storage
        .device
        .context
        .record(|r| {
            r.rows_f32(&Rows {
                op,
                rows,
                cols,
                eps,
                x: &storage.buffer,
                x_offset: l.start_offset(),
                alpha: alpha.map(|(a, al)| (&a.buffer, al.start_offset())),
                out: &out.buffer,
                out_offset: 0,
            })
        })
        .map_err(vk_err)?;
    Ok(Some((out, Shape::from(out_dims))))
}

pub(crate) fn sigmoid(
    storage: &VulkanStorage,
    l: &Layout,
) -> Result<Option<(VulkanStorage, Shape)>> {
    if !storage.native(l) {
        return Ok(None);
    }
    Ok(Some((
        storage.map_f32(MapOp::Sigmoid, l, None, [0.0; 2])?,
        l.shape().clone(),
    )))
}

// a contiguous f32 or u32 Vulkan tensor's buffer and first element
fn bound(t: &Tensor, dtype: DType) -> Result<(Arc<HostBuffer>, usize)> {
    if t.dtype() != dtype || !t.is_contiguous() {
        crate::bail!(
            "a Vulkan kernel takes contiguous {dtype:?}, got {:?} {:?}",
            t.dtype(),
            t.layout()
        );
    }
    match &*t.storage() {
        crate::Storage::Vulkan(s) => Ok((s.buffer.clone(), t.layout().start_offset())),
        _ => crate::bail!("a Vulkan kernel's operand is on {:?}", t.device()),
    }
}

fn tensor_of(storage: VulkanStorage, shape: impl Into<Shape>) -> Tensor {
    crate::tensor::from_storage(
        crate::Storage::Vulkan(storage),
        shape,
        crate::op::BackpropOp::none(),
        false,
    )
}

pub(crate) fn ssm_scan_step(step: &crate::vulkan::ScanStep<'_>) -> Result<Tensor> {
    let (batch, seq, heads, head_dim) = step.x.dims4()?;
    let (_, _, groups, state) = step.b.dims4()?;
    let f32 = DType::F32;
    let (x, dt, b, c) = (
        bound(step.x, f32)?,
        bound(step.dt, f32)?,
        bound(step.b, f32)?,
        bound(step.c, f32)?,
    );
    let (a, d, dt_bias) = (
        bound(step.a, f32)?,
        bound(step.d, f32)?,
        bound(step.dt_bias, f32)?,
    );
    let (pool, slots) = (bound(step.pool, f32)?, bound(step.slots, DType::U32)?);
    if [a.1, d.1, dt_bias.1] != [0; 3] {
        crate::bail!("the scan step reads a, d and dt_bias from their buffers' start");
    }
    let device = step.x.device().as_vulkan_device()?.clone();
    let count = batch * seq * heads * head_dim;
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(count * 4)?),
        device: device.clone(),
        count,
        dtype: f32,
    };
    device
        .context
        .record(|r| {
            r.ssm_scan_step(&inference_vulkan::ScanStep {
                batch,
                seq,
                heads,
                head_dim,
                groups,
                state,
                dt_range: step.dt_range,
                x: (&x.0, x.1),
                dt: (&dt.0, dt.1),
                a: &a.0,
                b: (&b.0, b.1),
                c: (&c.0, c.1),
                d: &d.0,
                dt_bias: &dt_bias.0,
                pool: (&pool.0, pool.1),
                slots: (&slots.0, slots.1),
                out: &out.buffer,
            })
        })
        .map_err(vk_err)?;
    Ok(tensor_of(out, (batch, seq, heads, head_dim)))
}

pub(crate) fn ssm_conv_step(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    pool: &Tensor,
    slots: &Tensor,
) -> Result<Tensor> {
    let (batch, seq, channels) = x.dims3()?;
    let kernel = weight.dim(1)?;
    let f32 = DType::F32;
    let (xb, wb, pb, sb) = (
        bound(x, f32)?,
        bound(weight, f32)?,
        bound(pool, f32)?,
        bound(slots, DType::U32)?,
    );
    let bias = bias.map(|b| bound(b, f32)).transpose()?;
    let device = x.device().as_vulkan_device()?.clone();
    let count = batch * seq * channels;
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(count * 4)?),
        device: device.clone(),
        count,
        dtype: f32,
    };
    device
        .context
        .record(|r| {
            r.ssm_conv_step(&inference_vulkan::ConvStep {
                batch,
                seq,
                channels,
                kernel,
                x: (&xb.0, xb.1),
                weight: (&wb.0, wb.1),
                bias: bias.as_ref().map(|b| (&b.0, b.1)),
                pool: (&pb.0, pb.1),
                slots: (&sb.0, sb.1),
                out: &out.buffer,
            })
        })
        .map_err(vk_err)?;
    Ok(tensor_of(out, (batch, seq, channels)))
}

pub(crate) fn moe_router(route: &crate::vulkan::Route<'_>) -> Result<(Tensor, Tensor)> {
    use crate::vulkan::RouterFn as F;
    let (tokens, experts) = route.logits.dims2()?;
    let logits = bound(route.logits, DType::F32)?;
    let bias = route
        .selection_bias
        .map(|b| bound(b, DType::F32))
        .transpose()?;
    if bias.as_ref().is_some_and(|b| b.1 != 0) {
        crate::bail!("the router reads its selection bias from its buffer's start");
    }
    let device = route.logits.device().as_vulkan_device()?.clone();
    let count = tokens * route.top_k;
    let storage = |dtype| -> Result<VulkanStorage> {
        Ok(VulkanStorage {
            buffer: Arc::new(device.buffer(count * 4)?),
            device: device.clone(),
            count,
            dtype,
        })
    };
    let (values, indices) = (storage(DType::F32)?, storage(DType::U32)?);
    let to_fn = |f: F| match f {
        F::Raw => inference_vulkan::RouterFn::Raw,
        F::Softmax => inference_vulkan::RouterFn::Softmax,
        F::Sigmoid => inference_vulkan::RouterFn::Sigmoid,
    };
    device
        .context
        .record(|r| {
            r.moe_router(&inference_vulkan::Router {
                tokens,
                experts,
                top_k: route.top_k,
                score_fn: to_fn(route.score_fn),
                weight_fn: to_fn(route.weight_fn),
                renormalize: route.renormalize,
                norm_min: route.norm_min,
                output_scale: route.output_scale,
                clip: route.clip,
                logits: (&logits.0, logits.1),
                bias: bias.as_ref().map(|b| &b.0),
                values: &values.buffer,
                indices: &indices.buffer,
            })
        })
        .map_err(vk_err)?;
    let shape = (tokens, route.top_k);
    Ok((tensor_of(values, shape), tensor_of(indices, shape)))
}

/// `y[slot] = W[ids[slot]] x[slot / x_div]` over a Q5_0 expert stack `(experts, rows, cols)` copied to `weights`, `ids`
/// `(tokens, k)`.
pub(crate) fn expert_mat_vec(
    weights: &Arc<HostBuffer>,
    (experts, rows, cols): (usize, usize, usize),
    x: &Tensor,
    ids: &Tensor,
    x_div: usize,
) -> Result<Tensor> {
    let (xb, ib) = (bound(x, DType::F32)?, bound(ids, DType::U32)?);
    let slots = ids.elem_count();
    let device = x.device().as_vulkan_device()?.clone();
    let scratch = Arc::new(device.buffer(MatVec::scratch_bytes(slots.div_ceil(x_div), cols))?);
    let routing =
        Arc::new(device.buffer(inference_vulkan::MoeMatVec::routing_bytes(slots, experts))?);
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(slots * rows * 4)?),
        device: device.clone(),
        count: slots * rows,
        dtype: DType::F32,
    };
    let weights: Arc<dyn Bound> = weights.clone();
    device
        .context
        .record(|r| {
            r.moe_mat_vec_q5_0(&inference_vulkan::MoeMatVec {
                weights: &weights,
                experts,
                rows,
                cols,
                slots,
                x_div,
                x: (&xb.0, xb.1),
                ids: (&ib.0, ib.1),
                y: &out.buffer,
                scratch: &scratch,
                routing: &routing,
            })
        })
        .map_err(vk_err)?;
    Ok(tensor_of(out, (slots, rows)))
}

// the geometry of f32 vLLM-layout caches, keys `[blocks, kv_heads, head_size / x, block_size, x]`
fn cache_shape(key_cache: &Tensor, value_cache: &Tensor) -> Result<inference_vulkan::CacheShape> {
    let (_, kv_heads, _, block_size, x) = key_cache.dims5()?;
    let (_, _, head_size, _) = value_cache.dims4()?;
    Ok(inference_vulkan::CacheShape {
        kv_heads,
        head_size,
        block_size,
        x,
    })
}

pub(crate) fn paged_cache_write(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let shape = cache_shape(key_cache, value_cache)?;
    let tokens = key.dim(0)?;
    let f32 = DType::F32;
    let (k, v) = (
        bound(&key.contiguous()?, f32)?,
        bound(&value.contiguous()?, f32)?,
    );
    let (kc, vc) = (bound(key_cache, f32)?, bound(value_cache, f32)?);
    let slots = slot_mapping
        .to_device(key.device())?
        .to_dtype(DType::I64)?
        .contiguous()?;
    let sb = match &*slots.storage() {
        crate::Storage::Vulkan(s) => (s.buffer.clone(), slots.layout().start_offset()),
        _ => crate::bail!("paged_cache_write: the slots are not on the Vulkan device"),
    };
    let device = key.device().as_vulkan_device()?.clone();
    device
        .context
        .record(|r| {
            r.paged_cache_write(&inference_vulkan::PagedWrite {
                shape,
                tokens,
                key: (&k.0, k.1),
                value: (&v.0, v.1),
                key_cache: &kc.0,
                value_cache: &vc.0,
                slots: (&sb.0, sb.1),
            })
        })
        .map_err(vk_err)
}

pub(crate) fn paged_decode(op: &crate::vulkan::PagedDecode<'_>) -> Result<Tensor> {
    let shape = cache_shape(op.key_cache, op.value_cache)?;
    let (seqs, heads, head_size) = op.q.dims3()?;
    let (_, max_blocks) = op.block_tables.dims2()?;
    let f32 = DType::F32;
    let device = op.q.device().as_vulkan_device()?.clone();
    let on_device = |t: &Tensor| -> Result<Tensor> {
        t.to_device(op.q.device())?
            .to_dtype(DType::U32)?
            .contiguous()
    };
    let (tables, lens) = (on_device(op.block_tables)?, on_device(op.context_lens)?);
    let (q, kc, vc) = (
        bound(&op.q.contiguous()?, f32)?,
        bound(op.key_cache, f32)?,
        bound(op.value_cache, f32)?,
    );
    let (tb, lb) = (bound(&tables, DType::U32)?, bound(&lens, DType::U32)?);
    let count = seqs * heads * head_size;
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(count * 4)?),
        device: device.clone(),
        count,
        dtype: f32,
    };
    let partials = Arc::new(device.buffer(inference_vulkan::PagedDecode::partials_bytes(
        seqs,
        heads,
        head_size,
        op.max_context,
    ))?);
    device
        .context
        .record(|r| {
            r.paged_decode(&inference_vulkan::PagedDecode {
                shape,
                seqs,
                heads,
                max_blocks,
                max_context: op.max_context,
                scale: op.scale,
                q: (&q.0, q.1),
                key_cache: &kc.0,
                value_cache: &vc.0,
                block_tables: (&tb.0, tb.1),
                context_lens: (&lb.0, lb.1),
                partials: &partials,
                out: &out.buffer,
            })
        })
        .map_err(vk_err)?;
    Ok(tensor_of(out, (seqs, heads, head_size)))
}

// the stride between batches of a matmul operand whose batch dims (all but its last two) collapse into one
fn batch_stride(l: &Layout, batch: usize) -> Option<usize> {
    let (dims, strides) = (l.shape().dims(), l.stride());
    let rank = dims.len();
    if rank <= 2 || batch == 1 {
        return Some(0);
    }
    for d in 0..rank - 3 {
        if dims[d + 1] != 1 && strides[d] != strides[d + 1] * dims[d + 1] {
            return None;
        }
    }
    Some(strides[rank - 3])
}

/// `xs`'s rows of `k` (laid out by `l`, contiguous) times the `n x k` weight of `ty` blocks in `weights`, transposed,
/// in one matvec reading the weight once.
pub(crate) fn quantized_mat_vec(
    xs: &VulkanStorage,
    l: &Layout,
    weights: &Arc<HostBuffer>,
    ty: BlockType,
    n: usize,
    k: usize,
) -> Result<VulkanStorage> {
    let rows = l.shape().elem_count() / k;
    let out = xs.empty(rows * n, DType::F32)?;
    let weights: Arc<dyn Bound> = weights.clone();
    let scratch = Arc::new(xs.device.buffer(MatVec::scratch_bytes(rows, k))?);
    xs.device
        .context
        .record(|r| {
            r.mat_vec(&MatVec {
                ty,
                weights: &weights,
                rows: n,
                cols: k,
                x: &xs.buffer,
                x_offset: l.start_offset(),
                y: &out.buffer,
                y_offset: 0,
                x_rows: rows,
                scratch: &scratch,
            })
        })
        .map_err(vk_err)?;
    Ok(out)
}

/// Runs a CPU kernel on the storages' copies, and gives back the Vulkan storage of its result.
macro_rules! on_cpu {
    ($self:ident, |$cpu:ident $(, $other:ident)*| $body:expr) => {{
        let $cpu = $self.cpu()?;
        $(let $other = $other.cpu()?;)*
        let out: CpuStorage = $body?;
        $self.upload(&out)
    }};
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        if word_sized(self.dtype) {
            let out = self.empty(self.count, self.dtype)?;
            self.copy_words(&Layout::contiguous(self.count), &out, 0, &[1])?;
            return Ok(out);
        }
        on_cpu!(self, |cpu| Ok::<_, crate::Error>(cpu))
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.cpu()
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.native(l) {
            return self.map_f32(MapOp::Affine, l, None, [mul as f32, add as f32]);
        }
        on_cpu!(self, |cpu| cpu.affine(l, mul, add))
    }

    fn powf(&self, l: &Layout, e: f64) -> Result<Self> {
        if self.native(l) {
            return self.map_f32(MapOp::Powf, l, None, [e as f32, 0.0]);
        }
        on_cpu!(self, |cpu| cpu.powf(l, e))
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        if self.native(l) {
            return self.map_f32(MapOp::Elu, l, None, [alpha as f32, 0.0]);
        }
        on_cpu!(self, |cpu| cpu.elu(l, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, dims: &[usize]) -> Result<Self> {
        let row_op = match op {
            ReduceOp::Sum => RowOp::Sum,
            ReduceOp::Max => RowOp::Max,
            ReduceOp::Min => RowOp::Min,
            ReduceOp::ArgMax => RowOp::ArgMax,
        };
        if dims == [l.shape().rank().saturating_sub(1)]
            && let Some((out, _)) = rows(self, l, None, row_op, 0.0)?
        {
            return Ok(out);
        }
        on_cpu!(self, |cpu| cpu.reduce_op(op, l, dims))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, l: &Layout, rhs_l: &Layout) -> Result<Self> {
        on_cpu!(self, |cpu, rhs| cpu.cmp(op, &rhs, l, rhs_l))
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        if dtype == self.dtype && word_sized(dtype) && l.shape().rank() <= MAX_DIMS {
            let out = self.empty(l.shape().elem_count(), dtype)?;
            self.copy_words(l, &out, 0, &contiguous_strides(l.shape().dims()))?;
            return Ok(out);
        }
        on_cpu!(self, |cpu| cpu.to_dtype(l, dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        if let Some(op) = unary_op(B::NAME).filter(|_| self.native(l)) {
            return self.map_f32(op, l, None, [0.0; 2]);
        }
        on_cpu!(self, |cpu| cpu.unary_impl::<B>(l))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if let Some(op) = binary_op(B::NAME).filter(|_| self.native(l) && rhs.dtype == DType::F32) {
            return self.map_f32(op, l, Some((rhs, rhs_l)), [0.0; 2]);
        }
        on_cpu!(self, |cpu, rhs| cpu.binary_impl::<B>(&rhs, l, rhs_l))
    }

    fn where_cond(
        &self,
        l: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        on_cpu!(self, |cpu, t, f| cpu.where_cond(l, &t, t_l, &f, f_l))
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Self> {
        on_cpu!(self, |cpu, kernel| cpu.conv1d(l, &kernel, kernel_l, params))
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        on_cpu!(self, |cpu, kernel| cpu
            .conv_transpose1d(l, &kernel, kernel_l, params))
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Self> {
        on_cpu!(self, |cpu, kernel| cpu.conv2d(l, &kernel, kernel_l, params))
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        on_cpu!(self, |cpu, kernel| cpu
            .conv_transpose2d(l, &kernel, kernel_l, params))
    }

    fn avg_pool2d(
        &self,
        l: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        on_cpu!(self, |cpu| cpu.avg_pool2d(l, kernel, stride))
    }

    fn max_pool2d(
        &self,
        l: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        on_cpu!(self, |cpu| cpu.max_pool2d(l, kernel, stride))
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        on_cpu!(self, |cpu| cpu.upsample_nearest2d(l, h, w))
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        on_cpu!(self, |cpu| cpu.upsample_bilinear2d(
            l,
            h,
            w,
            align_corners,
            scale_h,
            scale_w
        ))
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let (src, idx) = (l.shape().dims(), ids_l.shape().dims());
        let same_but_dim =
            src.len() == idx.len() && (0..src.len()).all(|d| d == dim || src[d] == idx[d]);
        if let Some(out) = self
            .select(l, ids, ids_l, dim, true)
            .filter(|_| same_but_dim)
        {
            return out;
        }
        on_cpu!(self, |cpu, ids| cpu.gather(l, &ids, ids_l, dim))
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut cpu = self.cpu()?;
        cpu.scatter_set(l, &ids.cpu()?, ids_l, &src.cpu()?, src_l, dim)?;
        self.replace(&cpu)
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut cpu = self.cpu()?;
        cpu.scatter_add_set(l, &ids.cpu()?, ids_l, &src.cpu()?, src_l, dim)?;
        self.replace(&cpu)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if let Some(out) = self.select(l, ids, ids_l, dim, false) {
            return out;
        }
        on_cpu!(self, |cpu, ids| cpu.index_select(&ids, l, ids_l, dim))
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        on_cpu!(self, |cpu, ids, src| cpu
            .index_add(l, &ids, ids_l, &src, src_l, dim))
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let (batch, m, n, k) = bmnk;
        let strides = (batch_stride(l, batch), batch_stride(rhs_l, batch));
        if let (DType::F32, DType::F32, (Some(a_bs), Some(b_bs))) = (self.dtype, rhs.dtype, strides)
        {
            let (a_st, b_st) = (l.stride(), rhs_l.stride());
            let (ar, br) = (a_st.len(), b_st.len());
            let out = self.empty(batch * m * n, DType::F32)?;
            self.device
                .context
                .record(|r| {
                    r.matmul_f32(&MatMul {
                        batch,
                        m,
                        n,
                        k,
                        a: MatOperand {
                            buffer: &self.buffer,
                            offset: l.start_offset(),
                            batch_stride: a_bs,
                            row_stride: a_st[ar - 2],
                            col_stride: a_st[ar - 1],
                        },
                        b: MatOperand {
                            buffer: &rhs.buffer,
                            offset: rhs_l.start_offset(),
                            batch_stride: b_bs,
                            row_stride: b_st[br - 2],
                            col_stride: b_st[br - 1],
                        },
                        out: &out.buffer,
                    })
                })
                .map_err(vk_err)?;
            return Ok(out);
        }
        on_cpu!(self, |cpu, rhs| cpu.matmul(&rhs, bmnk, l, rhs_l))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        if word_sized(self.dtype) && dst.dtype == self.dtype && src_l.shape().rank() <= MAX_DIMS {
            return self.copy_words(
                src_l,
                dst,
                dst_offset,
                &contiguous_strides(src_l.shape().dims()),
            );
        }
        let mut out = dst.cpu()?;
        self.cpu()?.copy_strided_src(&mut out, dst_offset, src_l)?;
        dst.replace(&out)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        if word_sized(self.dtype) && dst.dtype == self.dtype {
            let src_l = Layout::new(Shape::from((d1, d2)), vec![src_stride1, 1], src_offset);
            return self.copy_words(&src_l, dst, dst_offset, &[dst_stride1, 1]);
        }
        let mut out = dst.cpu()?;
        self.cpu()?.copy2d(
            &mut out,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )?;
        dst.replace(&out)
    }

    fn const_set(&mut self, v: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let fill = match v {
            crate::scalar::Scalar::F32(v) => Some(v.to_bits()),
            crate::scalar::Scalar::U32(v) => Some(v),
            crate::scalar::Scalar::I32(v) => Some(v as u32),
            _ => None,
        };
        if let Some(fill) = fill.filter(|_| word_sized(self.dtype) && l.shape().rank() <= MAX_DIMS)
        {
            let out = Operand {
                buffer: &self.buffer,
                offset: l.start_offset(),
                strides: l.stride(),
            };
            let map = Map {
                op: MapOp::Fill,
                dims: l.shape().dims(),
                a: out,
                b: None,
                out,
                params: [0.0; 2],
            };
            return self
                .device
                .context
                .record(|r| r.copy_b32(&map, fill))
                .map_err(vk_err);
        }
        let mut cpu = self.cpu()?;
        cpu.const_set(v, l)?;
        self.replace(&cpu)
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        if ordinal != 0 {
            crate::bail!("the Vulkan backend drives one device, ordinal 0, not {ordinal}");
        }
        let context = CONTEXT
            .get_or_init(|| Context::new().map_err(|err| err.to_string()))
            .clone()
            .map_err(crate::Error::msg)?;
        Ok(Self {
            context,
            seed: Arc::new(Mutex::new(0)),
        })
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Vulkan { gpu_id: 0 }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.context, &rhs.context)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let mut buffer = self.buffer(count * dtype.size_in_bytes())?;
        buffer.as_bytes_mut().fill(0);
        Ok(VulkanStorage {
            buffer: Arc::new(buffer),
            device: self.clone(),
            count,
            dtype,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        Ok(VulkanStorage {
            buffer: Arc::new(self.buffer(count * dtype.size_in_bytes())?),
            device: self.clone(),
            count,
            dtype,
        })
    }

    fn storage_from_slice<T: WithDType>(&self, values: &[T]) -> Result<Self::Storage> {
        self.storage_from_bytes(bytes_of(values), values.len(), T::DTYPE)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let dtype = storage.dtype();
        cpu_bytes!(storage, |values| self.storage_from_bytes(
            bytes_of(values),
            values.len(),
            dtype
        ))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(
            &crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, lo, hi)?,
        )
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(
            &crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)?,
        )
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed.lock().expect("vulkan seed poisoned") = seed;
        crate::cpu_backend::CpuDevice.set_seed(seed)
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed.lock().expect("vulkan seed poisoned"))
    }

    fn synchronize(&self) -> Result<()> {
        self.context.synchronize().map_err(vk_err)
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, IndexOp, Result, Tensor};

    fn device() -> Option<Device> {
        match Device::new_vulkan(0) {
            Ok(device) => Some(device),
            Err(err) => {
                eprintln!("skipping: {err}");
                None
            }
        }
    }

    // the GPU's exp, tanh and the like round differently from the CPU's
    const TOLERANCE: f32 = 1e-5;
    // a 2688-term dot product summed in another order
    const DOT_TOLERANCE: f32 = 1e-3;
    // a 2688-term dot product over activations rounded to int8 blocks, against the outputs' RMS: unit-normal weights
    // and activations leave ~0.6 % (one deviation), and the worst of a few hundred outputs lands near three
    const INT8_DOT_TOLERANCE: f32 = 3e-2;

    fn assert_same(got: &Tensor, want: &Tensor) -> Result<()> {
        assert_close(got, want, TOLERANCE)
    }

    // within int8 activations' rounding, measured against the outputs' RMS (an output near zero keeps the same
    // absolute error)
    fn assert_int8_close(got: &Tensor, want: &Tensor) -> Result<()> {
        assert_eq!(got.dims(), want.dims());
        let (got, want) = (
            got.flatten_all()?.to_vec1::<f32>()?,
            want.flatten_all()?.to_vec1::<f32>()?,
        );
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= INT8_DOT_TOLERANCE * rms,
                "{i}: {g} vs {w} (rms {rms})"
            );
        }
        Ok(())
    }

    fn assert_close(got: &Tensor, want: &Tensor, tolerance: f32) -> Result<()> {
        assert_eq!(got.dims(), want.dims());
        assert!(got.device().is_vulkan());
        let (got, want) = (
            got.flatten_all()?.to_vec1::<f32>()?,
            want.flatten_all()?.to_vec1::<f32>()?,
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= tolerance * w.abs().max(1.0),
                "{i}: {g} vs {w}"
            );
        }
        Ok(())
    }

    // the native maps: broadcast binary ops, unary ops on strided views, affine, and fills and copies into views
    #[test]
    fn vulkan_elementwise_ops_match_the_cpu() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let a = ((Tensor::arange(0f32, 24., &Device::Cpu)?.reshape((2, 3, 4))? - 11.5)? / 7.)?;
        let row = Tensor::new(&[0.5f32, -2., 3., 1.5], &Device::Cpu)?;
        let (va, vrow) = (a.to_device(&vk)?, row.to_device(&vk)?);
        assert_same(&va.broadcast_mul(&vrow)?, &a.broadcast_mul(&row)?)?;
        assert_same(
            &va.broadcast_sub(&vrow)?.maximum(&va)?,
            &a.broadcast_sub(&row)?.maximum(&a)?,
        )?;
        let view = |t: &Tensor| t.transpose(0, 2).and_then(|t| t.narrow(1, 1, 2));
        assert_same(&view(&va)?.exp()?, &view(&a)?.exp()?)?;
        assert_same(&view(&va)?.silu()?, &view(&a)?.silu()?)?;
        assert_same(&view(&va)?.tanh()?, &view(&a)?.tanh()?)?;
        assert_same(&view(&va)?.relu()?, &view(&a)?.relu()?)?;
        assert_same(&((view(&va)? * 3.)? + 0.25)?, &((view(&a)? * 3.)? + 0.25)?)?;
        let vz = Tensor::zeros((4, 6), DType::F32, &vk)?;
        let z = Tensor::zeros((4, 6), DType::F32, &Device::Cpu)?;
        vz.slice_set(&va.reshape((4, 6))?.narrow(1, 0, 3)?.contiguous()?, 1, 2)?;
        z.slice_set(&a.reshape((4, 6))?.narrow(1, 0, 3)?.contiguous()?, 1, 2)?;
        assert_same(&vz, &z)?;
        let ids = Tensor::new(&[3u32, 1, 4], &Device::Cpu)?;
        assert_eq!(
            ids.to_device(&vk)?.contiguous()?.to_vec1::<u32>()?,
            ids.to_vec1::<u32>()?
        );
        Ok(())
    }

    #[test]
    fn vulkan_tensors_compute_as_the_cpu_does() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let a = Tensor::arange(0f32, 24., &Device::Cpu)?.reshape((4, 6))?;
        let b = (Tensor::arange(0f32, 24., &Device::Cpu)?.reshape((6, 4))? * 0.5)?;
        let (va, vb) = (a.to_device(&vk)?, b.to_device(&vk)?);
        assert_same(&va, &a)?;
        assert_same(&(&va + &va)?, &(&a + &a)?)?;
        assert_same(&va.matmul(&vb)?, &a.matmul(&b)?)?;
        assert_same(&va.t()?.contiguous()?, &a.t()?.contiguous()?)?;
        assert_same(
            &va.i((.., 1..3))?.contiguous()?,
            &a.i((.., 1..3))?.contiguous()?,
        )?;
        assert_same(
            &crate::nn::ops::softmax_last_dim(&va)?,
            &crate::nn::ops::softmax_last_dim(&a)?,
        )?;
        assert_same(&va.sum_keepdim(1)?, &a.sum_keepdim(1)?)?;
        assert_eq!(
            va.to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?,
            a.to_vec2::<f32>()?
        );
        Ok(())
    }

    // Q8_0 and Q5_0 weights on the Vulkan device take a decode step's and a verify step's rows on the GPU, against the
    // dequantized weight in f32
    #[test]
    fn vulkan_quantized_matmuls_match_the_dequantized_weight() -> Result<()> {
        use crate::Module;
        use crate::quantized::{GgmlDType, QMatMul, QTensor};
        let Some(vk) = device() else { return Ok(()) };
        let w = Tensor::randn(0f32, 1., (96, 2688), &Device::Cpu)?;
        for dtype in [GgmlDType::Q8_0, GgmlDType::Q5_0] {
            let weight = QTensor::quantize(&w.to_device(&vk)?, dtype)?;
            let dense = weight.dequantize(&Device::Cpu)?;
            let matmul = QMatMul::from_qtensor(weight)?;
            for rows in [1usize, 3] {
                let x = Tensor::randn(0f32, 1., (1, rows, 2688), &Device::Cpu)?;
                let want = x.broadcast_matmul(&dense.t()?)?;
                assert_int8_close(&matmul.forward(&x.to_device(&vk)?)?, &want)?;
            }
        }
        Ok(())
    }

    // a Q5_0 expert stack's routes: one token's (a kernel a route), then several tokens sharing experts (the kernel
    // reading each expert once), over a token's input (x_div k) and over each route's own row (x_div 1)
    #[test]
    fn vulkan_expert_matvecs_match_the_dequantized_experts() -> Result<()> {
        use crate::quantized::{GgmlDType, QTensor};
        let Some(vk) = device() else { return Ok(()) };
        let (experts, rows, cols, k) = (4usize, 71usize, 256usize, 2usize);
        let w = Tensor::randn(0f32, 1., (experts, rows, cols), &Device::Cpu)?;
        let weight = QTensor::quantize(&w.to_device(&vk)?, GgmlDType::Q5_0)?;
        let dense = weight.dequantize(&Device::Cpu)?;
        // the last: 18 slots of one expert, more than a leader takes
        // a prompt's 150 routes, past the grouped kernel's, take the tiled matmul over routes sorted by expert, one
        // expert past a tile of them
        let prompt: Vec<u32> = (0..150u32)
            .map(|i| if i % 3 == 0 { 2 } else { i * 7 % 4 })
            .collect();
        let routes: [&[u32]; 4] = [&[3, 1], &[2, 0, 0, 2, 1, 2], &[1; 18], &prompt];
        for ids in routes {
            let tokens = ids.len() / k;
            let id_t = Tensor::from_slice(ids, (tokens, k), &Device::Cpu)?;
            for x_div in [k, 1] {
                let x = Tensor::randn(0f32, 1., (ids.len() / x_div, cols), &Device::Cpu)?;
                let got = weight
                    .vulkan_expert_mat_vec(&x.to_device(&vk)?, &id_t.to_device(&vk)?, x_div)?
                    .expect("Q5_0 runs on Vulkan");
                let mut want = Vec::new();
                for (slot, &e) in ids.iter().enumerate() {
                    let row = x.narrow(0, slot / x_div, 1)?;
                    want.push(row.matmul(&dense.get(e as usize)?.t()?)?);
                }
                assert_int8_close(&got, &Tensor::cat(&want, 0)?)?;
            }
        }
        Ok(())
    }

    // the tiled matmul over sizes off its tiles: attention's q k^T (b strided along n) and p v (b unit-stride along n)
    #[test]
    fn vulkan_tiled_matmuls_match_the_cpu() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let q = Tensor::randn(0f32, 1., (3, 70, 40), &Device::Cpu)?;
        let k = Tensor::randn(0f32, 1., (3, 131, 40), &Device::Cpu)?;
        let v = Tensor::randn(0f32, 1., (3, 131, 24), &Device::Cpu)?;
        let (vq, vk_, vv) = (q.to_device(&vk)?, k.to_device(&vk)?, v.to_device(&vk)?);
        let scores = q.matmul(&k.t()?)?;
        assert_close(&vq.matmul(&vk_.t()?)?, &scores, DOT_TOLERANCE)?;
        let vscores = scores.to_device(&vk)?;
        assert_close(&vscores.matmul(&vv)?, &scores.matmul(&v)?, DOT_TOLERANCE)?;
        Ok(())
    }

    // row ops, row selection and matmuls on the GPU: RMS norm, softmax, sigmoid, last-dim reductions, embedding and
    // gather lookups, and batched matmuls with a transposed operand
    #[test]
    fn vulkan_row_ops_selection_and_matmuls_match_the_cpu() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let x = Tensor::randn(0f32, 1., (3, 5, 300), &Device::Cpu)?;
        let alpha = Tensor::randn(1f32, 0.1, 300, &Device::Cpu)?;
        let (vx, valpha) = (x.to_device(&vk)?, alpha.to_device(&vk)?);
        let ops = crate::nn::ops::rms_norm;
        assert_same(&ops(&vx, &valpha, 1e-6)?, &ops(&x, &alpha, 1e-6)?)?;
        assert_same(
            &crate::nn::ops::softmax_last_dim(&vx)?,
            &crate::nn::ops::softmax_last_dim(&x)?,
        )?;
        assert_same(
            &crate::nn::ops::sigmoid(&vx)?,
            &crate::nn::ops::sigmoid(&x)?,
        )?;
        assert_close(&vx.sum_keepdim(2)?, &x.sum_keepdim(2)?, DOT_TOLERANCE)?;
        assert_same(&vx.max_keepdim(2)?, &x.max_keepdim(2)?)?;
        assert_eq!(
            vx.argmax_keepdim(2)?.to_vec3::<u32>()?,
            x.argmax_keepdim(2)?.to_vec3::<u32>()?
        );
        let table = Tensor::randn(0f32, 1., (50, 16), &Device::Cpu)?;
        let ids = Tensor::new(&[7u32, 0, 49, 7], &Device::Cpu)?;
        let (vtable, vids) = (table.to_device(&vk)?, ids.to_device(&vk)?);
        assert_same(
            &vtable.index_select(&vids, 0)?,
            &table.index_select(&ids, 0)?,
        )?;
        let picks = Tensor::new(&[[3u32, 1], [0, 15]], &Device::Cpu)?;
        let (rows, vrows) = (table.narrow(0, 0, 2)?, vtable.narrow(0, 0, 2)?);
        assert_same(
            &vrows.contiguous()?.gather(&picks.to_device(&vk)?, 1)?,
            &rows.contiguous()?.gather(&picks, 1)?,
        )?;
        let a = Tensor::randn(0f32, 1., (4, 3, 64), &Device::Cpu)?;
        let b = Tensor::randn(0f32, 1., (4, 7, 64), &Device::Cpu)?;
        let (va, vb) = (a.to_device(&vk)?, b.to_device(&vk)?);
        assert_close(&va.matmul(&vb.t()?)?, &a.matmul(&b.t()?)?, DOT_TOLERANCE)?;
        let w = Tensor::randn(0f32, 1., (128, 64), &Device::Cpu)?;
        let row = a.narrow(1, 0, 1)?;
        assert_close(
            &row.to_device(&vk)?
                .broadcast_matmul(&w.to_device(&vk)?.t()?)?,
            &row.broadcast_matmul(&w.t()?)?,
            DOT_TOLERANCE,
        )?;
        Ok(())
    }

    // the scan against its formula over 128-state rows and 3 steps, one batch row advancing pool row 1 and one padding row
    #[test]
    fn a_vulkan_scan_step_advances_its_pool_rows() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let (seq, heads, head_dim, groups, state) = (
            3usize,
            4usize,
            8usize,
            2usize,
            crate::vulkan::SCAN_STEP_STATE,
        );
        let r = |shape: &[usize]| Tensor::randn(0f32, 0.5, shape, &Device::Cpu);
        let (x, dt, b, c) = (
            r(&[2, seq, heads, head_dim])?,
            r(&[2, seq, heads])?,
            r(&[2, seq, groups, state])?,
            r(&[2, seq, groups, state])?,
        );
        let (a, d, dt_bias) = ((r(&[heads])?.abs()? * -1.)?, r(&[heads])?, r(&[heads])?);
        let pool = r(&[3, heads, head_dim, state])?;
        let vpool = pool.to_device(&vk)?;
        let slots = Tensor::new(&[1u32, u32::MAX], &Device::Cpu)?.to_device(&vk)?;
        let on = |t: &Tensor| t.to_device(&vk);
        let got = crate::vulkan::ssm_scan_step(&crate::vulkan::ScanStep {
            x: &on(&x)?,
            dt: &on(&dt)?,
            a: &on(&a)?,
            b: &on(&b)?,
            c: &on(&c)?,
            d: &on(&d)?,
            dt_bias: &on(&dt_bias)?,
            dt_range: [0.0, f32::INFINITY],
            pool: &vpool,
            slots: &slots,
        })?
        .flatten_all()?
        .to_vec1::<f32>()?;
        let (xv, dtv, bv, cv) = (
            x.flatten_all()?.to_vec1::<f32>()?,
            dt.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?,
            c.flatten_all()?.to_vec1::<f32>()?,
        );
        let (av, dv, biasv) = (
            a.to_vec1::<f32>()?,
            d.to_vec1::<f32>()?,
            dt_bias.to_vec1::<f32>()?,
        );
        let mut states = pool.flatten_all()?.to_vec1::<f32>()?;
        let row_len = heads * head_dim * state;
        let step_len = heads * head_dim;
        let mut want = vec![0f32; seq * step_len];
        for t in 0..seq {
            for h in 0..heads {
                let dt_t = (1.0 + (dtv[t * heads + h] + biasv[h]).exp()).ln();
                let decay = (dt_t * av[h]).exp();
                let g = t * groups * state + h / (heads / groups) * state;
                for p in 0..head_dim {
                    let xp = xv[t * step_len + h * head_dim + p];
                    let mut acc = 0f32;
                    for n in 0..state {
                        let at = row_len + (h * head_dim + p) * state + n;
                        states[at] = states[at] * decay + dt_t * xp * bv[g + n];
                        acc += states[at] * cv[g + n];
                    }
                    want[t * step_len + h * head_dim + p] = acc + dv[h] * xp;
                }
            }
        }
        for (i, (g, w)) in got[..seq * step_len].iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= DOT_TOLERANCE * w.abs().max(1.0),
                "{i}: {g} vs {w}"
            );
        }
        assert!(
            got[seq * step_len..].iter().all(|&v| v == 0.0),
            "the padding row scores 0"
        );
        let pool_after = vpool.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            &pool_after[..row_len],
            &pool.flatten_all()?.to_vec1::<f32>()?[..row_len],
            "row 0 is untouched"
        );
        for (i, (g, w)) in pool_after[row_len..2 * row_len]
            .iter()
            .zip(&states[row_len..2 * row_len])
            .enumerate()
        {
            assert!(
                (g - w).abs() <= DOT_TOLERANCE * w.abs().max(1.0),
                "state {i}: {g} vs {w}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_vulkan_tensor_takes_in_place_writes() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let dst = Tensor::zeros((3, 4), DType::F32, &vk)?;
        let src = Tensor::ones((3, 2), DType::F32, &vk)?;
        dst.slice_set(&src, 1, 1)?;
        let want = Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?;
        want.slice_set(&Tensor::ones((3, 2), DType::F32, &Device::Cpu)?, 1, 1)?;
        assert_same(&dst, &want)?;
        assert_eq!(
            dst.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
            want.to_vec2::<f32>()?
        );
        Ok(())
    }
}
