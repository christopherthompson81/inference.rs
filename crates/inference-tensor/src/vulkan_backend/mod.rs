//! The Vulkan backend: storage in host-visible memory the iGPU and the CPU share, on `inference_vulkan`'s stream. An op
//! with no Vulkan kernel yet waits for the GPU, runs the CPU backend's on a copy and uploads the result, so a model
//! runs whole while its kernels are ported one at a time.

use std::sync::{Arc, Mutex, OnceLock};

use inference_vulkan::{
    BlockType, Bound, Buffer, Context, MAX_DIMS, Map, MapOp, MatMul, MatOperand, MatVec,
    MatrixCores, Operand, RowOp, Rows, Select,
};

use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, DeviceLocation, Layout, Result, Shape, Tensor, WithDType};

// a zero-element tensor still gets a buffer
const MIN_BUFFER_BYTES: usize = 16;
// below this an iGPU's serial scan beats chunked SSD (Radeon 780M, a Lightning layer: 38 against 43 ms at 3072, 61
// against 57 at 4096; `prompt_scan_time`)
const UNIFIED_CHUNKED_SCAN_MIN_SEQ: usize = 3584;
// paged prefill work (rows x context x heads x head size) a submit takes, ~100 ms on a Radeon 780M; whole 4096-row
// chunks there passed amdgpu's job timeout by a 16K-token prompt
const PAGED_PREFILL_WORK_PER_SUBMIT: usize = 16_000_000_000;
// rows a piece keeps at least, the scalar kernel's workgroup
const PAGED_PREFILL_MIN_PIECE_ROWS: usize = 8;

static CONTEXT: OnceLock<std::result::Result<Arc<Context>, String>> = OnceLock::new();

// the device goes before the libraries' own exit handlers, which otherwise tear the driver down under its threads
#[cfg(unix)]
extern "C" fn shut_down_at_exit() {
    if let Some(Ok(context)) = CONTEXT.get() {
        context.shut_down();
    }
}

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

    fn buffer(&self, bytes: usize) -> Result<Buffer> {
        Buffer::new(&self.context, bytes.max(MIN_BUFFER_BYTES)).map_err(vk_err)
    }

    /// A buffer of `bytes` the device reads without an import's per-submit cost, allocated to size and never pooled,
    /// for a weight that lives as long as the model.
    pub(crate) fn weight_copy(&self, bytes: &[u8]) -> Result<Arc<Buffer>> {
        let mut buffer = Buffer::exact(&self.context, bytes.len()).map_err(vk_err)?;
        buffer.write(0, bytes).map_err(vk_err)?;
        Ok(Arc::new(buffer))
    }

    fn storage_from_bytes(
        &self,
        bytes: &[u8],
        count: usize,
        dtype: DType,
    ) -> Result<VulkanStorage> {
        let mut buffer = self.buffer(bytes.len())?;
        buffer.write(0, bytes).map_err(vk_err)?;
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
    buffer: Arc<Buffer>,
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

// the buffer's first `count` values, once the GPU has finished writing them
fn values_of<T: WithDType>(buffer: &Buffer, count: usize) -> Result<Vec<T>> {
    let mut out = Vec::<T>::with_capacity(count);
    // SAFETY: the bytes are written in full below, and any bit pattern of a `WithDType` is valid
    unsafe {
        let bytes =
            std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), count * size_of::<T>());
        buffer.read_into(bytes).map_err(vk_err)?;
        out.set_len(count);
    }
    Ok(out)
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
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    pub fn count(&self) -> usize {
        self.count
    }

    // the CPU backend's copy, after the GPU has finished every write to it
    fn cpu(&self) -> Result<CpuStorage> {
        let (buffer, count) = (&self.buffer, self.count);
        Ok(match self.dtype {
            DType::U8 => CpuStorage::U8(values_of(buffer, count)?),
            DType::U32 => CpuStorage::U32(values_of(buffer, count)?),
            DType::I16 => CpuStorage::I16(values_of(buffer, count)?),
            DType::I32 => CpuStorage::I32(values_of(buffer, count)?),
            DType::I64 => CpuStorage::I64(values_of(buffer, count)?),
            DType::BF16 => CpuStorage::BF16(values_of(buffer, count)?),
            DType::F16 => CpuStorage::F16(values_of(buffer, count)?),
            DType::F32 => CpuStorage::F32(values_of(buffer, count)?),
            DType::F64 => CpuStorage::F64(values_of(buffer, count)?),
            DType::F8E4M3 => CpuStorage::F8E4M3(values_of(buffer, count)?),
            DType::F6E2M3 => CpuStorage::F6E2M3(values_of(buffer, count)?),
            DType::F6E3M2 => CpuStorage::F6E3M2(values_of(buffer, count)?),
            DType::F4 => CpuStorage::F4(values_of(buffer, count)?),
            DType::F8E8M0 => CpuStorage::F8E8M0(values_of(buffer, count)?),
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
        "gelu_erf" => MapOp::GeluErf,
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
    beta: Option<(&VulkanStorage, &Layout)>,
    op: RowOp,
    eps: f32,
) -> Result<Option<(VulkanStorage, Shape)>> {
    let param_ok = |p: Option<(&VulkanStorage, &Layout)>| {
        p.is_none_or(|(a, al)| a.dtype == DType::F32 && al.is_contiguous())
    };
    if storage.dtype != DType::F32
        || !l.is_contiguous()
        || l.shape().rank() == 0
        || !param_ok(alpha)
        || !param_ok(beta)
    {
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
                beta: beta.map(|(b, bl)| (&b.buffer, bl.start_offset())),
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
pub(crate) fn rel_pos_attn(op: &crate::vulkan::RelPosAttn<'_>) -> Result<Option<Tensor>> {
    let f32 = DType::F32;
    let (batch, heads, positions, head_size) = op.q_content.dims4()?;
    let (_, _, keys, _) = op.k.dims4()?;
    let (_, _, p_rows, _) = op.positions.dims4()?;
    let device = op.q_content.device().as_vulkan_device()?.clone();
    let shapes_match = op.q_position.dims() == op.q_content.dims()
        && op.k.dims() == [batch, heads, keys, head_size]
        && op.v.dims() == op.k.dims()
        && op.positions.dims() == [batch, heads, p_rows, head_size];
    let operands = [op.q_content, op.q_position, op.k, op.v, op.positions];
    if !shapes_match
        || !operands.iter().all(|t| t.dtype() == f32)
        || !inference_vulkan::RelPosAttn::supports(head_size)
        || !device.context.has_matrix_cores()
        || op.pos_base + 1 < positions
        || op.pos_base + keys > p_rows
    {
        return Ok(None);
    }
    let [qc, qp, k, v, p] = operands.map(|t| t.contiguous());
    let [qc, qp, k, v, p] = [qc?, qp?, k?, v?, p?];
    let [qcb, qpb, kb, vb, pb] = [&qc, &qp, &k, &v, &p].map(|t| bound(t, f32));
    let [qcb, qpb, kb, vb, pb] = [qcb?, qpb?, kb?, vb?, pb?];
    if [qcb.1, qpb.1].iter().any(|o| o % 4 != 0) {
        return Ok(None);
    }
    // keys, values and positions as padded f16 tables the kernel loads matrices from
    let table = |x: &(Arc<Buffer>, usize), rows: usize| -> Result<Arc<Buffer>> {
        let bytes = inference_vulkan::RelPosAttn::table_bytes(batch, heads, rows, head_size);
        let table = Arc::new(device.buffer(bytes)?);
        let count = batch * heads * rows * head_size;
        device
            .context
            .record(|r| r.relpos_table((&x.0, x.1), count, head_size, &table))
            .map_err(vk_err)?;
        Ok(table)
    };
    let (kt, vt, pt) = (table(&kb, keys)?, table(&vb, keys)?, table(&pb, p_rows)?);
    let count = batch * heads * positions * head_size;
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(count * 4)?),
        device: device.clone(),
        count,
        dtype: f32,
    };
    let piece = inference_vulkan::RelPosAttn::piece_positions(heads, keys, head_size);
    // a piece a record, so the stream submits between them
    for first in (0..positions).step_by(piece) {
        device
            .context
            .record(|r| {
                r.relpos_attn(&inference_vulkan::RelPosAttn {
                    batch,
                    heads,
                    positions,
                    keys,
                    head_size,
                    p_rows,
                    pos_base: op.pos_base,
                    first,
                    count: piece.min(positions - first),
                    qc: (&qcb.0, qcb.1),
                    qp: (&qpb.0, qpb.1),
                    k: &kt,
                    v: &vt,
                    p: &pt,
                    out: &out.buffer,
                })
            })
            .map_err(vk_err)?;
    }
    Ok(Some(tensor_of(out, (batch, heads, positions, head_size))))
}

fn bound(t: &Tensor, dtype: DType) -> Result<(Arc<Buffer>, usize)> {
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
    let unified = step.x.device().as_vulkan_device()?.context.is_unified();
    let chunked = !unified || step.x.dim(1)? >= UNIFIED_CHUNKED_SCAN_MIN_SEQ;
    scan_step(step, chunked)
}

fn scan_step(step: &crate::vulkan::ScanStep<'_>, chunked: bool) -> Result<Tensor> {
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
    let scratch = inference_vulkan::ScanStep::chunked_scratch_bytes(batch, seq, heads, head_dim)
        .filter(|_| chunked)
        .map(|bytes| device.buffer(bytes).map(Arc::new))
        .transpose()?;
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
                scratch: scratch.as_ref(),
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

// max(x, 0)^2 in one map, where `x`'s storage is an f32 Vulkan buffer the map reads
pub(crate) fn relu_sqr(x: &Tensor) -> Result<Option<Tensor>> {
    let out = match &*x.storage() {
        crate::Storage::Vulkan(s) if s.native(x.layout()) => {
            s.map_f32(MapOp::Relu2, x.layout(), None, [0.0; 2])?
        }
        _ => return Ok(None),
    };
    Ok(Some(tensor_of(out, x.dims())))
}

pub(crate) fn gated_rmsnorm(
    x: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f32,
    group_size: usize,
) -> Result<Tensor> {
    let width = x.dim(crate::D::Minus1)?;
    if gate.dims() != x.dims()
        || weight.elem_count() != width
        || group_size == 0
        || width % group_size != 0
    {
        crate::bail!(
            "gated RMSNorm of x {:?}, gate {:?}, weight {:?}, groups of {group_size}",
            x.dims(),
            gate.dims(),
            weight.dims()
        );
    }
    let f32 = DType::F32;
    let (x, gate, weight) = (x.contiguous()?, gate.contiguous()?, weight.contiguous()?);
    let (xb, gb, wb) = (bound(&x, f32)?, bound(&gate, f32)?, bound(&weight, f32)?);
    let device = x.device().as_vulkan_device()?.clone();
    let count = x.elem_count();
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(count * 4)?),
        device: device.clone(),
        count,
        dtype: f32,
    };
    device
        .context
        .record(|r| {
            r.gated_rmsnorm(&inference_vulkan::GatedRmsNorm {
                rows: count / width,
                groups: width / group_size,
                group_size,
                eps,
                x: (&xb.0, xb.1),
                gate: (&gb.0, gb.1),
                weight: (&wb.0, wb.1),
                out: &out.buffer,
            })
        })
        .map_err(vk_err)?;
    Ok(tensor_of(out, x.dims()))
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
    weights: &Arc<Buffer>,
    ty: BlockType,
    (experts, rows, cols): (usize, usize, usize),
    x: &Tensor,
    ids: &Tensor,
    x_div: usize,
) -> Result<Tensor> {
    let (xb, ib) = (bound(x, DType::F32)?, bound(ids, DType::U32)?);
    let slots = ids.elem_count();
    let device = x.device().as_vulkan_device()?.clone();
    let scratch = Arc::new(device.buffer(inference_vulkan::MoeMatVec::scratch_bytes(
        &device.context,
        slots,
        x_div,
        cols,
    ))?);
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
            r.moe_mat_vec(&inference_vulkan::MoeMatVec {
                ty,
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

// the geometry of vLLM-layout f32 or f16 caches (keys `[blocks, kv_heads, head_size / x, block_size, x]`), or of
// quantized HND rows `[blocks, kv_heads, block_size, row bytes]` holding `head_size` values each
fn cache_shape(
    key_cache: &Tensor,
    value_cache: &Tensor,
    head_size: usize,
) -> Result<inference_vulkan::CacheShape> {
    if key_cache.dtype() == DType::U8 && value_cache.dtype() == DType::U8 {
        let (_, kv_heads, block_size, k_row) = key_cache.dims4()?;
        let (_, _, _, v_row) = value_cache.dims4()?;
        let bits = |row| inference_vulkan::CacheShape::quant_bits(row, head_size);
        let (Some(k), Some(v)) = (bits(k_row), bits(v_row)) else {
            crate::bail!(
                "Vulkan's quantized cache rows of {k_row} / {v_row} bytes do not hold head size {head_size}"
            );
        };
        return Ok(inference_vulkan::CacheShape {
            kv_heads,
            head_size,
            block_size,
            x: 0,
            f16: false,
            quant: Some((k, v)),
        });
    }
    let (_, kv_heads, _, block_size, x) = key_cache.dims5()?;
    let (_, _, cache_head, _) = value_cache.dims4()?;
    let f16 = match (key_cache.dtype(), value_cache.dtype()) {
        (DType::F32, DType::F32) => false,
        (DType::F16, DType::F16) => true,
        (k, v) => {
            crate::bail!("Vulkan's paged caches are f32, f16 or quantized, got {k:?} / {v:?}")
        }
    };
    if cache_head != head_size {
        crate::bail!("Vulkan's paged caches hold head size {cache_head}, not {head_size}");
    }
    Ok(inference_vulkan::CacheShape {
        kv_heads,
        head_size,
        block_size,
        x,
        f16,
        quant: None,
    })
}

// a quantized cache's query in its rotated space, in a buffer of its own
fn rotated_query(
    device: &VulkanDevice,
    shape: &inference_vulkan::CacheShape,
    q: (Arc<inference_vulkan::Buffer>, usize),
    count: usize,
) -> Result<(Arc<inference_vulkan::Buffer>, usize)> {
    if shape.quant.is_none() {
        return Ok(q);
    }
    let rotated = Arc::new(device.buffer(count * 4)?);
    device
        .context
        .record(|r| r.hadamard_32((&q.0, q.1), (&rotated, 0), count))
        .map_err(vk_err)?;
    Ok((rotated, 0))
}

pub(crate) fn paged_cache_write(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let shape = cache_shape(key_cache, value_cache, key.dim(2)?)?;
    let tokens = key.dim(0)?;
    let f32 = DType::F32;
    let (k, v) = (
        bound(&key.contiguous()?, f32)?,
        bound(&value.contiguous()?, f32)?,
    );
    let (kc, vc) = (
        bound(key_cache, key_cache.dtype())?,
        bound(value_cache, value_cache.dtype())?,
    );
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

pub(crate) fn flash_attn(op: &crate::vulkan::FlashAttn<'_>) -> Result<Option<Tensor>> {
    let f32 = DType::F32;
    let (batch, heads, positions, head_size) = op.q.dims4()?;
    let (k_batch, kv_heads, keys, k_head) = op.k.dims4()?;
    let shapes_match = k_batch == batch && k_head == head_size && op.v.dims() == op.k.dims();
    let all_f32 = [op.q, op.k, op.v].iter().all(|t| t.dtype() == f32);
    if !shapes_match
        || !all_f32
        || !inference_vulkan::FlashAttn::supports(heads, kv_heads, head_size)
        || (op.causal && keys < positions)
    {
        return Ok(None);
    }
    // a mask shared by every batch and head, as one (positions, keys) matrix
    let mask = match op.mask {
        Some(m) => {
            let dims = m.dims();
            let rank = dims.len();
            let leading_ones = dims[..rank.saturating_sub(2)].iter().all(|&d| d == 1);
            if m.dtype() != f32 || rank < 2 || !leading_ones {
                return Ok(None);
            }
            let m = m.reshape((dims[rank - 2], dims[rank - 1]))?;
            Some(m.broadcast_as((positions, keys))?.contiguous()?)
        }
        None => None,
    };
    let device = op.q.device().as_vulkan_device()?.clone();
    let (q, k, v) = (op.q.contiguous()?, op.k.contiguous()?, op.v.contiguous()?);
    let (qb, kb, vb) = (bound(&q, f32)?, bound(&k, f32)?, bound(&v, f32)?);
    let mb = mask.as_ref().map(|m| bound(m, f32)).transpose()?;
    if [qb.1, kb.1, vb.1].iter().any(|o| o % 4 != 0) {
        return Ok(None);
    }
    let count = batch * heads * positions * head_size;
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(count * 4)?),
        device: device.clone(),
        count,
        dtype: f32,
    };
    let piece = inference_vulkan::FlashAttn::piece_positions(heads, keys, head_size);
    // a piece a record, so the stream submits between them
    for first in (0..positions).step_by(piece) {
        device
            .context
            .record(|r| {
                r.flash_attn(&inference_vulkan::FlashAttn {
                    batch,
                    heads,
                    kv_heads,
                    positions,
                    keys,
                    head_size,
                    scale: op.scale,
                    causal: op.causal,
                    first,
                    count: piece.min(positions - first),
                    q: (&qb.0, qb.1),
                    k: (&kb.0, kb.1),
                    v: (&vb.0, vb.1),
                    mask: mb.as_ref().map(|m| (&m.0, m.1)),
                    out: &out.buffer,
                })
            })
            .map_err(vk_err)?;
    }
    Ok(Some(tensor_of(out, (batch, heads, positions, head_size))))
}

// attention's output over a quantized cache, rotated back out of the cache's space in place
fn unrotate(
    r: &inference_vulkan::Recorder<'_>,
    shape: &inference_vulkan::CacheShape,
    out: &Arc<inference_vulkan::Buffer>,
    count: usize,
) -> std::result::Result<(), inference_vulkan::Error> {
    if shape.quant.is_some() {
        r.hadamard_32((out, 0), (out, 0), count)?;
    }
    Ok(())
}

pub(crate) fn paged_prefill(op: &crate::vulkan::PagedPrefill<'_>) -> Result<Tensor> {
    paged_prefill_in_pieces(op, PAGED_PREFILL_WORK_PER_SUBMIT)
}

fn paged_prefill_in_pieces(
    op: &crate::vulkan::PagedPrefill<'_>,
    work_per_submit: usize,
) -> Result<Tensor> {
    let (rows, heads, head_size) = op.q.dims3()?;
    let shape = cache_shape(op.key_cache, op.value_cache, head_size)?;
    let f32 = DType::F32;
    let device = op.q.device().as_vulkan_device()?.clone();
    let table = op
        .block_table
        .to_device(op.q.device())?
        .to_dtype(DType::U32)?
        .contiguous()?;
    let (q, kc, vc, tb) = (
        bound(&op.q.contiguous()?, f32)?,
        bound(op.key_cache, op.key_cache.dtype())?,
        bound(op.value_cache, op.value_cache.dtype())?,
        bound(&table, DType::U32)?,
    );
    let count = rows * heads * head_size;
    let q = rotated_query(&device, &shape, q, count)?;
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(count * 4)?),
        device: device.clone(),
        count,
        dtype: f32,
    };
    // rows [r0, r1) over the first `context - rows + r1` positions see the keys they would in one call
    let row_work = (op.context * heads * head_size).max(1);
    let piece = (work_per_submit / row_work)
        .max(1)
        .next_multiple_of(PAGED_PREFILL_MIN_PIECE_ROWS)
        .min(rows.max(1));
    let row_len = heads * head_size;
    for r0 in (0..rows).step_by(piece) {
        let r1 = (r0 + piece).min(rows);
        let context = op.context - rows + r1;
        let partials = Arc::new(
            device.buffer(inference_vulkan::PagedPrefill::partials_bytes(
                r1 - r0,
                heads,
                head_size,
                context,
            ))?,
        );
        device
            .context
            .record(|r| {
                r.paged_prefill(&inference_vulkan::PagedPrefill {
                    shape,
                    rows: r1 - r0,
                    heads,
                    context,
                    scale: op.scale,
                    q: (&q.0, q.1 + r0 * row_len),
                    key_cache: &kc.0,
                    value_cache: &vc.0,
                    block_table: (&tb.0, tb.1),
                    partials: &partials,
                    out: (&out.buffer, r0 * row_len),
                })
            })
            .map_err(vk_err)?;
    }
    device
        .context
        .record(|r| unrotate(r, &shape, &out.buffer, count))
        .map_err(vk_err)?;
    Ok(tensor_of(out, (rows, heads, head_size)))
}

pub(crate) fn penalize(
    logits: &Tensor,
    entries: &[[u32; 4]],
    frequency: f32,
    presence: f32,
    repetition: f32,
) -> Result<Tensor> {
    let (_, vocab) = logits.dims2()?;
    // a copy, so the model's logits stay as they were
    let out = logits.contiguous()?.copy()?;
    if entries.is_empty() {
        return Ok(out);
    }
    let flat: Vec<u32> = entries.iter().flatten().copied().collect();
    let entries_t = Tensor::from_vec(flat, entries.len() * 4, logits.device())?;
    let (x, x_off) = bound(&out, DType::F32)?;
    let (e, e_off) = bound(&entries_t, DType::U32)?;
    let device = logits.device().as_vulkan_device()?.clone();
    device
        .context
        .record(|r| {
            r.penalize(&inference_vulkan::Penalize {
                vocab,
                count: entries.len(),
                logits: (&x, x_off),
                entries: (&e, e_off),
                frequency,
                presence,
                repetition,
            })
        })
        .map_err(vk_err)?;
    Ok(out)
}

pub(crate) fn ranked_top_k(logits: &Tensor, temperature: f64) -> Result<Tensor> {
    const _: () = assert!(crate::vulkan::RANKED_TOP_K == inference_vulkan::RankedTopK::K);
    let (rows, vocab) = logits.dims2()?;
    let logits = logits.contiguous()?;
    let (x, x_off) = bound(&logits, DType::F32)?;
    let device = logits.device().as_vulkan_device()?.clone();
    let count = rows * inference_vulkan::RankedTopK::row_floats();
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(count * 4)?),
        device: device.clone(),
        count,
        dtype: DType::F32,
    };
    let scratch =
        Arc::new(device.buffer(inference_vulkan::RankedTopK::scratch_bytes(rows, vocab))?);
    device
        .context
        .record(|r| {
            r.ranked_top_k(&inference_vulkan::RankedTopK {
                rows,
                vocab,
                inv_temperature: (1.0 / temperature) as f32,
                logits: (&x, x_off),
                scratch: &scratch,
                out: &out.buffer,
            })
        })
        .map_err(vk_err)?;
    Ok(tensor_of(
        out,
        (rows, inference_vulkan::RankedTopK::row_floats()),
    ))
}

pub(crate) fn ranked_draw(packed: &Tensor, params: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let k = inference_vulkan::RankedTopK::K;
    let (rows, width) = packed.dims2()?;
    if width != inference_vulkan::RankedTopK::row_floats()
        || params.dims2()? != (rows, inference_vulkan::RankedDraw::PARAM_WORDS)
    {
        crate::bail!(
            "a ranked draw takes packed (rows, {}) and params (rows, {}), got {:?} and {:?}",
            inference_vulkan::RankedTopK::row_floats(),
            inference_vulkan::RankedDraw::PARAM_WORDS,
            packed.dims(),
            params.dims()
        );
    }
    let (packed, params) = (packed.contiguous()?, params.contiguous()?);
    let (p, p_off) = bound(&packed, DType::F32)?;
    let (q, q_off) = bound(&params, DType::U32)?;
    let device = packed.device().as_vulkan_device()?.clone();
    let storage = |count: usize, dtype: DType| -> Result<VulkanStorage> {
        Ok(VulkanStorage {
            buffer: Arc::new(device.buffer(count * 4)?),
            device: device.clone(),
            count,
            dtype,
        })
    };
    let (ids, probs, tokens) = (
        storage(rows * k, DType::U32)?,
        storage(rows * k, DType::F32)?,
        storage(rows, DType::U32)?,
    );
    device
        .context
        .record(|r| {
            r.ranked_draw(&inference_vulkan::RankedDraw {
                rows,
                packed: (&p, p_off),
                params: (&q, q_off),
                ids: &ids.buffer,
                probs: &probs.buffer,
                tokens: &tokens.buffer,
            })
        })
        .map_err(vk_err)?;
    Ok((
        tensor_of(ids, (rows, k)),
        tensor_of(probs, (rows, k)),
        tensor_of(tokens, (rows,)),
    ))
}

pub(crate) fn paged_decode(op: &crate::vulkan::PagedDecode<'_>) -> Result<Tensor> {
    let (seqs, heads, head_size) = op.q.dims3()?;
    let shape = cache_shape(op.key_cache, op.value_cache, head_size)?;
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
        bound(op.key_cache, op.key_cache.dtype())?,
        bound(op.value_cache, op.value_cache.dtype())?,
    );
    let (tb, lb) = (bound(&tables, DType::U32)?, bound(&lens, DType::U32)?);
    let count = seqs * heads * head_size;
    let q = rotated_query(&device, &shape, q, count)?;
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
            })?;
            unrotate(r, &shape, &out.buffer, count)
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
    weights: &Arc<Buffer>,
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

/// The `count` rows of a `vocab x cols` quantized weight at the u32 `ids` (bound at their element offset).
pub(crate) fn quantized_embedding(
    ids: (&Arc<Buffer>, usize),
    count: usize,
    weights: &Arc<Buffer>,
    ty: BlockType,
    vocab: usize,
    cols: usize,
    device: &VulkanDevice,
) -> Result<VulkanStorage> {
    let len = count * cols;
    let out = VulkanStorage {
        buffer: Arc::new(device.buffer(len * 4)?),
        device: device.clone(),
        count: len,
        dtype: DType::F32,
    };
    let weights: Arc<dyn Bound> = weights.clone();
    device
        .context
        .record(|r| {
            r.embed(&inference_vulkan::Embed {
                ty,
                weights: &weights,
                vocab,
                cols,
                ids,
                count,
                y: &out.buffer,
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
            && let Some((out, _)) = rows(self, l, None, None, row_op, 0.0)?
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
        let native = self.dtype == DType::F32
            && kernel.dtype == DType::F32
            && l.stride().len() == 4
            && kernel_l.is_contiguous();
        if native {
            let p = params;
            let (o_h, o_w) = (p.out_h(), p.out_w());
            let out = self.empty(p.b_size * p.groups * p.c_out * o_h * o_w, DType::F32)?;
            let s = l.stride();
            self.device
                .context
                .record(|r| {
                    r.conv2d_f32(&inference_vulkan::Conv2d {
                        batch: p.b_size,
                        c_in: p.c_in,
                        c_out: p.c_out,
                        groups: p.groups,
                        i_h: p.i_h,
                        i_w: p.i_w,
                        k_h: p.k_h,
                        k_w: p.k_w,
                        o_h,
                        o_w,
                        padding: p.padding,
                        stride: p.stride,
                        dilation: p.dilation,
                        x: (&self.buffer, l.start_offset()),
                        x_strides: [s[0], s[1], s[2], s[3]],
                        w: (&kernel.buffer, kernel_l.start_offset()),
                        out: &out.buffer,
                    })
                })
                .map_err(vk_err)?;
            return Ok(out);
        }
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
            .get_or_init(|| {
                let set = |name| std::env::var_os(name).is_some();
                let mode = if set(crate::vulkan::NO_MATRIX_CORES_ENV) {
                    MatrixCores::Off
                } else if set(crate::vulkan::IGPU_MATRIX_CORES_ENV) {
                    MatrixCores::Any
                } else {
                    MatrixCores::Default
                };
                let context = Context::with_matrix_cores(mode).map_err(|err| err.to_string())?;
                #[cfg(unix)]
                // SAFETY: the handler only reads the initialized static
                unsafe {
                    libc::atexit(shut_down_at_exit);
                }
                Ok(context)
            })
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
        buffer.fill(0).map_err(vk_err)?;
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
    // erf GELU from -10 to 10 in steps of 0.01: both tails and the bend between
    const GELU_ERF_STEPS: f32 = 1000.;
    const GELU_ERF_STEP: f64 = 0.01;
    // a 2688-term dot product summed in another order
    const DOT_TOLERANCE: f32 = 1e-3;
    // attention with f16 queries, keys, values and probabilities (f32 sums): unit-variance inputs land within ~1.5e-3
    const F16_ATTENTION_TOLERANCE: f32 = 4e-3;
    // a 2688-term dot product over activations rounded to int8 blocks, against the outputs' RMS: unit-normal weights
    // and activations leave ~0.6 % (one deviation), and the worst of the expert test's ~28k outputs reaches ~3.1 %
    const INT8_DOT_TOLERANCE: f32 = 5e-2;

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
        let wide = Tensor::arange(-GELU_ERF_STEPS, GELU_ERF_STEPS, &Device::Cpu)?
            .affine(GELU_ERF_STEP, 0.)?;
        assert_same(&wide.to_device(&vk)?.gelu_erf()?, &wide.gelu_erf()?)?;
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
        // 2816 columns: whole 256-value blocks for the k-quants, and a block count no thread stride divides evenly
        let w = Tensor::randn(0f32, 1., (97, 2816), &Device::Cpu)?;
        for dtype in [
            GgmlDType::Q8_0,
            GgmlDType::Q5_0,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_1,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let weight = QTensor::quantize(&w.to_device(&vk)?, dtype)?;
            let dense = weight.dequantize(&Device::Cpu)?;
            let matmul = QMatMul::from_qtensor(weight)?;
            // past 8 rows, the prompt matmul
            for rows in [1usize, 3, 8, 70] {
                let x = Tensor::randn(0f32, 1., (1, rows, 2816), &Device::Cpu)?;
                let want = x.broadcast_matmul(&dense.t()?)?;
                assert_int8_close(&matmul.forward(&x.to_device(&vk)?)?, &want)?;
            }
        }
        Ok(())
    }

    // an expert stack's routes, for each block type the kernels take: one token's (a kernel a route), then several
    // tokens sharing experts (the kernel reading each expert once), over a token's input (x_div k) and over each
    // route's own row (x_div 1)
    #[test]
    fn vulkan_expert_matvecs_match_the_dequantized_experts() -> Result<()> {
        use crate::quantized::{GgmlDType, QTensor};
        let Some(vk) = device() else { return Ok(()) };
        let (experts, rows, cols, k) = (4usize, 71usize, 256usize, 2usize);
        let w = Tensor::randn(0f32, 1., (experts, rows, cols), &Device::Cpu)?;
        // the last: 18 slots of one expert, more than a leader takes
        // a prompt's 150 routes, past the grouped kernel's, take the tiled matmul over routes sorted by expert, one
        // expert past a tile of them; 400 average past what sends a prompt's experts to the matrix cores
        let prompt: Vec<u32> = (0..150u32)
            .map(|i| if i % 3 == 0 { 2 } else { i * 7 % 4 })
            .collect();
        let long: Vec<u32> = (0..400u32).map(|i| (i * 5 + i / 7) % 4).collect();
        // one expert of 10 routes beside busy ones (the narrowest tiles)
        let uneven: Vec<u32> = (0..140u32)
            .map(|i| if i % 14 == 0 { 3 } else { i % 3 })
            .collect();
        let routes: [&[u32]; 6] = [
            &[3, 1],
            &[2, 0, 0, 2, 1, 2],
            &[1; 18],
            &prompt,
            &long,
            &uneven,
        ];
        for dtype in [
            GgmlDType::Q5_0,
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_1,
            GgmlDType::Q6K,
        ] {
            let weight = QTensor::quantize(&w.to_device(&vk)?, dtype)?;
            let dense = weight.dequantize(&Device::Cpu)?;
            for ids in routes {
                let tokens = ids.len() / k;
                let id_t = Tensor::from_slice(ids, (tokens, k), &Device::Cpu)?;
                for x_div in [k, 1] {
                    let x = Tensor::randn(0f32, 1., (ids.len() / x_div, cols), &Device::Cpu)?;
                    let got = weight
                        .vulkan_expert_mat_vec(&x.to_device(&vk)?, &id_t.to_device(&vk)?, x_div)?
                        .expect("the type runs on Vulkan");
                    let mut want = Vec::new();
                    for (slot, &e) in ids.iter().enumerate() {
                        let row = x.narrow(0, slot / x_div, 1)?;
                        want.push(row.matmul(&dense.get(e as usize)?.t()?)?);
                    }
                    assert_int8_close(&got, &Tensor::cat(&want, 0)?)?;
                }
            }
        }
        Ok(())
    }

    // the tiled matmul over sizes off its tiles: attention's q k^T (b strided along n) and p v (b unit-stride along n)
    #[test]
    fn vulkan_tiled_matmuls_match_the_cpu() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        // on matrix cores the inputs are rounded to f16, whose products f32 holds exactly: the reference rounds them too
        let cores = vk.as_vulkan_device()?.context.has_matrix_cores();
        let rounded = |t: &Tensor| -> Result<Tensor> {
            if cores {
                t.to_dtype(DType::F16)?.to_dtype(DType::F32)
            } else {
                Ok(t.clone())
            }
        };
        let check = |a: &Tensor, b: &Tensor| -> Result<()> {
            let got = a.to_device(&vk)?.matmul(&b.to_device(&vk)?)?;
            assert_close(&got, &rounded(a)?.matmul(&rounded(b)?)?, DOT_TOLERANCE)
        };
        let q = Tensor::randn(0f32, 1., (3, 70, 40), &Device::Cpu)?;
        let k = Tensor::randn(0f32, 1., (3, 131, 40), &Device::Cpu)?;
        let v = Tensor::randn(0f32, 1., (3, 131, 24), &Device::Cpu)?;
        // b transposed (k its unit stride), then an unaligned k against an n-contiguous b
        check(&q, &k.t()?)?;
        check(&q.matmul(&k.t()?)?, &v)?;
        // whole 128 x 128 tiles and their edges, against a transposed weight and an n-contiguous b
        let x = Tensor::randn(0f32, 1., (1, 300, 256), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 1., (260, 256), &Device::Cpu)?;
        check(&x, &w.t()?.unsqueeze(0)?)?;
        let y = Tensor::randn(0f32, 1., (2, 64, 200), &Device::Cpu)?;
        check(
            &x.narrow(1, 0, 130)?.narrow(2, 0, 64)?.repeat((2, 1, 1))?,
            &y,
        )?;
        // a strided a (its k not unit): no vector loads
        check(&q.transpose(1, 2)?.contiguous()?.transpose(1, 2)?, &k.t()?)?;
        Ok(())
    }

    // f32 matmul throughput at a speech encoder's shapes (Parakeet over 10 minutes: 7500 frames, 1024 wide, a
    // transposed weight), on whichever kernel the device takes
    #[test]
    #[ignore]
    fn f32_matmul_throughput() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        const RUNS: usize = 10;
        for (m, n, k) in [
            (7500, 4096, 1024),
            (7500, 1024, 4096),
            (7500, 1024, 1024),
            (7500, 256, 1024),
            (1024, 256, 1024),
        ] {
            let x = Tensor::randn(0f32, 1., (m, k), &vk)?;
            let w = Tensor::randn(0f32, 1., (n, k), &vk)?;
            x.matmul(&w.t()?)?.sum_all()?.to_scalar::<f32>()?;
            let start = std::time::Instant::now();
            let mut last = None;
            for _ in 0..RUNS {
                last = Some(x.matmul(&w.t()?)?);
            }
            last.expect("runs").sum_all()?.to_scalar::<f32>()?;
            let secs = start.elapsed().as_secs_f64() / RUNS as f64;
            let tflops = 2.0 * (m * n * k) as f64 / secs / 1e12;
            eprintln!("{m} x {n} x {k}: {:.1} ms, {tflops:.2} TFLOPS", secs * 1e3);
        }
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
        // off-centre rows, so the mean matters
        let shifted = (&x * 1.5)?.affine(1., 2.)?;
        let beta = Tensor::randn(0f32, 0.5, 300, &Device::Cpu)?;
        let (vshifted, vbeta) = (shifted.to_device(&vk)?, beta.to_device(&vk)?);
        let norm = crate::nn::ops::layer_norm;
        assert_same(
            &norm(&vshifted, &valpha, &vbeta, 1e-5)?,
            &norm(&shifted, &alpha, &beta, 1e-5)?,
        )?;
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

    // fused attention against softmax(scale q k^T + mask) v on the CPU: GQA, keys past the positions (a cached prefix), a
    // a direct conv2d against the CPU's: one input channel strided and padded, depthwise, two groups, dilated, and a
    // strided input view; a pointwise conv is a matmul, its inputs rounded to f16 where it runs on matrix cores
    #[test]
    fn vulkan_conv2d_matches_the_cpu() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let check = |x: &Tensor, w: &Tensor, padding, stride, dilation, groups| -> Result<()> {
            let got =
                x.to_device(&vk)?
                    .conv2d(&w.to_device(&vk)?, padding, stride, dilation, groups)?;
            assert_close(
                &got,
                &x.conv2d(w, padding, stride, dilation, groups)?,
                DOT_TOLERANCE,
            )
        };
        let randn = |dims: &[usize]| Tensor::randn(0f32, 1., dims, &Device::Cpu);
        let x1 = randn(&[2, 1, 37, 20])?;
        check(&x1, &randn(&[8, 1, 3, 3])?, 1, 2, 1, 1)?;
        let x8 = randn(&[2, 8, 19, 10])?;
        check(&x8, &randn(&[8, 1, 3, 3])?, 1, 2, 1, 8)?;
        let pointwise = randn(&[16, 8, 1, 1])?;
        let got = x8
            .to_device(&vk)?
            .conv2d(&pointwise.to_device(&vk)?, 0, 1, 1, 1)?;
        let rounded = |t: &Tensor| -> Result<Tensor> {
            if vk.as_vulkan_device()?.context.has_matrix_cores() {
                t.to_dtype(DType::F16)?.to_dtype(DType::F32)
            } else {
                Ok(t.clone())
            }
        };
        let want = rounded(&x8)?.conv2d(&rounded(&pointwise)?, 0, 1, 1, 1)?;
        assert_close(&got, &want, DOT_TOLERANCE)?;
        check(&x8, &randn(&[6, 4, 3, 3])?, 2, 1, 2, 2)?;
        check(&x8.transpose(2, 3)?, &randn(&[8, 8, 3, 3])?, 1, 1, 1, 1)?;
        Ok(())
    }

    // relative-position attention against its definition over the same f16-rounded inputs: content and position
    // scores (row i reading position row pos_base - i + j), the softmax and its values; a partial last workgroup (its
    // later subgroups past the positions) and key tile, two heads, head sizes of 128 and 32
    #[test]
    fn relative_position_attention_matches_its_definition() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        if !vk.as_vulkan_device()?.context.has_matrix_cores() {
            return Ok(());
        }
        let f16 = |t: &Tensor| t.to_dtype(DType::F16)?.to_dtype(DType::F32);
        for (t, head) in [(150, 128), (70, 32)] {
            let (heads, p_rows, pos_base) = (2, 2 * t - 1, t - 1);
            let randn =
                |rows: usize| Tensor::randn(0f32, 0.5, (1, heads, rows, head), &Device::Cpu);
            let (qc, qp, k, v, p) = (randn(t)?, randn(t)?, randn(t)?, randn(t)?, randn(p_rows)?);
            let got = crate::vulkan::rel_pos_attn(&crate::vulkan::RelPosAttn {
                q_content: &qc.to_device(&vk)?,
                q_position: &qp.to_device(&vk)?,
                k: &k.to_device(&vk)?,
                v: &v.to_device(&vk)?,
                positions: &p.to_device(&vk)?,
                pos_base,
            })?
            .expect("the device takes it");
            let ids: Vec<u32> = (0..t)
                .flat_map(|i| (0..t).map(move |j| (pos_base - i + j) as u32))
                .collect();
            let ids =
                Tensor::from_vec(ids, (1, 1, t, t), &Device::Cpu)?.repeat((1, heads, 1, 1))?;
            let content = f16(&qc)?.matmul(&f16(&k)?.t()?)?;
            let position = f16(&qp)?
                .matmul(&f16(&p)?.t()?)?
                .contiguous()?
                .gather(&ids, 3)?;
            let probs = crate::nn::ops::softmax_last_dim(&(content + position)?)?;
            let want = probs.matmul(&f16(&v)?)?;
            assert_close(&got, &want, F16_ATTENTION_TOLERANCE)?;
        }
        Ok(())
    }

    // f32 products against the CPU's over the inputs as the device rounds them: widths that leave c's rows off 16-byte
    // alignment (2T - 1 relative positions), with b along n and transposed, a partial tile and a batch of heads
    #[test]
    fn vulkan_matmul_matches_the_cpu_at_odd_widths() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let rounded = |t: &Tensor| -> Result<Tensor> {
            if vk.as_vulkan_device()?.context.has_matrix_cores() {
                t.to_dtype(DType::F16)?.to_dtype(DType::F32)
            } else {
                Ok(t.clone())
            }
        };
        for (batch, m, n) in [(1usize, 300usize, 599usize), (8, 257, 513), (2, 256, 258)] {
            for transposed in [false, true] {
                let a = Tensor::randn(0f32, 1.0, (batch, m, 128), &Device::Cpu)?;
                let b = if transposed {
                    Tensor::randn(0f32, 1.0, (batch, n, 128), &Device::Cpu)?.t()?
                } else {
                    Tensor::randn(0f32, 1.0, (batch, 128, n), &Device::Cpu)?
                };
                let got = a.to_device(&vk)?.matmul(&b.to_device(&vk)?)?;
                assert_close(&got, &rounded(&a)?.matmul(&rounded(&b)?)?, DOT_TOLERANCE)?;
            }
        }
        Ok(())
    }

    // one Parakeet layer's full attention over 10 minutes (7,564 frames, 8 heads of 128): GPU time a call and the
    // useful rate (content scores, position scores and values, 2 flops a multiply-add each)
    #[test]
    #[ignore = "benchmark"]
    fn relative_position_attention_throughput() -> Result<()> {
        const FRAMES: usize = 7564;
        const HEADS: usize = 8;
        const HEAD: usize = 128;
        const REPEATS: usize = 10;
        let Some(vk) = device() else { return Ok(()) };
        if !vk.as_vulkan_device()?.context.has_matrix_cores() {
            return Ok(());
        }
        let randn = |rows: usize| Tensor::randn(0f32, 0.5, (1, HEADS, rows, HEAD), &vk);
        let (qc, qp, k, v) = (
            randn(FRAMES)?,
            randn(FRAMES)?,
            randn(FRAMES)?,
            randn(FRAMES)?,
        );
        let p = randn(2 * FRAMES - 1)?;
        let call = || -> Result<Tensor> {
            Ok(crate::vulkan::rel_pos_attn(&crate::vulkan::RelPosAttn {
                q_content: &qc,
                q_position: &qp,
                k: &k,
                v: &v,
                positions: &p,
                pos_base: FRAMES - 1,
            })?
            .expect("the device takes it"))
        };
        call()?;
        vk.synchronize()?;
        let start = std::time::Instant::now();
        for _ in 0..REPEATS {
            call()?;
        }
        vk.synchronize()?;
        let secs = start.elapsed().as_secs_f64() / REPEATS as f64;
        let flops = 6.0 * (FRAMES * FRAMES * HEADS * HEAD) as f64;
        println!(
            "relpos attention {FRAMES} x {HEADS} x {HEAD}: {:.2} ms, {:.1} TFLOPS",
            secs * 1e3,
            flops / secs / 1e12
        );
        Ok(())
    }

    // partial last tile, with the model's causal mask and without one, causal without a mask; and decode's one position
    // over a cache (its own kernel), fewer keys than a workgroup's subgroups included. Heads of 16s run on matrix cores
    // where the device has them, in f16.
    #[test]
    fn fused_vulkan_attention_matches_its_definition() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        for (batch, heads, kv_heads, positions, keys, head_size, masked, causal_only) in [
            (
                2usize, 32usize, 2usize, 75usize, 75usize, 128usize, true, false,
            ),
            (1, 8, 8, 40, 107, 80, true, false),
            (1, 6, 2, 33, 33, 64, false, false),
            (1, 32, 2, 261, 300, 128, false, true),
            (1, 4, 1, 70, 70, 32, true, false),
            (2, 32, 2, 1, 1037, 128, true, false),
            (1, 8, 8, 1, 5, 80, true, false),
            (1, 6, 2, 1, 300, 64, false, false),
        ] {
            let r = |shape: &[usize]| Tensor::randn(0f32, 1.0, shape, &Device::Cpu);
            let q = r(&[batch, heads, positions, head_size])?;
            let k = r(&[batch, kv_heads, keys, head_size])?;
            let v = r(&[batch, kv_heads, keys, head_size])?;
            let offset = keys - positions;
            let causal: Vec<f32> = (0..positions * keys)
                .map(|i| {
                    if i % keys > i / keys + offset {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect();
            let mask = Tensor::from_vec(causal, (positions, keys), &Device::Cpu)?;
            let scale = 1.0 / (head_size as f32).sqrt();
            let group = heads / kv_heads;
            let expand = |t: &Tensor| -> Result<Tensor> {
                t.unsqueeze(2)?
                    .broadcast_as((batch, kv_heads, group, keys, head_size))?
                    .reshape((batch, heads, keys, head_size))
            };
            let scores = (q.matmul(&expand(&k)?.t()?)? * f64::from(scale))?;
            let scores = if masked || causal_only {
                scores.broadcast_add(&mask)?
            } else {
                scores
            };
            let want = crate::nn::ops::softmax_last_dim(&scores)?.matmul(&expand(&v)?)?;
            let on = |t: &Tensor| t.to_device(&vk);
            let mask_vk = on(&mask)?;
            let got = crate::vulkan::flash_attn(&crate::vulkan::FlashAttn {
                q: &on(&q)?,
                k: &on(&k)?,
                v: &on(&v)?,
                mask: masked.then_some(&mask_vk),
                scale,
                causal: masked || causal_only,
            })?
            .expect("the kernel takes this shape");
            let on_matrix_cores = positions > 1 && head_size.is_multiple_of(16);
            let tolerance = if on_matrix_cores {
                F16_ATTENTION_TOLERANCE
            } else {
                1e-4
            };
            assert_close(&got, &want, tolerance)?;
        }
        Ok(())
    }

    // relu(x)^2 in one map against the CPU's relu then sqr, over a strided view
    // a prompt chunk's rows split into pieces, each over its own causal prefix, score as one call does
    #[test]
    fn paged_prefill_in_pieces_matches_one_call() -> Result<()> {
        let Some(vk) = device() else {
            return Ok(());
        };
        const BLOCK: usize = 16;
        const KV_HEADS: usize = 2;
        const HEADS: usize = 8;
        const HEAD: usize = 64;
        // f16 caches hold 8 dimensions a key row
        const X: usize = 8;
        const CONTEXT: usize = 900;
        const ROWS: usize = 37;
        // 8 rows a piece, the last one short
        const PIECE_WORK: usize = 8 * CONTEXT * HEADS * HEAD;
        let blocks = CONTEXT.div_ceil(BLOCK);
        let kc = Tensor::randn(0f32, 1., (blocks, KV_HEADS, HEAD / X, BLOCK, X), &vk)?
            .to_dtype(DType::F16)?;
        let vc =
            Tensor::randn(0f32, 1., (blocks, KV_HEADS, HEAD, BLOCK), &vk)?.to_dtype(DType::F16)?;
        let table: Vec<u32> = (0..blocks as u32).rev().collect();
        let op = crate::vulkan::PagedPrefill {
            q: &Tensor::randn(0f32, 1., (ROWS, HEADS, HEAD), &vk)?,
            key_cache: &kc,
            value_cache: &vc,
            block_table: &Tensor::new(table.as_slice(), &vk)?,
            context: CONTEXT,
            scale: 1.0 / (HEAD as f32).sqrt(),
        };
        let whole = super::paged_prefill_in_pieces(&op, usize::MAX)?;
        let pieces = super::paged_prefill_in_pieces(&op, PIECE_WORK)?;
        assert_same(&pieces, &whole)
    }

    #[test]
    fn a_fused_relu_sqr_matches_relu_then_sqr() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let x = Tensor::randn(0f32, 1.0, (6, 1856), &Device::Cpu)?;
        let want = x.t()?.relu()?.sqr()?.flatten_all()?.to_vec1::<f32>()?;
        let got = crate::nn::ops::relu_sqr(&x.to_device(&vk)?.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(got, want);
        Ok(())
    }

    // Mamba-2's gated RMSNorm in one pass against its composite on the CPU: a decode row and a prompt's rows, groups
    // narrower and wider than a workgroup
    #[test]
    fn a_fused_gated_rmsnorm_matches_its_composite() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let eps = 1e-5f32;
        // the last past 65535 (rows x groups) workgroups, a long prompt's
        let shapes = [
            (1usize, 8usize, 512usize),
            (5, 3, 100),
            (2, 2, 700),
            (8300, 8, 32),
        ];
        for (rows, groups, group_size) in shapes {
            let width = groups * group_size;
            let x = Tensor::randn(0f32, 1.0, (rows, width), &Device::Cpu)?;
            let gate = Tensor::randn(0f32, 2.0, (rows, width), &Device::Cpu)?;
            let weight = Tensor::randn(1f32, 0.3, width, &Device::Cpu)?;
            let gated = (&x * crate::nn::ops::silu(&gate)?)?.reshape((rows, groups, group_size))?;
            let rms = (gated.sqr()?.mean_keepdim(crate::D::Minus1)? + f64::from(eps))?.sqrt()?;
            let want = gated
                .broadcast_div(&rms)?
                .reshape((rows, width))?
                .broadcast_mul(&weight)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let got = crate::vulkan::gated_rmsnorm(
                &x.to_device(&vk)?,
                &gate.to_device(&vk)?,
                &weight.to_device(&vk)?,
                eps,
                group_size,
            )?
            .flatten_all()?
            .to_vec1::<f32>()?;
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                    "{rows}x{groups}x{group_size}: {g} vs {w}"
                );
            }
        }
        Ok(())
    }

    // the device's top-K a row against a sort here: ties to the lower index, masked entries never in the sum
    #[test]
    fn ranked_top_k_matches_a_sort() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let k = crate::vulkan::RANKED_TOP_K;
        // whole 4096-logit chunks, a remainder chunk, and one chunk
        for (rows, vocab, temperature) in [
            (3usize, 131_072usize, 0.7f64),
            (1, 151_936, 0.8),
            (2, 1000, 1.0),
        ] {
            let mut logits =
                Tensor::randn(0f32, 3.0, (rows, vocab), &Device::Cpu)?.to_vec2::<f32>()?;
            for row in logits.iter_mut() {
                // ties straddling the cut, and masked tokens
                for i in (0..vocab).step_by(97).take(200) {
                    row[i] = 5.5;
                }
                for i in (3..vocab).step_by(11) {
                    row[i] = f32::NEG_INFINITY;
                }
            }
            let flat: Vec<f32> = logits.iter().flatten().copied().collect();
            let packed = crate::vulkan::ranked_top_k(
                &Tensor::from_vec(flat, (rows, vocab), &vk)?,
                temperature,
            )?
            .to_vec2::<f32>()?;
            for (row, got) in logits.iter().zip(&packed) {
                let mut order: Vec<usize> = (0..vocab).collect();
                order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
                let want_idx: Vec<f32> = order[..k].iter().map(|&i| i as f32).collect();
                let want_val: Vec<f32> = order[..k].iter().map(|&i| row[i]).collect();
                assert_eq!(&got[k..2 * k], &want_idx[..]);
                assert_eq!(&got[..k], &want_val[..]);
                let inv = (1.0 / temperature) as f32;
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) * inv;
                let z = row
                    .iter()
                    .filter(|v| v.is_finite())
                    .map(|v| f64::from((v * inv - max).exp()))
                    .sum::<f64>() as f32;
                assert!(
                    (got[2 * k + 1] - max).abs() < 1e-5,
                    "{} vs {max}",
                    got[2 * k + 1]
                );
                assert!((got[2 * k] - z).abs() <= 1e-4 * z, "{} vs {z}", got[2 * k]);
            }
        }
        Ok(())
    }

    // a prompt long enough to run chunked (a partial last chunk, a padded row) against the same scan in short steps
    #[test]
    fn a_chunked_vulkan_scan_matches_its_steps() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let (seq, heads, head_dim, groups, state) = (
            150usize,
            4usize,
            80usize,
            2usize,
            crate::vulkan::SCAN_STEP_STATE,
        );
        let r = |shape: &[usize]| Tensor::randn(0f32, 0.5, shape, &vk);
        let (x, dt, b, c) = (
            r(&[2, seq, heads, head_dim])?,
            r(&[2, seq, heads])?,
            r(&[2, seq, groups, state])?,
            r(&[2, seq, groups, state])?,
        );
        let (a, d, dt_bias) = ((r(&[heads])?.abs()? * -1.)?, r(&[heads])?, r(&[heads])?);
        let start = r(&[3, heads, head_dim, state])?;
        let slots = Tensor::new(&[2u32, u32::MAX], &vk)?;
        let scan = |pool: &Tensor, from: usize, len: usize| {
            super::scan_step(
                &crate::vulkan::ScanStep {
                    x: &x.narrow(1, from, len)?.contiguous()?,
                    dt: &dt.narrow(1, from, len)?.contiguous()?,
                    a: &a,
                    b: &b.narrow(1, from, len)?.contiguous()?,
                    c: &c.narrow(1, from, len)?.contiguous()?,
                    d: &d,
                    dt_bias: &dt_bias,
                    dt_range: [0.0, f32::INFINITY],
                    pool,
                    slots: &slots,
                },
                true,
            )
        };
        let chunked_pool = start.copy()?;
        let chunked = scan(&chunked_pool, 0, seq)?;
        let stepped_pool = start.copy()?;
        let pieces = [0usize, 40, 70, seq];
        let stepped = Tensor::cat(
            &pieces
                .windows(2)
                .map(|w| scan(&stepped_pool, w[0], w[1] - w[0]))
                .collect::<Result<Vec<_>>>()?,
            1,
        )?;
        assert_close(&chunked, &stepped, 2e-4)?;
        assert_close(&chunked_pool, &stepped_pool, 2e-4)?;
        Ok(())
    }

    // a Nemotron 3.5 Lightning Mamba layer's scan over prompts of a few lengths, serial and chunked
    #[test]
    #[ignore = "benchmark"]
    fn prompt_scan_time() -> Result<()> {
        let Some(vk) = device() else { return Ok(()) };
        let (heads, head_dim, groups, state) =
            (64usize, 64usize, 8usize, crate::vulkan::SCAN_STEP_STATE);
        for (seq, chunked) in [128usize, 512, 1024, 2048, 3072, 4096]
            .into_iter()
            .flat_map(|seq| [(seq, false), (seq, true)])
        {
            let r = |shape: &[usize]| Tensor::randn(0f32, 0.5, shape, &vk);
            let (x, dt, b, c) = (
                r(&[1, seq, heads, head_dim])?,
                r(&[1, seq, heads])?,
                r(&[1, seq, groups, state])?,
                r(&[1, seq, groups, state])?,
            );
            let (a, d, dt_bias) = ((r(&[heads])?.abs()? * -1.)?, r(&[heads])?, r(&[heads])?);
            let pool = Tensor::zeros((1, heads, head_dim, state), DType::F32, &vk)?;
            let slots = Tensor::new(&[0u32], &vk)?;
            let run = || {
                super::scan_step(
                    &crate::vulkan::ScanStep {
                        x: &x,
                        dt: &dt,
                        a: &a,
                        b: &b,
                        c: &c,
                        d: &d,
                        dt_bias: &dt_bias,
                        dt_range: [0.0, f32::INFINITY],
                        pool: &pool,
                        slots: &slots,
                    },
                    chunked,
                )
            };
            // synchronized runs, so each reuses the buffers the last one freed rather than allocating its own
            run()?;
            vk.synchronize()?;
            let start = std::time::Instant::now();
            const RUNS: u32 = 20;
            for _ in 0..RUNS {
                run()?;
                vk.synchronize()?;
            }
            eprintln!(
                "seq {seq} chunked {chunked}: {:.3} ms a scan",
                start.elapsed().as_secs_f64() * 1e3 / f64::from(RUNS)
            );
        }
        Ok(())
    }
}
