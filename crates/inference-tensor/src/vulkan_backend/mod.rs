//! The Vulkan backend: storage in host-visible memory the iGPU and the CPU share, on `inference_vulkan`'s stream. An op
//! with no Vulkan kernel yet waits for the GPU, runs the CPU backend's on a copy and uploads the result, so a model
//! runs whole while its kernels are ported one at a time.

use std::sync::{Arc, Mutex, OnceLock};

use inference_vulkan::{Context, HostBuffer, MAX_DIMS, Map, MapOp, Operand};

use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, DeviceLocation, Layout, Result, Shape, WithDType};

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

    fn native(&self, l: &Layout) -> bool {
        self.dtype == DType::F32 && l.shape().rank() <= MAX_DIMS
    }
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

    fn assert_same(got: &Tensor, want: &Tensor) -> Result<()> {
        assert_eq!(got.dims(), want.dims());
        assert!(got.device().is_vulkan());
        let (got, want) = (
            got.flatten_all()?.to_vec1::<f32>()?,
            want.flatten_all()?.to_vec1::<f32>()?,
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= TOLERANCE * w.abs().max(1.0),
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
