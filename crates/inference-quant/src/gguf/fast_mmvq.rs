//! CUDA fast path for GGUF matmul with BF16/F32 activations.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use candle_core::cuda::cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut, SyncOnDrop};
use candle_core::{
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor, quantized::QTensor,
};

use super::kernel::{GgufType, KernelWeight};

use super::ffi;
use crate::{
    GluActivationType,
    utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream},
};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36; // 2 halves (4 bytes) + QK8_1 int8 = 4 + 32 = 36
const MATRIX_ROW_PADDING: usize = 512;

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

fn output_shape(xs: &Tensor, nrows: usize) -> Shape {
    let mut out_dims = xs.dims().to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = nrows;
    Shape::from(out_dims)
}

/// Quant types supported by `mmvq_gguf.cu`.
pub fn supports(dtype: impl Into<GgufType>) -> bool {
    let dtype = dtype.into();
    matches!(
        dtype,
        GgufType::Q4_0
            | GgufType::Q4_1
            | GgufType::Q5_0
            | GgufType::Q5_1
            | GgufType::Q8_0
            | GgufType::Q2K
            | GgufType::Q3K
            | GgufType::Q4K
            | GgufType::Q5K
            | GgufType::Q6K
            | GgufType::Iq4Nl
            | GgufType::Iq4Xs
            | GgufType::Iq2Xxs
            | GgufType::Iq2Xs
            | GgufType::Iq2S
            | GgufType::Iq3Xxs
            | GgufType::Iq3S
            | GgufType::Iq1S
            | GgufType::Iq1M
            | GgufType::Iq1Kt
            | GgufType::Iq2Kt
            | GgufType::Iq3Kt
            | GgufType::Iq4Kt
    )
}

/// Maximum flattened batch handled by the CUDA launcher table.
pub const MMVQ_MAX_BATCH: usize = 8;

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
}

struct WorkspaceGuard<'a> {
    slot: MutexGuard<'static, WorkspaceSlot>,
    stream: &'a CudaStream,
}

impl WorkspaceGuard<'_> {
    fn ptr_mut(&mut self) -> (u64, SyncOnDrop<'_>) {
        self.slot.slice.device_ptr_mut(self.stream)
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: candle_core::cuda::DeviceId,
    stream: usize,
    thread: ThreadId,
    capacity: usize,
}

type WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<WorkspaceSlot>>>;

static WORKSPACE: OnceLock<WsMap> = OnceLock::new();

fn workspace_ensure<'a>(
    dev: &CudaDevice,
    bytes: usize,
    stream: &'a CudaStream,
) -> Result<WorkspaceGuard<'a>> {
    let map = WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let capacity = bytes.max(1).next_power_of_two();
    let key = WorkspaceKey {
        device: dev.id(),
        stream: stream.cu_stream() as usize,
        thread: std::thread::current().id(),
        capacity,
    };
    let workspace_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(capacity)? };
                // CUDA graph replay requires process-stable workspace addresses.
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot { slice })));
                guard.insert(key, leaked);
                leaked
            }
        }
    };
    Ok(WorkspaceGuard {
        slot: workspace_mtx.lock().unwrap(),
        stream,
    })
}

// Launcher dispatch by weight and output dtype.

type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

type FusedGluLauncher = unsafe extern "C" fn(
    vx_gate: *const std::ffi::c_void,
    vx_up: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

type FusedQkvLauncher = unsafe extern "C" fn(
    vx_q: *const std::ffi::c_void,
    vx_k: *const std::ffi::c_void,
    vx_v: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    q_dst: *mut std::ffi::c_void,
    k_dst: *mut std::ffi::c_void,
    v_dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_q: i32,
    nrows_k: i32,
    nrows_v: i32,
    stride_col_y: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

fn plain_launcher_bf16(dtype: GgufType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgufType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_bf16_plain,
        GgufType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_bf16_plain,
        GgufType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_bf16_plain,
        GgufType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_bf16_plain,
        GgufType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_bf16_plain,
        GgufType::Q2K => ffi::launch_mmvq_gguf_q2_k_bf16_plain,
        GgufType::Q3K => ffi::launch_mmvq_gguf_q3_k_bf16_plain,
        GgufType::Q4K => ffi::launch_mmvq_gguf_q4_k_bf16_plain,
        GgufType::Q5K => ffi::launch_mmvq_gguf_q5_k_bf16_plain,
        GgufType::Q6K => ffi::launch_mmvq_gguf_q6_k_bf16_plain,
        GgufType::Iq4Nl => ffi::launch_mmvq_gguf_iq4_nl_bf16_plain,
        GgufType::Iq4Xs => ffi::launch_mmvq_gguf_iq4_xs_bf16_plain,
        GgufType::Iq2Xxs => ffi::launch_mmvq_gguf_iq2_xxs_bf16_plain,
        GgufType::Iq2Xs => ffi::launch_mmvq_gguf_iq2_xs_bf16_plain,
        GgufType::Iq2S => ffi::launch_mmvq_gguf_iq2_s_bf16_plain,
        GgufType::Iq3Xxs => ffi::launch_mmvq_gguf_iq3_xxs_bf16_plain,
        GgufType::Iq3S => ffi::launch_mmvq_gguf_iq3_s_bf16_plain,
        GgufType::Iq1S => ffi::launch_mmvq_gguf_iq1_s_bf16_plain,
        GgufType::Iq1M => ffi::launch_mmvq_gguf_iq1_m_bf16_plain,
        GgufType::Iq1Kt => ffi::launch_mmvq_gguf_iq1_kt_bf16_plain,
        GgufType::Iq2Kt => ffi::launch_mmvq_gguf_iq2_kt_bf16_plain,
        GgufType::Iq3Kt => ffi::launch_mmvq_gguf_iq3_kt_bf16_plain,
        GgufType::Iq4Kt => ffi::launch_mmvq_gguf_iq4_kt_bf16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f16(dtype: GgufType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgufType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f16_plain,
        GgufType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f16_plain,
        GgufType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f16_plain,
        GgufType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f16_plain,
        GgufType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f16_plain,
        GgufType::Q2K => ffi::launch_mmvq_gguf_q2_k_f16_plain,
        GgufType::Q3K => ffi::launch_mmvq_gguf_q3_k_f16_plain,
        GgufType::Q4K => ffi::launch_mmvq_gguf_q4_k_f16_plain,
        GgufType::Q5K => ffi::launch_mmvq_gguf_q5_k_f16_plain,
        GgufType::Q6K => ffi::launch_mmvq_gguf_q6_k_f16_plain,
        GgufType::Iq4Nl => ffi::launch_mmvq_gguf_iq4_nl_f16_plain,
        GgufType::Iq4Xs => ffi::launch_mmvq_gguf_iq4_xs_f16_plain,
        GgufType::Iq2Xxs => ffi::launch_mmvq_gguf_iq2_xxs_f16_plain,
        GgufType::Iq2Xs => ffi::launch_mmvq_gguf_iq2_xs_f16_plain,
        GgufType::Iq2S => ffi::launch_mmvq_gguf_iq2_s_f16_plain,
        GgufType::Iq3Xxs => ffi::launch_mmvq_gguf_iq3_xxs_f16_plain,
        GgufType::Iq3S => ffi::launch_mmvq_gguf_iq3_s_f16_plain,
        GgufType::Iq1S => ffi::launch_mmvq_gguf_iq1_s_f16_plain,
        GgufType::Iq1M => ffi::launch_mmvq_gguf_iq1_m_f16_plain,
        GgufType::Iq1Kt => ffi::launch_mmvq_gguf_iq1_kt_f16_plain,
        GgufType::Iq2Kt => ffi::launch_mmvq_gguf_iq2_kt_f16_plain,
        GgufType::Iq3Kt => ffi::launch_mmvq_gguf_iq3_kt_f16_plain,
        GgufType::Iq4Kt => ffi::launch_mmvq_gguf_iq4_kt_f16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f32(dtype: GgufType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgufType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f32_plain,
        GgufType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f32_plain,
        GgufType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f32_plain,
        GgufType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f32_plain,
        GgufType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f32_plain,
        GgufType::Q2K => ffi::launch_mmvq_gguf_q2_k_f32_plain,
        GgufType::Q3K => ffi::launch_mmvq_gguf_q3_k_f32_plain,
        GgufType::Q4K => ffi::launch_mmvq_gguf_q4_k_f32_plain,
        GgufType::Q5K => ffi::launch_mmvq_gguf_q5_k_f32_plain,
        GgufType::Q6K => ffi::launch_mmvq_gguf_q6_k_f32_plain,
        GgufType::Iq4Nl => ffi::launch_mmvq_gguf_iq4_nl_f32_plain,
        GgufType::Iq4Xs => ffi::launch_mmvq_gguf_iq4_xs_f32_plain,
        GgufType::Iq2Xxs => ffi::launch_mmvq_gguf_iq2_xxs_f32_plain,
        GgufType::Iq2Xs => ffi::launch_mmvq_gguf_iq2_xs_f32_plain,
        GgufType::Iq2S => ffi::launch_mmvq_gguf_iq2_s_f32_plain,
        GgufType::Iq3Xxs => ffi::launch_mmvq_gguf_iq3_xxs_f32_plain,
        GgufType::Iq3S => ffi::launch_mmvq_gguf_iq3_s_f32_plain,
        GgufType::Iq1S => ffi::launch_mmvq_gguf_iq1_s_f32_plain,
        GgufType::Iq1M => ffi::launch_mmvq_gguf_iq1_m_f32_plain,
        GgufType::Iq1Kt => ffi::launch_mmvq_gguf_iq1_kt_f32_plain,
        GgufType::Iq2Kt => ffi::launch_mmvq_gguf_iq2_kt_f32_plain,
        GgufType::Iq3Kt => ffi::launch_mmvq_gguf_iq3_kt_f32_plain,
        GgufType::Iq4Kt => ffi::launch_mmvq_gguf_iq4_kt_f32_plain,
        _ => return None,
    };
    Some(f)
}

fn fused_glu_launcher(input_ty: DType, dtype: GgufType) -> Option<FusedGluLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgufType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_bf16_fused_glu),
        (DType::BF16, GgufType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_bf16_fused_glu),
        (DType::BF16, GgufType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_bf16_fused_glu),
        (DType::BF16, GgufType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_bf16_fused_glu),
        (DType::BF16, GgufType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_bf16_fused_glu),
        (DType::BF16, GgufType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_bf16_fused_glu),
        (DType::BF16, GgufType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_bf16_fused_glu),
        (DType::BF16, GgufType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_bf16_fused_glu),
        (DType::BF16, GgufType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_bf16_fused_glu),
        (DType::BF16, GgufType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_bf16_fused_glu),

        (DType::F16, GgufType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f16_fused_glu),
        (DType::F16, GgufType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f16_fused_glu),
        (DType::F16, GgufType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f16_fused_glu),
        (DType::F16, GgufType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f16_fused_glu),
        (DType::F16, GgufType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f16_fused_glu),
        (DType::F16, GgufType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f16_fused_glu),
        (DType::F16, GgufType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f16_fused_glu),
        (DType::F16, GgufType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f16_fused_glu),
        (DType::F16, GgufType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f16_fused_glu),
        (DType::F16, GgufType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f16_fused_glu),

        (DType::F32, GgufType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f32_fused_glu),
        (DType::F32, GgufType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f32_fused_glu),
        (DType::F32, GgufType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f32_fused_glu),
        (DType::F32, GgufType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f32_fused_glu),
        (DType::F32, GgufType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f32_fused_glu),
        (DType::F32, GgufType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f32_fused_glu),
        (DType::F32, GgufType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f32_fused_glu),
        (DType::F32, GgufType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f32_fused_glu),
        (DType::F32, GgufType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f32_fused_glu),
        (DType::F32, GgufType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f32_fused_glu),
        _ => None,
    }
}

pub fn supports_fused_glu(input_ty: DType, dtype: impl Into<GgufType>) -> bool {
    let dtype = dtype.into();
    fused_glu_launcher(input_ty, dtype).is_some()
}

fn fused_qkv_launcher(input_ty: DType, dtype: GgufType) -> Option<FusedQkvLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgufType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_bf16_fused_qkv),
        (DType::BF16, GgufType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_bf16_fused_qkv),
        (DType::BF16, GgufType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_bf16_fused_qkv),
        (DType::BF16, GgufType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_bf16_fused_qkv),
        (DType::BF16, GgufType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_bf16_fused_qkv),
        (DType::BF16, GgufType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_bf16_fused_qkv),
        (DType::BF16, GgufType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_bf16_fused_qkv),
        (DType::BF16, GgufType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_bf16_fused_qkv),
        (DType::BF16, GgufType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_bf16_fused_qkv),
        (DType::BF16, GgufType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_bf16_fused_qkv),

        (DType::F16, GgufType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f16_fused_qkv),
        (DType::F16, GgufType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f16_fused_qkv),
        (DType::F16, GgufType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f16_fused_qkv),
        (DType::F16, GgufType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f16_fused_qkv),
        (DType::F16, GgufType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f16_fused_qkv),
        (DType::F16, GgufType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f16_fused_qkv),
        (DType::F16, GgufType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f16_fused_qkv),
        (DType::F16, GgufType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f16_fused_qkv),
        (DType::F16, GgufType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f16_fused_qkv),
        (DType::F16, GgufType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f16_fused_qkv),

        (DType::F32, GgufType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f32_fused_qkv),
        (DType::F32, GgufType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f32_fused_qkv),
        (DType::F32, GgufType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f32_fused_qkv),
        (DType::F32, GgufType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f32_fused_qkv),
        (DType::F32, GgufType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f32_fused_qkv),
        (DType::F32, GgufType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f32_fused_qkv),
        (DType::F32, GgufType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f32_fused_qkv),
        (DType::F32, GgufType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f32_fused_qkv),
        (DType::F32, GgufType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f32_fused_qkv),
        (DType::F32, GgufType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f32_fused_qkv),
        _ => None,
    }
}

/// Compute `w @ xs^T` where `w` is a Q8_1-quantizable GGUF weight tensor and
/// `xs` is a contiguous BF16 / F16 / F32 activation on the same CUDA device.
///
/// The product of the leading input dimensions must be in `1..=8`.
///
/// Output has the same leading dimensions as `xs` with the last axis replaced
/// by `w.shape().dims2()?.0` (nrows of the weight).
///
/// The output dtype matches the input dtype (BF16 → BF16, F16 → F16, F32 → F32).
pub fn plain<W: KernelWeight + ?Sized>(w: &W, xs: &Tensor) -> Result<Tensor> {
    let dtype = w.gguf_type();
    if !supports(dtype) {
        candle_core::bail!("fast_mmvq: unsupported quant dtype {dtype:?}");
    }
    let Device::Cuda(dev) = w.kernel_device() else {
        candle_core::bail!("fast_mmvq: weight must live on CUDA");
    };
    if !xs.device().same_device(&w.kernel_device()) {
        candle_core::bail!("fast_mmvq: input and weight are on different devices");
    }
    let (nrows, ncols) = w.kernel_shape().dims2()?;

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("fast_mmvq: input dtype must be BF16, F16, or F32, got {input_ty:?}");
    }

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let (weight_ptr, _weight_guard) = w.kernel_ptr(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = plain_launcher_bf16(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = plain_launcher_f16(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = plain_launcher_f32(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        _ => unreachable!(),
    }
}

pub fn fused_glu(
    gate_w: &QTensor,
    up_w: &QTensor,
    xs: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    let dtype = gate_w.gguf_type();
    if dtype != up_w.gguf_type() {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up dtype mismatch {:?} vs {:?}",
            dtype,
            up_w.gguf_type()
        );
    }
    let Some(launcher) = fused_glu_launcher(xs.dtype(), dtype) else {
        candle_core::bail!("fast_mmvq fused_glu: unsupported dtype combination");
    };

    let Device::Cuda(dev) = gate_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: gate weight must live on CUDA");
    };
    let Device::Cuda(up_dev) = up_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: up weight must live on CUDA");
    };
    if dev.id() != up_dev.id() {
        candle_core::bail!("fast_mmvq fused_glu: gate/up weights are on different CUDA devices");
    }
    if !xs.device().same_device(&gate_w.device()) {
        candle_core::bail!("fast_mmvq fused_glu: input and weights are on different devices");
    }

    let (nrows, ncols) = gate_w.shape().dims2()?;
    let (up_nrows, up_ncols) = up_w.shape().dims2()?;
    if (nrows, ncols) != (up_nrows, up_ncols) {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up shape mismatch [{nrows}, {ncols}] vs [{up_nrows}, {up_ncols}]"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq fused_glu: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq fused_glu: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq fused_glu: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmvq fused_glu: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_glu: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let (gate_ptr, _gate_guard) = gate_w.device_ptr_with_guard(&stream)?;
    let (up_ptr, _up_guard) = up_w.device_ptr_with_guard(&stream)?;
    let gate_ptr = gate_ptr as *const std::ffi::c_void;
    let up_ptr = up_ptr as *const std::ffi::c_void;
    let activation = activation as i32;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        _ => unreachable!(),
    }
}

/// Compute Q, K, and V matvecs with one input quantization pass and one MMVQ
/// kernel. The result tensors match the unfused `plain` outputs for each
/// projection and preserve the input dtype.
pub fn fused_qkv(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: &QTensor,
    xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dtype = q_w.gguf_type();
    if dtype != k_w.gguf_type() || dtype != v_w.gguf_type() {
        candle_core::bail!(
            "fast_mmvq fused_qkv: q/k/v dtype mismatch {:?}, {:?}, {:?}",
            dtype,
            k_w.gguf_type(),
            v_w.gguf_type()
        );
    }
    let Some(launcher) = fused_qkv_launcher(xs.dtype(), dtype) else {
        candle_core::bail!("fast_mmvq fused_qkv: unsupported dtype combination");
    };

    let Device::Cuda(dev) = q_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: q weight must live on CUDA");
    };
    let Device::Cuda(k_dev) = k_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: k weight must live on CUDA");
    };
    let Device::Cuda(v_dev) = v_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: v weight must live on CUDA");
    };
    if dev.id() != k_dev.id() || dev.id() != v_dev.id() {
        candle_core::bail!("fast_mmvq fused_qkv: q/k/v weights are on different CUDA devices");
    }
    if !xs.device().same_device(&q_w.device()) {
        candle_core::bail!("fast_mmvq fused_qkv: input and weights are on different devices");
    }

    let (q_nrows, ncols) = q_w.shape().dims2()?;
    let (k_nrows, k_ncols) = k_w.shape().dims2()?;
    let (v_nrows, v_ncols) = v_w.shape().dims2()?;
    if ncols != k_ncols || ncols != v_ncols {
        candle_core::bail!(
            "fast_mmvq fused_qkv: q/k/v ncols mismatch {ncols}, {k_ncols}, {v_ncols}"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq fused_qkv: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq fused_qkv: shape mismatch: weight ncols {ncols} vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq fused_qkv: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmvq fused_qkv: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_qkv: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let (q_ptr, _q_guard) = q_w.device_ptr_with_guard(&stream)?;
    let (k_ptr, _k_guard) = k_w.device_ptr_with_guard(&stream)?;
    let (v_ptr, _v_guard) = v_w.device_ptr_with_guard(&stream)?;
    let q_ptr = q_ptr as *const std::ffi::c_void;
    let k_ptr = k_ptr as *const std::ffi::c_void;
    let v_ptr = v_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut q_out = unsafe { dev.alloc::<half::bf16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::bf16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::bf16>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut q_out = unsafe { dev.alloc::<half::f16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::f16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::f16>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut q_out = unsafe { dev.alloc::<f32>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<f32>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<f32>(v_nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        _ => unreachable!(),
    }
}

type DequantizeLauncher = unsafe extern "C" fn(
    *const std::ffi::c_void,
    *mut std::ffi::c_void,
    i64,
    i64,
    *mut std::ffi::c_void,
);

fn dequantize_launcher(ty: GgufType, dtype: DType) -> Option<DequantizeLauncher> {
    Some(match (ty, dtype) {
        (GgufType::Iq1M, DType::BF16) => ffi::launch_dequantize_iq1_m_bf16,
        (GgufType::Iq1M, DType::F16) => ffi::launch_dequantize_iq1_m_f16,
        (GgufType::Iq1M, DType::F32) => ffi::launch_dequantize_iq1_m_f32,
        (GgufType::Iq1Kt, DType::BF16) => ffi::launch_dequantize_iq1_kt_bf16,
        (GgufType::Iq1Kt, DType::F16) => ffi::launch_dequantize_iq1_kt_f16,
        (GgufType::Iq1Kt, DType::F32) => ffi::launch_dequantize_iq1_kt_f32,
        (GgufType::Iq2Kt, DType::BF16) => ffi::launch_dequantize_iq2_kt_bf16,
        (GgufType::Iq2Kt, DType::F16) => ffi::launch_dequantize_iq2_kt_f16,
        (GgufType::Iq2Kt, DType::F32) => ffi::launch_dequantize_iq2_kt_f32,
        (GgufType::Iq3Kt, DType::BF16) => ffi::launch_dequantize_iq3_kt_bf16,
        (GgufType::Iq3Kt, DType::F16) => ffi::launch_dequantize_iq3_kt_f16,
        (GgufType::Iq3Kt, DType::F32) => ffi::launch_dequantize_iq3_kt_f32,
        (GgufType::Iq4Kt, DType::BF16) => ffi::launch_dequantize_iq4_kt_bf16,
        (GgufType::Iq4Kt, DType::F16) => ffi::launch_dequantize_iq4_kt_f16,
        (GgufType::Iq4Kt, DType::F32) => ffi::launch_dequantize_iq4_kt_f32,
        _ => return None,
    })
}

/// The weight dequantized on its GPU, as ggml does for prefill on types without an mmq tile.
pub fn dequantize<W: KernelWeight + ?Sized>(w: &W, dtype: DType) -> Result<Tensor> {
    let ty = w.gguf_type();
    let Some(launcher) = dequantize_launcher(ty, dtype) else {
        candle_core::bail!("no CUDA dequantizer for {ty:?} to {dtype:?}");
    };
    let Device::Cuda(dev) = w.kernel_device() else {
        candle_core::bail!("fast_mmvq: weight must live on CUDA");
    };
    let shape = w.kernel_shape().clone();
    let elems = shape.elem_count();
    let (nrows, ncols) = shape.dims2()?;
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let (weight_ptr, _weight_guard) = w.kernel_ptr(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;
    macro_rules! run {
        ($t:ty) => {{
            let mut out = unsafe { dev.alloc::<$t>(elems)? };
            {
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
                unsafe {
                    launcher(
                        weight_ptr,
                        out_ptr as *mut std::ffi::c_void,
                        nrows as i64,
                        ncols as i64,
                        stream_ptr,
                    )
                };
            }
            CudaStorage::wrap_cuda_slice(out, dev.clone())
        }};
    }
    let storage = match dtype {
        DType::BF16 => run!(half::bf16),
        DType::F16 => run!(half::f16),
        _ => run!(f32),
    };
    Ok(Tensor::from((Storage::Cuda(storage), shape)))
}
