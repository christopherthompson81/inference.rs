//! CUDA fast path for GGUF matmul with BF16/F32 activations.

use inference_tensor::cuda::cudarc::driver::CudaStream;
use inference_tensor::{
    CudaStorage, DType, Device, Result, Shape, Storage, Tensor, quantized::QTensor,
};

use super::kernel::{GgufType, KernelWeight};
use super::thread_workspace::ThreadWorkspaces;

use super::ffi;
use crate::{
    GluActivationType,
    utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream},
};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36; // 2 halves (4 bytes) + QK8_1 int8 = 4 + 32 = 36
const MATRIX_ROW_PADDING: usize = 512;
const QK_K: usize = 256;
// The second Q8_1 region of the MoE workspace starts on this boundary
const WORKSPACE_ALIGN: usize = 256;
// Down reads one activation row per routed pair, and fuses no activation
const MOE_DOWN_Y_DIV: i32 = 1;
const MOE_NO_ACTIVATION: i32 = 0;

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

/// Quant types with matvec kernels (`mmvq_instance_<type>.cu`).
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
            | GgufType::Iq2K
            | GgufType::Iq3K
            | GgufType::Iq4K
            | GgufType::Iq5K
            | GgufType::Iq6K
            | GgufType::Iq4Ks
            | GgufType::Iq4KsTiled
            | GgufType::Iq2Ks
            | GgufType::Iq4Kss
            | GgufType::Iq5Ks
            | GgufType::Iq3Ks
            | GgufType::Iq2Kl
    )
}

/// Maximum flattened batch handled by the CUDA launcher table.
pub const MMVQ_MAX_BATCH: usize = 8;

// sm_80's int8 mma.m16n8k32, which the tensor-core matvec (mmvq_tc.cuh) needs
const TENSOR_CORE_MIN_CC: usize = 80;
// mmvq_tc_quantize's bytes per 32-value block: 32 lanes' int2 fragments and 8 rows' f32 scales
const TENSOR_CORE_ACTIVATION_BLOCK_BYTES: usize = 32 * 8 + 8 * 4;

// The most input rows MMVQ's dp4a kernels take before MMQ's int8 tensor-core tiles, which cost about the same from 1
// to 8 rows, are faster (`fast_mmq::bench::small_batch_mmvq_vs_mmq`, RTX 3090).
fn dp4a_max_rows(dtype: GgufType) -> usize {
    match dtype {
        GgufType::Iq4Xs => 5,
        GgufType::Q4K | GgufType::Q5K => 3,
        // the crossover moves with the output width (4 for a 248K-row head, 6 for a 2K-row FFN)
        GgufType::Q6K => 4,
        GgufType::Q8_0 => 7,
        // the ik types' MMVQ rows cost far more than the k-quants' (IQ4_KS 5.8x at 8 rows)
        GgufType::Iq4Ks | GgufType::Iq4Kss => 2,
        GgufType::Iq4Kt | GgufType::Iq3K | GgufType::Iq3Kt => 1,
        _ => MMVQ_MAX_BATCH,
    }
}

/// Whether `rows` input rows of a `dtype` weight take MMVQ's dp4a kernels, the fused GLU and QKV forms included.
pub fn prefers_fused_mmvq(dtype: impl Into<GgufType>, rows: usize) -> bool {
    let dtype = dtype.into();
    rows <= dp4a_max_rows(dtype) || rows <= MMVQ_MAX_BATCH && !super::fast_mmq::supports(dtype)
}

/// Whether `rows` input rows of a `dtype` weight take MMVQ (`plain`) rather than MMQ: its dp4a kernels, or for 2-8
/// rows of a type with one, the tensor-core matvec, which reads and decodes each weight block once for all rows.
pub fn prefers_mmvq(dtype: impl Into<GgufType>, rows: usize) -> bool {
    let dtype = dtype.into();
    prefers_fused_mmvq(dtype, rows) || rows <= MMVQ_MAX_BATCH && has_tensor_core_rows(dtype)
}

/// Whether a `dtype` weight has the tensor-core matvec for 1-8 input rows (a tile-ordered layout).
pub fn has_tensor_core_rows(dtype: impl Into<GgufType>) -> bool {
    dtype.into() == GgufType::Iq4KsTiled
}

static WORKSPACE: ThreadWorkspaces = ThreadWorkspaces::new();

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
        GgufType::Iq2K => ffi::launch_mmvq_gguf_iq2_k_bf16_plain,
        GgufType::Iq3K => ffi::launch_mmvq_gguf_iq3_k_bf16_plain,
        GgufType::Iq4K => ffi::launch_mmvq_gguf_iq4_k_bf16_plain,
        GgufType::Iq5K => ffi::launch_mmvq_gguf_iq5_k_bf16_plain,
        GgufType::Iq6K => ffi::launch_mmvq_gguf_iq6_k_bf16_plain,
        GgufType::Iq4Ks => ffi::launch_mmvq_gguf_iq4_ks_bf16_plain,
        GgufType::Iq2Ks => ffi::launch_mmvq_gguf_iq2_ks_bf16_plain,
        GgufType::Iq4Kss => ffi::launch_mmvq_gguf_iq4_kss_bf16_plain,
        GgufType::Iq5Ks => ffi::launch_mmvq_gguf_iq5_ks_bf16_plain,
        GgufType::Iq3Ks => ffi::launch_mmvq_gguf_iq3_ks_bf16_plain,
        GgufType::Iq2Kl => ffi::launch_mmvq_gguf_iq2_kl_bf16_plain,
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
        GgufType::Iq2K => ffi::launch_mmvq_gguf_iq2_k_f16_plain,
        GgufType::Iq3K => ffi::launch_mmvq_gguf_iq3_k_f16_plain,
        GgufType::Iq4K => ffi::launch_mmvq_gguf_iq4_k_f16_plain,
        GgufType::Iq5K => ffi::launch_mmvq_gguf_iq5_k_f16_plain,
        GgufType::Iq6K => ffi::launch_mmvq_gguf_iq6_k_f16_plain,
        GgufType::Iq4Ks => ffi::launch_mmvq_gguf_iq4_ks_f16_plain,
        GgufType::Iq2Ks => ffi::launch_mmvq_gguf_iq2_ks_f16_plain,
        GgufType::Iq4Kss => ffi::launch_mmvq_gguf_iq4_kss_f16_plain,
        GgufType::Iq5Ks => ffi::launch_mmvq_gguf_iq5_ks_f16_plain,
        GgufType::Iq3Ks => ffi::launch_mmvq_gguf_iq3_ks_f16_plain,
        GgufType::Iq2Kl => ffi::launch_mmvq_gguf_iq2_kl_f16_plain,
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
        GgufType::Iq2K => ffi::launch_mmvq_gguf_iq2_k_f32_plain,
        GgufType::Iq3K => ffi::launch_mmvq_gguf_iq3_k_f32_plain,
        GgufType::Iq4K => ffi::launch_mmvq_gguf_iq4_k_f32_plain,
        GgufType::Iq5K => ffi::launch_mmvq_gguf_iq5_k_f32_plain,
        GgufType::Iq6K => ffi::launch_mmvq_gguf_iq6_k_f32_plain,
        GgufType::Iq4Ks => ffi::launch_mmvq_gguf_iq4_ks_f32_plain,
        GgufType::Iq2Ks => ffi::launch_mmvq_gguf_iq2_ks_f32_plain,
        GgufType::Iq4Kss => ffi::launch_mmvq_gguf_iq4_kss_f32_plain,
        GgufType::Iq5Ks => ffi::launch_mmvq_gguf_iq5_ks_f32_plain,
        GgufType::Iq3Ks => ffi::launch_mmvq_gguf_iq3_ks_f32_plain,
        GgufType::Iq2Kl => ffi::launch_mmvq_gguf_iq2_kl_f32_plain,
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
/// The output dtype matches the input dtype (BF16 -> BF16, F16 -> F16, F32 -> F32).
pub fn plain<W: KernelWeight + ?Sized>(w: &W, xs: &Tensor) -> Result<Tensor> {
    let dtype = w.gguf_type();
    if !supports(dtype) {
        inference_tensor::bail!("fast_mmvq: unsupported quant dtype {dtype:?}");
    }
    let Device::Cuda(dev) = w.kernel_device() else {
        inference_tensor::bail!("fast_mmvq: weight must live on CUDA");
    };
    if !xs.device().same_device(&w.kernel_device()) {
        inference_tensor::bail!("fast_mmvq: input and weight are on different devices");
    }
    let (nrows, ncols) = w.kernel_shape().dims2()?;

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        inference_tensor::bail!("fast_mmvq: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        inference_tensor::bail!(
            "fast_mmvq: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        inference_tensor::bail!(
            "fast_mmvq: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        inference_tensor::bail!(
            "fast_mmvq: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    if dtype == GgufType::Iq4KsTiled {
        let cfg = tiled_config(dev.cuda_stream().context().ordinal(), nrows, ncols);
        return tiled_plain(w, xs, cfg);
    }
    if !prefers_fused_mmvq(dtype, b_size) {
        return super::fast_mmq::plain(w, xs);
    }

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        inference_tensor::bail!("fast_mmvq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = WORKSPACE.ensure(&dev, scratch_bytes, &stream)?;
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
        inference_tensor::bail!(
            "fast_mmvq fused_glu: gate/up dtype mismatch {:?} vs {:?}",
            dtype,
            up_w.gguf_type()
        );
    }
    let Some(launcher) = fused_glu_launcher(xs.dtype(), dtype) else {
        inference_tensor::bail!("fast_mmvq fused_glu: unsupported dtype combination");
    };

    let Device::Cuda(dev) = gate_w.device() else {
        inference_tensor::bail!("fast_mmvq fused_glu: gate weight must live on CUDA");
    };
    let Device::Cuda(up_dev) = up_w.device() else {
        inference_tensor::bail!("fast_mmvq fused_glu: up weight must live on CUDA");
    };
    if dev.id() != up_dev.id() {
        inference_tensor::bail!(
            "fast_mmvq fused_glu: gate/up weights are on different CUDA devices"
        );
    }
    if !xs.device().same_device(&gate_w.device()) {
        inference_tensor::bail!("fast_mmvq fused_glu: input and weights are on different devices");
    }

    let (nrows, ncols) = gate_w.shape().dims2()?;
    let (up_nrows, up_ncols) = up_w.shape().dims2()?;
    if (nrows, ncols) != (up_nrows, up_ncols) {
        inference_tensor::bail!(
            "fast_mmvq fused_glu: gate/up shape mismatch [{nrows}, {ncols}] vs [{up_nrows}, {up_ncols}]"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        inference_tensor::bail!("fast_mmvq fused_glu: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        inference_tensor::bail!(
            "fast_mmvq fused_glu: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        inference_tensor::bail!(
            "fast_mmvq fused_glu: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        inference_tensor::bail!(
            "fast_mmvq fused_glu: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        inference_tensor::bail!("fast_mmvq fused_glu: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = WORKSPACE.ensure(&dev, scratch_bytes, &stream)?;
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
        inference_tensor::bail!(
            "fast_mmvq fused_qkv: q/k/v dtype mismatch {:?}, {:?}, {:?}",
            dtype,
            k_w.gguf_type(),
            v_w.gguf_type()
        );
    }
    let Some(launcher) = fused_qkv_launcher(xs.dtype(), dtype) else {
        inference_tensor::bail!("fast_mmvq fused_qkv: unsupported dtype combination");
    };

    let Device::Cuda(dev) = q_w.device() else {
        inference_tensor::bail!("fast_mmvq fused_qkv: q weight must live on CUDA");
    };
    let Device::Cuda(k_dev) = k_w.device() else {
        inference_tensor::bail!("fast_mmvq fused_qkv: k weight must live on CUDA");
    };
    let Device::Cuda(v_dev) = v_w.device() else {
        inference_tensor::bail!("fast_mmvq fused_qkv: v weight must live on CUDA");
    };
    if dev.id() != k_dev.id() || dev.id() != v_dev.id() {
        inference_tensor::bail!("fast_mmvq fused_qkv: q/k/v weights are on different CUDA devices");
    }
    if !xs.device().same_device(&q_w.device()) {
        inference_tensor::bail!("fast_mmvq fused_qkv: input and weights are on different devices");
    }

    let (q_nrows, ncols) = q_w.shape().dims2()?;
    let (k_nrows, k_ncols) = k_w.shape().dims2()?;
    let (v_nrows, v_ncols) = v_w.shape().dims2()?;
    if ncols != k_ncols || ncols != v_ncols {
        inference_tensor::bail!(
            "fast_mmvq fused_qkv: q/k/v ncols mismatch {ncols}, {k_ncols}, {v_ncols}"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        inference_tensor::bail!("fast_mmvq fused_qkv: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        inference_tensor::bail!(
            "fast_mmvq fused_qkv: shape mismatch: weight ncols {ncols} vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        inference_tensor::bail!(
            "fast_mmvq fused_qkv: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        inference_tensor::bail!(
            "fast_mmvq fused_qkv: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        inference_tensor::bail!("fast_mmvq fused_qkv: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = WORKSPACE.ensure(&dev, scratch_bytes, &stream)?;
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

type MoeLauncher = unsafe extern "C" fn(
    vx_gate: *const std::ffi::c_void,
    vx_up: *const std::ffi::c_void,
    expert_stride: i64,
    vy: *const std::ffi::c_void,
    ids: *const std::ffi::c_void,
    dst: *mut f32,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    y_div: i32,
    n_pairs: i32,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

fn moe_launcher(dtype: GgufType) -> Option<MoeLauncher> {
    let f: MoeLauncher = match dtype {
        GgufType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_moe,
        GgufType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_moe,
        GgufType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_moe,
        GgufType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_moe,
        GgufType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_moe,
        GgufType::Q2K => ffi::launch_mmvq_gguf_q2_k_moe,
        GgufType::Q3K => ffi::launch_mmvq_gguf_q3_k_moe,
        GgufType::Q4K => ffi::launch_mmvq_gguf_q4_k_moe,
        GgufType::Q5K => ffi::launch_mmvq_gguf_q5_k_moe,
        GgufType::Q6K => ffi::launch_mmvq_gguf_q6_k_moe,
        GgufType::Iq4Nl => ffi::launch_mmvq_gguf_iq4_nl_moe,
        GgufType::Iq4Xs => ffi::launch_mmvq_gguf_iq4_xs_moe,
        GgufType::Iq2Xxs => ffi::launch_mmvq_gguf_iq2_xxs_moe,
        GgufType::Iq2Xs => ffi::launch_mmvq_gguf_iq2_xs_moe,
        GgufType::Iq2S => ffi::launch_mmvq_gguf_iq2_s_moe,
        GgufType::Iq3Xxs => ffi::launch_mmvq_gguf_iq3_xxs_moe,
        GgufType::Iq3S => ffi::launch_mmvq_gguf_iq3_s_moe,
        GgufType::Iq1S => ffi::launch_mmvq_gguf_iq1_s_moe,
        GgufType::Iq1M => ffi::launch_mmvq_gguf_iq1_m_moe,
        GgufType::Iq1Kt => ffi::launch_mmvq_gguf_iq1_kt_moe,
        GgufType::Iq2Kt => ffi::launch_mmvq_gguf_iq2_kt_moe,
        GgufType::Iq3Kt => ffi::launch_mmvq_gguf_iq3_kt_moe,
        GgufType::Iq4Kt => ffi::launch_mmvq_gguf_iq4_kt_moe,
        GgufType::Iq2K => ffi::launch_mmvq_gguf_iq2_k_moe,
        GgufType::Iq3K => ffi::launch_mmvq_gguf_iq3_k_moe,
        GgufType::Iq4K => ffi::launch_mmvq_gguf_iq4_k_moe,
        GgufType::Iq5K => ffi::launch_mmvq_gguf_iq5_k_moe,
        GgufType::Iq6K => ffi::launch_mmvq_gguf_iq6_k_moe,
        GgufType::Iq4Ks => ffi::launch_mmvq_gguf_iq4_ks_moe,
        GgufType::Iq2Ks => ffi::launch_mmvq_gguf_iq2_ks_moe,
        GgufType::Iq4Kss => ffi::launch_mmvq_gguf_iq4_kss_moe,
        GgufType::Iq5Ks => ffi::launch_mmvq_gguf_iq5_ks_moe,
        GgufType::Iq3Ks => ffi::launch_mmvq_gguf_iq3_ks_moe,
        GgufType::Iq2Kl => ffi::launch_mmvq_gguf_iq2_kl_moe,
        _ => return None,
    };
    Some(f)
}

/// Whether `moe_decode` reads these stacked experts: one matvec kernel per type, gate and up of one type.
pub fn supports_moe(
    gate: &dyn KernelWeight,
    up: &dyn KernelWeight,
    down: &dyn KernelWeight,
) -> bool {
    gate.gguf_type() == up.gguf_type()
        && [gate, up, down]
            .into_iter()
            .all(|w| moe_launcher(w.gguf_type()).is_some() && w.kernel_shape().rank() == 3)
}

// Quantizes `rows` rows of `xs` (row length k) to Q8_1 rows of `k_padded`, as the matvec kernels read them
fn quantize_rows_q8_1(
    xs: &Tensor,
    dst: *mut std::ffi::c_void,
    k_padded: usize,
    stream: &CudaStream,
) -> Result<()> {
    let xs = xs.contiguous()?;
    let (rows, k) = xs.dims2()?;
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Cuda(cuda) = &*storage else {
        inference_tensor::bail!("fast_mmvq: input must live on CUDA");
    };
    let offset = layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let args = (k as i32, k_padded as i32, rows as i32, stream_ptr);
    match xs.dtype() {
        DType::BF16 => {
            let (ptr, _guard) =
                slice_ptr_on_stream(cuda.as_cuda_slice::<half::bf16>()?, offset, stream);
            unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                    ptr as *const _,
                    dst,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                )
            };
        }
        DType::F16 => {
            let (ptr, _guard) =
                slice_ptr_on_stream(cuda.as_cuda_slice::<half::f16>()?, offset, stream);
            unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                    ptr as *const _,
                    dst,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                )
            };
        }
        DType::F32 => {
            let (ptr, _guard) = slice_ptr_on_stream(cuda.as_cuda_slice::<f32>()?, offset, stream);
            unsafe {
                ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                    ptr as *const _,
                    dst,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                )
            };
        }
        other => inference_tensor::bail!(
            "fast_mmvq: input dtype must be BF16, F16 or F32, got {other:?}"
        ),
    }
    Ok(())
}

/// Matvec MoE over stacked `[E, out, in]` experts, `topk` ids per row of `xs`: `[tokens * topk, hidden]` F32 outputs.
pub fn moe_decode(
    gate: &dyn KernelWeight,
    up: &dyn KernelWeight,
    down: &dyn KernelWeight,
    xs: &Tensor,
    ids: &Tensor,
    topk: usize,
    activation: GluActivationType,
) -> Result<Tensor> {
    if !supports_moe(gate, up, down) {
        inference_tensor::bail!(
            "fast_mmvq: no expert-indexed matvec for {:?}/{:?}/{:?}",
            gate.gguf_type(),
            up.gguf_type(),
            down.gguf_type()
        );
    }
    let Device::Cuda(dev) = gate.kernel_device() else {
        inference_tensor::bail!("fast_mmvq: experts must live on CUDA");
    };
    if !xs.device().same_device(&gate.kernel_device())
        || !ids.device().same_device(&gate.kernel_device())
    {
        inference_tensor::bail!(
            "fast_mmvq: MoE input, expert ids and experts are on different devices"
        );
    }
    let (_, intermediate, hidden) = gate.kernel_shape().dims3()?;
    let (tokens, k) = xs.dims2()?;
    if up.kernel_shape().dims3()? != gate.kernel_shape().dims3()?
        || down.kernel_shape().dims3()?.1 != hidden
        || down.kernel_shape().dims3()?.2 != intermediate
        || k != hidden
    {
        inference_tensor::bail!(
            "fast_mmvq: expert shapes gate {:?} up {:?} down {:?} do not fit input {:?}",
            gate.kernel_shape(),
            up.kernel_shape(),
            down.kernel_shape(),
            xs.shape()
        );
    }
    let pairs = tokens * topk;
    if ids.elem_count() != pairs || ids.dtype() != DType::U32 {
        inference_tensor::bail!(
            "fast_mmvq: expected {pairs} u32 expert ids, got {:?} {:?}",
            ids.shape(),
            ids.dtype()
        );
    }
    let expert_bytes = |w: &dyn KernelWeight, rows: usize, cols: usize| -> Result<i64> {
        let row = w.gguf_type().row_bytes(cols).ok_or_else(|| {
            inference_tensor::Error::Msg(format!("{:?} rows cannot hold {cols}", w.gguf_type()))
        })?;
        Ok((rows * row) as i64)
    };
    let gate_stride = expert_bytes(gate, intermediate, hidden)?;
    let down_stride = expert_bytes(down, hidden, intermediate)?;

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let x_padded = pad(hidden, MATRIX_ROW_PADDING);
    let mid_padded = pad(intermediate, MATRIX_ROW_PADDING);
    // one workspace for both quantized operands: two locked slots could share a key and deadlock
    let x_bytes = pad(
        tokens * x_padded / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE,
        WORKSPACE_ALIGN,
    );
    let mid_bytes = pairs * mid_padded / Q8_1_BLOCK_SIZE * Q8_1_TYPE_SIZE;
    let mut workspace = WORKSPACE.ensure(&dev, x_bytes + mid_bytes, &stream)?;
    let (scratch, _scratch_guard) = workspace.ptr_mut();
    let x_q8 = scratch as *mut std::ffi::c_void;
    let mid_q8 = (scratch + x_bytes as u64) as *mut std::ffi::c_void;

    quantize_rows_q8_1(xs, x_q8, x_padded, &stream)?;

    let ids = ids.flatten_all()?.contiguous()?;
    let (ids_storage, ids_layout) = ids.storage_and_layout();
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        inference_tensor::bail!("fast_mmvq: expert ids must live on CUDA");
    };
    let (ids_ptr, _ids_guard) = slice_ptr_on_stream(
        ids_cuda.as_cuda_slice::<u32>()?,
        ids_layout.start_offset(),
        &stream,
    );
    let ids_ptr = ids_ptr as *const std::ffi::c_void;

    let mut activated = unsafe { dev.alloc::<f32>(pairs * intermediate)? };
    let mut out = unsafe { dev.alloc::<f32>(pairs * hidden)? };
    {
        let (gate_ptr, _gate_guard) = gate.kernel_ptr(&stream)?;
        let (up_ptr, _up_guard) = up.kernel_ptr(&stream)?;
        let (mid_ptr, _mid_guard) = slice_ptr_mut_on_stream(&mut activated, 0, &stream);
        let launch = moe_launcher(gate.gguf_type()).expect("supports_moe checked");
        unsafe {
            launch(
                gate_ptr as *const _,
                up_ptr as *const _,
                gate_stride,
                x_q8,
                ids_ptr,
                mid_ptr as *mut f32,
                hidden as i32,
                intermediate as i32,
                (x_padded / Q8_1_BLOCK_SIZE) as i32,
                topk as i32,
                pairs as i32,
                activation as i32,
                stream_ptr,
            );
            ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                mid_ptr as *const _,
                mid_q8,
                intermediate as i32,
                mid_padded as i32,
                pairs as i32,
                stream_ptr,
            );
        }
    }
    {
        let (down_ptr, _down_guard) = down.kernel_ptr(&stream)?;
        let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
        let launch = moe_launcher(down.gguf_type()).expect("supports_moe checked");
        unsafe {
            launch(
                down_ptr as *const _,
                std::ptr::null(),
                down_stride,
                mid_q8,
                ids_ptr,
                out_ptr as *mut f32,
                intermediate as i32,
                hidden as i32,
                (mid_padded / Q8_1_BLOCK_SIZE) as i32,
                MOE_DOWN_Y_DIV,
                pairs as i32,
                MOE_NO_ACTIVATION,
                stream_ptr,
            );
        }
    }
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from((pairs, hidden)),
    )))
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
        (GgufType::Iq2Xxs, DType::BF16) => ffi::launch_dequantize_iq2_xxs_bf16,
        (GgufType::Iq2Xxs, DType::F16) => ffi::launch_dequantize_iq2_xxs_f16,
        (GgufType::Iq2Xxs, DType::F32) => ffi::launch_dequantize_iq2_xxs_f32,
        (GgufType::Iq2Xs, DType::BF16) => ffi::launch_dequantize_iq2_xs_bf16,
        (GgufType::Iq2Xs, DType::F16) => ffi::launch_dequantize_iq2_xs_f16,
        (GgufType::Iq2Xs, DType::F32) => ffi::launch_dequantize_iq2_xs_f32,
        (GgufType::Iq2S, DType::BF16) => ffi::launch_dequantize_iq2_s_bf16,
        (GgufType::Iq2S, DType::F16) => ffi::launch_dequantize_iq2_s_f16,
        (GgufType::Iq2S, DType::F32) => ffi::launch_dequantize_iq2_s_f32,
        (GgufType::Iq3Xxs, DType::BF16) => ffi::launch_dequantize_iq3_xxs_bf16,
        (GgufType::Iq3Xxs, DType::F16) => ffi::launch_dequantize_iq3_xxs_f16,
        (GgufType::Iq3Xxs, DType::F32) => ffi::launch_dequantize_iq3_xxs_f32,
        (GgufType::Iq3S, DType::BF16) => ffi::launch_dequantize_iq3_s_bf16,
        (GgufType::Iq3S, DType::F16) => ffi::launch_dequantize_iq3_s_f16,
        (GgufType::Iq3S, DType::F32) => ffi::launch_dequantize_iq3_s_f32,
        (GgufType::Iq1S, DType::BF16) => ffi::launch_dequantize_iq1_s_bf16,
        (GgufType::Iq1S, DType::F16) => ffi::launch_dequantize_iq1_s_f16,
        (GgufType::Iq1S, DType::F32) => ffi::launch_dequantize_iq1_s_f32,
        (GgufType::Iq4Nl, DType::BF16) => ffi::launch_dequantize_iq4_nl_bf16,
        (GgufType::Iq4Nl, DType::F16) => ffi::launch_dequantize_iq4_nl_f16,
        (GgufType::Iq4Nl, DType::F32) => ffi::launch_dequantize_iq4_nl_f32,
        (GgufType::Iq4Xs, DType::BF16) => ffi::launch_dequantize_iq4_xs_bf16,
        (GgufType::Iq4Xs, DType::F16) => ffi::launch_dequantize_iq4_xs_f16,
        (GgufType::Iq4Xs, DType::F32) => ffi::launch_dequantize_iq4_xs_f32,
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
        (GgufType::Iq2K, DType::BF16) => ffi::launch_dequantize_iq2_k_bf16,
        (GgufType::Iq2K, DType::F16) => ffi::launch_dequantize_iq2_k_f16,
        (GgufType::Iq2K, DType::F32) => ffi::launch_dequantize_iq2_k_f32,
        (GgufType::Iq3K, DType::BF16) => ffi::launch_dequantize_iq3_k_bf16,
        (GgufType::Iq3K, DType::F16) => ffi::launch_dequantize_iq3_k_f16,
        (GgufType::Iq3K, DType::F32) => ffi::launch_dequantize_iq3_k_f32,
        (GgufType::Iq4K, DType::BF16) => ffi::launch_dequantize_iq4_k_bf16,
        (GgufType::Iq4K, DType::F16) => ffi::launch_dequantize_iq4_k_f16,
        (GgufType::Iq4K, DType::F32) => ffi::launch_dequantize_iq4_k_f32,
        (GgufType::Iq5K, DType::BF16) => ffi::launch_dequantize_iq5_k_bf16,
        (GgufType::Iq5K, DType::F16) => ffi::launch_dequantize_iq5_k_f16,
        (GgufType::Iq5K, DType::F32) => ffi::launch_dequantize_iq5_k_f32,
        (GgufType::Iq6K, DType::BF16) => ffi::launch_dequantize_iq6_k_bf16,
        (GgufType::Iq6K, DType::F16) => ffi::launch_dequantize_iq6_k_f16,
        (GgufType::Iq6K, DType::F32) => ffi::launch_dequantize_iq6_k_f32,
        (GgufType::Iq4Ks, DType::BF16) => ffi::launch_dequantize_iq4_ks_bf16,
        (GgufType::Iq4Ks, DType::F16) => ffi::launch_dequantize_iq4_ks_f16,
        (GgufType::Iq4Ks, DType::F32) => ffi::launch_dequantize_iq4_ks_f32,
        (GgufType::Iq2Ks, DType::BF16) => ffi::launch_dequantize_iq2_ks_bf16,
        (GgufType::Iq2Ks, DType::F16) => ffi::launch_dequantize_iq2_ks_f16,
        (GgufType::Iq2Ks, DType::F32) => ffi::launch_dequantize_iq2_ks_f32,
        (GgufType::Iq4Kss, DType::BF16) => ffi::launch_dequantize_iq4_kss_bf16,
        (GgufType::Iq4Kss, DType::F16) => ffi::launch_dequantize_iq4_kss_f16,
        (GgufType::Iq4Kss, DType::F32) => ffi::launch_dequantize_iq4_kss_f32,
        (GgufType::Iq5Ks, DType::BF16) => ffi::launch_dequantize_iq5_ks_bf16,
        (GgufType::Iq5Ks, DType::F16) => ffi::launch_dequantize_iq5_ks_f16,
        (GgufType::Iq5Ks, DType::F32) => ffi::launch_dequantize_iq5_ks_f32,
        (GgufType::Iq3Ks, DType::BF16) => ffi::launch_dequantize_iq3_ks_bf16,
        (GgufType::Iq3Ks, DType::F16) => ffi::launch_dequantize_iq3_ks_f16,
        (GgufType::Iq3Ks, DType::F32) => ffi::launch_dequantize_iq3_ks_f32,
        (GgufType::Iq2Kl, DType::BF16) => ffi::launch_dequantize_iq2_kl_bf16,
        (GgufType::Iq2Kl, DType::F16) => ffi::launch_dequantize_iq2_kl_f16,
        (GgufType::Iq2Kl, DType::F32) => ffi::launch_dequantize_iq2_kl_f32,
        _ => return None,
    })
}

pub fn can_dequantize(ty: GgufType, dtype: DType) -> bool {
    dequantize_launcher(ty, dtype).is_some()
}

/// The weight dequantized on its GPU, as ggml does for prefill on types without an mmq tile.
pub fn dequantize<W: KernelWeight + ?Sized>(w: &W, dtype: DType) -> Result<Tensor> {
    let ty = w.gguf_type();
    let Some(launcher) = dequantize_launcher(ty, dtype) else {
        inference_tensor::bail!("no CUDA dequantizer for {ty:?} to {dtype:?}");
    };
    let Device::Cuda(dev) = w.kernel_device() else {
        inference_tensor::bail!("fast_mmvq: weight must live on CUDA");
    };
    let shape = w.kernel_shape().clone();
    let elems = shape.elem_count();
    let (mut nrows, mut ncols) = shape.dims2()?;
    // ggml's IQ4_NL dequantizer fills whole 256-element super-blocks; the weight's zeroed padding covers the reads
    let filled = if ty.block_size() < QK_K {
        (nrows, ncols) = (1, elems.next_multiple_of(QK_K));
        ncols
    } else {
        elems
    };
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let (weight_ptr, _weight_guard) = w.kernel_ptr(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;
    macro_rules! run {
        ($t:ty) => {{
            let mut out = unsafe { dev.alloc::<$t>(filled)? };
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
    let out = Tensor::from((Storage::Cuda(storage), Shape::from(filled)));
    if filled == elems {
        return out.reshape(shape);
    }
    out.narrow(0, 0, elems)?.reshape(shape)
}

// IQ4_KS block layout (ik_llama.cpp): a row is an f32 scale then 136-byte blocks of 256 values, 8 scale bytes and
// 128 bytes of nibbles
const IQ4_KS_ROW_SCALE_BYTES: usize = 4;
const IQ4_KS_BLOCK_BYTES: usize = 136;
const IQ4_KS_BLOCK_SCALES: usize = 8;
// mmvq_tc.cuh's tile-ordered IQ4_KS: rows per group and the group's row-scale header; per superblock the block
// scales ([block][row] bytes), then per block two halves of 8 rows x 4 words
const TILED_GROUP_ROWS: usize = 16;
const TILED_GROUP_HEADER: usize = 64;
const TILED_SCALE_BYTES: usize = IQ4_KS_BLOCK_SCALES * TILED_GROUP_ROWS;
const TILED_HALF_BYTES: usize = 128;
const TILED_BLOCK_BYTES: usize = 2 * TILED_HALF_BYTES;
const TILED_SB_BYTES: usize = TILED_SCALE_BYTES + IQ4_KS_BLOCK_SCALES * TILED_BLOCK_BYTES;
// IQ4_KS nibble bytes per 32-value block, read as 4 words
const IQ4_KS_BLOCK_QS: usize = 16;

// Where row `r`'s word `t` of block `j` sits in a tiled superblock (the mma A fragment's lane order)
fn tiled_word(j: usize, r: usize, t: usize) -> usize {
    TILED_SCALE_BYTES + j * TILED_BLOCK_BYTES + (r / 8) * TILED_HALF_BYTES + ((r % 8) * 4 + t) * 4
}

/// IQ4_KS rows regrouped into mmvq_tc.cuh's tile order (`mmvq_tc_iq4_ks_kernel`), rows padded to 16 with zeros.
pub(crate) fn iq4_ks_to_tiled(bytes: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    use rayon::prelude::*;
    let nsb = ncols / QK_K;
    let row_bytes = IQ4_KS_ROW_SCALE_BYTES + nsb * IQ4_KS_BLOCK_BYTES;
    let group_bytes = TILED_GROUP_HEADER + nsb * TILED_SB_BYTES;
    let mut out = vec![0u8; nrows.div_ceil(TILED_GROUP_ROWS) * group_bytes];
    out.par_chunks_exact_mut(group_bytes)
        .zip(bytes.par_chunks(TILED_GROUP_ROWS * row_bytes))
        .for_each(|(group, rows)| {
            for (r, src) in rows.chunks_exact(row_bytes).enumerate() {
                group[4 * r..4 * r + 4].copy_from_slice(&src[..IQ4_KS_ROW_SCALE_BYTES]);
                for sb in 0..nsb {
                    let block = &src[IQ4_KS_ROW_SCALE_BYTES + sb * IQ4_KS_BLOCK_BYTES..]
                        [..IQ4_KS_BLOCK_BYTES];
                    let (scales, qs) = block.split_at(IQ4_KS_BLOCK_SCALES);
                    let dst =
                        &mut group[TILED_GROUP_HEADER + sb * TILED_SB_BYTES..][..TILED_SB_BYTES];
                    for j in 0..IQ4_KS_BLOCK_SCALES {
                        dst[j * TILED_GROUP_ROWS + r] = scales[j];
                        for t in 0..4 {
                            let (word, q) = (tiled_word(j, r, t), IQ4_KS_BLOCK_QS * j + 4 * t);
                            dst[word..word + 4].copy_from_slice(&qs[q..q + 4]);
                        }
                    }
                }
            }
        });
    out
}

/// A tile-ordered IQ4_KS weight's first `nrows` rows back in IQ4_KS's row-major layout (`iq4_ks_to_tiled`'s inverse).
pub(crate) fn iq4_ks_from_tiled(tiled: &[u8], nrows: usize, ncols: usize) -> Vec<u8> {
    let nsb = ncols / QK_K;
    let row_bytes = IQ4_KS_ROW_SCALE_BYTES + nsb * IQ4_KS_BLOCK_BYTES;
    let group_bytes = TILED_GROUP_HEADER + nsb * TILED_SB_BYTES;
    let mut out = vec![0u8; nrows * row_bytes];
    for row in 0..nrows {
        let dst = &mut out[row * row_bytes..(row + 1) * row_bytes];
        let (rg, r) = (row / TILED_GROUP_ROWS, row % TILED_GROUP_ROWS);
        let group = &tiled[rg * group_bytes..(rg + 1) * group_bytes];
        dst[..IQ4_KS_ROW_SCALE_BYTES].copy_from_slice(&group[4 * r..4 * r + 4]);
        for sb in 0..nsb {
            let src = &group[TILED_GROUP_HEADER + sb * TILED_SB_BYTES..][..TILED_SB_BYTES];
            let block =
                &mut dst[IQ4_KS_ROW_SCALE_BYTES + sb * IQ4_KS_BLOCK_BYTES..][..IQ4_KS_BLOCK_BYTES];
            let (scales, qs) = block.split_at_mut(IQ4_KS_BLOCK_SCALES);
            for j in 0..IQ4_KS_BLOCK_SCALES {
                scales[j] = src[j * TILED_GROUP_ROWS + r];
                for t in 0..4 {
                    let (word, q) = (tiled_word(j, r, t), IQ4_KS_BLOCK_QS * j + 4 * t);
                    qs[q..q + 4].copy_from_slice(&src[word..word + 4]);
                }
            }
        }
    }
    out
}

type TiledLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_dst: i32,
    b_size: i32,
    cfg: i32,
    stream: *mut std::ffi::c_void,
);

// mmvq_tc.cuh's launch_mmvq_tc_iq4_ks shapes; `tune_tiled` times each per weight shape
pub(crate) const TILED_CONFIGS: i32 = 4;
const TILED_TUNE_WARMUP: usize = 3;
const TILED_TUNE_ITERS: usize = 20;

// (device ordinal, rows, cols)
type TiledShape = (usize, usize, usize);

// The fastest config per shape, filled at load by `tune_tiled`
static TILED_CONFIG: std::sync::RwLock<Vec<(TiledShape, i32)>> = std::sync::RwLock::new(Vec::new());

fn tiled_launcher(input_ty: DType) -> TiledLauncher {
    match input_ty {
        DType::BF16 => ffi::launch_mmvq_tc_gguf_iq4_ks_bf16_plain,
        DType::F16 => ffi::launch_mmvq_tc_gguf_iq4_ks_f16_plain,
        _ => ffi::launch_mmvq_tc_gguf_iq4_ks_f32_plain,
    }
}

/// Whether `dev` runs the tile-ordered IQ4_KS matvec (int8 `mma.m16n8k32`).
pub fn tensor_core_capable(dev: &inference_tensor::CudaDevice) -> bool {
    dev.compute_cap() >= TENSOR_CORE_MIN_CC
}

// The quantized activations and a launch of the tiled matvec with config `cfg`, on raw pointers
struct TiledCall {
    weight: *const std::ffi::c_void,
    scratch: *mut std::ffi::c_void,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    stream: *mut std::ffi::c_void,
}

impl TiledCall {
    unsafe fn quantize(&self, input_ty: DType, xs: *const std::ffi::c_void) {
        let quantize = match input_ty {
            DType::BF16 => ffi::launch_mmvq_tc_quantize_bf16,
            DType::F16 => ffi::launch_mmvq_tc_quantize_f16,
            _ => ffi::launch_mmvq_tc_quantize_f32,
        };
        unsafe {
            quantize(
                xs,
                self.scratch,
                self.ncols as i32,
                self.b_size as i32,
                self.stream,
            )
        };
    }

    unsafe fn launch(&self, input_ty: DType, out: *mut std::ffi::c_void, cfg: i32) {
        unsafe {
            tiled_launcher(input_ty)(
                self.weight,
                self.scratch as *const std::ffi::c_void,
                out,
                self.ncols as i32,
                self.nrows as i32,
                self.nrows as i32,
                self.b_size as i32,
                cfg,
                self.stream,
            )
        };
    }
}

fn tiled_config(ordinal: usize, nrows: usize, ncols: usize) -> i32 {
    let configs = TILED_CONFIG.read().unwrap_or_else(|e| e.into_inner());
    configs
        .iter()
        .find(|(key, _)| *key == (ordinal, nrows, ncols))
        .map_or(0, |(_, cfg)| *cfg)
}

/// Times the tiled matvec's configs on a tile-ordered IQ4_KS weight at 8 rows and keeps the fastest for its shape.
pub(crate) fn tune_tiled<W: KernelWeight + ?Sized>(w: &W) -> Result<()> {
    let Device::Cuda(dev) = w.kernel_device() else {
        inference_tensor::bail!("tune_tiled: weight must live on CUDA");
    };
    let (nrows, ncols) = w.kernel_shape().dims2()?;
    let key = (dev.cuda_stream().context().ordinal(), nrows, ncols);
    if TILED_CONFIG
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|(k, _)| *k == key)
    {
        return Ok(());
    }
    let stream = dev.cuda_stream();
    let b_size = MMVQ_MAX_BATCH;
    let xs = unsafe { dev.alloc::<f32>(b_size * ncols)? };
    let mut out = unsafe { dev.alloc::<f32>(b_size * nrows)? };
    let scratch_bytes = (ncols / Q8_1_BLOCK_SIZE) * TENSOR_CORE_ACTIVATION_BLOCK_BYTES;
    let mut workspace = WORKSPACE.ensure(&dev, scratch_bytes, &stream)?;
    let (scratch, _scratch_guard) = workspace.ptr_mut();
    let (weight, _weight_guard) = w.kernel_ptr(&stream)?;
    let (xs_ptr, _xs_guard) = slice_ptr_on_stream(&xs, 0, &stream);
    let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    let call = TiledCall {
        weight: weight as *const std::ffi::c_void,
        scratch: scratch as *mut std::ffi::c_void,
        ncols,
        nrows,
        b_size,
        stream: stream.cu_stream() as *mut std::ffi::c_void,
    };
    let out_ptr = out_ptr as *mut std::ffi::c_void;
    unsafe { call.quantize(DType::F32, xs_ptr as *const std::ffi::c_void) };
    let timing = Some(inference_tensor::cuda::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let mut best = (f32::INFINITY, 0);
    for cfg in 0..TILED_CONFIGS {
        for _ in 0..TILED_TUNE_WARMUP {
            unsafe { call.launch(DType::F32, out_ptr, cfg) };
        }
        let start = stream
            .record_event(timing)
            .map_err(inference_tensor::Error::wrap)?;
        for _ in 0..TILED_TUNE_ITERS {
            unsafe { call.launch(DType::F32, out_ptr, cfg) };
        }
        let end = stream
            .record_event(timing)
            .map_err(inference_tensor::Error::wrap)?;
        let ms = start
            .elapsed_ms(&end)
            .map_err(inference_tensor::Error::wrap)?;
        if ms < best.0 {
            best = (ms, cfg);
        }
    }
    TILED_CONFIG
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .push((key, best.1));
    Ok(())
}

// `plain` for a tile-ordered IQ4_KS weight: the tensor-core matvec reads each weight tile once for all 1-8 rows
pub(crate) fn tiled_plain<W: KernelWeight + ?Sized>(
    w: &W,
    xs: &Tensor,
    cfg: i32,
) -> Result<Tensor> {
    let Device::Cuda(dev) = w.kernel_device() else {
        inference_tensor::bail!("fast_mmvq: weight must live on CUDA");
    };
    let dev = &dev;
    let (nrows, ncols) = w.kernel_shape().dims2()?;
    let input_ty = xs.dtype();
    let b_size = xs.elem_count() / ncols;
    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        inference_tensor::bail!("fast_mmvq: input must live on CUDA");
    };
    let scratch_bytes = (ncols / Q8_1_BLOCK_SIZE) * TENSOR_CORE_ACTIVATION_BLOCK_BYTES;
    let mut workspace = WORKSPACE.ensure(dev, scratch_bytes, &stream)?;
    let (scratch, _scratch_guard) = workspace.ptr_mut();
    let (weight, _weight_guard) = w.kernel_ptr(&stream)?;
    let call = TiledCall {
        weight: weight as *const std::ffi::c_void,
        scratch: scratch as *mut std::ffi::c_void,
        ncols,
        nrows,
        b_size,
        stream: stream.cu_stream() as *mut std::ffi::c_void,
    };
    macro_rules! run {
        ($t:ty) => {{
            let slice = xs_cuda.as_cuda_slice::<$t>()?;
            let mut out = unsafe { dev.alloc::<$t>(nrows * b_size)? };
            {
                let (xs_ptr, _xs_guard) =
                    slice_ptr_on_stream(slice, xs_layout.start_offset(), &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
                unsafe {
                    call.quantize(input_ty, xs_ptr as *const std::ffi::c_void);
                    call.launch(input_ty, out_ptr as *mut std::ffi::c_void, cfg);
                }
            }
            CudaStorage::wrap_cuda_slice(out, dev.clone())
        }};
    }
    let storage = match input_ty {
        DType::BF16 => run!(half::bf16),
        DType::F16 => run!(half::f16),
        _ => run!(f32),
    };
    Ok(Tensor::from((
        Storage::Cuda(storage),
        output_shape(&xs, nrows),
    )))
}
