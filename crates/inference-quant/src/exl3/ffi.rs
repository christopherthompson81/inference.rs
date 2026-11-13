//! exllamav3's EXL3 matmul kernels (`kernels/cuda/exl3`, over `third_party/exllamav3`), a library of their own,
//! opened when an EXL3 layer first runs on a GPU.

use std::ffi::c_void;

use half::f16;

inference_tensor::kernel_library! {
    library = "inferencequant_exl3";
    path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT_EXL3").unwrap_or("");
    id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT_EXL3").unwrap_or("");
    unsafe extern "C" {
        /// C (`m x n`, f16 or with `c_fp32` f32) = A (`m x k`, f16) times the trellis (`k/16 x n/16 x tile_u16`), A
        /// rotated by `suh` and the Hadamard into `a_had` (A's size) on the way in, C by the Hadamard and `svh` on the
        /// way out. Returns the kernel that ran, 0 for nothing to do, -1 when no kernel fits.
        pub(crate) fn launch_exl3_gemm(
            a: *const f16,
            b: *const u16,
            c: *mut c_void,
            size_m: i32,
            size_k: i32,
            size_n: i32,
            tile_u16: i32,
            cb: i32,
            c_fp32: bool,
            suh: *const f16,
            a_had: *mut f16,
            svh: *const f16,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// Slot j's C (`m x n`) = A[j] (`m x k`, f16) times expert `indices[j]`, whose trellis, `suh` and `svh` the
        /// three lists address; `a_had` holds `slots x m x k`. Returns as `launch_exl3_gemm`.
        pub(crate) fn launch_exl3_mgemm(
            a: *const f16,
            b_list: *const u64,
            c: *mut c_void,
            size_m: i32,
            size_k: i32,
            size_n: i32,
            tile_u16: i32,
            cb: i32,
            c_fp32: bool,
            suh_list: *const u64,
            a_had: *mut f16,
            svh_list: *const u64,
            indices: *const i64,
            slots: i32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;
    }
}
