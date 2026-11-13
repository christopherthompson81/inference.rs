//! exllamav3's EXL3 matmul kernels (`kernels/cuda/exl3`, over `third_party/exllamav3`), a library of their own,
//! opened when an EXL3 layer first runs on a GPU.

use std::ffi::c_void;

use half::f16;

/// `launch_exl3_moe`'s arguments (`kernels/cuda/exl3/exl3_moe_launch_instance_exl3.cu`, field for field): a MoE
/// layer's routed experts for `bsz` tokens, each projection's table its experts' trellis, `suh` and `svh` addresses.
#[repr(C)]
pub(crate) struct Exl3MoeArgs {
    pub x: *const f16,
    pub sel: *const i64,
    pub rw: *const f16,
    pub gate_tables: *const i64,
    pub up_tables: *const i64,
    pub down_tables: *const i64,
    pub had_g: *mut f16,
    pub had_u: *mut f16,
    pub gu_g: *mut f32,
    pub gu_u: *mut f32,
    pub act_out: *mut f16,
    pub d_out: *mut f32,
    pub ctr: *mut i32,
    pub out: *mut f32,
    pub bsz: i32,
    pub topk: i32,
    pub hidden: i32,
    pub inter: i32,
    pub experts: i32,
    pub act: i32,
    pub gate_up_words: i32,
    pub down_words: i32,
    pub cb: i32,
    pub slots_max: i32,
}

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

        /// A (`m x k`, f16) through `num_src` matrices of their own rotations, cut into `slices` column slices
        /// `slice_width` wide: `b_list` and `svh_list` per slice (offset to its first column), `suh_list` per source,
        /// `ints` the slices' widths, source widths, sources, first columns and source offsets; the outputs end to end
        /// from `c` (each source `m x` its width) through `c_ptrs` (device scratch for `slices` addresses). `a_had`
        /// holds `num_src x m x k`.
        pub(crate) fn launch_exl3_mgemm_sliced(
            a: *const f16,
            b_list: *const u64,
            suh_list: *const u64,
            svh_list: *const u64,
            ints: *const i32,
            num_src: i32,
            slices: i32,
            slice_width: i32,
            c: *mut c_void,
            c_ptrs: *mut u64,
            size_m: i32,
            size_k: i32,
            tile_u16: i32,
            cb: i32,
            c_fp32: bool,
            a_had: *mut f16,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// The counters `launch_exl3_moe` needs, in i32s.
        pub(crate) fn exl3_moe_ctr_len(slots_max: i32, bsz: i32, inter: i32, hidden: i32) -> i64;

        /// Runs the fused MoE layer `args` describes; 0 when launched, -1 when no kernel fits.
        pub(crate) fn launch_exl3_moe(
            args: *const Exl3MoeArgs,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;
    }
}
