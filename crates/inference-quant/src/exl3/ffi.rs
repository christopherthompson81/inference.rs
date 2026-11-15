//! exllamav3's EXL3 matmul kernels (`kernels/cuda/exl3`, over `third_party/exllamav3`), a library of their own,
//! opened when an EXL3 layer first runs on a GPU.

use std::ffi::c_void;

use half::f16;

/// `launch_exl3_moe`'s arguments (`kernels/cuda/exl3/exl3_moe_launch_instance_exl3.cu`, field for field): a MoE
/// layer's routed experts for `bsz` tokens, each projection's table its experts' trellis, `suh` and `svh` addresses.
#[repr(C)]
pub(crate) struct Exl3MoeArgs {
    pub x: *const c_void,
    pub sel: *const c_void,
    pub rw: *const c_void,
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
    pub out: *mut c_void,
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
    pub x_bf16: bool,
    pub sel_u32: bool,
    pub rw_f32: bool,
    pub out_bf16: bool,
}

inference_tensor::kernel_library! {
    library = "inferencequant_exl3";
    path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT_EXL3").unwrap_or("");
    id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT_EXL3").unwrap_or("");
    unsafe extern "C" {
        /// One row `x` (f16, or bf16 with `x_bf16`) through `jobs` trellises of one k (`b`, `suh`, `svh`, `out`, `n`
        /// apiece) into each `out` (f32, f16 or bf16 by `out_kind`), rotated by its `suh`/`svh` and the Hadamards
        /// inside; `ws` is the stream's zeroed workspace of `ws_floats` words, its counters left zeroed. 0 when
        /// launched, -1 for a shape, rate or workspace it does not take.
        pub(crate) fn launch_exl3_dgemv(
            x: *const c_void,
            x_bf16: bool,
            jobs: i32,
            b: *const *const u16,
            suh: *const *const f16,
            svh: *const *const f16,
            out: *const *mut c_void,
            n: *const i32,
            size_k: i32,
            tile_u16: i32,
            cb: i32,
            ws: *mut f32,
            ws_floats: i64,
            out_kind: i32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// One token's routed experts through the decode kernels: gate, up (silu) and down, each slot weighted by
        /// `rw` and summed in order into `out` (`hidden`, of `out_kind`); `x` f16 or bf16, `sel` i64 or u32, `rw` f16
        /// or f32. Each table holds `experts` trellis, then suh, then svh addresses. `ws` as for launch_exl3_dgemv.
        /// 0 when launched, -1 for shapes, rates or a workspace it does not take.
        pub(crate) fn launch_exl3_dmoe(
            x: *const c_void,
            x_bf16: bool,
            sel: *const c_void,
            sel_u32: bool,
            rw: *const c_void,
            rw_f32: bool,
            gate: *const i64,
            up: *const i64,
            down: *const i64,
            experts: i32,
            topk: i32,
            hidden: i32,
            inter: i32,
            gu_tile_u16: i32,
            d_tile_u16: i32,
            cb: i32,
            ws: *mut f32,
            ws_floats: i64,
            out: *mut c_void,
            out_kind: i32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

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
        /// from `c` (each source `m x` its width) through `c_ptrs` (device scratch for `slices` addresses, twice that
        /// with `c_bf`). `a_had` holds `num_src x m x k`. `a` is bf16 when `a_bf16`; a non-null `c_bf` takes the
        /// outputs as bf16 (`c` f32 scratch then).
        pub(crate) fn launch_exl3_mgemm_sliced(
            a: *const c_void,
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
            a_bf16: bool,
            c_bf: *mut c_void,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// `rows x k` x (f16, or bf16 with `x_bf16`) times `suh`, rotated by 128-wide Hadamards, into `out` (f16); -1
        /// for a shape it does not take.
        pub(crate) fn launch_exl3_had_in(
            x: *const c_void,
            x_bf16: bool,
            out: *mut f16,
            suh: *const f16,
            rows: i32,
            k: i32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// `rows x n` f32 `y` rotated by 128-wide Hadamards, times `svh`: in place, or into `out_bf16` when not null.
        pub(crate) fn launch_exl3_had_out(
            y: *mut f32,
            out_bf16: *mut c_void,
            svh: *const f16,
            rows: i32,
            n: i32,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;

        /// The trellis (`k / 16 x n / 16` tiles of `words` uint16) as its f16 weight `k x n` in the rotated basis;
        /// -1 for a rate or shape it does not take.
        pub(crate) fn launch_exl3_reconstruct(
            out: *mut f16,
            trellis: *const u16,
            k: i32,
            n: i32,
            words: i32,
            cb: i32,
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
