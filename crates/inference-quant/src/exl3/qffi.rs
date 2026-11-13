//! exllamav3's trellis quantizer (`kernels/cuda/exl3/*_instance_exl3q.cu`), a library of its own, opened when an EXL3
//! tensor is first written.

use half::f16;

inference_tensor::kernel_library! {
    library = "inferencequant_exl3q";
    path = option_env!("INFERENCE_KERNEL_LIB_INFERENCEQUANT_EXL3Q").unwrap_or("");
    id = option_env!("INFERENCE_KERNEL_ID_INFERENCEQUANT_EXL3Q").unwrap_or("");
    unsafe extern "C" {
        /// Per tile, the cost and history scratch bytes the quantizer needs for `k` bits (`half_k`: k + 0.5) on
        /// codebook `cb`; returns the tiles one wave keeps resident, -1 for a rate it has no kernel for.
        pub(crate) fn exl3_quantize_scratch(
            k: i32,
            cb: i32,
            half_k: bool,
            cost_bytes: *mut i64,
            edge_bytes: *mut i64,
        ) -> i64;

        /// `tiles` tiles of 256 f32 values to their trellis encodings: `out` the values, `idx` each position's
        /// state; the scratch holds `scratch_tiles` tiles. Returns 0, or -1 when no kernel fits.
        pub(crate) fn launch_exl3_quantize_tiles(
            input: *const f32,
            out: *mut f32,
            idx: *mut u16,
            costs: *mut f16,
            edges: *mut u16,
            tiles: i64,
            scratch_tiles: i64,
            k: i32,
            cb: i32,
            half_k: bool,
            stream: inference_tensor::cuda::cudarc::driver::sys::CUstream,
        ) -> i32;
    }
}
