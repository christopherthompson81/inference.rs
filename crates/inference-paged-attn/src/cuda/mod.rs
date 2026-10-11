pub const USE_FP8: bool = cfg!(has_fp8);

mod backend;
mod ffi;

/// Opens this crate's CUDA kernel library now, so a missing one fails at model load rather than at a launch.
pub fn ensure_kernels_loaded() -> std::result::Result<(), String> {
    ffi::ensure_loaded()
}

pub use backend::{
    FA3_DECODE_MAX_QUERY_LEN, Fa3DecodeMetadata, Fa3DecodeParams, Fa3DecodeSchedule,
    Fa3PagedMetadataLayout, USE_FA3_FP8_PAGED, concat_and_cache_mla, context_attention_fwd_mla,
    copy_blocks, fa3_fp8_decode, fa3_prepare_decode_metadata, fa3_prepare_paged_metadata,
    flashinfer_mla_decode, gather_kv_cache, gather_kv_cache_flashinfer, gather_mla_cache,
    is_flashinfer_cache, kv_scale_update, reshape_and_cache, reshape_and_cache_flashinfer,
    swap_blocks,
};
