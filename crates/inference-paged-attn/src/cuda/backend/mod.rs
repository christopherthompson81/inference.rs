mod cache;
mod context_attention_mla;
mod fa3;
mod flashinfer;
mod gather_kv;
mod mla;
mod reshape_cache;
mod scale_update;
pub use cache::{copy_blocks, swap_blocks};
pub use context_attention_mla::context_attention_fwd_mla;
pub use fa3::{
    FA3_DECODE_MAX_QUERY_LEN, Fa3DecodeMetadata, Fa3DecodeParams, Fa3DecodeSchedule,
    Fa3PagedMetadataLayout, USE_FA3_FP8_PAGED, fa3_fp8_decode, fa3_prepare_decode_metadata,
    fa3_prepare_paged_metadata,
};
pub use flashinfer::{
    gather_kv_cache_flashinfer, is_flashinfer_cache, reshape_and_cache_flashinfer,
};
pub use gather_kv::gather_kv_cache;
use inference_tensor::cuda::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr},
};
use inference_tensor::{Layout, Result};
pub use mla::{concat_and_cache_mla, flashinfer_mla_decode, gather_mla_cache};
pub use reshape_cache::reshape_and_cache;
pub use scale_update::kv_scale_update;

fn cache_input_layout(
    layout: &Layout,
    name: &str,
    op: &str,
) -> Result<(usize, usize, usize, usize)> {
    let (num_tokens, num_heads, head_size, row_stride) = match *layout.dims() {
        [num_tokens, num_heads, head_size] => {
            (num_tokens, num_heads, head_size, layout.stride()[0])
        }
        [batch, seq_len, num_heads, head_size] => {
            let num_tokens = batch
                .checked_mul(seq_len)
                .ok_or_else(|| inference_tensor::Error::msg("cache input token count overflow"))?;
            let row_stride = if seq_len == 1 {
                layout.stride()[0]
            } else {
                layout.stride()[1]
            };
            if batch > 1 && seq_len > 1 && layout.stride()[0] != seq_len.saturating_mul(row_stride)
            {
                inference_tensor::bail!(
                    "{op} cannot flatten {name} batch/sequence strides: {layout:?}"
                );
            }
            (num_tokens, num_heads, head_size, row_stride)
        }
        _ => inference_tensor::bail!("{op} expects rank-3 or rank-4 {name} input, got {layout:?}"),
    };
    // a single head's stride never addresses anything, so a transposed one-head view is still dense
    if layout.stride()[layout.stride().len() - 1] != 1
        || (num_heads > 1 && layout.stride()[layout.stride().len() - 2] != head_size)
        || row_stride < num_heads.saturating_mul(head_size)
    {
        inference_tensor::bail!("{op} expects dense {name} heads, got {layout:?}");
    }
    Ok((num_tokens, num_heads, head_size, row_stride))
}

pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    slice_ptr_on_stream(v, lo, v.stream())
}

pub fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_input_layout_flattens_uniform_rank_four_rows() -> Result<()> {
        let prefill = Layout::new((2, 3, 2, 4).into(), vec![48, 16, 4, 1], 7);
        assert_eq!(cache_input_layout(&prefill, "key", "test")?, (6, 2, 4, 16));

        let decode = Layout::new((8, 1, 2, 4).into(), vec![24, 8, 4, 1], 5);
        assert_eq!(cache_input_layout(&decode, "value", "test")?, (8, 2, 4, 24));
        Ok(())
    }

    #[test]
    fn cache_input_layout_accepts_a_transposed_single_head() -> Result<()> {
        let one_head = Layout::new((1, 27, 1, 64).into(), vec![1728, 64, 1728, 1], 0);
        assert_eq!(
            cache_input_layout(&one_head, "key", "test")?,
            (27, 1, 64, 64)
        );
        let two_heads = Layout::new((1, 27, 2, 64).into(), vec![3456, 64, 1728, 1], 0);
        assert!(cache_input_layout(&two_heads, "key", "test").is_err());
        Ok(())
    }
}
