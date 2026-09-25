#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdio.h>

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/default_decode_params.cuh"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/variants.cuh"
#include "flashinfer_decode.cuh"

namespace inference_flashinfer {

using namespace flashinfer;

template <typename DType, typename CacheType, uint32_t HEAD_DIM,
          bool USE_SLIDING_WINDOW, bool USE_LOGITS_SOFT_CAP>
cudaError_t run_flashinfer_decode(const DecodeArgs &a) {
  using Params = BatchDecodeParams<DType, CacheType, DType, int32_t>;
  using AttentionVariant =
      DefaultAttention<false, USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, false>;

  paged_kv_t<CacheType, int32_t> paged_kv(
      a.num_kv_heads, a.page_size, HEAD_DIM, a.batch_size, QKVLayout::kHND,
      static_cast<CacheType *>(a.key_cache),
      static_cast<CacheType *>(a.value_cache),
      const_cast<int32_t *>(a.kv_indices), const_cast<int32_t *>(a.kv_indptr),
      const_cast<int32_t *>(a.kv_last_page_len));

  Params params(static_cast<DType *>(a.q), /*q_rope_offset=*/nullptr, paged_kv,
                static_cast<DType *>(a.o), /*lse=*/nullptr,
                /*maybe_alibi_slopes=*/nullptr, a.num_qo_heads, a.q_stride_n,
                a.q_stride_h, a.window_left, a.logits_soft_cap,
                a.sm_scale * a.k_scale, 1.0f, 1.0f);
  params.v_scale = a.v_scale;
  params.request_indices = const_cast<int32_t *>(a.request_indices);
  params.kv_tile_indices = const_cast<int32_t *>(a.kv_tile_indices);
  params.o_indptr = const_cast<int32_t *>(a.o_indptr);
  params.kv_chunk_size_ptr = const_cast<int32_t *>(a.kv_chunk_size_ptr);
  params.block_valid_mask = const_cast<bool *>(a.block_valid_mask);
  params.padded_batch_size = a.padded_batch_size;

  cudaError_t status =
      BatchDecodeWithPagedKVCacheDispatched<HEAD_DIM, PosEncodingMode::kNone,
                                            AttentionVariant, Params>(
          params, static_cast<DType *>(a.tmp_v), static_cast<float *>(a.tmp_s),
          /*enable_pdl=*/false, a.stream);
  if (status != cudaSuccess) {
    fprintf(stderr, "FlashInfer decode failed: %s\n",
            cudaGetErrorString(status));
  }
  return status;
}

template <typename DType, typename CacheType, uint32_t HEAD_DIM>
cudaError_t dispatch_flashinfer_decode_softcap(const DecodeArgs &a) {
  const bool soft_cap = a.logits_soft_cap > 0.0f;
  if (a.window_left >= 0) {
    return soft_cap
               ? run_flashinfer_decode<DType, CacheType, HEAD_DIM, true, true>(a)
               : run_flashinfer_decode<DType, CacheType, HEAD_DIM, true, false>(a);
  }
  return soft_cap
             ? run_flashinfer_decode<DType, CacheType, HEAD_DIM, false, true>(a)
             : run_flashinfer_decode<DType, CacheType, HEAD_DIM, false, false>(a);
}

} // namespace inference_flashinfer

#define INFERENCE_FLASHINFER_DECODE_INSTANTIATE(DTYPE, CACHE_DTYPE, HEAD_DIM)  \
  template cudaError_t                                                         \
  inference_flashinfer::dispatch_flashinfer_decode_softcap<DTYPE, CACHE_DTYPE, \
                                                           HEAD_DIM>(          \
      const inference_flashinfer::DecodeArgs &);

#define INFERENCE_FLASHINFER_DECODE_INSTANTIATE_ALL_HEAD_DIMS(DTYPE, CACHE_DTYPE) \
  INFERENCE_FLASHINFER_DECODE_INSTANTIATE(DTYPE, CACHE_DTYPE, 64)              \
  INFERENCE_FLASHINFER_DECODE_INSTANTIATE(DTYPE, CACHE_DTYPE, 128)             \
  INFERENCE_FLASHINFER_DECODE_INSTANTIATE(DTYPE, CACHE_DTYPE, 256)             \
  INFERENCE_FLASHINFER_DECODE_INSTANTIATE(DTYPE, CACHE_DTYPE, 512)
