#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace inference_flashinfer {

struct DecodeArgs {
  void *q;
  void *key_cache;
  void *value_cache;
  const int32_t *kv_indptr;
  const int32_t *kv_indices;
  const int32_t *kv_last_page_len;
  const int32_t *request_indices;
  const int32_t *kv_tile_indices;
  const int32_t *o_indptr;
  const int32_t *kv_chunk_size_ptr;
  const bool *block_valid_mask;
  void *o;
  void *tmp_v;
  void *tmp_s;
  int32_t batch_size;
  int32_t padded_batch_size;
  int32_t num_qo_heads;
  int32_t num_kv_heads;
  int32_t page_size;
  int32_t q_stride_n;
  int32_t q_stride_h;
  float sm_scale;
  int32_t window_left;
  float logits_soft_cap;
  float k_scale;
  float v_scale;
  cudaStream_t stream;
};

// Instantiated in flashinfer_decode/*.cu, one TU per fp8 (dtype, head dim) since each costs minutes of ptxas.
template <typename DType, typename CacheType, uint32_t HEAD_DIM>
cudaError_t dispatch_flashinfer_decode_softcap(const DecodeArgs &args);

} // namespace inference_flashinfer
