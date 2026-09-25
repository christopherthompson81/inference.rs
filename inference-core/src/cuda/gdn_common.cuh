#pragma once

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cmath>
#include <cstdint>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
using gdn_fp8_e4m3 = __nv_fp8_e4m3;
#else
struct alignas(1) gdn_fp8_e4m3 {
  unsigned char value;

  __device__ explicit gdn_fp8_e4m3(float) : value(0) {}
};
#endif

constexpr int GDN_CHANNEL_BLOCK_SIZE = 256;
constexpr int GDN_DECODE_VALUE_TILE = 64;
constexpr int GDN_DECODE_STATE_LOAD_UNROLL = 128;
constexpr int GDN_DECODE_STATE_UPDATE_TILE_ROWS = 32;
constexpr int GDN_DECODE_COOPERATIVE_K = 128;
constexpr int GDN_DECODE_COOPERATIVE_V = 16;
constexpr int GDN_DECODE_COOPERATIVE_V_PADDED = 20;
constexpr int GDN_DECODE_COOPERATIVE_THREADS = 128;
constexpr int GDN_DECODE_COOPERATIVE_VALUES_PER_WARP = 4;
constexpr int GDN_DECODE_PIPELINED_K = 128;
constexpr int GDN_DECODE_PIPELINED_V = 32;
constexpr int GDN_DECODE_PIPELINED_V_PADDED = 36;
constexpr int GDN_DECODE_PIPELINED_STAGES = 2;
constexpr int GDN_DECODE_PIPELINED_THREADS = 256;
constexpr int GDN_DECODE_PIPELINED_WARPS = GDN_DECODE_PIPELINED_THREADS / 32;
constexpr int GDN_DECODE_PIPELINED_VALUES_PER_WARP = 4;
constexpr int GDN_DECODE_KERNEL_COOPERATIVE = 1;
constexpr int GDN_DECODE_KERNEL_PIPELINED = 2;
constexpr int GDN_DECODE_KERNEL_VALUE_MAJOR_4 = 3;
constexpr int GDN_DECODE_KERNEL_VALUE_MAJOR_32 = 4;
constexpr int GDN_DECODE_VALUE_MAJOR_K = 128;
constexpr int GDN_DECODE_VALUE_MAJOR_V = 128;
constexpr int GDN_PACKED_CONV_WIDTH = 4;
constexpr int GDN_PREFILL_CONV_THREADS = 128;
constexpr int GDN_PREFILL_CONV_TOKEN_TILE = 64;
constexpr int GDN_RMSNORM_FAST_HIDDEN = 128;
constexpr int GDN_RMSNORM_ROWS_PER_BLOCK = 8;
constexpr int GDN_RMSNORM_TILED_LANES_PER_ROW = 16;
constexpr int GDN_RMSNORM_TILED_ROW_PAIR_OFFSET = 2;
constexpr int GDN_RMSNORM_TILED_ROWS_PER_BLOCK = 4;
constexpr int GDN_RMSNORM_TILED_MIN_ROWS = 1024;
constexpr int GDN_RMSNORM_TILED_VALUES_PER_LANE = 8;
constexpr int GDN_STATE_DTYPE_F16 = 0;
constexpr int GDN_STATE_DTYPE_BF16 = 1;
constexpr int GDN_STATE_DTYPE_F32 = 2;
constexpr uint32_t GDN_PENDING_KEY_BANK_MASK = 1u;

__device__ __forceinline__ float4 gdn_load_state_x4(const float *source) {
  return *reinterpret_cast<const float4 *>(source);
}

__device__ __forceinline__ float4
gdn_load_state_x4(const __half *source) {
  const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(source));
  const float2 hi =
      __half22float2(*reinterpret_cast<const __half2 *>(source + 2));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ float4
gdn_load_state_x4(const __nv_bfloat16 *source) {
  const float2 lo =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(source));
  const float2 hi = __bfloat1622float2(
      *reinterpret_cast<const __nv_bfloat162 *>(source + 2));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

__device__ __forceinline__ void gdn_store_state_x4(float *destination,
                                                   float4 value) {
  *reinterpret_cast<float4 *>(destination) = value;
}

__device__ __forceinline__ void gdn_store_state_x4(__half *destination,
                                                   float4 value) {
  *reinterpret_cast<__half2 *>(destination) =
      __floats2half2_rn(value.x, value.y);
  *reinterpret_cast<__half2 *>(destination + 2) =
      __floats2half2_rn(value.z, value.w);
}

__device__ __forceinline__ void
gdn_store_state_x4(__nv_bfloat16 *destination, float4 value) {
  *reinterpret_cast<__nv_bfloat162 *>(destination) =
      __floats2bfloat162_rn(value.x, value.y);
  *reinterpret_cast<__nv_bfloat162 *>(destination + 2) =
      __floats2bfloat162_rn(value.z, value.w);
}

__device__ __forceinline__ bool
gdn_is_padding_row(const int32_t *__restrict__ slot_indices, int bidx) {
  return slot_indices && slot_indices[bidx] < 0;
}

template <typename T>
__device__ __forceinline__ T gdn_ragged_from_float(float value);

template <>
__device__ __forceinline__ float gdn_ragged_from_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16
gdn_ragged_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

// (batch, head) -> row of the state buffer: identity on a gathered [B*H, ...] copy, or through the
// per-batch slot table when a kernel updates the recurrent state pool in place
__device__ __forceinline__ size_t gdn_state_row(const int32_t *__restrict__ slot_indices,
                                                int bidx, int h, int num_heads) {
  const size_t slot = slot_indices ? (size_t)slot_indices[bidx] : (size_t)bidx;
  return slot * num_heads + h;
}

template <int WARP_SIZE>
__device__ __forceinline__ float gdn_warp_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset, WARP_SIZE);
  }
  return __shfl_sync(0xffffffff, x, 0, WARP_SIZE);
}

template <typename T> struct alignas(8) GdnConvWidth4 {
  T values[GDN_PACKED_CONV_WIDTH];
};

template <typename T>
__device__ __forceinline__ T gdn_conv_width4_update(
    T input, const GdnConvWidth4<T> &weights, GdnConvWidth4<T> *state) {
  GdnConvWidth4<T> values = *state;

#pragma unroll
  for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
    values.values[i] = values.values[i + 1];
  }
  values.values[GDN_PACKED_CONV_WIDTH - 1] = input;
  *state = values;

  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < GDN_PACKED_CONV_WIDTH; i++) {
    acc += (float)values.values[i] * (float)weights.values[i];
  }
  const float sig = 1.0f / (1.0f + expf(-acc));
  return (T)(acc * sig);
}

template <typename T>
__device__ __forceinline__ float causal_conv1d_width4_load(
    const T *__restrict__ x, const T *__restrict__ state, int pos,
    size_t x_batch_offset, int64_t x_stride_s, int64_t x_stride_c, int ch) {
  if (pos >= 0) {
    return (float)x[x_batch_offset + (size_t)pos * x_stride_s +
                    (size_t)ch * x_stride_c];
  }
  return (float)state[GDN_PACKED_CONV_WIDTH + pos];
}

__device__ __forceinline__ float gdn_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <int VALUES_PER_WARP>
__device__ __forceinline__ float gdn_grouped_k_sum(float value) {
#pragma unroll
  for (int offset = 16; offset >= VALUES_PER_WARP; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ void gdn_cp_async_cg_16(void *dst,
                                                   const void *src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t dst_smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               : : "r"(dst_smem), "l"(src));
#else
  *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(src);
#endif
}

__device__ __forceinline__ void gdn_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" : :);
#endif
}

__device__ __forceinline__ void gdn_cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}

__device__ __forceinline__ float gdn_silu(float x) {
  if (isnan(x)) {
    return x;
  }
  if (isinf(x)) {
    return x > 0.0f ? x : 0.0f;
  }
  if (x >= 0.0f) {
    return x / (1.0f + expf(-x));
  }
  const float ex = expf(x);
  return x * ex / (1.0f + ex);
}

__device__ __forceinline__ float gdn_warp_max(float value, int width = 32) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset, width));
  }
  return value;
}

__device__ __forceinline__ size_t gdn_fp8_scale_offset(
    int normalized_row, int groups, int scale_stride_m) {
  const int projection_row = normalized_row / groups;
  const int group = normalized_row - projection_row * groups;
  return (size_t)group * scale_stride_m + projection_row;
}

__device__ __forceinline__ void gdn_fp8_quant_params(
    float maximum, int scale_layout, float &quant_scale,
    float &inverse_quant_scale) {
  if (scale_layout == 0) {
    inverse_quant_scale = fmaxf(maximum, 1.0e-10f) / 448.0f;
    quant_scale = 1.0f / inverse_quant_scale;
    return;
  }
  quant_scale = maximum == 0.0f ? 1.0f : 448.0f / maximum;
  inverse_quant_scale = 1.0f / quant_scale;
}

__device__ __forceinline__ gdn_fp8_e4m3 gdn_fp8_quantize(
    float value, float quant_scale, float inverse_quant_scale,
    int scale_layout) {
  const float scaled = scale_layout == 0
                           ? value / inverse_quant_scale
                           : value * quant_scale;
  return gdn_fp8_e4m3(fminf(fmaxf(scaled, -448.0f), 448.0f));
}

template <typename T> struct alignas(16) gdn_rmsnorm_vec8 {
  T data[GDN_RMSNORM_TILED_VALUES_PER_LANE];
};

constexpr int GDN_SPEC_COMMIT_WARPS = 4;

constexpr int GDN_SPEC_COMMIT_MAX_K = 256;

constexpr int GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH = 16;

constexpr int GDN_SPEC_CHECKPOINT_MAX_K = 256;

constexpr int GDN_SPEC_CHECKPOINT_VALUE_TILE = 32;

constexpr int GDN_SPEC_CHECKPOINT_WARPS = 4;

constexpr int GDN_SPEC_CHECKPOINT_VALUES_PER_WARP =
    GDN_SPEC_CHECKPOINT_VALUE_TILE / GDN_SPEC_CHECKPOINT_WARPS;

constexpr int GDN_SPEC_FUSED_MAX_TOKENS = 8;

constexpr int GDN_SPEC_FUSED_THREADS = 256;

constexpr int GDN_SPEC_FUSED_WARPS = GDN_SPEC_FUSED_THREADS / 32;

constexpr int GDN_SPEC_FUSED_VALUE_CHUNK = 32;

constexpr int GDN_SPEC_FUSED_VALUE_CHUNKS =
    GDN_DECODE_VALUE_MAJOR_V / GDN_SPEC_FUSED_VALUE_CHUNK;

constexpr int GDN_SPEC_FUSED_VALUES_PER_WARP =
    GDN_SPEC_FUSED_VALUE_CHUNK / GDN_SPEC_FUSED_WARPS;

// Paired reductions favor 16-bit states and underfilled or saturated F32 grids; serial sustains mid-grid bandwidth.
constexpr int GDN_SPEC_FUSED_PAIR_LOW_GRID_MAX = 144;

constexpr int GDN_SPEC_FUSED_PAIR_HIGH_GRID_MIN = 768;

constexpr uint32_t GDN_SPEC_CHECKPOINT_PAD_SLOT = 0xffffffffu;

constexpr int GDN_DEFERRED_STATE_DEPTH = 4;

__device__ __forceinline__ size_t
gdn_spec_checkpoint_base(uint32_t active_slot, int checkpoint_lanes) {
  return ((size_t)active_slot / checkpoint_lanes) * checkpoint_lanes;
}

__device__ __forceinline__ float gdn_spec_softplus(float value) {
  return value > 20.0f
             ? value
             : (value > 0.0f ? value + log1pf(expf(-value))
                             : log1pf(expf(value)));
}

__device__ __forceinline__ float2 gdn_spec_warp_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset);
    y += __shfl_xor_sync(0xffffffff, y, offset);
  }
  return make_float2(x, y);
}

template <typename StateT>
__device__ __forceinline__ void gdn_spec_copy_state_chunk(
    StateT *shared_state, const StateT *state, int chunk, int thread) {
  constexpr int ELEMENTS_PER_COPY = 16 / sizeof(StateT);
  constexpr int ELEMENTS_PER_CHUNK =
      GDN_SPEC_FUSED_VALUE_CHUNK * GDN_DECODE_VALUE_MAJOR_K;
  constexpr int COPIES_PER_CHUNK = ELEMENTS_PER_CHUNK / ELEMENTS_PER_COPY;
  for (int copy = thread; copy < COPIES_PER_CHUNK;
       copy += GDN_SPEC_FUSED_THREADS) {
    const int element = copy * ELEMENTS_PER_COPY;
    gdn_cp_async_cg_16(shared_state + element,
                       state + chunk * ELEMENTS_PER_CHUNK + element);
  }
  gdn_cp_async_commit();
}

constexpr int GDN_TRANSITION_CONV_INPUT = 0;

constexpr int GDN_TRANSITION_KEY = 1;

constexpr int GDN_TRANSITION_DELTA = 2;

constexpr int GDN_TRANSITION_DECAY = 3;

constexpr int GDN_TRANSITION_CONV_STATE = 4;

constexpr int GDN_TRANSITION_RECURRENT_STATE = 5;

constexpr int GDN_TRANSITION_STAGE_SRC_CONV = 0;

constexpr int GDN_TRANSITION_STAGE_SRC_KEY = 1;

constexpr int GDN_TRANSITION_STAGE_SRC_DELTA = 2;

constexpr int GDN_TRANSITION_STAGE_SRC_DECAY = 3;

constexpr int GDN_TRANSITION_STAGE_DST_CONV = 4;

constexpr int GDN_TRANSITION_STAGE_DST_KEY = 5;

constexpr int GDN_TRANSITION_STAGE_DST_DELTA = 6;

constexpr int GDN_TRANSITION_STAGE_DST_DECAY = 7;

constexpr int GDN_TRANSITION_STAGE_DST_KEEP = 8;

constexpr int GDN_TRANSITION_STAGE_DST_EPOCH = 9;

constexpr int GDN_TRANSITION_STAGE_COPY_BLOCKS = 4;

constexpr int GDN_TRANSITION_PUBLISH_KEEP = 0;

constexpr int GDN_TRANSITION_PUBLISH_EPOCH = 1;

constexpr int GDN_TRANSITION_PUBLISH_KEY_BANK = 2;

constexpr int GDN_TRANSITION_APPLY_PENDING_CONV = 0;

constexpr int GDN_TRANSITION_APPLY_PENDING_KEY_BANKS = 1;

constexpr int GDN_TRANSITION_APPLY_PENDING_KEY_BANK = 2;

constexpr int GDN_TRANSITION_APPLY_PENDING_DELTA = 3;

constexpr int GDN_TRANSITION_APPLY_PENDING_DECAY = 4;

constexpr int GDN_TRANSITION_APPLY_PENDING_KEEP = 5;

constexpr int GDN_TRANSITION_APPLY_PENDING_EPOCH = 6;

constexpr int GDN_TRANSITION_APPLY_CONV_EPOCH = 7;

constexpr int GDN_TRANSITION_APPLY_RECURRENT_EPOCH = 8;

constexpr int GDN_TRANSITION_APPLY_CONV_STATE = 9;

constexpr int GDN_TRANSITION_APPLY_RECURRENT_STATE = 10;

constexpr int GDN_CHUNKED_BT = 64;
constexpr int GDN_CHUNKED_BV = 64;

template <typename StateT>
using GdnChunkedKernel = void (*)(const float *, const float *, const float *,
                                  const float *, const float *, StateT *,
                                  float *, int, int, const int32_t *, int);

// Each chunked configuration costs minutes of cicc, so each is explicitly instantiated in its own TU in gdn_chunked/.
template <typename StateT, int BT, int BK, int BV, bool VALUE_MAJOR>
GdnChunkedKernel<StateT> gdn_chunked_kernel();
