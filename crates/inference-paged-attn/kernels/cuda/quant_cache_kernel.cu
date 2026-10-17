// Quantized HND paged KV cache ([blocks, kv_heads, block_size, row_bytes] bytes): each row is rotated by Hadamard
// transforms `width` values wide (a power of two, 32..512), then every group of 32 values is scaled by its absmax and
// quantized to `bits` (2..8) on a midpoint grid. A row holds head_size*bits/8 data bytes, 8 consecutive values per
// `bits` bytes, then one f16 scale per group.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

namespace inference_kv_quant {

constexpr int GROUP = 32;
constexpr int VALUES_PER_CHUNK = 8;
constexpr int MAX_WARPS = 8;
// keeps an all-zero group's scale positive, so its values come back as (near) zero
constexpr float MIN_SCALE = 1e-10f;

template <typename T> __device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_float(float v) { return static_cast<T>(v); }
template <> __device__ __forceinline__ __half from_float(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }

// The normalized Hadamard transform of the warp's 32*N values, value j of each lane at index lane + 32*j; it is its
// own inverse. Its stages commute: strides below 32 cross lanes, the rest pair a lane's own values.
template <int N> __device__ __forceinline__ void hadamard(float (&v)[N]) {
  const int lane = threadIdx.x & (GROUP - 1);
#pragma unroll
  for (int s = 1; s < GROUP; s <<= 1) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const float o = __shfl_xor_sync(0xffffffffu, v[j], s);
      v[j] = (lane & s) ? o - v[j] : v[j] + o;
    }
  }
#pragma unroll
  for (int t = 1; t < N; t <<= 1) {
#pragma unroll
    for (int j = 0; j < N; ++j) {
      if (!(j & t)) {
        const float a = v[j], b = v[j | t];
        v[j] = a + b;
        v[j | t] = a - b;
      }
    }
  }
  const float norm = rsqrtf(float(GROUP * N));
#pragma unroll
  for (int j = 0; j < N; ++j) {
    v[j] *= norm;
  }
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, s));
  }
  return v;
}

__device__ __forceinline__ int64_t row_offset(int64_t block, int head, int64_t slot, int num_heads, int block_size,
                                              int row_bytes) {
  return ((block * num_heads + head) * block_size + slot) * row_bytes;
}

// Quantizes one rotated group (this warp's 32 lanes) into the row at `row`.
__device__ __forceinline__ void quantize_group(float x, uint8_t *row, int group, int bits, int data_bytes) {
  const int lane = threadIdx.x & (GROUP - 1);
  const float scale = fmaxf(warp_max(fabsf(x)), MIN_SCALE);
  const float mid = float(1 << (bits - 1));
  const int qmax = (1 << bits) - 1;
  const int q = max(0, min(qmax, __float2int_rd(fmaf(x / scale, mid, mid))));
  uint64_t packed = uint64_t(q) << ((lane % VALUES_PER_CHUNK) * bits);
#pragma unroll
  for (int s = 1; s < VALUES_PER_CHUNK; s <<= 1) {
    packed |= __shfl_xor_sync(0xffffffffu, packed, s);
  }
  if (lane % VALUES_PER_CHUNK == 0) {
    uint8_t *dst = row + (group * (GROUP / VALUES_PER_CHUNK) + lane / VALUES_PER_CHUNK) * bits;
    for (int b = 0; b < bits; ++b) {
      dst[b] = uint8_t(packed >> (8 * b));
    }
  }
  if (lane == 0) {
    reinterpret_cast<__half *>(row + data_bytes)[group] = __float2half_rn(scale);
  }
}

// This lane's (still rotated) value of group `group` in the row at `row`.
__device__ __forceinline__ float dequantize_group(const uint8_t *row, int group, int bits, int data_bytes) {
  const int lane = threadIdx.x & (GROUP - 1);
  const uint8_t *src = row + (group * (GROUP / VALUES_PER_CHUNK) + lane / VALUES_PER_CHUNK) * bits;
  uint64_t packed = 0;
  for (int b = 0; b < bits; ++b) {
    packed |= uint64_t(src[b]) << (8 * b);
  }
  const int q = int((packed >> ((lane % VALUES_PER_CHUNK) * bits)) & ((1u << bits) - 1));
  const float mid = float(1 << (bits - 1));
  const float scale = __half2float(reinterpret_cast<const __half *>(row + data_bytes)[group]);
  return (float(q) + 0.5f - mid) / mid * scale;
}

// One cache's rows (K or V) of `head_size` values: each warp rotates and quantizes rotation blocks of 32*N values.
template <typename T, int N>
__global__ void reshape_and_cache_quant_kernel(const T *__restrict__ src, uint8_t *__restrict__ cache,
                                               const int64_t *__restrict__ slot_mapping, int num_heads, int head_size,
                                               int block_size, int stride, int bits) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  const int data_bytes = head_size * bits / 8;
  uint8_t *row =
      cache + row_offset(slot / block_size, head, slot % block_size, num_heads, block_size, data_bytes + head_size / 16);
  const T *values = src + int64_t(token) * stride + head * head_size;
  const int lane = threadIdx.x & (GROUP - 1);
  for (int r = threadIdx.x / GROUP; r < head_size / (GROUP * N); r += blockDim.x / GROUP) {
    float v[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
      v[j] = to_float(values[r * GROUP * N + lane + GROUP * j]);
    }
    hadamard<N>(v);
#pragma unroll
    for (int j = 0; j < N; ++j) {
      quantize_group(v[j], row, r * N + j, bits, data_bytes);
    }
  }
}

// One cache's rows dequantized and rotated back into `[tokens, kv_heads, head_size]`.
template <typename T, int N>
__global__ void gather_kv_cache_quant_kernel(const uint8_t *__restrict__ cache, T *__restrict__ out,
                                             const int32_t *__restrict__ block_table,
                                             const int32_t *__restrict__ cu_seq_lens, int num_seqs, int block_size,
                                             int block_table_stride, int num_kv_heads, int head_size, int bits) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x & (GROUP - 1);
  T *dst = out + (int64_t(token) * num_kv_heads + head) * head_size;
  const int rotations = head_size / (GROUP * N);
  if (token >= cu_seq_lens[num_seqs]) {
    for (int r = threadIdx.x / GROUP; r < rotations; r += blockDim.x / GROUP) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        dst[r * GROUP * N + lane + GROUP * j] = from_float<T>(0.0f);
      }
    }
    return;
  }
  int seq = 0;
  while (seq + 1 < num_seqs && cu_seq_lens[seq + 1] <= token) {
    seq++;
  }
  const int position = token - cu_seq_lens[seq];
  const int64_t block = block_table[seq * block_table_stride + position / block_size];
  const int data_bytes = head_size * bits / 8;
  const uint8_t *row =
      cache + row_offset(block, head, position % block_size, num_kv_heads, block_size, data_bytes + head_size / 16);
  for (int r = threadIdx.x / GROUP; r < rotations; r += blockDim.x / GROUP) {
    float v[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
      v[j] = dequantize_group(row, r * N + j, bits, data_bytes);
    }
    hadamard<N>(v);
#pragma unroll
    for (int j = 0; j < N; ++j) {
      dst[r * GROUP * N + lane + GROUP * j] = from_float<T>(v[j]);
    }
  }
}

// Rotates contiguous rows' last dim in blocks of 32*N values; `blocks` is the count of such blocks.
template <typename T, int N>
__global__ void hadamard_rows_kernel(const T *__restrict__ src, T *__restrict__ dst, int64_t blocks) {
  const int64_t r = int64_t(blockIdx.x) * (blockDim.x / GROUP) + threadIdx.x / GROUP;
  if (r >= blocks) {
    return;
  }
  const int lane = threadIdx.x & (GROUP - 1);
  float v[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    v[j] = to_float(src[r * GROUP * N + lane + GROUP * j]);
  }
  hadamard<N>(v);
#pragma unroll
  for (int j = 0; j < N; ++j) {
    dst[r * GROUP * N + lane + GROUP * j] = from_float<T>(v[j]);
  }
}

inline dim3 rotation_block(int head_size, int width) { return dim3(GROUP * min(head_size / width, MAX_WARPS)); }

// Calls f with the values per lane of a rotation `width` wide as a compile-time constant.
template <typename F> bool with_width(int width, F f) {
  switch (width) {
  case 32: f(std::integral_constant<int, 1>{}); return true;
  case 64: f(std::integral_constant<int, 2>{}); return true;
  case 128: f(std::integral_constant<int, 4>{}); return true;
  case 256: f(std::integral_constant<int, 8>{}); return true;
  case 512: f(std::integral_constant<int, 16>{}); return true;
  default: return false;
  }
}

// Calls f with a typed null pointer standing for dtype code 0 (f16), 1 (bf16) or 2 (f32), returning what it does.
template <typename F> bool with_dtype(uint32_t dtype, F f) {
  switch (dtype) {
  case 0: return f(static_cast<__half *>(nullptr));
  case 1: return f(static_cast<__nv_bfloat16 *>(nullptr));
  case 2: return f(static_cast<float *>(nullptr));
  default: return false;
  }
}

} // namespace inference_kv_quant

using namespace inference_kv_quant;

// Quantizes `src` ([tokens, num_heads, head_size] rows `stride` apart) into one quantized cache at `slot_mapping`.
extern "C" void reshape_and_cache_quant(const void *src, void *cache, const int64_t *slot_mapping, int32_t num_tokens,
                                        int32_t num_heads, int32_t head_size, int32_t block_size, int32_t stride,
                                        int32_t bits, int32_t width, uint32_t dtype, cudaStream_t stream) {
  const dim3 grid(num_tokens, num_heads);
  const dim3 block = rotation_block(head_size, width);
  const bool ok = with_dtype(dtype, [&](auto type) {
    using T = std::remove_pointer_t<decltype(type)>;
    return with_width(width, [&](auto n) {
      reshape_and_cache_quant_kernel<T, decltype(n)::value><<<grid, block, 0, stream>>>(
          static_cast<const T *>(src), static_cast<uint8_t *>(cache), slot_mapping, num_heads, head_size, block_size,
          stride, bits);
    });
  });
  if (!ok) {
    fprintf(stderr, "reshape_and_cache_quant received unsupported dtype %u or width %d\n", dtype, width);
  }
}

extern "C" void gather_kv_cache_quant(const void *cache, void *out, const int32_t *block_table,
                                      const int32_t *cu_seq_lens, int32_t num_tokens, int32_t num_seqs,
                                      int32_t block_size, int32_t block_table_stride, int32_t num_kv_heads,
                                      int32_t head_size, int32_t bits, int32_t width, uint32_t out_dtype,
                                      cudaStream_t stream) {
  const dim3 grid(num_tokens, num_kv_heads);
  const dim3 block = rotation_block(head_size, width);
  const bool ok = with_dtype(out_dtype, [&](auto type) {
    using T = std::remove_pointer_t<decltype(type)>;
    return with_width(width, [&](auto n) {
      gather_kv_cache_quant_kernel<T, decltype(n)::value><<<grid, block, 0, stream>>>(
          static_cast<const uint8_t *>(cache), static_cast<T *>(out), block_table, cu_seq_lens, num_seqs, block_size,
          block_table_stride, num_kv_heads, head_size, bits);
    });
  });
  if (!ok) {
    fprintf(stderr, "gather_kv_cache_quant received unsupported dtype %u or width %d\n", out_dtype, width);
  }
}

extern "C" void hadamard_rows(const void *src, void *dst, int64_t elements, int32_t width, uint32_t dtype,
                              cudaStream_t stream) {
  const int64_t blocks = elements / width;
  if (blocks == 0) {
    return;
  }
  const dim3 grid(unsigned((blocks + MAX_WARPS - 1) / MAX_WARPS));
  const dim3 block(GROUP * MAX_WARPS);
  const bool ok = with_dtype(dtype, [&](auto type) {
    using T = std::remove_pointer_t<decltype(type)>;
    return with_width(width, [&](auto n) {
      hadamard_rows_kernel<T, decltype(n)::value>
          <<<grid, block, 0, stream>>>(static_cast<const T *>(src), static_cast<T *>(dst), blocks);
    });
  });
  if (!ok) {
    fprintf(stderr, "hadamard_rows received unsupported dtype %u or width %d\n", dtype, width);
  }
}
