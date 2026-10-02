#undef __CUDA_FP8_TYPES_EXIST__
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

namespace vllm {

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#else
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
#endif
  return tmp.u16[0];
}

inline __device__ float half_to_float(uint16_t h) {
  float f;
#ifndef USE_ROCM
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
#else
  asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
#endif
  return f;
}

inline __device__ void from_float(half &dst, float src) {
  dst = static_cast<half>(float_to_half(src));
}

inline __device__ void from_float(__nv_bfloat16 &dst, float src) {
  dst = __float2bfloat16(src);
}

inline __device__ float to_float(half u) {
  return half_to_float(static_cast<uint16_t>(u));
}

inline __device__ float to_float(__nv_bfloat16 u) {
  return __bfloat162float(u);
}

} // namespace vllm

#define ASSERT_THROW(cond, msg)                                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      throw std::runtime_error(msg);                                           \
    }                                                                          \
  } while (0)

// offsets[e] is the lower bound of e in the sorted ids: no host sync, temp buffer or thrust scan, so graph-capture safe
static __global__ void expert_offsets_kernel(const int32_t *expert_ids,
                                             int size_m,
                                             int32_t *expert_offsets,
                                             int num_experts) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e > num_experts) {
    return;
  }
  int lo = 0;
  int hi = size_m;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (expert_ids[mid] < e) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  expert_offsets[e] = lo;
}

/**
 * @brief Calculates expert offsets array on the GPU.
 *
 * @param d_expert_ids     Device pointer to sorted expert IDs [size_m].
 * @param size_m           Total number of tokens.
 * @param d_expert_offsets Device pointer for output offsets [num_experts + 1].
 * @param num_experts      Number of experts.
 * @param stream           CUDA stream.
 */
static void calculate_expert_offsets(const int32_t *d_expert_ids, int size_m,
                                     int32_t *d_expert_offsets, int num_experts,
                                     cudaStream_t stream) {
  int threads = 256;
  int blocks = (num_experts + 1 + threads - 1) / threads;
  expert_offsets_kernel<<<blocks, threads, 0, stream>>>(
      d_expert_ids, size_m, d_expert_offsets, num_experts);
}
