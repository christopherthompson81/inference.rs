// What an offloaded MoE layer's decode step hands the CPU, prepared in one launch: each pick's cache slot from the
// mapped slot table, written for the cached experts' kernel, and the picks, slots and f32 activations written straight
// into the mapped buffers the host function reads.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define PREPARE_THREADS 256

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__global__ void moe_split_prepare_kernel(const uint32_t *__restrict__ ids, const float *__restrict__ weights,
                                         const T *__restrict__ xs, const uint32_t *table, uint32_t *__restrict__ slots,
                                         uint32_t *host_ids, float *host_weights, uint32_t *host_slots,
                                         float *host_xs, int picks, int values) {
  for (int i = threadIdx.x; i < picks; i += blockDim.x) {
    const uint32_t id = ids[i];
    const uint32_t slot = table[id];
    slots[i] = slot;
    host_ids[i] = id;
    host_weights[i] = weights[i];
    host_slots[i] = slot;
  }
  for (int i = threadIdx.x; i < values; i += blockDim.x) {
    host_xs[i] = to_f32(xs[i]);
  }
}

extern "C" int moe_split_prepare(const void *ids, const void *weights, const void *xs, int xs_bf16,
                                 const void *table, void *slots, void *host_ids, void *host_weights,
                                 void *host_slots, void *host_xs, int picks, int values, int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const uint32_t *i = static_cast<const uint32_t *>(ids);
  const float *w = static_cast<const float *>(weights);
  const uint32_t *t = static_cast<const uint32_t *>(table);
  uint32_t *out = static_cast<uint32_t *>(slots);
  uint32_t *hi = static_cast<uint32_t *>(host_ids);
  float *hw = static_cast<float *>(host_weights);
  uint32_t *hs = static_cast<uint32_t *>(host_slots);
  float *hx = static_cast<float *>(host_xs);
  if (xs_bf16) {
    moe_split_prepare_kernel<__nv_bfloat16><<<1, PREPARE_THREADS, 0, s>>>(
        i, w, static_cast<const __nv_bfloat16 *>(xs), t, out, hi, hw, hs, hx, picks, values);
  } else {
    moe_split_prepare_kernel<float>
        <<<1, PREPARE_THREADS, 0, s>>>(i, w, static_cast<const float *>(xs), t, out, hi, hw, hs, hx, picks, values);
  }
  return static_cast<int>(cudaGetLastError());
}
