// Qwen3.8-Flash-Next's hyper-connection reads and writes, each fused into one launch (see cuda/hyper.rs).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kHyperThreads = 256;
constexpr int kWarp = 32;

__device__ __forceinline__ float load_f(const float *p) { return *p; }
__device__ __forceinline__ float load_f(const __nv_bfloat16 *p) {
  return __bfloat162float(*p);
}
__device__ __forceinline__ void store_f(float *p, float v) { *p = v; }
__device__ __forceinline__ void store_f(__nv_bfloat16 *p, float v) {
  *p = __float2bfloat16(v);
}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

// One block per (group, row): each `group`-wide slice RMS-normed, times its
// element's scale. Each thread keeps its values between the sum and the write,
// so the slice is read once.
constexpr int kNormThreads = 1024;
constexpr int kNormPerThread = 4;

// the sum over a block of kNormThreads, returned to every thread
__device__ float norm_block_sum(float v) {
  __shared__ float partial[kNormThreads / kWarp];
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  if (threadIdx.x % kWarp == 0) {
    partial[threadIdx.x / kWarp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < kNormThreads / kWarp; ++w) {
    total += partial[w];
  }
  return total;
}

template <typename T>
__global__ void __launch_bounds__(kNormThreads)
    hc_group_norm_kernel(const T *__restrict__ x, const T *__restrict__ scale,
                         T *__restrict__ out, int width, int group,
                         float eps) {
  const int64_t base = (int64_t)blockIdx.y * width + (int64_t)blockIdx.x * group;
  const int scale_base = blockIdx.x * group;
  float v[kNormPerThread];
  float ss = 0.0f;
#pragma unroll
  for (int k = 0; k < kNormPerThread; ++k) {
    const int i = threadIdx.x + k * kNormThreads;
    v[k] = i < group ? load_f(x + base + i) : 0.0f;
    ss += v[k] * v[k];
  }
  const float inv = rsqrtf(norm_block_sum(ss) / group + eps);
#pragma unroll
  for (int k = 0; k < kNormPerThread; ++k) {
    const int i = threadIdx.x + k * kNormThreads;
    if (i < group) {
      store_f(out + base + i, v[k] * inv * load_f(scale + scale_base + i));
    }
  }
}

// The block's input: the mean over streams of sigmoid(mix) * normed, `mix` and
// `normed` being (rows, streams * hidden), the output (rows, hidden).
template <typename T>
__global__ void hc_mix_kernel(const T *__restrict__ mix,
                              const T *__restrict__ normed, T *__restrict__ out,
                              int streams, int hidden) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hidden) {
    return;
  }
  const int64_t row = (int64_t)blockIdx.y * streams * hidden;
  float acc = 0.0f;
  for (int s = 0; s < streams; ++s) {
    const int64_t at = row + (int64_t)s * hidden + h;
    acc += sigmoidf(load_f(mix + at)) * load_f(normed + at);
  }
  store_f(out + (int64_t)blockIdx.y * hidden + h, acc / streams);
}

// silu(x * scale) elementwise: the low-rank mix's activation, and with scale
// 1 / streams.
template <typename T>
__global__ void hc_scaled_silu_kernel(const T *__restrict__ x,
                                      T *__restrict__ out, int64_t n,
                                      float scale) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = load_f(x + i) * scale;
    store_f(out + i, v / (1.0f + __expf(-v)));
  }
}

// 2 sigmoid(x * scale) elementwise: the write-back weights.
template <typename T>
__global__ void hc_inject_weights_kernel(const T *__restrict__ x,
                                         T *__restrict__ out, int64_t n,
                                         float scale) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    store_f(out + i, 2.0f * sigmoidf(load_f(x + i) * scale));
  }
}

// streams + output * inject[s] into every stream: streams (rows, streams *
// hidden), output (rows, hidden), inject (rows, streams).
template <typename T>
__global__ void hc_inject_add_kernel(const T *__restrict__ streams_in,
                                     const T *__restrict__ output,
                                     const T *__restrict__ inject,
                                     T *__restrict__ out, int streams,
                                     int hidden, int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int64_t width = (int64_t)streams * hidden;
  const int64_t row = i / width;
  const int64_t col = i - row * width;
  const int s = (int)(col / hidden);
  const int h = (int)(col - (int64_t)s * hidden);
  store_f(out + i, load_f(streams_in + i) +
                       load_f(output + row * hidden + h) *
                           load_f(inject + row * streams + s));
}

int blocks_for(int64_t n) {
  return (int)((n + kHyperThreads - 1) / kHyperThreads);
}

} // namespace

#define HYPER_ENTRIES(tag, T)                                                  \
  extern "C" int hc_group_norm_##tag(const void *x, const void *scale,         \
                                     void *out, int rows, int width,           \
                                     int group, float eps, int64_t stream) {   \
    if (group > kNormThreads * kNormPerThread) {                               \
      return (int)cudaErrorInvalidValue;                                       \
    }                                                                          \
    hc_group_norm_kernel<T><<<dim3(width / group, rows), kNormThreads, 0,      \
                              reinterpret_cast<cudaStream_t>(stream)>>>(       \
        (const T *)x, (const T *)scale, (T *)out, width, group, eps);          \
    return (int)cudaGetLastError();                                            \
  }                                                                            \
  extern "C" int hc_mix_##tag(const void *mix, const void *normed, void *out,  \
                              int rows, int streams, int hidden,               \
                              int64_t stream) {                                \
    hc_mix_kernel<T><<<dim3(blocks_for(hidden), rows), kHyperThreads, 0,       \
                       reinterpret_cast<cudaStream_t>(stream)>>>(              \
        (const T *)mix, (const T *)normed, (T *)out, streams, hidden);         \
    return (int)cudaGetLastError();                                            \
  }                                                                            \
  extern "C" int hc_scaled_silu_##tag(const void *x, void *out, int64_t n,     \
                                      float scale, int64_t stream) {           \
    hc_scaled_silu_kernel<T><<<blocks_for(n), kHyperThreads, 0,                \
                               reinterpret_cast<cudaStream_t>(stream)>>>(      \
        (const T *)x, (T *)out, n, scale);                                     \
    return (int)cudaGetLastError();                                            \
  }                                                                            \
  extern "C" int hc_inject_weights_##tag(const void *x, void *out, int64_t n,  \
                                         float scale, int64_t stream) {        \
    hc_inject_weights_kernel<T><<<blocks_for(n), kHyperThreads, 0,             \
                                  reinterpret_cast<cudaStream_t>(stream)>>>(   \
        (const T *)x, (T *)out, n, scale);                                     \
    return (int)cudaGetLastError();                                            \
  }                                                                            \
  extern "C" int hc_inject_add_##tag(const void *streams_in,                   \
                                     const void *output, const void *inject,   \
                                     void *out, int rows, int streams,         \
                                     int hidden, int64_t stream) {             \
    const int64_t n = (int64_t)rows * streams * hidden;                        \
    hc_inject_add_kernel<T><<<blocks_for(n), kHyperThreads, 0,                 \
                              reinterpret_cast<cudaStream_t>(stream)>>>(       \
        (const T *)streams_in, (const T *)output, (const T *)inject, (T *)out, \
        streams, hidden, n);                                                   \
    return (int)cudaGetLastError();                                            \
  }

HYPER_ENTRIES(bf16, __nv_bfloat16)
HYPER_ENTRIES(f32, float)
