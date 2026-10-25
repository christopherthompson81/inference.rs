// Fused row ops of diffusion transformers over a bf16 residual stream, computed in f32 as the reference's mixed
// precision does: LayerNorm with an f32 affine and adaLN modulation, the gated residual, and full-width QK RMSNorm with
// an interleaved-pair RoPE written head-major.
#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr int kDitThreads = 256;
constexpr int kDitWarp = 32;

__device__ __forceinline__ float dit_block_sum(float value, float *scratch) {
  for (int offset = kDitWarp / 2; offset > 0; offset /= 2)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  const int warp = threadIdx.x / kDitWarp, lane = threadIdx.x % kDitWarp;
  if (lane == 0)
    scratch[warp] = value;
  __syncthreads();
  value = lane < blockDim.x / kDitWarp ? scratch[lane] : 0.f;
  for (int offset = kDitWarp / 2; offset > 0; offset /= 2)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  __syncthreads();
  return value;
}

// one block per row: ((x - mean) / std * weight + bias) * (1 + scale) + shift, each of the four optional
__global__ void dit_layer_norm_kernel(const __nv_bfloat16 *xs, const float *weight, const float *bias,
                                      const float *scale, const float *shift, __nv_bfloat16 *out, int cols,
                                      float eps) {
  __shared__ float scratch[kDitWarp];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  const __nv_bfloat16 *x = xs + base;
  float sum = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x)
    sum += __bfloat162float(x[i]);
  const float mean = dit_block_sum(sum, scratch) / cols;
  float squares = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float d = __bfloat162float(x[i]) - mean;
    squares += d * d;
  }
  const float inv_std = rsqrtf(dit_block_sum(squares, scratch) / cols + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    float v = (__bfloat162float(x[i]) - mean) * inv_std;
    if (weight != nullptr)
      v *= weight[i];
    if (bias != nullptr)
      v += bias[i];
    if (scale != nullptr)
      v *= 1.f + scale[i];
    if (shift != nullptr)
      v += shift[i];
    out[base + i] = __float2bfloat16(v);
  }
}

// x + y * gate, gate one value per column
__global__ void dit_gated_residual_kernel(const __nv_bfloat16 *xs, const __nv_bfloat16 *ys, const float *gate,
                                          __nv_bfloat16 *out, int64_t n, int cols) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = __float2bfloat16(__bfloat162float(xs[i]) + __bfloat162float(ys[i]) * gate[i % cols]);
  }
}

// one block per token of xs (batch * tokens, heads * head_dim): RMSNorm over the whole row times weight, rounded to
// bf16, then (with cos/sin, each (tokens, head_dim / 2)) adjacent pairs rotated in f32; out is (batch, heads, tokens,
// head_dim)
__global__ void dit_rms_norm_rope_kernel(const __nv_bfloat16 *xs, const float *weight, const float *cos,
                                         const float *sin, __nv_bfloat16 *out, int tokens, int heads, int head_dim,
                                         float eps) {
  __shared__ float scratch[kDitWarp];
  const int row = blockIdx.x;
  const int b = row / tokens, t = row % tokens;
  const int cols = heads * head_dim;
  const __nv_bfloat16 *x = xs + static_cast<int64_t>(row) * cols;
  float squares = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = __bfloat162float(x[i]);
    squares += v * v;
  }
  const float rms = sqrtf(dit_block_sum(squares, scratch) / cols + eps);
  const int half = head_dim / 2;
  for (int p = threadIdx.x; p < cols / 2; p += blockDim.x) {
    const int i = 2 * p, head = i / head_dim, k = i % head_dim;
    float v0 = __bfloat162float(__float2bfloat16(__bfloat162float(x[i]) / rms * weight[i]));
    float v1 = __bfloat162float(__float2bfloat16(__bfloat162float(x[i + 1]) / rms * weight[i + 1]));
    if (cos != nullptr) {
      const int64_t r = static_cast<int64_t>(t) * half + k / 2;
      const float c = cos[r], s = sin[r];
      const float r0 = v0 * c - v1 * s, r1 = v0 * s + v1 * c;
      v0 = r0;
      v1 = r1;
    }
    __nv_bfloat16 *o = out + ((static_cast<int64_t>(b) * heads + head) * tokens + t) * head_dim + k;
    o[0] = __float2bfloat16(v0);
    o[1] = __float2bfloat16(v1);
  }
}

extern "C" int dit_layer_norm_bf16(const void *xs, const void *weight, const void *bias, const void *scale,
                                   const void *shift, void *out, int rows, int cols, float eps, int64_t stream) {
  dit_layer_norm_kernel<<<rows, kDitThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(xs), static_cast<const float *>(weight), static_cast<const float *>(bias),
      static_cast<const float *>(scale), static_cast<const float *>(shift), static_cast<__nv_bfloat16 *>(out), cols,
      eps);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int dit_gated_residual_bf16(const void *xs, const void *ys, const void *gate, void *out, int64_t n,
                                       int cols, int64_t stream) {
  constexpr int64_t kMaxBlocks = 4096;
  const int64_t blocks = std::min<int64_t>((n + kDitThreads - 1) / kDitThreads, kMaxBlocks);
  dit_gated_residual_kernel<<<static_cast<int>(blocks), kDitThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(xs), static_cast<const __nv_bfloat16 *>(ys), static_cast<const float *>(gate),
      static_cast<__nv_bfloat16 *>(out), n, cols);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int dit_rms_norm_rope_bf16(const void *xs, const void *weight, const void *cos, const void *sin, void *out,
                                      int batch, int tokens, int heads, int head_dim, float eps, int64_t stream) {
  dit_rms_norm_rope_kernel<<<batch * tokens, kDitThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(xs), static_cast<const float *>(weight), static_cast<const float *>(cos),
      static_cast<const float *>(sin), static_cast<__nv_bfloat16 *>(out), tokens, heads, head_dim, eps);
  return static_cast<int>(cudaGetLastError());
}
