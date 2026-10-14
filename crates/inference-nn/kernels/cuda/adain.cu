#include <cstdint>
#include <cuda_runtime.h>

constexpr int kAdainThreads = 256;
constexpr int kWarp = 32;

__device__ __forceinline__ float adain_block_sum(float value, float *scratch) {
  for (int offset = kWarp / 2; offset > 0; offset /= 2)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
  if (lane == 0)
    scratch[warp] = value;
  __syncthreads();
  value = lane < blockDim.x / kWarp ? scratch[lane] : 0.f;
  for (int offset = kWarp / 2; offset > 0; offset /= 2)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  __syncthreads();
  return value;
}

// one block per (batch, channel) row of xs (batch, channels, len): instance
// norm, the style's affine from h (batch, 2 * channels) holding scale - 1 then
// shift, then Snake (alpha per channel) or a leaky ReLU
__global__ void adain_act_f32_kernel(const float *xs, const float *h,
                                     const float *alpha, float *out,
                                     int channels, int len, float eps,
                                     float slope) {
  __shared__ float scratch[kWarp];
  const int row = blockIdx.x;
  const int batch = row / channels, channel = row % channels;
  const float *x = xs + static_cast<int64_t>(row) * len;
  float *y = out + static_cast<int64_t>(row) * len;
  float sum = 0.f;
  for (int i = threadIdx.x; i < len; i += blockDim.x)
    sum += x[i];
  const float mean = adain_block_sum(sum, scratch) / len;
  float squares = 0.f;
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    const float d = x[i] - mean;
    squares += d * d;
  }
  const float inv_std = rsqrtf(adain_block_sum(squares, scratch) / len + eps);
  const float scale =
      h[static_cast<int64_t>(batch) * 2 * channels + channel] + 1.f;
  const float shift =
      h[static_cast<int64_t>(batch) * 2 * channels + channels + channel];
  if (alpha != nullptr) {
    const float a = alpha[channel], inv_a = 1.f / a;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      const float v = (x[i] - mean) * inv_std * scale + shift;
      const float s = sinf(a * v);
      y[i] = v + inv_a * s * s;
    }
  } else {
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
      const float v = (x[i] - mean) * inv_std * scale + shift;
      y[i] = v < 0.f ? v * slope : v;
    }
  }
}

extern "C" int adain_act_f32(const void *xs, const void *h, const void *alpha,
                             void *out, int rows, int channels, int len,
                             float eps, float slope, int64_t stream) {
  adain_act_f32_kernel<<<rows, kAdainThreads, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const float *>(xs), static_cast<const float *>(h),
      static_cast<const float *>(alpha), static_cast<float *>(out), channels,
      len, eps, slope);
  return static_cast<int>(cudaGetLastError());
}
