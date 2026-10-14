#include <cstdint>
#include <cuda_runtime.h>

constexpr int kLstmThreads = 1024;
constexpr int kLstmUnroll = 4;

__device__ __forceinline__ float lstm_sigmoid(float x) {
  return 1.f / (1.f + expf(-x));
}

// one block per direction runs all its steps: the recurrent weights stay in L2
// and the state in shared memory gx (steps, 8H): input projections with both
// biases, forward gates then backward; w_hh_t (2, H, 4H); out (steps, 2H)
__global__ void lstm_bidirectional_f32_kernel(const float *gx,
                                              const float *w_hh_t, float *out,
                                              int steps, int hidden) {
  extern __shared__ float shared[];
  const int gates = 4 * hidden;
  float *h = shared;
  float *c = h + hidden;
  float *g = c + hidden;
  const int direction = blockIdx.x;
  const float *w = w_hh_t + static_cast<int64_t>(direction) * hidden * gates;
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    h[j] = 0.f;
    c[j] = 0.f;
  }
  __syncthreads();
  for (int step = 0; step < steps; ++step) {
    const int t = direction == 0 ? step : steps - 1 - step;
    const float *gx_t =
        gx + static_cast<int64_t>(t) * 2 * gates + direction * gates;
    for (int j = threadIdx.x; j < gates; j += blockDim.x) {
      float acc[kLstmUnroll] = {gx_t[j], 0.f, 0.f, 0.f};
      int k = 0;
      for (; k + kLstmUnroll <= hidden; k += kLstmUnroll) {
#pragma unroll
        for (int u = 0; u < kLstmUnroll; ++u)
          acc[u] += h[k + u] * w[static_cast<int64_t>(k + u) * gates + j];
      }
      for (; k < hidden; ++k)
        acc[0] += h[k] * w[static_cast<int64_t>(k) * gates + j];
      g[j] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
      const float i = lstm_sigmoid(g[j]);
      const float f = lstm_sigmoid(g[hidden + j]);
      const float cell = tanhf(g[2 * hidden + j]);
      const float o = lstm_sigmoid(g[3 * hidden + j]);
      c[j] = f * c[j] + i * cell;
      h[j] = o * tanhf(c[j]);
      out[static_cast<int64_t>(t) * 2 * hidden + direction * hidden + j] = h[j];
    }
    __syncthreads();
  }
}

extern "C" int lstm_bidirectional_f32(const void *gx, const void *w_hh_t,
                                      void *out, int steps, int hidden,
                                      int64_t stream) {
  // h, c and the four gates
  const size_t shared = static_cast<size_t>(6) * hidden * sizeof(float);
  if (shared > 48 * 1024) {
    cudaError_t status = cudaFuncSetAttribute(
        lstm_bidirectional_f32_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess)
      return static_cast<int>(status);
  }
  lstm_bidirectional_f32_kernel<<<2, kLstmThreads, shared,
                                  reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const float *>(gx), static_cast<const float *>(w_hh_t),
      static_cast<float *>(out), steps, hidden);
  return static_cast<int>(cudaGetLastError());
}
