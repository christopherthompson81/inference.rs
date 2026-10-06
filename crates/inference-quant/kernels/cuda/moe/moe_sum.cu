// Derived from vLLM (Apache-2.0): https://github.com/vllm-project/vllm

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

__global__ void moe_sum_bf16_kernel(__nv_bfloat16* __restrict__ out,
                                    const __nv_bfloat16* __restrict__ input,
                                    int32_t num_tokens, int32_t hidden,
                                    int32_t topk) {
  const int token = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= num_tokens || col >= hidden) return;

  const size_t base = ((size_t)token * topk * hidden) + col;
  float acc = 0.0f;
  for (int slot = 0; slot < topk; ++slot) {
    acc += __bfloat162float(input[base + (size_t)slot * hidden]);
  }
  out[(size_t)token * hidden + col] = __float2bfloat16(acc);
}

extern "C" void launch_moe_sum_bf16(void* out, const void* input,
                                     int32_t num_tokens, int32_t hidden,
                                     int32_t topk, cudaStream_t stream) {
  if (num_tokens == 0 || hidden == 0 || topk == 0) return;
  const int threads = 256;
  dim3 grid(num_tokens, (hidden + threads - 1) / threads);
  moe_sum_bf16_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<const __nv_bfloat16*>(input), num_tokens, hidden, topk);
}
