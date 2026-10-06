// Expert dispatch tables and the weighted top-k reduce the grouped mmq MoE
// uses.

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

static __global__ void moe_dispatch_count_kernel(
    const int32_t *__restrict__ topk_ids, // [total_assignments] flattened
    int32_t *__restrict__ expert_counts,  // [num_experts] zero-initialized
    const int total_assignments) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_assignments) {
    const int expert = topk_ids[idx];
    atomicAdd(&expert_counts[expert], 1);
  }
}

// one thread: fine for the expert counts MoE checkpoints use
static __global__ void moe_dispatch_prefix_sum_kernel(
    const int32_t *__restrict__ expert_counts, // [num_experts]
    int32_t *__restrict__ expert_bounds,       // [num_experts + 1]
    const int num_experts) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    expert_bounds[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
      expert_bounds[i + 1] = expert_bounds[i] + expert_counts[i];
    }
  }
}

static __global__ void moe_dispatch_scatter_kernel(
    const int32_t *__restrict__ topk_ids, // [total_assignments] flattened
    int32_t
        *__restrict__ expert_cursors, // [num_experts] init to expert_bounds[i]
    int32_t *__restrict__ sorted_token_ids,  // [total_assignments] output
    int32_t *__restrict__ sorted_source_ids, // [total_assignments] output
    const int total_assignments, const int topk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_assignments) {
    const int expert = topk_ids[idx];
    const int pos = atomicAdd(&expert_cursors[expert], 1);
    sorted_token_ids[pos] = idx; // flat index into topk_ids: token = idx/topk
    sorted_source_ids[pos] = idx / topk;
  }
}

template <typename InputT, typename OutputT>
static __global__ void moe_weighted_reduce_flat_kernel(
    const InputT *__restrict__ inputs, const float *__restrict__ topk_weights,
    OutputT *__restrict__ outputs, const int num_tokens, const int hidden,
    const int topk) {
  const int token = blockIdx.x;
  const int h = blockIdx.y * blockDim.x + threadIdx.x;
  if (token >= num_tokens)
    return;

  extern __shared__ float weights[];
  for (int slot = threadIdx.x; slot < topk; slot += blockDim.x) {
    weights[slot] = topk_weights[token * topk + slot];
  }
  __syncthreads();

  if (h >= hidden)
    return;

  const size_t input_base = (size_t)token * topk * hidden + h;
  float acc = 0.0f;
  for (int slot = 0; slot < topk; ++slot) {
    acc += (float)inputs[input_base + (size_t)slot * hidden] * weights[slot];
  }
  outputs[(size_t)token * hidden + h] = (OutputT)acc;
}

extern "C" void launch_moe_dispatch(
    const int32_t *topk_ids, int32_t *expert_bounds, int32_t *sorted_token_ids,
    int32_t *sorted_source_ids, int total_assignments, int num_experts,
    int topk, int32_t *expert_counts, int32_t *expert_cursors, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), s);
  {
    int threads = 256;
    int blocks = (total_assignments + threads - 1) / threads;
    moe_dispatch_count_kernel<<<blocks, threads, 0, s>>>(
        topk_ids, expert_counts, total_assignments);
  }

  moe_dispatch_prefix_sum_kernel<<<1, 1, 0, s>>>(expert_counts, expert_bounds,
                                                 num_experts);

  cudaMemcpyAsync(expert_cursors, expert_bounds, num_experts * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, s);

  {
    int threads = 256;
    int blocks = (total_assignments + threads - 1) / threads;
    moe_dispatch_scatter_kernel<<<blocks, threads, 0, s>>>(
        topk_ids, expert_cursors, sorted_token_ids, sorted_source_ids,
        total_assignments, topk);
  }
}

extern "C" int launch_moe_weighted_reduce_flat(const float *inputs,
                                               const float *topk_weights,
                                               float *outputs, int num_tokens,
                                               int hidden, int topk,
                                               void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  dim3 grid(num_tokens, 1 + (hidden - 1) / threads);
  moe_weighted_reduce_flat_kernel<<<grid, threads, topk * sizeof(float), s>>>(
      inputs, topk_weights, outputs, num_tokens, hidden, topk);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_moe_weighted_reduce_flat_bf16(const float *inputs,
                                                    const float *topk_weights,
                                                    __nv_bfloat16 *outputs,
                                                    int num_tokens, int hidden,
                                                    int topk, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  dim3 grid(num_tokens, 1 + (hidden - 1) / threads);
  moe_weighted_reduce_flat_kernel<<<grid, threads, topk * sizeof(float), s>>>(
      inputs, topk_weights, outputs, num_tokens, hidden, topk);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_moe_weighted_reduce_flat_f16_input(
    const half *inputs, const float *topk_weights, half *outputs,
    int num_tokens, int hidden, int topk, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  dim3 grid(num_tokens, 1 + (hidden - 1) / threads);
  moe_weighted_reduce_flat_kernel<<<grid, threads, topk * sizeof(float), s>>>(
      inputs, topk_weights, outputs, num_tokens, hidden, topk);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_moe_weighted_reduce_flat_bf16_input(
    const __nv_bfloat16 *inputs, const float *topk_weights,
    __nv_bfloat16 *outputs, int num_tokens, int hidden, int topk,
    void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int threads = 256;
  dim3 grid(num_tokens, 1 + (hidden - 1) / threads);
  moe_weighted_reduce_flat_kernel<<<grid, threads, topk * sizeof(float), s>>>(
      inputs, topk_weights, outputs, num_tokens, hidden, topk);
  return static_cast<int>(cudaGetLastError());
}
