#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kQsaThreads = 512;
constexpr int kQsaWarp = 32;
constexpr int kQsaRadix = 256;
constexpr int kQsaDigitBits = 8;
constexpr uint32_t kQsaPaddedRow = UINT32_MAX;

// monotonic in the float's order; NaN sorts below every number
__device__ __forceinline__ uint32_t qsa_ordered(float value) {
  if (value != value) {
    return 0u;
  }
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// One CUDA block per query row: radix-select the top `k` of the row's visible block scores (ties to the lowest block),
// then write the row's additive mask over `n_kv` cells: causal, the chosen blocks and the incomplete tail block.
__global__ void qsa_select_mask_kernel(const float *__restrict__ scores,
                                       const uint32_t *__restrict__ positions,
                                       __half *__restrict__ mask, int n_pool,
                                       int n_kv, int k, int ratio) {
  extern __shared__ uint32_t chosen[];
  __shared__ int histogram[kQsaRadix];
  __shared__ int warp_equal[kQsaThreads / kQsaWarp];
  __shared__ uint32_t digit;
  __shared__ int remaining;
  __shared__ int equal_before;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float *row_scores = scores + static_cast<int64_t>(row) * n_pool;
  __half *row_mask = mask + static_cast<int64_t>(row) * n_kv;
  const __half keep = __float2half(0.f);
  const __half hide = __float2half(-INFINITY);
  const uint32_t position = positions[row];

  if (position == kQsaPaddedRow) {
    // a padding row reads one cell, so its output stays finite
    for (int t = tid; t < n_kv; t += blockDim.x) {
      row_mask[t] = t == 0 ? keep : hide;
    }
    return;
  }
  const int64_t seen = static_cast<int64_t>(position) + 1;
  const int complete = static_cast<int>(seen / ratio);
  const int64_t tail = static_cast<int64_t>(complete) * ratio;
  const int visible = min(complete, n_pool);
  if (visible <= k) {
    for (int t = tid; t < n_kv; t += blockDim.x) {
      row_mask[t] = t <= position ? keep : hide;
    }
    return;
  }

  uint32_t prefix = 0, prefix_mask = 0;
  if (tid == 0) {
    remaining = k;
  }
  for (int shift = 32 - kQsaDigitBits; shift >= 0; shift -= kQsaDigitBits) {
    for (int i = tid; i < kQsaRadix; i += blockDim.x) {
      histogram[i] = 0;
    }
    __syncthreads();
    for (int i = tid; i < visible; i += blockDim.x) {
      const uint32_t key = qsa_ordered(row_scores[i]);
      if ((key & prefix_mask) == prefix) {
        atomicAdd(&histogram[(key >> shift) & (kQsaRadix - 1)], 1);
      }
    }
    __syncthreads();
    if (tid == 0) {
      int above = 0;
      for (int d = kQsaRadix - 1; d >= 0; --d) {
        if (above + histogram[d] >= remaining) {
          digit = static_cast<uint32_t>(d);
          remaining -= above;
          break;
        }
        above += histogram[d];
      }
    }
    __syncthreads();
    prefix |= digit << shift;
    prefix_mask |= static_cast<uint32_t>(kQsaRadix - 1) << shift;
  }

  const int words = (n_pool + 31) / 32;
  for (int w = tid; w < words; w += blockDim.x) {
    chosen[w] = 0u;
  }
  if (tid == 0) {
    equal_before = 0;
  }
  __syncthreads();
  const int lane = tid % kQsaWarp, warp = tid / kQsaWarp;
  const int warps = blockDim.x / kQsaWarp;
  // `remaining` is now how many blocks tied at the threshold still fit, taken in block order
  for (int start = 0; start < visible; start += blockDim.x) {
    const int i = start + tid;
    const uint32_t key = i < visible ? qsa_ordered(row_scores[i]) : 0u;
    const bool above = i < visible && key > prefix;
    const bool equal = i < visible && key == prefix;
    const uint32_t ballot = __ballot_sync(0xffffffffu, equal);
    if (lane == 0) {
      warp_equal[warp] = __popc(ballot);
    }
    __syncthreads();
    int before = equal_before + __popc(ballot & ((1u << lane) - 1u));
    for (int w = 0; w < warp; ++w) {
      before += warp_equal[w];
    }
    if (above || (equal && before < remaining)) {
      atomicOr(&chosen[i / 32], 1u << (i % 32));
    }
    __syncthreads();
    if (tid == 0) {
      for (int w = 0; w < warps; ++w) {
        equal_before += warp_equal[w];
      }
    }
    __syncthreads();
  }

  for (int t = tid; t < n_kv; t += blockDim.x) {
    const int block = t / ratio;
    const bool picked =
        t >= tail || ((chosen[block / 32] >> (block % 32)) & 1u) != 0u;
    row_mask[t] = (t <= position && picked) ? keep : hide;
  }
}

} // namespace

extern "C" int qsa_select_mask(const float *scores, const uint32_t *positions,
                               void *mask, int rows, int n_pool, int n_kv,
                               int k, int ratio, int64_t stream) {
  const size_t chosen_bytes = static_cast<size_t>((n_pool + 31) / 32) * 4;
  qsa_select_mask_kernel<<<rows, kQsaThreads, chosen_bytes,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
      scores, positions, static_cast<__half *>(mask), n_pool, n_kv, k, ratio);
  return static_cast<int>(cudaGetLastError());
}
