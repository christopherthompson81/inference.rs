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

// The radix threshold of a row's top `k` visible block scores: blocks keyed
// above `*prefix` are chosen, and of those keyed equal, the first `*remaining`
// in block order. Every thread of the block calls it and gets the result.
__device__ void qsa_threshold(const float *__restrict__ row_scores, int visible,
                              int k, uint32_t *prefix_out, int *remaining_out) {
  __shared__ int histogram[kQsaRadix];
  __shared__ uint32_t digit;
  __shared__ int remaining;
  const int tid = threadIdx.x;
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
  *prefix_out = prefix;
  *remaining_out = remaining;
  __syncthreads();
}

// Walks the visible blocks in order, `blockDim.x` at a time, calling
// `pick(block, slot)` for each chosen one with its rank among the chosen. Every
// thread of the block calls it.
template <typename Pick>
__device__ void qsa_for_chosen(const float *__restrict__ row_scores,
                               int visible, uint32_t prefix, int remaining,
                               Pick pick) {
  __shared__ int warp_equal[kQsaThreads / kQsaWarp];
  __shared__ int warp_picked[kQsaThreads / kQsaWarp];
  __shared__ int equal_before;
  __shared__ int picked_before;
  const int tid = threadIdx.x;
  const int lane = tid % kQsaWarp, warp = tid / kQsaWarp;
  const int warps = blockDim.x / kQsaWarp;
  if (tid == 0) {
    equal_before = 0;
    picked_before = 0;
  }
  __syncthreads();
  for (int start = 0; start < visible; start += blockDim.x) {
    const int i = start + tid;
    const uint32_t key = i < visible ? qsa_ordered(row_scores[i]) : 0u;
    const bool above = i < visible && key > prefix;
    const bool equal = i < visible && key == prefix;
    const uint32_t equal_ballot = __ballot_sync(0xffffffffu, equal);
    if (lane == 0) {
      warp_equal[warp] = __popc(equal_ballot);
    }
    __syncthreads();
    int before = equal_before + __popc(equal_ballot & ((1u << lane) - 1u));
    for (int w = 0; w < warp; ++w) {
      before += warp_equal[w];
    }
    const bool picked = above || (equal && before < remaining);
    const uint32_t picked_ballot = __ballot_sync(0xffffffffu, picked);
    if (lane == 0) {
      warp_picked[warp] = __popc(picked_ballot);
    }
    __syncthreads();
    if (picked) {
      int slot = picked_before + __popc(picked_ballot & ((1u << lane) - 1u));
      for (int w = 0; w < warp; ++w) {
        slot += warp_picked[w];
      }
      pick(i, slot);
    }
    __syncthreads();
    if (tid == 0) {
      for (int w = 0; w < warps; ++w) {
        equal_before += warp_equal[w];
        picked_before += warp_picked[w];
      }
    }
    __syncthreads();
  }
}

// One CUDA block per query row: radix-select the top `k` of the row's visible
// block scores (ties to the lowest block), then write the row's additive mask
// over `n_kv` cells: causal, the chosen blocks and the incomplete tail block.
__global__ void qsa_select_mask_kernel(const float *__restrict__ scores,
                                       const uint32_t *__restrict__ positions,
                                       __half *__restrict__ mask, int n_pool,
                                       int n_kv, int k, int ratio) {
  extern __shared__ uint32_t chosen[];
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

  uint32_t prefix;
  int remaining;
  qsa_threshold(row_scores, visible, k, &prefix, &remaining);
  const int words = (n_pool + 31) / 32;
  for (int w = tid; w < words; w += blockDim.x) {
    chosen[w] = 0u;
  }
  __syncthreads();
  qsa_for_chosen(row_scores, visible, prefix, remaining, [&](int i, int) {
    atomicOr(&chosen[i / 32], 1u << (i % 32));
  });

  for (int t = tid; t < n_kv; t += blockDim.x) {
    const int block = t / ratio;
    const bool picked =
        t >= tail || ((chosen[block / 32] >> (block % 32)) & 1u) != 0u;
    row_mask[t] = (t <= position && picked) ? keep : hide;
  }
}

// One CUDA block per query row: the same selection as the mask, written as the
// row's kept positions in order into `width` slots (the chosen blocks' cells,
// then the tail), with every slot past the last kept one repeating the first.
__global__ void qsa_select_cells_kernel(const float *__restrict__ scores,
                                        const uint32_t *__restrict__ positions,
                                        uint32_t *__restrict__ cells,
                                        int n_pool, int width, int k,
                                        int ratio) {
  __shared__ int kept;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float *row_scores = scores + static_cast<int64_t>(row) * n_pool;
  uint32_t *row_cells = cells + static_cast<int64_t>(row) * width;
  const uint32_t position = positions[row];

  if (position == kQsaPaddedRow) {
    for (int t = tid; t < width; t += blockDim.x) {
      row_cells[t] = 0u;
    }
    return;
  }
  const int64_t seen = static_cast<int64_t>(position) + 1;
  const int complete = static_cast<int>(seen / ratio);
  const int visible = min(complete, n_pool);
  if (visible <= k) {
    for (int t = tid; t < width; t += blockDim.x) {
      row_cells[t] = t < seen ? static_cast<uint32_t>(t) : 0u;
    }
    return;
  }

  uint32_t prefix;
  int remaining;
  qsa_threshold(row_scores, visible, k, &prefix, &remaining);
  qsa_for_chosen(row_scores, visible, prefix, remaining, [&](int i, int slot) {
    for (int m = 0; m < ratio; ++m) {
      row_cells[slot * ratio + m] = static_cast<uint32_t>(i * ratio + m);
    }
  });
  // the selection always keeps exactly k blocks past the budget
  const int tail = complete * ratio;
  const int tail_len = static_cast<int>(seen - tail);
  if (tid == 0) {
    kept = k * ratio + tail_len;
  }
  for (int m = tid; m < tail_len; m += blockDim.x) {
    row_cells[k * ratio + m] = static_cast<uint32_t>(tail + m);
  }
  __syncthreads();
  const uint32_t first = row_cells[0];
  for (int t = kept + tid; t < width; t += blockDim.x) {
    row_cells[t] = first;
  }
}

// Copies each row's `width` listed cells out of a paged HND cache into its own
// run of `blocks_per_row` pages of a compact cache, one warp per (row, slot,
// head); slots past `width` repeat the row's first cell.
template <typename Word>
__global__ void qsa_gather_cells_kernel(const Word *__restrict__ cache,
                                        const uint32_t *__restrict__ table,
                                        const uint32_t *__restrict__ cells,
                                        Word *__restrict__ out, int width,
                                        int table_width, int blocks_per_row,
                                        int heads, int block_size,
                                        int row_words) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / kQsaWarp;
  const int lane = threadIdx.x % kQsaWarp;
  const int slots = blocks_per_row * block_size;
  const int row = blockIdx.y;
  const int slot = warp / heads;
  const int head = warp % heads;
  if (slot >= slots) {
    return;
  }
  const uint32_t *row_cells = cells + static_cast<int64_t>(row) * width;
  const uint32_t position = row_cells[slot < width ? slot : 0];
  const int64_t page =
      table[static_cast<int64_t>(row) * table_width + position / block_size];
  const int64_t src =
      ((page * heads + head) * block_size + position % block_size) * row_words;
  const int64_t out_page =
      static_cast<int64_t>(row) * blocks_per_row + slot / block_size;
  const int64_t dst =
      ((out_page * heads + head) * block_size + slot % block_size) * row_words;
  for (int w = lane; w < row_words; w += kQsaWarp) {
    out[dst + w] = cache[src + w];
  }
}

template <typename Word>
int qsa_gather_cells_launch(const void *cache, const uint32_t *table,
                            const uint32_t *cells, void *out, int rows,
                            int width, int table_width, int blocks_per_row,
                            int heads, int block_size, int row_bytes,
                            cudaStream_t stream) {
  const int warps = blocks_per_row * block_size * heads;
  const int warps_per_block = kQsaThreads / kQsaWarp;
  const dim3 grid((warps + warps_per_block - 1) / warps_per_block, rows);
  qsa_gather_cells_kernel<Word><<<grid, kQsaThreads, 0, stream>>>(
      static_cast<const Word *>(cache), table, cells, static_cast<Word *>(out),
      width, table_width, blocks_per_row, heads, block_size,
      row_bytes / static_cast<int>(sizeof(Word)));
  return static_cast<int>(cudaGetLastError());
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

extern "C" int qsa_select_cells(const float *scores, const uint32_t *positions,
                                uint32_t *cells, int rows, int n_pool,
                                int width, int k, int ratio, int64_t stream) {
  qsa_select_cells_kernel<<<rows, kQsaThreads, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
      scores, positions, cells, n_pool, width, k, ratio);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int qsa_gather_cells(const void *cache, const uint32_t *table,
                                const uint32_t *cells, void *out, int rows,
                                int width, int table_width, int blocks_per_row,
                                int heads, int block_size, int row_bytes,
                                int64_t stream) {
  const auto s = reinterpret_cast<cudaStream_t>(stream);
  if (row_bytes % 16 == 0) {
    return qsa_gather_cells_launch<uint4>(cache, table, cells, out, rows, width,
                                          table_width, blocks_per_row, heads,
                                          block_size, row_bytes, s);
  }
  if (row_bytes % 4 == 0) {
    return qsa_gather_cells_launch<uint32_t>(cache, table, cells, out, rows,
                                             width, table_width, blocks_per_row,
                                             heads, block_size, row_bytes, s);
  }
  return qsa_gather_cells_launch<uint8_t>(cache, table, cells, out, rows, width,
                                          table_width, blocks_per_row, heads,
                                          block_size, row_bytes, s);
}
