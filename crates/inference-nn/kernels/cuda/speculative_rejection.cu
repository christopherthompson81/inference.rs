#include <cuda_runtime.h>
#include <climits>
#include <stdint.h>

namespace {

constexpr int SPEC_REJECTION_BLOCK_SIZE = 1024;
constexpr int WARP_SIZE = 32;
constexpr int SPEC_REJECTION_WARPS = SPEC_REJECTION_BLOCK_SIZE / WARP_SIZE;
// loads in flight per thread over a vocabulary-wide row
constexpr int SPEC_REJECTION_LOAD_UNROLL = 8;
constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr int SPEC_CHUNK_BLOCK_SIZE = 256;
constexpr int SPEC_REJECTION_MAX_CHUNKS = 128;
// a chunk's residual below this share of its mass is rounding of a fully covered chunk
constexpr float SPEC_RESIDUAL_FLOOR = 1e-5f;
constexpr int SPEC_REJECTION_MAX_Q = 128;
constexpr int SPEC_REJECTION_MAX_TOP_K = 128;
constexpr uint32_t SPEC_REJECTION_INVALID_VALUE = UINT32_MAX;

enum SpecRejectionStatus : uint32_t {
  SPEC_REJECTION_OK = 0,
  SPEC_REJECTION_NEEDS_CPU = 1,
  SPEC_REJECTION_INVALID_Q = 2,
  SPEC_REJECTION_INVALID_TARGET = 3,
  SPEC_REJECTION_INVALID_RNG = 4,
};

__device__ __forceinline__ bool filter_active(const float value) {
  return value > 0.0f && value < 1.0f;
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(FULL_MASK, value, offset);
  }
  return value;
}

__device__ __forceinline__ float rescaled(const float sum, const float from,
                                          const float to) {
  return sum == 0.0f ? 0.0f : sum * expf(from - to);
}

// (max, sum of exp relative to max) of two partial softmax denominators
__device__ __forceinline__ void combine_partial(float *max, float *sum,
                                                const float other_max,
                                                const float other_sum) {
  const float merged = fmaxf(*max, other_max);
  if (merged == -INFINITY) {
    return;
  }
  *sum = rescaled(*sum, *max, merged) + rescaled(other_sum, other_max, merged);
  *max = merged;
}

__device__ __forceinline__ void warp_combine_partial(float *max, float *sum) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    combine_partial(max, sum, __shfl_xor_sync(FULL_MASK, *max, offset),
                    __shfl_xor_sync(FULL_MASK, *sum, offset));
  }
}

// One block per (chunk, row): the chunk's max and its sum of exp relative to
// that max, the max NaN when the chunk holds a NaN or +inf.
__global__ void __launch_bounds__(SPEC_CHUNK_BLOCK_SIZE)
    categorical_chunk_stats_kernel(const float *__restrict__ target_logits,
                                   const float *__restrict__ inverse_temperatures,
                                   float2 *__restrict__ chunk_stats,
                                   const int rows_per_sequence, const int vocab,
                                   const int chunks) {
  __shared__ float2 warp_stats[SPEC_CHUNK_BLOCK_SIZE / WARP_SIZE];
  const int row = blockIdx.y;
  const float inverse_temperature =
      inverse_temperatures[row / rows_per_sequence];
  const float *logits = target_logits + static_cast<size_t>(row) * vocab;
  const int width = (vocab - 1) / chunks + 1;
  const int start = min(static_cast<int>(blockIdx.x) * width, vocab);
  const int end = min(start + width, vocab);
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  bool invalid = false;
  const int stride = blockDim.x * SPEC_REJECTION_LOAD_UNROLL;
  for (int base = start + threadIdx.x; base < end; base += stride) {
    float values[SPEC_REJECTION_LOAD_UNROLL];
#pragma unroll
    for (int k = 0; k < SPEC_REJECTION_LOAD_UNROLL; ++k) {
      const int token = base + k * blockDim.x;
      values[k] = token < end ? logits[token] : -INFINITY;
    }
#pragma unroll
    for (int k = 0; k < SPEC_REJECTION_LOAD_UNROLL; ++k) {
      invalid |= isnan(values[k]) || values[k] == INFINITY;
      const float scaled = values[k] * inverse_temperature;
      if (scaled > local_max) {
        local_sum = rescaled(local_sum, local_max, scaled) + 1.0f;
        local_max = scaled;
      } else if (scaled != -INFINITY) {
        local_sum += expf(scaled - local_max);
      }
    }
  }
  const bool chunk_invalid = __syncthreads_or(invalid);
  warp_combine_partial(&local_max, &local_sum);
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  if (lane == 0) {
    warp_stats[warp] = make_float2(local_max, local_sum);
  }
  __syncthreads();
  if (warp == 0) {
    const bool present = lane < static_cast<int>(blockDim.x / WARP_SIZE);
    float max = present ? warp_stats[lane].x : -INFINITY;
    float sum = present ? warp_stats[lane].y : 0.0f;
    warp_combine_partial(&max, &sum);
    if (lane == 0) {
      chunk_stats[static_cast<size_t>(row) * chunks + blockIdx.x] =
          make_float2(chunk_invalid ? NAN : max, sum);
    }
  }
}

struct CategoricalShared {
  float segments[SPEC_REJECTION_MAX_CHUNKS];
  float q_adjust[SPEC_REJECTION_MAX_Q];
  float scan[SPEC_REJECTION_WARPS + 1];
  uint32_t q_raw_ids[SPEC_REJECTION_MAX_Q];
  float q_raw_probs[SPEC_REJECTION_MAX_Q];
  bool q_first[SPEC_REJECTION_MAX_Q];
  uint32_t sorted_q_ids[SPEC_REJECTION_MAX_Q];
  float scaled_q_probs[SPEC_REJECTION_MAX_Q];
  double q_sum;
  float q_draft;
  float row_max;
  float denominator;
  float segment_target;
  int selected_segment;
  int found_token;
  int last_positive;
};

// The row's max and softmax denominator from its chunks' partials
__device__ bool categorical_row_stats(const float2 *row_chunks,
                                      const int chunks, CategoricalShared &sh) {
  if (threadIdx.x < WARP_SIZE) {
    float max = -INFINITY;
    float sum = 0.0f;
    bool invalid = false;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += WARP_SIZE) {
      const float2 partial = row_chunks[chunk];
      invalid |= isnan(partial.x);
      combine_partial(&max, &sum, partial.x, partial.y);
    }
    invalid = __any_sync(FULL_MASK, invalid);
    warp_combine_partial(&max, &sum);
    if (threadIdx.x == 0) {
      sh.row_max = max;
      sh.denominator = invalid ? NAN : sum;
    }
  }
  __syncthreads();
  return isfinite(sh.row_max) && sh.denominator > 0.0f &&
         isfinite(sh.denominator);
}

__device__ SpecRejectionStatus validate_q_row(
    const uint32_t *q_ids, const float *q_probs, const uint32_t draft,
    const int vocab, const int q_width, double *q_sum, float *q_draft) {
  double sum = 0.0;
  for (int slot = 0; slot < q_width; ++slot) {
    const uint32_t id = q_ids[slot];
    const float probability = q_probs[slot];
    if (id >= static_cast<uint32_t>(vocab) || !isfinite(probability) ||
        probability < 0.0f) {
      return SPEC_REJECTION_INVALID_Q;
    }
    sum += static_cast<double>(probability);
  }
  if (!(sum > 0.0) || !isfinite(sum)) {
    return SPEC_REJECTION_INVALID_Q;
  }
  float draft_sum = 0.0f;
  for (int slot = 0; slot < q_width; ++slot) {
    if (q_ids[slot] == draft) {
      draft_sum += static_cast<float>(static_cast<double>(q_probs[slot]) / sum);
    }
  }
  if (!(draft_sum > 0.0f) || !isfinite(draft_sum)) {
    return SPEC_REJECTION_INVALID_Q;
  }
  *q_sum = sum;
  *q_draft = draft_sum;
  return SPEC_REJECTION_OK;
}

__device__ __forceinline__ double warp_sum_double(double value) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(FULL_MASK, value, offset);
  }
  return value;
}

// Stages a q row in shared memory and checks it: its sum and the draft's share
// of it. Every thread calls it; q_width <= the block's width.
__device__ SpecRejectionStatus load_q_row(const uint32_t *ids,
                                          const float *probs,
                                          const uint32_t draft, const int vocab,
                                          const int q_width,
                                          CategoricalShared &sh) {
  bool bad = false;
  if (threadIdx.x < q_width) {
    const uint32_t id = ids[threadIdx.x];
    const float probability = probs[threadIdx.x];
    bad = id >= static_cast<uint32_t>(vocab) || !isfinite(probability) ||
          probability < 0.0f;
    sh.q_raw_ids[threadIdx.x] = id;
    sh.q_raw_probs[threadIdx.x] = probability;
  }
  if (__syncthreads_or(bad)) {
    return SPEC_REJECTION_INVALID_Q;
  }
  if (threadIdx.x < WARP_SIZE) {
    double sum = 0.0;
    for (int slot = threadIdx.x; slot < q_width; slot += WARP_SIZE) {
      sum += static_cast<double>(sh.q_raw_probs[slot]);
    }
    sum = warp_sum_double(sum);
    float draft_sum = 0.0f;
    for (int slot = threadIdx.x; slot < q_width; slot += WARP_SIZE) {
      if (sh.q_raw_ids[slot] == draft) {
        draft_sum += static_cast<float>(
            static_cast<double>(sh.q_raw_probs[slot]) / sum);
      }
    }
    draft_sum = warp_sum(draft_sum);
    if (threadIdx.x == 0) {
      sh.q_sum = sum;
      sh.q_draft = draft_sum;
    }
  }
  __syncthreads();
  const bool valid = sh.q_sum > 0.0 && isfinite(sh.q_sum) &&
                     sh.q_draft > 0.0f && isfinite(sh.q_draft);
  return valid ? SPEC_REJECTION_OK : SPEC_REJECTION_INVALID_Q;
}

// The staged q row as sorted unique ids, duplicates merged in slot order, each
// scaled to the target's unnormalized weights. Every thread calls it.
__device__ int sort_q_row(const int q_width, const float target_denominator,
                          CategoricalShared &sh) {
  const int slot = threadIdx.x;
  uint32_t id = 0;
  float probability = 0.0f;
  bool first = false;
  if (slot < q_width) {
    id = sh.q_raw_ids[slot];
    probability = static_cast<float>(
        static_cast<double>(sh.q_raw_probs[slot]) / sh.q_sum);
    first = probability != 0.0f;
    for (int earlier = 0; first && earlier < slot; ++earlier) {
      first = !(sh.q_raw_ids[earlier] == id && sh.q_raw_probs[earlier] != 0.0f);
    }
  }
  __syncthreads();
  if (slot < q_width) {
    sh.q_raw_probs[slot] = probability;
    sh.q_first[slot] = first;
  }
  const int count = __syncthreads_count(first);
  if (first) {
    int rank = 0;
    float merged = 0.0f;
    for (int other = 0; other < q_width; ++other) {
      const uint32_t other_id = sh.q_raw_ids[other];
      rank += sh.q_first[other] && other_id < id;
      if (other >= slot && other_id == id) {
        merged += sh.q_raw_probs[other];
      }
    }
    sh.sorted_q_ids[rank] = id;
    sh.scaled_q_probs[rank] = merged * target_denominator;
  }
  __syncthreads();
  return count;
}

__device__ __forceinline__ int q_lower_bound(const uint32_t *q_ids,
                                             const int q_count,
                                             const int token) {
  int index = 0;
  while (index < q_count && q_ids[index] < static_cast<uint32_t>(token)) {
    ++index;
  }
  return index;
}

// The residual weight of `token`, whose window of tokens ends before
// window_end; q ids below the window were consumed by the earlier windows.
// The cursor walk is uniform across the block.
__device__ __forceinline__ float categorical_residual_weight(
    const float value, const int token, const float inverse_temperature,
    const float row_max, const uint32_t *q_ids, const float *q_scaled,
    const int q_count, int *q_cursor, const int window_end) {
  float weight = expf(value * inverse_temperature - row_max);
  int cursor = *q_cursor;
  for (; cursor < q_count && q_ids[cursor] < static_cast<uint32_t>(window_end);
       ++cursor) {
    if (q_ids[cursor] == static_cast<uint32_t>(token)) {
      weight = fmaxf(weight - q_scaled[cursor], 0.0f);
    }
  }
  *q_cursor = cursor;
  return weight;
}

// Each chunk's residual mass: its exp sum less what q takes from its q tokens,
// floored to zero when q covers all but rounding of it.
__device__ void categorical_segment_weights(
    const float *logits, const float2 *row_chunks, const int vocab,
    const int chunks, const float inverse_temperature, const int q_count,
    CategoricalShared &sh) {
  if (threadIdx.x < q_count) {
    const float target = expf(
        logits[sh.sorted_q_ids[threadIdx.x]] * inverse_temperature - sh.row_max);
    sh.q_adjust[threadIdx.x] = fminf(target, sh.scaled_q_probs[threadIdx.x]);
  }
  __syncthreads();
  if (threadIdx.x < chunks) {
    const float2 partial = row_chunks[threadIdx.x];
    const float plain = rescaled(partial.y, partial.x, sh.row_max);
    const int width = (vocab - 1) / chunks + 1;
    const uint32_t start = static_cast<uint32_t>(threadIdx.x * width);
    const uint32_t end = start + static_cast<uint32_t>(width);
    float adjust = 0.0f;
    for (int slot = 0; slot < q_count; ++slot) {
      const uint32_t id = sh.sorted_q_ids[slot];
      if (id >= start && id < end) {
        adjust += sh.q_adjust[slot];
      }
    }
    const float residual = plain - adjust;
    sh.segments[threadIdx.x] =
        residual > SPEC_RESIDUAL_FLOOR * plain ? residual : 0.0f;
  }
  __syncthreads();
}

__device__ bool select_categorical_segment(const int segments,
                                           const float uniform,
                                           CategoricalShared &sh) {
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int segment = 0; segment < segments; ++segment) {
      total += sh.segments[segment];
    }
    sh.selected_segment = -1;
    if (total > 0.0f && isfinite(total)) {
      const float target = fminf(uniform * total, nextafterf(total, -INFINITY));
      float cumulative = 0.0f;
      for (int segment = 0; segment < segments; ++segment) {
        const float next = cumulative + sh.segments[segment];
        if (target < next) {
          sh.selected_segment = segment;
          sh.segment_target = target - cumulative;
          break;
        }
        cumulative = next;
      }
    }
  }
  __syncthreads();
  return sh.selected_segment >= 0;
}

// Inclusive prefix sum over the block; scan[SPEC_REJECTION_WARPS] gets the
// block's total.
__device__ __forceinline__ float block_inclusive_scan(float value,
                                                      float *scan) {
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int warps = blockDim.x / WARP_SIZE;
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    const float other = __shfl_up_sync(FULL_MASK, value, offset);
    if (lane >= offset) {
      value += other;
    }
  }
  if (lane == WARP_SIZE - 1) {
    scan[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    float total = lane < warps ? scan[lane] : 0.0f;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
      const float other = __shfl_up_sync(FULL_MASK, total, offset);
      if (lane >= offset) {
        total += other;
      }
    }
    if (lane < warps) {
      scan[lane] = total;
    }
    if (lane == warps - 1) {
      scan[SPEC_REJECTION_WARPS] = total;
    }
  }
  __syncthreads();
  return warp > 0 ? value + scan[warp - 1] : value;
}

// The selected chunk's token at segment_target of its cumulative residual mass,
// or its last token with mass when rounding leaves the target past the end.
__device__ int walk_categorical_segment(const float *logits, const int vocab,
                                        const int chunks,
                                        const float inverse_temperature,
                                        const int q_count,
                                        CategoricalShared &sh) {
  const int width = (vocab - 1) / chunks + 1;
  const int start = min(sh.selected_segment * width, vocab);
  const int end = min(start + width, vocab);
  const float target = sh.segment_target;
  if (threadIdx.x == 0) {
    sh.found_token = INT_MAX;
    sh.last_positive = -1;
  }
  __syncthreads();
  int q_cursor = q_lower_bound(sh.sorted_q_ids, q_count, start);
  float carry = 0.0f;
  for (int base = start; base < end; base += blockDim.x) {
    const int token = base + threadIdx.x;
    const float value = token < end ? logits[token] : -INFINITY;
    const float weight = categorical_residual_weight(
        value, token, inverse_temperature, sh.row_max, sh.sorted_q_ids,
        sh.scaled_q_probs, q_count, &q_cursor, min(base + blockDim.x, end));
    const float cumulative = carry + block_inclusive_scan(weight, sh.scan);
    if (weight > 0.0f) {
      atomicMax(&sh.last_positive, token);
    }
    if (weight > 0.0f && target < cumulative) {
      atomicMin(&sh.found_token, token);
    }
    carry += sh.scan[SPEC_REJECTION_WARPS];
    __syncthreads();
    if (sh.found_token != INT_MAX) {
      break;
    }
  }
  return sh.found_token != INT_MAX ? sh.found_token : sh.last_positive;
}

// The residual's draw; the target's own when the residual has no mass left
__device__ SpecRejectionStatus sample_categorical_row(
    const float *logits, const float2 *row_chunks, const int vocab,
    const int chunks, const float inverse_temperature, const float uniform,
    const int q_count, CategoricalShared &sh, uint32_t *output_token) {
  if (!isfinite(uniform) || uniform < 0.0f || uniform >= 1.0f) {
    return SPEC_REJECTION_INVALID_RNG;
  }
  for (int effective_q = q_count;; effective_q = 0) {
    categorical_segment_weights(logits, row_chunks, vocab, chunks,
                                inverse_temperature, effective_q, sh);
    if (select_categorical_segment(chunks, uniform, sh)) {
      const int token = walk_categorical_segment(
          logits, vocab, chunks, inverse_temperature, effective_q, sh);
      if (token >= 0) {
        *output_token = static_cast<uint32_t>(token);
        return SPEC_REJECTION_OK;
      }
    }
    if (effective_q == 0) {
      return SPEC_REJECTION_INVALID_TARGET;
    }
  }
}

__global__ void __launch_bounds__(SPEC_REJECTION_BLOCK_SIZE)
    sparse_rejection_categorical_f32_kernel(
        const float *__restrict__ target_logits,
        const float2 *__restrict__ chunk_stats,
        const uint32_t *__restrict__ draft_tokens,
        const uint32_t *__restrict__ q_token_ids,
        const float *__restrict__ q_probs,
        const float *__restrict__ inverse_temperatures,
        const uint32_t *__restrict__ target_top_k,
        const float *__restrict__ top_p, const float *__restrict__ min_p,
        const float *__restrict__ accept_uniforms,
        const float *__restrict__ sample_uniforms,
        uint32_t *__restrict__ outcomes, const int batch, const int drafts,
        const int vocab, const int q_width, const int chunks) {
  const int sequence = blockIdx.x;
  if (sequence >= batch) {
    return;
  }

  __shared__ CategoricalShared sh;
  __shared__ uint32_t continuation;
  __shared__ uint32_t accepted;
  __shared__ uint32_t rejection_row;
  __shared__ uint32_t row_status;
  __shared__ int q_count;

  if (threadIdx.x == 0) {
    continuation = SPEC_REJECTION_INVALID_VALUE;
    accepted = 0;
    rejection_row = SPEC_REJECTION_INVALID_VALUE;
    row_status = SPEC_REJECTION_OK;
    q_count = 0;
    const float row_top_p = top_p[sequence];
    const float row_min_p = min_p[sequence];
    if (target_top_k[sequence] != 0 || filter_active(row_top_p) ||
        filter_active(row_min_p)) {
      row_status = SPEC_REJECTION_NEEDS_CPU;
    }
    const float inverse_temperature = inverse_temperatures[sequence];
    if (!(inverse_temperature > 0.0f) || !isfinite(inverse_temperature)) {
      row_status = SPEC_REJECTION_INVALID_TARGET;
    }
  }
  __syncthreads();
  if (row_status != SPEC_REJECTION_OK) {
    if (threadIdx.x == 0) {
      outcomes[sequence * 3] = SPEC_REJECTION_INVALID_VALUE;
      outcomes[sequence * 3 + 1] = SPEC_REJECTION_INVALID_VALUE;
      outcomes[sequence * 3 + 2] = row_status;
    }
    return;
  }

  const float inverse_temperature = inverse_temperatures[sequence];
  const size_t row_stride = static_cast<size_t>(vocab);
  const size_t sequence_row = static_cast<size_t>(sequence) * (drafts + 1);
  const auto row_chunks = [&](const size_t row) {
    return chunk_stats + (sequence_row + row) * chunks;
  };
  for (int position = 0; position < drafts; ++position) {
    const float *row = target_logits + (sequence_row + position) * row_stride;
    if (!categorical_row_stats(row_chunks(position), chunks, sh)) {
      if (threadIdx.x == 0) {
        row_status = SPEC_REJECTION_INVALID_TARGET;
      }
      __syncthreads();
      break;
    }

    const size_t draft_offset =
        static_cast<size_t>(sequence) * drafts + position;
    const uint32_t draft = draft_tokens[draft_offset];
    const SpecRejectionStatus q_status =
        draft >= static_cast<uint32_t>(vocab)
            ? SPEC_REJECTION_INVALID_Q
            : load_q_row(q_token_ids + draft_offset * q_width,
                         q_probs + draft_offset * q_width, draft, vocab,
                         q_width, sh);
    if (threadIdx.x == 0) {
      row_status = q_status;
      const float uniform = accept_uniforms[draft_offset];
      if (row_status == SPEC_REJECTION_OK &&
          (!isfinite(uniform) || uniform < 0.0f || uniform >= 1.0f)) {
        row_status = SPEC_REJECTION_INVALID_RNG;
      }
      if (row_status == SPEC_REJECTION_OK) {
        const float target_probability =
            expf(row[draft] * inverse_temperature - sh.row_max) /
            sh.denominator;
        const float accept_probability =
            fminf(target_probability / sh.q_draft, 1.0f);
        if (uniform < accept_probability) {
          accepted = static_cast<uint32_t>(position + 1);
        } else {
          rejection_row = static_cast<uint32_t>(position);
        }
      }
    }
    __syncthreads();
    if (row_status != SPEC_REJECTION_OK ||
        rejection_row != SPEC_REJECTION_INVALID_VALUE) {
      break;
    }
  }

  if (row_status == SPEC_REJECTION_OK &&
      rejection_row == SPEC_REJECTION_INVALID_VALUE) {
    if (!categorical_row_stats(row_chunks(drafts), chunks, sh) &&
        threadIdx.x == 0) {
      row_status = SPEC_REJECTION_INVALID_TARGET;
    }
    __syncthreads();
  } else if (row_status == SPEC_REJECTION_OK) {
    // the rejected row's q is still staged
    const int count = sort_q_row(q_width, sh.denominator, sh);
    if (threadIdx.x == 0) {
      q_count = count;
    }
    __syncthreads();
  }

  if (row_status == SPEC_REJECTION_OK) {
    const int selected_row = rejection_row == SPEC_REJECTION_INVALID_VALUE
                                 ? drafts
                                 : static_cast<int>(rejection_row);
    const SpecRejectionStatus sample_status = sample_categorical_row(
        target_logits + (sequence_row + selected_row) * row_stride,
        row_chunks(selected_row), vocab, chunks, inverse_temperature,
        sample_uniforms[sequence], q_count, sh, &continuation);
    if (threadIdx.x == 0) {
      row_status = sample_status;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (row_status == SPEC_REJECTION_OK) {
      outcomes[sequence * 3] = accepted;
      outcomes[sequence * 3 + 1] = continuation;
    } else {
      outcomes[sequence * 3] = SPEC_REJECTION_INVALID_VALUE;
      outcomes[sequence * 3 + 1] = SPEC_REJECTION_INVALID_VALUE;
    }
    outcomes[sequence * 3 + 2] = row_status;
  }
}

__device__ __forceinline__ float row_normalizer(const float *normalizers,
                                                const size_t row) {
  return normalizers == nullptr ? 0.0f : normalizers[row];
}

// requested_k 0: top-p / min-p over the whole vocabulary via full_normalizer; a
// wider nucleus needs the CPU. The top-K rows, a block of one thread a
// candidate slot: reading, the softmax weights, the duplicate check, the
// residual and the draw's ranking by token id run per slot; the sums whose
// order matters (top-k, top-p, min-p, the residual, the draw's cumulative walk)
// stay serial on thread 0 in slot order, so a row's outcome does not depend on
// the block's width. requested_k 0 is top-p / min-p over the whole vocabulary
// via full_normalizer; a nucleus wider than the packed K needs the CPU.
constexpr int SPEC_TOPK_BLOCK_SIZE = SPEC_REJECTION_MAX_TOP_K;
constexpr uint8_t SPEC_SLOT_OK = 0;
constexpr uint8_t SPEC_SLOT_END = 1;
constexpr uint8_t SPEC_SLOT_INVALID = 2;

struct SpecTopkShared {
  uint32_t ids[SPEC_REJECTION_MAX_TOP_K];
  float weights[SPEC_REJECTION_MAX_TOP_K];
  uint32_t sorted_ids[SPEC_REJECTION_MAX_TOP_K];
  float sorted_weights[SPEC_REJECTION_MAX_TOP_K];
  uint8_t event[SPEC_REJECTION_MAX_TOP_K];
  double q_sum;
  uint32_t status;
  uint32_t token;
  uint32_t flag;
  int count;
  float denominator;
};

__device__ SpecRejectionStatus prepare_topk_block(
    const float *packed, const int packed_k, const uint32_t requested_k,
    const int vocab, const float inverse_temperature, const float top_p,
    const float min_p, const float full_normalizer, SpecTopkShared &sh) {
  const int tid = threadIdx.x;
  const bool nucleus = requested_k == 0;
  if (nucleus && !(full_normalizer > 0.0f && isfinite(full_normalizer))) {
    return SPEC_REJECTION_NEEDS_CPU;
  }
  const int effective_k = nucleus ? min(packed_k, vocab)
                                  : min(static_cast<int>(requested_k), vocab);
  if (effective_k > packed_k) {
    return SPEC_REJECTION_NEEDS_CPU;
  }
  if (!(inverse_temperature > 0.0f) || !isfinite(inverse_temperature)) {
    return SPEC_REJECTION_INVALID_TARGET;
  }
  const float first_value = packed[0];
  if (!isfinite(first_value)) {
    return SPEC_REJECTION_INVALID_TARGET;
  }
  const float scaled_max = first_value * inverse_temperature;
  if (tid < effective_k) {
    const float value = packed[tid];
    const float packed_id = packed[packed_k + tid];
    bool bad = isnan(value) || value == INFINITY || !isfinite(packed_id) ||
               packed_id < 0.0f || floorf(packed_id) != packed_id;
    const uint32_t id = bad ? 0u : static_cast<uint32_t>(packed_id);
    bad = bad || id >= static_cast<uint32_t>(vocab);
    sh.ids[tid] = id;
    sh.event[tid] = bad ? SPEC_SLOT_INVALID
                        : (value == -INFINITY ? SPEC_SLOT_END : SPEC_SLOT_OK);
    sh.weights[tid] = sh.event[tid] == SPEC_SLOT_OK
                          ? expf(value * inverse_temperature - scaled_max)
                          : 0.0f;
  }
  __syncthreads();
  // a slot repeating an earlier slot's id; only slots before the first END or
  // invalid one count, as the scan enforces
  bool duplicate = false;
  if (tid < effective_k && sh.event[tid] == SPEC_SLOT_OK) {
    for (int previous = 0; previous < tid; ++previous) {
      if (sh.event[previous] == SPEC_SLOT_OK &&
          sh.ids[previous] == sh.ids[tid]) {
        duplicate = true;
        break;
      }
    }
  }
  __syncthreads();
  if (duplicate) {
    sh.event[tid] = SPEC_SLOT_INVALID;
  }
  __syncthreads();
  if (tid == 0) {
    SpecRejectionStatus status = SPEC_REJECTION_OK;
    float topk_sum = 0.0f;
    int count = 0;
    for (int slot = 0; slot < effective_k; ++slot) {
      if (sh.event[slot] == SPEC_SLOT_INVALID) {
        status = SPEC_REJECTION_INVALID_TARGET;
        break;
      }
      if (sh.event[slot] == SPEC_SLOT_END) {
        break;
      }
      topk_sum += sh.weights[slot];
      ++count;
    }
    if (status == SPEC_REJECTION_OK &&
        (!(topk_sum > 0.0f) || !isfinite(topk_sum))) {
      status = SPEC_REJECTION_INVALID_TARGET;
    }
    float filtered_sum = 0.0f;
    if (status == SPEC_REJECTION_OK) {
      const float last_weight = sh.weights[count - 1];
      bool top_p_covered = false;
      if (filter_active(top_p)) {
        const float cutoff = top_p * (nucleus ? full_normalizer : topk_sum);
        float cumulative = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
          if (cumulative >= cutoff) {
            sh.weights[slot] = 0.0f;
          } else {
            cumulative += sh.weights[slot];
          }
        }
        top_p_covered = cumulative >= cutoff;
      }
      if (nucleus && count == effective_k && effective_k < vocab &&
          !top_p_covered && !(filter_active(min_p) && min_p >= last_weight)) {
        status = SPEC_REJECTION_NEEDS_CPU;
      }
      if (status == SPEC_REJECTION_OK && filter_active(min_p)) {
        const float threshold = sh.weights[0] * min_p;
        for (int slot = 0; slot < count; ++slot) {
          if (threshold >= sh.weights[slot]) {
            sh.weights[slot] = 0.0f;
          }
        }
      }
      if (status == SPEC_REJECTION_OK) {
        for (int slot = 0; slot < count; ++slot) {
          filtered_sum += sh.weights[slot];
        }
        if (!(filtered_sum > 0.0f) || !isfinite(filtered_sum)) {
          status = SPEC_REJECTION_INVALID_TARGET;
        }
      }
    }
    sh.status = status;
    sh.count = count;
    sh.denominator = filtered_sum;
  }
  __syncthreads();
  return static_cast<SpecRejectionStatus>(sh.status);
}

// A draw over the candidates in ascending token order (each slot ranked by id;
// the ids are distinct), the cumulative walk serial on thread 0
__device__ SpecRejectionStatus sample_topk_block(const int count,
                                                 const float uniform,
                                                 SpecTopkShared &sh) {
  const int tid = threadIdx.x;
  if (!isfinite(uniform) || uniform < 0.0f || uniform >= 1.0f) {
    return SPEC_REJECTION_INVALID_RNG;
  }
  if (tid < count) {
    const uint32_t id = sh.ids[tid];
    int rank = 0;
    for (int other = 0; other < count; ++other) {
      rank += sh.ids[other] < id;
    }
    sh.sorted_ids[rank] = id;
    sh.sorted_weights[rank] = sh.weights[tid];
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    for (int index = 0; index < count; ++index) {
      total += sh.sorted_weights[index];
    }
    SpecRejectionStatus status = SPEC_REJECTION_INVALID_TARGET;
    if (total > 0.0f && isfinite(total)) {
      const float target = fminf(uniform * total, nextafterf(total, -INFINITY));
      float cumulative = 0.0f;
      int last_positive = -1;
      for (int index = 0; index < count; ++index) {
        if (sh.sorted_weights[index] > 0.0f) {
          last_positive = index;
        }
        cumulative += sh.sorted_weights[index];
        if (target < cumulative) {
          sh.token = sh.sorted_ids[index];
          status = SPEC_REJECTION_OK;
          break;
        }
      }
      if (status != SPEC_REJECTION_OK && last_positive >= 0) {
        sh.token = sh.sorted_ids[last_positive];
        status = SPEC_REJECTION_OK;
      }
    }
    sh.status = status;
  }
  __syncthreads();
  return static_cast<SpecRejectionStatus>(sh.status);
}

__global__ void
__launch_bounds__(SPEC_TOPK_BLOCK_SIZE) sparse_rejection_topk_f32_kernel(
    const float *__restrict__ packed_target,
    const uint32_t *__restrict__ draft_tokens,
    const uint32_t *__restrict__ q_token_ids, const float *__restrict__ q_probs,
    const float *__restrict__ inverse_temperatures,
    const uint32_t *__restrict__ target_top_k, const float *__restrict__ top_p,
    const float *__restrict__ min_p, const float *__restrict__ accept_uniforms,
    const float *__restrict__ sample_uniforms,
    const float *__restrict__ full_normalizers, uint32_t *__restrict__ outcomes,
    const int batch, const int drafts, const int vocab, const int q_width,
    const int packed_k) {
  const int sequence = blockIdx.x;
  if (sequence >= batch) {
    return;
  }
  __shared__ SpecTopkShared sh;
  const int tid = threadIdx.x;
  uint32_t accepted = 0;
  uint32_t continuation = SPEC_REJECTION_INVALID_VALUE;
  SpecRejectionStatus status = SPEC_REJECTION_OK;
  const int packed_width = 2 * packed_k;
  const float inverse_temperature = inverse_temperatures[sequence];
  const size_t sequence_row = static_cast<size_t>(sequence) * (drafts + 1);

  for (int position = 0; position < drafts; ++position) {
    const float *packed =
        packed_target + (sequence_row + position) * packed_width;
    status = prepare_topk_block(
        packed, packed_k, target_top_k[sequence], vocab, inverse_temperature,
        top_p[sequence], min_p[sequence],
        row_normalizer(full_normalizers, sequence_row + position), sh);
    if (status != SPEC_REJECTION_OK) {
      break;
    }
    const int support_size = sh.count;
    const float target_denominator = sh.denominator;

    const size_t draft_offset =
        static_cast<size_t>(sequence) * drafts + position;
    const uint32_t draft = draft_tokens[draft_offset];
    if (draft >= static_cast<uint32_t>(vocab)) {
      status = SPEC_REJECTION_INVALID_Q;
      break;
    }
    const size_t q_offset = draft_offset * q_width;
    const float uniform = accept_uniforms[draft_offset];
    // thread 0 validates q and decides; every thread reads the decision
    if (tid == 0) {
      double q_sum = 0.0;
      float q_draft = 0.0f;
      SpecRejectionStatus row =
          validate_q_row(q_token_ids + q_offset, q_probs + q_offset, draft,
                         vocab, q_width, &q_sum, &q_draft);
      if (row == SPEC_REJECTION_OK &&
          (!isfinite(uniform) || uniform < 0.0f || uniform >= 1.0f)) {
        row = SPEC_REJECTION_INVALID_RNG;
      }
      uint32_t accept = 0;
      if (row == SPEC_REJECTION_OK) {
        float draft_weight = 0.0f;
        for (int slot = 0; slot < support_size; ++slot) {
          if (sh.ids[slot] == draft) {
            draft_weight = sh.weights[slot];
            break;
          }
        }
        const float target_probability = draft_weight / target_denominator;
        const float accept_probability =
            fminf(target_probability / q_draft, 1.0f);
        accept = uniform < accept_probability;
      }
      sh.status = row;
      sh.flag = accept;
      sh.q_sum = q_sum;
    }
    __syncthreads();
    status = static_cast<SpecRejectionStatus>(sh.status);
    if (status != SPEC_REJECTION_OK) {
      break;
    }
    const bool accept = sh.flag != 0;
    const double q_sum = sh.q_sum;
    __syncthreads();
    if (accept) {
      accepted = static_cast<uint32_t>(position + 1);
      continue;
    }

    if (tid < support_size) {
      float candidate_q = 0.0f;
      for (int q_slot = 0; q_slot < q_width; ++q_slot) {
        if (q_token_ids[q_offset + q_slot] == sh.ids[tid]) {
          candidate_q += static_cast<float>(
              static_cast<double>(q_probs[q_offset + q_slot]) / q_sum);
        }
      }
      sh.weights[tid] =
          fmaxf(sh.weights[tid] - target_denominator * candidate_q, 0.0f);
    }
    __syncthreads();
    if (tid == 0) {
      float residual_total = 0.0f;
      for (int slot = 0; slot < support_size; ++slot) {
        residual_total += sh.weights[slot];
      }
      sh.flag = !(residual_total > 0.0f) || !isfinite(residual_total);
    }
    __syncthreads();
    int count = support_size;
    if (sh.flag) {
      status = prepare_topk_block(
          packed, packed_k, target_top_k[sequence], vocab, inverse_temperature,
          top_p[sequence], min_p[sequence],
          row_normalizer(full_normalizers, sequence_row + position), sh);
      count = sh.count;
    }
    if (status == SPEC_REJECTION_OK) {
      status = sample_topk_block(count, sample_uniforms[sequence], sh);
      continuation = sh.token;
    }
    break;
  }

  if (status == SPEC_REJECTION_OK &&
      accepted == static_cast<uint32_t>(drafts)) {
    const float *packed =
        packed_target + (sequence_row + drafts) * packed_width;
    status = prepare_topk_block(
        packed, packed_k, target_top_k[sequence], vocab, inverse_temperature,
        top_p[sequence], min_p[sequence],
        row_normalizer(full_normalizers, sequence_row + drafts), sh);
    if (status == SPEC_REJECTION_OK) {
      status = sample_topk_block(sh.count, sample_uniforms[sequence], sh);
      continuation = sh.token;
    }
  }

  if (tid == 0) {
    if (status == SPEC_REJECTION_OK) {
      outcomes[sequence * 3] = accepted;
      outcomes[sequence * 3 + 1] = continuation;
    } else {
      outcomes[sequence * 3] = SPEC_REJECTION_INVALID_VALUE;
      outcomes[sequence * 3 + 1] = SPEC_REJECTION_INVALID_VALUE;
    }
    outcomes[sequence * 3 + 2] = status;
  }
}

} // namespace

extern "C" void sparse_rejection_categorical_f32(
    const float *target_logits, float *chunk_stats,
    const uint32_t *draft_tokens, const uint32_t *q_token_ids,
    const float *q_probs, const float *inverse_temperatures,
    const uint32_t *target_top_k, const float *top_p, const float *min_p,
    const float *accept_uniforms, const float *sample_uniforms,
    uint32_t *outcomes, int batch, int drafts, int vocab, int q_width,
    int chunks, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  float2 *stats = reinterpret_cast<float2 *>(chunk_stats);
  const dim3 chunk_grid(chunks, batch * (drafts + 1));
  categorical_chunk_stats_kernel<<<chunk_grid, SPEC_CHUNK_BLOCK_SIZE, 0,
                                   cuda_stream>>>(
      target_logits, inverse_temperatures, stats, drafts + 1, vocab, chunks);
  sparse_rejection_categorical_f32_kernel<<<batch, SPEC_REJECTION_BLOCK_SIZE, 0,
                                            cuda_stream>>>(
      target_logits, stats, draft_tokens, q_token_ids, q_probs,
      inverse_temperatures, target_top_k, top_p, min_p, accept_uniforms,
      sample_uniforms, outcomes, batch, drafts, vocab, q_width, chunks);
}

extern "C" void sparse_rejection_topk_f32(
    const float *packed_target, const uint32_t *draft_tokens,
    const uint32_t *q_token_ids, const float *q_probs,
    const float *inverse_temperatures, const uint32_t *target_top_k,
    const float *top_p, const float *min_p, const float *accept_uniforms,
    const float *sample_uniforms, const float *full_normalizers,
    uint32_t *outcomes, int batch, int drafts, int vocab, int q_width,
    int packed_k, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  sparse_rejection_topk_f32_kernel<<<batch, SPEC_TOPK_BLOCK_SIZE, 0,
                                     cuda_stream>>>(
      packed_target, draft_tokens, q_token_ids, q_probs, inverse_temperatures,
      target_top_k, top_p, min_p, accept_uniforms, sample_uniforms,
      full_normalizers, outcomes, batch, drafts, vocab, q_width, packed_k);
}
