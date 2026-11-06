#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int SPEC_REJECTION_BLOCK_SIZE = 256;
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

__device__ __forceinline__ float block_reduce_max(float value, float *scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ float block_reduce_sum(float value, float *scratch) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ bool categorical_row_stats(const float *logits, const int vocab,
                                      const float inverse_temperature,
                                      float *scratch, float *row_max,
                                      float *denominator) {
  float local_max = -INFINITY;
  bool invalid = false;
  for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
    const float value = logits[token];
    invalid |= isnan(value) || value == INFINITY;
    local_max = fmaxf(local_max, value * inverse_temperature);
  }
  const bool row_invalid = __syncthreads_or(invalid);
  const float global_max = block_reduce_max(local_max, scratch);
  if (row_invalid || !isfinite(global_max)) {
    return false;
  }

  float local_sum = 0.0f;
  for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
    local_sum += expf(logits[token] * inverse_temperature - global_max);
  }
  const float sum = block_reduce_sum(local_sum, scratch);
  if (!(sum > 0.0f) || !isfinite(sum)) {
    return false;
  }
  if (threadIdx.x == 0) {
    *row_max = global_max;
    *denominator = sum;
  }
  __syncthreads();
  return true;
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

__device__ SpecRejectionStatus prepare_sorted_q(
    const uint32_t *input_ids, const float *input_probs, const int vocab,
    const int q_width, const float target_denominator, uint32_t *sorted_ids,
    float *scaled_probs, int *unique_count) {
  double q_sum = 0.0;
  for (int slot = 0; slot < q_width; ++slot) {
    const uint32_t id = input_ids[slot];
    const float probability = input_probs[slot];
    if (id >= static_cast<uint32_t>(vocab) || !isfinite(probability) ||
        probability < 0.0f) {
      return SPEC_REJECTION_INVALID_Q;
    }
    q_sum += static_cast<double>(probability);
  }
  if (!(q_sum > 0.0) || !isfinite(q_sum)) {
    return SPEC_REJECTION_INVALID_Q;
  }

  int count = 0;
  for (int slot = 0; slot < q_width; ++slot) {
    const uint32_t id = input_ids[slot];
    const float probability =
        static_cast<float>(static_cast<double>(input_probs[slot]) / q_sum);
    if (probability == 0.0f) {
      continue;
    }

    int duplicate = -1;
    for (int index = 0; index < count; ++index) {
      if (sorted_ids[index] == id) {
        duplicate = index;
        break;
      }
    }
    if (duplicate >= 0) {
      scaled_probs[duplicate] += probability;
      continue;
    }

    int position = count;
    while (position > 0 && sorted_ids[position - 1] > id) {
      sorted_ids[position] = sorted_ids[position - 1];
      scaled_probs[position] = scaled_probs[position - 1];
      --position;
    }
    sorted_ids[position] = id;
    scaled_probs[position] = probability;
    ++count;
  }
  for (int index = 0; index < count; ++index) {
    scaled_probs[index] *= target_denominator;
  }
  *unique_count = count;
  return SPEC_REJECTION_OK;
}

__device__ __forceinline__ float categorical_residual_weight(
    const float *logits, const int token, const float inverse_temperature,
    const float row_max, const uint32_t *q_ids, const float *q_scaled,
    const int q_count, int *q_cursor) {
  float weight = expf(logits[token] * inverse_temperature - row_max);
  while (*q_cursor < q_count &&
         q_ids[*q_cursor] < static_cast<uint32_t>(token)) {
    ++*q_cursor;
  }
  if (*q_cursor < q_count && q_ids[*q_cursor] == static_cast<uint32_t>(token)) {
    weight = fmaxf(weight - q_scaled[*q_cursor], 0.0f);
    ++*q_cursor;
  }
  return weight;
}

__device__ bool select_categorical_segment(const float *segment_sums,
                                           const float uniform,
                                           int *selected_segment,
                                           float *segment_target) {
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int segment = 0; segment < blockDim.x; ++segment) {
      total += segment_sums[segment];
    }
    *selected_segment = -1;
    if (total > 0.0f && isfinite(total)) {
      const float target = fminf(uniform * total, nextafterf(total, -INFINITY));
      float cumulative = 0.0f;
      for (int segment = 0; segment < blockDim.x; ++segment) {
        const float next = cumulative + segment_sums[segment];
        if (target < next) {
          *selected_segment = segment;
          *segment_target = target - cumulative;
          break;
        }
        cumulative = next;
      }
    }
  }
  __syncthreads();
  return *selected_segment >= 0;
}

__device__ SpecRejectionStatus sample_categorical_row(
    const float *logits, const int vocab, const float inverse_temperature,
    const float row_max, const float uniform, const uint32_t *q_ids,
    const float *q_scaled, const int q_count, float *segment_sums,
    int *selected_segment, float *segment_target, uint32_t *output_token) {
  if (!isfinite(uniform) || uniform < 0.0f || uniform >= 1.0f) {
    return SPEC_REJECTION_INVALID_RNG;
  }

  const int segment_width = (vocab - 1) / blockDim.x + 1;
  const int start = threadIdx.x * segment_width;
  const int end = min(start + segment_width, vocab);
  int effective_q_count = q_count;
  int q_cursor = 0;
  while (q_cursor < effective_q_count &&
         q_ids[q_cursor] < static_cast<uint32_t>(start)) {
    ++q_cursor;
  }
  float local_sum = 0.0f;
  for (int token = start; token < end; ++token) {
    local_sum += categorical_residual_weight(logits, token, inverse_temperature,
                                             row_max, q_ids, q_scaled,
                                             effective_q_count, &q_cursor);
  }
  segment_sums[threadIdx.x] = local_sum;
  __syncthreads();

  bool selected = select_categorical_segment(segment_sums, uniform,
                                             selected_segment, segment_target);
  if (!selected && effective_q_count > 0) {
    effective_q_count = 0;
    local_sum = 0.0f;
    for (int token = start; token < end; ++token) {
      local_sum += expf(logits[token] * inverse_temperature - row_max);
    }
    segment_sums[threadIdx.x] = local_sum;
    __syncthreads();
    selected = select_categorical_segment(segment_sums, uniform,
                                          selected_segment, segment_target);
  }
  if (!selected) {
    return SPEC_REJECTION_INVALID_TARGET;
  }

  if (threadIdx.x == *selected_segment) {
    int selected_q = 0;
    while (selected_q < effective_q_count &&
           q_ids[selected_q] < static_cast<uint32_t>(start)) {
      ++selected_q;
    }
    float cumulative = 0.0f;
    int last_positive = -1;
    for (int token = start; token < end; ++token) {
      const float weight = categorical_residual_weight(
          logits, token, inverse_temperature, row_max, q_ids, q_scaled,
          effective_q_count, &selected_q);
      if (weight > 0.0f) {
        last_positive = token;
      }
      cumulative += weight;
      if (*output_token == SPEC_REJECTION_INVALID_VALUE &&
          *segment_target < cumulative) {
        *output_token = static_cast<uint32_t>(token);
      }
    }
    if (*output_token == SPEC_REJECTION_INVALID_VALUE && last_positive >= 0) {
      *output_token = static_cast<uint32_t>(last_positive);
    }
  }
  __syncthreads();
  return *output_token == SPEC_REJECTION_INVALID_VALUE
             ? SPEC_REJECTION_INVALID_TARGET
             : SPEC_REJECTION_OK;
}

__global__ void sparse_rejection_categorical_f32_kernel(
    const float *__restrict__ target_logits,
    const uint32_t *__restrict__ draft_tokens,
    const uint32_t *__restrict__ q_token_ids, const float *__restrict__ q_probs,
    const float *__restrict__ inverse_temperatures,
    const uint32_t *__restrict__ target_top_k, const float *__restrict__ top_p,
    const float *__restrict__ min_p, const float *__restrict__ accept_uniforms,
    const float *__restrict__ sample_uniforms, uint32_t *__restrict__ outcomes,
    const int batch, const int drafts, const int vocab, const int q_width) {
  const int sequence = blockIdx.x;
  if (sequence >= batch) {
    return;
  }

  __shared__ float scratch[SPEC_REJECTION_BLOCK_SIZE];
  __shared__ uint32_t sorted_q_ids[SPEC_REJECTION_MAX_Q];
  __shared__ float scaled_q_probs[SPEC_REJECTION_MAX_Q];
  __shared__ float row_max;
  __shared__ float denominator;
  __shared__ float segment_target;
  __shared__ uint32_t continuation;
  __shared__ uint32_t accepted;
  __shared__ uint32_t rejection_row;
  __shared__ uint32_t row_status;
  __shared__ int q_count;
  __shared__ int selected_segment;

  if (threadIdx.x == 0) {
    continuation = SPEC_REJECTION_INVALID_VALUE;
    accepted = 0;
    rejection_row = SPEC_REJECTION_INVALID_VALUE;
    row_status = SPEC_REJECTION_OK;
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
  for (int position = 0; position < drafts; ++position) {
    const float *row = target_logits + (sequence_row + position) * row_stride;
    if (!categorical_row_stats(row, vocab, inverse_temperature, scratch,
                               &row_max, &denominator)) {
      if (threadIdx.x == 0) {
        row_status = SPEC_REJECTION_INVALID_TARGET;
      }
      __syncthreads();
      break;
    }

    if (threadIdx.x == 0) {
      const size_t draft_offset =
          static_cast<size_t>(sequence) * drafts + position;
      const uint32_t draft = draft_tokens[draft_offset];
      if (draft >= static_cast<uint32_t>(vocab)) {
        row_status = SPEC_REJECTION_INVALID_Q;
      } else {
        const size_t q_offset = draft_offset * q_width;
        double q_sum = 0.0;
        float q_draft = 0.0f;
        row_status = validate_q_row(q_token_ids + q_offset, q_probs + q_offset,
                                    draft, vocab, q_width, &q_sum, &q_draft);
        const float uniform = accept_uniforms[draft_offset];
        if (row_status == SPEC_REJECTION_OK &&
            (!isfinite(uniform) || uniform < 0.0f || uniform >= 1.0f)) {
          row_status = SPEC_REJECTION_INVALID_RNG;
        }
        if (row_status == SPEC_REJECTION_OK) {
          const float target_probability =
              expf(row[draft] * inverse_temperature - row_max) / denominator;
          const float draft_probability = q_draft;
          const float accept_probability =
              fminf(target_probability / draft_probability, 1.0f);
          if (uniform < accept_probability) {
            accepted = static_cast<uint32_t>(position + 1);
          } else {
            rejection_row = static_cast<uint32_t>(position);
          }
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
    const float *row = target_logits + (sequence_row + drafts) * row_stride;
    if (!categorical_row_stats(row, vocab, inverse_temperature, scratch,
                               &row_max, &denominator)) {
      if (threadIdx.x == 0) {
        row_status = SPEC_REJECTION_INVALID_TARGET;
      }
    } else if (threadIdx.x == 0) {
      q_count = 0;
    }
    __syncthreads();
  } else if (row_status == SPEC_REJECTION_OK) {
    if (threadIdx.x == 0) {
      const size_t q_offset =
          (static_cast<size_t>(sequence) * drafts + rejection_row) * q_width;
      row_status = prepare_sorted_q(q_token_ids + q_offset, q_probs + q_offset,
                                    vocab, q_width, denominator, sorted_q_ids,
                                    scaled_q_probs, &q_count);
    }
    __syncthreads();
  }

  if (row_status == SPEC_REJECTION_OK) {
    const int selected_row = rejection_row == SPEC_REJECTION_INVALID_VALUE
                                 ? drafts
                                 : static_cast<int>(rejection_row);
    const float *row =
        target_logits + (sequence_row + selected_row) * row_stride;
    const SpecRejectionStatus sample_status = sample_categorical_row(
        row, vocab, inverse_temperature, row_max, sample_uniforms[sequence],
        sorted_q_ids, scaled_q_probs, q_count, scratch, &selected_segment,
        &segment_target, &continuation);
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
    const float *target_logits, const uint32_t *draft_tokens,
    const uint32_t *q_token_ids, const float *q_probs,
    const float *inverse_temperatures, const uint32_t *target_top_k,
    const float *top_p, const float *min_p, const float *accept_uniforms,
    const float *sample_uniforms, uint32_t *outcomes, int batch, int drafts,
    int vocab, int q_width, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  sparse_rejection_categorical_f32_kernel<<<batch, SPEC_REJECTION_BLOCK_SIZE, 0,
                                            cuda_stream>>>(
      target_logits, draft_tokens, q_token_ids, q_probs, inverse_temperatures,
      target_top_k, top_p, min_p, accept_uniforms, sample_uniforms, outcomes,
      batch, drafts, vocab, q_width);
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
