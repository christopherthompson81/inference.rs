#include "gdn_common.cuh"

template <typename T>
__global__ void gdn_speculative_conv_state_commit_kernel(
    const T *__restrict__ x, const T *__restrict__ initial_state,
    T *__restrict__ state_pool, const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ slot_indices, int batch_size, int seq_len,
    int conv_dim, int kernel_size) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (channel >= conv_dim || batch_idx >= batch_size) {
    return;
  }

  const int rows = (int)keep_rows[batch_idx];
  if (rows == 0) {
    return;
  }

  const T *prior = initial_state +
                   ((size_t)batch_idx * conv_dim + channel) * kernel_size;
  T *destination =
      state_pool +
      ((size_t)slot_indices[batch_idx] * conv_dim + channel) * kernel_size;
  const int pad = kernel_size - rows;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      destination[i] = prior[i + rows];
    } else {
      const int position = rows - kernel_size + i;
      destination[i] =
          x[((size_t)batch_idx * seq_len + position) * conv_dim + channel];
    }
  }
}

template <typename T, typename StateT, bool VALUE_MAJOR>
__global__ void gdn_speculative_recurrent_state_commit_kernel(
    const T *__restrict__ convolved_qkv, const T *__restrict__ b,
    const T *__restrict__ a,
    const StateT *__restrict__ initial_recurrent_state,
    const float *__restrict__ a_log, const float *__restrict__ dt_bias,
    StateT *__restrict__ recurrent_state_pool,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ slot_indices, int batch_size, int seq_len,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads) {
  constexpr int WARP_SIZE = 32;
  constexpr int K_PER_LANE = GDN_SPEC_COMMIT_MAX_K / WARP_SIZE;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int value_idx = blockIdx.x * GDN_SPEC_COMMIT_WARPS + warp;
  const int batch_head = blockIdx.y;
  const int batch_idx = batch_head / num_v_heads;
  const int value_head = batch_head - batch_idx * num_v_heads;
  if (batch_idx >= batch_size) {
    return;
  }

  const int rows = (int)keep_rows[batch_idx];
  if (rows == 0) {
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head =
      tiled_v_heads ? value_head % num_k_heads : value_head / values_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t initial_head =
      ((size_t)batch_idx * num_v_heads + value_head) * head_k_dim *
      head_v_dim;
  const size_t destination_head =
      ((size_t)slot_indices[batch_idx] * num_v_heads + value_head) *
      head_k_dim * head_v_dim;
  const int thread_idx = warp * WARP_SIZE + lane;

  __shared__ float key_buffer[GDN_SPEC_COMMIT_MAX_K];
  __shared__ float norm_buffer[WARP_SIZE * GDN_SPEC_COMMIT_WARPS];
  __shared__ float key_multiplier;
  __shared__ float beta;
  __shared__ float decay;
  __shared__ float values[GDN_SPEC_COMMIT_WARPS];

  float state[K_PER_LANE];
#pragma unroll
  for (int r = 0; r < K_PER_LANE; r++) {
    const int key_idx = r * WARP_SIZE + lane;
    if (key_idx < head_k_dim) {
      const size_t offset = VALUE_MAJOR
                                ? (size_t)value_idx * head_k_dim + key_idx
                                : (size_t)key_idx * head_v_dim + value_idx;
      state[r] = initial_recurrent_state[initial_head + offset];
    }
  }

  for (int position = 0; position < rows; position++) {
    float key_norm_partial = 0.0f;
    for (int key_idx = thread_idx; key_idx < head_k_dim;
         key_idx += WARP_SIZE * GDN_SPEC_COMMIT_WARPS) {
      const int channel = key_dim + key_head * head_k_dim + key_idx;
      const float value =
          (float)convolved_qkv[((size_t)batch_idx * seq_len + position) *
                                   conv_dim +
                               channel];
      key_buffer[key_idx] = value;
      key_norm_partial = __fmaf_rn(value, value, key_norm_partial);
    }
    norm_buffer[thread_idx] = key_norm_partial;
    __syncthreads();
    for (int stride = (WARP_SIZE * GDN_SPEC_COMMIT_WARPS) / 2; stride > 0;
         stride >>= 1) {
      if (thread_idx < stride) {
        norm_buffer[thread_idx] += norm_buffer[thread_idx + stride];
      }
      __syncthreads();
    }

    if (thread_idx == 0) {
      key_multiplier = rsqrtf(norm_buffer[0] + 1.0e-6f);
      const size_t gate_offset =
          ((size_t)batch_idx * seq_len + position) * num_v_heads +
          value_head;
      const float b_value = (float)b[gate_offset];
      const float a_value = (float)a[gate_offset] + dt_bias[value_head];
      const float softplus =
          a_value > 20.0f
              ? a_value
              : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                                : log1pf(expf(a_value)));
      beta = 1.0f / (1.0f + expf(-b_value));
      decay = expf(-expf(a_log[value_head]) * softplus);
    }
    if (lane == 0) {
      const int channel = 2 * key_dim + value_head * head_v_dim + value_idx;
      values[warp] =
          (float)convolved_qkv[((size_t)batch_idx * seq_len + position) *
                                   conv_dim +
                               channel];
    }
    __syncthreads();

    float state_dot_key = 0.0f;
#pragma unroll
    for (int r = 0; r < K_PER_LANE; r++) {
      const int key_idx = r * WARP_SIZE + lane;
      if (key_idx < head_k_dim) {
        const float key = key_buffer[key_idx] * key_multiplier;
        state_dot_key = __fmaf_rn(state[r], key, state_dot_key);
      }
    }
    state_dot_key = gdn_warp_sum<WARP_SIZE>(state_dot_key);
    const float delta = (values[warp] - decay * state_dot_key) * beta;

#pragma unroll
    for (int r = 0; r < K_PER_LANE; r++) {
      const int key_idx = r * WARP_SIZE + lane;
      if (key_idx < head_k_dim) {
        const float key = key_buffer[key_idx] * key_multiplier;
        state[r] = __fmaf_rn(key, delta, decay * state[r]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < K_PER_LANE; r++) {
    const int key_idx = r * WARP_SIZE + lane;
    if (key_idx < head_k_dim) {
      const size_t offset = VALUE_MAJOR
                                ? (size_t)value_idx * head_k_dim + key_idx
                                : (size_t)key_idx * head_v_dim + value_idx;
      recurrent_state_pool[destination_head + offset] = state[r];
    }
  }
}

template <typename T, typename StateT>
void launch_gdn_speculative_state_commit(
    const T *mixed_qkv, const T *convolved_qkv, const T *b, const T *a,
    const T *initial_conv_state, const StateT *initial_recurrent_state,
    const float *a_log, const float *dt_bias, T *conv_state_pool,
    StateT *recurrent_state_pool, const uint32_t *keep_rows,
    const uint32_t *slot_indices, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int kernel_size,
    int tiled_v_heads, int value_major, cudaStream_t stream) {
  dim3 conv_block(GDN_CHANNEL_BLOCK_SIZE);
  const int conv_dim =
      2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
  dim3 conv_grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) /
                     GDN_CHANNEL_BLOCK_SIZE,
                 batch_size);
  gdn_speculative_conv_state_commit_kernel<T>
      <<<conv_grid, conv_block, 0, stream>>>(
          mixed_qkv, initial_conv_state, conv_state_pool, keep_rows,
          slot_indices, batch_size, seq_len, conv_dim, kernel_size);

  dim3 recurrence_block(32, GDN_SPEC_COMMIT_WARPS);
  dim3 recurrence_grid(
      (head_v_dim + GDN_SPEC_COMMIT_WARPS - 1) / GDN_SPEC_COMMIT_WARPS,
      batch_size * num_v_heads);
  if (value_major) {
    gdn_speculative_recurrent_state_commit_kernel<T, StateT, true>
        <<<recurrence_grid, recurrence_block, 0, stream>>>(
            convolved_qkv, b, a, initial_recurrent_state, a_log, dt_bias,
            recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            tiled_v_heads);
  } else {
    gdn_speculative_recurrent_state_commit_kernel<T, StateT, false>
        <<<recurrence_grid, recurrence_block, 0, stream>>>(
            convolved_qkv, b, a, initial_recurrent_state, a_log, dt_bias,
            recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            tiled_v_heads);
  }
}

template <typename T>
void dispatch_gdn_speculative_state_commit(
    const T *mixed_qkv, const T *convolved_qkv, const T *b, const T *a,
    const T *initial_conv_state, const void *initial_recurrent_state,
    const float *a_log, const float *dt_bias, T *conv_state_pool,
    void *recurrent_state_pool, const uint32_t *keep_rows,
    const uint32_t *slot_indices, int batch_size, int seq_len, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int kernel_size,
    int tiled_v_heads, int value_major, int state_dtype, cudaStream_t stream) {
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const __half *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (__half *)recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        stream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const __nv_bfloat16 *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (__nv_bfloat16 *)recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        stream);
  } else {
    launch_gdn_speculative_state_commit(
        mixed_qkv, convolved_qkv, b, a, initial_conv_state,
        (const float *)initial_recurrent_state, a_log, dt_bias,
        conv_state_pool, (float *)recurrent_state_pool, keep_rows, slot_indices,
        batch_size, seq_len, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        kernel_size, tiled_v_heads, value_major, stream);
  }
}

extern "C" void gdn_speculative_state_commit(
    const void *mixed_qkv, const void *convolved_qkv, const void *b,
    const void *a, const void *initial_conv_state,
    const void *initial_recurrent_state, const float *a_log,
    const float *dt_bias, void *conv_state_pool, void *recurrent_state_pool,
    const uint32_t *keep_rows, const uint32_t *slot_indices, int batch_size,
    int seq_len, int num_k_heads, int num_v_heads, int head_k_dim,
    int head_v_dim, int kernel_size, int tiled_v_heads, int value_major,
    int dtype, int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    dispatch_gdn_speculative_state_commit(
        (const __half *)mixed_qkv, (const __half *)convolved_qkv,
        (const __half *)b, (const __half *)a,
        (const __half *)initial_conv_state, initial_recurrent_state, a_log,
        dt_bias, (__half *)conv_state_pool, recurrent_state_pool, keep_rows,
        slot_indices, batch_size, seq_len, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, kernel_size, tiled_v_heads, value_major,
        state_dtype, custream);
  } else {
    dispatch_gdn_speculative_state_commit(
        (const __nv_bfloat16 *)mixed_qkv,
        (const __nv_bfloat16 *)convolved_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a,
        (const __nv_bfloat16 *)initial_conv_state, initial_recurrent_state,
        a_log, dt_bias, (__nv_bfloat16 *)conv_state_pool,
        recurrent_state_pool, keep_rows, slot_indices, batch_size, seq_len,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, kernel_size,
        tiled_v_heads, value_major, state_dtype, custream);
  }
}

template <typename T>
__global__ void gdn_speculative_conv_checkpoints_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ state_pool, T *__restrict__ output,
    const uint32_t *__restrict__ active_slots, int batch_size, int seq_len,
    int conv_dim, int kernel_size, int checkpoint_lanes,
    bool write_checkpoints, T *pending_conv_input,
    const uint32_t *__restrict__ pending_keep_rows,
    const uint32_t *__restrict__ pending_epochs,
    uint32_t *__restrict__ conv_applied_epochs, int max_pending_rows,
    int pending_capacity, int conv_blocks, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }
  const bool channel_valid = channel < conv_dim;

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT) {
    for (int position = 0; channel_valid && position < seq_len; position++) {
      output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
          (T)0.0f;
    }
    return;
  }

  T state[GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH];
  T *source = channel_valid
                  ? state_pool +
                        ((size_t)active_slot * conv_dim + channel) * kernel_size
                  : nullptr;
  if (channel_valid) {
    for (int i = 0; i < kernel_size; i++) {
      state[i] = source[i];
    }
  }
  const int conv_block = blockIdx.x;
  uint32_t pending_epoch = 0;
  int pending_rows = 0;
  bool apply_pending = false;
  if (pending_epochs != nullptr && active_slot < pending_capacity) {
    pending_epoch = pending_epochs[active_slot];
    pending_rows = (int)pending_keep_rows[active_slot];
    apply_pending = pending_epoch != 0 && pending_rows > 0 &&
                    pending_rows <= max_pending_rows &&
                    conv_applied_epochs[(size_t)active_slot * conv_blocks +
                                        conv_block] != pending_epoch;
  }
  if (channel_valid && apply_pending) {
    for (int position = 0; position < pending_rows; position++) {
      for (int i = 0; i < kernel_size - 1; i++) {
        state[i] = state[i + 1];
      }
      state[kernel_size - 1] =
          pending_conv_input[((size_t)active_slot * max_pending_rows +
                              position) *
                                 conv_dim +
                             channel];
    }
    for (int i = 0; i < kernel_size; i++) {
      source[i] = state[i];
    }
  }
  if (pending_epochs != nullptr) {
    __syncthreads();
    if (threadIdx.x == 0 && apply_pending) {
      conv_applied_epochs[(size_t)active_slot * conv_blocks + conv_block] =
          pending_epoch;
    }
  }
  if (!channel_valid) {
    return;
  }
  const T *channel_weight = weight + (size_t)channel * kernel_size;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);

  for (int position = 0; position < seq_len; position++) {
    for (int i = 0; i < kernel_size - 1; i++) {
      state[i] = state[i + 1];
    }
    const T input =
        x[(size_t)batch_idx * x_stride_b + (size_t)position * x_stride_s +
          (size_t)channel * x_stride_c];
    state[kernel_size - 1] = input;
    if (!write_checkpoints && pending_conv_input != nullptr) {
      pending_conv_input[((size_t)active_slot * max_pending_rows + position) *
                             conv_dim +
                         channel] = input;
    }

    float acc = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
      acc = __fmaf_rn((float)state[i], (float)channel_weight[i], acc);
    }
    const float result = acc / (1.0f + expf(-acc));
    output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
        (T)result;

    if (write_checkpoints) {
      T *destination =
          state_pool +
          (((base_slot + position) * conv_dim + channel) * kernel_size);
      for (int i = 0; i < kernel_size; i++) {
        destination[i] = state[i];
      }
    }
  }
}

template <typename T>
__global__ void gdn_speculative_conv_checkpoints_width4_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ state_pool, T *__restrict__ output,
    const uint32_t *__restrict__ active_slots, int batch_size, int seq_len,
    int conv_dim, int checkpoint_lanes, bool write_checkpoints,
    T *pending_conv_input, const uint32_t *__restrict__ pending_keep_rows,
    const uint32_t *__restrict__ pending_epochs,
    uint32_t *__restrict__ conv_applied_epochs, int max_pending_rows,
    int pending_capacity, int conv_blocks, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }
  const bool channel_valid = channel < conv_dim;

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT) {
    for (int position = 0; channel_valid && position < seq_len; position++) {
      output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
          (T)0.0f;
    }
    return;
  }

  auto *states = reinterpret_cast<GdnConvWidth4<T> *>(state_pool);
  const auto *weights = reinterpret_cast<const GdnConvWidth4<T> *>(weight);
  GdnConvWidth4<T> state;
  if (channel_valid) {
    state = states[(size_t)active_slot * conv_dim + channel];
  }
  const int conv_block = blockIdx.x;
  uint32_t pending_epoch = 0;
  int pending_rows = 0;
  bool apply_pending = false;
  if (pending_epochs != nullptr && active_slot < pending_capacity) {
    pending_epoch = pending_epochs[active_slot];
    pending_rows = (int)pending_keep_rows[active_slot];
    apply_pending = pending_epoch != 0 && pending_rows > 0 &&
                    pending_rows <= max_pending_rows &&
                    conv_applied_epochs[(size_t)active_slot * conv_blocks +
                                        conv_block] != pending_epoch;
  }
  if (channel_valid && apply_pending) {
    for (int position = 0; position < pending_rows; position++) {
#pragma unroll
      for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
        state.values[i] = state.values[i + 1];
      }
      state.values[GDN_PACKED_CONV_WIDTH - 1] =
          pending_conv_input[((size_t)active_slot * max_pending_rows +
                              position) *
                                 conv_dim +
                             channel];
    }
    states[(size_t)active_slot * conv_dim + channel] = state;
  }
  if (pending_epochs != nullptr) {
    __syncthreads();
    if (threadIdx.x == 0 && apply_pending) {
      conv_applied_epochs[(size_t)active_slot * conv_blocks + conv_block] =
          pending_epoch;
    }
  }
  if (!channel_valid) {
    return;
  }
  const GdnConvWidth4<T> channel_weight = weights[channel];
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);

  for (int position = 0; position < seq_len; position++) {
#pragma unroll
    for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
      state.values[i] = state.values[i + 1];
    }
    const T input =
        x[(size_t)batch_idx * x_stride_b + (size_t)position * x_stride_s +
          (size_t)channel * x_stride_c];
    state.values[GDN_PACKED_CONV_WIDTH - 1] = input;
    if (!write_checkpoints && pending_conv_input != nullptr) {
      pending_conv_input[((size_t)active_slot * max_pending_rows + position) *
                             conv_dim +
                         channel] = input;
    }

    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < GDN_PACKED_CONV_WIDTH; i++) {
      acc = __fmaf_rn((float)state.values[i],
                      (float)channel_weight.values[i], acc);
    }
    const float result = acc / (1.0f + expf(-acc));
    output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
        (T)result;
    if (write_checkpoints) {
      states[(base_slot + position) * conv_dim + channel] = state;
    }
  }
}

template <typename T>
void launch_gdn_speculative_conv_checkpoints(
    const T *x, const T *weight, T *state_pool, T *output,
    const uint32_t *active_slots, int batch_size, int seq_len, int conv_dim,
    int kernel_size, int checkpoint_lanes, bool write_checkpoints,
    T *pending_conv_input, const uint32_t *pending_keep_rows,
    const uint32_t *pending_epochs, uint32_t *conv_applied_epochs,
    int max_pending_rows, int pending_capacity, int conv_blocks,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    cudaStream_t stream) {
  if (pending_conv_input != nullptr &&
      (write_checkpoints || max_pending_rows < seq_len ||
       max_pending_rows > GDN_SPEC_FUSED_MAX_TOKENS ||
       pending_capacity <= 0 || conv_blocks <= 0 ||
       pending_keep_rows == nullptr || pending_epochs == nullptr ||
       conv_applied_epochs == nullptr)) {
    return;
  }
  dim3 block(GDN_CHANNEL_BLOCK_SIZE);
  dim3 grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) /
                GDN_CHANNEL_BLOCK_SIZE,
            batch_size);
  if (kernel_size == GDN_PACKED_CONV_WIDTH) {
    gdn_speculative_conv_checkpoints_width4_kernel<T>
        <<<grid, block, 0, stream>>>(
            x, weight, state_pool, output, active_slots, batch_size, seq_len,
            conv_dim, checkpoint_lanes, write_checkpoints, pending_conv_input,
            pending_keep_rows, pending_epochs, conv_applied_epochs,
            max_pending_rows, pending_capacity, conv_blocks, x_stride_b,
            x_stride_s, x_stride_c);
  } else {
    gdn_speculative_conv_checkpoints_kernel<T><<<grid, block, 0, stream>>>(
        x, weight, state_pool, output, active_slots, batch_size, seq_len,
        conv_dim, kernel_size, checkpoint_lanes, write_checkpoints,
        pending_conv_input, pending_keep_rows, pending_epochs,
        conv_applied_epochs, max_pending_rows, pending_capacity, conv_blocks,
        x_stride_b, x_stride_s, x_stride_c);
  }
}

extern "C" void gdn_speculative_conv_checkpoints(
    const void *x, const void *weight, void *state_pool, void *output,
    const uint32_t *active_slots, int batch_size, int seq_len, int conv_dim,
    int kernel_size, int checkpoint_lanes, int write_checkpoints,
    void *pending_conv_input, const uint32_t *pending_keep_rows,
    const uint32_t *pending_epochs, uint32_t *conv_applied_epochs,
    int max_pending_rows, int pending_capacity, int conv_blocks,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c, int dtype,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    launch_gdn_speculative_conv_checkpoints(
        (const __half *)x, (const __half *)weight, (__half *)state_pool,
        (__half *)output, active_slots, batch_size, seq_len, conv_dim,
        kernel_size, checkpoint_lanes, write_checkpoints != 0,
        (__half *)pending_conv_input, pending_keep_rows, pending_epochs,
        conv_applied_epochs, max_pending_rows, pending_capacity, conv_blocks,
        x_stride_b, x_stride_s, x_stride_c, custream);
  } else {
    launch_gdn_speculative_conv_checkpoints(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (__nv_bfloat16 *)state_pool, (__nv_bfloat16 *)output, active_slots,
        batch_size, seq_len, conv_dim, kernel_size, checkpoint_lanes,
        write_checkpoints != 0,
        (__nv_bfloat16 *)pending_conv_input, pending_keep_rows,
        pending_epochs, conv_applied_epochs, max_pending_rows,
        pending_capacity, conv_blocks, x_stride_b, x_stride_s, x_stride_c,
        custream);
  }
}
