#include "gdn_common.cuh"

// ============================================================================
// Kernel 1: gated_delta_rule_recurrence (optimized)
//
// V-tiled recurrence with compile-time K dimension for register residency.
// Grid: (ceil(V/BV), B*H), Block: (BV,). Each thread owns BK registers of
// state. Shared memory holds k_buf and q_buf (2*BK floats).
//
// Optimizations over naive version:
//   - Template BK -> float s[BK] lives in true registers (1 cycle vs ~30)
//   - #pragma unroll on all k-loops -> full ILP
//   - Fused decay+kv_mem pass and fused state_update+output pass
//   - __fmaf_rn intrinsics for guaranteed fused multiply-add
//   - BV=64 threads -> 2 warps, 6 blocks/SM on Ampere
//
// q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
// state: [BH, K, V] (in/out)  output: [BH, S, V]
// ============================================================================

// Optimized kernel: BK known at compile time -> registers + full unrolling
template <typename StateT, int BK, int BV>
__global__ void gated_delta_rule_recurrence_kernel_tiled(
    const float *__restrict__ q,    // [BH, S, K]
    const float *__restrict__ k,    // [BH, S, K]
    const float *__restrict__ v,    // [BH, S, V]
    const float *__restrict__ g,    // [BH, S]
    const float *__restrict__ beta, // [BH, S]
    StateT *__restrict__ state,     // [BH, K, V] or the pool with slot_indices
    float *__restrict__ output,     // [BH, S, V]
    int seq_len, int v_dim, const int32_t *__restrict__ slot_indices,
    int num_heads) {

  const int v_tile = blockIdx.x;       // which V-tile
  const int bh = blockIdx.y;           // batch*head index
  const int tid = threadIdx.x;         // thread within tile [0, BV)
  const int v_idx = v_tile * BV + tid; // global V index

  if (v_idx >= v_dim)
    return;

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int t = 0; t < seq_len; t++) {
      out_bh[t * v_dim + v_idx] = 0.0f;
    }
    return;
  }

  // Pointers for this (batch, head)
  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  // Shared memory: k_buf[BK] + q_buf[BK]
  __shared__ float k_buf[BK];
  __shared__ float q_buf[BK];

  // Load state column into registers — BK is compile-time, so this is
  // a true register array (not spilled to local memory)
  float s[BK];
#pragma unroll
  for (int j = 0; j < BK; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }

  for (int t = 0; t < seq_len; t++) {
// Collaboratively load k_t into shared memory
// BK / BV loads per thread (e.g. 128/64 = 2)
#pragma unroll
    for (int j = tid; j < BK; j += BV) {
      k_buf[j] = k_bh[t * BK + j];
    }
    __syncthreads();

    // Load scalars for this timestep
    float decay = expf(g_bh[t]);
    float beta_t = beta_bh[t];
    float v_t = v_bh[t * v_dim + v_idx];

    // Fused pass 1: decay state + compute kv_mem
    float kv_mem = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; j++) {
      s[j] *= decay;
      kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
    }

    // Delta rule
    float delta = (v_t - kv_mem) * beta_t;

// Collaboratively load q_t into shared memory
#pragma unroll
    for (int j = tid; j < BK; j += BV) {
      q_buf[j] = q_bh[t * BK + j];
    }
    __syncthreads();

    // Fused pass 2: update state + compute output
    float y_t = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; j++) {
      s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
      y_t = __fmaf_rn(s[j], q_buf[j], y_t);
    }

    out_bh[t * v_dim + v_idx] = y_t;

    __syncthreads();
  }

// Write state back
#pragma unroll
  for (int j = 0; j < BK; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

// Fallback kernel: runtime k_dim, still V-tiled for occupancy
template <typename StateT, int BV, int MAX_K>
__global__ void gated_delta_rule_recurrence_kernel_fallback(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, StateT *__restrict__ state,
    float *__restrict__ output, int seq_len, int k_dim, int v_dim,
    const int32_t *__restrict__ slot_indices, int num_heads) {

  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;

  if (v_idx >= v_dim)
    return;

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int t = 0; t < seq_len; t++) {
      out_bh[t * v_dim + v_idx] = 0.0f;
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * k_dim;
  const float *k_bh = k + (size_t)bh * seq_len * k_dim;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * k_dim * v_dim;

  extern __shared__ float shared[];
  float *k_buf = shared;
  float *q_buf = shared + k_dim;

  float s[MAX_K];
  for (int j = 0; j < k_dim; j++) {
    s[j] = state_bh[j * v_dim + v_idx];
  }

  for (int t = 0; t < seq_len; t++) {
    for (int j = tid; j < k_dim; j += BV) {
      k_buf[j] = k_bh[t * k_dim + j];
    }
    __syncthreads();

    float decay = expf(g_bh[t]);
    float beta_t = beta_bh[t];
    float v_t = v_bh[t * v_dim + v_idx];

    float kv_mem = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      s[j] *= decay;
      kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
    }

    float delta = (v_t - kv_mem) * beta_t;

    for (int j = tid; j < k_dim; j += BV) {
      q_buf[j] = q_bh[t * k_dim + j];
    }
    __syncthreads();

    float y_t = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
      y_t = __fmaf_rn(s[j], q_buf[j], y_t);
    }

    out_bh[t * v_dim + v_idx] = y_t;

    __syncthreads();
  }

  for (int j = 0; j < k_dim; j++) {
    state_bh[j * v_dim + v_idx] = s[j];
  }
}

template <typename StateT>
void launch_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim == 128) {
    // Fast path for Qwen3-Next (k_dim=128)
    constexpr int BK = 128;
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    gated_delta_rule_recurrence_kernel_tiled<StateT, BK, BV>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else if (k_dim == 64) {
    // Fast path for models with k_dim=64
    constexpr int BK = 64;
    constexpr int BV = 64;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    gated_delta_rule_recurrence_kernel_tiled<StateT, BK, BV>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else {
    // Fallback for other k_dim values (runtime loop, still V-tiled)
    constexpr int BV = 64;
    constexpr int MAX_K = 256;
    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    size_t smem = 2 * k_dim * sizeof(float);
    gated_delta_rule_recurrence_kernel_fallback<StateT, BV, MAX_K>
        <<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                        seq_len, k_dim, v_dim, slot_indices,
                                        num_heads);
  }
}

template void launch_gated_delta_rule_recurrence<__half>(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, __half *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream);
template void launch_gated_delta_rule_recurrence<__nv_bfloat16>(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, __nv_bfloat16 *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream);
template void launch_gated_delta_rule_recurrence<float>(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, float *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream);

extern "C" void gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT, int BK, int NUM_WARPS, bool VALUE_MAJOR = false>
__global__ __launch_bounds__(
    32 * NUM_WARPS,
    2) void gated_delta_rule_recurrence_kernel_warp(const float *__restrict__ q,
                                                    const float *__restrict__ k,
                                                    const float *__restrict__ v,
                                                    const float *__restrict__ g,
                                                    const float
                                                        *__restrict__ beta,
                                                    StateT *__restrict__ state,
                                                    float *__restrict__ output,
                                                    int seq_len, int v_dim,
                                                    const int32_t *__restrict__ slot_indices,
                                                    int num_heads) {

  constexpr int WARP_SIZE = 32;
  static_assert(BK % WARP_SIZE == 0, "BK must be a multiple of warp size");
  constexpr int ROWS_PER_LANE = BK / WARP_SIZE;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int v_idx = blockIdx.x * NUM_WARPS + warp;
  const int bh = blockIdx.y;

  if (v_idx >= v_dim) {
    return;
  }

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    if (lane == 0) {
      for (int t = 0; t < seq_len; t++) {
        out_bh[t * v_dim + v_idx] = 0.0f;
      }
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  float s[ROWS_PER_LANE];
#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; r++) {
    const int row = r * WARP_SIZE + lane;
    if constexpr (VALUE_MAJOR) {
      s[r] = state_bh[v_idx * BK + row];
    } else {
      s[r] = state_bh[row * v_dim + v_idx];
    }
  }

  for (int t = 0; t < seq_len; t++) {
    const float *q_t = q_bh + t * BK;
    const float *k_t = k_bh + t * BK;

    float k_reg[ROWS_PER_LANE];
    float q_reg[ROWS_PER_LANE];
    float kv_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
      const int row = r * WARP_SIZE + lane;
      const float k_val = k_t[row];
      k_reg[r] = k_val;
      q_reg[r] = q_t[row];
      kv_partial = __fmaf_rn(s[r], k_val, kv_partial);
    }

    const float decay = expf(g_bh[t]);
    const float kv_col = gdn_warp_sum<WARP_SIZE>(kv_partial);
    const float delta = (v_bh[t * v_dim + v_idx] - decay * kv_col) * beta_bh[t];

    float y_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
      s[r] = __fmaf_rn(k_reg[r], delta, decay * s[r]);
      y_partial = __fmaf_rn(s[r], q_reg[r], y_partial);
    }

    const float y_col = gdn_warp_sum<WARP_SIZE>(y_partial);
    if (lane == 0) {
      out_bh[t * v_dim + v_idx] = y_col;
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; r++) {
    const int row = r * WARP_SIZE + lane;
    if constexpr (VALUE_MAJOR) {
      state_bh[v_idx * BK + row] = s[r];
    } else {
      state_bh[row * v_dim + v_idx] = s[r];
    }
  }
}

template <typename StateT>
void launch_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  constexpr int NUM_WARPS = 4;
  dim3 grid((v_dim + NUM_WARPS - 1) / NUM_WARPS, bh);
  dim3 block(32, NUM_WARPS);

  if (k_dim == 128) {
    gated_delta_rule_recurrence_kernel_warp<StateT, 128, NUM_WARPS>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else if (k_dim == 64) {
    gated_delta_rule_recurrence_kernel_warp<StateT, 64, NUM_WARPS>
        <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                    v_dim, slot_indices, num_heads);
  } else {
    launch_gated_delta_rule_recurrence(q, k, v, g, beta, state, output, bh,
                                       seq_len, k_dim, v_dim, slot_indices,
                                       num_heads, stream);
  }
}

extern "C" void warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT>
void launch_vmajor_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) {
    return;
  }

  constexpr int NUM_WARPS = 4;
  dim3 grid((v_dim + NUM_WARPS - 1) / NUM_WARPS, bh);
  dim3 block(32, NUM_WARPS);
  gated_delta_rule_recurrence_kernel_warp<StateT, 128, NUM_WARPS, true>
      <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                   v_dim, slot_indices, num_heads);
}

extern "C" void vmajor_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_vmajor_warp_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT, int VALUES_PER_WARP, int NUM_WARPS>
__global__ __launch_bounds__(32 * NUM_WARPS, 2) void
gated_delta_rule_recurrence_kernel_vmajor_grouped(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, StateT *__restrict__ state,
    float *__restrict__ output, int seq_len,
    const int32_t *__restrict__ slot_indices, int num_heads) {
  constexpr int WARP_SIZE = 32;
  constexpr int BK = 128;
  constexpr int V_DIM = 128;
  constexpr int ROWS_PER_LANE = BK / WARP_SIZE;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int value_group = blockIdx.x * NUM_WARPS + warp;
  const int first_value = value_group * VALUES_PER_WARP;
  const int bh = blockIdx.y;

  if (first_value >= V_DIM) {
    return;
  }

  float *out_bh = output + (size_t)bh * seq_len * V_DIM;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    if (lane == 0) {
#pragma unroll
      for (int value = 0; value < VALUES_PER_WARP; value++) {
        for (int t = 0; t < seq_len; t++) {
          out_bh[t * V_DIM + first_value + value] = 0.0f;
        }
      }
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * V_DIM;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads,
                            num_heads) *
                  BK * V_DIM;

  float s[VALUES_PER_WARP][ROWS_PER_LANE];
#pragma unroll
  for (int value = 0; value < VALUES_PER_WARP; value++) {
#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      s[value][row] = state_bh[(first_value + value) * BK + key];
    }
  }

  for (int t = 0; t < seq_len; t++) {
    const float *q_t = q_bh + t * BK;
    const float *k_t = k_bh + t * BK;
    float k_reg[ROWS_PER_LANE];
    float q_reg[ROWS_PER_LANE];
    float kv_partial[VALUES_PER_WARP] = {};

#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      const float k_value = k_t[key];
      k_reg[row] = k_value;
      q_reg[row] = q_t[key];
#pragma unroll
      for (int value = 0; value < VALUES_PER_WARP; value++) {
        kv_partial[value] =
            __fmaf_rn(s[value][row], k_value, kv_partial[value]);
      }
    }

    const float decay = expf(g_bh[t]);
#pragma unroll
    for (int value = 0; value < VALUES_PER_WARP; value++) {
      const int value_idx = first_value + value;
      const float kv_col = gdn_warp_sum<WARP_SIZE>(kv_partial[value]);
      const float delta =
          (v_bh[t * V_DIM + value_idx] - decay * kv_col) * beta_bh[t];
      float y_partial = 0.0f;
#pragma unroll
      for (int row = 0; row < ROWS_PER_LANE; row++) {
        s[value][row] =
            __fmaf_rn(k_reg[row], delta, decay * s[value][row]);
        y_partial = __fmaf_rn(s[value][row], q_reg[row], y_partial);
      }
      const float y_col = gdn_warp_sum<WARP_SIZE>(y_partial);
      if (lane == 0) {
        out_bh[t * V_DIM + value_idx] = y_col;
      }
    }
  }

#pragma unroll
  for (int value = 0; value < VALUES_PER_WARP; value++) {
#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      state_bh[(first_value + value) * BK + key] = s[value][row];
    }
  }
}

template <typename StateT, int VALUES_PER_WARP>
cudaError_t launch_vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) {
    return cudaErrorInvalidValue;
  }

  constexpr int NUM_WARPS = 4;
  constexpr int VALUES_PER_BLOCK = NUM_WARPS * VALUES_PER_WARP;
  const dim3 grid((v_dim + VALUES_PER_BLOCK - 1) / VALUES_PER_BLOCK, bh);
  const dim3 block(32, NUM_WARPS);
  gated_delta_rule_recurrence_kernel_vmajor_grouped<StateT, VALUES_PER_WARP,
                                                     NUM_WARPS>
      <<<grid, block, 0, stream>>>(q, k, v, g, beta, state, output, seq_len,
                                  slot_indices, num_heads);
  return cudaGetLastError();
}

template <typename StateT>
cudaError_t dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int values_per_warp, cudaStream_t stream) {
  switch (values_per_warp) {
  case 2:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 2>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  case 4:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 4>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  case 8:
    return launch_vmajor_grouped_warp_gated_delta_rule_recurrence<StateT, 8>(
        q, k, v, g, beta, state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, stream);
  default:
    return cudaErrorInvalidValue;
  }
}

extern "C" int vmajor_grouped_warp_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int values_per_warp, int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim,
            v_dim, slot_indices, num_heads, values_per_warp, custream));
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len,
            k_dim, v_dim, slot_indices, num_heads, values_per_warp, custream));
  } else {
    return static_cast<int>(
        dispatch_vmajor_grouped_warp_gated_delta_rule_recurrence(
            q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim,
            v_dim, slot_indices, num_heads, values_per_warp, custream));
  }
}

template <typename StateT, bool VALUE_MAJOR = false>
void launch_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim == 128) {
    constexpr int BT = 64;
    constexpr int BK = 128;
    constexpr int BV = 64;
    // Shared memory: BT*BK + BT*BT + BT + BT + BK floats
    size_t smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);

    // Request extended shared memory
    auto kernel =
        gdn_chunked_kernel<StateT, BT, BK, BV, VALUE_MAJOR>();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    kernel<<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                          seq_len, v_dim, slot_indices,
                                          num_heads);
  } else if (k_dim == 64) {
    constexpr int BT = 64;
    constexpr int BK = 64;
    constexpr int BV = 64;
    size_t smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);

    auto kernel =
        gdn_chunked_kernel<StateT, BT, BK, BV, VALUE_MAJOR>();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);

    dim3 grid((v_dim + BV - 1) / BV, bh);
    dim3 block(BV);
    kernel<<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                          seq_len, v_dim, slot_indices,
                                          num_heads);
  } else if constexpr (!VALUE_MAJOR) {
    launch_gated_delta_rule_recurrence(q, k, v, g, beta, state, output, bh,
                                       seq_len, k_dim, v_dim, slot_indices,
                                       num_heads, stream);
  }
}

extern "C" void chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream);
  } else {
    launch_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream);
  }
}

template <typename StateT>
cudaError_t launch_vmajor_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, StateT *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    cudaStream_t stream) {
  if (k_dim != 128 || v_dim != 128) {
    return cudaErrorInvalidValue;
  }

  constexpr int BT = 64;
  constexpr int BK = 128;
  constexpr int BV = 64;
  const size_t smem = (BT * BK + BT * BT + 2 * BT + BK) * sizeof(float);
  auto kernel = gdn_chunked_kernel<StateT, BT, BK, BV, true>();
  const cudaError_t attribute_status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (attribute_status != cudaSuccess) {
    return attribute_status;
  }

  const dim3 grid((v_dim + BV - 1) / BV, bh);
  const dim3 block(BV);
  kernel<<<grid, block, smem, stream>>>(q, k, v, g, beta, state, output,
                                        seq_len, v_dim, slot_indices,
                                        num_heads);
  return cudaGetLastError();
}

extern "C" int vmajor_chunked_gated_delta_rule_recurrence(
    const float *q, const float *k, const float *v, const float *g,
    const float *beta, void *state, float *output, int bh, int seq_len,
    int k_dim, int v_dim, const int32_t *slot_indices, int num_heads,
    int state_dtype, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__half *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream));
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (__nv_bfloat16 *)state, output, bh, seq_len, k_dim,
        v_dim, slot_indices, num_heads, custream));
  } else {
    return static_cast<int>(launch_vmajor_chunked_gated_delta_rule_recurrence(
        q, k, v, g, beta, (float *)state, output, bh, seq_len, k_dim, v_dim,
        slot_indices, num_heads, custream));
  }
}
