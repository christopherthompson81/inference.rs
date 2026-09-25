#pragma once

#include "gdn_common.cuh"

// ============================================================================
// Kernel 1b: chunked_gated_delta_rule_recurrence (prefill optimization)
//
// Processes prefill tokens in BT-token chunks instead of one at a time.
// Within each chunk: parallel prefix sum of g, cooperative kk_dot computation,
// forward substitution (triangular solve), output computation, and state
// update.
//
// Same thread model as Kernel 1: one block per (v_tile, batch*head),
// one thread per V-column. Each thread owns BK registers of state.
//
// Shared memory holds:
//   k_chunk[BT * BK]  -- key vectors for current chunk
//   kk_dot[BT * BT]   -- dot(k[i], k[j]) lower-triangular matrix
//   gcum[BT]           -- cumulative sum of g within chunk
//   beta_s[BT]         -- beta values for chunk
//   q_buf[BK]          -- q vector (loaded one row at a time)
//
// q,k: [BH, S, K]  v: [BH, S, V]  g,beta: [BH, S]
// state: [BH, K, V] or [BH, V, K] (in/out)  output: [BH, S, V]
// ============================================================================

template <typename StateT, int BT, int BK, int BV, bool VALUE_MAJOR = false>
__global__ void
chunked_gated_delta_rule_kernel(const float *__restrict__ q,    // [BH, S, K]
                                const float *__restrict__ k,    // [BH, S, K]
                                const float *__restrict__ v,    // [BH, S, V]
                                const float *__restrict__ g,    // [BH, S]
                                const float *__restrict__ beta, // [BH, S]
                                StateT *__restrict__ state,     // key-major/value-major or pool
                                float *__restrict__ output,     // [BH, S, V]
                                int seq_len, int v_dim,
                                const int32_t *__restrict__ slot_indices,
                                int num_heads) {

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

  const int num_chunks = (seq_len + BT - 1) / BT;

  // Pointers for this (batch, head)
  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  // Dynamic shared memory layout
  extern __shared__ float smem[];
  float *k_chunk = smem;                  // [BT * BK]
  float *kk_dot = smem + BT * BK;         // [BT * BT]
  float *gcum = smem + BT * BK + BT * BT; // [BT]
  float *beta_s = gcum + BT;              // [BT]
  float *q_buf = beta_s + BT;             // [BK]

  // Load state column into registers
  float s[BK];
#pragma unroll
  for (int j = 0; j < BK; j++) {
    if constexpr (VALUE_MAJOR) {
      s[j] = state_bh[v_idx * BK + j];
    } else {
      s[j] = state_bh[j * v_dim + v_idx];
    }
  }

  // Per-thread register array for corrected deltas
  float delta[BT];

  for (int c = 0; c < num_chunks; c++) {
    const int chunk_start = c * BT;
    const int chunk_len = min(BT, seq_len - chunk_start);

    // === Phase 1: Cooperative load of k, beta, g into shared memory ===
    for (int t = 0; t < chunk_len; t++) {
      for (int j = tid; j < BK; j += BV) {
        k_chunk[t * BK + j] = k_bh[(chunk_start + t) * BK + j];
      }
    }
    if (tid < chunk_len) {
      beta_s[tid] = beta_bh[chunk_start + tid];
      gcum[tid] = g_bh[chunk_start + tid];
    }
    __syncthreads();

    // === Phase 1b: Parallel prefix sum of g (Hillis-Steele) ===
    for (int stride = 1; stride < BT; stride <<= 1) {
      float prev = 0.0f;
      if (tid < chunk_len && (int)tid >= stride)
        prev = gcum[tid - stride];
      __syncthreads();
      if (tid < chunk_len && (int)tid >= stride)
        gcum[tid] += prev;
      __syncthreads();
    }

    // === Phase 2: Compute kk_dot[i][j] = dot(k[i], k[j]) for j < i ===
    // Only lower-triangular entries needed (strictly lower)
    for (int idx = tid; idx < chunk_len * chunk_len; idx += BV) {
      int i = idx / chunk_len;
      int j = idx % chunk_len;
      if (j < i) {
        float dot = 0.0f;
        for (int d = 0; d < BK; d++) {
          dot = __fmaf_rn(k_chunk[i * BK + d], k_chunk[j * BK + d], dot);
        }
        kk_dot[i * BT + j] = dot;
      }
    }
    __syncthreads();

    // === Phase 3: Forward substitution (per V-column, in registers) ===
    // Computes corrected delta values via triangular solve
    for (int i = 0; i < chunk_len; i++) {
      float v_i = v_bh[(chunk_start + i) * v_dim + v_idx];
      float decay_i = expf(gcum[i]);
      float beta_i = beta_s[i];

      // Inter-chunk contribution: state @ k[i] with decay
      float kv_mem = 0.0f;
#pragma unroll
      for (int d = 0; d < BK; d++) {
        kv_mem = __fmaf_rn(s[d] * decay_i, k_chunk[i * BK + d], kv_mem);
      }

      float rhs = beta_i * (v_i - kv_mem);

      // Subtract lower-triangular contributions (intra-chunk)
      for (int j = 0; j < i; j++) {
        float a_ij = beta_i * kk_dot[i * BT + j] * expf(gcum[i] - gcum[j]);
        rhs -= a_ij * delta[j];
      }
      delta[i] = rhs;
    }

    // === Phase 4: Output computation (per V-column) ===
    for (int i = 0; i < chunk_len; i++) {
      // Cooperatively load q[i] into shared
      for (int j = tid; j < BK; j += BV) {
        q_buf[j] = q_bh[(chunk_start + i) * BK + j];
      }
      __syncthreads();

      float decay_i = expf(gcum[i]);

      // Inter-chunk contribution: q[i] @ (state * decay)
      float o_val = 0.0f;
#pragma unroll
      for (int d = 0; d < BK; d++) {
        o_val = __fmaf_rn(q_buf[d], s[d] * decay_i, o_val);
      }

      // Intra-chunk contribution: sum_{j<=i} dot(q[i], k[j]) * delta[j] *
      // exp(gcum[i] - gcum[j])
      for (int j = 0; j <= i; j++) {
        float qk_dot = 0.0f;
        for (int d = 0; d < BK; d++) {
          qk_dot = __fmaf_rn(q_buf[d], k_chunk[j * BK + d], qk_dot);
        }
        o_val += qk_dot * delta[j] * expf(gcum[i] - gcum[j]);
      }

      out_bh[(chunk_start + i) * v_dim + v_idx] = o_val;
      __syncthreads();
    }

    // === Phase 5: State update for next chunk ===
    float g_total = gcum[chunk_len - 1];
#pragma unroll
    for (int d = 0; d < BK; d++) {
      float s_new = s[d] * expf(g_total);
      for (int t = 0; t < chunk_len; t++) {
        s_new += k_chunk[t * BK + d] * delta[t] * expf(g_total - gcum[t]);
      }
      s[d] = s_new;
    }

    __syncthreads();
  }

  // Write final state back
#pragma unroll
  for (int j = 0; j < BK; j++) {
    if constexpr (VALUE_MAJOR) {
      state_bh[v_idx * BK + j] = s[j];
    } else {
      state_bh[j * v_dim + v_idx] = s[j];
    }
  }
}

template <typename StateT, int BT, int BK, int BV, bool VALUE_MAJOR>
GdnChunkedKernel<StateT> gdn_chunked_kernel() {
  return chunked_gated_delta_rule_kernel<StateT, BT, BK, BV, VALUE_MAJOR>;
}

#define GDN_CHUNKED_INSTANTIATE(STATE_T, BK, VALUE_MAJOR)                      \
  template GdnChunkedKernel<STATE_T>                                           \
  gdn_chunked_kernel<STATE_T, GDN_CHUNKED_BT, BK, GDN_CHUNKED_BV, VALUE_MAJOR>();
