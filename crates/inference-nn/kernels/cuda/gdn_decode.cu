#include "gdn_common.cuh"

template <typename T, typename OutT>
__global__ void gdn_prepare_recurrence_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, OutT *__restrict__ q_out,
    OutT *__restrict__ k_out, OutT *__restrict__ v_out,
    float *__restrict__ g_out, float *__restrict__ beta_out, int batch_size,
    int seq_len, int token_stride, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int tiled_v_heads) {
  const int token_head = blockIdx.x;
  const int hv = token_head % num_v_heads;
  const int token = token_head / num_v_heads;
  const int t = token % seq_len;
  const int bidx = token / seq_len;
  const int tid = threadIdx.x;

  if (bidx >= batch_size)
    return;

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const int bh = bidx * num_v_heads + hv;

  const size_t token_idx = (size_t)bidx * seq_len + t;
  const T *row = mixed_qkv + token_idx * conv_dim;
  const T *b_row = b + token_idx * num_v_heads;
  const T *a_row = a + token_idx * num_v_heads;

  __shared__ float red_q[256];
  __shared__ float red_k[256];
  __shared__ float q_mul;
  __shared__ float k_mul;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int d = tid; d < head_k_dim; d += blockDim.x) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_sum += q_val * q_val;
    k_sum += k_val * k_val;
  }

  red_q[tid] = q_sum;
  red_k[tid] = k_sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_q[tid] += red_q[tid + stride];
      red_k[tid] += red_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    q_mul = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)head_k_dim);
    k_mul = rsqrtf(red_k[0] + 1.0e-6f);

    float b_val = (float)b_row[hv];
    float a_val = (float)a_row[hv] + dt_bias[hv];
    float softplus_val = a_val > 20.0f
                             ? a_val
                             : (a_val > 0.0f ? a_val + log1pf(expf(-a_val))
                                             : log1pf(expf(a_val)));
    beta_out[(size_t)bh * token_stride + t] = 1.0f / (1.0f + expf(-b_val));
    g_out[(size_t)bh * token_stride + t] = -expf(a_log[hv]) * softplus_val;
  }
  __syncthreads();

  OutT *q_dst = q_out + ((size_t)bh * token_stride + t) * head_k_dim;
  OutT *k_dst = k_out + ((size_t)bh * token_stride + t) * head_k_dim;
  OutT *v_dst = v_out + ((size_t)bh * token_stride + t) * head_v_dim;

  for (int d = tid; d < head_k_dim; d += blockDim.x) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_dst[d] = (OutT)(q_val * q_mul);
    k_dst[d] = (OutT)(k_val * k_mul);
  }

  for (int d = tid; d < head_v_dim; d += blockDim.x) {
    v_dst[d] = (OutT)(float)row[2 * key_dim + hv * head_v_dim + d];
  }
}

template <typename T, typename OutT>
static void launch_gdn_prepare_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *q_out, void *k_out, void *v_out, float *g_out,
    float *beta_out, int batch_size, int seq_len, int token_stride,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid(batch_size * seq_len * num_v_heads);
  gdn_prepare_recurrence_kernel<T, OutT><<<grid, block, 0, stream>>>(
      (const T *)mixed_qkv, (const T *)b, (const T *)a, a_log, dt_bias,
      (OutT *)q_out, (OutT *)k_out, (OutT *)v_out, g_out, beta_out, batch_size,
      seq_len, token_stride, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
      tiled_v_heads);
}

extern "C" void gdn_prepare_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *q_out, void *k_out, void *v_out, float *g_out,
    float *beta_out, int batch_size, int seq_len, int token_stride,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int dtype, int out_bf16, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0 && out_bf16) {
    launch_gdn_prepare_recurrence<__half, __nv_bfloat16>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else if (dtype == 0) {
    launch_gdn_prepare_recurrence<__half, float>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else if (out_bf16) {
    launch_gdn_prepare_recurrence<__nv_bfloat16, __nv_bfloat16>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  } else {
    launch_gdn_prepare_recurrence<__nv_bfloat16, float>(
        mixed_qkv, b, a, a_log, dt_bias, q_out, k_out, v_out, g_out, beta_out,
        batch_size, seq_len, token_stride, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, custream);
  }
}

template <typename T, typename StateT, int BV>
__global__ __launch_bounds__(32) void gdn_decode_recurrence_kernel_value_major(
    const T *__restrict__ mixed_qkv,
    const T *__restrict__ b,
    const T *__restrict__ a,
    const float *__restrict__ a_log,
    const float *__restrict__ dt_bias,
    StateT *__restrict__ state,
    T *__restrict__ output,
    int batch_size,
    int num_k_heads,
    int num_v_heads,
    int tiled_v_heads,
    int64_t b_batch_stride,
    int64_t b_head_stride,
    int64_t a_batch_stride,
    int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  static_assert(GDN_DECODE_VALUE_MAJOR_V % BV == 0,
                "V must divide into value tiles");
  constexpr int NUM_V_TILES = GDN_DECODE_VALUE_MAJOR_V / BV;

  const int lane = threadIdx.x;
  const int linear_tile = blockIdx.x;
  const int total_tiles = batch_size * num_v_heads * NUM_V_TILES;
  if (linear_tile >= total_tiles) {
    return;
  }
  const int v_tile = linear_tile % NUM_V_TILES;
  const int bh = linear_tile / NUM_V_TILES;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;
  T *out_bh = output + (size_t)bh * GDN_DECODE_VALUE_MAJOR_V;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    if (lane < BV) {
      out_bh[v_tile * BV + lane] = (T)0.0f;
    }
    return;
  }
  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * GDN_DECODE_VALUE_MAJOR_K;
  const int value_dim = num_v_heads * GDN_DECODE_VALUE_MAJOR_V;
  const int conv_dim = 2 * key_dim + value_dim;
  const int v_base = v_tile * BV;
  const T *row = mixed_qkv + (size_t)bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) *
                  GDN_DECODE_VALUE_MAJOR_V * GDN_DECODE_VALUE_MAJOR_K;

  float4 qv = gdn_load_state_x4(
      row + hk * GDN_DECODE_VALUE_MAJOR_K + lane * 4);
  float4 kv = gdn_load_state_x4(
      row + key_dim + hk * GDN_DECODE_VALUE_MAJOR_K + lane * 4);
  float q_norm = qv.x * qv.x + qv.y * qv.y + qv.z * qv.z + qv.w * qv.w;
  float k_norm = kv.x * kv.x + kv.y * kv.y + kv.z * kv.z + kv.w * kv.w;
  q_norm = gdn_warp_sum<32>(q_norm);
  k_norm = gdn_warp_sum<32>(k_norm);
  const float q_mul = rsqrtf(q_norm + 1.0e-6f) *
                      rsqrtf((float)GDN_DECODE_VALUE_MAJOR_K);
  const float k_mul = rsqrtf(k_norm + 1.0e-6f);
  qv = make_float4(qv.x * q_mul, qv.y * q_mul, qv.z * q_mul, qv.w * q_mul);
  kv = make_float4(kv.x * k_mul, kv.y * k_mul, kv.z * k_mul, kv.w * k_mul);

  float beta_t = 0.0f;
  float decay_t = 0.0f;
  if (lane == 0) {
    const float b_value =
        (float)b[(size_t)bidx * b_batch_stride + hv * b_head_stride];
    const float a_value =
        (float)a[(size_t)bidx * a_batch_stride + hv * a_head_stride] +
        dt_bias[hv];
    const float softplus =
        a_value > 20.0f
            ? a_value
            : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                              : log1pf(expf(a_value)));
    beta_t = 1.0f / (1.0f + expf(-b_value));
    decay_t = expf(-expf(a_log[hv]) * softplus);
  }
  beta_t = __shfl_sync(0xffffffff, beta_t, 0);
  decay_t = __shfl_sync(0xffffffff, decay_t, 0);

  float4 h[BV];
#pragma unroll
  for (int vi = 0; vi < BV; vi++) {
    const float4 raw = gdn_load_state_x4(
        state_bh + (v_base + vi) * GDN_DECODE_VALUE_MAJOR_K + lane * 4);
    h[vi] = make_float4(raw.x * decay_t, raw.y * decay_t,
                        raw.z * decay_t, raw.w * decay_t);
  }

  const float v_owned =
      lane < BV
          ? (float)row[2 * key_dim + hv * GDN_DECODE_VALUE_MAJOR_V + v_base + lane]
          : 0.0f;
  float out_owned = 0.0f;
#pragma unroll
  for (int vi = 0; vi < BV; vi++) {
    float state_dot_k = h[vi].x * kv.x;
    state_dot_k = __fmaf_rn(h[vi].y, kv.y, state_dot_k);
    state_dot_k = __fmaf_rn(h[vi].z, kv.z, state_dot_k);
    state_dot_k = __fmaf_rn(h[vi].w, kv.w, state_dot_k);
    state_dot_k = gdn_warp_sum<32>(state_dot_k);
    const float delta =
        (__shfl_sync(0xffffffff, v_owned, vi) - state_dot_k) * beta_t;

    h[vi].x = __fmaf_rn(kv.x, delta, h[vi].x);
    h[vi].y = __fmaf_rn(kv.y, delta, h[vi].y);
    h[vi].z = __fmaf_rn(kv.z, delta, h[vi].z);
    h[vi].w = __fmaf_rn(kv.w, delta, h[vi].w);

    float state_dot_q = h[vi].x * qv.x;
    state_dot_q = __fmaf_rn(h[vi].y, qv.y, state_dot_q);
    state_dot_q = __fmaf_rn(h[vi].z, qv.z, state_dot_q);
    state_dot_q = __fmaf_rn(h[vi].w, qv.w, state_dot_q);
    state_dot_q = gdn_warp_sum<32>(state_dot_q);
    if (lane == vi) {
      out_owned = state_dot_q;
    }
  }

#pragma unroll
  for (int vi = 0; vi < BV; vi++) {
    gdn_store_state_x4(
        state_bh + (v_base + vi) * GDN_DECODE_VALUE_MAJOR_K + lane * 4,
        h[vi]);
  }
  if (lane < BV) {
    out_bh[v_base + lane] = (T)out_owned;
  }
}

// Adapted from FlashInfer's Apache-2.0 nontranspose GDN kernel, Copyright (c) 2025 FlashInfer team.
// Exact source revision and license notices are in third_party/flashinfer_gdn.
template <typename T, typename StateT>
__global__ void gdn_decode_recurrence_kernel_cooperative(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state,
    T *__restrict__ output, int batch_size, int num_k_heads,
    int num_v_heads, int head_v_dim, int tiled_v_heads,
    int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  constexpr int BK = GDN_DECODE_COOPERATIVE_K;
  constexpr int BV = GDN_DECODE_COOPERATIVE_V;
  constexpr int VECTORS_PER_ROW = BV / 4;
  constexpr int STATE_VECTORS = BK * VECTORS_PER_ROW;
  constexpr int K_LANES_PER_VALUE = 32 / GDN_DECODE_COOPERATIVE_VALUES_PER_WARP;
  constexpr int K_ITERATIONS = BK / K_LANES_PER_VALUE;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;
  const int v_local = lane % GDN_DECODE_COOPERATIVE_VALUES_PER_WARP;
  const int k_lane = lane / GDN_DECODE_COOPERATIVE_VALUES_PER_WARP;
  const int v_in_tile = warp * GDN_DECODE_COOPERATIVE_VALUES_PER_WARP + v_local;
  const int v_idx = v_tile * BV + v_in_tile;

  if (bidx >= batch_size) {
    return;
  }

  T *out_bh = output + (size_t)bh * head_v_dim;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    if (tid < BV) {
      out_bh[v_tile * BV + tid] = (T)0.0f;
    }
    return;
  }

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * BK;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const T *row = mixed_qkv + bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * BK * head_v_dim;

  __shared__ __align__(16) float
      state_buf[BK * GDN_DECODE_COOPERATIVE_V_PADDED];
  __shared__ float q_buf[BK];
  __shared__ float k_buf[BK];
  __shared__ float q_warp_sums[4];
  __shared__ float k_warp_sums[4];
  __shared__ float beta_t;
  __shared__ float decay_t;
  __shared__ float q_mul;
  __shared__ float k_mul;

#pragma unroll
  for (int vector_idx = tid; vector_idx < STATE_VECTORS;
       vector_idx += GDN_DECODE_COOPERATIVE_THREADS) {
    const int k_idx = vector_idx / VECTORS_PER_ROW;
    const int v_vector = vector_idx % VECTORS_PER_ROW;
    const StateT *src =
        state_bh + k_idx * head_v_dim + v_tile * BV + v_vector * 4;
    float *dst = state_buf + k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_vector * 4;
    if constexpr (sizeof(StateT) == sizeof(float)) {
      __pipeline_memcpy_async(dst, src, sizeof(float4));
    } else {
      *reinterpret_cast<float4 *>(dst) = gdn_load_state_x4(src);
    }
  }
  __pipeline_commit();

  const float q_value = (float)row[hk * BK + tid];
  const float k_value = (float)row[key_dim + hk * BK + tid];
  q_buf[tid] = q_value;
  k_buf[tid] = k_value;
  const float q_sum = gdn_warp_sum(q_value * q_value);
  const float k_sum = gdn_warp_sum(k_value * k_value);
  if (lane == 0) {
    q_warp_sums[warp] = q_sum;
    k_warp_sums[warp] = k_sum;
  }

  if (tid == 0) {
    const float b_value = (float)b[bidx * b_batch_stride + hv * b_head_stride];
    const float a_value =
        (float)a[bidx * a_batch_stride + hv * a_head_stride] + dt_bias[hv];
    const float softplus =
        a_value > 20.0f
            ? a_value
            : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                              : log1pf(expf(a_value)));
    beta_t = 1.0f / (1.0f + expf(-b_value));
    decay_t = expf(-expf(a_log[hv]) * softplus);
  }
  __syncthreads();

  if (tid == 0) {
    const float q_norm = q_warp_sums[0] + q_warp_sums[1] + q_warp_sums[2] +
                         q_warp_sums[3];
    const float k_norm = k_warp_sums[0] + k_warp_sums[1] + k_warp_sums[2] +
                         k_warp_sums[3];
    q_mul = rsqrtf(q_norm + 1.0e-6f) * rsqrtf((float)BK);
    k_mul = rsqrtf(k_norm + 1.0e-6f);
  }
  __syncthreads();

  q_buf[tid] *= q_mul;
  k_buf[tid] *= k_mul;
  __pipeline_wait_prior(0);
  __syncthreads();

  float state_dot_k = 0.0f;
#pragma unroll
  for (int iteration = 0; iteration < K_ITERATIONS; iteration++) {
    const int k_idx = iteration * K_LANES_PER_VALUE + k_lane;
    const float old_state =
        state_buf[k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_in_tile] * decay_t;
    state_dot_k = __fmaf_rn(old_state, k_buf[k_idx], state_dot_k);
  }
  state_dot_k =
      gdn_grouped_k_sum<GDN_DECODE_COOPERATIVE_VALUES_PER_WARP>(state_dot_k);

  float delta = 0.0f;
  if (k_lane == 0) {
    const float v_value = (float)row[2 * key_dim + hv * head_v_dim + v_idx];
    delta = (v_value - state_dot_k) * beta_t;
  }
  delta = __shfl_sync(0xffffffff, delta, v_local);

  float state_dot_q = 0.0f;
#pragma unroll
  for (int iteration = 0; iteration < K_ITERATIONS; iteration++) {
    const int k_idx = iteration * K_LANES_PER_VALUE + k_lane;
    const int state_idx = k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_in_tile;
    const float old_state = state_buf[state_idx] * decay_t;
    const float new_state = __fmaf_rn(k_buf[k_idx], delta, old_state);
    state_buf[state_idx] = new_state;
    state_dot_q = __fmaf_rn(new_state, q_buf[k_idx], state_dot_q);
  }
  state_dot_q =
      gdn_grouped_k_sum<GDN_DECODE_COOPERATIVE_VALUES_PER_WARP>(state_dot_q);
  if (k_lane == 0) {
    out_bh[v_idx] = (T)state_dot_q;
  }
  __syncthreads();

#pragma unroll
  for (int vector_idx = tid; vector_idx < STATE_VECTORS;
       vector_idx += GDN_DECODE_COOPERATIVE_THREADS) {
    const int k_idx = vector_idx / VECTORS_PER_ROW;
    const int v_vector = vector_idx % VECTORS_PER_ROW;
    const float *src =
        state_buf + k_idx * GDN_DECODE_COOPERATIVE_V_PADDED + v_vector * 4;
    StateT *dst = state_bh + k_idx * head_v_dim + v_tile * BV + v_vector * 4;
    gdn_store_state_x4(dst, *reinterpret_cast<const float4 *>(src));
  }
}

// Adapted from FlashInfer's Apache-2.0 large-batch nontranspose GDN kernel.
// Exact source revision and license notices are in third_party/flashinfer_gdn.
template <typename T, typename StateT>
__global__ void gdn_decode_recurrence_kernel_pipelined(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state,
    T *__restrict__ output, int batch_size, int num_k_heads,
    int num_v_heads, int head_v_dim, int tiled_v_heads,
    int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  constexpr int BK = GDN_DECODE_PIPELINED_K;
  constexpr int BV = GDN_DECODE_PIPELINED_V;
  constexpr int VECTORS_PER_ROW = BV / 4;
  constexpr int STATE_VECTORS = BK * VECTORS_PER_ROW;
  constexpr int K_LANES_PER_VALUE =
      32 / GDN_DECODE_PIPELINED_VALUES_PER_WARP;
  constexpr int K_ITERATIONS = BK / K_LANES_PER_VALUE;
  static_assert(BV == GDN_DECODE_PIPELINED_WARPS *
                          GDN_DECODE_PIPELINED_VALUES_PER_WARP,
                "each warp must own one group of values");
  static_assert(BK % K_LANES_PER_VALUE == 0,
                "K must divide across the grouped lanes");
  static_assert(GDN_DECODE_PIPELINED_V_PADDED % 4 == 0,
                "padded rows must preserve vector alignment");
  static_assert(GDN_DECODE_PIPELINED_STAGES == 2,
                "the decode pipeline expects two stages");

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bh = blockIdx.x;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;
  const int v_local = lane % GDN_DECODE_PIPELINED_VALUES_PER_WARP;
  const int k_lane = lane / GDN_DECODE_PIPELINED_VALUES_PER_WARP;
  const int v_in_tile = warp * GDN_DECODE_PIPELINED_VALUES_PER_WARP + v_local;

  if (bidx >= batch_size) {
    return;
  }

  T *out_bh = output + (size_t)bh * head_v_dim;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    for (int v_idx = tid; v_idx < head_v_dim; v_idx += blockDim.x) {
      out_bh[v_idx] = (T)0.0f;
    }
    return;
  }

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * BK;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const int num_v_tiles = head_v_dim / BV;
  const T *row = mixed_qkv + bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * BK * head_v_dim;

  __shared__ __align__(16)
      float state_buf[GDN_DECODE_PIPELINED_STAGES]
                     [BK * GDN_DECODE_PIPELINED_V_PADDED];
  __shared__ float q_buf[BK];
  __shared__ float k_buf[BK];
  __shared__ float q_warp_sums[GDN_DECODE_PIPELINED_WARPS];
  __shared__ float k_warp_sums[GDN_DECODE_PIPELINED_WARPS];
  __shared__ float beta_t;
  __shared__ float decay_t;
  __shared__ float q_mul;
  __shared__ float k_mul;

  if (tid < BK) {
    const float q_value = (float)row[hk * BK + tid];
    const float k_value = (float)row[key_dim + hk * BK + tid];
    q_buf[tid] = q_value;
    k_buf[tid] = k_value;
  }

  const float q_value = tid < BK ? q_buf[tid] : 0.0f;
  const float k_value = tid < BK ? k_buf[tid] : 0.0f;
  const float q_sum = gdn_warp_sum(q_value * q_value);
  const float k_sum = gdn_warp_sum(k_value * k_value);
  if (lane == 0) {
    q_warp_sums[warp] = q_sum;
    k_warp_sums[warp] = k_sum;
  }

  if (tid == 0) {
    const float b_value = (float)b[bidx * b_batch_stride + hv * b_head_stride];
    const float a_value =
        (float)a[bidx * a_batch_stride + hv * a_head_stride] + dt_bias[hv];
    const float softplus =
        a_value > 20.0f
            ? a_value
            : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                              : log1pf(expf(a_value)));
    beta_t = 1.0f / (1.0f + expf(-b_value));
    decay_t = expf(-expf(a_log[hv]) * softplus);
  }
  __syncthreads();

  if (tid == 0) {
    float q_norm = 0.0f;
    float k_norm = 0.0f;
#pragma unroll
    for (int warp_idx = 0; warp_idx < GDN_DECODE_PIPELINED_WARPS;
         warp_idx++) {
      q_norm += q_warp_sums[warp_idx];
      k_norm += k_warp_sums[warp_idx];
    }
    q_mul = rsqrtf(q_norm + 1.0e-6f) * rsqrtf((float)BK);
    k_mul = rsqrtf(k_norm + 1.0e-6f);
  }
  __syncthreads();

  if (tid < BK) {
    q_buf[tid] *= q_mul;
    k_buf[tid] *= k_mul;
  }

  if (num_v_tiles > 0) {
#pragma unroll
    for (int vector_idx = tid; vector_idx < STATE_VECTORS;
         vector_idx += GDN_DECODE_PIPELINED_THREADS) {
      const int k_idx = vector_idx / VECTORS_PER_ROW;
      const int v_vector = vector_idx % VECTORS_PER_ROW;
      const StateT *src = state_bh + k_idx * head_v_dim + v_vector * 4;
      float *dst = state_buf[0] +
                   k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_vector * 4;
      if constexpr (sizeof(StateT) == sizeof(float)) {
        gdn_cp_async_cg_16(dst, src);
      } else {
        *reinterpret_cast<float4 *>(dst) = gdn_load_state_x4(src);
      }
    }
    gdn_cp_async_commit();
  }
  __syncthreads();

  for (int v_tile = 0; v_tile < num_v_tiles; v_tile++) {
    const int stage = v_tile % GDN_DECODE_PIPELINED_STAGES;
    const int next_v_tile = v_tile + 1;
    gdn_cp_async_wait();
    __syncthreads();

    if (next_v_tile < num_v_tiles) {
      const int next_stage = next_v_tile % GDN_DECODE_PIPELINED_STAGES;
#pragma unroll
      for (int vector_idx = tid; vector_idx < STATE_VECTORS;
           vector_idx += GDN_DECODE_PIPELINED_THREADS) {
        const int k_idx = vector_idx / VECTORS_PER_ROW;
        const int v_vector = vector_idx % VECTORS_PER_ROW;
        const StateT *src = state_bh + k_idx * head_v_dim +
                            next_v_tile * BV + v_vector * 4;
        float *dst = state_buf[next_stage] +
                     k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_vector * 4;
        if constexpr (sizeof(StateT) == sizeof(float)) {
          gdn_cp_async_cg_16(dst, src);
        } else {
          *reinterpret_cast<float4 *>(dst) = gdn_load_state_x4(src);
        }
      }
      gdn_cp_async_commit();
    }

    float state_dot_k = 0.0f;
#pragma unroll
    for (int iteration = 0; iteration < K_ITERATIONS; iteration++) {
      const int k_idx = iteration * K_LANES_PER_VALUE + k_lane;
      const float old_state =
          state_buf[stage][k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_in_tile] *
          decay_t;
      state_dot_k = __fmaf_rn(old_state, k_buf[k_idx], state_dot_k);
    }
    state_dot_k =
        gdn_grouped_k_sum<GDN_DECODE_PIPELINED_VALUES_PER_WARP>(state_dot_k);

    const int v_idx = v_tile * BV + v_in_tile;
    float delta = 0.0f;
    if (k_lane == 0) {
      const float v_value = (float)row[2 * key_dim + hv * head_v_dim + v_idx];
      delta = (v_value - state_dot_k) * beta_t;
    }
    delta = __shfl_sync(0xffffffff, delta, v_local);

    float state_dot_q = 0.0f;
#pragma unroll
    for (int iteration = 0; iteration < K_ITERATIONS; iteration++) {
      const int k_idx = iteration * K_LANES_PER_VALUE + k_lane;
      const int state_idx =
          k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_in_tile;
      const float old_state = state_buf[stage][state_idx] * decay_t;
      const float new_state = __fmaf_rn(k_buf[k_idx], delta, old_state);
      state_buf[stage][state_idx] = new_state;
      state_dot_q = __fmaf_rn(new_state, q_buf[k_idx], state_dot_q);
    }
    state_dot_q =
        gdn_grouped_k_sum<GDN_DECODE_PIPELINED_VALUES_PER_WARP>(state_dot_q);
    if (k_lane == 0) {
      out_bh[v_idx] = (T)state_dot_q;
    }
    __syncthreads();

#pragma unroll
    for (int vector_idx = tid; vector_idx < STATE_VECTORS;
         vector_idx += GDN_DECODE_PIPELINED_THREADS) {
      const int k_idx = vector_idx / VECTORS_PER_ROW;
      const int v_vector = vector_idx % VECTORS_PER_ROW;
      const float *src = state_buf[stage] +
                         k_idx * GDN_DECODE_PIPELINED_V_PADDED + v_vector * 4;
      StateT *dst =
          state_bh + k_idx * head_v_dim + v_tile * BV + v_vector * 4;
      gdn_store_state_x4(dst, *reinterpret_cast<const float4 *>(src));
    }
    __syncthreads();
  }
}

template <typename T, typename StateT, int BK, int BV>
__global__ void gdn_decode_recurrence_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state,
    T *__restrict__ output, int batch_size, int num_k_heads,
    int num_v_heads, int head_v_dim, int tiled_v_heads,
    int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;

  if (bidx >= batch_size)
    return;

  T *out_bh = output + (size_t)bh * head_v_dim;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    if (v_idx < head_v_dim) {
      out_bh[v_idx] = (T)0.0f;
    }
    return;
  }

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * BK;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  const T *row = mixed_qkv + bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * BK * head_v_dim;

  __shared__ float red_q[BV];
  __shared__ float red_k[BV];
  __shared__ float q_buf[BK];
  __shared__ float k_buf[BK];
  __shared__ float beta_t;
  __shared__ float decay_t;
  __shared__ float q_mul;
  __shared__ float k_mul;
  __shared__ float state_buf[BK * BV];

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int d = tid; d < BK; d += BV) {
    float q_val = (float)row[hk * BK + d];
    float k_val = (float)row[key_dim + hk * BK + d];
    q_sum += q_val * q_val;
    k_sum += k_val * k_val;
  }

  red_q[tid] = q_sum;
  red_k[tid] = k_sum;
  __syncthreads();

  for (int stride = BV >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_q[tid] += red_q[tid + stride];
      red_k[tid] += red_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    q_mul = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)BK);
    k_mul = rsqrtf(red_k[0] + 1.0e-6f);
    float b_val = (float)b[bidx * b_batch_stride + hv * b_head_stride];
    float a_val =
        (float)a[bidx * a_batch_stride + hv * a_head_stride] + dt_bias[hv];
    float softplus_val = a_val > 20.0f
                             ? a_val
                             : (a_val > 0.0f ? a_val + log1pf(expf(-a_val))
                                             : log1pf(expf(a_val)));
    beta_t = 1.0f / (1.0f + expf(-b_val));
    decay_t = expf(-expf(a_log[hv]) * softplus_val);
  }
  __syncthreads();

  for (int d = tid; d < BK; d += BV) {
    q_buf[d] = (float)row[hk * BK + d] * q_mul;
    k_buf[d] = (float)row[key_dim + hk * BK + d] * k_mul;
  }
  __syncthreads();

  if (v_idx >= head_v_dim)
    return;

  float v_t = (float)row[2 * key_dim + hv * head_v_dim + v_idx];
  float kv_mem = 0.0f;
#pragma unroll GDN_DECODE_STATE_LOAD_UNROLL
  for (int j = 0; j < BK; j++) {
      const float s = (float)state_bh[j * head_v_dim + v_idx] * decay_t;
    state_buf[j * BV + tid] = s;
    kv_mem = __fmaf_rn(s, k_buf[j], kv_mem);
  }

  float delta = (v_t - kv_mem) * beta_t;
  float y_t = 0.0f;
  static_assert(BK % GDN_DECODE_STATE_UPDATE_TILE_ROWS == 0,
                "BK must be divisible by the update tile");
#pragma unroll 1
  for (int base = 0; base < BK; base += GDN_DECODE_STATE_UPDATE_TILE_ROWS) {
#pragma unroll GDN_DECODE_STATE_UPDATE_TILE_ROWS
    for (int offset = 0; offset < GDN_DECODE_STATE_UPDATE_TILE_ROWS; offset++) {
      const int j = base + offset;
      const float s = __fmaf_rn(k_buf[j], delta, state_buf[j * BV + tid]);
      state_bh[j * head_v_dim + v_idx] = s;
      y_t = __fmaf_rn(s, q_buf[j], y_t);
    }
  }
  out_bh[v_idx] = (T)y_t;
}

template <typename T, typename StateT, int BV, int MAX_K>
__global__ void gdn_decode_recurrence_kernel_fallback(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state,
    T *__restrict__ output, int batch_size, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int tiled_v_heads,
    int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *__restrict__ slot_indices) {
  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;
  const int bidx = bh / num_v_heads;
  const int hv = bh - bidx * num_v_heads;

  if (bidx >= batch_size)
    return;

  T *out_bh = output + (size_t)bh * head_v_dim;
  if (gdn_is_padding_row(slot_indices, bidx)) {
    if (v_idx < head_v_dim) {
      out_bh[v_idx] = (T)0.0f;
    }
    return;
  }

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;

  const T *row = mixed_qkv + bidx * conv_dim;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bidx, hv, num_v_heads) * head_k_dim * head_v_dim;

  extern __shared__ float shared[];
  float *red_q = shared;
  float *red_k = red_q + BV;
  float *q_buf = red_k + BV;
  float *k_buf = q_buf + head_k_dim;

  __shared__ float beta_t;
  __shared__ float decay_t;
  __shared__ float q_mul;
  __shared__ float k_mul;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int d = tid; d < head_k_dim; d += BV) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_sum += q_val * q_val;
    k_sum += k_val * k_val;
  }

  red_q[tid] = q_sum;
  red_k[tid] = k_sum;
  __syncthreads();

  for (int stride = BV >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_q[tid] += red_q[tid + stride];
      red_k[tid] += red_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    q_mul = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)head_k_dim);
    k_mul = rsqrtf(red_k[0] + 1.0e-6f);
    float b_val = (float)b[bidx * b_batch_stride + hv * b_head_stride];
    float a_val =
        (float)a[bidx * a_batch_stride + hv * a_head_stride] + dt_bias[hv];
    float softplus_val = a_val > 20.0f
                             ? a_val
                             : (a_val > 0.0f ? a_val + log1pf(expf(-a_val))
                                             : log1pf(expf(a_val)));
    beta_t = 1.0f / (1.0f + expf(-b_val));
    decay_t = expf(-expf(a_log[hv]) * softplus_val);
  }
  __syncthreads();

  for (int d = tid; d < head_k_dim; d += BV) {
    q_buf[d] = (float)row[hk * head_k_dim + d] * q_mul;
    k_buf[d] = (float)row[key_dim + hk * head_k_dim + d] * k_mul;
  }
  __syncthreads();

  if (v_idx >= head_v_dim)
    return;

  float s[MAX_K];
  for (int j = 0; j < head_k_dim; j++) {
    s[j] = (float)state_bh[j * head_v_dim + v_idx] * decay_t;
  }

  float v_t = (float)row[2 * key_dim + hv * head_v_dim + v_idx];
  float kv_mem = 0.0f;
  for (int j = 0; j < head_k_dim; j++) {
    kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
  }

  float delta = (v_t - kv_mem) * beta_t;
  float y_t = 0.0f;
  for (int j = 0; j < head_k_dim; j++) {
    s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
    y_t = __fmaf_rn(s[j], q_buf[j], y_t);
  }

  for (int j = 0; j < head_k_dim; j++) {
    state_bh[j * head_v_dim + v_idx] = s[j];
  }
  out_bh[v_idx] = (T)y_t;
}

template <typename T, typename StateT>
void launch_gdn_decode_recurrence(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, StateT *state, T *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, cudaStream_t stream) {
  if (kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_4 ||
      kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_32) {
    if (head_k_dim == GDN_DECODE_VALUE_MAJOR_K &&
        head_v_dim == GDN_DECODE_VALUE_MAJOR_V) {
      const int bh = batch_size * num_v_heads;
      if (kernel_kind == GDN_DECODE_KERNEL_VALUE_MAJOR_4) {
        gdn_decode_recurrence_kernel_value_major<T, StateT, 4>
            <<<bh * (GDN_DECODE_VALUE_MAJOR_V / 4), 32, 0, stream>>>(
                mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
                num_k_heads, num_v_heads, tiled_v_heads, b_batch_stride,
                b_head_stride, a_batch_stride, a_head_stride, slot_indices);
      } else {
        gdn_decode_recurrence_kernel_value_major<T, StateT, 32>
            <<<bh * (GDN_DECODE_VALUE_MAJOR_V / 32), 32, 0, stream>>>(
                mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
                num_k_heads, num_v_heads, tiled_v_heads, b_batch_stride,
                b_head_stride, a_batch_stride, a_head_stride, slot_indices);
      }
    }
    return;
  }
  constexpr int BV = GDN_DECODE_VALUE_TILE;
  dim3 grid((head_v_dim + BV - 1) / BV, batch_size * num_v_heads);
  dim3 block(BV);

  if (kernel_kind == GDN_DECODE_KERNEL_PIPELINED &&
      head_k_dim == GDN_DECODE_PIPELINED_K &&
      head_v_dim % GDN_DECODE_PIPELINED_V == 0) {
    dim3 pipelined_grid(batch_size * num_v_heads);
    dim3 pipelined_block(GDN_DECODE_PIPELINED_THREADS);
    gdn_decode_recurrence_kernel_pipelined<T, StateT>
        <<<pipelined_grid, pipelined_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (kernel_kind == GDN_DECODE_KERNEL_COOPERATIVE &&
             head_k_dim == GDN_DECODE_COOPERATIVE_K &&
             head_v_dim % GDN_DECODE_COOPERATIVE_V == 0) {
    constexpr int COOPERATIVE_BV = GDN_DECODE_COOPERATIVE_V;
    dim3 cooperative_grid(head_v_dim / COOPERATIVE_BV,
                          batch_size * num_v_heads);
    dim3 cooperative_block(GDN_DECODE_COOPERATIVE_THREADS);
    gdn_decode_recurrence_kernel_cooperative<T, StateT>
        <<<cooperative_grid, cooperative_block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (head_k_dim == 128) {
    gdn_decode_recurrence_kernel<T, StateT, 128, BV>
        <<<grid, block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else if (head_k_dim == 64) {
    gdn_decode_recurrence_kernel<T, StateT, 64, BV>
        <<<grid, block, 0, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  } else {
    constexpr int MAX_K = 256;
    size_t smem = (2 * BV + 2 * head_k_dim) * sizeof(float);
    gdn_decode_recurrence_kernel_fallback<T, StateT, BV, MAX_K>
        <<<grid, block, smem, stream>>>(
            mixed_qkv, b, a, a_log, dt_bias, state, output, batch_size,
            num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
            b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
            slot_indices);
  }
}

template <typename T>
void dispatch_gdn_decode_recurrence(
    const T *mixed_qkv, const T *b, const T *a, const float *a_log,
    const float *dt_bias, void *state, T *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, int state_dtype,
    cudaStream_t stream) {
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (__half *)state, output, batch_size,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
        b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
        slot_indices, kernel_kind, stream);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (__nv_bfloat16 *)state, output,
        batch_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        tiled_v_heads, b_batch_stride, b_head_stride, a_batch_stride,
        a_head_stride, slot_indices, kernel_kind, stream);
  } else {
    launch_gdn_decode_recurrence(
        mixed_qkv, b, a, a_log, dt_bias, (float *)state, output, batch_size,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, tiled_v_heads,
        b_batch_stride, b_head_stride, a_batch_stride, a_head_stride,
        slot_indices, kernel_kind, stream);
  }
}

extern "C" void gdn_decode_recurrence(
    const void *mixed_qkv, const void *b, const void *a, const float *a_log,
    const float *dt_bias, void *state, void *output, int batch_size,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int64_t b_batch_stride, int64_t b_head_stride,
    int64_t a_batch_stride, int64_t a_head_stride,
    const int32_t *slot_indices, int kernel_kind, int dtype, int state_dtype,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == 0) {
    dispatch_gdn_decode_recurrence(
        (const __half *)mixed_qkv, (const __half *)b, (const __half *)a,
        a_log, dt_bias, state, (__half *)output, batch_size, num_k_heads,
        num_v_heads, head_k_dim, head_v_dim, tiled_v_heads, b_batch_stride,
        b_head_stride, a_batch_stride, a_head_stride, slot_indices, kernel_kind,
        state_dtype, custream);
  } else {
    dispatch_gdn_decode_recurrence(
        (const __nv_bfloat16 *)mixed_qkv, (const __nv_bfloat16 *)b,
        (const __nv_bfloat16 *)a, a_log, dt_bias, state,
        (__nv_bfloat16 *)output, batch_size, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, tiled_v_heads, b_batch_stride, b_head_stride,
        a_batch_stride, a_head_stride, slot_indices, kernel_kind, state_dtype,
        custream);
  }
}
