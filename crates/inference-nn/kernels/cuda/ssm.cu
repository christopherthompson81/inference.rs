// Selective SSM (Mamba-2 style) CUDA kernel for inference.rs
// Adapted from llama.cpp's ssm-scan implementation.
//
// This kernel implements the selective scan operation:
//   state[t] = state[t-1] * exp(dt[t] * A) + dt[t] * B[t] * x[t]
//   y[t] = sum_d(state[t] * C[t]) + D * x[t]
//
// Supports Mamba-2 style: multiple heads, head_dim, n_groups, d_state.

#include <cmath>
#include <cstdint>

#define WARP_SIZE 32

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
  }
  return x;
}

// Mamba-2 SSM scan: each warp runs one (head, head_offset) row across the
// timesteps, each lane `c_factor` state items. Shapes: x (batch, seq_len,
// n_heads, head_dim); dt (batch, seq_len, n_heads), before dt_bias and
// softplus; A, D and dt_bias (n_heads,), A already -exp(A_log); B and C (batch,
// seq_len, n_groups, d_state), the heads split evenly over the groups; y
// (batch, seq_len, n_heads, head_dim). state is (rows, n_heads, head_dim,
// d_state): row b without slot_indices, else row slot_indices[b], a negative
// slot a padding row that leaves the state and outputs 0.
template <int c_factor>
__global__ void __launch_bounds__(WARP_SIZE, 1)
    ssm_scan_kernel(const float *__restrict__ x, const float *__restrict__ dt,
                    const float *__restrict__ A, const float *__restrict__ B,
                    const float *__restrict__ C, const float *__restrict__ D,
                    const float *__restrict__ dt_bias,
                    float *__restrict__ state, float *__restrict__ y,
                    const int32_t *__restrict__ slot_indices, int n_heads,
                    int n_groups, int head_dim, int d_state, int seq_len,
                    float dt_min, float dt_max) {
  // Grid: (n_heads * head_dim, batch_size)
  const int batch_idx = blockIdx.y;
  const int warp_idx = blockIdx.x;
  const int lane = threadIdx.x;

  const int head_idx = warp_idx / head_dim;
  const int head_off = warp_idx % head_dim;

  if (head_idx >= n_heads)
    return;

  const int d_inner = n_heads * head_dim;
  const size_t y_row =
      (size_t)batch_idx * seq_len * d_inner + head_idx * head_dim + head_off;
  const int slot = slot_indices ? slot_indices[batch_idx] : batch_idx;
  if (slot < 0) {
    if (lane == 0) {
      for (int t = 0; t < seq_len; t++) {
        y[y_row + (size_t)t * d_inner] = 0.0f;
      }
    }
    return;
  }

  const float a_val = A[head_idx];
  const float d_val = D[head_idx];
  const float dt_bias_val = dt_bias[head_idx];
  const int group = head_idx / (n_heads / n_groups);

  const size_t state_base =
      (((size_t)slot * n_heads + head_idx) * head_dim + head_off) * d_state;
  // a d_state that is not a whole number of lanes leaves the last lanes' items
  // past the row
  float reg_state[c_factor];
#pragma unroll
  for (int j = 0; j < c_factor; j++) {
    const int state_off = lane * c_factor + j;
    reg_state[j] = state_off < d_state ? state[state_base + state_off] : 0.0f;
  }

  for (int t = 0; t < seq_len; t++) {
    float dt_val = dt[((size_t)batch_idx * seq_len + t) * n_heads + head_idx];
    dt_val += dt_bias_val;
    // softplus: log(1 + exp(x))
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
    dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);

    const float dA = expf(dt_val * a_val);

    const float x_val = x[((size_t)batch_idx * seq_len + t) * d_inner +
                          head_idx * head_dim + head_off];
    const float x_dt = x_val * dt_val;

    float state_sum = 0.0f;
    const size_t bc_base =
        (((size_t)batch_idx * seq_len + t) * n_groups + group) * d_state;
#pragma unroll
    for (int j = 0; j < c_factor; j++) {
      const int state_off = lane * c_factor + j;
      if (state_off >= d_state)
        continue;
      const float b_val = B[bc_base + state_off];
      const float c_val = C[bc_base + state_off];

      // state = state * dA + B * x_dt
      reg_state[j] = reg_state[j] * dA + b_val * x_dt;
      state_sum += reg_state[j] * c_val;
    }

    state_sum = warp_reduce_sum(state_sum);

    if (lane == 0) {
      y[y_row + (size_t)t * d_inner] = state_sum + d_val * x_val;
    }
  }

#pragma unroll
  for (int j = 0; j < c_factor; j++) {
    const int state_off = lane * c_factor + j;
    if (state_off < d_state)
      state[state_base + state_off] = reg_state[j];
  }
}

template <int c_factor>
static void launch_ssm_scan(dim3 grid, cudaStream_t stream, const float *x,
                            const float *dt, const float *A, const float *B,
                            const float *C, const float *D,
                            const float *dt_bias, float *state, float *y,
                            const int32_t *slot_indices, int n_heads,
                            int n_groups, int head_dim, int d_state,
                            int seq_len, float dt_min, float dt_max) {
  ssm_scan_kernel<c_factor><<<grid, WARP_SIZE, 0, stream>>>(
      x, dt, A, B, C, D, dt_bias, state, y, slot_indices, n_heads, n_groups,
      head_dim, d_state, seq_len, dt_min, dt_max);
}

extern "C" void
selective_scan_cuda(const float *x, const float *dt, const float *A,
                    const float *B, const float *C, const float *D,
                    const float *dt_bias, float *state, float *y,
                    const int32_t *slot_indices, int batch_size, int n_heads,
                    int n_groups, int head_dim, int d_state, int seq_len,
                    float dt_min, float dt_max, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 grid(n_heads * head_dim, batch_size);
  // items per lane: d_state up to 256
  const int c_factor = (d_state + WARP_SIZE - 1) / WARP_SIZE;
  if (c_factor == 1) {
    launch_ssm_scan<1>(grid, custream, x, dt, A, B, C, D, dt_bias, state, y,
                       slot_indices, n_heads, n_groups, head_dim, d_state,
                       seq_len, dt_min, dt_max);
  } else if (c_factor == 2) {
    launch_ssm_scan<2>(grid, custream, x, dt, A, B, C, D, dt_bias, state, y,
                       slot_indices, n_heads, n_groups, head_dim, d_state,
                       seq_len, dt_min, dt_max);
  } else if (c_factor <= 4) {
    launch_ssm_scan<4>(grid, custream, x, dt, A, B, C, D, dt_bias, state, y,
                       slot_indices, n_heads, n_groups, head_dim, d_state,
                       seq_len, dt_min, dt_max);
  } else {
    launch_ssm_scan<8>(grid, custream, x, dt, A, B, C, D, dt_bias, state, y,
                       slot_indices, n_heads, n_groups, head_dim, d_state,
                       seq_len, dt_min, dt_max);
  }
}
