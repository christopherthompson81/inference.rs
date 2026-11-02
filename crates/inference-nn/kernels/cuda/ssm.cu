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

// Chunked SSD (Mamba-2's state space duality) for one sequence's prompt, a
// window of chunks at a time: per chunk the scan becomes matmuls (cuBLAS, run
// by the caller between these kernels), only the state carried from chunk to
// chunk running in order. Scratch layouts, k a chunk of the window and L the
// chunk length: cs [k][h][L] (dt summed within the chunk), decay [k][h], xdt
// and xw (x * dt, xw also decayed to the chunk's end) [k][L][h][p], bg and cg
// (B and C, zero past the prompt) [k][g][L][n], cb [k][g][t_in][t_out], m
// [k][h][t_in][t_out], ds and sprev [k][h][p][n], yi and ys (the chunk's own
// and the carried state's outputs) [k][L][h][p].
#define SSD_CHUNK 256

struct SsdParams {
  const float *x, *dt, *A, *B, *C, *D, *dt_bias;
  float *state, *y;
  float *cs, *decay, *xdt, *xw, *bg, *cg, *cb, *m, *ds, *sprev, *yi, *ys;
  int seq_len, first_chunk, n_chunks, n_heads, n_groups, head_dim, d_state;
  float dt_min, dt_max;
};

// grid (heads, chunks), one thread per chunk position
__global__ void __launch_bounds__(SSD_CHUNK)
    ssd_prepare_kernel(const SsdParams p) {
  __shared__ float s_cs[SSD_CHUNK];
  __shared__ float s_dt[SSD_CHUNK];
  const int h = blockIdx.x, k = blockIdx.y, t = threadIdx.x;
  const int H = p.n_heads, P = p.head_dim, N = p.d_state, G = p.n_groups;
  const int base = (p.first_chunk + k) * SSD_CHUNK;
  const int tok = base + t;

  float v = 0.0f;
  if (tok < p.seq_len) {
    v = p.dt[(size_t)tok * H + h] + p.dt_bias[h];
    v = (v > 20.0f) ? v : logf(1.0f + expf(v));
    v = fminf(fmaxf(v, p.dt_min), p.dt_max);
  }
  s_dt[t] = v;
  s_cs[t] = v;
  __syncthreads();
  for (int off = 1; off < SSD_CHUNK; off <<= 1) {
    const float add = t >= off ? s_cs[t - off] : 0.0f;
    __syncthreads();
    s_cs[t] += add;
    __syncthreads();
  }
  const float a = p.A[h];
  const float cs_last = s_cs[SSD_CHUNK - 1];
  p.cs[((size_t)k * H + h) * SSD_CHUNK + t] = s_cs[t];
  if (t == 0)
    p.decay[k * H + h] = expf(a * cs_last);
  // s_cs becomes each position's decay to the chunk's end
  const float to_end = expf(a * (cs_last - s_cs[t]));
  __syncthreads();
  s_cs[t] = to_end;
  __syncthreads();

  for (int i = t; i < SSD_CHUNK * P; i += SSD_CHUNK) {
    const int tt = i / P, d = i % P;
    const int tk = base + tt;
    const float xd =
        tk < p.seq_len ? p.x[((size_t)tk * H + h) * P + d] * s_dt[tt] : 0.0f;
    const size_t o = (((size_t)k * SSD_CHUNK + tt) * H + h) * P + d;
    p.xdt[o] = xd;
    p.xw[o] = xd * s_cs[tt];
  }

  // the group's heads copy its B and C, a share of the rows each
  const int heads_per_group = H / G;
  const int g = h / heads_per_group;
  const int rows = (SSD_CHUNK + heads_per_group - 1) / heads_per_group;
  const int first_row = (h % heads_per_group) * rows;
  for (int i = t; i < rows * N; i += SSD_CHUNK) {
    const int tt = first_row + i / N, n = i % N;
    if (tt >= SSD_CHUNK)
      break;
    const int tk = base + tt;
    const size_t src = ((size_t)tk * G + g) * N + n;
    const size_t o = (((size_t)k * G + g) * SSD_CHUNK + tt) * N + n;
    p.bg[o] = tk < p.seq_len ? p.B[src] : 0.0f;
    p.cg[o] = tk < p.seq_len ? p.C[src] : 0.0f;
  }
}

// grid (SSD_CHUNK, heads, chunks): row t_in of each head's decayed causal C.B
__global__ void __launch_bounds__(SSD_CHUNK)
    ssd_mask_kernel(const SsdParams p) {
  const int t_in = blockIdx.x, h = blockIdx.y, k = blockIdx.z;
  const int t_out = threadIdx.x;
  const int H = p.n_heads, G = p.n_groups;
  const int g = h / (H / G);
  const float *cs = p.cs + ((size_t)k * H + h) * SSD_CHUNK;
  float v = 0.0f;
  if (t_in <= t_out) {
    const size_t cb = (((size_t)k * G + g) * SSD_CHUNK + t_in) * SSD_CHUNK;
    v = expf(p.A[h] * (cs[t_out] - cs[t_in])) * p.cb[cb + t_out];
  }
  p.m[(((size_t)k * H + h) * SSD_CHUNK + t_in) * SSD_CHUNK + t_out] = v;
}

// one thread per state item: keeps the state each chunk starts from and
// leaves the window's last in `state`
__global__ void ssd_state_pass_kernel(const SsdParams p) {
  const int per_head = p.head_dim * p.d_state;
  const int items = p.n_heads * per_head;
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= items)
    return;
  const int h = e / per_head;
  float s = p.state[e];
  for (int k = 0; k < p.n_chunks; k++) {
    p.sprev[(size_t)k * items + e] = s;
    s = p.decay[k * p.n_heads + h] * s + p.ds[(size_t)k * items + e];
  }
  p.state[e] = s;
}

// y = carried state's output decayed to the position + the chunk's own + D x
__global__ void ssd_finish_kernel(const SsdParams p) {
  const int H = p.n_heads, P = p.head_dim;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)p.n_chunks * SSD_CHUNK * H * P)
    return;
  const int h = (i / P) % H;
  const int t = (i / ((size_t)P * H)) % SSD_CHUNK;
  const int k = i / ((size_t)P * H * SSD_CHUNK);
  const int tok = (p.first_chunk + k) * SSD_CHUNK + t;
  if (tok >= p.seq_len)
    return;
  const size_t o = ((size_t)tok * H + h) * P + i % P;
  const float decay = expf(p.A[h] * p.cs[((size_t)k * H + h) * SSD_CHUNK + t]);
  p.y[o] = p.ys[i] * decay + p.yi[i] + p.D[h] * p.x[o];
}

extern "C" int ssd_chunk_length() { return SSD_CHUNK; }

extern "C" void ssd_prepare_cuda(const SsdParams *p, int64_t stream) {
  dim3 grid(p->n_heads, p->n_chunks);
  ssd_prepare_kernel<<<grid, SSD_CHUNK, 0, (cudaStream_t)stream>>>(*p);
}

extern "C" void ssd_mask_cuda(const SsdParams *p, int64_t stream) {
  dim3 grid(SSD_CHUNK, p->n_heads, p->n_chunks);
  ssd_mask_kernel<<<grid, SSD_CHUNK, 0, (cudaStream_t)stream>>>(*p);
}

extern "C" void ssd_state_pass_cuda(const SsdParams *p, int64_t stream) {
  const int items = p->n_heads * p->head_dim * p->d_state;
  ssd_state_pass_kernel<<<(items + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
      *p);
}

extern "C" void ssd_finish_cuda(const SsdParams *p, int64_t stream) {
  const size_t items =
      (size_t)p->n_chunks * SSD_CHUNK * p->n_heads * p->head_dim;
  ssd_finish_kernel<<<(items + 255) / 256, 256, 0, (cudaStream_t)stream>>>(*p);
}
