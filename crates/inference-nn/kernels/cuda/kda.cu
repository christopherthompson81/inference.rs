// Kimi Delta Attention's recurrence with its output norm, for one (sequence,
// head) per block and one value column j per thread, which holds S[:, j]
// (state (rows, heads, key i, value j)). Per token:
//   q, k L2-normed over the head (x / sqrt(sum x^2 + eps)), q times K^-0.5
//   w_i = exp(-exp(a_log[h]) * softplus(f_i + dt_bias_i)), beta = sigmoid(b)
//   S_ij = S_ij w_i, S_ij += k_i beta (v_j - sum_i k_i S_ij)
//   y_j = sum_i q_i S_ij, out_j = rmsnorm(y)_j norm_w_j sigmoid(gate_j)
// Keys and values share the head width N.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// fla's L2 norm adds this under the square root
#define KDA_L2_EPS 1e-6f
// softplus is the identity past this, as torch and fla take it
#define KDA_SOFTPLUS_THRESHOLD 20.0f

template <typename T> __device__ __forceinline__ float kda_to_float(T value) {
  return static_cast<float>(value);
}
template <> __device__ __forceinline__ float kda_to_float(__half value) {
  return __half2float(value);
}
template <> __device__ __forceinline__ float kda_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <typename T> __device__ __forceinline__ T kda_from_float(float value) {
  return static_cast<T>(value);
}
template <> __device__ __forceinline__ __half kda_from_float(float value) {
  return __float2half(value);
}
template <>
__device__ __forceinline__ __nv_bfloat16 kda_from_float(float value) {
  return __float2bfloat16(value);
}

// the sum over the block's N threads, on every thread
template <int N>
__device__ __forceinline__ float kda_head_sum(float value, float *scratch) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  __syncthreads();
  if ((threadIdx.x & 31) == 0)
    scratch[threadIdx.x >> 5] = value;
  __syncthreads();
  float sum = 0.0f;
#pragma unroll
  for (int w = 0; w < N / 32; w++)
    sum += scratch[w];
  return sum;
}

struct KdaParams {
  // (batch, seq, heads * N) in the activations' dtype: q, k, v after their
  // convs, the decay LoRA's output before its bias, the output gate's
  // pre-sigmoid; b (batch, seq, heads) before its sigmoid
  const void *q, *k, *v, *f, *gate, *b;
  // f32: dt_bias (heads * N), a_log (heads), the output norm's weight (N)
  const float *dt_bias, *a_log, *norm_w;
  float *state;
  // the state row of each sequence, or null for row = sequence
  const uint32_t *slots;
  void *out;
  int seq_len, heads;
  float norm_eps;
};

template <typename T, int N>
__global__ void __launch_bounds__(N) kda_kernel(const KdaParams p) {
  const int j = threadIdx.x;
  const int batch = blockIdx.x / p.heads, head = blockIdx.x % p.heads;
  const int hidden = p.heads * N, c = head * N + j;
  const int row = p.slots ? (int)p.slots[batch] : batch;
  __shared__ float s_q[N], s_k[N], s_w[N], scratch[N / 32];
  const T *q = (const T *)p.q, *k = (const T *)p.k, *v = (const T *)p.v;
  const T *f = (const T *)p.f, *gate = (const T *)p.gate, *b = (const T *)p.b;
  T *out = (T *)p.out;
  const float dt_bias = p.dt_bias[c], rate = expf(p.a_log[head]);
  const float norm_w = p.norm_w[j], scale = rsqrtf((float)N);

  float *column = p.state + ((size_t)row * p.heads + head) * N * N + j;
  float s[N];
#pragma unroll
  for (int i = 0; i < N; i++)
    s[i] = column[(size_t)i * N];

  for (int t = 0; t < p.seq_len; t++) {
    const size_t token = (size_t)batch * p.seq_len + t;
    const size_t at = token * hidden + c;
    const float qj = kda_to_float(q[at]), kj = kda_to_float(k[at]);
    const float q_norm = sqrtf(kda_head_sum<N>(qj * qj, scratch) + KDA_L2_EPS);
    const float k_norm = sqrtf(kda_head_sum<N>(kj * kj, scratch) + KDA_L2_EPS);
    const float fj = kda_to_float(f[at]) + dt_bias;
    const float softplus = fj > KDA_SOFTPLUS_THRESHOLD ? fj : log1pf(expf(fj));
    s_q[j] = qj / q_norm * scale;
    s_k[j] = kj / k_norm;
    s_w[j] = expf(-rate * softplus);
    __syncthreads();
    const float beta =
        1.0f / (1.0f + expf(-kda_to_float(b[token * p.heads + head])));
    float kv = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++) {
      s[i] *= s_w[i];
      kv += s_k[i] * s[i];
    }
    const float delta = beta * (kda_to_float(v[at]) - kv);
    float y = 0.0f;
#pragma unroll
    for (int i = 0; i < N; i++) {
      s[i] += s_k[i] * delta;
      y += s_q[i] * s[i];
    }
    const float mean_sq = kda_head_sum<N>(y * y, scratch) / N;
    const float g = kda_to_float(gate[at]);
    out[at] = kda_from_float<T>(y * rsqrtf(mean_sq + p.norm_eps) * norm_w /
                                (1.0f + expf(-g)));
  }
#pragma unroll
  for (int i = 0; i < N; i++)
    column[(size_t)i * N] = s[i];
}

template <int N>
static void kda_launch(const KdaParams *p, int batch, int dtype,
                       cudaStream_t s) {
  const int blocks = batch * p->heads;
  if (dtype == 0)
    kda_kernel<__half, N><<<blocks, N, 0, s>>>(*p);
  else if (dtype == 1)
    kda_kernel<__nv_bfloat16, N><<<blocks, N, 0, s>>>(*p);
  else
    kda_kernel<float, N><<<blocks, N, 0, s>>>(*p);
}

// dtype: 0 f16, 1 bf16, 2 f32; head_size 64 or 128, else returns 0
extern "C" int kda_cuda(const KdaParams *p, int batch, int head_size,
                        int dtype, int64_t stream) {
  const cudaStream_t s = (cudaStream_t)stream;
  if (head_size == 64)
    kda_launch<64>(p, batch, dtype, s);
  else if (head_size == 128)
    kda_launch<128>(p, batch, dtype, s);
  else
    return 0;
  return 1;
}
