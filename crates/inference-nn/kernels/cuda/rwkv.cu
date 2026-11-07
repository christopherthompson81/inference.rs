// RWKV-7's WKV recurrence, adapted from llama.cpp's rwkv_wkv7_f32 (ggml-cuda
// wkv.cu). Per head, with the state S[i][j] (i a value row, j a key column):
//   sa = sum_j a[j] S[i][j]
//   S[i][j] = S[i][j] w[j] + v[i] k[j] + sa b[j]
//   y[i] = sum_j S[i][j] r[j]
// r, w, k, v, a and b are (batch, seq_len, heads, head_size); state (batch,
// heads, head_size, head_size), advanced in place; y like r.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define RWKV7_HEAD_SIZE 64
#define RWKV7_SHIFT_THREADS 256
// fla's `w = -exp(-0.5) * sigmoid(..)`, the decay exp(w)
#define RWKV7_DECAY_SCALE (-0.6065306597126334f)
// F.normalize's floor on a head's key norm
#define RWKV7_KK_EPS 1e-12f

template <typename T> __device__ __forceinline__ float rwkv_to_float(T value) {
  return static_cast<float>(value);
}
template <> __device__ __forceinline__ float rwkv_to_float(__half value) {
  return __half2float(value);
}
template <>
__device__ __forceinline__ float rwkv_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <typename T>
__device__ __forceinline__ T rwkv_from_float(float value) {
  return static_cast<T>(value);
}
template <> __device__ __forceinline__ __half rwkv_from_float(float value) {
  return __float2half(value);
}
template <>
__device__ __forceinline__ __nv_bfloat16 rwkv_from_float(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float rwkv_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// the sum over a head's RWKV7_HEAD_SIZE threads (two warps), on every thread
__device__ __forceinline__ float rwkv_head_sum(float value, float *scratch) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  __syncthreads();
  if ((threadIdx.x & 31) == 0)
    scratch[threadIdx.x >> 5] = value;
  __syncthreads();
  return scratch[0] + scratch[1];
}

__global__ void __launch_bounds__(RWKV7_HEAD_SIZE)
    rwkv7_wkv_kernel(const float *__restrict__ r, const float *__restrict__ w,
                     const float *__restrict__ k, const float *__restrict__ v,
                     const float *__restrict__ a, const float *__restrict__ b,
                     float *__restrict__ state, float *__restrict__ y,
                     int seq_len, int heads) {
  constexpr int N = RWKV7_HEAD_SIZE;
  const int i = threadIdx.x;
  const int batch = blockIdx.x / heads, head = blockIdx.x % heads;
  const int channels = heads * N;
  __shared__ float s_r[N], s_w[N], s_k[N], s_a[N], s_b[N];

  float *row = state + (((size_t)batch * heads + head) * N + i) * N;
  float s[N];
#pragma unroll
  for (int j = 0; j < N; j++)
    s[j] = row[j];

  for (int t = 0; t < seq_len; t++) {
    const size_t at = ((size_t)batch * seq_len + t) * channels + head * N + i;
    __syncthreads();
    s_r[i] = r[at];
    s_w[i] = w[at];
    s_k[i] = k[at];
    s_a[i] = a[at];
    s_b[i] = b[at];
    __syncthreads();
    float sa = 0.0f;
#pragma unroll
    for (int j = 0; j < N; j++)
      sa += s_a[j] * s[j];
    const float vi = v[at];
    float out = 0.0f;
#pragma unroll
    for (int j = 0; j < N; j++) {
      s[j] = s[j] * s_w[j] + vi * s_k[j] + sa * s_b[j];
      out += s[j] * s_r[j];
    }
    y[at] = out;
  }
#pragma unroll
  for (int j = 0; j < N; j++)
    row[j] = s[j];
}

extern "C" int rwkv7_head_size() { return RWKV7_HEAD_SIZE; }

extern "C" void rwkv7_wkv_cuda(const float *r, const float *w, const float *k,
                               const float *v, const float *a, const float *b,
                               float *state, float *y, int batch, int seq_len,
                               int heads, int64_t stream) {
  rwkv7_wkv_kernel<<<batch * heads, RWKV7_HEAD_SIZE, 0, (cudaStream_t)stream>>>(
      r, w, k, v, a, b, state, y, seq_len, heads);
}

// The token shift's lerps, `out[m] = x + (prev - x) * mix[m]` for each of
// `mixes` mixes, `prev` the previous token's `x` and `last` before the first;
// `new_last` is the sequence's last token. x (batch, seq, hidden); last and
// new_last (batch, hidden); mix (mixes, hidden); out (mixes, batch, seq,
// hidden).
template <typename T>
__global__ void
rwkv7_token_shift_kernel(const T *__restrict__ x, const T *__restrict__ last,
                         const T *__restrict__ mix, T *__restrict__ out,
                         T *__restrict__ new_last, int batch, int seq_len,
                         int hidden, int mixes) {
  const size_t items = (size_t)batch * seq_len * hidden;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= items)
    return;
  const int c = i % hidden;
  const int t = (i / hidden) % seq_len;
  const int b = i / ((size_t)hidden * seq_len);
  const float xi = rwkv_to_float(x[i]);
  const float prev = t == 0 ? rwkv_to_float(last[(size_t)b * hidden + c])
                            : rwkv_to_float(x[i - hidden]);
  for (int m = 0; m < mixes; m++)
    out[m * items + i] = rwkv_from_float<T>(
        xi + (prev - xi) * rwkv_to_float(mix[(size_t)m * hidden + c]));
  if (t == seq_len - 1)
    new_last[(size_t)b * hidden + c] = x[i];
}

struct Rwkv7TimeMixParams {
  // (batch, seq, hidden): the projections and LoRA outputs, the value mix's
  // first-layer value and its LoRA output (null on the first layer)
  const void *r, *w, *k, *v, *a, *g, *v_first, *v_mix;
  // (hidden,) f32: k_k, k_a, r_k (per head), the group norm's weight and bias
  const float *k_k, *k_a, *r_k, *norm_w, *norm_b;
  float *state;
  void *out;
  int seq_len, heads;
  float norm_eps;
};

// The time-mix between its projections and o_proj for one (sequence, head), a
// thread per channel: the decay, rate and value mix, kk and k, the WKV state
// update, the group norm, the bonus and the gate.
template <typename T>
__global__ void __launch_bounds__(RWKV7_HEAD_SIZE)
    rwkv7_time_mix_kernel(const Rwkv7TimeMixParams p) {
  constexpr int N = RWKV7_HEAD_SIZE;
  const int i = threadIdx.x;
  const int batch = blockIdx.x / p.heads, head = blockIdx.x % p.heads;
  const int hidden = p.heads * N, c = head * N + i;
  __shared__ float s_r[N], s_w[N], s_k[N], s_a[N], s_b[N], scratch[2];
  const T *r = (const T *)p.r, *w = (const T *)p.w, *k = (const T *)p.k;
  const T *v = (const T *)p.v, *a = (const T *)p.a, *g = (const T *)p.g;
  const T *v_first = (const T *)p.v_first, *v_mix = (const T *)p.v_mix;
  T *out = (T *)p.out;
  const float k_k = p.k_k[c], k_a = p.k_a[c], r_k = p.r_k[c];
  const float norm_w = p.norm_w[c], norm_b = p.norm_b[c];

  float *row = p.state + (((size_t)batch * p.heads + head) * N + i) * N;
  float s[N];
#pragma unroll
  for (int j = 0; j < N; j++)
    s[j] = row[j];

  for (int t = 0; t < p.seq_len; t++) {
    const size_t at = ((size_t)batch * p.seq_len + t) * hidden + c;
    const float ri = rwkv_to_float(r[at]);
    const float wi =
        expf(RWKV7_DECAY_SCALE * rwkv_sigmoid(rwkv_to_float(w[at])));
    const float ki = rwkv_to_float(k[at]);
    const float ai = rwkv_sigmoid(rwkv_to_float(a[at]));
    float vi = rwkv_to_float(v[at]);
    if (v_first) {
      const float mix = rwkv_sigmoid(rwkv_to_float(v_mix[at]));
      vi += (rwkv_to_float(v_first[at]) - vi) * mix;
    }
    const float kk = ki * k_k;
    const float kk_norm =
        fmaxf(sqrtf(rwkv_head_sum(kk * kk, scratch)), RWKV7_KK_EPS);
    const float kkn = kk / kk_norm;
    const float kn = ki * (1.0f + (ai - 1.0f) * k_a);
    __syncthreads();
    s_r[i] = ri;
    s_w[i] = wi;
    s_k[i] = kn;
    s_a[i] = -kkn;
    s_b[i] = kkn * ai;
    __syncthreads();
    float sa = 0.0f;
#pragma unroll
    for (int j = 0; j < N; j++)
      sa += s_a[j] * s[j];
    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < N; j++) {
      s[j] = s[j] * s_w[j] + vi * s_k[j] + sa * s_b[j];
      y += s[j] * s_r[j];
    }
    const float mean = rwkv_head_sum(y, scratch) / N;
    const float centered = y - mean;
    const float var = rwkv_head_sum(centered * centered, scratch) / N;
    const float normed = centered * rsqrtf(var + p.norm_eps) * norm_w + norm_b;
    const float bonus = rwkv_head_sum(ri * kn * r_k, scratch);
    out[at] = rwkv_from_float<T>((normed + bonus * vi) * rwkv_to_float(g[at]));
  }
#pragma unroll
  for (int j = 0; j < N; j++)
    row[j] = s[j];
}

// dtype: 0 f16, 1 bf16, 2 f32
extern "C" void rwkv7_token_shift_cuda(const void *x, const void *last,
                                       const void *mix, void *out,
                                       void *new_last, int batch, int seq_len,
                                       int hidden, int mixes, int dtype,
                                       int64_t stream) {
  const size_t items = (size_t)batch * seq_len * hidden;
  const int blocks = (items + RWKV7_SHIFT_THREADS - 1) / RWKV7_SHIFT_THREADS;
  const cudaStream_t s = (cudaStream_t)stream;
#define RWKV7_SHIFT(T)                                                         \
  rwkv7_token_shift_kernel<T><<<blocks, RWKV7_SHIFT_THREADS, 0, s>>>(          \
      (const T *)x, (const T *)last, (const T *)mix, (T *)out, (T *)new_last,  \
      batch, seq_len, hidden, mixes)
  if (dtype == 0)
    RWKV7_SHIFT(__half);
  else if (dtype == 1)
    RWKV7_SHIFT(__nv_bfloat16);
  else
    RWKV7_SHIFT(float);
#undef RWKV7_SHIFT
}

extern "C" void rwkv7_time_mix_cuda(const Rwkv7TimeMixParams *p, int batch,
                                    int dtype, int64_t stream) {
  const int blocks = batch * p->heads;
  const cudaStream_t s = (cudaStream_t)stream;
  if (dtype == 0)
    rwkv7_time_mix_kernel<__half><<<blocks, RWKV7_HEAD_SIZE, 0, s>>>(*p);
  else if (dtype == 1)
    rwkv7_time_mix_kernel<__nv_bfloat16><<<blocks, RWKV7_HEAD_SIZE, 0, s>>>(*p);
  else
    rwkv7_time_mix_kernel<float><<<blocks, RWKV7_HEAD_SIZE, 0, s>>>(*p);
}
