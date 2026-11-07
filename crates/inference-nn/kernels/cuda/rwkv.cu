// RWKV-7's WKV recurrence, adapted from llama.cpp's rwkv_wkv7_f32 (ggml-cuda
// wkv.cu). Per head, with the state S[i][j] (i a value row, j a key column):
//   sa = sum_j a[j] S[i][j]
//   S[i][j] = S[i][j] w[j] + v[i] k[j] + sa b[j]
//   y[i] = sum_j S[i][j] r[j]
// r, w, k, v, a and b are (batch, seq_len, heads, head_size); state (batch,
// heads, head_size, head_size), advanced in place; y like r.

#include <cstdint>

#define RWKV7_HEAD_SIZE 64

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
