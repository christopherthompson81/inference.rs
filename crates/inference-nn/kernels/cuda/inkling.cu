// Inkling 2's per-slot elementwise chains, each one launch: the rows' L2 clip, unit RMS, the depth mix (softmax
// attention of learned queries over the unit-normed states) and the short causal convolution with its state. Inputs
// in f32 or bf16, outputs f32.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int WARP = 32;
constexpr int ROW_THREADS = 256;
constexpr int MAX_STATES = 16;
constexpr int MAX_QUERIES = 4;
// a depth mix's block
constexpr int MIX_THREADS = 512;

__device__ __forceinline__ float load(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void store(float* p, int64_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store(__nv_bfloat16* p, int64_t i, float v) { p[i] = __float2bfloat16(v); }

__device__ __forceinline__ float warp_sum(float x) {
  for (int o = WARP / 2; o > 0; o /= 2) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

// a warp a row of `n`: x * scale[row % scale_rows] (when given), its L2 norm brought down to at most `threshold`
template <typename T>
__global__ void l2_clip_kernel(const T* x, const float* scale, float* out, int64_t rows, int n, int scale_rows,
                               float threshold, float eps) {
  const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / WARP;
  const int lane = threadIdx.x % WARP;
  if (row >= rows) return;
  const float* s = scale ? scale + (row % scale_rows) * n : nullptr;
  float sq = 0.0f;
  for (int i = lane; i < n; i += WARP) {
    const float v = load(x, row * n + i) * (s ? s[i] : 1.0f);
    sq += v * v;
  }
  const float norm = sqrtf(warp_sum(sq)) + eps;
  const float factor = fminf(fmaxf(threshold / norm, 0.0f), 1.0f);
  for (int i = lane; i < n; i += WARP) out[row * n + i] = load(x, row * n + i) * (s ? s[i] : 1.0f) * factor;
}

// a warp a row of `n`: x / sqrt(mean(x^2) + eps)
template <typename T>
__global__ void unit_rms_kernel(const T* x, float* out, int64_t rows, int n, float eps) {
  const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / WARP;
  const int lane = threadIdx.x % WARP;
  if (row >= rows) return;
  float sq = 0.0f;
  for (int i = lane; i < n; i += WARP) {
    const float v = load(x, row * n + i);
    sq += v * v;
  }
  const float inv = rsqrtf(warp_sum(sq) / n + eps);
  for (int i = lane; i < n; i += WARP) out[row * n + i] = load(x, row * n + i) * inv;
}

// a block a row: the states (`blocks` (s - 1, rows, hidden) then `current` (rows, hidden)) unit-normed, each query's
// softmax over them (scores over sqrt(hidden)), out (rows, queries, hidden) the weighted sum of the normed states.
// Each thread takes a few hidden indices of every state at once, its partial sums reduced across the block once.
template <typename T, typename O>
__global__ void depth_mix_kernel(const T* blocks, const T* current, const float* queries, O* out, int s_count,
                                 int64_t rows, int q_count, int hidden, float eps) {
  // per state: its sum of squares, then each query's dot
  constexpr int SUMS = MAX_STATES * (1 + MAX_QUERIES);
  __shared__ float partial[MIX_THREADS / WARP][SUMS];
  __shared__ float total[SUMS];
  __shared__ float inv_rms[MAX_STATES];
  __shared__ float weights[MAX_QUERIES][MAX_STATES];
  const int64_t row = blockIdx.x;
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  auto state = [&](int s) -> const T* {
    return s < s_count - 1 ? blocks + (int64_t(s) * rows + row) * hidden : current + row * hidden;
  };
  float sums[SUMS];
#pragma unroll
  for (int k = 0; k < SUMS; k++) sums[k] = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float qv[MAX_QUERIES];
#pragma unroll
    for (int q = 0; q < MAX_QUERIES; q++) qv[q] = q < q_count ? queries[q * hidden + i] : 0.0f;
#pragma unroll
    for (int s = 0; s < MAX_STATES; s++) {
      if (s >= s_count) break;
      const float v = load(state(s), i);
      sums[s * (1 + MAX_QUERIES)] += v * v;
#pragma unroll
      for (int q = 0; q < MAX_QUERIES; q++) sums[s * (1 + MAX_QUERIES) + 1 + q] += v * qv[q];
    }
  }
#pragma unroll
  for (int k = 0; k < SUMS; k++) {
    const float v = warp_sum(sums[k]);
    if (lane == 0) partial[warp][k] = v;
  }
  __syncthreads();
  for (int k = threadIdx.x; k < SUMS; k += blockDim.x) {
    float acc = 0.0f;
    for (int w = 0; w < MIX_THREADS / WARP; w++) acc += partial[w][k];
    total[k] = acc;
  }
  __syncthreads();
  if (threadIdx.x < q_count) {
    const int q = threadIdx.x;
    float m = -INFINITY;
    for (int s = 0; s < s_count; s++) {
      const float inv = rsqrtf(total[s * (1 + MAX_QUERIES)] / hidden + eps);
      if (q == 0) inv_rms[s] = inv;
      weights[q][s] = total[s * (1 + MAX_QUERIES) + 1 + q] * inv * rsqrtf((float)hidden);
      m = fmaxf(m, weights[q][s]);
    }
    float l = 0.0f;
    for (int s = 0; s < s_count; s++) {
      weights[q][s] = __expf(weights[q][s] - m);
      l += weights[q][s];
    }
    for (int s = 0; s < s_count; s++) weights[q][s] /= l;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float acc[MAX_QUERIES] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < MAX_STATES; s++) {
      if (s >= s_count) break;
      const float v = load(state(s), i) * inv_rms[s];
#pragma unroll
      for (int q = 0; q < MAX_QUERIES; q++) acc[q] += weights[q][s] * v;
    }
    for (int q = 0; q < q_count; q++) store(out, (row * q_count + q) * hidden + i, acc[q]);
  }
}

// a loop controller's input: x and message (rows, hidden) each unit-normed, side by side in out (rows, 2 hidden) f32;
// a warp a (row, half)
template <typename X, typename M>
__global__ void controller_in_kernel(const X* x, const M* message, float* out, int64_t rows, int hidden, float eps) {
  const int64_t w = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / WARP;
  const int lane = threadIdx.x % WARP;
  if (w >= 2 * rows) return;
  const int64_t row = w / 2;
  const bool is_x = w % 2 == 0;
  auto at = [&](int i) { return is_x ? load(x, row * hidden + i) : load(message, row * hidden + i); };
  float sq = 0.0f;
  for (int i = lane; i < hidden; i += WARP) {
    const float v = at(i);
    sq += v * v;
  }
  const float inv = rsqrtf(warp_sum(sq) / hidden + eps);
  float* dst = out + row * 2 * hidden + (w % 2) * hidden;
  for (int i = lane; i < hidden; i += WARP) dst[i] = at(i) * inv;
}

// a loop controller's output: (1 - tanh(forget)) x + tanh(gain) message, in x's dtype
template <typename T, typename M>
__global__ void controller_out_kernel(const T* x, const M* message, const float* gain, const float* forget, T* out,
                                      int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float xv = load(x, i);
  store(out, i, xv - tanhf(forget[i]) * xv + tanhf(gain[i]) * load(message, i));
}

// the softcapped SwiGLU: gate_cap tanh(a / gate_cap) sigmoid(a) times up_cap tanh(b / up_cap), a and b (rows, cols)
// with their own row strides, out (rows, cols) contiguous, all of one dtype
template <typename T>
__global__ void softcap_silu_mul_kernel(const T* a, int64_t a_row, const T* b, int64_t b_row, T* out, int64_t rows,
                                        int cols, float gate_cap, float up_cap) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  const int64_t r = i / cols;
  const int c = i % cols;
  const float g = load(a, r * a_row + c);
  const float u = load(b, r * b_row + c);
  const float capped = gate_cap * tanhf(g / gate_cap) / (1.0f + __expf(-g));
  store(out, i, capped * up_cap * tanhf(u / up_cap));
}

// the attention's output gate: x (f32) times sigmoid(gate), written in out's dtype
template <typename G, typename O>
__global__ void sigmoid_gate_kernel(const float* x, const G* gate, O* out, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  store(out, i, x[i] / (1.0f + __expf(-load(gate, i))));
}

// x (b, t, c) through a causal depthwise convolution of `width` taps with a residual, its last width - 1 inputs
// carried in `state` (b, c, width - 1; batch stride `state_batch`); `taps` (b, t, heads, width) adds an input-dependent
// weight a lag shared within a head; the sum over sqrt(terms). Writes out (b, t, c) and new_state (b, c, width - 1).
template <typename X, typename S, typename O>
__global__ void short_conv_kernel(const X* x, const S* state, int64_t state_batch, const float* weight,
                                  const float* taps, int heads, O* out, float* new_state, int b, int t, int c,
                                  int width) {
  const int64_t outs = int64_t(b) * t * c;
  const int64_t states = int64_t(b) * c * (width - 1);
  const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // ctx[j], j < width - 1 the state, else the input at j - (width - 1)
  auto ctx = [&](int bi, int ci, int j) -> float {
    if (j < width - 1) return load(state, bi * state_batch + int64_t(ci) * (width - 1) + j);
    return load(x, (int64_t(bi) * t + (j - (width - 1))) * c + ci);
  };
  if (e < outs) {
    const int ci = e % c;
    const int ti = (e / c) % t;
    const int bi = e / (int64_t(c) * t);
    float acc = load(x, e);
    for (int k = 0; k < width; k++) acc += weight[ci * width + k] * ctx(bi, ci, ti + k);
    int terms = 1 + width;
    if (taps) {
      const int head = ci / (c / heads);
      const float* tap = taps + ((int64_t(bi) * t + ti) * heads + head) * width;
      for (int lag = 0; lag < width; lag++) acc += ctx(bi, ci, ti + width - 1 - lag) * tap[lag];
      terms += width;
    }
    store(out, e, acc * rsqrtf((float)terms));
  } else if (e < outs + states) {
    const int64_t r = e - outs;
    const int j = r % (width - 1);
    const int ci = (r / (width - 1)) % c;
    const int bi = r / (int64_t(c) * (width - 1));
    new_state[r] = ctx(bi, ci, t + j);
  }
}

int64_t blocks_for(int64_t threads) { return (threads + ROW_THREADS - 1) / ROW_THREADS; }

}  // namespace

// dtype 0 f32, 1 bf16
// two dtype codes (0 f32, 1 bf16) choosing a template pair
#define PAIR(XC, MC, CALL)                                       \
  if (XC == 0 && MC == 0) {                                      \
    CALL(float, float);                                          \
  } else if (XC == 0) {                                          \
    CALL(float, __nv_bfloat16);                                  \
  } else if (MC == 0) {                                          \
    CALL(__nv_bfloat16, float);                                  \
  } else {                                                       \
    CALL(__nv_bfloat16, __nv_bfloat16);                          \
  }

extern "C" int inkling_l2_clip(const void* x, int dtype, const float* scale, float* out, int64_t rows, int n,
                               int scale_rows, float threshold, float eps, int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t grid = blocks_for(rows * WARP);
  if (dtype == 0) {
    l2_clip_kernel<<<grid, ROW_THREADS, 0, s>>>((const float*)x, scale, out, rows, n, scale_rows, threshold, eps);
  } else {
    l2_clip_kernel<<<grid, ROW_THREADS, 0, s>>>((const __nv_bfloat16*)x, scale, out, rows, n, scale_rows, threshold,
                                                eps);
  }
  return (int)cudaGetLastError();
}

extern "C" int inkling_unit_rms(const void* x, int dtype, float* out, int64_t rows, int n, float eps,
                                int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t grid = blocks_for(rows * WARP);
  if (dtype == 0) {
    unit_rms_kernel<<<grid, ROW_THREADS, 0, s>>>((const float*)x, out, rows, n, eps);
  } else {
    unit_rms_kernel<<<grid, ROW_THREADS, 0, s>>>((const __nv_bfloat16*)x, out, rows, n, eps);
  }
  return (int)cudaGetLastError();
}

extern "C" int inkling_depth_mix(const void* blocks, const void* current, int dtype, const float* queries, void* out,
                                 int out_dtype, int s_count, int64_t rows, int q_count, int hidden, float eps,
                                 int64_t stream) {
  if (s_count > MAX_STATES || q_count > MAX_QUERIES) return -1;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
#define MIX(T, O)                                                                                                \
  depth_mix_kernel<<<rows, MIX_THREADS, 0, s>>>((const T*)blocks, (const T*)current, queries, (O*)out, s_count, rows, \
                                                q_count, hidden, eps)
  PAIR(dtype, out_dtype, MIX)
#undef MIX
  return (int)cudaGetLastError();
}

// out in f32, or in x's dtype when `out_like_x`
extern "C" int inkling_short_conv(const void* x, int x_dtype, const void* state, int state_dtype,
                                  int64_t state_batch, const float* weight, const float* taps, int heads, void* out,
                                  int out_like_x, float* new_state, int b, int t, int c, int width, int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t grid = blocks_for(int64_t(b) * t * c + int64_t(b) * c * (width - 1));
#define CONV_OUT(X, S, O)                                                                                   \
  short_conv_kernel<<<grid, ROW_THREADS, 0, s>>>((const X*)x, (const S*)state, state_batch, weight, taps, heads, \
                                                 (O*)out, new_state, b, t, c, width)
#define CONV(X, S)            \
  if (out_like_x) {           \
    CONV_OUT(X, S, X);        \
  } else {                    \
    CONV_OUT(X, S, float);    \
  }
  PAIR(x_dtype, state_dtype, CONV)
#undef CONV
#undef CONV_OUT
  return (int)cudaGetLastError();
}

extern "C" int inkling_controller_in(const void* x, int x_dtype, const void* message, int m_dtype, float* out,
                                     int64_t rows, int hidden, float eps, int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t grid = blocks_for(2 * rows * WARP);
#define IN(X, M) controller_in_kernel<<<grid, ROW_THREADS, 0, s>>>((const X*)x, (const M*)message, out, rows, hidden, eps)
  PAIR(x_dtype, m_dtype, IN)
#undef IN
  return (int)cudaGetLastError();
}

extern "C" int inkling_controller_out(const void* x, int x_dtype, const void* message, int m_dtype, const float* gain,
                                      const float* forget, void* out, int64_t n, int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t grid = blocks_for(n);
#define OUT(X, M) \
  controller_out_kernel<<<grid, ROW_THREADS, 0, s>>>((const X*)x, (const M*)message, gain, forget, (X*)out, n)
  PAIR(x_dtype, m_dtype, OUT)
#undef OUT
  return (int)cudaGetLastError();
}

extern "C" int inkling_softcap_silu_mul(const void* a, int64_t a_row, const void* b, int64_t b_row, int dtype,
                                        void* out, int64_t rows, int cols, float gate_cap, float up_cap,
                                        int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t grid = blocks_for(rows * cols);
  if (dtype == 0) {
    softcap_silu_mul_kernel<<<grid, ROW_THREADS, 0, s>>>((const float*)a, a_row, (const float*)b, b_row, (float*)out,
                                                         rows, cols, gate_cap, up_cap);
  } else {
    softcap_silu_mul_kernel<<<grid, ROW_THREADS, 0, s>>>((const __nv_bfloat16*)a, a_row, (const __nv_bfloat16*)b,
                                                         b_row, (__nv_bfloat16*)out, rows, cols, gate_cap, up_cap);
  }
  return (int)cudaGetLastError();
}

extern "C" int inkling_sigmoid_gate(const float* x, const void* gate, int gate_dtype, void* out, int out_dtype,
                                    int64_t n, int64_t stream) {
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t grid = blocks_for(n);
#define GATE(G, O) sigmoid_gate_kernel<<<grid, ROW_THREADS, 0, s>>>(x, (const G*)gate, (O*)out, n)
  PAIR(gate_dtype, out_dtype, GATE)
#undef GATE
  return (int)cudaGetLastError();
}
