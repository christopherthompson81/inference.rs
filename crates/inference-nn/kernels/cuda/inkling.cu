// Inkling 2's per-slot elementwise chains, each one launch: the rows' L2 clip, unit RMS, the depth mix (softmax
// attention of learned queries over the unit-normed states) and the short causal convolution with its state. Inputs
// in f32 or bf16, outputs f32.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>

namespace {

constexpr int WARP = 32;
constexpr int ROW_THREADS = 256;
constexpr int MAX_STATES = 16;
constexpr int MAX_QUERIES = 4;
// a depth mix's block
constexpr int MIX_THREADS = 512;
// a few rows (decode, verify) split each row's hidden dim over blocks of MIX_CHUNK values instead of one block a row
constexpr int64_t MIX_SPLIT_MAX_ROWS = 16;
constexpr int MIX_CHUNK = 128;
constexpr int MIX_SPLIT_THREADS = 128;
constexpr int MIX_SUMS = MAX_STATES * (1 + MAX_QUERIES);
// a fused loop controller's rows (decode, verify) and its widest bottleneck
constexpr int64_t CONTROLLER_MAX_ROWS = 16;
constexpr int CONTROLLER_MAX_RANK = 128;
constexpr int CONTROLLER_THREADS = 256;
// a Markov pick's blocks a batch row (each a slice of the vocabulary) and its block
constexpr int PICK_BLOCKS = 256;
constexpr int PICK_THREADS = 256;
// the down projection's dots split over this many slices of the hidden dim, their partial sums the up kernel's to add
constexpr int CONTROLLER_DOWN_CHUNKS = 4;
// a down block's partials: the x half's dot, the message half's, and their sums of squares
constexpr int CONTROLLER_PARTIALS = 4;

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

// one of a paged attention step's row operations: rows of `n` read in `x_dtype`, written f32 or bf16 (`out_dtype`):
// an L2 clip (after the optional per-row scale, `scale_rows` of them), a unit RMS, or a cast (n 1)
constexpr int ROWS_OPS = 5;
enum RowsMode : int { ROWS_CLIP = 0, ROWS_RMS = 1, ROWS_CAST = 2 };
struct RowsOp {
  const void* x;
  void* out;
  const float* scale;
  int64_t rows;
  int n;
  int scale_rows;
  int x_dtype;
  int out_dtype;
  int mode;
  float param;
};
struct RowsOps {
  RowsOp op[ROWS_OPS];
  int count;
};

__device__ __forceinline__ void rows_store(void* out, int dtype, int64_t i, float v) {
  if (dtype == 0) {
    static_cast<float*>(out)[i] = v;
  } else {
    static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16(v);
  }
}

template <typename T>
__device__ __forceinline__ void rows_op(const RowsOp& o, int64_t row, int lane, float eps) {
  const T* x = static_cast<const T*>(o.x);
  if (o.mode == ROWS_CAST) {
    if (lane == 0) rows_store(o.out, o.out_dtype, row, load(x, row));
    return;
  }
  const float* s = o.scale ? o.scale + (row % o.scale_rows) * o.n : nullptr;
  float sq = 0.0f;
  for (int i = lane; i < o.n; i += WARP) {
    const float v = load(x, row * o.n + i) * (s ? s[i] : 1.0f);
    sq += v * v;
  }
  const float total = warp_sum(sq);
  const float factor = o.mode == ROWS_CLIP ? fminf(fmaxf(o.param / (sqrtf(total) + eps), 0.0f), 1.0f)
                                           : rsqrtf(total / o.n + o.param);
  for (int i = lane; i < o.n; i += WARP)
    rows_store(o.out, o.out_dtype, row * o.n + i, load(x, row * o.n + i) * (s ? s[i] : 1.0f) * factor);
}

// a warp a row over every op's rows in turn: one launch for a step's q, k, v, r and sinks
__global__ void attention_rows_kernel(RowsOps ops, float eps) {
  int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / WARP;
  const int lane = threadIdx.x % WARP;
  for (int k = 0; k < ops.count; k++) {
    const RowsOp& o = ops.op[k];
    if (row < o.rows) {
      if (o.x_dtype == 0) {
        rows_op<float>(o, row, lane, eps);
      } else {
        rows_op<__nv_bfloat16>(o, row, lane, eps);
      }
      return;
    }
    row -= o.rows;
  }
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

// the split depth mix's first half: block (row, chunk)'s sums of squares and query dots over its chunk of every state,
// partials (rows, chunks, MIX_SUMS)
template <typename T>
__global__ void depth_mix_partials_kernel(const T* blocks, const T* current, const float* queries, float* partials,
                                          int s_count, int64_t rows, int q_count, int hidden) {
  __shared__ float warp_sums[MIX_SPLIT_THREADS / WARP][MIX_SUMS];
  const int64_t row = blockIdx.x;
  const int chunk = blockIdx.y;
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int lo = chunk * MIX_CHUNK;
  const int hi = min(lo + MIX_CHUNK, hidden);
  float sums[MIX_SUMS];
#pragma unroll
  for (int k = 0; k < MIX_SUMS; k++) sums[k] = 0.0f;
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    float qv[MAX_QUERIES];
#pragma unroll
    for (int q = 0; q < MAX_QUERIES; q++) qv[q] = q < q_count ? queries[q * hidden + i] : 0.0f;
#pragma unroll
    for (int s = 0; s < MAX_STATES; s++) {
      if (s >= s_count) break;
      const T* state = s < s_count - 1 ? blocks + (int64_t(s) * rows + row) * hidden : current + row * hidden;
      const float v = load(state, i);
      sums[s * (1 + MAX_QUERIES)] += v * v;
#pragma unroll
      for (int q = 0; q < MAX_QUERIES; q++) sums[s * (1 + MAX_QUERIES) + 1 + q] += v * qv[q];
    }
  }
  const int used = s_count * (1 + MAX_QUERIES);
#pragma unroll
  for (int k = 0; k < MIX_SUMS; k++) {
    if (k >= used) break;
    const float v = warp_sum(sums[k]);
    if (lane == 0) warp_sums[warp][k] = v;
  }
  __syncthreads();
  float* dst = partials + (row * gridDim.y + chunk) * MIX_SUMS;
  for (int k = threadIdx.x; k < used; k += blockDim.x) {
    float acc = 0.0f;
    for (int w = 0; w < MIX_SPLIT_THREADS / WARP; w++) acc += warp_sums[w][k];
    dst[k] = acc;
  }
}

// the split depth mix's second half: block (row, chunk) sums the row's partials, takes each query's softmax and
// writes its chunk of the output
template <typename T, typename O>
__global__ void depth_mix_apply_kernel(const T* blocks, const T* current, const float* partials, O* out, int s_count,
                                       int64_t rows, int q_count, int hidden, float eps) {
  __shared__ float total[MIX_SUMS];
  __shared__ float inv_rms[MAX_STATES];
  __shared__ float weights[MAX_QUERIES][MAX_STATES];
  const int64_t row = blockIdx.x;
  const int chunks = gridDim.y;
  const int used = s_count * (1 + MAX_QUERIES);
  for (int k = threadIdx.x; k < used; k += blockDim.x) {
    float acc = 0.0f;
    for (int c = 0; c < chunks; c++) acc += partials[(row * chunks + c) * MIX_SUMS + k];
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
  const int lo = blockIdx.y * MIX_CHUNK;
  const int hi = min(lo + MIX_CHUNK, hidden);
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    float acc[MAX_QUERIES] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < MAX_STATES; s++) {
      if (s >= s_count) break;
      const T* state = s < s_count - 1 ? blocks + (int64_t(s) * rows + row) * hidden : current + row * hidden;
      const float v = load(state, i) * inv_rms[s];
#pragma unroll
      for (int q = 0; q < MAX_QUERIES; q++) acc[q] += weights[q][s] * v;
    }
    for (int q = 0; q < q_count; q++) store(out, (row * q_count + q) * hidden + i, acc[q]);
  }
}

// a fused loop controller's bottleneck, block (r, chunk): over its slice of the hidden dim, down[r]'s dots with x's
// half and the message's, and their sums of squares, a row at a time (the slice read from memory once); partials
// (rows, rank, chunks, 4)
template <typename X, typename M>
__global__ void controller_down_kernel(const X* x, const M* message, const float* down, float* partials, int64_t rows,
                                       int hidden, int rank) {
  __shared__ float warp_sums[CONTROLLER_THREADS / WARP][CONTROLLER_PARTIALS];
  const int r = blockIdx.x;
  const int chunk = blockIdx.y;
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int span = (hidden + CONTROLLER_DOWN_CHUNKS - 1) / CONTROLLER_DOWN_CHUNKS;
  const int lo = chunk * span;
  const int hi = min(lo + span, hidden);
  const float* dx = down + int64_t(r) * 2 * hidden;
  const float* dm = dx + hidden;
  for (int64_t row = 0; row < rows; row++) {
    float acc_x = 0.0f, acc_m = 0.0f, sq_x = 0.0f, sq_m = 0.0f;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      const float xv = load(x, row * hidden + i);
      const float mv = load(message, row * hidden + i);
      acc_x += dx[i] * xv;
      acc_m += dm[i] * mv;
      sq_x += xv * xv;
      sq_m += mv * mv;
    }
    const float sums[CONTROLLER_PARTIALS] = {warp_sum(acc_x), warp_sum(acc_m), warp_sum(sq_x), warp_sum(sq_m)};
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < CONTROLLER_PARTIALS; k++) warp_sums[warp][k] = sums[k];
    }
    __syncthreads();
    if (threadIdx.x < CONTROLLER_PARTIALS) {
      float t = 0.0f;
      for (int w = 0; w < CONTROLLER_THREADS / WARP; w++) t += warp_sums[w][threadIdx.x];
      partials[((row * rank + r) * CONTROLLER_DOWN_CHUNKS + chunk) * CONTROLLER_PARTIALS + threadIdx.x] = t;
    }
    __syncthreads();
  }
}

// a fused loop controller's output, a warp an index i for every row: the bottleneck of each row from the down slices,
// gain and forget from it through `gain_forget` (2 hidden, rank) by lanes over the rank (each row of the weight read
// once, kept in registers across the rows), (1 - tanh(forget)) x + tanh(gain) message in x's dtype
template <typename X, typename M>
__global__ void controller_up_kernel(const X* x, const M* message, const float* gain_forget, const float* partials,
                                     X* out, int64_t rows, int hidden, int rank, float eps) {
  constexpr int PER_LANE = CONTROLLER_MAX_RANK / WARP;
  __shared__ float zs[CONTROLLER_MAX_ROWS][CONTROLLER_MAX_RANK];
  for (int k = threadIdx.x; k < rows * rank; k += blockDim.x) {
    const int64_t row = k / rank;
    const int r = k % rank;
    float t[CONTROLLER_PARTIALS] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float* p = partials + (row * rank + r) * CONTROLLER_DOWN_CHUNKS * CONTROLLER_PARTIALS;
    for (int c = 0; c < CONTROLLER_DOWN_CHUNKS; c++) {
#pragma unroll
      for (int q = 0; q < CONTROLLER_PARTIALS; q++) t[q] += p[c * CONTROLLER_PARTIALS + q];
    }
    zs[row][r] = rsqrtf(t[2] / hidden + eps) * t[0] + rsqrtf(t[3] / hidden + eps) * t[1];
  }
  __syncthreads();
  const int i = blockIdx.x * (CONTROLLER_THREADS / WARP) + threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  if (i >= hidden) return;
  const float* g = gain_forget + int64_t(i) * rank;
  const float* f = gain_forget + int64_t(hidden + i) * rank;
  float gw[PER_LANE], fw[PER_LANE];
#pragma unroll
  for (int k = 0; k < PER_LANE; k++) {
    const int r = lane + k * WARP;
    gw[k] = r < rank ? g[r] : 0.0f;
    fw[k] = r < rank ? f[r] : 0.0f;
  }
  for (int64_t row = 0; row < rows; row++) {
    float gain = 0.0f, forget = 0.0f;
#pragma unroll
    for (int k = 0; k < PER_LANE; k++) {
      const int r = lane + k * WARP;
      const float zr = r < rank ? zs[row][r] : 0.0f;
      gain += gw[k] * zr;
      forget += fw[k] * zr;
    }
    gain = warp_sum(gain);
    forget = warp_sum(forget);
    if (lane == 0) {
      const float xv = load(x, row * hidden + i);
      store(out, row * hidden + i, xv - tanhf(forget) * xv + tanhf(gain) * load(message, row * hidden + i));
    }
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

// the larger value, the lower index on a tie; NaN loses
__device__ __forceinline__ void pick_better(float& v, int& i, float ov, int oi) {
  if (ov != ov) return;
  if (v != v || ov > v || (ov == v && oi < i)) {
    v = ov;
    i = oi;
  }
}

__device__ __forceinline__ void pick_block(float& v, int& i) {
  __shared__ float vs[PICK_THREADS / WARP];
  __shared__ int is[PICK_THREADS / WARP];
  for (int off = WARP / 2; off > 0; off /= 2) pick_better(v, i, __shfl_down_sync(0xffffffff, v, off), __shfl_down_sync(0xffffffff, i, off));
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  if (lane == 0) {
    vs[warp] = v;
    is[warp] = i;
  }
  __syncthreads();
  if (warp == 0) {
    v = lane < PICK_THREADS / WARP ? vs[lane] : NAN;
    i = lane < PICK_THREADS / WARP ? is[lane] : INT_MAX;
    for (int off = WARP / 2; off > 0; off /= 2) pick_better(v, i, __shfl_down_sync(0xffffffff, v, off), __shfl_down_sync(0xffffffff, i, off));
  }
}

// a slice of the vocabulary: each row's `logits[v] + fold_b[v] + fold_w[v] . latent`, a warp a row, its best kept
template <typename L>
__global__ void markov_pick_partial_kernel(const L* logits, int64_t logits_batch, const __nv_bfloat16* fold_w,
                                           const float* fold_b, const float* latent, int rank, int vocab,
                                           float* part_v, int* part_i) {
  extern __shared__ float lat[];
  const int bi = blockIdx.y;
  for (int r = threadIdx.x; r < rank; r += blockDim.x) lat[r] = latent[int64_t(bi) * rank + r];
  __syncthreads();
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  const int per = (vocab + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * per, hi = min(vocab, lo + per);
  float best = NAN;
  int at = INT_MAX;
  for (int v = lo + warp; v < hi; v += PICK_THREADS / WARP) {
    const __nv_bfloat16* w = fold_w + int64_t(v) * rank;
    float dot = 0.0f;
    for (int r = lane * 8; r < rank; r += WARP * 8) {
      const uint4 raw = *reinterpret_cast<const uint4*>(w + r);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const float2 f = __bfloat1622float2(h[k]);
        dot += f.x * lat[r + 2 * k] + f.y * lat[r + 2 * k + 1];
      }
    }
    for (int off = WARP / 2; off > 0; off /= 2) dot += __shfl_xor_sync(0xffffffff, dot, off);
    if (lane == 0) pick_better(best, at, load(logits, int64_t(bi) * logits_batch + v) + fold_b[v] + dot, v);
  }
  pick_block(best, at);
  if (threadIdx.x == 0) {
    part_v[bi * gridDim.x + blockIdx.x] = best;
    part_i[bi * gridDim.x + blockIdx.x] = at;
  }
}

__global__ void markov_pick_final_kernel(const float* part_v, const int* part_i, int parts, uint32_t* out) {
  const int bi = blockIdx.x;
  float best = NAN;
  int at = INT_MAX;
  for (int p = threadIdx.x; p < parts; p += blockDim.x) pick_better(best, at, part_v[bi * parts + p], part_i[bi * parts + p]);
  pick_block(best, at);
  if (threadIdx.x == 0) out[bi] = at == INT_MAX ? 0u : uint32_t(at);
}

// the state window a thread keeps in registers
constexpr int CONV_MAX_HIST = 8;

// short_conv_kernel over a slot pool `[pool_rows, pool_c, hist]`, a thread a (row, channel): the state read from the
// row's slot at `channel_off`, the rows run through the window, the new state written back in place. A slot past
// the pool (a padded row) reads zeros and writes nothing. `before` and `stash` (when given) take the state as read
// `[b, pool_c, hist]` and the inputs `[b, t, pool_c]`, a verify's rollback.
template <typename X, typename O>
__global__ void short_conv_pool_kernel(const X* x, float* pool, const uint32_t* slots, int pool_rows, int pool_c,
                                       int channel_off, const float* weight, const float* taps, int heads, O* out,
                                       float* before, X* stash, int b, int t, int c, int width) {
  const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= int64_t(b) * c) return;
  const int ci = e % c;
  const int bi = e / c;
  const int hist = width - 1;
  const uint32_t slot = slots[bi];
  const bool live = slot < uint32_t(pool_rows);
  float* row = pool + (int64_t(live ? slot : 0) * pool_c + channel_off + ci) * hist;
  float win[CONV_MAX_HIST];
#pragma unroll
  for (int j = 0; j < CONV_MAX_HIST; j++) win[j] = j < hist && live ? row[j] : 0.0f;
  if (before) {
    float* dst = before + (int64_t(bi) * pool_c + channel_off + ci) * hist;
    for (int j = 0; j < hist; j++) dst[j] = win[j];
  }
  const int terms = 1 + width + (taps ? width : 0);
  const float scale = rsqrtf((float)terms);
  const int head = ci / (c / heads);
  for (int ti = 0; ti < t; ti++) {
    const int64_t at = (int64_t(bi) * t + ti) * c + ci;
    const float xv = load(x, at);
    if (stash) stash[(int64_t(bi) * t + ti) * pool_c + channel_off + ci] = x[at];
    float acc = xv + weight[ci * width + hist] * xv;
#pragma unroll
    for (int k = 0; k < CONV_MAX_HIST; k++)
      if (k < hist) acc += weight[ci * width + k] * win[k];
    if (taps) {
      const float* tap = taps + ((int64_t(bi) * t + ti) * heads + head) * width;
      acc += xv * tap[0];
#pragma unroll
      for (int lag = 1; lag < CONV_MAX_HIST + 1; lag++)
        if (lag < width) acc += win[hist - lag] * tap[lag];
    }
    store(out, at, acc * scale);
#pragma unroll
    for (int j = 0; j + 1 < CONV_MAX_HIST; j++)
      if (j + 1 < hist) win[j] = win[j + 1];
    win[hist - 1] = xv;
  }
  if (live)
    for (int j = 0; j < hist; j++) row[j] = win[j];
}

// a slot of a verify step's convolution rollback: its pool `[slots, c, hist]`, the batch's pool rows, its state
// before the step `[b, c, hist]` and the conv inputs over the step `[b, t, c]`
struct ConvRollbackSlot {
  float* pool;
  const uint32_t* rows;
  const float* state;
  const void* inputs;
};

// each batch row that kept `keep[row]` of the step's rows takes the window of `[state, inputs]` ending at its last kept
// row; a negative `keep` leaves the row as the step wrote it
template <typename I>
__global__ void conv_rollback_kernel(const ConvRollbackSlot* slots, const int32_t* keep, int b, int t, int c,
                                     int hist) {
  const ConvRollbackSlot s = slots[blockIdx.y];
  const int64_t e = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= int64_t(b) * c * hist) return;
  const int j = e % hist;
  const int ci = (e / hist) % c;
  const int bi = e / (int64_t(c) * hist);
  const int k = keep[bi];
  if (k < 0) return;
  const int src = k + j;
  const float v = src < hist ? s.state[(int64_t(bi) * c + ci) * hist + src]
                             : load((const I*)s.inputs, (int64_t(bi) * t + (src - hist)) * c + ci);
  s.pool[(int64_t(s.rows[bi]) * c + ci) * hist + j] = v;
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

// the ops (`RowsOps`, up to ROWS_OPS) in one launch; `eps` the clips' norm epsilon
extern "C" int inkling_attention_rows(const void* ops, int64_t total_rows, float eps, int64_t stream) {
  const RowsOps& o = *static_cast<const RowsOps*>(ops);
  if (o.count > ROWS_OPS) return -1;
  attention_rows_kernel<<<blocks_for(total_rows * WARP), ROW_THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      o, eps);
  return (int)cudaGetLastError();
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

// the f32 partials a depth mix of `rows` takes (0: one block a row, no scratch)
extern "C" int64_t inkling_depth_mix_scratch(int64_t rows, int hidden) {
  if (rows > MIX_SPLIT_MAX_ROWS) return 0;
  return rows * ((hidden + MIX_CHUNK - 1) / MIX_CHUNK) * MIX_SUMS;
}

// `partials` of inkling_depth_mix_scratch's size, null when it is 0
extern "C" int inkling_depth_mix(const void* blocks, const void* current, int dtype, const float* queries, void* out,
                                 int out_dtype, float* partials, int s_count, int64_t rows, int q_count, int hidden,
                                 float eps, int64_t stream) {
  if (s_count > MAX_STATES || q_count > MAX_QUERIES) return -1;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (partials) {
    const dim3 grid((unsigned)rows, (unsigned)((hidden + MIX_CHUNK - 1) / MIX_CHUNK));
#define SPLIT(T, O)                                                                                           \
  depth_mix_partials_kernel<<<grid, MIX_SPLIT_THREADS, 0, s>>>((const T*)blocks, (const T*)current, queries, \
                                                               partials, s_count, rows, q_count, hidden);    \
  depth_mix_apply_kernel<<<grid, MIX_SPLIT_THREADS, 0, s>>>((const T*)blocks, (const T*)current, partials,  \
                                                            (O*)out, s_count, rows, q_count, hidden, eps)
    PAIR(dtype, out_dtype, SPLIT)
#undef SPLIT
    return (int)cudaGetLastError();
  }
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

// the f32 partials a fused controller of `rows` takes (0: too many rows or too wide a rank, the unfused path)
extern "C" int64_t inkling_controller_scratch(int64_t rows, int rank) {
  if (rows > CONTROLLER_MAX_ROWS || rank > CONTROLLER_MAX_RANK) return 0;
  return rows * rank * CONTROLLER_DOWN_CHUNKS * CONTROLLER_PARTIALS;
}

// a whole loop controller in two launches, `z` of inkling_controller_scratch's size, out in x's dtype
extern "C" int inkling_controller(const void* x, int x_dtype, const void* message, int m_dtype, const float* down,
                                  const float* gain_forget, float* z, void* out, int64_t rows, int hidden, int rank,
                                  float eps, int64_t stream) {
  if (inkling_controller_scratch(rows, rank) == 0) return -1;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const dim3 down_grid((unsigned)rank, CONTROLLER_DOWN_CHUNKS);
  constexpr int up_per_block = CONTROLLER_THREADS / WARP;
  const unsigned up_grid = (unsigned)((hidden + up_per_block - 1) / up_per_block);
#define CTRL(X, M)                                                                                              \
  controller_down_kernel<<<down_grid, CONTROLLER_THREADS, 0, s>>>((const X*)x, (const M*)message, down, z,    \
                                                                  rows, hidden, rank);                        \
  controller_up_kernel<<<up_grid, CONTROLLER_THREADS, 0, s>>>((const X*)x, (const M*)message, gain_forget, z, \
                                                              (X*)out, rows, hidden, rank, eps)
  PAIR(x_dtype, m_dtype, CTRL)
#undef CTRL
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

// the f32 + i32 partials a Markov pick over `b` rows takes
extern "C" int64_t inkling_markov_pick_scratch(int b) { return int64_t(b) * PICK_BLOCKS; }

// each batch row's argmax over `v < vocab` of `logits[v] + fold_b[v] + fold_w[v] . latent` (fold_w bf16
// `[>= vocab, rank]`, rank a multiple of 8 and its rows 16-byte aligned), `part_v`/`part_i` of the scratch's size
extern "C" int inkling_markov_pick(const void* logits, int logits_dtype, int64_t logits_batch, const void* fold_w,
                                   const float* fold_b, const float* latent, int b, int rank, int vocab,
                                   float* part_v, int* part_i, uint32_t* out, int64_t stream) {
  if (rank % 8 != 0) return -1;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const dim3 grid(PICK_BLOCKS, b);
  const size_t smem = size_t(rank) * sizeof(float);
  if (logits_dtype == 0) {
    markov_pick_partial_kernel<<<grid, PICK_THREADS, smem, s>>>((const float*)logits, logits_batch,
                                                                (const __nv_bfloat16*)fold_w, fold_b, latent, rank,
                                                                vocab, part_v, part_i);
  } else {
    markov_pick_partial_kernel<<<grid, PICK_THREADS, smem, s>>>((const __nv_bfloat16*)logits, logits_batch,
                                                                (const __nv_bfloat16*)fold_w, fold_b, latent, rank,
                                                                vocab, part_v, part_i);
  }
  markov_pick_final_kernel<<<b, PICK_THREADS, 0, s>>>(part_v, part_i, PICK_BLOCKS, out);
  return (int)cudaGetLastError();
}

// the widest state the pool kernel keeps (inkling_short_conv_pool refuses wider)
extern "C" int inkling_short_conv_pool_max_hist() { return CONV_MAX_HIST; }

extern "C" int inkling_short_conv_pool(const void* x, int x_dtype, float* pool, const uint32_t* slots, int pool_rows,
                                       int pool_c, int channel_off, const float* weight, const float* taps, int heads,
                                       void* out, int out_like_x, float* before, void* stash, int b, int t, int c,
                                       int width, int64_t stream) {
  if (width - 1 > CONV_MAX_HIST || width < 2) return -1;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t grid = blocks_for(int64_t(b) * c);
#define POOL_OUT(X, O)                                                                                        \
  short_conv_pool_kernel<<<grid, ROW_THREADS, 0, s>>>((const X*)x, pool, slots, pool_rows, pool_c, channel_off, \
                                                      weight, taps, heads, (O*)out, before, (X*)stash, b, t, c,   \
                                                      width)
  if (x_dtype == 0) {
    POOL_OUT(float, float);
  } else if (out_like_x) {
    POOL_OUT(__nv_bfloat16, __nv_bfloat16);
  } else {
    POOL_OUT(__nv_bfloat16, float);
  }
#undef POOL_OUT
  return (int)cudaGetLastError();
}

// inputs_dtype 0 f32, 1 bf16
extern "C" int inkling_conv_rollback(const void* slots, const int32_t* keep, int n_slots, int b, int t, int c,
                                     int hist, int inputs_dtype, int64_t stream) {
  const dim3 grid(blocks_for(int64_t(b) * c * hist), n_slots);
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  if (inputs_dtype == 0) {
    conv_rollback_kernel<float><<<grid, ROW_THREADS, 0, s>>>((const ConvRollbackSlot*)slots, keep, b, t, c, hist);
  } else {
    conv_rollback_kernel<__nv_bfloat16>
        <<<grid, ROW_THREADS, 0, s>>>((const ConvRollbackSlot*)slots, keep, b, t, c, hist);
  }
  return (int)cudaGetLastError();
}
