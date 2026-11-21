// Causal attention with a relative bias by distance and a sink (Inkling 2): each (batch, query, head) row of `table`
// holds the bias at distances 0..extent (none past them), `sinks` a logit of its own with a zero value per (batch,
// query, head); q is (batch, t, heads, head_dim). Keys `window` or more back are hidden (0 for none); query i sits at position past + i. f32 throughout;
// K and V rows are contiguous, their heads and batches strided (a cache's narrowed view). The output is (batch, t,
// heads, head_dim).
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int WARP = 32;
constexpr int PREFILL_WARPS = 8;
constexpr int DECODE_WARPS = 32;
constexpr int KEY_TILE = WARP;
// keys a decode warp loads before it folds them
constexpr int DECODE_KEYS = 4;

struct Args {
  const float* q;
  const float* k;
  const float* v;
  const float* table;
  const float* sinks;
  float* out;
  int heads;
  int kv_heads;
  int t;
  int keys;
  int past;
  int extent;
  int window;
  float scale;
  int64_t k_batch;
  int64_t k_head;
  int64_t v_batch;
  int64_t v_head;
};

__device__ __forceinline__ float warp_max(float x) {
  for (int o = WARP / 2; o > 0; o /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
  return x;
}

__device__ __forceinline__ float warp_sum(float x) {
  for (int o = WARP / 2; o > 0; o /= 2) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

__device__ __forceinline__ int first_key(const Args& a, int pos) {
  return a.window > 0 && pos + 1 > a.window ? pos + 1 - a.window : 0;
}

// one key per lane: its score folded into the running (m, l, acc) of a warp's query, the values from `vrow(j)`
template <int R, typename VRow>
__device__ __forceinline__ void fold(float s, bool visible, float& m, float& l, float (&acc)[R], VRow vrow) {
  const int lane = threadIdx.x % WARP;
  const float m_new = fmaxf(m, warp_max(visible ? s : -INFINITY));
  const float alpha = __expf(m - m_new);
  const float p = visible ? __expf(s - m_new) : 0.0f;
  l = l * alpha + warp_sum(p);
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] *= alpha;
  for (int j = 0; j < WARP; j++) {
    const float pj = __shfl_sync(0xffffffffu, p, j);
    if (pj == 0.0f) continue;
    const float* row = vrow(j);
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] += pj * row[lane + r * WARP];
  }
  m = m_new;
}

// a block a tile of queries of one head, a warp a query, the keys through shared memory a tile at a time
template <int R>
__global__ void prefill_kernel(Args a) {
  constexpr int HD = R * WARP;
  __shared__ float kt[KEY_TILE][HD + 1];
  __shared__ float vt[KEY_TILE][HD];
  __shared__ float qs[PREFILL_WARPS][HD];
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kvh = head / (a.heads / a.kv_heads);
  const int i0 = blockIdx.x * PREFILL_WARPS;
  const int i = i0 + warp;
  const bool valid = i < a.t;
  const int pos = a.past + i;
  const int64_t row = (int64_t(batch) * a.t + i) * a.heads + head;
  if (valid) {
#pragma unroll
    for (int r = 0; r < R; r++) qs[warp][lane + r * WARP] = a.q[row * HD + lane + r * WARP] * a.scale;
  }
  const int last = min(i0 + PREFILL_WARPS, a.t) - 1;
  const int lo = first_key(a, a.past + i0);
  const int hi = a.past + last;
  const float* kb = a.k + batch * a.k_batch + kvh * a.k_head;
  const float* vb = a.v + batch * a.v_batch + kvh * a.v_head;
  const float* bias = a.table + row * (a.extent + 1);
  float m = valid ? a.sinks[row] : 0.0f;
  float l = 1.0f;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.0f;
  const int first = valid ? first_key(a, pos) : 0;
  for (int base = lo; base <= hi; base += KEY_TILE) {
    __syncthreads();
    for (int e = threadIdx.x; e < KEY_TILE * HD; e += blockDim.x) {
      const int j = e / HD;
      const int d = e % HD;
      const int key = base + j;
      const bool in = key < a.keys;
      kt[j][d] = in ? kb[int64_t(key) * HD + d] : 0.0f;
      vt[j][d] = in ? vb[int64_t(key) * HD + d] : 0.0f;
    }
    __syncthreads();
    if (!valid) continue;
    const int key = base + lane;
    const bool visible = key <= pos && key >= first && key < a.keys;
    float s = 0.0f;
#pragma unroll 8
    for (int d = 0; d < HD; d++) s += qs[warp][d] * kt[lane][d];
    const int dist = pos - key;
    s += dist < a.extent && dist >= 0 ? bias[dist] : 0.0f;
    fold<R>(s, visible, m, l, acc, [&](int j) { return &vt[j][0]; });
  }
  if (!valid) return;
  float* o = a.out + ((int64_t(batch) * a.t + i) * a.heads + head) * HD;
  const float inv = 1.0f / l;
#pragma unroll
  for (int r = 0; r < R; r++) o[lane + r * WARP] = acc[r] * inv;
}

// a block a query of one head, its warps splitting the keys, a lane R of the head's dims (so a key's K and V rows load
// coalesced), DECODE_KEYS keys a step to keep loads in flight; the warps merged in shared memory at the end
template <int R>
__global__ void decode_kernel(Args a) {
  constexpr int HD = R * WARP;
  __shared__ float sm[DECODE_WARPS];
  __shared__ float sl[DECODE_WARPS];
  __shared__ float sacc[DECODE_WARPS][HD];
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int i = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kvh = head / (a.heads / a.kv_heads);
  const int pos = a.past + i;
  const int64_t row = (int64_t(batch) * a.t + i) * a.heads + head;
  float qr[R];
#pragma unroll
  for (int r = 0; r < R; r++) qr[r] = a.q[row * HD + lane * R + r] * a.scale;
  const float* kb = a.k + batch * a.k_batch + kvh * a.k_head + lane * R;
  const float* vb = a.v + batch * a.v_batch + kvh * a.v_head + lane * R;
  const float* bias = a.table + row * (a.extent + 1);
  // the sink starts the first warp's softmax: its weight and no value
  float m = warp == 0 ? a.sinks[row] : -INFINITY;
  float l = warp == 0 ? 1.0f : 0.0f;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.0f;
  const int first = first_key(a, pos);
  for (int base = first + warp * DECODE_KEYS; base <= pos; base += DECODE_WARPS * DECODE_KEYS) {
    float s[DECODE_KEYS];
    float vr[DECODE_KEYS][R];
#pragma unroll
    for (int j = 0; j < DECODE_KEYS; j++) {
      const int key = min(base + j, pos);
      float part = 0.0f;
#pragma unroll
      for (int r = 0; r < R; r++) {
        part += qr[r] * kb[int64_t(key) * HD + r];
        vr[j][r] = vb[int64_t(key) * HD + r];
      }
      s[j] = part;
    }
#pragma unroll
    for (int j = 0; j < DECODE_KEYS; j++) s[j] = warp_sum(s[j]);
#pragma unroll
    for (int j = 0; j < DECODE_KEYS; j++) {
      const int key = base + j;
      if (key > pos) break;
      const int dist = pos - key;
      const float score = s[j] + (dist < a.extent ? bias[dist] : 0.0f);
      const float m_new = fmaxf(m, score);
      const float alpha = __expf(m - m_new);
      const float p = __expf(score - m_new);
      l = l * alpha + p;
#pragma unroll
      for (int r = 0; r < R; r++) acc[r] = acc[r] * alpha + p * vr[j][r];
      m = m_new;
    }
  }
  if (lane == 0) {
    sm[warp] = m;
    sl[warp] = l;
  }
#pragma unroll
  for (int r = 0; r < R; r++) sacc[warp][lane * R + r] = acc[r];
  __syncthreads();
  if (warp != 0) return;
  float mt = -INFINITY;
  for (int w = 0; w < DECODE_WARPS; w++) mt = fmaxf(mt, sm[w]);
  float lt = 0.0f;
  float out[R];
#pragma unroll
  for (int r = 0; r < R; r++) out[r] = 0.0f;
  for (int w = 0; w < DECODE_WARPS; w++) {
    if (sm[w] == -INFINITY) continue;
    const float wt = __expf(sm[w] - mt);
    lt += sl[w] * wt;
#pragma unroll
    for (int r = 0; r < R; r++) out[r] += sacc[w][lane * R + r] * wt;
  }
  float* o = a.out + ((int64_t(batch) * a.t + i) * a.heads + head) * HD;
  const float inv = 1.0f / lt;
#pragma unroll
  for (int r = 0; r < R; r++) o[lane * R + r] = out[r] * inv;
}

template <int R>
cudaError_t launch(const Args& a, int batch, bool decode, cudaStream_t stream) {
  if (decode) {
    decode_kernel<R><<<dim3(a.t, a.heads, batch), DECODE_WARPS * WARP, 0, stream>>>(a);
  } else {
    const int tiles = (a.t + PREFILL_WARPS - 1) / PREFILL_WARPS;
    prefill_kernel<R><<<dim3(tiles, a.heads, batch), PREFILL_WARPS * WARP, 0, stream>>>(a);
  }
  return cudaGetLastError();
}

}  // namespace

// queries up to `decode_rows` take the key-splitting kernel; head_dim a multiple of 32 up to 128
extern "C" int biased_attention_f32(const float* q, const float* k, const float* v, const float* table,
                                    const float* sinks, float* out, int batch, int heads, int kv_heads, int t,
                                    int keys, int past, int head_dim, int extent, int window, float scale,
                                    int64_t k_batch, int64_t k_head, int64_t v_batch, int64_t v_head,
                                    int decode_rows, int64_t stream) {
  Args a{q, k, v, table, sinks, out, heads, kv_heads, t, keys, past, extent, window, scale,
         k_batch, k_head, v_batch, v_head};
  const bool decode = t <= decode_rows;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t err;
  switch (head_dim) {
    case 32: err = launch<1>(a, batch, decode, s); break;
    case 64: err = launch<2>(a, batch, decode, s); break;
    case 128: err = launch<4>(a, batch, decode, s); break;
    default: return -1;
  }
  return static_cast<int>(err);
}
