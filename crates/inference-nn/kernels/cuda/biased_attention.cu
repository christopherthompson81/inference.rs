// Causal attention with a relative bias by distance and a sink (Inkling 2): each (batch, query, head) row of `table`
// holds the bias at distances 0..extent (none past them), `sinks` a logit of its own with a zero value per (batch,
// query, head); q is (batch, t, heads, head_dim) f32. Query i of batch b sits at position past_b + i, past_b from a
// device array when given (a paged cache's lengths, so a captured graph reads them) or one value; keys `window` or
// more back are hidden (0 for none). With a log scale (alpha > 0) each query's score before the softmax, the sink's
// aside, is scaled by 1 + alpha ln(max(1, (pos + 1) / floor)). K and V come from a contiguous view (rows contiguous,
// heads and batches strided) or a paged cache [blocks, kv_heads, block_size, head_dim] through block tables, in f32,
// f16 or bf16. The output is (batch, t, heads, head_dim) f32.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

__device__ __forceinline__ float load(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float load(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }

// where a (batch, kv head, key)'s row of K or V sits
template <typename T>
struct Kv {
  const T* base;
  // contiguous: the batch and head strides
  int64_t batch_stride;
  int64_t head_stride;
  // paged (when tables is set): the block table rows and the cache's shape
  const uint32_t* tables;
  int64_t table_stride;
  int table_width;
  int block_size;
  int kv_heads;

  __device__ __forceinline__ const T* row(int b, int h, int key, int hd) const {
    if (tables) {
      const int slot = min(key / block_size, table_width - 1);
      const int64_t block = tables[int64_t(b) * table_stride + slot];
      return base + ((block * kv_heads + h) * block_size + key % block_size) * hd;
    }
    return base + b * batch_stride + h * head_stride + int64_t(key) * hd;
  }
};

template <typename T>
struct Args {
  const float* q;
  Kv<T> k;
  Kv<T> v;
  const float* table;
  const float* sinks;
  float* out;
  int heads;
  int kv_heads;
  int t;
  // past_b = past_lens[b] - past_sub when past_lens is set, else past
  const uint32_t* past_lens;
  int past_sub;
  int past;
  int extent;
  int window;
  float scale;
  float log_alpha;
  float log_floor;
};

template <typename T>
__device__ __forceinline__ int past_of(const Args<T>& a, int b) {
  return a.past_lens ? int(a.past_lens[b]) - a.past_sub : a.past;
}

template <typename T>
__device__ __forceinline__ float tau_of(const Args<T>& a, int pos) {
  return a.log_alpha > 0.0f ? 1.0f + a.log_alpha * logf(fmaxf(1.0f, (pos + 1) / a.log_floor)) : 1.0f;
}

__device__ __forceinline__ float warp_max(float x) {
  for (int o = WARP / 2; o > 0; o /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
  return x;
}

__device__ __forceinline__ float warp_sum(float x) {
  for (int o = WARP / 2; o > 0; o /= 2) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

__device__ __forceinline__ int first_key(int window, int pos) {
  return window > 0 && pos + 1 > window ? pos + 1 - window : 0;
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
template <int R, typename T>
__global__ void prefill_kernel(Args<T> a) {
  constexpr int HD = R * WARP;
  __shared__ float kt[KEY_TILE][HD + 1];
  __shared__ float vt[KEY_TILE][HD];
  __shared__ float qs[PREFILL_WARPS][HD];
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kvh = head / (a.heads / a.kv_heads);
  const int past = past_of(a, batch);
  const int keys = past + a.t;
  const int i0 = blockIdx.x * PREFILL_WARPS;
  const int i = i0 + warp;
  const bool valid = i < a.t;
  const int pos = past + i;
  const float tau = tau_of(a, pos);
  const int64_t row = (int64_t(batch) * a.t + i) * a.heads + head;
  if (valid) {
#pragma unroll
    for (int r = 0; r < R; r++) qs[warp][lane + r * WARP] = a.q[row * HD + lane + r * WARP] * a.scale;
  }
  const int last = min(i0 + PREFILL_WARPS, a.t) - 1;
  const int lo = first_key(a.window, past + i0);
  const int hi = past + last;
  const float* bias = a.table + row * (a.extent + 1);
  float m = valid ? a.sinks[row] : 0.0f;
  float l = 1.0f;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.0f;
  const int first = valid ? first_key(a.window, pos) : 0;
  for (int base = lo; base <= hi; base += KEY_TILE) {
    __syncthreads();
    for (int e = threadIdx.x; e < KEY_TILE * HD; e += blockDim.x) {
      const int j = e / HD;
      const int d = e % HD;
      const int key = base + j;
      const bool in = key < keys;
      kt[j][d] = in ? load(a.k.row(batch, kvh, key, HD), d) : 0.0f;
      vt[j][d] = in ? load(a.v.row(batch, kvh, key, HD), d) : 0.0f;
    }
    __syncthreads();
    if (!valid) continue;
    const int key = base + lane;
    const bool visible = key <= pos && key >= first && key < keys;
    float s = 0.0f;
#pragma unroll 8
    for (int d = 0; d < HD; d++) s += qs[warp][d] * kt[lane][d];
    const int dist = pos - key;
    s += dist < a.extent && dist >= 0 ? bias[dist] : 0.0f;
    fold<R>(s * tau, visible, m, l, acc, [&](int j) { return &vt[j][0]; });
  }
  if (!valid) return;
  float* o = a.out + ((int64_t(batch) * a.t + i) * a.heads + head) * HD;
  const float inv = 1.0f / l;
#pragma unroll
  for (int r = 0; r < R; r++) o[lane + r * WARP] = acc[r] * inv;
}

// a block a query of one head, its warps splitting the keys, a lane R of the head's dims (so a key's K and V rows load
// coalesced), DECODE_KEYS keys a step to keep loads in flight; the warps merged in shared memory at the end
template <int R, typename T>
__global__ void decode_kernel(Args<T> a) {
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
  const int pos = past_of(a, batch) + i;
  const float tau = tau_of(a, pos);
  const int64_t row = (int64_t(batch) * a.t + i) * a.heads + head;
  float qr[R];
#pragma unroll
  for (int r = 0; r < R; r++) qr[r] = a.q[row * HD + lane * R + r] * a.scale;
  const float* bias = a.table + row * (a.extent + 1);
  // the sink starts the first warp's softmax: its weight and no value
  float m = warp == 0 ? a.sinks[row] : -INFINITY;
  float l = warp == 0 ? 1.0f : 0.0f;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.0f;
  const int first = first_key(a.window, pos);
  for (int base = first + warp * DECODE_KEYS; base <= pos; base += DECODE_WARPS * DECODE_KEYS) {
    float s[DECODE_KEYS];
    float vr[DECODE_KEYS][R];
#pragma unroll
    for (int j = 0; j < DECODE_KEYS; j++) {
      const int key = min(base + j, pos);
      const T* kr = a.k.row(batch, kvh, key, HD) + lane * R;
      const T* vrow = a.v.row(batch, kvh, key, HD) + lane * R;
      float part = 0.0f;
#pragma unroll
      for (int r = 0; r < R; r++) {
        part += qr[r] * load(kr, r);
        vr[j][r] = load(vrow, r);
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
      const float score = (s[j] + (dist < a.extent ? bias[dist] : 0.0f)) * tau;
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

template <int R, typename T>
cudaError_t launch(const Args<T>& a, int batch, bool decode, cudaStream_t stream) {
  if (decode) {
    decode_kernel<R, T><<<dim3(a.t, a.heads, batch), DECODE_WARPS * WARP, 0, stream>>>(a);
  } else {
    const int tiles = (a.t + PREFILL_WARPS - 1) / PREFILL_WARPS;
    prefill_kernel<R, T><<<dim3(tiles, a.heads, batch), PREFILL_WARPS * WARP, 0, stream>>>(a);
  }
  return cudaGetLastError();
}

template <typename T>
int dispatch(const Args<T>& a, int batch, int head_dim, bool decode, cudaStream_t s) {
  cudaError_t err;
  switch (head_dim) {
    case 32: err = launch<1>(a, batch, decode, s); break;
    case 64: err = launch<2>(a, batch, decode, s); break;
    case 128: err = launch<4>(a, batch, decode, s); break;
    default: return -1;
  }
  return static_cast<int>(err);
}

}  // namespace

// the shapes and positions every launch takes
struct BiasedAttentionShape {
  int batch;
  int heads;
  int kv_heads;
  int t;
  int head_dim;
  int extent;
  int window;
  float scale;
  float log_alpha;
  float log_floor;
  // past_b = past_lens[b] - past_sub when past_lens is set, else past
  const uint32_t* past_lens;
  int past_sub;
  int past;
  // queries up to this many take the key-splitting kernel
  int decode_rows;
};

// K/V from a contiguous f32 view
extern "C" int biased_attention_f32(const float* q, const float* k, const float* v, const float* table,
                                    const float* sinks, float* out, const BiasedAttentionShape* shape,
                                    int64_t k_batch, int64_t k_head, int64_t v_batch, int64_t v_head,
                                    int64_t stream) {
  const BiasedAttentionShape& p = *shape;
  Args<float> a{q,
                {k, k_batch, k_head, nullptr, 0, 0, 0, 0},
                {v, v_batch, v_head, nullptr, 0, 0, 0, 0},
                table, sinks, out, p.heads, p.kv_heads, p.t, p.past_lens, p.past_sub, p.past, p.extent, p.window,
                p.scale, p.log_alpha, p.log_floor};
  return dispatch(a, p.batch, p.head_dim, p.t <= p.decode_rows, reinterpret_cast<cudaStream_t>(stream));
}

// K/V from a paged cache [blocks, kv_heads, block_size, head_dim] through block tables (batch, table_width) u32;
// cache_dtype 0 f32, 1 f16, 2 bf16
extern "C" int biased_attention_paged(const float* q, const void* k_cache, const void* v_cache, int cache_dtype,
                                      const uint32_t* tables, int64_t table_stride, int table_width, int block_size,
                                      const float* table, const float* sinks, float* out,
                                      const BiasedAttentionShape* shape, int64_t stream) {
  const BiasedAttentionShape& p = *shape;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const bool decode = p.t <= p.decode_rows;
#define PAGED(T)                                                                                                   \
  {                                                                                                                \
    Args<T> a{q,                                                                                                   \
              {(const T*)k_cache, 0, 0, tables, table_stride, table_width, block_size, p.kv_heads},                \
              {(const T*)v_cache, 0, 0, tables, table_stride, table_width, block_size, p.kv_heads},                \
              table, sinks, out, p.heads, p.kv_heads, p.t, p.past_lens, p.past_sub, p.past, p.extent, p.window,    \
              p.scale, p.log_alpha, p.log_floor};                                                                  \
    return dispatch(a, p.batch, p.head_dim, decode, s);                                                            \
  }
  switch (cache_dtype) {
    case 0: PAGED(float)
    case 1: PAGED(__half)
    case 2: PAGED(__nv_bfloat16)
    default: return -1;
  }
#undef PAGED
}
