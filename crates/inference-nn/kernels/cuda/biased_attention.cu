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

#include <algorithm>
#include <type_traits>

namespace {

constexpr int WARP = 32;
constexpr int PREFILL_WARPS = 8;
constexpr int DECODE_WARPS = 8;
constexpr int KEY_TILE = WARP;
// a decode warp's lane groups: KEY_LANES lanes a key, each lane HD / KEY_LANES of its dims, DECODE_KEYS keys a group
// loaded before it folds them
constexpr int KEY_LANES = 8;
constexpr int KEY_GROUPS = WARP / KEY_LANES;
constexpr int DECODE_KEYS = 2;
// a decode's keys split over up to this many blocks a query, enough for 2 blocks an SM; a row takes a split a
// DECODE_SPLIT_KEYS of its keys, the grid's other blocks exiting
constexpr int DECODE_MAX_SPLITS = 16;
constexpr int DECODE_BLOCKS_PER_SM = 6;
constexpr int DECODE_SPLIT_KEYS = 64;
// the tensor-core prefill: a warp 16 queries, a block 4 warps, keys a tile at a time; prompts from this many queries
constexpr int MMA_WARPS = 4;
constexpr int MMA_ROWS = 16;
constexpr int MMA_KEYS = 64;
constexpr int MMA_MIN_QUERIES = 32;
// shared rows padded by 8 halves against bank conflicts
constexpr int MMA_PAD = 8;

__device__ __forceinline__ float load(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float load(const __half* p, int64_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }

// an E4M3 cache byte (the writer's scale 1): its bits placed as f16 are 2^-8 of the value (bias 7 against 15), exact
// for every byte, subnormals too; sm86 has no fp8 conversion
struct E4m3 {
  uint8_t bits;
};
constexpr float E4M3_BIAS_GAP = 256.0f;
__device__ __forceinline__ float load(const E4m3* p, int64_t i) {
  const unsigned b = p[i].bits;
  return __half2float(__ushort_as_half((unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7)))) * E4M3_BIAS_GAP;
}

// a lane's R values of a row in one load where the type packs them (rows are R-aligned: lane * R of a head dim)
template <int R, typename T>
__device__ __forceinline__ void load_row(const T* p, float (&o)[R]) {
#pragma unroll
  for (int r = 0; r < R; r++) o[r] = load(p, r);
}

template <>
__device__ __forceinline__ void load_row<4, __nv_bfloat16>(const __nv_bfloat16* p, float (&o)[4]) {
  const uint2 raw = *reinterpret_cast<const uint2*>(p);
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
  o[0] = a.x, o[1] = a.y, o[2] = b.x, o[3] = b.y;
}

template <>
__device__ __forceinline__ void load_row<4, E4m3>(const E4m3* p, float (&o)[4]) {
  const uint32_t raw = *reinterpret_cast<const uint32_t*>(p);
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const unsigned b = (raw >> (8 * r)) & 0xFFu;
    o[r] = __half2float(__ushort_as_half((unsigned short)(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7)))) * E4M3_BIAS_GAP;
  }
}

// a q8 cache's row (paged-attn's kv_quant at 8 bits): byte i is value i, (q + 0.5 - 128) / 128 times its group's f16
// scale, the scales after the HD data bytes, one a 32 values; the values Hadamard-rotated a 32 at a time, which the
// caller undoes on q and the output
struct Q8 {
  uint8_t bits;
};
constexpr int Q8_GROUP = 32;
constexpr float Q8_MID = 128.0f;

// a row's length in its elements: a q8 row carries its scales after its data
template <typename T>
__device__ __forceinline__ constexpr int row_len(int hd) {
  return hd;
}
template <>
__device__ __forceinline__ constexpr int row_len<Q8>(int hd) {
  return hd + hd / Q8_GROUP * 2;
}

// value i of a row of HD values
template <int HD, typename T>
__device__ __forceinline__ float load_at(const T* row, int64_t i) {
  return load(row, i);
}
template <int HD>
__device__ __forceinline__ float load_at(const Q8* row, int64_t i) {
  const float scale = __half2float(reinterpret_cast<const __half*>(row + HD)[i / Q8_GROUP]);
  return (float(row[i].bits) + 0.5f - Q8_MID) * (scale / Q8_MID);
}

// a lane's N values (N a multiple of 4), 4 a load
template <int N, typename T>
__device__ __forceinline__ void load_span(const T* p, float (&o)[N]) {
#pragma unroll
  for (int c = 0; c < N / 4; c++) load_row<4>(p + 4 * c, *reinterpret_cast<float(*)[4]>(o + 4 * c));
}

// a lane's N values from `off` of a row of HD values (N a multiple of 4, within one q8 group)
template <int HD, int N, typename T>
__device__ __forceinline__ void load_span_at(const T* row, int off, float (&o)[N]) {
  load_span<N>(row + off, o);
}
template <int HD, int N>
__device__ __forceinline__ void load_span_at(const Q8* row, int off, float (&o)[N]) {
  const float scale = __half2float(reinterpret_cast<const __half*>(row + HD)[off / Q8_GROUP]) / Q8_MID;
  // rows are 4-byte aligned (HD + HD / 16 bytes)
#pragma unroll
  for (int c = 0; c < N / 4; c++) {
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(row + off + 4 * c);
#pragma unroll
    for (int r = 0; r < 4; r++) o[4 * c + r] = (float((raw >> (8 * r)) & 0xFFu) + 0.5f - Q8_MID) * scale;
  }
}

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
  // a ring of this many blocks a sequence (its table row holds the ring's first block): key block k sits at block
  // k % ring, which only a window shorter than the ring reads back
  int ring;

  __device__ __forceinline__ const T* row(int b, int h, int key, int hd) const {
    if (tables) {
      const int64_t block =
          ring > 0 ? int64_t(tables[int64_t(b) * table_stride]) + (key / block_size) % ring
                   : tables[int64_t(b) * table_stride + min(key / block_size, table_width - 1)];
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
  // a split decode's partials, `(rows, splits, head_dim + 2)`: each split's max, sum and unnormalized values, and
  // each row's count of finished splits (zero between launches: the last split resets it)
  float* partials;
  unsigned int* arrived;
  int splits;
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
      kt[j][d] = in ? load_at<HD>(a.k.row(batch, kvh, key, row_len<T>(HD)), d) : 0.0f;
      vt[j][d] = in ? load_at<HD>(a.v.row(batch, kvh, key, row_len<T>(HD)), d) : 0.0f;
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

// a = A (16x16 f16, row-major) fragment regs, b = B (16x8, k-major) regs, c = the f32 accumulators
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// a pair of f32 values as f16 and its remainder in f16: the two together carry ~21 bits
__device__ __forceinline__ void split_half2(float x, float y, uint32_t& hi, uint32_t& lo) {
  const __half2 h = __floats2half2_rn(x, y);
  const float2 back = __half22float2(h);
  hi = *reinterpret_cast<const uint32_t*>(&h);
  lo = pack_half2(x - back.x, y - back.y);
}

// prefill on tensor cores (sm80+): a block 64 queries of one head, a warp 16 of them; each key tile's K and V^T in
// shared memory as f16 (a BF16 cache's values exactly), S = Q K^T and O += P V by mma with f32 sums, q and the
// probabilities each as an f16 part and its f16 remainder (two mma) so the products keep ~f32 precision; the scores take the scalar kernel's bias, log
// scale, causal and window masks and sink, softmax online. Fragment layout (m16n8k16): lane g = lane / 4 holds rows g
// and g + 8, t = lane % 4 the column pairs 2t and 2t + 8 of A and 2t of C.
template <int R, typename T>
__global__ void __launch_bounds__(MMA_WARPS * WARP) prefill_mma_kernel(Args<T> a) {
#if __CUDA_ARCH__ >= 800
  constexpr int HD = R * WARP;
  constexpr int KSTEPS = HD / 16;
  constexpr int SN = MMA_KEYS / 8;
  constexpr int ON = HD / 8;
  __shared__ __half ks[MMA_KEYS][HD + MMA_PAD];
  __shared__ __half vt[HD][MMA_KEYS + MMA_PAD];
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int g = lane / 4;
  const int t4 = lane % 4;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kvh = head / (a.heads / a.kv_heads);
  const int past = past_of(a, batch);
  const int keys = past + a.t;
  const int q0 = blockIdx.x * MMA_WARPS * MMA_ROWS;
  // this lane's two query rows
  int qi[2], pos[2], first[2];
  bool valid[2];
  float tau[2];
  const float* bias[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    qi[h] = q0 + warp * MMA_ROWS + g + 8 * h;
    valid[h] = qi[h] < a.t;
    pos[h] = past + qi[h];
    first[h] = first_key(a.window, pos[h]);
    tau[h] = tau_of(a, pos[h]);
    bias[h] = a.table + ((int64_t(batch) * a.t + min(qi[h], a.t - 1)) * a.heads + head) * (a.extent + 1);
  }
  auto q_at = [&](int h, int d) -> float {
    return valid[h] ? a.q[((int64_t(batch) * a.t + qi[h]) * a.heads + head) * HD + d] * a.scale : 0.0f;
  };
  uint32_t qf[KSTEPS][4], qr[KSTEPS][4];
#pragma unroll
  for (int kk = 0; kk < KSTEPS; kk++) {
    const int d = kk * 16 + 2 * t4;
    split_half2(q_at(0, d), q_at(0, d + 1), qf[kk][0], qr[kk][0]);
    split_half2(q_at(1, d), q_at(1, d + 1), qf[kk][1], qr[kk][1]);
    split_half2(q_at(0, d + 8), q_at(0, d + 9), qf[kk][2], qr[kk][2]);
    split_half2(q_at(1, d + 8), q_at(1, d + 9), qf[kk][3], qr[kk][3]);
  }
  float m[2], l[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    m[h] = valid[h] ? a.sinks[(int64_t(batch) * a.t + qi[h]) * a.heads + head] : 0.0f;
    l[h] = 1.0f;
  }
  float o[ON][4];
#pragma unroll
  for (int n = 0; n < ON; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  const int last = min(q0 + MMA_WARPS * MMA_ROWS, a.t) - 1;
  const int lo = first_key(a.window, past + q0) / MMA_KEYS * MMA_KEYS;
  const int hi = past + last;
  for (int base = lo; base <= hi; base += MMA_KEYS) {
    __syncthreads();
    for (int e = threadIdx.x; e < MMA_KEYS * HD; e += blockDim.x) {
      const int j = e / HD;
      const int d = e % HD;
      const int key = base + j;
      const bool in = key < keys;
      ks[j][d] = __float2half(in ? load_at<HD>(a.k.row(batch, kvh, key, row_len<T>(HD)), d) : 0.0f);
      vt[d][j] = __float2half(in ? load_at<HD>(a.v.row(batch, kvh, key, row_len<T>(HD)), d) : 0.0f);
    }
    __syncthreads();
    float sc[SN][4];
#pragma unroll
    for (int n = 0; n < SN; n++) {
      sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.0f;
#pragma unroll
      for (int kk = 0; kk < KSTEPS; kk++) {
        const __half* krow = &ks[n * 8 + g][kk * 16 + 2 * t4];
        const uint32_t k0 = *reinterpret_cast<const uint32_t*>(krow);
        const uint32_t k1 = *reinterpret_cast<const uint32_t*>(krow + 8);
        mma_f16(sc[n], qf[kk], k0, k1);
        mma_f16(sc[n], qr[kk], k0, k1);
      }
    }
    // the scores: bias by distance, the log scale, the masks; each row's max over the tile
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < SN; n++) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const int h = c / 2;
        const int key = base + n * 8 + 2 * t4 + c % 2;
        const int dist = pos[h] - key;
        const bool visible = valid[h] && key <= pos[h] && key >= first[h] && key < keys;
        const float b = dist >= 0 && dist < a.extent ? bias[h][dist] : 0.0f;
        sc[n][c] = visible ? (sc[n][c] + b) * tau[h] : -INFINITY;
        tile_max[h] = fmaxf(tile_max[h], sc[n][c]);
      }
    }
    float alpha[2], rowsum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int h = 0; h < 2; h++) {
      tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffu, tile_max[h], 1));
      tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffu, tile_max[h], 2));
      const float m_new = fmaxf(m[h], tile_max[h]);
      alpha[h] = __expf(m[h] - m_new);
      m[h] = m_new;
    }
#pragma unroll
    for (int n = 0; n < SN; n++) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const int h = c / 2;
        sc[n][c] = sc[n][c] == -INFINITY ? 0.0f : __expf(sc[n][c] - m[h]);
        rowsum[h] += sc[n][c];
      }
    }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      rowsum[h] += __shfl_xor_sync(0xffffffffu, rowsum[h], 1);
      rowsum[h] += __shfl_xor_sync(0xffffffffu, rowsum[h], 2);
      l[h] = l[h] * alpha[h] + rowsum[h];
    }
#pragma unroll
    for (int n = 0; n < ON; n++) {
      o[n][0] *= alpha[0];
      o[n][1] *= alpha[0];
      o[n][2] *= alpha[1];
      o[n][3] *= alpha[1];
    }
    // O += P V: two score tiles (16 keys) are an A fragment
#pragma unroll
    for (int kk = 0; kk < MMA_KEYS / 16; kk++) {
      uint32_t pf[4], pr[4];
      split_half2(sc[2 * kk][0], sc[2 * kk][1], pf[0], pr[0]);
      split_half2(sc[2 * kk][2], sc[2 * kk][3], pf[1], pr[1]);
      split_half2(sc[2 * kk + 1][0], sc[2 * kk + 1][1], pf[2], pr[2]);
      split_half2(sc[2 * kk + 1][2], sc[2 * kk + 1][3], pf[3], pr[3]);
#pragma unroll
      for (int n = 0; n < ON; n++) {
        const __half* vrow = &vt[n * 8 + g][kk * 16 + 2 * t4];
        const uint32_t v0 = *reinterpret_cast<const uint32_t*>(vrow);
        const uint32_t v1 = *reinterpret_cast<const uint32_t*>(vrow + 8);
        mma_f16(o[n], pf, v0, v1);
        mma_f16(o[n], pr, v0, v1);
      }
    }
  }
#pragma unroll
  for (int h = 0; h < 2; h++) {
    if (!valid[h]) continue;
    float* out = a.out + ((int64_t(batch) * a.t + qi[h]) * a.heads + head) * HD;
    const float inv = 1.0f / l[h];
#pragma unroll
    for (int n = 0; n < ON; n++) {
      out[n * 8 + 2 * t4] = o[n][2 * h] * inv;
      out[n * 8 + 2 * t4 + 1] = o[n][2 * h + 1] * inv;
    }
  }
#endif
}

// a block a query of one head (or a split of its keys), its warps' lane groups each taking keys (a group a key, its
// lanes splitting the head's dims, so a key's K and V rows load coalesced and its dot reduces over KEY_LANES lanes),
// DECODE_KEYS keys a group a step to keep loads in flight; the groups merged by shuffles, the warps in shared memory
template <int R, typename T>
__global__ void __launch_bounds__(DECODE_WARPS * WARP) decode_kernel(Args<T> a) {
  constexpr int HD = R * WARP;
  constexpr int PER = HD / KEY_LANES;
  __shared__ float sm[DECODE_WARPS];
  __shared__ float sl[DECODE_WARPS];
  __shared__ float sacc[DECODE_WARPS][HD];
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int group = lane / KEY_LANES;
  const int j = lane % KEY_LANES;
  const int splits = max(1, a.splits);
  const int i = blockIdx.x / splits;
  const int split = blockIdx.x % splits;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kvh = head / (a.heads / a.kv_heads);
  const int pos = past_of(a, batch) + i;
  const float tau = tau_of(a, pos);
  const int64_t row = (int64_t(batch) * a.t + i) * a.heads + head;
  float qr[PER];
#pragma unroll
  for (int r = 0; r < PER; r++) qr[r] = a.q[row * HD + j * PER + r] * a.scale;
  const float* bias = a.table + row * (a.extent + 1);
  // the sink starts the first split's first group's softmax: its weight and no value
  const bool sink = warp == 0 && group == 0 && split == 0;
  float m = sink ? a.sinks[row] : -INFINITY;
  float l = sink ? 1.0f : 0.0f;
  float acc[PER];
#pragma unroll
  for (int r = 0; r < PER; r++) acc[r] = 0.0f;
  // this split's contiguous share of the visible keys, over the splits the row's keys need
  const int first = first_key(a.window, pos);
  const int keys = pos - first + 1;
  const int used = min(splits, (keys + DECODE_SPLIT_KEYS - 1) / DECODE_SPLIT_KEYS);
  if (split >= used) return;
  const int per = (keys + used - 1) / used;
  const int lo = first + split * per;
  const int last = min(pos, lo + per - 1);
  constexpr int WARP_KEYS = KEY_GROUPS * DECODE_KEYS;
  // the warp steps as one (its shuffles stay converged), a group masking the keys past the split
  for (int base = lo + warp * WARP_KEYS; base <= last; base += DECODE_WARPS * WARP_KEYS) {
    float s[DECODE_KEYS];
    float vr[DECODE_KEYS][PER];
#pragma unroll
    for (int k = 0; k < DECODE_KEYS; k++) {
      const int key = min(base + group * DECODE_KEYS + k, last);
      float kv[PER];
      load_span_at<HD, PER>(a.k.row(batch, kvh, key, row_len<T>(HD)), j * PER, kv);
      load_span_at<HD, PER>(a.v.row(batch, kvh, key, row_len<T>(HD)), j * PER, vr[k]);
      float part = 0.0f;
#pragma unroll
      for (int r = 0; r < PER; r++) part += qr[r] * kv[r];
      s[k] = part;
    }
#pragma unroll
    for (int k = 0; k < DECODE_KEYS; k++) {
#pragma unroll
      for (int o = KEY_LANES / 2; o > 0; o /= 2) s[k] += __shfl_xor_sync(0xffffffffu, s[k], o);
    }
#pragma unroll
    for (int k = 0; k < DECODE_KEYS; k++) {
      const int key = base + group * DECODE_KEYS + k;
      if (key > last) break;
      const int dist = pos - key;
      const float score = (s[k] + (dist < a.extent ? bias[dist] : 0.0f)) * tau;
      const float m_new = fmaxf(m, score);
      const float alpha = __expf(m - m_new);
      const float p = __expf(score - m_new);
      l = l * alpha + p;
#pragma unroll
      for (int r = 0; r < PER; r++) acc[r] = acc[r] * alpha + p * vr[k][r];
      m = m_new;
    }
  }
  // the warp's groups merged: lanes of the same dims across the groups
  float mw = m;
#pragma unroll
  for (int o = KEY_LANES; o < WARP; o *= 2) mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, o));
  const float wt = m == -INFINITY ? 0.0f : __expf(m - mw);
  l *= wt;
#pragma unroll
  for (int r = 0; r < PER; r++) acc[r] *= wt;
#pragma unroll
  for (int o = KEY_LANES; o < WARP; o *= 2) {
    l += __shfl_xor_sync(0xffffffffu, l, o);
#pragma unroll
    for (int r = 0; r < PER; r++) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
  }
  m = mw;
  if (group == 0) {
#pragma unroll
    for (int r = 0; r < PER; r++) sacc[warp][j * PER + r] = acc[r];
  }
  if (lane == 0) {
    sm[warp] = m;
    sl[warp] = l;
  }
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
  if (used > 1) {
    float* parts = a.partials + row * splits * (HD + 2);
    float* part = parts + split * (HD + 2);
    if (lane == 0) {
      part[0] = mt;
      part[1] = lt;
    }
#pragma unroll
    for (int r = 0; r < R; r++) part[2 + lane * R + r] = out[r];
    __threadfence();
    __syncwarp();
    unsigned int before = 0;
    if (lane == 0) before = atomicAdd(a.arrived + row, 1u);
    before = __shfl_sync(0xffffffffu, before, 0);
    // the last split merges every split's partials
    if (before != unsigned(used) - 1) return;
    __threadfence();
    if (lane == 0) a.arrived[row] = 0;
    mt = -INFINITY;
    for (int sp = 0; sp < used; sp++) mt = fmaxf(mt, __ldcg(parts + sp * (HD + 2)));
    lt = 0.0f;
#pragma unroll
    for (int r = 0; r < R; r++) out[r] = 0.0f;
    for (int sp = 0; sp < used; sp++) {
      const float ms = __ldcg(parts + sp * (HD + 2));
      if (ms == -INFINITY) continue;
      const float wt = __expf(ms - mt);
      lt += __ldcg(parts + sp * (HD + 2) + 1) * wt;
#pragma unroll
      for (int r = 0; r < R; r++) out[r] += __ldcg(parts + sp * (HD + 2) + 2 + lane * R + r) * wt;
    }
  }
  const float inv = 1.0f / lt;
#pragma unroll
  for (int r = 0; r < R; r++) o[lane * R + r] = out[r] * inv;
}

// whether the device runs the tensor-core prefill (sm80+), asked once
bool mma_device() {
  static const bool mma = [] {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major >= 8;
  }();
  return mma;
}

// the SMs of the current device, asked once
int sm_count() {
  static const int count = [] {
    int dev = 0, n = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n;
  }();
  return count;
}

// blocks a decode's (query, head) row takes: enough rows of blocks for the SMs; a function of the shapes alone, so a
// captured graph keeps it at every position
int decode_splits(int64_t rows) {
  const int64_t want = (int64_t(sm_count()) * DECODE_BLOCKS_PER_SM + rows - 1) / rows;
  return int(std::min<int64_t>(DECODE_MAX_SPLITS, std::max<int64_t>(1, want)));
}

template <int R, typename T>
cudaError_t launch(const Args<T>& a, int batch, bool decode, cudaStream_t stream) {
  if (decode) {
    if (a.partials && a.arrived && a.splits > 1) {
      decode_kernel<R, T><<<dim3(a.t * a.splits, a.heads, batch), DECODE_WARPS * WARP, 0, stream>>>(a);
    } else {
      Args<T> one = a;
      one.splits = 1;
      decode_kernel<R, T><<<dim3(a.t, a.heads, batch), DECODE_WARPS * WARP, 0, stream>>>(one);
    }
  } else if (!std::is_same<T, float>::value && a.t >= MMA_MIN_QUERIES && mma_device()) {
    // a half-precision cache's values are exact in f16; an f32 one takes the f32 kernel
    const int tiles = (a.t + MMA_WARPS * MMA_ROWS - 1) / (MMA_WARPS * MMA_ROWS);
    prefill_mma_kernel<R, T><<<dim3(tiles, a.heads, batch), MMA_WARPS * WARP, 0, stream>>>(a);
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
  // a split decode's partials, of biased_attention_decode_partials' size, and a zeroed count a row (rows of them,
  // left zeroed); null for one block a row
  float* partials;
  unsigned int* arrived;
};

// the f32 partials a decode of `t` queries over `batch` sequences and `heads` takes (0: one block a row)
extern "C" int64_t biased_attention_decode_partials(int batch, int t, int heads, int head_dim) {
  const int64_t rows = int64_t(batch) * t * heads;
  const int splits = decode_splits(rows);
  return splits > 1 ? rows * splits * (head_dim + 2) : 0;
}

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
                p.scale, p.log_alpha, p.log_floor, p.partials, p.arrived,
                decode_splits(int64_t(p.batch) * p.t * p.heads)};
  return dispatch(a, p.batch, p.head_dim, p.t <= p.decode_rows, reinterpret_cast<cudaStream_t>(stream));
}

// K/V from a paged cache [blocks, kv_heads, block_size, head_dim] through block tables (batch, table_width) u32, or
// with `ring` a ring of that many blocks a sequence from each table row's first block;
// cache_dtype 0 f32, 1 f16, 2 bf16, 3 e4m3, 4 q8 (rows of head_dim + head_dim / 16 bytes; the caller rotates q and
// the output)
extern "C" int biased_attention_paged(const float* q, const void* k_cache, const void* v_cache, int cache_dtype,
                                      const uint32_t* tables, int64_t table_stride, int table_width, int block_size,
                                      int ring, const float* table, const float* sinks, float* out,
                                      const BiasedAttentionShape* shape, int64_t stream) {
  const BiasedAttentionShape& p = *shape;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const bool decode = p.t <= p.decode_rows;
#define PAGED(T)                                                                                                   \
  {                                                                                                                \
    Args<T> a{q,                                                                                                   \
              {(const T*)k_cache, 0, 0, tables, table_stride, table_width, block_size, p.kv_heads, ring},          \
              {(const T*)v_cache, 0, 0, tables, table_stride, table_width, block_size, p.kv_heads, ring},          \
              table, sinks, out, p.heads, p.kv_heads, p.t, p.past_lens, p.past_sub, p.past, p.extent, p.window,    \
              p.scale, p.log_alpha, p.log_floor, p.partials, p.arrived,                                          \
              decode_splits(int64_t(p.batch) * p.t * p.heads)};                                                   \
    return dispatch(a, p.batch, p.head_dim, decode, s);                                                            \
  }
  switch (cache_dtype) {
    case 0: PAGED(float)
    case 1: PAGED(__half)
    case 2: PAGED(__nv_bfloat16)
    case 3: PAGED(E4m3)
    case 4: PAGED(Q8)
    default: return -1;
  }
#undef PAGED
}
