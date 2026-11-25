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
// a decode's keys split over up to this many blocks a row of blocks (a query of a KV head's heads), enough to fill
// the SMs at long context; a row takes a split a DECODE_SPLIT_KEYS of its keys, the grid's other blocks exiting
constexpr int DECODE_MAX_SPLITS = 64;
constexpr int DECODE_BLOCKS_PER_SM = 6;
constexpr int DECODE_SPLIT_KEYS = 64;
// the tensor-core prefill: a warp 16 queries, a block 4 warps, keys a tile at a time; prompts from this many queries
constexpr int MMA_WARPS = 4;
constexpr int MMA_ROWS = 16;
constexpr int MMA_KEYS = 64;
constexpr int MMA_MIN_QUERIES = 32;
// shared rows padded by 8 halves against bank conflicts
constexpr int MMA_PAD = 8;
// a verify on tensor cores: a block every query and head of a KV head, up to two m16 tiles of rows, its warps a tile
// and half of each tile of keys; a split at least VERIFY_SPLIT_KEYS keys
constexpr int VERIFY_ROWS = 2 * MMA_ROWS;
constexpr int VERIFY_WARPS = 4;
constexpr int VERIFY_KEYS = 32;
constexpr int VERIFY_SPLIT_KEYS = 128;
// a sequence's verify rows take the tensor-core kernel from this many keys; under it the decode kernel's, whose rows
// sum as a decode row's do, bit for bit (a drafted greedy reply stays the plain one at short context)
constexpr int VERIFY_MIN_KEYS = 8192;

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
// a packed cache's row (kv_quant at 2..6 bits, the width a launch's `bits`): each 8 values take `bits` bytes, value v
// at bit v * bits of them (little-endian), (q + 0.5 - 2^(bits - 1)) / 2^(bits - 1) times its group's f16 scale; the
// scales after the HD * bits / 8 data bytes, rotated as q8's
struct Qp {
  uint8_t byte;
};
constexpr int PACK_VALUES = 8;
constexpr int MIN_PACKED_BITS = 2;
constexpr int MAX_PACKED_BITS = 6;

template <typename T>
constexpr bool quantized() {
  return std::is_same<T, Q8>::value || std::is_same<T, Qp>::value;
}

// a row's length in its elements (bytes for the quantized rows, whose scales follow their data)
template <typename T>
__device__ __forceinline__ constexpr int row_len(int hd, int bits) {
  return std::is_same<T, Q8>::value ? hd + hd / Q8_GROUP * 2
         : std::is_same<T, Qp>::value ? hd * bits / 8 + hd / Q8_GROUP * 2
                                       : hd;
}

// a quantized row's group scales
template <typename T>
__device__ __forceinline__ const __half* scales_of(const T* row, int hd, int bits) {
  return reinterpret_cast<const __half*>(row + hd * bits / 8);
}

// the midpoint of a width's grid, 2^(bits - 1)
__device__ __forceinline__ float grid_mid(int bits) { return float(1 << (bits - 1)); }

// the 8 values of a packed row's chunk `chunk`, its `bits` bytes in the low bits
__device__ __forceinline__ uint64_t packed_chunk(const Qp* row, int chunk, int bits) {
  const uint8_t* p = &row[chunk * bits].byte;
  uint64_t v = 0;
  for (int b = 0; b < bits; b++) v |= uint64_t(p[b]) << (8 * b);
  return v;
}

// values off..off + 3 of a packed row (off a multiple of 4) as 4 fields of `bits` bits in the low bits
__device__ __forceinline__ uint32_t packed4(const Qp* row, int off, int bits) {
  return uint32_t(packed_chunk(row, off / PACK_VALUES, bits) >> ((off % PACK_VALUES) * bits));
}

// value i of a row of HD values
template <int HD, typename T>
__device__ __forceinline__ float load_at(const T* row, int64_t i, int bits) {
  return load(row, i);
}
template <int HD>
__device__ __forceinline__ float load_at(const Q8* row, int64_t i, int bits) {
  const float scale = __half2float(reinterpret_cast<const __half*>(row + HD)[i / Q8_GROUP]);
  return (float(row[i].bits) + 0.5f - Q8_MID) * (scale / Q8_MID);
}
template <int HD>
__device__ __forceinline__ float load_at(const Qp* row, int64_t i, int bits) {
  const float mid = grid_mid(bits);
  const float scale = __half2float(scales_of(row, HD, bits)[i / Q8_GROUP]);
  const int q = int((packed_chunk(row, int(i) / PACK_VALUES, bits) >> ((i % PACK_VALUES) * bits)) & ((1u << bits) - 1));
  return (float(q) + 0.5f - mid) * (scale / mid);
}

// a quantized row's values off..off + 3 less the grid's midpoint (the +0.5 left to the caller), exact
template <typename T>
__device__ __forceinline__ void centered4(const T* row, int off, int bits, float* o);

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

// the power of 2 putting x's magnitude in [0.5, 1) (0 for 0), within what pow2 makes; its value
constexpr int POW2_LIMIT = 125;
__device__ __forceinline__ int exponent_of(float x) {
  return x > 0.0f ? max(-POW2_LIMIT, min(POW2_LIMIT, int((__float_as_uint(x) >> 23) & 0xFFu) - 126)) : 0;
}
__device__ __forceinline__ float pow2(int e) { return __uint_as_float(uint32_t(e + 127) << 23); }

// a q8 byte b as the f32 b - 128, exact: b under the bits of 2^23, less 2^23 + 128
constexpr unsigned F32_TWO23_BITS = 0x4B000000u;
constexpr float Q8_F32_OFFSET = 8388736.0f;
__device__ __forceinline__ float q8_centered(uint32_t raw, int byte) {
  return __uint_as_float(__byte_perm(raw, F32_TWO23_BITS, 0x7540 | byte)) - Q8_F32_OFFSET;
}

template <>
__device__ __forceinline__ void centered4<Q8>(const Q8* row, int off, int bits, float* o) {
  const uint32_t raw = *reinterpret_cast<const uint32_t*>(row + off);
#pragma unroll
  for (int b = 0; b < 4; b++) o[b] = q8_centered(raw, b);
}
template <>
__device__ __forceinline__ void centered4<Qp>(const Qp* row, int off, int bits, float* o) {
  const uint32_t raw = packed4(row, off, bits);
  const uint32_t top = (1u << bits) - 1;
  const float mid = grid_mid(bits);
#pragma unroll
  for (int b = 0; b < 4; b++) o[b] = float((raw >> (b * bits)) & top) - mid;
}

// 4 values of a row in one load, as they sit in the cache
template <int BYTES>
struct RawOf;
template <>
struct RawOf<1> {
  using type = uint32_t;
};
template <>
struct RawOf<2> {
  using type = uint2;
};
template <>
struct RawOf<4> {
  using type = uint4;
};
template <typename T>
using Raw4 = typename RawOf<sizeof(T)>::type;

// 4 loaded values as f16 pairs for the tensor cores: a q8 row's as the integers 2b - 255 (exact; its scales carry the
// rest), others as they are
template <typename T>
__device__ __forceinline__ void stage4(const Raw4<T>& raw, __half2& lo, __half2& hi) {
  float v[4];
  load_row<4>(reinterpret_cast<const T*>(&raw), v);
  lo = __floats2half2_rn(v[0], v[1]);
  hi = __floats2half2_rn(v[2], v[3]);
}
// a byte b under a high byte of 0x64 is the f16 1024 + b; less 1152, times 2, plus 1 is 2b - 255, each step exact
constexpr unsigned Q8_HALF_HIGH = 0x64646464u;
constexpr float Q8_HALF_OFFSET = 1152.0f;
// value = (2b - 255) x scale / (2 x 128)
// 4 loaded values of a 32-value group (8 lanes) as f16 pairs under a power of 2 of the group's largest, which it
// returns as the group's scale: f16 keeps small and large values' bits
template <typename T>
__device__ __forceinline__ float stage4_scaled(const Raw4<T>& raw, __half2& lo, __half2& hi) {
  float v[4];
  load_row<4>(reinterpret_cast<const T*>(&raw), v);
  float mx = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int o = 1; o < Q8_GROUP / 4; o *= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const int e = exponent_of(mx);
  const float down = pow2(-e);
  lo = __floats2half2_rn(v[0] * down, v[1] * down);
  hi = __floats2half2_rn(v[2] * down, v[3] * down);
  return pow2(e);
}

template <>
__device__ __forceinline__ void stage4<Q8>(const uint32_t& raw, __half2& lo, __half2& hi) {
  const __half2 two = __float2half2_rn(2.0f);
  const __half2 one = __float2half2_rn(1.0f);
  const __half2 off = __float2half2_rn(Q8_HALF_OFFSET);
  const uint32_t a = __byte_perm(raw, Q8_HALF_HIGH, 0x4140);
  const uint32_t b = __byte_perm(raw, Q8_HALF_HIGH, 0x4342);
  lo = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&a), off), two, one);
  hi = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&b), off), two, one);
}

// 4 values of a row at value d as the verify kernel stages them: as they sit, a packed row's as 4 fields
template <typename T>
__device__ __forceinline__ Raw4<T> fetch4(const T* row, int d, int bits) {
  return *reinterpret_cast<const Raw4<T>*>(row + d);
}
__device__ __forceinline__ uint32_t fetch4(const Qp* row, int d, int bits) { return packed4(row, d, bits); }

// a quantized row's 4 fetched values as f16 pairs of the exact integers 2q - (2^bits - 1), the scale / 2^bits after
template <typename T>
__device__ __forceinline__ void stage_quant(const uint32_t& raw, int bits, __half2& lo, __half2& hi) {
  if constexpr (std::is_same<T, Q8>::value) {
    stage4<Q8>(raw, lo, hi);
  } else {
    const uint32_t top = (1u << bits) - 1;
    float v[4];
#pragma unroll
    for (int r = 0; r < 4; r++) v[r] = float(2 * int((raw >> (r * bits)) & top) - int(top));
    lo = __floats2half2_rn(v[0], v[1]);
    hi = __floats2half2_rn(v[2], v[3]);
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
  // a quantized cache's width (8 for Q8, 2..6 for Qp), else unused
  int bits;

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
  // with a verify beside it, the decode kernel leaves sequences of this many keys or more to it (0: takes all)
  int verify_min_keys;
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
      kt[j][d] = in ? load_at<HD>(a.k.row(batch, kvh, key, row_len<T>(HD, a.k.bits)), d, a.k.bits) : 0.0f;
      vt[j][d] = in ? load_at<HD>(a.v.row(batch, kvh, key, row_len<T>(HD, a.v.bits)), d, a.v.bits) : 0.0f;
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

// four 8x8 f16 tiles from shared rows (lane l gives row l % 8 of tile l / 8), each transposed
__device__ __forceinline__ void ldmatrix_x4_trans(const __half* row, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3) {
#if __CUDA_ARCH__ >= 800
  const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(row));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
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
      ks[j][d] = __float2half(in ? load_at<HD>(a.k.row(batch, kvh, key, row_len<T>(HD, a.k.bits)), d, a.k.bits) : 0.0f);
      vt[d][j] = __float2half(in ? load_at<HD>(a.v.row(batch, kvh, key, row_len<T>(HD, a.v.bits)), d, a.v.bits) : 0.0f);
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

// a row's split partials `(splits, head_dim + 2)` merged into its output, a lane R of its values
template <int R>
__device__ __forceinline__ void merge_splits(const float* parts, int used, int lane, float* o) {
  constexpr int HD = R * WARP;
  float mt = -INFINITY;
  for (int sp = 0; sp < used; sp++) mt = fmaxf(mt, __ldcg(parts + sp * (HD + 2)));
  float lt = 0.0f;
  float out[R];
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
  const float inv = 1.0f / lt;
#pragma unroll
  for (int r = 0; r < R; r++) o[lane * R + r] = out[r] * inv;
}

// a block a query of G heads sharing a KV head (or a split of its keys), its warps' lane groups each taking keys (a
// group a key, its lanes splitting the head's dims, so a key's K and V rows load coalesced, once for all G heads, and
// each head's dot reduces over KEY_LANES lanes), DECODE_KEYS keys a group a step to keep loads in flight; the groups
// merged by shuffles, the warps in shared memory, warp g finishing head g
template <int R, int G, typename T>
__global__ void __launch_bounds__(DECODE_WARPS * WARP) decode_kernel(Args<T> a) {
  constexpr int HD = R * WARP;
  constexpr int PER = HD / KEY_LANES;
  static_assert(G <= DECODE_WARPS, "a warp finishes each head");
  __shared__ float sm[G][DECODE_WARPS];
  __shared__ float sl[G][DECODE_WARPS];
  __shared__ float sacc[G][DECODE_WARPS][HD];
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int group = lane / KEY_LANES;
  const int j = lane % KEY_LANES;
  const int splits = max(1, a.splits);
  const int i = blockIdx.x / splits;
  const int split = blockIdx.x % splits;
  const int head0 = blockIdx.y * G;
  const int batch = blockIdx.z;
  if (a.verify_min_keys > 0 && past_of(a, batch) + a.t >= a.verify_min_keys) return;
  const int kvh = head0 / (a.heads / a.kv_heads);
  const int pos = past_of(a, batch) + i;
  const float tau = tau_of(a, pos);
  const int64_t row0 = (int64_t(batch) * a.t + i) * a.heads + head0;
  // a q8 cache's values read as b - 128 times their group's scale / 128: K's +0.5 comes back through the lane's sum of
  // q, V's through a sum of the weights (c) every value of the lane's group shares
  constexpr bool QUANT = quantized<T>();
  float qr[G][PER], qsum[G];
#pragma unroll
  for (int g = 0; g < G; g++) {
    qsum[g] = 0.0f;
#pragma unroll
    for (int r = 0; r < PER; r++) {
      qr[g][r] = a.q[(row0 + g) * HD + j * PER + r] * a.scale;
      qsum[g] += qr[g][r];
    }
  }
  // the sink starts the first split's first group's softmax: its weight and no value
  const bool sink = warp == 0 && group == 0 && split == 0;
  float m[G], l[G], acc[G][PER], c[G];
#pragma unroll
  for (int g = 0; g < G; g++) {
    m[g] = sink ? a.sinks[row0 + g] : -INFINITY;
    l[g] = sink ? 1.0f : 0.0f;
    c[g] = 0.0f;
#pragma unroll
    for (int r = 0; r < PER; r++) acc[g][r] = 0.0f;
  }
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
    float s[G][DECODE_KEYS];
    float vr[DECODE_KEYS][PER];
    float vscale[DECODE_KEYS];
#pragma unroll
    for (int k = 0; k < DECODE_KEYS; k++) {
      const int key = min(base + group * DECODE_KEYS + k, last);
      const T* krow = a.k.row(batch, kvh, key, row_len<T>(HD, a.k.bits));
      const T* vrow = a.v.row(batch, kvh, key, row_len<T>(HD, a.v.bits));
      float kv[PER];
      float kscale = 1.0f;
      vscale[k] = 1.0f;
      if constexpr (QUANT) {
#pragma unroll
        for (int w = 0; w < PER / 4; w++) {
          centered4<T>(krow, j * PER + 4 * w, a.k.bits, kv + 4 * w);
          centered4<T>(vrow, j * PER + 4 * w, a.v.bits, vr[k] + 4 * w);
        }
        const int grp = j * PER / Q8_GROUP;
        kscale = __half2float(scales_of(krow, HD, a.k.bits)[grp]) / grid_mid(a.k.bits);
        vscale[k] = __half2float(scales_of(vrow, HD, a.v.bits)[grp]) / grid_mid(a.v.bits);
      } else {
        load_span_at<HD, PER>(krow, j * PER, kv);
        load_span_at<HD, PER>(vrow, j * PER, vr[k]);
      }
#pragma unroll
      for (int g = 0; g < G; g++) {
        float part = QUANT ? 0.5f * qsum[g] : 0.0f;
#pragma unroll
        for (int r = 0; r < PER; r++) part += qr[g][r] * kv[r];
        s[g][k] = part * kscale;
      }
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
#pragma unroll
      for (int k = 0; k < DECODE_KEYS; k++) {
#pragma unroll
        for (int o = KEY_LANES / 2; o > 0; o /= 2) s[g][k] += __shfl_xor_sync(0xffffffffu, s[g][k], o);
      }
    }
    // the group's keys fold in together: one rescale of the sums for both
#pragma unroll
    for (int g = 0; g < G; g++) {
      const float* bias = a.table + (row0 + g) * (a.extent + 1);
      float score[DECODE_KEYS];
      float mx = m[g];
#pragma unroll
      for (int k = 0; k < DECODE_KEYS; k++) {
        const int key = base + group * DECODE_KEYS + k;
        const int dist = pos - key;
        score[k] = key <= last ? (s[g][k] + (dist < a.extent ? bias[dist] : 0.0f)) * tau : -INFINITY;
        mx = fmaxf(mx, score[k]);
      }
      if (mx == -INFINITY) continue;
      const float alpha = __expf(m[g] - mx);
      float pv[DECODE_KEYS];
      float psum = 0.0f, pvsum = 0.0f;
#pragma unroll
      for (int k = 0; k < DECODE_KEYS; k++) {
        const float p = __expf(score[k] - mx);
        psum += p;
        pv[k] = p * vscale[k];
        pvsum += pv[k];
      }
      l[g] = l[g] * alpha + psum;
#pragma unroll
      for (int r = 0; r < PER; r++) {
        float x = acc[g][r] * alpha;
#pragma unroll
        for (int k = 0; k < DECODE_KEYS; k++) x += pv[k] * vr[k][r];
        acc[g][r] = x;
      }
      if constexpr (QUANT) c[g] = c[g] * alpha + 0.5f * pvsum;
      m[g] = mx;
    }
  }
  if constexpr (QUANT) {
#pragma unroll
    for (int g = 0; g < G; g++) {
#pragma unroll
      for (int r = 0; r < PER; r++) acc[g][r] += c[g];
    }
  }
  // the warp's groups merged: lanes of the same dims across the groups
#pragma unroll
  for (int g = 0; g < G; g++) {
    float mw = m[g];
#pragma unroll
    for (int o = KEY_LANES; o < WARP; o *= 2) mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, o));
    const float wt = m[g] == -INFINITY ? 0.0f : __expf(m[g] - mw);
    l[g] *= wt;
#pragma unroll
    for (int r = 0; r < PER; r++) acc[g][r] *= wt;
#pragma unroll
    for (int o = KEY_LANES; o < WARP; o *= 2) {
      l[g] += __shfl_xor_sync(0xffffffffu, l[g], o);
#pragma unroll
      for (int r = 0; r < PER; r++) acc[g][r] += __shfl_xor_sync(0xffffffffu, acc[g][r], o);
    }
    m[g] = mw;
    if (group == 0) {
#pragma unroll
      for (int r = 0; r < PER; r++) sacc[g][warp][j * PER + r] = acc[g][r];
    }
    if (lane == 0) {
      sm[g][warp] = m[g];
      sl[g][warp] = l[g];
    }
  }
  __syncthreads();
  if (warp >= G) return;
  const int g = warp;
  const int64_t row = row0 + g;
  float mt = -INFINITY;
  for (int w = 0; w < DECODE_WARPS; w++) mt = fmaxf(mt, sm[g][w]);
  float lt = 0.0f;
  float out[R];
#pragma unroll
  for (int r = 0; r < R; r++) out[r] = 0.0f;
  for (int w = 0; w < DECODE_WARPS; w++) {
    if (sm[g][w] == -INFINITY) continue;
    const float wt = __expf(sm[g][w] - mt);
    lt += sl[g][w] * wt;
#pragma unroll
    for (int r = 0; r < R; r++) out[r] += sacc[g][w][lane * R + r] * wt;
  }
  float* o = a.out + row * HD;
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
    merge_splits<R>(parts, used, lane, o);
    return;
  }
  const float inv = 1.0f / lt;
#pragma unroll
  for (int r = 0; r < R; r++) o[lane * R + r] = out[r] * inv;
}

// a verify's rows on tensor cores (sm80+): a block every query and head of one KV head (row r is query r / G, head
// r % G) over a split of the keys, so each K and V row loads once for all of them. Each key tile's K and V^T staged
// in shared memory as f16, a q8 cache's as the exact integers 2b - 255 with its scales applied to the f32 sums by
// group; q and the probabilities each as an f16 part and its f16 remainder, as in the prefill. Warp w takes row tile
// w % 2 and half w / 2 of each key tile; the halves merge in shared memory, the splits through the partials.
template <int R, typename T>
__global__ void __launch_bounds__(VERIFY_WARPS * WARP) verify_mma_kernel(Args<T> a) {
#if __CUDA_ARCH__ >= 800
  constexpr bool QUANT = quantized<T>();
  constexpr int HD = R * WARP;
  constexpr int KSTEPS = HD / 16;
  constexpr int GROUPS = HD / Q8_GROUP;
  constexpr int GROUP_TILES = Q8_GROUP / 8;
  constexpr int WARP_KEYS = VERIFY_KEYS / 2;
  constexpr int SN = WARP_KEYS / 8;
  constexpr int ON = HD / 8;
  static_assert(GROUP_TILES % 2 == 0 && GROUPS * GROUP_TILES == ON, "value tiles load in pairs, a group's");
  constexpr int ROW_BYTES = VERIFY_KEYS * (HD + MMA_PAD) * 2;
  constexpr int SC_BYTES = 2 * VERIFY_KEYS * GROUPS * 4;
  constexpr int MERGE_BYTES = VERIFY_ROWS * (HD + 2) * 4;
  constexpr int STAGE_BYTES = 2 * ROW_BYTES + SC_BYTES;
  // a thread's chunks of 4 values a tile, and its scales
  constexpr int CHUNKS = VERIFY_KEYS * HD / 4 / (VERIFY_WARPS * WARP);
  constexpr int SCALES = QUANT ? (VERIFY_KEYS * GROUPS + VERIFY_WARPS * WARP - 1) / (VERIFY_WARPS * WARP) : 0;
  __shared__ __align__(16) unsigned char smem[STAGE_BYTES > MERGE_BYTES ? STAGE_BYTES : MERGE_BYTES];
  __shared__ unsigned int ticket;
  auto ks = reinterpret_cast<__half(*)[HD + MMA_PAD]>(smem);
  auto vs = reinterpret_cast<__half(*)[HD + MMA_PAD]>(smem + ROW_BYTES);
  auto ksc = reinterpret_cast<float(*)[GROUPS]>(smem + 2 * ROW_BYTES);
  auto vsc = ksc + VERIFY_KEYS;
  auto merged = reinterpret_cast<float(*)[HD + 2]>(smem);
  const int warp = threadIdx.x / WARP;
  const int lane = threadIdx.x % WARP;
  const int g = lane / 4;
  const int t4 = lane % 4;
  const int mt = warp % 2;
  const int kb = warp / 2 * WARP_KEYS;
  const int splits = max(1, a.splits);
  const int split = blockIdx.x;
  const int kvh = blockIdx.y;
  const int batch = blockIdx.z;
  const int group = a.heads / a.kv_heads;
  const int nr = a.t * group;
  const int past = past_of(a, batch);
  if (past + a.t < a.verify_min_keys) return;
  // this lane's two rows
  int pos[2], first[2];
  bool valid[2];
  float tau[2];
  const float* bias[2];
  int64_t qrow[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int r = mt * MMA_ROWS + g + 8 * h;
    valid[h] = r < nr;
    const int i = valid[h] ? r / group : 0;
    qrow[h] = (int64_t(batch) * a.t + i) * a.heads + kvh * group + (valid[h] ? r % group : 0);
    pos[h] = past + i;
    first[h] = first_key(a.window, pos[h]);
    tau[h] = tau_of(a, pos[h]);
    bias[h] = a.table + qrow[h] * (a.extent + 1);
  }
  // q's fragments, an f16 part and its remainder, in shared memory for every warp of the row tile
  __shared__ uint4 qs[2][2][KSTEPS][WARP];
  // each row under a power of 2 of its largest before its f16 split, so small or large q keep their bits; the factor
  // comes back on its scores
  auto q_of = [&](int h, int d) -> float { return valid[h] ? a.q[qrow[h] * HD + d] * a.scale : 0.0f; };
  float qdown[2], qup[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    float mx = 0.0f;
#pragma unroll
    for (int kk = 0; kk < KSTEPS; kk++) {
      const int d = kk * 16 + 2 * t4;
      mx = fmaxf(mx, fmaxf(fmaxf(fabsf(q_of(h, d)), fabsf(q_of(h, d + 1))),
                           fmaxf(fabsf(q_of(h, d + 8)), fabsf(q_of(h, d + 9)))));
    }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const int e = exponent_of(mx);
    qdown[h] = pow2(-e);
    qup[h] = pow2(e);
  }
  auto q_at = [&](int h, int d) -> float { return q_of(h, d) * qdown[h]; };
#pragma unroll
  for (int kk = 0; kk < KSTEPS; kk++) {
    const int d = kk * 16 + 2 * t4;
    uint32_t qf[4], qr[4];
    split_half2(q_at(0, d), q_at(0, d + 1), qf[0], qr[0]);
    split_half2(q_at(1, d), q_at(1, d + 1), qf[1], qr[1]);
    split_half2(q_at(0, d + 8), q_at(0, d + 9), qf[2], qr[2]);
    split_half2(q_at(1, d + 8), q_at(1, d + 9), qf[3], qr[3]);
    // each key half's warp writes one of them
    qs[kb != 0][mt][kk][lane] =
        kb == 0 ? make_uint4(qf[0], qf[1], qf[2], qf[3]) : make_uint4(qr[0], qr[1], qr[2], qr[3]);
  }
  // the sink starts the first split's first key half
  const bool sink = split == 0 && kb == 0;
  float m[2], l[2];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    m[h] = valid[h] && sink ? a.sinks[qrow[h]] : -INFINITY;
    l[h] = valid[h] && sink ? 1.0f : 0.0f;
  }
  float o[ON][4];
#pragma unroll
  for (int n = 0; n < ON; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  // this split's share of the keys any row sees
  const int lo = first_key(a.window, past);
  const int hi = past + a.t - 1;
  const int keys = hi - lo + 1;
  const int used = min(splits, (keys + VERIFY_SPLIT_KEYS - 1) / VERIFY_SPLIT_KEYS);
  if (split >= used) return;
  const int per = (keys + used - 1) / used;
  const int s_lo = lo + split * per;
  const int s_hi = min(hi, s_lo + per - 1);
  // a tile's row addresses, worked out once a tile (two tiles' worth, the next one's ahead); its values and scales
  // load into registers while the tile before computes
  __shared__ const T* rows[2][2][VERIFY_KEYS];
  const auto locate = [&](int base, int buf) {
    if (threadIdx.x >= 2 * VERIFY_KEYS) return;
    const int j = threadIdx.x % VERIFY_KEYS;
    const int key = min(base + j, s_hi);
    const Kv<T>& kv = threadIdx.x < VERIFY_KEYS ? a.k : a.v;
    rows[buf][threadIdx.x / VERIFY_KEYS][j] = kv.row(batch, kvh, key, row_len<T>(HD, kv.bits));
  };
  static_assert(VERIFY_WARPS * WARP >= 2 * VERIFY_KEYS, "a thread a K or V row address");
  Raw4<T> kraw[CHUNKS], vraw[CHUNKS];
  float kscale[SCALES > 0 ? SCALES : 1], vscale[SCALES > 0 ? SCALES : 1];
  const auto fetch = [&](int buf) {
#pragma unroll
    for (int c = 0; c < CHUNKS; c++) {
      const int e = threadIdx.x + c * VERIFY_WARPS * WARP;
      const int j = e / (HD / 4);
      const int d = e % (HD / 4) * 4;
      kraw[c] = fetch4(rows[buf][0][j], d, a.k.bits);
      vraw[c] = fetch4(rows[buf][1][j], d, a.v.bits);
    }
    if constexpr (QUANT) {
#pragma unroll
      for (int c = 0; c < SCALES; c++) {
        const int e = min(int(threadIdx.x) + c * VERIFY_WARPS * WARP, VERIFY_KEYS * GROUPS - 1);
        const auto scale = [&](int kv) {
          const int bits = kv == 0 ? a.k.bits : a.v.bits;
          return __half2float(scales_of(rows[buf][kv][e / GROUPS], HD, bits)[e % GROUPS]) /
                 (2.0f * grid_mid(bits));
        };
        kscale[c] = scale(0);
        vscale[c] = scale(1);
      }
    }
  };
  locate(s_lo, 0);
  __syncthreads();
  fetch(0);
  for (int base = s_lo, it = 0; base <= s_hi; base += VERIFY_KEYS, it++) {
    const bool more = base + VERIFY_KEYS <= s_hi;
    __syncthreads();
    // keys past the split stage as zeros: their scores are masked, their values must not be NaN
#pragma unroll
    for (int c = 0; c < CHUNKS; c++) {
      const int e = threadIdx.x + c * VERIFY_WARPS * WARP;
      const int j = e / (HD / 4);
      const int d = e % (HD / 4) * 4;
      __half2 k01, k23, v01, v23;
      if constexpr (QUANT) {
        stage_quant<T>(kraw[c], a.k.bits, k01, k23);
        stage_quant<T>(vraw[c], a.v.bits, v01, v23);
      } else {
        const float kup = stage4_scaled<T>(kraw[c], k01, k23);
        const float vup = stage4_scaled<T>(vraw[c], v01, v23);
        if (d % Q8_GROUP == 0) ksc[j][d / Q8_GROUP] = kup, vsc[j][d / Q8_GROUP] = vup;
      }
      if (base + j > s_hi) k01 = k23 = v01 = v23 = __float2half2_rn(0.0f);
      *reinterpret_cast<uint2*>(&ks[j][d]) =
          make_uint2(*reinterpret_cast<uint32_t*>(&k01), *reinterpret_cast<uint32_t*>(&k23));
      *reinterpret_cast<uint2*>(&vs[j][d]) =
          make_uint2(*reinterpret_cast<uint32_t*>(&v01), *reinterpret_cast<uint32_t*>(&v23));
    }
    if constexpr (QUANT) {
#pragma unroll
      for (int c = 0; c < SCALES; c++) {
        const int e = threadIdx.x + c * VERIFY_WARPS * WARP;
        if (e >= VERIFY_KEYS * GROUPS) break;
        const bool in = base + e / GROUPS <= s_hi;
        ksc[e / GROUPS][e % GROUPS] = in ? kscale[c] : 0.0f;
        vsc[e / GROUPS][e % GROUPS] = in ? vscale[c] : 0.0f;
      }
    }
    if (more) locate(base + VERIFY_KEYS, (it + 1) % 2);
    __syncthreads();
    if (more) fetch((it + 1) % 2);
    // S = Q K^T a 32-value group of dims at a time, each group's sums times its keys' scales
    constexpr int GSTEPS = Q8_GROUP / 16;
    float sc[SN][4];
#pragma unroll
    for (int n = 0; n < SN; n++) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.0f;
#pragma unroll
    for (int gi = 0; gi < KSTEPS / GSTEPS; gi++) {
      float part[SN][4];
#pragma unroll
      for (int n = 0; n < SN; n++) part[n][0] = part[n][1] = part[n][2] = part[n][3] = 0.0f;
#pragma unroll
      for (int kk = gi * GSTEPS; kk < (gi + 1) * GSTEPS; kk++) {
        const uint4 hi4 = qs[0][mt][kk][lane];
        const uint4 lo4 = qs[1][mt][kk][lane];
        const uint32_t qf[4] = {hi4.x, hi4.y, hi4.z, hi4.w};
        const uint32_t qr[4] = {lo4.x, lo4.y, lo4.z, lo4.w};
#pragma unroll
        for (int n = 0; n < SN; n++) {
          const __half* krow = &ks[kb + n * 8 + g][kk * 16 + 2 * t4];
          const uint32_t k0 = *reinterpret_cast<const uint32_t*>(krow);
          const uint32_t k1 = *reinterpret_cast<const uint32_t*>(krow + 8);
          mma_f16(part[n], qf, k0, k1);
          mma_f16(part[n], qr, k0, k1);
        }
      }
#pragma unroll
      for (int n = 0; n < SN; n++) {
        const int kc = kb + n * 8 + 2 * t4;
#pragma unroll
        for (int c = 0; c < 4; c++) sc[n][c] += ksc[kc + c % 2][gi] * part[n][c];
      }
    }
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < SN; n++) {
#pragma unroll
      for (int c = 0; c < 4; c++) {
        const int h = c / 2;
        const int key = base + kb + n * 8 + 2 * t4 + c % 2;
        const int dist = pos[h] - key;
        const bool visible = valid[h] && key <= pos[h] && key >= first[h] && key <= s_hi;
        const float b = dist >= 0 && dist < a.extent ? bias[h][dist] : 0.0f;
        sc[n][c] = visible ? (sc[n][c] * qup[h] + b) * tau[h] : -INFINITY;
        tile_max[h] = fmaxf(tile_max[h], sc[n][c]);
      }
    }
    float alpha[2], rowsum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int h = 0; h < 2; h++) {
      tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffu, tile_max[h], 1));
      tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffu, tile_max[h], 2));
      const float m_new = fmaxf(m[h], tile_max[h]);
      alpha[h] = m[h] == -INFINITY ? 0.0f : __expf(m[h] - m_new);
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
    // O += P V: two score tiles (16 keys) are an A fragment, a group's value tiles taking P times its keys' value
    // scales, under a power of 2 of the warp's largest (back on the group's f32 sums), so the weights stay in f16 range
#pragma unroll
    for (int kk = 0; kk < WARP_KEYS / 16; kk++) {
      const int kc = kb + kk * 16 + 2 * t4;
#pragma unroll
      for (int gi = 0; gi < GROUPS; gi++) {
        float w[4] = {vsc[kc][gi], vsc[kc + 1][gi], vsc[kc + 8][gi], vsc[kc + 9][gi]};
        float wmax = fmaxf(fmaxf(w[0], w[1]), fmaxf(w[2], w[3]));
        wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, 1));
        wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, 2));
        const int e = exponent_of(wmax);
        const float down = pow2(-e);
#pragma unroll
        for (int x = 0; x < 4; x++) w[x] *= down;
        uint32_t pf[4], pr[4];
        split_half2(sc[2 * kk][0] * w[0], sc[2 * kk][1] * w[1], pf[0], pr[0]);
        split_half2(sc[2 * kk][2] * w[0], sc[2 * kk][3] * w[1], pf[1], pr[1]);
        split_half2(sc[2 * kk + 1][0] * w[2], sc[2 * kk + 1][1] * w[3], pf[2], pr[2]);
        split_half2(sc[2 * kk + 1][2] * w[2], sc[2 * kk + 1][3] * w[3], pf[3], pr[3]);
        float part[GROUP_TILES][4];
#pragma unroll
        for (int n = 0; n < GROUP_TILES; n++) part[n][0] = part[n][1] = part[n][2] = part[n][3] = 0.0f;
        // B fragments of two value tiles a load, from key-major rows transposed
#pragma unroll
        for (int n = 0; n < GROUP_TILES; n += 2) {
          const int mat = lane / 8;
          const __half* vrow = &vs[kb + kk * 16 + mat % 2 * 8 + lane % 8][(gi * GROUP_TILES + n + mat / 2) * 8];
          uint32_t v0, v1, v2, v3;
          ldmatrix_x4_trans(vrow, v0, v1, v2, v3);
          mma_f16(part[n], pf, v0, v1);
          mma_f16(part[n], pr, v0, v1);
          mma_f16(part[n + 1], pf, v2, v3);
          mma_f16(part[n + 1], pr, v2, v3);
        }
        const float up = pow2(e);
#pragma unroll
        for (int n = 0; n < GROUP_TILES; n++) {
#pragma unroll
          for (int x = 0; x < 4; x++) o[gi * GROUP_TILES + n][x] += up * part[n][x];
        }
      }
    }
  }
  // the second key half's warps hand theirs to the first's
  __syncthreads();
  if (kb != 0) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      float* dst = merged[mt * MMA_ROWS + g + 8 * h];
      if (t4 == 0) dst[0] = m[h], dst[1] = l[h];
#pragma unroll
      for (int n = 0; n < ON; n++) {
        dst[2 + n * 8 + 2 * t4] = o[n][2 * h];
        dst[2 + n * 8 + 2 * t4 + 1] = o[n][2 * h + 1];
      }
    }
  }
  __syncthreads();
  if (kb == 0) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const float* src = merged[mt * MMA_ROWS + g + 8 * h];
      const float mo = fmaxf(m[h], src[0]);
      const float w0 = m[h] == -INFINITY ? 0.0f : __expf(m[h] - mo);
      const float w1 = src[0] == -INFINITY ? 0.0f : __expf(src[0] - mo);
      const float lt = l[h] * w0 + src[1] * w1;
      if (!valid[h]) continue;
      float* dst;
      float scale_out;
      if (used > 1) {
        dst = a.partials + (qrow[h] * splits + split) * (HD + 2);
        if (t4 == 0) dst[0] = mo, dst[1] = lt;
        dst += 2;
        scale_out = 1.0f;
      } else {
        dst = a.out + qrow[h] * HD;
        scale_out = 1.0f / lt;
      }
#pragma unroll
      for (int n = 0; n < ON; n++) {
        const int d = n * 8 + 2 * t4;
        dst[d] = (o[n][2 * h] * w0 + src[2 + d] * w1) * scale_out;
        dst[d + 1] = (o[n][2 * h + 1] * w0 + src[2 + d + 1] * w1) * scale_out;
      }
    }
  }
  if (used == 1) return;
  __threadfence();
  __syncthreads();
  // one count a block: its first row's
  const int64_t counter = int64_t(batch) * a.t * a.heads + kvh * group;
  if (threadIdx.x == 0) ticket = atomicAdd(a.arrived + counter, 1u);
  __syncthreads();
  // the last split merges every row's partials
  if (ticket != unsigned(used) - 1) return;
  __threadfence();
  if (threadIdx.x == 0) a.arrived[counter] = 0;
  for (int r = warp; r < nr; r += VERIFY_WARPS) {
    const int64_t row = (int64_t(batch) * a.t + r / group) * a.heads + kvh * group + r % group;
    merge_splits<R>(a.partials + row * splits * (HD + 2), used, lane, a.out + row * HD);
  }
#endif
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

// heads a decode block takes: every head of a KV head where the kernel is built for that many, else one
int decode_group(int heads, int kv_heads) {
  const int group = kv_heads > 0 && heads % kv_heads == 0 ? heads / kv_heads : 1;
  return group >= 2 && group <= 4 ? group : 1;
}

// blocks a decode's row of blocks takes: enough for the SMs; a function of the shapes alone, so a captured graph keeps
// it at every position
int decode_splits(int batch, int t, int heads, int kv_heads) {
  const int64_t rows = int64_t(batch) * t * (heads / decode_group(heads, kv_heads));
  const int64_t want = (int64_t(sm_count()) * DECODE_BLOCKS_PER_SM + rows - 1) / rows;
  return int(std::min<int64_t>(DECODE_MAX_SPLITS, std::max<int64_t>(1, want)));
}

// a verify's blocks a KV head: enough for the SMs
int verify_splits(int batch, int kv_heads) {
  const int64_t rows = int64_t(batch) * kv_heads;
  const int64_t want = (int64_t(sm_count()) * DECODE_BLOCKS_PER_SM + rows - 1) / rows;
  return int(std::min<int64_t>(DECODE_MAX_SPLITS, std::max<int64_t>(1, want)));
}

// whether a decode of `t` queries takes the tensor-core verify: several queries, every row of a KV head in its tiles
bool verify_mma(int t, int heads, int kv_heads) {
  return t > 1 && kv_heads > 0 && heads % kv_heads == 0 && t * (heads / kv_heads) <= VERIFY_ROWS && mma_device();
}

// the splits a decode's partials are sized for: the most either kernel takes
int alloc_splits(int batch, int t, int heads, int kv_heads) {
  const int splits = decode_splits(batch, t, heads, kv_heads);
  return verify_mma(t, heads, kv_heads) ? std::max(splits, verify_splits(batch, kv_heads)) : splits;
}

template <int R, typename T>
cudaError_t launch(const Args<T>& a, int batch, bool decode, cudaStream_t stream) {
  const bool split = a.partials && a.arrived && a.splits > 1;
  if (decode) {
    // an f32 cache keeps the f32 kernel; with the verify, both kernels launch, each taking the sequences of its
    // length, and both split as the partials were sized (their layouts the same)
    const bool verify = !std::is_same<T, float>::value && verify_mma(a.t, a.heads, a.kv_heads);
    Args<T> d = a;
    d.splits = !split ? 1 : verify ? a.splits : std::min(a.splits, decode_splits(batch, a.t, a.heads, a.kv_heads));
    d.verify_min_keys = verify ? VERIFY_MIN_KEYS : 0;
    if (verify) verify_mma_kernel<R, T><<<dim3(d.splits, a.kv_heads, batch), VERIFY_WARPS * WARP, 0, stream>>>(d);
    const dim3 block(DECODE_WARPS * WARP);
    // heads sharing a KV head go to one block, which reads each K and V row once for all of them
    switch (decode_group(a.heads, a.kv_heads)) {
      case 2: decode_kernel<R, 2, T><<<dim3(a.t * d.splits, a.kv_heads, batch), block, 0, stream>>>(d); break;
      case 3: decode_kernel<R, 3, T><<<dim3(a.t * d.splits, a.kv_heads, batch), block, 0, stream>>>(d); break;
      case 4: decode_kernel<R, 4, T><<<dim3(a.t * d.splits, a.kv_heads, batch), block, 0, stream>>>(d); break;
      default: decode_kernel<R, 1, T><<<dim3(a.t * d.splits, a.heads, batch), block, 0, stream>>>(d); break;
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

// a packed cache's head sizes: the quantized row format's (a multiple of 64)
int dispatch_packed(const Args<Qp>& a, int batch, int head_dim, bool decode, cudaStream_t s) {
  cudaError_t err;
  switch (head_dim) {
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
extern "C" int64_t biased_attention_decode_partials(int batch, int t, int heads, int kv_heads, int head_dim) {
  const int64_t rows = int64_t(batch) * t * heads;
  const int splits = alloc_splits(batch, t, heads, kv_heads);
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
                alloc_splits(p.batch, p.t, p.heads, p.kv_heads)};
  return dispatch(a, p.batch, p.head_dim, p.t <= p.decode_rows, reinterpret_cast<cudaStream_t>(stream));
}

// K/V from a paged cache [blocks, kv_heads, block_size, head_dim] through block tables (batch, table_width) u32, or
// with `ring` a ring of that many blocks a sequence from each table row's first block;
// cache_dtype 0 f32, 1 f16, 2 bf16, 3 e4m3, 4 q8 (rows of head_dim + head_dim / 16 bytes; the caller rotates q and
// the output), 5 packed at `cache_bits` 2..6 (rows of head_dim * bits / 8 + head_dim / 16 bytes, rotated as q8's)
extern "C" int biased_attention_paged(const float* q, const void* k_cache, const void* v_cache, int cache_dtype,
                                      int cache_bits,
                                      const uint32_t* tables, int64_t table_stride, int table_width, int block_size,
                                      int ring, const float* table, const float* sinks, float* out,
                                      const BiasedAttentionShape* shape, int64_t stream) {
  const BiasedAttentionShape& p = *shape;
  const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const bool decode = p.t <= p.decode_rows;
#define PAGED(T, BITS, DISPATCH)                                                                                   \
  {                                                                                                                \
    Args<T> a{q,                                                                                                   \
              {(const T*)k_cache, 0, 0, tables, table_stride, table_width, block_size, p.kv_heads, ring, BITS},    \
              {(const T*)v_cache, 0, 0, tables, table_stride, table_width, block_size, p.kv_heads, ring, BITS},    \
              table, sinks, out, p.heads, p.kv_heads, p.t, p.past_lens, p.past_sub, p.past, p.extent, p.window,    \
              p.scale, p.log_alpha, p.log_floor, p.partials, p.arrived,                                          \
              alloc_splits(p.batch, p.t, p.heads, p.kv_heads)};                                                    \
    return DISPATCH(a, p.batch, p.head_dim, decode, s);                                                            \
  }
  switch (cache_dtype) {
    case 0: PAGED(float, 0, dispatch)
    case 1: PAGED(__half, 0, dispatch)
    case 2: PAGED(__nv_bfloat16, 0, dispatch)
    case 3: PAGED(E4m3, 0, dispatch)
    case 4: PAGED(Q8, 8, dispatch)
    case 5:
      if (cache_bits < MIN_PACKED_BITS || cache_bits > MAX_PACKED_BITS) return -1;
      PAGED(Qp, cache_bits, dispatch_packed)
    default: return -1;
  }
#undef PAGED
}
