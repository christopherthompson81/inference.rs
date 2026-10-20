// Tensor-core matvec for 1-8 input rows over row-major Q4_K and Q6_K weights
// (sm_80+): like mmvq_tc.cuh's IQ4_KS kernel, a warp multiplies 16 weight rows
// by every input row's int8 activations (mmvq_tc_quantize), so each weight
// block is read and decoded once whatever the row count. Lane (g, t) holds rows
// g and g + 8 of its 16-row group and the block's values 4t..4t+3 and
// 16+4t..16+4t+3, which are words 4t and 16+4t of the k-quants' 32-byte chunks.

#pragma once
#include "mmvq_gguf.cuh"

#define MMVQ_TCK_ROWS 16
#define MMVQ_TCK_BLOCKS 8

static __device__ __forceinline__ void
tck_mma_k32(int (&c)[4], const uint32_t a0, const uint32_t a1,
            const uint32_t a2, const uint32_t a3, const int2 b) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, "
      "%5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y));
}

static __device__ __forceinline__ void
tck_mma_k16(int (&c)[4], const uint32_t a0, const uint32_t a1, const int b) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, "
      "%5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(b));
}

static __device__ __forceinline__ uint32_t tck_byte(const uint32_t w,
                                                    const int i) {
  return (w >> (8 * i)) & 0xff;
}

// Q6_K superblocks are 210 bytes, so a word inside one is only 2-byte aligned
static __device__ __forceinline__ uint32_t tck_ld_b2(const char *p) {
  const unsigned short *h = (const unsigned short *)p;
  return (uint32_t)__ldcs(h) | (uint32_t)__ldcs(h + 1) << 16;
}

// One superblock of rows g and g + 8: Q4_K's header (dm, 12 scale bytes) and
// per 64-value chunk its two words
struct TckQ4K {
  static constexpr int WORDS = 4 + 8;
  static __device__ __forceinline__ void load(const char *sb, const int t,
                                              uint32_t (&w)[WORDS]) {
    const uint4 h = __ldg((const uint4 *)sb);
    w[0] = h.x;
    w[1] = h.y;
    w[2] = h.z;
    w[3] = h.w;
    const char *qs = sb + 16;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      w[4 + 2 * c] = __ldcs((const uint32_t *)(qs + 32 * c + 4 * t));
      w[5 + 2 * c] = __ldcs((const uint32_t *)(qs + 32 * c + 16 + 4 * t));
    }
  }
};

// One superblock of rows g and g + 8: Q6_K's ql words per (half, quarter
// parity), qh words per half, 16 scale bytes, d
struct TckQ6K {
  static constexpr int WORDS = 8 + 4 + 4 + 1;
  static __device__ __forceinline__ void load(const char *sb, const int t,
                                              uint32_t (&w)[WORDS]) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        w[4 * n + 2 * h] = tck_ld_b2(sb + 64 * n + 32 * h + 4 * t);
        w[4 * n + 2 * h + 1] = tck_ld_b2(sb + 64 * n + 32 * h + 16 + 4 * t);
      }
      w[8 + 2 * n] = tck_ld_b2(sb + 128 + 32 * n + 4 * t);
      w[9 + 2 * n] = tck_ld_b2(sb + 128 + 32 * n + 16 + 4 * t);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
      w[12 + i] = tck_ld_b2(sb + 192 + 4 * i);
    w[16] = __ldg((const unsigned short *)(sb + 208));
  }
};

// Q4_K's 6-bit scale and min of sub-block j (get_scale_min_k4)
template <int j>
static __device__ __forceinline__ void
tck_q4k_scale_min(const uint32_t (&w)[TckQ4K::WORDS], int &sc, int &m) {
  if (j < 4) {
    sc = tck_byte(w[1], j) & 63;
    m = tck_byte(w[2], j) & 63;
  } else {
    sc = (tck_byte(w[3], j - 4) & 0xf) | ((tck_byte(w[1], j - 4) >> 6) << 4);
    m = (tck_byte(w[3], j - 4) >> 4) | ((tck_byte(w[2], j - 4) >> 6) << 4);
  }
}

template <int j>
static __device__ __forceinline__ void
tck_q4k_block(const uint32_t (&wa)[TckQ4K::WORDS],
              const uint32_t (&wb)[TckQ4K::WORDS], const int2 bq,
              const float2 dy, const int2 xs, float (&acc)[4]) {
  constexpr int c = j / 2;
  constexpr int shift = 4 * (j & 1);
  int r[4] = {0, 0, 0, 0};
  tck_mma_k32(r, (wa[4 + 2 * c] >> shift) & 0x0f0f0f0f,
              (wb[4 + 2 * c] >> shift) & 0x0f0f0f0f,
              (wa[5 + 2 * c] >> shift) & 0x0f0f0f0f,
              (wb[5 + 2 * c] >> shift) & 0x0f0f0f0f, bq);
  int sca, ma, scb, mb;
  tck_q4k_scale_min<j>(wa, sca, ma);
  tck_q4k_scale_min<j>(wb, scb, mb);
  const float2 dma = __half22float2(*(const half2 *)&wa[0]);
  const float2 dmb = __half22float2(*(const half2 *)&wb[0]);
  const float da = dma.x * sca, mina = dma.y * ma;
  const float db = dmb.x * scb, minb = dmb.y * mb;
  acc[0] += dy.x * (da * (float)r[0] - mina * (float)xs.x);
  acc[1] += dy.y * (da * (float)r[1] - mina * (float)xs.y);
  acc[2] += dy.x * (db * (float)r[2] - minb * (float)xs.x);
  acc[3] += dy.y * (db * (float)r[3] - minb * (float)xs.y);
}

// block j's 6-bit values (word 0 for values 4t.., 1 for 16+4t..) recentred to
// int8
template <int j, int half>
static __device__ __forceinline__ uint32_t
tck_q6k_values(const uint32_t (&w)[TckQ6K::WORDS]) {
  constexpr int n = j / 4;
  constexpr int q = j % 4;
  const uint32_t lo =
      (w[4 * n + 2 * (q & 1) + half] >> (4 * (q >> 1))) & 0x0f0f0f0f;
  const uint32_t hi = ((w[8 + 2 * n + half] >> (2 * q)) & 0x03030303) << 4;
  return __vsub4(lo | hi, 0x20202020);
}

template <int j, int half>
static __device__ __forceinline__ int
tck_q6k_scale(const uint32_t (&w)[TckQ6K::WORDS]) {
  constexpr int i = 8 * (j / 4) + 2 * (j % 4) + half;
  return (int)(int8_t)tck_byte(w[12 + i / 4], i % 4);
}

template <int j>
static __device__ __forceinline__ void
tck_q6k_block(const uint32_t (&wa)[TckQ6K::WORDS],
              const uint32_t (&wb)[TckQ6K::WORDS], const int2 bq,
              const float2 dy, const int2 xs, float (&acc)[4]) {
  int lo[4] = {0, 0, 0, 0};
  int hi[4] = {0, 0, 0, 0};
  tck_mma_k16(lo, tck_q6k_values<j, 0>(wa), tck_q6k_values<j, 0>(wb), bq.x);
  tck_mma_k16(hi, tck_q6k_values<j, 1>(wa), tck_q6k_values<j, 1>(wb), bq.y);
  const int sa0 = tck_q6k_scale<j, 0>(wa), sa1 = tck_q6k_scale<j, 1>(wa);
  const int sb0 = tck_q6k_scale<j, 0>(wb), sb1 = tck_q6k_scale<j, 1>(wb);
  const float da = __half2float(__ushort_as_half((unsigned short)wa[16]));
  const float db = __half2float(__ushort_as_half((unsigned short)wb[16]));
  acc[0] += dy.x * da * (float)(sa0 * lo[0] + sa1 * hi[0]);
  acc[1] += dy.y * da * (float)(sa0 * lo[1] + sa1 * hi[1]);
  acc[2] += dy.x * db * (float)(sb0 * lo[2] + sb1 * hi[2]);
  acc[3] += dy.y * db * (float)(sb0 * lo[3] + sb1 * hi[3]);
}

template <typename T> struct TckBlocks;
template <> struct TckBlocks<TckQ4K> {
  static constexpr int SB_BYTES = sizeof(block_q4_K);
  template <int j>
  static __device__ __forceinline__ void
  run(const uint32_t (&wa)[TckQ4K::WORDS], const uint32_t (&wb)[TckQ4K::WORDS],
      const int2 bq, const float2 dy, const int2 xs, float (&acc)[4]) {
    tck_q4k_block<j>(wa, wb, bq, dy, xs, acc);
  }
};
template <> struct TckBlocks<TckQ6K> {
  static constexpr int SB_BYTES = sizeof(block_q6_K);
  template <int j>
  static __device__ __forceinline__ void
  run(const uint32_t (&wa)[TckQ6K::WORDS], const uint32_t (&wb)[TckQ6K::WORDS],
      const int2 bq, const float2 dy, const int2 xs, float (&acc)[4]) {
    tck_q6k_block<j>(wa, wb, bq, dy, xs, acc);
  }
};

template <typename Q, int GROUPS> struct TckStage {
  uint32_t a[GROUPS][Q::WORDS];
  uint32_t b[GROUPS][Q::WORDS];
};

template <typename Q, int j, int GROUPS>
static __device__ __forceinline__ void
tck_blocks(const TckStage<Q, GROUPS> &st, const int2 *yq, const float2 *yd,
           const int2 *ys, const int ib0, const int lane, const int t,
           float (&acc)[GROUPS][4]) {
  if constexpr (j < MMVQ_TCK_BLOCKS) {
    const int ib = ib0 + j;
    const int2 bq = yq[(int64_t)ib * WARP_SIZE + lane];
    const float2 dy = yd[(int64_t)ib * (WARP_SIZE / 8) + t];
    const int2 xs = ys[(int64_t)ib * (WARP_SIZE / 8) + t];
#pragma unroll
    for (int n = 0; n < GROUPS; ++n)
      TckBlocks<Q>::template run<j>(st.a[n], st.b[n], bq, dy, xs, acc[n]);
    tck_blocks<Q, j + 1, GROUPS>(st, yq, yd, ys, ib0, lane, t, acc);
  }
}

// KSPLIT warps splitting the superblocks, GROUPS 16-row groups per warp
template <typename Q, typename dst_t, int KSPLIT, int GROUPS, bool PREFETCH>
static __global__ void __launch_bounds__(KSPLIT *WARP_SIZE)
    mmvq_tc_k_kernel(const char *__restrict__ vx, const void *__restrict__ vy,
                     dst_t *__restrict__ dst, const int ncols_x,
                     const int nrows_x, const int stride_col_dst,
                     const int b_size) {
#if __CUDA_ARCH__ >= 800
  constexpr int SB_BYTES = TckBlocks<Q>::SB_BYTES;
  const int warp = threadIdx.y;
  const int lane = threadIdx.x;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int nb = ncols_x / QK8_1;
  const int nsb = ncols_x / QK_K;
  const int64_t row_bytes = (int64_t)nsb * SB_BYTES;
  const int rg0 = blockIdx.x * GROUPS;
  const int2 *yq = (const int2 *)vy;
  const float2 *yd = (const float2 *)(yq + (int64_t)nb * WARP_SIZE);
  const int2 *ys = (const int2 *)(yd + (int64_t)nb * (WARP_SIZE / 8));

  const char *ra[GROUPS], *rb[GROUPS];
#pragma unroll
  for (int n = 0; n < GROUPS; ++n) {
    const int r = (rg0 + n) * MMVQ_TCK_ROWS + g;
    ra[n] = vx + (int64_t)min(r, nrows_x - 1) * row_bytes;
    rb[n] = vx + (int64_t)min(r + 8, nrows_x - 1) * row_bytes;
  }
  const int chunk = (nsb + KSPLIT - 1) / KSPLIT;
  const int s0 = warp * chunk;
  const int ns = max(0, min(chunk, nsb - s0));

  auto load = [&](const int s, TckStage<Q, GROUPS> &st) {
#pragma unroll
    for (int n = 0; n < GROUPS; ++n) {
      Q::load(ra[n] + (int64_t)s * SB_BYTES, t, st.a[n]);
      Q::load(rb[n] + (int64_t)s * SB_BYTES, t, st.b[n]);
    }
  };

  float acc[GROUPS][4] = {};
  if constexpr (PREFETCH) {
    TckStage<Q, GROUPS> cur, nxt;
    if (ns > 0)
      load(s0, cur);
    for (int i = 0; i < ns; ++i) {
      if (i + 1 < ns)
        load(s0 + i + 1, nxt);
      tck_blocks<Q, 0, GROUPS>(cur, yq, yd, ys, (s0 + i) * MMVQ_TCK_BLOCKS,
                               lane, t, acc);
      cur = nxt;
    }
  } else {
    for (int i = 0; i < ns; ++i) {
      TckStage<Q, GROUPS> cur;
      load(s0 + i, cur);
      tck_blocks<Q, 0, GROUPS>(cur, yq, yd, ys, (s0 + i) * MMVQ_TCK_BLOCKS,
                               lane, t, acc);
    }
  }

  __shared__ float red[KSPLIT][GROUPS][4][WARP_SIZE];
#pragma unroll
  for (int n = 0; n < GROUPS; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k)
      red[warp][n][k][lane] = acc[n][k];
  __syncthreads();
  if (warp >= GROUPS)
    return;
  const int n = warp;
  float s[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    s[k] = 0.0f;
#pragma unroll
    for (int w2 = 0; w2 < KSPLIT; ++w2)
      s[k] += red[w2][n][k][lane];
  }
  const int r0 = (rg0 + n) * MMVQ_TCK_ROWS + g;
  const int r1 = r0 + 8;
  const int c0 = 2 * t;
  const int c1 = 2 * t + 1;
  if (r0 < nrows_x) {
    if (c0 < b_size)
      dst[c0 * stride_col_dst + r0] = (dst_t)s[0];
    if (c1 < b_size)
      dst[c1 * stride_col_dst + r0] = (dst_t)s[1];
  }
  if (r1 < nrows_x) {
    if (c0 < b_size)
      dst[c0 * stride_col_dst + r1] = (dst_t)s[2];
    if (c1 < b_size)
      dst[c1 * stride_col_dst + r1] = (dst_t)s[3];
  }
#else
  __trap();
#endif
}

template <typename Q, typename dst_t, int KSPLIT, int GROUPS,
          bool PREFETCH = true>
static void launch_mmvq_tc_k_cfg(const void *vx, const void *vy, void *dst,
                                 int ncols_x, int nrows_x, int stride_col_dst,
                                 int b_size, cudaStream_t stream) {
  const int ngroups = (nrows_x + MMVQ_TCK_ROWS - 1) / MMVQ_TCK_ROWS;
  const dim3 grid((ngroups + GROUPS - 1) / GROUPS, 1, 1);
  const dim3 block(WARP_SIZE, KSPLIT, 1);
  mmvq_tc_k_kernel<Q, dst_t, KSPLIT, GROUPS, PREFETCH>
      <<<grid, block, 0, stream>>>((const char *)vx, vy, (dst_t *)dst, ncols_x,
                                   nrows_x, stride_col_dst, b_size);
}

// (KSPLIT, GROUPS, PREFETCH) by `cfg`, which fast_mmvq picks per weight shape
// (RTX 3090: 8x1 with prefetch, 4x1 without on 4096-wide inputs)
template <typename Q, typename dst_t>
static void launch_mmvq_tc_k(const void *vx, const void *vy, void *dst,
                             int ncols_x, int nrows_x, int stride_col_dst,
                             int b_size, int cfg, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (cfg) {
  case 1:
    launch_mmvq_tc_k_cfg<Q, dst_t, 8, 1>(vx, vy, dst, ncols_x, nrows_x,
                                         stride_col_dst, b_size, s);
    break;
  case 2:
    launch_mmvq_tc_k_cfg<Q, dst_t, 4, 1, false>(vx, vy, dst, ncols_x, nrows_x,
                                                stride_col_dst, b_size, s);
    break;
  case 3:
    launch_mmvq_tc_k_cfg<Q, dst_t, 8, 1, false>(vx, vy, dst, ncols_x, nrows_x,
                                                stride_col_dst, b_size, s);
    break;
  default:
    launch_mmvq_tc_k_cfg<Q, dst_t, 4, 2>(vx, vy, dst, ncols_x, nrows_x,
                                         stride_col_dst, b_size, s);
    break;
  }
}

#define MMVQ_TC_K_LAUNCHER(type, Q, dst_tag, dst_c_type)                       \
  extern "C" void launch_mmvq_tc_gguf_##type##_##dst_tag##_plain(              \
      const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x,     \
      int stride_col_dst, int b_size, int cfg, void *stream) {                 \
    launch_mmvq_tc_k<Q, dst_c_type>(vx, vy, dst, ncols_x, nrows_x,             \
                                    stride_col_dst, b_size, cfg, stream);      \
  }

#define MMVQ_TC_K_LAUNCHERS(type, Q)                                           \
  MMVQ_TC_K_LAUNCHER(type, Q, bf16, __nv_bfloat16)                             \
  MMVQ_TC_K_LAUNCHER(type, Q, f16, half)                                       \
  MMVQ_TC_K_LAUNCHER(type, Q, f32, float)
