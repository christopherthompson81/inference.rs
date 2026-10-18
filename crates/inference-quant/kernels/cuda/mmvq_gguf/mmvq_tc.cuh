// Tensor-core matvec for 1-8 input rows over a tile-ordered weight: a warp multiplies 16 weight rows by every
// input row's int8 activations (mmvq_tc_quantize) with one mma per 32 values, so each weight tile is read and
// decoded once whatever the row count (sm_80+). After ExLlamaV3's exl3_gemv_kernel.

#include "mmvq_rows.cuh"

// weight rows one warp's mma covers
#define MMVQ_TC_ROWS 16

// Tile-ordered IQ4_KS (`iq4_ks_tiled`), the bytes of 16 rows regrouped so a warp reads the mma's A fragments with
// coalesced word loads: per 16-row group, the rows' f32 scales (64 bytes), then per 256-value superblock its block
// scales as [block][row] bytes (128) and per 32-value block two 128-byte halves, word (g * 4 + t) of half h holding
// row h * 8 + g's bytes 4t..4t+3 of the block (values 4t..4t+3 low nibbles, 16+4t..16+4t+3 high).
#define MMVQ_TC_GROUP_HEADER 64
#define MMVQ_TC_SB_BYTES (128 + 8 * 256)
// iq4k_values' lower half packed four bytes a word; the upper half is each value plus 4, which the kernel applies as
// 4 x the block's activation sum instead of a second table
static constexpr __host__ __device__ uint32_t tc_pack4(int a, int b, int c, int d) {
  return (uint32_t)(uint8_t)a | (uint32_t)(uint8_t)b << 8 | (uint32_t)(uint8_t)c << 16 | (uint32_t)(uint8_t)d << 24;
}
#define MMVQ_TC_IQ4K_LO0 tc_pack4(-127, -104, -83, -65)
#define MMVQ_TC_IQ4K_LO1 tc_pack4(-49, -35, -22, -10)
#define MMVQ_TC_IQ4K_LO2 tc_pack4(1, 13, 25, 38)
#define MMVQ_TC_IQ4K_LO3 tc_pack4(53, 69, 89, 113)
#define MMVQ_TC_IQ4K_HI_OFFSET 4

// get_int_from_table_16 with the table in immediates: eight nibbles to eight int8s
static __device__ __forceinline__ int2 tc_decode_iq4k_lo(uint32_t q4) {
  const uint32_t mask = 0x32103210 | ((q4 & 0x88888888) >> 1);
  const uint32_t a3 = __byte_perm(__byte_perm(MMVQ_TC_IQ4K_LO0, MMVQ_TC_IQ4K_LO1, q4),
                                  __byte_perm(MMVQ_TC_IQ4K_LO2, MMVQ_TC_IQ4K_LO3, q4), mask);
  const uint32_t a4 = __byte_perm(__byte_perm(MMVQ_TC_IQ4K_LO0, MMVQ_TC_IQ4K_LO1, q4 >> 16),
                                  __byte_perm(MMVQ_TC_IQ4K_LO2, MMVQ_TC_IQ4K_LO3, q4 >> 16), mask >> 16);
  return make_int2(__byte_perm(a3, a4, 0x6420), __byte_perm(a3, a4, 0x7531));
}

// KSPLIT warps splitting K, GROUPS row groups per warp, PREFETCH blocks prefetched ahead
template <typename dst_t, int KSPLIT, int GROUPS, int PREFETCH>
static __global__ void __launch_bounds__(KSPLIT *WARP_SIZE)
    mmvq_tc_iq4_ks_kernel(const char *__restrict__ vx, const void *__restrict__ vy, dst_t *__restrict__ dst,
                           const int ncols_x, const int nrows_x, const int stride_col_dst, const int b_size) {
#if __CUDA_ARCH__ >= 800
  const int warp = threadIdx.y;
  const int lane = threadIdx.x;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int nb = ncols_x / QK8_1;
  const int nsb = ncols_x / QK_K;
  const int64_t group_bytes = MMVQ_TC_GROUP_HEADER + (int64_t)nsb * MMVQ_TC_SB_BYTES;
  const int ngroups = (nrows_x + MMVQ_TC_ROWS - 1) / MMVQ_TC_ROWS;
  const int rg0 = blockIdx.x * GROUPS;
  const int2 *yq = (const int2 *)vy;
  const float2 *yd = (const float2 *)(yq + (int64_t)nb * WARP_SIZE);
  const int2 *ys = (const int2 *)(yd + (int64_t)nb * (WARP_SIZE / 8));

  const char *gp[GROUPS];
  float sa[GROUPS], sb[GROUPS];
#pragma unroll
  for (int n = 0; n < GROUPS; ++n) {
    gp[n] = vx + (int64_t)min(rg0 + n, ngroups - 1) * group_bytes;
    sa[n] = ((const float *)gp[n])[g];
    sb[n] = ((const float *)gp[n])[g + 8];
  }
  const int chunk = (nb + KSPLIT - 1) / KSPLIT;
  const int i0 = warp * chunk;
  const int n_blocks = max(0, min(chunk, nb - i0));

  // block ib's three words for every row group: the two A halves and the scale word (rows 4t..4t+3); the offsets
  // within a group are the same for all groups and fit 32 bits, so they are computed once per block
  auto load_groups = [&](int ib, uint32_t (&words)[GROUPS][3]) {
    const uint32_t sb = (uint32_t)ib >> 3;
    const uint32_t j = (uint32_t)ib & 7;
    const uint32_t q_off = MMVQ_TC_GROUP_HEADER + 128 + (uint32_t)ib * 256 + sb * 128 + lane * 4;
    const uint32_t s_off = MMVQ_TC_GROUP_HEADER + sb * MMVQ_TC_SB_BYTES + (j * 4 + t) * 4;
#pragma unroll
    for (int n = 0; n < GROUPS; ++n) {
      words[n][0] = __ldcs((const uint32_t *)(gp[n] + q_off));
      words[n][1] = __ldcs((const uint32_t *)(gp[n] + q_off + 128));
      words[n][2] = __ldg((const uint32_t *)(gp[n] + s_off));
    }
  };

  // block ib's activations, the same for every row group: B fragment, row scales and row sums
  struct Act {
    int2 bq;
    float2 dy;
    int2 xs;
  };
  auto load_act = [&](int ib, Act &a) {
    a.bq = yq[(int64_t)ib * WARP_SIZE + lane];
    a.dy = yd[(int64_t)ib * (WARP_SIZE / 8) + t];
    a.xs = ys[(int64_t)ib * (WARP_SIZE / 8) + t];
  };

  uint32_t pf[PREFETCH][GROUPS][3];
  Act pa[PREFETCH];
#pragma unroll
  for (int d = 0; d < PREFETCH; ++d)
    if (d < n_blocks) {
      load_groups(i0 + d, pf[d]);
      load_act(i0 + d, pa[d]);
    }

  float acc[GROUPS][4] = {};
  for (int ib0 = 0; ib0 < n_blocks; ib0 += PREFETCH) {
#pragma unroll
    for (int d = 0; d < PREFETCH; ++d) {
      const int i = ib0 + d;
      if (i >= n_blocks) break;
      // used straight from the ring slot, which is refilled once this block's math is issued
      uint32_t (&w)[GROUPS][3] = pf[d];
      const int2 bq = pa[d].bq;
      const float2 dy = pa[d].dy;
      const int2 xs = pa[d].xs;
#pragma unroll
      for (int n = 0; n < GROUPS; ++n) {
        // the scale bytes of rows g and g + 8 sit in the words lanes with t = g / 4 and (g + 8) / 4 loaded
        const uint32_t wa = __shfl_sync(0xffffffffu, w[n][2], g >> 2);
        const uint32_t wb = __shfl_sync(0xffffffffu, w[n][2], (g + 8) >> 2);
        const int sca = (wa >> (8 * (g & 3))) & 0xff;
        const int scb = (wb >> (8 * (g & 3))) & 0xff;
        const int2 va = tc_decode_iq4k_lo(w[n][0]);
        const int2 vb = tc_decode_iq4k_lo(w[n][1]);
        const int hia = (sca & 1) * MMVQ_TC_IQ4K_HI_OFFSET;
        const int hib = (scb & 1) * MMVQ_TC_IQ4K_HI_OFFSET;
        int c[4] = {hia * xs.x, hia * xs.y, hib * xs.x, hib * xs.y};
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
            : "r"(va.x), "r"(vb.x), "r"(va.y), "r"(vb.y), "r"(bq.x), "r"(bq.y));
        const float da = sa[n] * ((sca & 254) - 127);
        const float db = sb[n] * ((scb & 254) - 127);
        acc[n][0] += da * dy.x * (float)c[0];
        acc[n][1] += da * dy.y * (float)c[1];
        acc[n][2] += db * dy.x * (float)c[2];
        acc[n][3] += db * dy.y * (float)c[3];
      }
      if (i + PREFETCH < n_blocks) {
        load_groups(i0 + i + PREFETCH, pf[d]);
        load_act(i0 + i + PREFETCH, pa[d]);
      }
    }
  }

  __shared__ float red[KSPLIT][GROUPS][4][WARP_SIZE];
#pragma unroll
  for (int n = 0; n < GROUPS; ++n)
#pragma unroll
    for (int k = 0; k < 4; ++k) red[warp][n][k][lane] = acc[n][k];
  __syncthreads();
  // warp n of the first GROUPS sums row group n
  if (warp >= GROUPS) return;
  const int n = warp;
  float s[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    s[k] = 0.0f;
#pragma unroll
    for (int w2 = 0; w2 < KSPLIT; ++w2) s[k] += red[w2][n][k][lane];
  }
  const int ra = (rg0 + n) * MMVQ_TC_ROWS + g;
  const int rb = ra + 8;
  const int c0 = 2 * t;
  const int c1 = 2 * t + 1;
  if (ra < nrows_x) {
    if (c0 < b_size) dst[c0 * stride_col_dst + ra] = from_float<dst_t>(s[0]);
    if (c1 < b_size) dst[c1 * stride_col_dst + ra] = from_float<dst_t>(s[1]);
  }
  if (rb < nrows_x) {
    if (c0 < b_size) dst[c0 * stride_col_dst + rb] = from_float<dst_t>(s[2]);
    if (c1 < b_size) dst[c1 * stride_col_dst + rb] = from_float<dst_t>(s[3]);
  }
#else
  __trap();
#endif
}

template <typename dst_t, int KSPLIT, int GROUPS, int PREFETCH>
static void launch_mmvq_tc_cfg(const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x,
                                int stride_col_dst, int b_size, cudaStream_t stream) {
  const int ngroups = (nrows_x + MMVQ_TC_ROWS - 1) / MMVQ_TC_ROWS;
  const dim3 grid((ngroups + GROUPS - 1) / GROUPS, 1, 1);
  const dim3 block(WARP_SIZE, KSPLIT, 1);
  mmvq_tc_iq4_ks_kernel<dst_t, KSPLIT, GROUPS, PREFETCH><<<grid, block, 0, stream>>>(
      (const char *)vx, vy, (dst_t *)dst, ncols_x, nrows_x, stride_col_dst, b_size);
}

// The (KSPLIT, GROUPS, PREFETCH) shapes fast_mmvq times per weight shape at load: wide groups win on tall weights, one
// group (fewer registers, 3 blocks per SM) on short ones (RTX 3090: FFN 8x2x4, 6144x5120 8x1x4)
template <typename dst_t>
static void launch_mmvq_tc_iq4_ks(const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x,
                                   int stride_col_dst, int b_size, int cfg, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (cfg) {
  case 1: launch_mmvq_tc_cfg<dst_t, 8, 1, 6>(vx, vy, dst, ncols_x, nrows_x, stride_col_dst, b_size, s); break;
  case 2: launch_mmvq_tc_cfg<dst_t, 8, 4, 2>(vx, vy, dst, ncols_x, nrows_x, stride_col_dst, b_size, s); break;
  case 3: launch_mmvq_tc_cfg<dst_t, 8, 1, 4>(vx, vy, dst, ncols_x, nrows_x, stride_col_dst, b_size, s); break;
  default: launch_mmvq_tc_cfg<dst_t, 8, 2, 4>(vx, vy, dst, ncols_x, nrows_x, stride_col_dst, b_size, s); break;
  }
}

#define MMVQ_TC_IQ4_KS_LAUNCHER(dst_tag, dst_c_type)                                                          \
  extern "C" void launch_mmvq_tc_gguf_iq4_ks_##dst_tag##_plain(const void *vx, const void *vy, void *dst,       \
                                                                 int ncols_x, int nrows_x, int stride_col_dst,  \
                                                                 int b_size, int cfg, void *stream) {           \
    launch_mmvq_tc_iq4_ks<dst_c_type>(vx, vy, dst, ncols_x, nrows_x, stride_col_dst, b_size, cfg, stream);     \
  }

// Rows `ids` of a tile-ordered IQ4_KS weight back in row-major order, a thread per (row, superblock)
static __global__ void iq4_ks_untile_kernel(const uint8_t *__restrict__ src, const uint32_t *__restrict__ ids,
                                            uint8_t *__restrict__ dst, int n, int nsb) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (int64_t)n * nsb) return;
  const int out = (int)(idx / nsb);
  const int sb = (int)(idx - (int64_t)out * nsb);
  const int row = (int)ids[out];
  const int r = row % MMVQ_TC_ROWS;
  const int64_t row_bytes = sizeof(float) + (int64_t)nsb * sizeof(block_iq4_ks);
  const int64_t group = (int64_t)(row / MMVQ_TC_ROWS) * (MMVQ_TC_GROUP_HEADER + (int64_t)nsb * MMVQ_TC_SB_BYTES);
  const int64_t sbase = group + MMVQ_TC_GROUP_HEADER + (int64_t)sb * MMVQ_TC_SB_BYTES;
  uint8_t *d = dst + out * row_bytes;
  if (sb == 0) *(uint32_t *)d = *(const uint32_t *)(src + group + 4 * r);
  uint8_t *blk = d + sizeof(float) + (int64_t)sb * sizeof(block_iq4_ks);
  for (int j = 0; j < 8; ++j) {
    blk[j] = src[sbase + j * 16 + r];
    const int64_t q = sbase + 128 + j * 256 + (r / 8) * 128 + (r % 8) * 16;
    for (int t = 0; t < 4; ++t) *(uint32_t *)(blk + 8 + 16 * j + 4 * t) = *(const uint32_t *)(src + q + 4 * t);
  }
}

extern "C" void launch_iq4_ks_untile_rows(const void *src, const void *ids, void *dst, int64_t n, int64_t ncols,
                                          void *stream) {
  const int nsb = (int)(ncols / QK_K);
  const int64_t m = n * nsb;
  if (m == 0) return;
  iq4_ks_untile_kernel<<<(unsigned)((m + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      (const uint8_t *)src, (const uint32_t *)ids, (uint8_t *)dst, (int)n, nsb);
}
