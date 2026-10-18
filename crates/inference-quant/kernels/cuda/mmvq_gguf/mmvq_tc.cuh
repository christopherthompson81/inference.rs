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

  // block ib's three words for row group n: the two A halves and the scale word (rows 4t..4t+3)
  auto load = [&](int ib, int n, uint32_t &qa, uint32_t &qb, uint32_t &sw) {
    const char *s = gp[n] + MMVQ_TC_GROUP_HEADER + (int64_t)(ib >> 3) * MMVQ_TC_SB_BYTES;
    const int j = ib & 7;
    const uint32_t *q = (const uint32_t *)(s + 128 + j * 256);
    qa = __ldcs(q + lane);
    qb = __ldcs(q + 32 + lane);
    sw = __ldg((const uint32_t *)s + j * 4 + t);
  };

  uint32_t pf[PREFETCH][GROUPS][3];
#pragma unroll
  for (int d = 0; d < PREFETCH; ++d)
    if (d < n_blocks)
#pragma unroll
      for (int n = 0; n < GROUPS; ++n) load(i0 + d, n, pf[d][n][0], pf[d][n][1], pf[d][n][2]);

  float acc[GROUPS][4] = {};
  for (int ib0 = 0; ib0 < n_blocks; ib0 += PREFETCH) {
#pragma unroll
    for (int d = 0; d < PREFETCH; ++d) {
      const int i = ib0 + d;
      if (i >= n_blocks) break;
      uint32_t w[GROUPS][3];
#pragma unroll
      for (int n = 0; n < GROUPS; ++n) {
        w[n][0] = pf[d][n][0];
        w[n][1] = pf[d][n][1];
        w[n][2] = pf[d][n][2];
      }
      if (i + PREFETCH < n_blocks)
#pragma unroll
        for (int n = 0; n < GROUPS; ++n) load(i0 + i + PREFETCH, n, pf[d][n][0], pf[d][n][1], pf[d][n][2]);
      const int ib = i0 + i;
      const int2 bq = yq[(int64_t)ib * WARP_SIZE + lane];
      const float2 dy = yd[(int64_t)ib * (WARP_SIZE / 8) + t];
#pragma unroll
      for (int n = 0; n < GROUPS; ++n) {
        // the scale bytes of rows g and g + 8 sit in the words lanes with t = g / 4 and (g + 8) / 4 loaded
        const uint32_t wa = __shfl_sync(0xffffffffu, w[n][2], g >> 2);
        const uint32_t wb = __shfl_sync(0xffffffffu, w[n][2], (g + 8) >> 2);
        const int sca = (wa >> (8 * (g & 3))) & 0xff;
        const int scb = (wb >> (8 * (g & 3))) & 0xff;
        const int2 va = get_int_from_table_16(w[n][0], iq4k_values + ((sca & 1) << 4));
        const int2 vb = get_int_from_table_16(w[n][1], iq4k_values + ((scb & 1) << 4));
        int c[4] = {0, 0, 0, 0};
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

// The (KSPLIT, GROUPS, PREFETCH) shapes fast_mmvq times per weight shape at load: which wins moves with the row count and K
// (5120x17408 16x1x4, 10240x5120 8x4x2, 17408x5120 8x1x4 on an RTX 3090)
template <typename dst_t>
static void launch_mmvq_tc_iq4_ks(const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x,
                                   int stride_col_dst, int b_size, int cfg, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (cfg) {
  case 1: launch_mmvq_tc_cfg<dst_t, 16, 1, 4>(vx, vy, dst, ncols_x, nrows_x, stride_col_dst, b_size, s); break;
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
