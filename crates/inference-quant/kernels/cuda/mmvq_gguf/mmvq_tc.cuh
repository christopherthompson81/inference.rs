// Tensor-core matvec for 2-8 input rows: a warp multiplies 16 weight rows by every input row's q8_1 activations
// with one int8 mma per 32 values, so a weight block is read and decoded once whatever the row count (sm_80+).
// A traits type supplies, for a group of 4 consecutive 32-value blocks (contiguous in a row), `tc_word`: word w of
// the group, whose block w/4's word w%4 holds values 4(w%4)..+3 (low nibbles) and 16+4(w%4)..+3 (high); `tc_scales`:
// the 4 blocks' scales and value tables (`tc_scale_word` loads them, `tc_scales_from` decodes them with the row's
// `tc_row_scale`, read once); and `tc_decode`: a word as int8s. The warp copies its 16 rows' groups into a private
// tile with row-contiguous loads, then each lane takes its words.
#pragma once
#include "mmvq_rows.cuh"

// weight rows one warp's mma covers
#define MMVQ_TC_ROWS 16
// consecutive 32-value blocks a warp reads per iteration (16 bytes each at 4 bits)
#define MMVQ_TC_BLOCKS 4
// 4-byte words of a row's group, a lane each
#define MMVQ_TC_GROUP_WORDS (MMVQ_TC_BLOCKS * 4)
// loads a lane makes per group to copy the warp's 16 rows
#define MMVQ_TC_COPIES (MMVQ_TC_ROWS * MMVQ_TC_GROUP_WORDS / WARP_SIZE)
// warps of a block, each taking every MMVQ_TC_NWARPS-th group of blocks of the same 16 rows
#define MMVQ_TC_NWARPS 8
// ints between tile rows, padded so a warp's word reads hit distinct banks
#define MMVQ_TC_TILE_STRIDE (MMVQ_TC_GROUP_WORDS + 4)

template <typename kt, typename dst_t>
static __global__ void __launch_bounds__(MMVQ_TC_NWARPS *WARP_SIZE)
    mmvq_tc_kernel(const void *__restrict__ vx, const void *__restrict__ vy, dst_t *__restrict__ dst,
                   const int ncols_x, const int nrows_x, const int stride_col_y, const int stride_col_dst,
                   const int b_size) {
#if __CUDA_ARCH__ >= 800
  const int lane = threadIdx.x;
  // the fragment row (and B column) this thread holds, and its slot in the row's four
  const int g = lane >> 2;
  const int t = lane & 3;
  const int row0 = blockIdx.x * MMVQ_TC_ROWS;
  const int64_t row_size = kt::row_size(ncols_x);
  // rows past the end read the last row; their results are not written
  // for the tile copy: this lane's word (and row, when a load spans several), and the row whose scales it reads
  const int copy_word = lane % MMVQ_TC_GROUP_WORDS;
  const int copy_row = lane / MMVQ_TC_GROUP_WORDS;
  const char *xs = (const char *)vx + (int64_t)min(row0 + lane % MMVQ_TC_ROWS, nrows_x - 1) * row_size;
  const float row_scale = kt::tc_row_scale(xs);
  // the activations in mmvq_tc_quantize's layout: each block's B fragments (an int2 a lane), then the rows' scales
  const int nb = ncols_x / QK8_1;
  const int2 *yq = (const int2 *)vy;
  const float2 *yd = (const float2 *)(yq + (int64_t)nb * WARP_SIZE);
  const int c0 = 2 * t;
  const int c1 = 2 * t + 1;

  __shared__ int tile_qs[MMVQ_TC_NWARPS][MMVQ_TC_ROWS][MMVQ_TC_TILE_STRIDE];
  // a row's block scales and tables, each written by its lane as vectors (conflict-free)
  __shared__ float4 tile_d[MMVQ_TC_NWARPS][MMVQ_TC_ROWS][MMVQ_TC_BLOCKS / 4];
  __shared__ int4 tile_table[MMVQ_TC_NWARPS][MMVQ_TC_ROWS][MMVQ_TC_BLOCKS / 4];
  int(*qs)[MMVQ_TC_TILE_STRIDE] = tile_qs[threadIdx.y];
  float(*ds)[MMVQ_TC_BLOCKS] = (float(*)[MMVQ_TC_BLOCKS])tile_d[threadIdx.y];
  int(*tables)[MMVQ_TC_BLOCKS] = (int(*)[MMVQ_TC_BLOCKS])tile_table[threadIdx.y];

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int ngroups = ncols_x / (QK8_1 * MMVQ_TC_BLOCKS);
  // no register double buffer: it costs occupancy, and the resident warps already overlap one another's loads
  for (int gi = threadIdx.y; gi < ngroups; gi += MMVQ_TC_NWARPS) {
    const int ib0 = gi * MMVQ_TC_BLOCKS;
#pragma unroll
    for (int i = 0; i < MMVQ_TC_COPIES; ++i) {
      const int row = i * (WARP_SIZE / MMVQ_TC_GROUP_WORDS) + copy_row;
      const char *x = (const char *)vx + (int64_t)min(row0 + row, nrows_x - 1) * row_size;
      qs[row][copy_word] = kt::tc_word(x, ib0, copy_word);
    }
    if (lane < MMVQ_TC_ROWS) {
#pragma unroll
      for (int q = 0; q < MMVQ_TC_BLOCKS / 4; ++q) {
        float d[4];
        int table[4];
        kt::tc_scales_from(kt::tc_scale_word(xs, ib0 + 4 * q), row_scale, d, table);
        ((float4 *)ds[lane])[q] = make_float4(d[0], d[1], d[2], d[3]);
        ((int4 *)tables[lane])[q] = make_int4(table[0], table[1], table[2], table[3]);
      }
    }
    __syncwarp();
#pragma unroll
    for (int w = 0; w < MMVQ_TC_BLOCKS; ++w) {
      const int ib = ib0 + w;
      const int2 va = kt::tc_decode(qs[g][4 * w + t], tables[g][w]);
      const int2 vb = kt::tc_decode(qs[g + 8][4 * w + t], tables[g + 8][w]);
      const int2 bq = yq[(int64_t)ib * WARP_SIZE + lane];
      const float2 dy = yd[(int64_t)ib * (WARP_SIZE / 8) + t];
      int c[4] = {0, 0, 0, 0};
      asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
          "{%0, %1, %2, %3};"
          : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
          : "r"(va.x), "r"(vb.x), "r"(va.y), "r"(vb.y), "r"(bq.x), "r"(bq.y));
      const float d0 = dy.x;
      const float d1 = dy.y;
      const float da = ds[g][w];
      const float db = ds[g + 8][w];
      acc[0] += da * d0 * (float)c[0];
      acc[1] += da * d1 * (float)c[1];
      acc[2] += db * d0 * (float)c[2];
      acc[3] += db * d1 * (float)c[3];
    }
    __syncwarp();
  }

  __shared__ float partial[MMVQ_TC_NWARPS][4][WARP_SIZE];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    partial[threadIdx.y][i][lane] = acc[i];
  }
  __syncthreads();
  if (threadIdx.y != 0) {
    return;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < MMVQ_TC_NWARPS; ++w) {
      sum += partial[w][i][lane];
    }
    acc[i] = sum;
  }
  const int ra = row0 + g;
  const int rb = row0 + g + 8;
  if (ra < nrows_x) {
    if (c0 < b_size) dst[c0 * stride_col_dst + ra] = from_float<dst_t>(acc[0]);
    if (c1 < b_size) dst[c1 * stride_col_dst + ra] = from_float<dst_t>(acc[1]);
  }
  if (rb < nrows_x) {
    if (c0 < b_size) dst[c0 * stride_col_dst + rb] = from_float<dst_t>(acc[2]);
    if (c1 < b_size) dst[c1 * stride_col_dst + rb] = from_float<dst_t>(acc[3]);
  }
#else
  __trap();
#endif
}

template <typename kt, typename dst_t>
static void launch_mmvq_tc(const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x, int stride_col_y,
                           int stride_col_dst, int b_size, void *stream) {
  const dim3 grid((nrows_x + MMVQ_TC_ROWS - 1) / MMVQ_TC_ROWS, 1, 1);
  const dim3 block(WARP_SIZE, MMVQ_TC_NWARPS, 1);
  mmvq_tc_kernel<kt, dst_t><<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
      vx, vy, (dst_t *)dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, b_size);
}

#define MMVQ_TC_LAUNCHER(tag, traits, dst_tag, dst_c_type)                                                      \
  extern "C" void launch_mmvq_tc_gguf_##tag##_##dst_tag##_plain(const void *vx, const void *vy, void *dst,       \
                                                                int ncols_x, int nrows_x, int stride_col_y,     \
                                                                int stride_col_dst, int b_size, void *stream) { \
    launch_mmvq_tc<traits, dst_c_type>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, b_size,     \
                                       stream);                                                                 \
  }

#define MMVQ_TC_LAUNCHERS(tag, traits)                    \
  MMVQ_TC_LAUNCHER(tag, traits, bf16, __nv_bfloat16)      \
  MMVQ_TC_LAUNCHER(tag, traits, f16, half)                \
  MMVQ_TC_LAUNCHER(tag, traits, f32, float)
