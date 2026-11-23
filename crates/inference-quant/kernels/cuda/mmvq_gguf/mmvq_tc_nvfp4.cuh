// Tensor-core MoE matvec for NVFP4 experts in nvfp4_tiled.rs's tile order, for one token's experts up to a few
// tokens': a block is one distinct expert's 16-row group (a slot of mmvq_moe_group_pairs's plan, or with no plan
// pair blockIdx.y alone) and every pair routed to it, up to 8 pairs as the columns of an int8 mma.m16n8k16 per
// 16-value sub-block, so each weight is read and decoded once for all of them (sm_80+). A lane's fragment is one
// 16-byte load and its rows' scales one 8-byte load, tiles loaded ahead of the math. Each sub-block's UE4M3 scale and
// each q8_1 block's activation scale apply to the mma's int32 sums, as vec_dot_nvfp4_q8_1 applies them to its own.
#pragma once
#include "mmvq_gguf.cuh"

// weight rows one tile and mma cover, and the pairs (columns) an mma takes
#define MMVQ_TC_NVFP4_ROWS 16
#define MMVQ_TC_NVFP4_COLS 8
// warps of a block, splitting K: the fused gate and up projection has half the row groups of the down projection it
// pairs with and twice its K, so twice the warps a group keep as many loads in flight
#define MMVQ_TC_NVFP4_KSPLIT 4
#define MMVQ_TC_NVFP4_KSPLIT_GLU 6
// tiles loaded ahead of the one the math is on
#define MMVQ_TC_NVFP4_AHEAD 2
#define MMVQ_TC_NVFP4_TILE_BYTES 576
#define MMVQ_TC_NVFP4_SCALE_BYTES 64
#define MMVQ_TC_NVFP4_SUBS (QK_NVFP4 / QK_NVFP4_SUB)

// kvalues_fp4 four bytes a word, in immediates rather than a table in memory
static constexpr __host__ __device__ uint32_t tc_nvfp4_pack4(int a, int b, int c, int d) {
  return (uint32_t)(uint8_t)a | (uint32_t)(uint8_t)b << 8 | (uint32_t)(uint8_t)c << 16 | (uint32_t)(uint8_t)d << 24;
}
#define MMVQ_TC_FP4_0 tc_nvfp4_pack4(0, 1, 2, 3)
#define MMVQ_TC_FP4_1 tc_nvfp4_pack4(4, 6, 8, 12)
#define MMVQ_TC_FP4_2 tc_nvfp4_pack4(0, -1, -2, -3)
#define MMVQ_TC_FP4_3 tc_nvfp4_pack4(-4, -6, -8, -12)

// get_int_from_table_16 over kvalues_fp4: a tile word's low nibbles (row g's four values) and high (row g + 8's)
static __device__ __forceinline__ int2 tc_nvfp4_rows(const uint32_t q4) {
  const uint32_t mask = 0x32103210 | ((q4 & 0x88888888) >> 1);
  const uint32_t a3 = __byte_perm(__byte_perm(MMVQ_TC_FP4_0, MMVQ_TC_FP4_1, q4),
                                  __byte_perm(MMVQ_TC_FP4_2, MMVQ_TC_FP4_3, q4), mask);
  const uint32_t a4 = __byte_perm(__byte_perm(MMVQ_TC_FP4_0, MMVQ_TC_FP4_1, q4 >> 16),
                                  __byte_perm(MMVQ_TC_FP4_2, MMVQ_TC_FP4_3, q4 >> 16), mask >> 16);
  return make_int2((int)__byte_perm(a3, a4, 0x6420), (int)__byte_perm(a3, a4, 0x7531));
}

template <int nsums> struct TcNvfp4Tile {
  uint4 q[nsums];
  uint2 sc[nsums];
};

// UE4M3 scale bytes as ggml_cuda_ue4m3_to_fp32 reads them (bit 7 ignored), a table each block builds in shared memory
#define MMVQ_TC_NVFP4_UE4M3_VALUES 128
// the dynamic shared memory a launch may take without opting in
#define MMVQ_TC_NVFP4_SMEM_MAX (48 * 1024)

// Shared memory a block takes for `cols` columns of `ncols_x`-long activations: per column and tile the four t lanes'
// B fragments (an int4 of sub-blocks 0-3 each), per tile and column the two q8_1 blocks' scales, and the UE4M3 table
static __host__ __device__ size_t tc_nvfp4_smem(const int cols, const int ncols_x) {
  const int nb = ncols_x / QK_NVFP4;
  return (size_t)cols * nb * 4 * sizeof(int4) + (size_t)nb * cols * sizeof(float2) +
         MMVQ_TC_NVFP4_UE4M3_VALUES * sizeof(float);
}

template <bool glu, int KSPLIT>
static __global__ void __launch_bounds__(KSPLIT *WARP_SIZE)
    mmvq_tc_nvfp4_moe_kernel(const char *__restrict__ vx_gate, const char *__restrict__ vx_up,
                             const int64_t expert_stride, const block_q8_1 *__restrict__ y,
                             const uint32_t *__restrict__ ids, const int *__restrict__ groups, const int n_pairs,
                             const int cols, const float *__restrict__ scales, const float *__restrict__ scales_up,
                             float *__restrict__ dst, const int ncols_x, const int nrows_x, const int stride_col_y,
                             const int y_div, const int activation, const float gate_cap, const float up_cap) {
#if __CUDA_ARCH__ >= 800
  constexpr int nsums = glu ? 2 : 1;
  const int slot = blockIdx.y;
  int expert, first, last;
  const int *pair_list = nullptr;
  if (groups) {
    if (slot >= groups[0]) {
      return;
    }
    expert = groups[1 + slot];
    first = groups[1 + n_pairs + slot];
    last = groups[2 + n_pairs + slot];
    pair_list = groups + 2 + 2 * n_pairs;
  } else {
    expert = (int)ids[slot];
    first = slot;
    last = slot + 1;
  }
  auto pair_at = [&](const int i) { return pair_list ? pair_list[i] : i; };

  const int warp = threadIdx.y;
  const int lane = threadIdx.x;
  const int tid = warp * WARP_SIZE + lane;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int nb = ncols_x / QK_NVFP4;
  const int64_t group_bytes = (int64_t)nb * MMVQ_TC_NVFP4_TILE_BYTES;
  const int ra = blockIdx.x * MMVQ_TC_NVFP4_ROWS + g;
  const int rb = ra + 8;
  const char *base[nsums];
  base[0] = vx_gate + expert * expert_stride + blockIdx.x * group_bytes;
  if constexpr (glu) {
    base[1] = vx_up + expert * expert_stride + blockIdx.x * group_bytes;
  }

  extern __shared__ int4 tc_nvfp4_smem_base[];
  int4 *s_b = tc_nvfp4_smem_base;
  float2 *s_d = (float2 *)(s_b + cols * nb * 4);
  float *s_ue4m3 = (float *)(s_d + nb * cols);
  for (int i = tid; i < MMVQ_TC_NVFP4_UE4M3_VALUES; i += KSPLIT * WARP_SIZE) {
    s_ue4m3[i] = ggml_cuda_ue4m3_to_fp32((uint8_t)i);
  }

  const int chunk = (nb + KSPLIT - 1) / KSPLIT;
  const int kb0 = warp * chunk;
  const int kb1 = min(nb, kb0 + chunk);

  auto load = [&](const int kb, TcNvfp4Tile<nsums> &tile) {
#pragma unroll
    for (int s = 0; s < nsums; ++s) {
      const char *p = base[s] + (int64_t)kb * MMVQ_TC_NVFP4_TILE_BYTES;
      tile.sc[s] = __ldg((const uint2 *)p + g);
      tile.q[s] = __ldcs((const uint4 *)(p + MMVQ_TC_NVFP4_SCALE_BYTES) + lane);
    }
  };

  __shared__ float red[KSPLIT][nsums][4][WARP_SIZE];

  for (int at = first; at < last; at += cols) {
    const int count = min(cols, last - at);
    // the weights stream in while this chunk's activations are staged
    TcNvfp4Tile<nsums> ring[MMVQ_TC_NVFP4_AHEAD];
#pragma unroll
    for (int d = 0; d < MMVQ_TC_NVFP4_AHEAD; ++d) {
      if (kb0 + d < kb1) {
        load(kb0 + d, ring[d]);
      }
    }
    // column c's B fragments: lane t's ints of sub-blocks 0-3 (q8_1 block 2 kb + is / 2, int (is % 2) * 4 + t)
    for (int i = tid; i < count * nb * 4; i += KSPLIT * WARP_SIZE) {
      const int c = i / (nb * 4);
      const int kb = (i / 4) % nb;
      const int tt = i % 4;
      const block_q8_1 *yc = y + (int64_t)(pair_at(at + c) / y_div) * stride_col_y + 2 * kb;
      s_b[i] = make_int4(get_int_b4(yc[0].qs, tt), get_int_b4(yc[0].qs, 4 + tt), get_int_b4(yc[1].qs, tt),
                         get_int_b4(yc[1].qs, 4 + tt));
    }
    for (int i = tid; i < count * nb; i += KSPLIT * WARP_SIZE) {
      const int kb = i / count;
      const int c = i % count;
      const block_q8_1 *yc = y + (int64_t)(pair_at(at + c) / y_div) * stride_col_y + 2 * kb;
      s_d[i] = make_float2(__low2float(yc[0].ds), __low2float(yc[1].ds));
    }
    __syncthreads();

    // this lane's B column g and accumulator columns 2t, 2t + 1, repeating the last column past count
    const int cb = min(g, count - 1);
    const int c0 = min(2 * t, count - 1);
    const int c1 = min(2 * t + 1, count - 1);
    float acc[nsums][4] = {};
    for (int kb00 = kb0; kb00 < kb1; kb00 += MMVQ_TC_NVFP4_AHEAD) {
#pragma unroll
      for (int d = 0; d < MMVQ_TC_NVFP4_AHEAD; ++d) {
        const int kb = kb00 + d;
        if (kb >= kb1) {
          break;
        }
        const TcNvfp4Tile<nsums> cur = ring[d];
        if (kb + MMVQ_TC_NVFP4_AHEAD < kb1) {
          load(kb + MMVQ_TC_NVFP4_AHEAD, ring[d]);
        }
        const int4 bq = s_b[(cb * nb + kb) * 4 + t];
        const float2 d0 = s_d[kb * count + c0];
        const float2 d1 = s_d[kb * count + c1];
#pragma unroll
        for (int is = 0; is < MMVQ_TC_NVFP4_SUBS; ++is) {
          const int b0 = is == 0 ? bq.x : is == 1 ? bq.y : is == 2 ? bq.z : bq.w;
          const float dy0 = is < 2 ? d0.x : d0.y;
          const float dy1 = is < 2 ? d1.x : d1.y;
#pragma unroll
          for (int s = 0; s < nsums; ++s) {
            const uint32_t word = is == 0 ? cur.q[s].x : is == 1 ? cur.q[s].y : is == 2 ? cur.q[s].z : cur.q[s].w;
            const int2 a = tc_nvfp4_rows(word);
            int c[4] = {0, 0, 0, 0};
            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                "{%0, %1, %2, %3};"
                : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                : "r"(a.x), "r"(a.y), "r"(b0));
            const float sa = s_ue4m3[(cur.sc[s].x >> (8 * is)) & 0x7F];
            const float sb = s_ue4m3[(cur.sc[s].y >> (8 * is)) & 0x7F];
            acc[s][0] += sa * dy0 * (float)c[0];
            acc[s][1] += sa * dy1 * (float)c[1];
            acc[s][2] += sb * dy0 * (float)c[2];
            acc[s][3] += sb * dy1 * (float)c[3];
          }
        }
      }
    }

#pragma unroll
    for (int s = 0; s < nsums; ++s)
#pragma unroll
      for (int k = 0; k < 4; ++k) red[warp][s][k][lane] = acc[s][k];
    __syncthreads();
    if (warp == 0) {
      float sum[nsums][4];
#pragma unroll
      for (int s = 0; s < nsums; ++s)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          sum[s][k] = 0.0f;
#pragma unroll
          for (int w2 = 0; w2 < KSPLIT; ++w2) sum[s][k] += red[w2][s][k][lane];
        }
      const float gate_scale = scales ? scales[expert] : 1.0f;
      const float up_scale = scales_up ? scales_up[expert] : 1.0f;
      const int p0 = pair_at(at + c0);
      const int p1 = pair_at(at + c1);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int col = 2 * t + (k & 1);
        const int row = (k < 2) ? ra : rb;
        if (col < count && row < nrows_x) {
          float out = sum[0][k] * gate_scale;
          if constexpr (glu) {
            out = apply_glu(out, sum[nsums - 1][k] * up_scale, activation, gate_cap, up_cap);
          }
          dst[(int64_t)((k & 1) ? p1 : p0) * nrows_x + row] = out;
        }
      }
    }
    __syncthreads();
  }
#else
  __trap();
#endif
}

// `groups` null runs each of `n_pairs` pairs on its own expert (one token's routes, which are distinct)
extern "C" void launch_mmvq_tc_nvfp4_moe(const void *vx_gate, const void *vx_up, int64_t expert_stride,
                                         const void *vy, const void *ids, const int *groups, int n_pairs,
                                         const float *scales, const float *scales_up, float *dst, int ncols_x,
                                         int nrows_x, int stride_col_y, int y_div, int activation, float gate_cap,
                                         float up_cap, void *stream) {
  const dim3 grid((nrows_x + MMVQ_TC_NVFP4_ROWS - 1) / MMVQ_TC_NVFP4_ROWS, n_pairs, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const uint32_t *id = (const uint32_t *)ids;
  // fewer columns a pass when a wide K's staged activations would pass the default shared memory a block takes
  int cols = groups ? min(MMVQ_TC_NVFP4_COLS, n_pairs) : 1;
  while (cols > 1 && tc_nvfp4_smem(cols, ncols_x) > MMVQ_TC_NVFP4_SMEM_MAX) {
    --cols;
  }
  const size_t smem = tc_nvfp4_smem(cols, ncols_x);
  if (vx_up) {
    const dim3 block(WARP_SIZE, MMVQ_TC_NVFP4_KSPLIT_GLU, 1);
    mmvq_tc_nvfp4_moe_kernel<true, MMVQ_TC_NVFP4_KSPLIT_GLU><<<grid, block, smem, s>>>(
        (const char *)vx_gate, (const char *)vx_up, expert_stride, (const block_q8_1 *)vy, id, groups, n_pairs,
        cols, scales, scales_up, dst, ncols_x, nrows_x, stride_col_y, y_div, activation, gate_cap, up_cap);
  } else {
    const dim3 block(WARP_SIZE, MMVQ_TC_NVFP4_KSPLIT, 1);
    mmvq_tc_nvfp4_moe_kernel<false, MMVQ_TC_NVFP4_KSPLIT><<<grid, block, smem, s>>>(
        (const char *)vx_gate, nullptr, expert_stride, (const block_q8_1 *)vy, id, groups, n_pairs, cols,
        scales, scales_up, dst, ncols_x, nrows_x, stride_col_y, y_div, activation, gate_cap, up_cap);
  }
}

// Rows `ids` of a tiled NVFP4 stack (row id = expert * rows + row, rows a multiple of 16) back as GGUF's row-major
// 36-byte blocks: a thread per output block.
static __global__ void nvfp4_untile_kernel(const uint8_t *__restrict__ src, const uint32_t *__restrict__ ids,
                                           uint8_t *__restrict__ dst, const int64_t n, const int nb) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * nb) {
    return;
  }
  const int64_t row = idx / nb;
  const int kb = (int)(idx % nb);
  const uint32_t id = ids[row];
  const int r = id % MMVQ_TC_NVFP4_ROWS;
  const int e = r % 8;
  const int upper = r / 8;
  const uint8_t *tile = src + ((int64_t)(id / MMVQ_TC_NVFP4_ROWS) * nb + kb) * MMVQ_TC_NVFP4_TILE_BYTES;
  uint8_t *out = dst + (row * nb + kb) * (MMVQ_TC_NVFP4_TILE_BYTES / MMVQ_TC_NVFP4_ROWS);
#pragma unroll
  for (int s = 0; s < MMVQ_TC_NVFP4_SUBS; ++s) {
    out[s] = tile[e * 8 + upper * 4 + s];
  }
  // value v of sub-block s sits in lane (e, v / 4)'s word s, byte v % 4; GGUF keeps it at byte 8 s + v % 8
  auto nibble = [&](const int s, const int v) {
    const uint8_t byte = tile[MMVQ_TC_NVFP4_SCALE_BYTES + (4 * e + v / 4) * 16 + 4 * s + v % 4];
    return upper ? byte >> 4 : byte & 0x0F;
  };
#pragma unroll
  for (int s = 0; s < MMVQ_TC_NVFP4_SUBS; ++s) {
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      out[MMVQ_TC_NVFP4_SUBS + 8 * s + b] = (uint8_t)(nibble(s, b) | nibble(s, b + 8) << 4);
    }
  }
}

extern "C" void launch_nvfp4_untile_rows(const void *src, const void *ids, void *dst, int64_t n, int64_t ncols,
                                         void *stream) {
  constexpr int threads = 256;
  const int nb = (int)(ncols / QK_NVFP4);
  const int64_t total = n * nb;
  const int blocks = (int)((total + threads - 1) / threads);
  nvfp4_untile_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      (const uint8_t *)src, (const uint32_t *)ids, (uint8_t *)dst, n, nb);
}
