// Tensor-core MoE matvec over tiled expert stacks (nvfp4_tiled.rs, q8_0_tiled.rs, q6k_tiled.rs), for one token's
// experts up to a few tokens': a block is one distinct expert's 16-row group (a slot of mmvq_moe_group_pairs's plan,
// or with no plan pair blockIdx.y alone) and every pair routed to it, up to 8 pairs as the columns of an int8
// mma.m16n8k16 per 16-value sub-block, so each weight is read and decoded once for all of them (sm_80+). A tile type
// gives a step of K (its loads, its sub-blocks' A fragments, both rows' scales); each sub-block's scale and each q8_1
// block's activation scale apply to the mma's int32 sums, as the type's vec_dot applies them to its own.
#pragma once
#include "mmvq_gguf.cuh"

// weight rows one tile and mma cover, and the pairs (columns) an mma takes
#define MMVQ_TC_MOE_ROWS 16
#define MMVQ_TC_MOE_COLS 8
#define MMVQ_TC_MOE_SUB 16
// warps of a block, splitting K: the fused gate and up projection has half the row groups of the down projection it
// pairs with and twice its K, so more warps a group keep as many loads in flight
#define MMVQ_TC_MOE_KSPLIT 4
#define MMVQ_TC_MOE_KSPLIT_GLU 6
// UE4M3 scale bytes as ggml_cuda_ue4m3_to_fp32 reads them (bit 7 ignored), a table each NVFP4 block builds
#define MMVQ_TC_UE4M3_VALUES 128
// the dynamic shared memory a launch may take without opting in
#define MMVQ_TC_MOE_SMEM_MAX (48 * 1024)

static constexpr __host__ __device__ uint32_t tc_pack4_bytes(int a, int b, int c, int d) {
  return (uint32_t)(uint8_t)a | (uint32_t)(uint8_t)b << 8 | (uint32_t)(uint8_t)c << 16 | (uint32_t)(uint8_t)d << 24;
}

// NVFP4 (nvfp4_tiled.rs): a step is one 576-byte tile of 64 values, a lane's 16 bytes and its rows' 8 scale bytes
struct TcNvfp4 {
  static constexpr int STEP = 64;
  static constexpr int SUBS = STEP / MMVQ_TC_MOE_SUB;
  static constexpr int AHEAD = 2;
  static constexpr bool UE4M3 = true;
  static constexpr int TILE_VALUES = 64;
  static constexpr int TILE_BYTES = 576;
  struct Regs {
    uint4 q;
    uint2 sc;
  };
  static __device__ __forceinline__ void load(const char *group, const int step, const int lane, const int g,
                                              Regs &r) {
    const char *p = group + (int64_t)step * TILE_BYTES;
    r.sc = __ldg((const uint2 *)p + g);
    r.q = __ldcs((const uint4 *)(p + 64) + lane);
  }
  // get_int_from_table_16 over kvalues_fp4 in immediates: the word's low nibbles (row g) and high (row g + 8)
  static __device__ __forceinline__ int2 frag(const Regs &r, const int is) {
    constexpr uint32_t t0 = tc_pack4_bytes(0, 1, 2, 3), t1 = tc_pack4_bytes(4, 6, 8, 12);
    constexpr uint32_t t2 = tc_pack4_bytes(0, -1, -2, -3), t3 = tc_pack4_bytes(-4, -6, -8, -12);
    const uint32_t q4 = is == 0 ? r.q.x : is == 1 ? r.q.y : is == 2 ? r.q.z : r.q.w;
    const uint32_t mask = 0x32103210 | ((q4 & 0x88888888) >> 1);
    const uint32_t a3 = __byte_perm(__byte_perm(t0, t1, q4), __byte_perm(t2, t3, q4), mask);
    const uint32_t a4 = __byte_perm(__byte_perm(t0, t1, q4 >> 16), __byte_perm(t2, t3, q4 >> 16), mask >> 16);
    return make_int2((int)__byte_perm(a3, a4, 0x6420), (int)__byte_perm(a3, a4, 0x7531));
  }
  static __device__ __forceinline__ float2 scales(const Regs &r, const int is, const float *ue4m3) {
    return make_float2(ue4m3[(r.sc.x >> (8 * is)) & 0x7F], ue4m3[(r.sc.y >> (8 * is)) & 0x7F]);
  }
};

// Q8_0 (q8_0_tiled.rs): a step is one 544-byte tile of 32 values, a lane's 16 bytes (both sub-blocks, both rows) and
// its rows' f16 scales
struct TcQ8_0 {
  static constexpr int STEP = 32;
  static constexpr int SUBS = STEP / MMVQ_TC_MOE_SUB;
  static constexpr int AHEAD = 2;
  static constexpr bool UE4M3 = false;
  static constexpr int TILE_VALUES = 32;
  static constexpr int TILE_BYTES = 544;
  struct Regs {
    uint4 q;
    uint32_t d;
  };
  static __device__ __forceinline__ void load(const char *group, const int step, const int lane, const int g,
                                              Regs &r) {
    const char *p = group + (int64_t)step * TILE_BYTES;
    r.d = __ldg((const uint32_t *)p + g);
    r.q = __ldcs((const uint4 *)(p + 32) + lane);
  }
  static __device__ __forceinline__ int2 frag(const Regs &r, const int is) {
    return is == 0 ? make_int2((int)r.q.x, (int)r.q.y) : make_int2((int)r.q.z, (int)r.q.w);
  }
  static __device__ __forceinline__ float2 scales(const Regs &r, const int, const float *) {
    return __half22float2(*reinterpret_cast<const half2 *>(&r.d));
  }
};

// Q6_K (q6k_tiled.rs): a step is one 128-value half of a 3360-byte tile, a lane's 48 bytes (8 words of low bits, 16
// bytes of high bits), its rows' 8 sub-block scales each and their f16 `d`
struct TcQ6K {
  static constexpr int STEP = 128;
  static constexpr int SUBS = STEP / MMVQ_TC_MOE_SUB;
  static constexpr int AHEAD = 1;
  static constexpr bool UE4M3 = false;
  static constexpr int TILE_VALUES = 256;
  static constexpr int TILE_BYTES = 3360;
  struct Regs {
    uint4 lo[2];
    uint4 hi;
    uint4 sc;
    uint32_t d;
  };
  static __device__ __forceinline__ void load(const char *group, const int step, const int lane, const int g,
                                              Regs &r) {
    const int h = step & 1;
    const char *p = group + (int64_t)(step >> 1) * TILE_BYTES;
    r.d = __ldg((const uint32_t *)p + g);
    r.sc = __ldg((const uint4 *)(p + 32 + g * 32 + h * 16));
    const uint4 *l = (const uint4 *)(p + 288 + h * 1536 + lane * 48);
    r.lo[0] = __ldcs(l);
    r.lo[1] = __ldcs(l + 1);
    r.hi = __ldcs(l + 2);
  }
  // four values' top two bits (bits 2 j of `b`) moved to bits 4-5 of byte j
  static __device__ __forceinline__ uint32_t high_bits(const uint32_t b) {
    return ((b & 0x03) | ((b & 0x0C) << 6) | ((b & 0x30) << 12) | ((b & 0xC0) << 18)) << 4;
  }
  static __device__ __forceinline__ uint32_t pick(const uint4 v, const int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
  }
  static __device__ __forceinline__ int2 frag(const Regs &r, const int is) {
    const uint32_t lo = pick(r.lo[is >> 2], is & 3);
    const uint32_t hi = pick(r.hi, is >> 1) >> (16 * (is & 1));
    const uint32_t a = (lo & 0x0F0F0F0F) | high_bits(hi & 0xFF);
    const uint32_t b = ((lo >> 4) & 0x0F0F0F0F) | high_bits((hi >> 8) & 0xFF);
    return make_int2((int)__vsub4(a, 0x20202020), (int)__vsub4(b, 0x20202020));
  }
  static __device__ __forceinline__ float2 scales(const Regs &r, const int is, const float *) {
    const float2 d = __half22float2(*reinterpret_cast<const half2 *>(&r.d));
    const int8_t sa = (int8_t)(pick(r.sc, is >> 2) >> (8 * (is & 3)));
    const int8_t sb = (int8_t)(pick(r.sc, 2 + (is >> 2)) >> (8 * (is & 3)));
    return make_float2(d.x * sa, d.y * sb);
  }
};

// Shared memory a block takes for `cols` columns of `ncols_x`-long activations: per column, step and t lane the
// sub-blocks' B fragment ints, per step and column the q8_1 blocks' scales, and the UE4M3 table where the type has one
template <typename Tile> static __host__ __device__ size_t tc_moe_smem(const int cols, const int ncols_x) {
  const int steps = ncols_x / Tile::STEP;
  return (size_t)cols * steps * 4 * Tile::SUBS * sizeof(int) +
         (size_t)steps * cols * (Tile::STEP / QK8_1) * sizeof(float) +
         (Tile::UE4M3 ? MMVQ_TC_UE4M3_VALUES * sizeof(float) : 0);
}

template <typename Tile, bool glu, int KSPLIT>
static __global__ void __launch_bounds__(KSPLIT *WARP_SIZE)
    mmvq_tc_moe_kernel(const char *__restrict__ vx_gate, const char *__restrict__ vx_up,
                       const int64_t expert_stride, const block_q8_1 *__restrict__ y,
                       const uint32_t *__restrict__ ids, const int *__restrict__ groups, const int n_pairs,
                       const int cols, const float *__restrict__ scales, const float *__restrict__ scales_up,
                       float *__restrict__ dst, const int ncols_x, const int nrows_x, const int stride_col_y,
                       const int y_div, const int activation, const float gate_cap, const float up_cap) {
#if __CUDA_ARCH__ >= 800
  constexpr int nsums = glu ? 2 : 1;
  constexpr int SUBS = Tile::SUBS;
  constexpr int QBS = Tile::STEP / QK8_1;
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
  const int steps = ncols_x / Tile::STEP;
  const int64_t group_bytes = (int64_t)(ncols_x / Tile::TILE_VALUES) * Tile::TILE_BYTES;
  const int ra = blockIdx.x * MMVQ_TC_MOE_ROWS + g;
  const int rb = ra + 8;
  const char *base[nsums];
  base[0] = vx_gate + expert * expert_stride + blockIdx.x * group_bytes;
  if constexpr (glu) {
    base[1] = vx_up + expert * expert_stride + blockIdx.x * group_bytes;
  }

  extern __shared__ int tc_moe_smem_base[];
  int *s_b = tc_moe_smem_base;
  float *s_d = (float *)(s_b + cols * steps * 4 * SUBS);
  float *s_ue4m3 = s_d + steps * cols * QBS;
  if constexpr (Tile::UE4M3) {
    for (int i = tid; i < MMVQ_TC_UE4M3_VALUES; i += KSPLIT * WARP_SIZE) {
      s_ue4m3[i] = ggml_cuda_ue4m3_to_fp32((uint8_t)i);
    }
  }

  const int chunk = (steps + KSPLIT - 1) / KSPLIT;
  const int k0 = warp * chunk;
  const int k1 = min(steps, k0 + chunk);

  struct Weights {
    typename Tile::Regs r[nsums];
  };
  auto load = [&](const int step, Weights &w) {
#pragma unroll
    for (int s = 0; s < nsums; ++s) {
      Tile::load(base[s], step, lane, g, w.r[s]);
    }
  };

  __shared__ float red[KSPLIT][nsums][4][WARP_SIZE];

  for (int at = first; at < last; at += cols) {
    const int count = min(cols, last - at);
    // the weights stream in while this chunk's activations are staged
    Weights ring[Tile::AHEAD];
#pragma unroll
    for (int d = 0; d < Tile::AHEAD; ++d) {
      if (k0 + d < k1) {
        load(k0 + d, ring[d]);
      }
    }
    // column c's B fragments: lane t's int of each sub-block is (q8_1 block STEP / 32 step + is / 2, int (is % 2) 4 + t)
    for (int i = tid; i < count * steps * 4; i += KSPLIT * WARP_SIZE) {
      const int c = i / (steps * 4);
      const int step = (i / 4) % steps;
      const int tt = i % 4;
      const block_q8_1 *yc = y + (int64_t)(pair_at(at + c) / y_div) * stride_col_y + QBS * step;
#pragma unroll
      for (int is = 0; is < SUBS; ++is) {
        s_b[i * SUBS + is] = get_int_b4(yc[is >> 1].qs, (is & 1) * 4 + tt);
      }
    }
    for (int i = tid; i < count * steps; i += KSPLIT * WARP_SIZE) {
      const int step = i / count;
      const int c = i % count;
      const block_q8_1 *yc = y + (int64_t)(pair_at(at + c) / y_div) * stride_col_y + QBS * step;
#pragma unroll
      for (int q = 0; q < QBS; ++q) {
        s_d[i * QBS + q] = __low2float(yc[q].ds);
      }
    }
    __syncthreads();

    // this lane's B column g and accumulator columns 2t, 2t + 1, repeating the last column past count
    const int cb = min(g, count - 1);
    const int c0 = min(2 * t, count - 1);
    const int c1 = min(2 * t + 1, count - 1);
    float acc[nsums][4] = {};
    for (int s00 = k0; s00 < k1; s00 += Tile::AHEAD) {
#pragma unroll
      for (int d = 0; d < Tile::AHEAD; ++d) {
        const int step = s00 + d;
        if (step >= k1) {
          break;
        }
        const Weights cur = ring[d];
        if (step + Tile::AHEAD < k1) {
          load(step + Tile::AHEAD, ring[d]);
        }
        const int *bq = s_b + ((cb * steps + step) * 4 + t) * SUBS;
        const float *d0 = s_d + (step * count + c0) * QBS;
        const float *d1 = s_d + (step * count + c1) * QBS;
#pragma unroll
        for (int is = 0; is < SUBS; ++is) {
          const int b0 = bq[is];
          const float dy0 = d0[is >> 1];
          const float dy1 = d1[is >> 1];
#pragma unroll
          for (int s = 0; s < nsums; ++s) {
            const int2 a = Tile::frag(cur.r[s], is);
            int c[4] = {0, 0, 0, 0};
            asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, "
                "{%0, %1, %2, %3};"
                : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                : "r"(a.x), "r"(a.y), "r"(b0));
            const float2 sab = Tile::scales(cur.r[s], is, s_ue4m3);
            acc[s][0] += sab.x * dy0 * (float)c[0];
            acc[s][1] += sab.x * dy1 * (float)c[1];
            acc[s][2] += sab.y * dy0 * (float)c[2];
            acc[s][3] += sab.y * dy1 * (float)c[3];
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
template <typename Tile>
static void launch_mmvq_tc_moe(const void *vx_gate, const void *vx_up, int64_t expert_stride, const void *vy,
                               const void *ids, const int *groups, int n_pairs, const float *scales,
                               const float *scales_up, float *dst, int ncols_x, int nrows_x, int stride_col_y,
                               int y_div, int activation, float gate_cap, float up_cap, void *stream) {
  const dim3 grid((nrows_x + MMVQ_TC_MOE_ROWS - 1) / MMVQ_TC_MOE_ROWS, n_pairs, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const uint32_t *id = (const uint32_t *)ids;
  // fewer columns a pass when a wide K's staged activations would pass the default shared memory a block takes
  int cols = groups ? min(MMVQ_TC_MOE_COLS, n_pairs) : 1;
  while (cols > 1 && tc_moe_smem<Tile>(cols, ncols_x) > MMVQ_TC_MOE_SMEM_MAX) {
    --cols;
  }
  const size_t smem = tc_moe_smem<Tile>(cols, ncols_x);
  if (vx_up) {
    const dim3 block(WARP_SIZE, MMVQ_TC_MOE_KSPLIT_GLU, 1);
    mmvq_tc_moe_kernel<Tile, true, MMVQ_TC_MOE_KSPLIT_GLU><<<grid, block, smem, s>>>(
        (const char *)vx_gate, (const char *)vx_up, expert_stride, (const block_q8_1 *)vy, id, groups, n_pairs,
        cols, scales, scales_up, dst, ncols_x, nrows_x, stride_col_y, y_div, activation, gate_cap, up_cap);
  } else {
    const dim3 block(WARP_SIZE, MMVQ_TC_MOE_KSPLIT, 1);
    mmvq_tc_moe_kernel<Tile, false, MMVQ_TC_MOE_KSPLIT><<<grid, block, smem, s>>>(
        (const char *)vx_gate, nullptr, expert_stride, (const block_q8_1 *)vy, id, groups, n_pairs, cols, scales,
        scales_up, dst, ncols_x, nrows_x, stride_col_y, y_div, activation, gate_cap, up_cap);
  }
}

#define MMVQ_TC_MOE_LAUNCHER(tag, Tile)                                                                          \
  extern "C" void launch_mmvq_tc_##tag##_moe(const void *vx_gate, const void *vx_up, int64_t expert_stride,       \
                                             const void *vy, const void *ids, const int *groups, int n_pairs,    \
                                             const float *scales, const float *scales_up, float *dst,            \
                                             int ncols_x, int nrows_x, int stride_col_y, int y_div,               \
                                             int activation, float gate_cap, float up_cap, void *stream) {       \
    launch_mmvq_tc_moe<Tile>(vx_gate, vx_up, expert_stride, vy, ids, groups, n_pairs, scales, scales_up, dst,    \
                             ncols_x, nrows_x, stride_col_y, y_div, activation, gate_cap, up_cap, stream);       \
  }

#define MMVQ_TC_UNTILE_LAUNCHER(tag, kernel, values)                                                             \
  extern "C" void launch_##tag##_untile_rows(const void *src, const void *ids, void *dst, int64_t n,             \
                                             int64_t ncols, void *stream) {                                      \
    constexpr int threads = 256;                                                                                \
    const int nb = (int)(ncols / (values));                                                                     \
    const int64_t total = n * nb;                                                                               \
    kernel<<<(int)((total + threads - 1) / threads), threads, 0, static_cast<cudaStream_t>(stream)>>>(          \
        (const uint8_t *)src, (const uint32_t *)ids, (uint8_t *)dst, n, nb);                                     \
  }
