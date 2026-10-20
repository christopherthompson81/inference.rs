#include "gdn_common.cuh"

// Chunked gated delta rule for prefill, the same recurrence as
// gated_delta_rule_recurrence_kernel_vmajor_grouped over value-major K=V=128
// state. Within a chunk with start state S0 and cumulative log decay G_t:
//   A_tj = beta_t * exp(G_t - G_j) * k_t.k_j (j < t),  T = (I + A)^-1
//   W = T (beta exp(G) k),  U = T (beta v),  Delta = U - W S0
//   y = exp(G) (q S0) + M Delta with M_tj = exp(G_t - G_j) q_t.k_j (j <= t)
//   S = exp(G_L) S0 + (exp(G_L - G_t) k_t)^T Delta
// Every exponent is a difference that is never positive, so nothing overflows
// or divides by an underflowed decay. The prepare kernel builds W, U, M and G
// for every chunk at once; the scan kernel walks the chunks of one head and one
// value tile in order. Both are register-tiled fp32, so results differ from the
// token scan only in rounding.

namespace {

constexpr int GDN_CHUNK = 64;
constexpr int GDN_CHUNK_DIM = 128;
constexpr int GDN_PREPARE_THREADS = 256;
constexpr int GDN_SCAN_THREADS = 128;
constexpr int GDN_SCAN_VTILE = 16;
constexpr int GDN_SCAN_SLICE = 32;
// floats a chunk's workspace holds: W^T [K, C], Q^T [K, C], U [C, V], M^T [C,
// C], G [C]; the transposes let the scan copy its operand slices straight into
// shared memory
constexpr int GDN_CHUNK_WS =
    GDN_CHUNK * GDN_CHUNK_DIM * 3 + GDN_CHUNK * GDN_CHUNK + GDN_CHUNK;
// K and Q pass through shared memory in halves of the head dimension, so two
// blocks fit an SM
constexpr int GDN_PREPARE_HALF = GDN_CHUNK_DIM / 2;
// transposed rows padded to spread the transposing stores over the banks; still
// 16-byte aligned
constexpr int GDN_PREPARE_LD = GDN_CHUNK + 4;
constexpr size_t GDN_PREPARE_SMEM = (2 * GDN_PREPARE_HALF * GDN_PREPARE_LD +
                                     GDN_CHUNK * GDN_CHUNK + 2 * GDN_CHUNK) *
                                    sizeof(float);

__device__ __forceinline__ float4 ld4(const float *p) {
  return *reinterpret_cast<const float4 *>(p);
}

__device__ __forceinline__ void fma4x4(float (&acc)[4][4], const float4 a,
                                       const float4 b) {
  const float av[4] = {a.x, a.y, a.z, a.w};
  const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
  for (int r = 0; r < 4; ++r) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      acc[r][c] = __fmaf_rn(av[r], bv[c], acc[r][c]);
    }
  }
}

__device__ __forceinline__ void fma4x2(float (&acc)[4][2], const float4 a,
                                       const float2 b) {
  const float av[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    acc[r][0] = __fmaf_rn(av[r], b.x, acc[r][0]);
    acc[r][1] = __fmaf_rn(av[r], b.y, acc[r][1]);
  }
}

__device__ __forceinline__ float2 ld2(const float *p) {
  return *reinterpret_cast<const float2 *>(p);
}

__global__ __launch_bounds__(GDN_PREPARE_THREADS) void gdn_chunk_prepare_kernel(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, float *__restrict__ ws, int seq_len,
    int chunks) {
  constexpr int C = GDN_CHUNK;
  constexpr int D = GDN_CHUNK_DIM;
  constexpr int H = GDN_PREPARE_HALF;
  constexpr int LD = GDN_PREPARE_LD;
  extern __shared__ float4 smem4[];
  float *kt = reinterpret_cast<float *>(smem4); // [H][LD], one half of K^T
  float *qt = kt + H * LD;                      // [H][LD], the same half of Q^T
  float *as = qt + H * LD;                      // [C][C]
  float *gs = as + C * C;                       // [C] cumulative log decay
  float *bs = gs + C;                           // [C]

  const int chunk = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int t0 = chunk * C;
  const int len = min(C, seq_len - t0);
  const size_t row0 = (size_t)bh * seq_len + t0;
  float *wt_out = ws + ((size_t)bh * chunks + chunk) * GDN_CHUNK_WS;
  float *qt_out = wt_out + D * C;
  float *u_out = qt_out + D * C;
  float *mt_out = u_out + C * D;
  float *g_out = mt_out + C * C;

  if (tid < C) {
    gs[tid] = tid < len ? g[row0 + tid] : 0.0f;
    bs[tid] = tid < len ? beta[row0 + tid] : 0.0f;
  }
  __syncthreads();
  if (tid < 32) {
    // inclusive scan of the 64 log decays, two per lane
    float a = gs[2 * tid], b = a + gs[2 * tid + 1];
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const float n = __shfl_up_sync(0xffffffff, b, off);
      if (tid >= off) {
        a += n;
        b += n;
      }
    }
    gs[2 * tid] = a;
    gs[2 * tid + 1] = b;
  }
  __syncthreads();
  if (tid < C) {
    g_out[tid] = gs[tid];
  }

  // K K^T and Q K^T, rows ty*4..+4 and columns tx*4..+4 of each
  const int ty = tid / 16, tx = tid % 16;
  float kk[4][4] = {}, qk[4][4] = {};
  for (int h0 = 0; h0 < D; h0 += H) {
    __syncthreads();
    for (int i = tid; i < C * H; i += GDN_PREPARE_THREADS) {
      const int t = i / H, d = i % H;
      const bool live = t < len;
      kt[d * LD + t] = live ? k[(row0 + t) * D + h0 + d] : 0.0f;
      qt[d * LD + t] = live ? q[(row0 + t) * D + h0 + d] : 0.0f;
    }
    __syncthreads();
    for (int i = tid; i < H * C; i += GDN_PREPARE_THREADS) {
      qt_out[(h0 + i / C) * C + i % C] = qt[(i / C) * LD + i % C];
    }
#pragma unroll 4
    for (int d = 0; d < H; ++d) {
      const float4 kj = ld4(&kt[d * LD + tx * 4]);
      fma4x4(kk, ld4(&kt[d * LD + ty * 4]), kj);
      fma4x4(qk, ld4(&qt[d * LD + ty * 4]), kj);
    }
  }
  __syncthreads();
  float *mt_s =
      kt; // M^T staged as [C][C + 1] over the K and Q halves, which are done

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int t = ty * 4 + r;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int j = tx * 4 + c;
      const float decay = j <= t ? expf(gs[t] - gs[j]) : 0.0f;
      as[t * C + j] = j < t ? bs[t] * decay * kk[r][c] : 0.0f;
      mt_s[j * (C + 1) + t] = decay * qk[r][c];
    }
  }
  __syncthreads();
  for (int i = tid; i < C * C; i += GDN_PREPARE_THREADS) {
    mt_out[i] = mt_s[(i / C) * (C + 1) + i % C];
  }

  // forward substitution, one column per thread: columns [0, D) solve for W,
  // [D, 2D) for U
  const int col = tid;
  // the right-hand sides load first, so their latencies overlap instead of
  // stalling every step
  float x[C];
#pragma unroll
  for (int t = 0; t < C; ++t) {
    x[t] = 0.0f;
    if (t < len) {
      x[t] = col < D ? k[(row0 + t) * D + col] : v[(row0 + t) * D + col - D];
    }
  }
#pragma unroll
  for (int t = 0; t < C; ++t) {
    float b = bs[t] * (col < D ? expf(gs[t]) : 1.0f) * x[t];
#pragma unroll
    for (int j = 0; j < (t & ~3); j += 4) {
      const float4 a = ld4(&as[t * C + j]);
      b = __fmaf_rn(-a.x, x[j], b);
      b = __fmaf_rn(-a.y, x[j + 1], b);
      b = __fmaf_rn(-a.z, x[j + 2], b);
      b = __fmaf_rn(-a.w, x[j + 3], b);
    }
#pragma unroll
    for (int j = t & ~3; j < t; ++j) {
      b = __fmaf_rn(-as[t * C + j], x[j], b);
    }
    x[t] = b;
  }
  if (col < D) {
#pragma unroll
    for (int t = 0; t < C; t += 4) {
      *reinterpret_cast<float4 *>(&wt_out[col * C + t]) =
          make_float4(x[t], x[t + 1], x[t + 2], x[t + 3]);
    }
  } else {
#pragma unroll
    for (int t = 0; t < C; ++t) {
      u_out[t * D + col - D] = x[t];
    }
  }
}

template <typename StateT>
__global__ __launch_bounds__(GDN_SCAN_THREADS) void gdn_chunk_scan_kernel(
    const float *__restrict__ k, const float *__restrict__ ws,
    StateT *__restrict__ state, float *__restrict__ output, int seq_len,
    int chunks, const int32_t *__restrict__ slot_indices, int num_heads) {
  constexpr int C = GDN_CHUNK;
  constexpr int D = GDN_CHUNK_DIM;
  constexpr int VT = GDN_SCAN_VTILE;
  constexpr int SL = GDN_SCAN_SLICE;
  // a chunk's operand slices in order: W^T x4, Q^T x4 ([SL][C]), M^T x2
  // ([SL][C]), scaled K rows x2 ([SL][D])
  constexpr int SLICES = 12;
  constexpr int SMALL =
      C * SL / 4 /
      GDN_SCAN_THREADS; // float4 a thread copies of a [SL][C] slice
  constexpr int LARGE = SL * D / 4 / GDN_SCAN_THREADS; // and of a [SL][D] slice
  __shared__ __align__(16) float s_s[D * VT]; // state tile [key][value]
  __shared__ __align__(16) float d_s[C * VT]; // Delta tile [t][value]
  __shared__ __align__(16) float a_s[SL * D]; // the current operand slice
  __shared__ float g_s[C];
  __shared__ float
      f_s[C]; // exp(G_L - G_t), the weight of row t in the state update

  const int v0 = blockIdx.x * VT;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  // [C, VT] products: rows ty*4..+4, values tx*2..+2; the [D, VT] state update
  // takes rows ty*8..+8
  const int ty = tid / 8, tx = tid % 8;
  float *out_bh = output + (size_t)bh * seq_len * D;

  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int i = tid; i < seq_len * VT; i += GDN_SCAN_THREADS) {
      out_bh[(size_t)(i / VT) * D + v0 + i % VT] = 0.0f;
    }
    return;
  }
  StateT *state_bh = state + gdn_state_row(slot_indices, bh / num_heads,
                                           bh % num_heads, num_heads) *
                                 D * D;
  for (int i = tid; i < D * VT; i += GDN_SCAN_THREADS) {
    const int value = i / D, key = i % D;
    s_s[key * VT + value] =
        static_cast<float>(state_bh[(v0 + value) * D + key]);
  }

  const auto chunk_ws = [&](int chunk) {
    return ws + ((size_t)bh * chunks + chunk) * GDN_CHUNK_WS;
  };
  // loads slice `sl` of `chunk` into registers; K rows past the chunk's end
  // read as zero
  const auto fetch = [&](int chunk, int sl, float4(&r)[LARGE]) {
    const float *wt = chunk_ws(chunk);
    if (sl < 10) {
      const float *src = sl < 4   ? wt + sl * SL * C
                         : sl < 8 ? wt + D * C + (sl - 4) * SL * C
                                  : wt + 3 * D * C + (sl - 8) * SL * C;
#pragma unroll
      for (int n = 0; n < SMALL; ++n) {
        r[n] = ld4(&src[(tid + n * GDN_SCAN_THREADS) * 4]);
      }
    } else {
      const int t0 = chunk * C;
      const int len = min(C, seq_len - t0);
      const size_t row0 = (size_t)bh * seq_len + t0;
#pragma unroll
      for (int n = 0; n < LARGE; ++n) {
        const int i = (tid + n * GDN_SCAN_THREADS) * 4;
        const int row = (sl - 10) * SL + i / D;
        r[n] = row < len ? ld4(&k[(row0 + row) * D + i % D])
                         : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
  };

  float4 pre[LARGE];
  fetch(0, 0, pre);
  float acc[4][2] = {}, sa[4][2] = {}, sb[4][2] = {};
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int t0 = chunk * C;
    const int len = min(C, seq_len - t0);
    const float *wt = chunk_ws(chunk);
    const float *u = wt + 2 * D * C;
    const float *gc = wt + 3 * D * C + C * C;
    __syncthreads();
    if (tid < C) {
      const float g_last = gc[len - 1];
      g_s[tid] = gc[tid];
      f_s[tid] = tid < len ? expf(g_last - gc[tid]) : 0.0f;
    }
    for (int sl = 0; sl < SLICES; ++sl) {
      __syncthreads();
      if (sl < 10) {
#pragma unroll
        for (int n = 0; n < SMALL; ++n) {
          *reinterpret_cast<float4 *>(&a_s[(tid + n * GDN_SCAN_THREADS) * 4]) =
              pre[n];
        }
      } else {
#pragma unroll
        for (int n = 0; n < LARGE; ++n) {
          const int i = (tid + n * GDN_SCAN_THREADS) * 4;
          const float f = f_s[(sl - 10) * SL + i / D];
          float4 kv = pre[n];
          kv.x *= f;
          kv.y *= f;
          kv.z *= f;
          kv.w *= f;
          *reinterpret_cast<float4 *>(&a_s[i]) = kv;
        }
      }
      if (sl + 1 < SLICES) {
        fetch(chunk, sl + 1, pre);
      } else if (chunk + 1 < chunks) {
        fetch(chunk + 1, 0, pre);
      }
      __syncthreads();
      if (sl < 8) {
        // Delta = U - W S0, then exp(G) (q S0)
        const int s0 = (sl % 4) * SL;
#pragma unroll 8
        for (int kk = 0; kk < SL; ++kk) {
          fma4x2(acc, ld4(&a_s[kk * C + ty * 4]),
                 ld2(&s_s[(s0 + kk) * VT + tx * 2]));
        }
      } else if (sl < 10) {
        // + M Delta
        const int s0 = (sl - 8) * SL;
#pragma unroll 8
        for (int kk = 0; kk < SL; ++kk) {
          fma4x2(acc, ld4(&a_s[kk * C + ty * 4]),
                 ld2(&d_s[(s0 + kk) * VT + tx * 2]));
        }
      } else {
        // (exp(G_L - G_t) k_t)^T Delta
        const int s0 = (sl - 10) * SL;
#pragma unroll 8
        for (int t = 0; t < SL; ++t) {
          const float2 dd = ld2(&d_s[(s0 + t) * VT + tx * 2]);
          fma4x2(sa, ld4(&a_s[t * D + ty * 8]), dd);
          fma4x2(sb, ld4(&a_s[t * D + ty * 8 + 4]), dd);
        }
      }
      if (sl == 3) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int row = ty * 4 + r;
          const float2 uu = ld2(&u[row * D + v0 + tx * 2]);
          d_s[row * VT + tx * 2] = uu.x - acc[r][0];
          d_s[row * VT + tx * 2 + 1] = uu.y - acc[r][1];
          acc[r][0] = 0.0f;
          acc[r][1] = 0.0f;
        }
      } else if (sl == 7) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float decay_row = expf(g_s[ty * 4 + r]);
          acc[r][0] *= decay_row;
          acc[r][1] *= decay_row;
        }
      } else if (sl == 9) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int row = ty * 4 + r;
          if (row < len) {
            *reinterpret_cast<float2 *>(
                &out_bh[(size_t)(t0 + row) * D + v0 + tx * 2]) =
                make_float2(acc[r][0], acc[r][1]);
          }
          acc[r][0] = 0.0f;
          acc[r][1] = 0.0f;
        }
      }
    }
    // S = exp(G_L) S0 + the update; only this thread reads its own state
    // entries
    const float decay_last = expf(g_s[len - 1]);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float &lo = s_s[(ty * 8 + r) * VT + tx * 2 + c];
        lo = __fmaf_rn(decay_last, lo, sa[r][c]);
        float &hi = s_s[(ty * 8 + 4 + r) * VT + tx * 2 + c];
        hi = __fmaf_rn(decay_last, hi, sb[r][c]);
        sa[r][c] = 0.0f;
        sb[r][c] = 0.0f;
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < D * VT; i += GDN_SCAN_THREADS) {
    const int value = i / D, key = i % D;
    state_bh[(v0 + value) * D + key] =
        static_cast<StateT>(s_s[key * VT + value]);
  }
}

template <typename StateT>
cudaError_t launch_gdn_chunked(const float *q, const float *k, const float *v,
                               const float *g, const float *beta, StateT *state,
                               float *output, float *ws, int bh, int seq_len,
                               const int32_t *slot_indices, int num_heads,
                               cudaStream_t stream) {
  const int chunks = (seq_len + GDN_CHUNK - 1) / GDN_CHUNK;
  cudaError_t err = cudaFuncSetAttribute(
      gdn_chunk_prepare_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)GDN_PREPARE_SMEM);
  if (err != cudaSuccess) {
    return err;
  }
  gdn_chunk_prepare_kernel<<<dim3(chunks, bh), GDN_PREPARE_THREADS,
                             GDN_PREPARE_SMEM, stream>>>(q, k, v, g, beta, ws,
                                                         seq_len, chunks);
  gdn_chunk_scan_kernel<StateT><<<dim3(GDN_CHUNK_DIM / GDN_SCAN_VTILE, bh),
                                  GDN_SCAN_THREADS, 0, stream>>>(
      k, ws, state, output, seq_len, chunks, slot_indices, num_heads);
  return cudaGetLastError();
}

} // namespace

extern "C" size_t gdn_chunked_workspace_floats(int bh, int seq_len) {
  return (size_t)bh * ((seq_len + GDN_CHUNK - 1) / GDN_CHUNK) * GDN_CHUNK_WS;
}

extern "C" int gdn_chunked_recurrence(const float *q, const float *k,
                                      const float *v, const float *g,
                                      const float *beta, void *state,
                                      float *output, float *workspace, int bh,
                                      int seq_len, const int32_t *slot_indices,
                                      int num_heads, int state_dtype,
                                      int64_t stream) {
  const cudaStream_t s = (cudaStream_t)stream;
  if (state_dtype == GDN_STATE_DTYPE_F16) {
    return (int)launch_gdn_chunked(q, k, v, g, beta, (__half *)state, output,
                                   workspace, bh, seq_len, slot_indices,
                                   num_heads, s);
  } else if (state_dtype == GDN_STATE_DTYPE_BF16) {
    return (int)launch_gdn_chunked(q, k, v, g, beta, (__nv_bfloat16 *)state,
                                   output, workspace, bh, seq_len, slot_indices,
                                   num_heads, s);
  }
  return (int)launch_gdn_chunked(q, k, v, g, beta, (float *)state, output,
                                 workspace, bh, seq_len, slot_indices,
                                 num_heads, s);
}
