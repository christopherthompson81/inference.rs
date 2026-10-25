// Bidirectional attention with an int8 Q.K^T (SageAttention's scheme): K less its per-head mean over tokens (a constant
// shift of each logit row, so softmax is unchanged), Q and K quantized per token, S from int8 MMA rescaled in f32, an
// online softmax, and P.V on fp16 MMA accumulating each key tile in fp16 into f32. GeForce Ampere runs int8 MMA at four
// times and fp16-accumulate MMA at twice the rate of the f32-accumulate MMA a bf16 attention uses.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kSaWarps = 8;
constexpr int kSaRows = 16 * kSaWarps;
constexpr int kSaKeys = 64;
constexpr int kSaThreads = 32 * kSaWarps;
constexpr float kSaInt8Max = 127.f;

// shared-memory rows padded by 16 bytes so the 8 rows of an ldmatrix land on distinct banks
template <int D> struct SaLayout {
  static constexpr int k_pitch = D + 16;
  static constexpr int v_pitch = 2 * D + 16;
  static constexpr int k_bytes = kSaKeys * k_pitch;
  static constexpr int v_bytes = kSaKeys * v_pitch;
  static constexpr int stage = k_bytes + v_bytes + kSaKeys * 4;
  static constexpr int total = 2 * stage;
};

__device__ __forceinline__ void sa_cp_async(void *smem, const void *gmem, int bytes) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}

__device__ __forceinline__ void sa_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N> __device__ __forceinline__ void sa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void sa_ldmatrix_x4(uint32_t (&r)[4], const void *p) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

__device__ __forceinline__ void sa_ldmatrix_x4_trans(uint32_t (&r)[4], const void *p) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

__device__ __forceinline__ void sa_mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void sa_mma_f16(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t sa_pack_half2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t *>(&h);
}

// one key tile (int8 K rows, fp16 V rows, f32 K scales) into a stage; rows past n_kv are zero-filled
template <int D>
__device__ __forceinline__ void sa_load_tile(uint8_t *stage, const int8_t *k, const __half *v, const float *ks,
                                             int key0, int n_kv) {
  using L = SaLayout<D>;
  constexpr int k_chunks = D / 16, v_chunks = 2 * D / 16;
  for (int c = threadIdx.x; c < kSaKeys * k_chunks; c += kSaThreads) {
    const int r = c / k_chunks, col = c % k_chunks, key = key0 + r;
    const bool ok = key < n_kv;
    sa_cp_async(stage + r * L::k_pitch + col * 16, k + static_cast<int64_t>(ok ? key : 0) * D + col * 16, ok ? 16 : 0);
  }
  uint8_t *vs = stage + L::k_bytes;
  for (int c = threadIdx.x; c < kSaKeys * v_chunks; c += kSaThreads) {
    const int r = c / v_chunks, col = c % v_chunks, key = key0 + r;
    const bool ok = key < n_kv;
    sa_cp_async(vs + r * L::v_pitch + col * 16,
                reinterpret_cast<const uint8_t *>(v + static_cast<int64_t>(ok ? key : 0) * D) + col * 16, ok ? 16 : 0);
  }
  // a head's scales start at any float, too unaligned for 16-byte copies
  if (threadIdx.x < kSaKeys) {
    const int key = key0 + threadIdx.x;
    reinterpret_cast<float *>(stage + L::k_bytes + L::v_bytes)[threadIdx.x] = key < n_kv ? ks[key] : 0.f;
  }
}

// grid (query blocks of kSaRows, heads, batch); q8/k8 (batch, heads, tokens, D) int8 with per-token scales, v
// (batch, heads, n_kv, D) fp16; out (batch, n_q, heads, D) bf16
template <int D>
__global__ void __launch_bounds__(kSaThreads, 1)
    sa_attention_kernel(const int8_t *q8, const float *qs, const int8_t *k8, const float *ks, const __half *v,
                        __nv_bfloat16 *out, int n_q, int n_kv, int heads, float scale_log2) {
  using L = SaLayout<D>;
  constexpr int KS = D / 32, NT = D / 8;
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t4 = lane % 4;
  const int64_t bh = static_cast<int64_t>(blockIdx.z) * heads + blockIdx.y;
  const int8_t *qh = q8 + bh * n_q * D;
  const int8_t *kh = k8 + bh * n_kv * D;
  const float *ksh = ks + bh * n_kv;
  const __half *vh = v + bh * n_kv * D;

  const int ra = blockIdx.x * kSaRows + warp * 16 + g, rb = ra + 8;
  uint32_t qa[KS][4];
#pragma unroll
  for (int kk = 0; kk < KS; ++kk) {
    const int c = kk * 32 + t4 * 4;
    qa[kk][0] = ra < n_q ? *reinterpret_cast<const uint32_t *>(qh + static_cast<int64_t>(ra) * D + c) : 0u;
    qa[kk][1] = rb < n_q ? *reinterpret_cast<const uint32_t *>(qh + static_cast<int64_t>(rb) * D + c) : 0u;
    qa[kk][2] = ra < n_q ? *reinterpret_cast<const uint32_t *>(qh + static_cast<int64_t>(ra) * D + c + 16) : 0u;
    qa[kk][3] = rb < n_q ? *reinterpret_cast<const uint32_t *>(qh + static_cast<int64_t>(rb) * D + c + 16) : 0u;
  }
  const float sqa = ra < n_q ? qs[bh * n_q + ra] * scale_log2 : 0.f;
  const float sqb = rb < n_q ? qs[bh * n_q + rb] * scale_log2 : 0.f;

  float o[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
    o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.f;
  float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;

  const int tiles = (n_kv + kSaKeys - 1) / kSaKeys;
  sa_load_tile<D>(smem, kh, vh, ksh, 0, n_kv);
  sa_commit();
  for (int tile = 0; tile < tiles; ++tile) {
    const uint8_t *cur = smem + (tile & 1) * L::stage;
    if (tile + 1 < tiles)
      sa_load_tile<D>(smem + ((tile + 1) & 1) * L::stage, kh, vh, ksh, (tile + 1) * kSaKeys, n_kv);
    sa_commit();
    sa_wait<1>();
    __syncthreads();

    int s[kSaKeys / 8][4];
#pragma unroll
    for (int j = 0; j < kSaKeys / 8; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0;
#pragma unroll
      for (int kk = 0; kk < KS; kk += 2) {
        uint32_t b[4];
        sa_ldmatrix_x4(b, cur + (j * 8 + lane % 8) * L::k_pitch + kk * 32 + (lane / 8) * 16);
        sa_mma_s8(s[j], qa[kk], b[0], b[1]);
        sa_mma_s8(s[j], qa[kk + 1], b[2], b[3]);
      }
    }

    const float *kscale = reinterpret_cast<const float *>(cur + L::k_bytes + L::v_bytes);
    float p[kSaKeys / 8][4];
    float mx_a = m_a, mx_b = m_b;
#pragma unroll
    for (int j = 0; j < kSaKeys / 8; ++j) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int col = j * 8 + 2 * t4 + c;
        const bool ok = tile * kSaKeys + col < n_kv;
        const float kv = kscale[col];
        p[j][c] = ok ? static_cast<float>(s[j][c]) * (sqa * kv) : -INFINITY;
        p[j][2 + c] = ok ? static_cast<float>(s[j][2 + c]) * (sqb * kv) : -INFINITY;
        mx_a = fmaxf(mx_a, p[j][c]);
        mx_b = fmaxf(mx_b, p[j][2 + c]);
      }
    }
    mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
    mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
    mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
    mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
    const float alpha_a = exp2f(m_a - mx_a), alpha_b = exp2f(m_b - mx_b);
    m_a = mx_a;
    m_b = mx_b;
    float sum_a = 0.f, sum_b = 0.f;
#pragma unroll
    for (int j = 0; j < kSaKeys / 8; ++j) {
      p[j][0] = exp2f(p[j][0] - m_a);
      p[j][1] = exp2f(p[j][1] - m_a);
      p[j][2] = exp2f(p[j][2] - m_b);
      p[j][3] = exp2f(p[j][3] - m_b);
      sum_a += p[j][0] + p[j][1];
      sum_b += p[j][2] + p[j][3];
    }
    l_a = l_a * alpha_a + sum_a;
    l_b = l_b * alpha_b + sum_b;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      o[nt][0] *= alpha_a;
      o[nt][1] *= alpha_a;
      o[nt][2] *= alpha_b;
      o[nt][3] *= alpha_b;
    }

    const uint8_t *vs = cur + L::k_bytes;
    uint32_t acc[NT][2];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
      acc[nt][0] = acc[nt][1] = 0u;
#pragma unroll
    for (int kt = 0; kt < kSaKeys / 16; ++kt) {
      const uint32_t pa[4] = {sa_pack_half2(p[2 * kt][0], p[2 * kt][1]), sa_pack_half2(p[2 * kt][2], p[2 * kt][3]),
                              sa_pack_half2(p[2 * kt + 1][0], p[2 * kt + 1][1]),
                              sa_pack_half2(p[2 * kt + 1][2], p[2 * kt + 1][3])};
      const int m = lane / 8;
      const int key = kt * 16 + (m % 2) * 8 + lane % 8;
#pragma unroll
      for (int np = 0; np < NT / 2; ++np) {
        uint32_t b[4];
        sa_ldmatrix_x4_trans(b, vs + key * L::v_pitch + (np * 16 + (m / 2) * 8) * 2);
        sa_mma_f16(acc[2 * np], pa, b[0], b[1]);
        sa_mma_f16(acc[2 * np + 1], pa, b[2], b[3]);
      }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc[nt][0]));
      const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc[nt][1]));
      o[nt][0] += lo.x;
      o[nt][1] += lo.y;
      o[nt][2] += hi.x;
      o[nt][3] += hi.y;
    }
    __syncthreads();
  }

  l_a += __shfl_xor_sync(0xffffffff, l_a, 1);
  l_a += __shfl_xor_sync(0xffffffff, l_a, 2);
  l_b += __shfl_xor_sync(0xffffffff, l_b, 1);
  l_b += __shfl_xor_sync(0xffffffff, l_b, 2);
  const float inv_a = 1.f / l_a, inv_b = 1.f / l_b;
  const int64_t row_stride = static_cast<int64_t>(heads) * D;
  __nv_bfloat16 *oa = out + (static_cast<int64_t>(blockIdx.z) * n_q + ra) * row_stride + blockIdx.y * D;
  __nv_bfloat16 *ob = oa + 8 * row_stride;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int c = nt * 8 + 2 * t4;
    if (ra < n_q)
      *reinterpret_cast<__nv_bfloat162 *>(oa + c) = __floats2bfloat162_rn(o[nt][0] * inv_a, o[nt][1] * inv_a);
    if (rb < n_q)
      *reinterpret_cast<__nv_bfloat162 *>(ob + c) = __floats2bfloat162_rn(o[nt][2] * inv_b, o[nt][3] * inv_b);
  }
}

// one block per (batch, head): the mean over tokens of x (bh, n, D), per column
template <int D> __global__ void sa_mean_kernel(const __nv_bfloat16 *x, float *mean, int n) {
  constexpr int R = kSaThreads / D;
  __shared__ float part[kSaThreads];
  const int col = threadIdx.x % D, r0 = threadIdx.x / D;
  const __nv_bfloat16 *xh = x + static_cast<int64_t>(blockIdx.x) * n * D;
  float s = 0.f;
  for (int r = r0; r < n; r += R)
    s += __bfloat162float(xh[static_cast<int64_t>(r) * D + col]);
  part[threadIdx.x] = s;
  __syncthreads();
  if (threadIdx.x < D) {
    float t = 0.f;
    for (int i = 0; i < R; ++i)
      t += part[i * D + threadIdx.x];
    mean[static_cast<int64_t>(blockIdx.x) * D + threadIdx.x] = t / n;
  }
}

// one warp per token row of x (bh * n, D): less mean (per bh, optional), quantized to int8 with a per-row scale
template <int D>
__global__ void sa_quantize_kernel(const __nv_bfloat16 *x, const float *mean, int8_t *q, float *scale, int64_t rows,
                                   int n) {
  constexpr int PER = D / 32;
  const int64_t row = blockIdx.x * static_cast<int64_t>(kSaWarps) + threadIdx.x / 32;
  if (row >= rows)
    return;
  const int lane = threadIdx.x % 32;
  const float *mu = mean != nullptr ? mean + (row / n) * D : nullptr;
  float v[PER];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int c = lane * PER + i;
    v[i] = __bfloat162float(x[row * D + c]) - (mu != nullptr ? mu[c] : 0.f);
    amax = fmaxf(amax, fabsf(v[i]));
  }
  for (int offset = 16; offset > 0; offset /= 2)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
  const float s = amax / kSaInt8Max, inv = amax > 0.f ? kSaInt8Max / amax : 0.f;
#pragma unroll
  for (int i = 0; i < PER; ++i)
    q[row * D + lane * PER + i] = static_cast<int8_t>(__float2int_rn(v[i] * inv));
  if (lane == 0)
    scale[row] = s;
}

extern "C" int sa_mean_bf16(const void *x, void *mean, int bh, int n, int head_dim, int64_t stream) {
  const auto s = reinterpret_cast<cudaStream_t>(stream);
  if (head_dim == 128)
    sa_mean_kernel<128><<<bh, kSaThreads, 0, s>>>(static_cast<const __nv_bfloat16 *>(x), static_cast<float *>(mean), n);
  else if (head_dim == 64)
    sa_mean_kernel<64><<<bh, kSaThreads, 0, s>>>(static_cast<const __nv_bfloat16 *>(x), static_cast<float *>(mean), n);
  else
    return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int sa_quantize_bf16(const void *x, const void *mean, void *q, void *scale, int64_t rows, int n,
                                int head_dim, int64_t stream) {
  const auto s = reinterpret_cast<cudaStream_t>(stream);
  const int64_t blocks = (rows + kSaWarps - 1) / kSaWarps;
  const auto *xs = static_cast<const __nv_bfloat16 *>(x);
  const auto *mu = static_cast<const float *>(mean);
  if (head_dim == 128)
    sa_quantize_kernel<128><<<blocks, kSaThreads, 0, s>>>(xs, mu, static_cast<int8_t *>(q),
                                                          static_cast<float *>(scale), rows, n);
  else if (head_dim == 64)
    sa_quantize_kernel<64><<<blocks, kSaThreads, 0, s>>>(xs, mu, static_cast<int8_t *>(q),
                                                         static_cast<float *>(scale), rows, n);
  else
    return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(cudaGetLastError());
}

template <int D>
static int sa_launch(const void *q8, const void *qs, const void *k8, const void *ks, const void *v, void *out,
                     int batch, int heads, int n_q, int n_kv, float scale_log2, cudaStream_t s) {
  const cudaError_t attr = cudaFuncSetAttribute(sa_attention_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                SaLayout<D>::total);
  if (attr != cudaSuccess)
    return static_cast<int>(attr);
  const dim3 grid((n_q + kSaRows - 1) / kSaRows, heads, batch);
  sa_attention_kernel<D><<<grid, kSaThreads, SaLayout<D>::total, s>>>(
      static_cast<const int8_t *>(q8), static_cast<const float *>(qs), static_cast<const int8_t *>(k8),
      static_cast<const float *>(ks), static_cast<const __half *>(v), static_cast<__nv_bfloat16 *>(out), n_q, n_kv,
      heads, scale_log2);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int sa_attention(const void *q8, const void *qs, const void *k8, const void *ks, const void *v, void *out,
                            int batch, int heads, int n_q, int n_kv, int head_dim, float scale_log2, int64_t stream) {
  const auto s = reinterpret_cast<cudaStream_t>(stream);
  if (head_dim == 128)
    return sa_launch<128>(q8, qs, k8, ks, v, out, batch, heads, n_q, n_kv, scale_log2, s);
  if (head_dim == 64)
    return sa_launch<64>(q8, qs, k8, ks, v, out, batch, heads, n_q, n_kv, scale_log2, s);
  return static_cast<int>(cudaErrorInvalidValue);
}
