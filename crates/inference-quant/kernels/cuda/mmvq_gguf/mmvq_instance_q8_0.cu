#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                     vec_dot_q8_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                        vec_dot_q8_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                        vec_dot_q8_0_q8_1)
MMVQ_LAUNCHER_PLAIN(q8_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q8_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q8_0, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q8_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q8_0)

// Q8_0 rows against unquantized activations, in F32: for reads whose
// activations 8 bits per 32 values perturb too much. `WARPS` warps share a row
// (lane l takes value l of every WARPS-th block), and a block of threads covers
// `ROWS` rows; short rows take a warp each.
#define Q8_0_EXACT_SHORT_COLS 1024

// two consecutive activations as floats; `col` is even, so the pair is aligned
static __device__ __forceinline__ float2 q8_0_exact_pair(const float *y,
                                                         size_t at) {
  return *reinterpret_cast<const float2 *>(y + at);
}
static __device__ __forceinline__ float2 q8_0_exact_pair(const __nv_bfloat16 *y,
                                                         size_t at) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(y + at));
}

template <typename T, int B, int WARPS, int ROWS>
static __global__ void
q8_0_exact_rows(const block_q8_0 *__restrict__ w, const T *__restrict__ y,
                T *__restrict__ dst, int ncols, int nrows) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int row = blockIdx.x * ROWS + warp / WARPS;
  const int part = warp % WARPS;
  const int nblocks = ncols / QK8_0;
  // a warp covers two blocks a step: lanes 0-15 the first, 16-31 the second,
  // two values each
  const int half_warp = lane / (WARP_SIZE / 2);
  const int pair = lane % (WARP_SIZE / 2);
  float acc[B];
#pragma unroll
  for (int b = 0; b < B; ++b)
    acc[b] = 0.0f;
  if (row < nrows) {
    const block_q8_0 *wr = w + (size_t)row * nblocks;
    for (int i = 2 * part + half_warp; i < nblocks; i += 2 * WARPS) {
      const float d = __half2float(wr[i].d);
      const char2 q = *reinterpret_cast<const char2 *>(wr[i].qs + 2 * pair);
      const float v0 = d * (float)q.x, v1 = d * (float)q.y;
      const size_t col = (size_t)i * QK8_0 + 2 * pair;
#pragma unroll
      for (int b = 0; b < B; ++b) {
        const float2 x = q8_0_exact_pair(y, (size_t)b * ncols + col);
        acc[b] += v0 * x.x + v1 * x.y;
      }
    }
  }
#pragma unroll
  for (int b = 0; b < B; ++b) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
      acc[b] += __shfl_xor_sync(0xffffffff, acc[b], off);
  }
  if (WARPS == 1) {
    if (lane < B && row < nrows) {
      float s = 0.0f;
#pragma unroll
      for (int b = 0; b < B; ++b)
        s = lane == b ? acc[b] : s;
      dst[(size_t)lane * nrows + row] = (T)s;
    }
    return;
  }
  __shared__ float partial[WARPS * ROWS][B];
  if (lane == 0) {
#pragma unroll
    for (int b = 0; b < B; ++b)
      partial[warp][b] = acc[b];
  }
  __syncthreads();
  if (threadIdx.x < B && row < nrows) {
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < WARPS; ++k)
      s += partial[k][threadIdx.x];
    dst[(size_t)threadIdx.x * nrows + row] = (T)s;
  }
}

template <typename T, int B>
static void q8_0_exact_launch(const block_q8_0 *w, const T *y, T *d, int ncols,
                              int nrows, cudaStream_t s) {
  if (ncols <= Q8_0_EXACT_SHORT_COLS) {
    constexpr int ROWS = 4;
    q8_0_exact_rows<T, B, 1, ROWS>
        <<<(nrows + ROWS - 1) / ROWS, ROWS * WARP_SIZE, 0, s>>>(w, y, d, ncols,
                                                                nrows);
  } else {
    constexpr int WARPS = 8;
    q8_0_exact_rows<T, B, WARPS, 1>
        <<<nrows, WARPS * WARP_SIZE, 0, s>>>(w, y, d, ncols, nrows);
  }
}

#define Q8_0_EXACT_LAUNCHER(dst_tag, c_type)                                   \
  extern "C" void launch_mmvq_gguf_q8_0_##dst_tag##_exact(                     \
      const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x,     \
      int b_size, void *stream) {                                              \
    cudaStream_t s = static_cast<cudaStream_t>(stream);                        \
    const block_q8_0 *w = (const block_q8_0 *)vx;                              \
    const c_type *y = (const c_type *)vy;                                      \
    c_type *d = (c_type *)dst;                                                 \
    switch (b_size) {                                                          \
    case 1:                                                                    \
      q8_0_exact_launch<c_type, 1>(w, y, d, ncols_x, nrows_x, s);              \
      break;                                                                   \
    case 2:                                                                    \
      q8_0_exact_launch<c_type, 2>(w, y, d, ncols_x, nrows_x, s);              \
      break;                                                                   \
    case 3:                                                                    \
      q8_0_exact_launch<c_type, 3>(w, y, d, ncols_x, nrows_x, s);              \
      break;                                                                   \
    case 4:                                                                    \
      q8_0_exact_launch<c_type, 4>(w, y, d, ncols_x, nrows_x, s);              \
      break;                                                                   \
    case 5:                                                                    \
      q8_0_exact_launch<c_type, 5>(w, y, d, ncols_x, nrows_x, s);              \
      break;                                                                   \
    case 6:                                                                    \
      q8_0_exact_launch<c_type, 6>(w, y, d, ncols_x, nrows_x, s);              \
      break;                                                                   \
    case 7:                                                                    \
      q8_0_exact_launch<c_type, 7>(w, y, d, ncols_x, nrows_x, s);              \
      break;                                                                   \
    default:                                                                   \
      q8_0_exact_launch<c_type, 8>(w, y, d, ncols_x, nrows_x, s);              \
      break;                                                                   \
    }                                                                          \
  }

Q8_0_EXACT_LAUNCHER(bf16, __nv_bfloat16)
Q8_0_EXACT_LAUNCHER(f32, float)
