// Shared pieces of the matvec kernels for GGUF types whose rows are read by byte stride (ik_llama.cpp's types).
// Adapted from ik_llama.cpp (MIT, Iwan Kawrakow): iqk_mmvq_templates.cuh.
#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#define WARP_SIZE 32
#define QK_K 256
#define QK8_1 32
#define QI4_XS (QK_K / 8)
#define MMVQ_ROWS_VDR 4
// ik_llama.cpp tail sub-blocks after a row's whole blocks
#define KT_TAIL_BLOCK 32

typedef struct {
  half2 ds;
  int8_t qs[QK8_1];
} block_q8_1;

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE);
  }
  return x;
}

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  const int8_t *a8 = (const int8_t *)&a;
  const int8_t *b8 = (const int8_t *)&b;
  return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

template <typename dst_t> static __device__ __forceinline__ dst_t from_float(float x) { return dst_t(x); }

// iqk_mul_mat_vec_q_kernel: rows are `T::row_size` bytes apart; T supplies the vec_dot (and tail) over one row
template <typename kt, int ncols_y, typename dst_t>
static __global__ void mmvq_rows_kernel(const void *__restrict__ vx, const void *__restrict__ vy, dst_t *__restrict__ dst,
                                      const int ncols_x, const int nrows_x, const int stride_col_y,
                                      const int stride_col_dst) {
  constexpr int qk = QK_K;
  constexpr int qi = QI4_XS;
  constexpr int vdr = MMVQ_ROWS_VDR;
  constexpr int nwarps = ncols_y <= 4 ? 4 : 2;
  constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
  const int64_t row_size = kt::row_size(ncols_x);

  float tmp[ncols_y][rows_per_cuda_block] = {{0.0f}};
  const block_q8_1 *y = (const block_q8_1 *)vy;

  int kbx = tid / (qi / vdr);
  for (; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        if (row0 + i < nrows_x) {
          kt::vec_dot((const char *)vx + (row0 + i) * row_size, &y[j * stride_col_y + kby], kbx, kqs, &tmp[j][i]);
        }
      }
    }
  }
  if constexpr (kt::has_tail) {
    const int nt = (ncols_x % qk) / KT_TAIL_BLOCK;
    if (nt > 0 && kbx == blocks_per_row_x) {
      const int kby = kbx * (qk / QK8_1);
      const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
      for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
          if (row0 + i < nrows_x) {
            kt::vec_dot_tail((const char *)vx + (row0 + i) * row_size, &y[j * stride_col_y + kby], kbx, kqs, nt,
                             &tmp[j][i]);
          }
        }
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      tmp[j][i] = warp_reduce_sum(tmp[j][i]);
    }
    if (threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_x) {
      dst[j * stride_col_dst + row0 + threadIdx.x] = from_float<dst_t>(tmp[j][threadIdx.x]);
    }
  }
}

template <typename kt, int ncols_y, typename dst_t>
static void launch_mmvq_rows_cols(const void *vx, const void *vy, dst_t *dst, int ncols_x, int nrows_x,
                                int stride_col_y, int stride_col_dst, cudaStream_t stream) {
  constexpr int nwarps = ncols_y <= 4 ? 4 : 2;
  constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
  const dim3 grid((nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block, 1, 1);
  const dim3 block(WARP_SIZE, nwarps, 1);
  mmvq_rows_kernel<kt, ncols_y, dst_t>
      <<<grid, block, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <typename kt, typename dst_t>
static void launch_mmvq_rows(const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x, int stride_col_y,
                           int stride_col_dst, int b_size, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  dst_t *out = (dst_t *)dst;
  switch (b_size) {
  case 1: launch_mmvq_rows_cols<kt, 1>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 2: launch_mmvq_rows_cols<kt, 2>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 3: launch_mmvq_rows_cols<kt, 3>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 4: launch_mmvq_rows_cols<kt, 4>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 5: launch_mmvq_rows_cols<kt, 5>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 6: launch_mmvq_rows_cols<kt, 6>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 7: launch_mmvq_rows_cols<kt, 7>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  default: launch_mmvq_rows_cols<kt, 8>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  }
}

#define MMVQ_ROWS_LAUNCHER(tag, traits, dst_tag, dst_c_type)                                                    \
  extern "C" void launch_mmvq_gguf_##tag##_##dst_tag##_plain(const void *vx, const void *vy, void *dst,        \
                                                            int ncols_x, int nrows_x, int stride_col_y,        \
                                                            int stride_col_dst, int b_size, void *stream) {   \
    launch_mmvq_rows<traits, dst_c_type>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, b_size,    \
                                         stream);                                                              \
  }
