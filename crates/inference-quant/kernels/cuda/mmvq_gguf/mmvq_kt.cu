// Matvec and dequantize kernels for ik_llama.cpp's trellis GGUF types (IQ1_KT through IQ4_KT), Q8_1 activations.
// Adapted from ik_llama.cpp (MIT, Iwan Kawrakow): iqk_mmvq_templates.cuh, mmvq-instance-iq*_kt.cu and convert.cu.
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <stdint.h>

#define WARP_SIZE 32
#define QK_K 256
#define QK8_1 32
#define QI4_XS (QK_K / 8)
#define VDR_KT 4
#define KT_ROW_META 4
#define KT_TAIL_BLOCK 32
#define KT_INDEX_OFFSET 4096

typedef struct {
  half2 ds;
  int8_t qs[QK8_1];
} block_q8_1;

typedef struct { uint8_t sh[QK_K / 32]; uint8_t ql[QK_K / 8]; uint8_t qh[QK_K / 16]; } block_iq1_kt;
typedef struct { uint8_t scales[QK_K / 64]; uint8_t ql[QK_K / 4]; } block_iq2_kt;
typedef struct { uint8_t scales[QK_K / 64]; uint8_t ql[QK_K / 4]; uint8_t qh[QK_K / 8]; } block_iq3_kt;
typedef struct { uint32_t qs[QK_K / 8]; } block_iq4_kt;

static __constant__ int8_t iq4k_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                              1,    13,   25,  38,  53,  69,  89,  113};

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

// ggml_row_size for these types: the f32 row scale, whole blocks, then IQ3_KT / IQ4_KT 32-element tail sub-blocks
static __host__ __device__ int64_t kt_row_size(int type_size, bool iq3, int ncols) {
  const int nt = (ncols % QK_K) / KT_TAIL_BLOCK;
  const int64_t tail = iq3 ? (nt + 1) / 2 + 12 * nt : 16 * nt;
  const int64_t bytes = KT_ROW_META + (int64_t)type_size * (ncols / QK_K) + tail;
  return (bytes + 3) / 4 * 4;
}

static __device__ __forceinline__ int trellis_next_int(uint32_t &val) {
  constexpr uint32_t ka = 0xCBAC1FED;
  val = ka * val;
  return ggml_cuda_dp4a(val & 0x3f3f3f3f, 0x01010101, -126);
}

static __device__ __forceinline__ int kt_dot4(uint32_t &val, int q8, int sumi) {
  int v4 = 0;
  for (int k = 0; k < 4; ++k) {
    v4 |= (trellis_next_int(val) & 0xff) << 8 * k;
  }
  return ggml_cuda_dp4a(v4, q8, sumi);
}

static __device__ __forceinline__ int kt_dot4_signed(uint32_t &val, uint32_t signs, int q8, int sumi) {
  int v4 = 0;
  for (int k = 0; k < 4; ++k) {
    v4 |= abs(trellis_next_int(val)) << 8 * k;
  }
  v4 = __vsub4(v4 ^ signs, signs);
  return ggml_cuda_dp4a(v4, q8, sumi);
}

struct kt_iq1 {
  static constexpr int type_size = sizeof(block_iq1_kt);
  static constexpr bool iq3 = false;
  static constexpr bool has_tail = false;
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs,
                                                 float *result) {
    const float scale = *(const float *)vbq;
    const block_iq1_kt *bq1 = (const block_iq1_kt *)((const char *)vbq + KT_ROW_META) + kbx;
    const int ib32 = iqs / 4;
    const int32_t *q8 = (const int *)bq8_1[ib32].qs;
    const float dl = scale * iq4k_values[bq1->sh[ib32] & 0xf];
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
      uint32_t val = bq1->ql[4 * ib32 + j] + KT_INDEX_OFFSET +
                     ((bq1->qh[4 * (ib32 % 4) + j] << (8 - 4 * (ib32 / 4))) & 0xf00) +
                     ((bq1->sh[ib32] << (8 - j)) & 0x1000);
      sumi = kt_dot4(val, q8[2 * j + 0], sumi);
      sumi = kt_dot4(val, q8[2 * j + 1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct kt_iq2 {
  static constexpr int type_size = sizeof(block_iq2_kt);
  static constexpr bool iq3 = false;
  static constexpr bool has_tail = false;
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs,
                                                 float *result) {
    const float scale = *(const float *)vbq;
    const block_iq2_kt *bq2 = (const block_iq2_kt *)((const char *)vbq + KT_ROW_META) + kbx;
    const int ib32 = iqs / 4;
    const int32_t *q8 = (const int *)bq8_1[ib32].qs;
    const int ls = iq4k_values[(bq2->scales[ib32 % 4] >> 4 * (ib32 / 4)) & 0xf];
    const float dl = scale * ls * 1.05f;
    const uint16_t *ql = (const uint16_t *)bq2->ql;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
      uint32_t val = ql[4 * ib32 + j] + KT_INDEX_OFFSET;
      sumi = kt_dot4(val, q8[2 * j + 0], sumi);
      sumi = kt_dot4(val, q8[2 * j + 1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct kt_iq3 {
  static constexpr int type_size = sizeof(block_iq3_kt);
  static constexpr bool iq3 = true;
  static constexpr bool has_tail = true;
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs,
                                                 float *result) {
    const float scale = *(const float *)vbq;
    const block_iq3_kt *bq3 = (const block_iq3_kt *)((const char *)vbq + KT_ROW_META) + kbx;
    const int ib32 = iqs / 4;
    const int32_t *q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (bq3->scales[ib32 % 4] >> 4 * (ib32 / 4)) & 0xf;
    const float dl = scale * ls * 1.01f;
    const uint16_t *ql = (const uint16_t *)bq3->ql;
    const uint32_t mask = 0x01010101 << ib32;
    const uint32_t *qh = (const uint32_t *)bq3->qh;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
      uint32_t val = ql[4 * ib32 + j] + KT_INDEX_OFFSET;
      sumi = kt_dot4_signed(val, __vcmpne4(qh[2 * j + 0] & mask, 0), q8[2 * j + 0], sumi);
      sumi = kt_dot4_signed(val, __vcmpne4(qh[2 * j + 1] & mask, 0), q8[2 * j + 1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs,
                                                      int nt, float *result) {
    const int ib32 = iqs / 4;
    if (ib32 >= nt) return;
    const float scale = *(const float *)vbq;
    const uint8_t *tail = (const uint8_t *)((const block_iq3_kt *)((const char *)vbq + KT_ROW_META) + kbx);
    const uint8_t *ql = tail + 8 * ib32;
    const uint8_t *qh = tail + 8 * nt + 4 * ib32;
    const uint8_t *scales = tail + 12 * nt;
    const int32_t *q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (scales[ib32 / 2] >> 4 * (ib32 & 1)) & 0xf;
    const float dl = scale * ls * 1.01f;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
      uint32_t val = (ql[2 * j] | (ql[2 * j + 1] << 8)) + KT_INDEX_OFFSET;
      const uint32_t sb = qh[j];
      sumi = kt_dot4_signed(val, __vcmpne4(((sb & 0x0f) * 0x00204081) & 0x01010101, 0), q8[2 * j + 0], sumi);
      sumi = kt_dot4_signed(val, __vcmpne4(((sb >> 4) * 0x00204081) & 0x01010101, 0), q8[2 * j + 1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
  }
};

struct kt_iq4 {
  static constexpr int type_size = sizeof(block_iq4_kt);
  static constexpr bool iq3 = false;
  static constexpr bool has_tail = true;
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs,
                                                 float *result) {
    const float scale = *(const float *)vbq;
    const block_iq4_kt *bq4 = (const block_iq4_kt *)((const char *)vbq + KT_ROW_META) + kbx;
    const int ib32 = iqs / 4;
    const int32_t *q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (bq4->qs[ib32] & 0xff) >> 1;
    const float dl = scale * (ls - 64);
    const uint32_t idx0 = ((bq4->qs[ib32] & 1) << 15) + KT_INDEX_OFFSET;
    const uint8_t *ql = (const uint8_t *)(bq4->qs + 8) + 8 * ib32;
    const uint8_t *qh = (const uint8_t *)(bq4->qs + 8) + 64 + 8 * (ib32 % 4);
    const int shift1 = 8 - 4 * (ib32 / 4);
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
      const uint32_t sh = bq4->qs[ib32] >> (8 + 3 * j);
      uint32_t val = ql[j] + ((qh[j] << shift1) & 0xf00) + ((sh & 7) << 12) + idx0;
      sumi = kt_dot4(val, q8[j], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs,
                                                      int nt, float *result) {
    const int ib32 = iqs / 4;
    if (ib32 >= nt) return;
    const float scale = *(const float *)vbq;
    const uint8_t *tail = (const uint8_t *)((const block_iq4_kt *)((const char *)vbq + KT_ROW_META) + kbx);
    const uint32_t sh = *(const uint32_t *)(tail + 16 * ib32);
    const uint8_t *ql = tail + 16 * ib32 + 4;
    const uint8_t *qh = tail + 16 * ib32 + 12;
    const int32_t *q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (sh & 0xff) >> 1;
    const float dl = scale * (ls - 64);
    const uint32_t idx0 = ((sh & 1) << 15) + KT_INDEX_OFFSET;
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
      const uint32_t shj = sh >> (8 + 3 * j);
      uint32_t val = ql[j] + (((qh[j / 2] >> 4 * (j & 1)) & 0xf) << 8) + ((shj & 7) << 12) + idx0;
      sumi = kt_dot4(val, q8[j], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
  }
};

template <typename dst_t> static __device__ __forceinline__ dst_t from_float(float x) { return dst_t(x); }

// iqk_mul_mat_vec_q_kernel: rows are `row_size` bytes apart, each starting with its f32 scale
template <typename kt, int ncols_y, typename dst_t>
static __global__ void mmvq_kt_kernel(const void *__restrict__ vx, const void *__restrict__ vy, dst_t *__restrict__ dst,
                                      const int ncols_x, const int nrows_x, const int stride_col_y,
                                      const int stride_col_dst) {
  constexpr int qk = QK_K;
  constexpr int qi = QI4_XS;
  constexpr int vdr = VDR_KT;
  constexpr int nwarps = ncols_y <= 4 ? 4 : 2;
  constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
  const int64_t row_size = kt_row_size(kt::type_size, kt::iq3, ncols_x);

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
static void launch_mmvq_kt_cols(const void *vx, const void *vy, dst_t *dst, int ncols_x, int nrows_x,
                                int stride_col_y, int stride_col_dst, cudaStream_t stream) {
  constexpr int nwarps = ncols_y <= 4 ? 4 : 2;
  constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
  const dim3 grid((nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block, 1, 1);
  const dim3 block(WARP_SIZE, nwarps, 1);
  mmvq_kt_kernel<kt, ncols_y, dst_t>
      <<<grid, block, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <typename kt, typename dst_t>
static void launch_mmvq_kt(const void *vx, const void *vy, void *dst, int ncols_x, int nrows_x, int stride_col_y,
                           int stride_col_dst, int b_size, void *stream) {
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  dst_t *out = (dst_t *)dst;
  switch (b_size) {
  case 1: launch_mmvq_kt_cols<kt, 1>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 2: launch_mmvq_kt_cols<kt, 2>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 3: launch_mmvq_kt_cols<kt, 3>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 4: launch_mmvq_kt_cols<kt, 4>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 5: launch_mmvq_kt_cols<kt, 5>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 6: launch_mmvq_kt_cols<kt, 6>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  case 7: launch_mmvq_kt_cols<kt, 7>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  default: launch_mmvq_kt_cols<kt, 8>(vx, vy, out, ncols_x, nrows_x, stride_col_y, stride_col_dst, s); break;
  }
}

// convert.cu dequantize_block_iq*_kt: one 32-thread block per 256 elements of a row, eight values per thread
template <typename dst_t>
static __global__ void dequantize_iq1_kt(const void *__restrict__ vx, dst_t *__restrict__ yy, int64_t n_per_row) {
  const int64_t row_size = kt_row_size(sizeof(block_iq1_kt), false, n_per_row);
  const int64_t bpr = n_per_row / QK_K;
  const int64_t row = blockIdx.x / bpr;
  const int64_t i = blockIdx.x % bpr;
  const char *cx = (const char *)vx + row * row_size;
  const float scale = *(const float *)cx;
  const block_iq1_kt *x = (const block_iq1_kt *)(cx + KT_ROW_META);
  const int ib = threadIdx.x;
  dst_t *y = yy + row * n_per_row + i * QK_K + 8 * ib;
  uint32_t idx = (x[i].ql[ib] | ((x[i].qh[ib % 16] << (8 - 4 * (ib / 16))) & 0xf00) |
                  ((x[i].sh[ib / 4] << (8 - (ib % 4))) & 0x1000)) + KT_INDEX_OFFSET;
  const float dl = scale * iq4k_values[x[i].sh[ib / 4] & 0xf];
  for (int j = 0; j < 8; ++j) {
    y[j] = from_float<dst_t>(dl * trellis_next_int(idx));
  }
}

template <typename dst_t>
static __global__ void dequantize_iq2_kt(const void *__restrict__ vx, dst_t *__restrict__ yy, int64_t n_per_row) {
  const int64_t row_size = kt_row_size(sizeof(block_iq2_kt), false, n_per_row);
  const int64_t bpr = n_per_row / QK_K;
  const int64_t row = blockIdx.x / bpr;
  const int64_t i = blockIdx.x % bpr;
  const char *cx = (const char *)vx + row * row_size;
  const float scale = *(const float *)cx;
  const block_iq2_kt *x = (const block_iq2_kt *)(cx + KT_ROW_META);
  const int ib = threadIdx.x;
  dst_t *y = yy + row * n_per_row + i * QK_K + 8 * ib;
  uint32_t idx = ((const uint16_t *)x[i].ql)[ib] + KT_INDEX_OFFSET;
  const float dl = scale * iq4k_values[(x[i].scales[(ib / 4) % 4] >> 4 * (ib / 16)) & 0xf] * 1.05f;
  for (int j = 0; j < 8; ++j) {
    y[j] = from_float<dst_t>(dl * trellis_next_int(idx));
  }
}

template <typename dst_t>
static __global__ void dequantize_iq3_kt(const void *__restrict__ vx, dst_t *__restrict__ yy, int64_t n_per_row) {
  const int64_t row_size = kt_row_size(sizeof(block_iq3_kt), true, n_per_row);
  const int64_t bpr = (n_per_row + QK_K - 1) / QK_K;
  const int64_t row = blockIdx.x / bpr;
  const int64_t i = blockIdx.x % bpr;
  const char *cx = (const char *)vx + row * row_size;
  const float scale = *(const float *)cx;
  const block_iq3_kt *x = (const block_iq3_kt *)(cx + KT_ROW_META);
  const int ib = threadIdx.x;
  dst_t *y = yy + row * n_per_row + i * QK_K + 8 * ib;
  if (i < n_per_row / QK_K) {
    uint32_t idx = ((const uint16_t *)x[i].ql)[ib] + KT_INDEX_OFFSET;
    const float dl = scale * ((x[i].scales[(ib / 4) % 4] >> 4 * (ib / 16)) & 0xf) * 1.01f;
    const uint8_t mask = 1 << (ib / 4);
    for (int j = 0; j < 8; ++j) {
      const float v = dl * abs(trellis_next_int(idx));
      y[j] = from_float<dst_t>(x[i].qh[(8 * ib + j) % 32] & mask ? -v : v);
    }
    return;
  }
  const int nt = (n_per_row % QK_K) / KT_TAIL_BLOCK;
  if (ib / 4 >= nt) return;
  const uint8_t *ql = (const uint8_t *)(x + i);
  const uint8_t *qh = ql + 8 * nt;
  const uint8_t *scales = qh + 4 * nt;
  uint32_t idx = (ql[2 * ib] | (ql[2 * ib + 1] << 8)) + KT_INDEX_OFFSET;
  const float dl = scale * ((scales[ib / 8] >> 4 * ((ib / 4) & 1)) & 0xf) * 1.01f;
  const uint32_t sgn = qh[ib];
  for (int j = 0; j < 8; ++j) {
    const float v = dl * abs(trellis_next_int(idx));
    y[j] = from_float<dst_t>(sgn & (1 << j) ? -v : v);
  }
}

template <typename dst_t>
static __global__ void dequantize_iq4_kt(const void *__restrict__ vx, dst_t *__restrict__ yy, int64_t n_per_row) {
  constexpr int kNumGroups = 64;
  const int64_t row_size = kt_row_size(sizeof(block_iq4_kt), false, n_per_row);
  const int64_t bpr = (n_per_row + QK_K - 1) / QK_K;
  const int64_t row = blockIdx.x / bpr;
  const int64_t i = blockIdx.x % bpr;
  const float *dptr = (const float *)((const char *)vx + row * row_size);
  const float scale = dptr[0];
  const block_iq4_kt *x = (const block_iq4_kt *)(dptr + 1);
  const int ib = threadIdx.x;
  dst_t *y = yy + row * n_per_row + i * QK_K + 8 * ib;
  const int ib32 = ib / 4;
  const int ig = ib % 4;
  const int jj = ib32 * 8 + 2 * ig;
  const uint8_t *ql;
  uint32_t qh1, qh2, sh;
  int qj;
  if (i < n_per_row / QK_K) {
    const uint32_t *shb = x[i].qs;
    ql = (const uint8_t *)(shb + 8);
    const uint8_t *qh = ql + kNumGroups;
    qj = jj;
    qh1 = (qh[(jj + 0) % (kNumGroups / 2)] << (8 - 4 * ((jj + 0) / (kNumGroups / 2)))) & 0xf00;
    qh2 = (qh[(jj + 1) % (kNumGroups / 2)] << (8 - 4 * ((jj + 1) / (kNumGroups / 2)))) & 0xf00;
    sh = shb[ib32];
  } else {
    const int nt = (n_per_row % QK_K) / KT_TAIL_BLOCK;
    if (ib32 >= nt) return;
    const uint8_t *tail = (const uint8_t *)(x + i);
    ql = tail + 16 * ib32 + 4;
    const uint8_t *qh = tail + 16 * ib32 + 12;
    qj = 2 * ig;
    qh1 = (qh[qj / 2] & 0x0f) << 8;
    qh2 = (qh[qj / 2] & 0xf0) << 4;
    sh = *(const uint32_t *)(tail + 16 * ib32);
  }
  const uint32_t offset = sh & 1 ? KT_INDEX_OFFSET + 32768 : KT_INDEX_OFFSET;
  uint32_t idx1 = ql[qj + 0] + qh1 + (((sh >> (8 + 6 * ig + 0)) & 7) << 12) + offset;
  uint32_t idx2 = ql[qj + 1] + qh2 + (((sh >> (8 + 6 * ig + 3)) & 7) << 12) + offset;
  const float dl = scale * ((int)((sh & 0xff) >> 1) - 64);
  for (int j = 0; j < 4; ++j) {
    y[j + 0] = from_float<dst_t>(dl * trellis_next_int(idx1));
    y[j + 4] = from_float<dst_t>(dl * trellis_next_int(idx2));
  }
}

#define MMVQ_KT_LAUNCHER(tag, kt, dst_tag, dst_c_type)                                                          \
  extern "C" void launch_mmvq_gguf_##tag##_##dst_tag##_plain(const void *vx, const void *vy, void *dst,        \
                                                            int ncols_x, int nrows_x, int stride_col_y,        \
                                                            int stride_col_dst, int b_size, void *stream) {   \
    launch_mmvq_kt<kt, dst_c_type>(vx, vy, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, b_size, stream); \
  }

#define DEQUANTIZE_KT_LAUNCHER(tag, dst_tag, dst_c_type)                                                        \
  extern "C" void launch_dequantize_##tag##_##dst_tag(const void *vx, void *dst, int64_t nrows, int64_t ncols,  \
                                                      void *stream) {                                          \
    const int64_t blocks = nrows * ((ncols + QK_K - 1) / QK_K);                                                \
    dequantize_##tag<dst_c_type>                                                                              \
        <<<(unsigned int)blocks, WARP_SIZE, 0, static_cast<cudaStream_t>(stream)>>>(vx, (dst_c_type *)dst, ncols); \
  }

#define KT_LAUNCHERS(tag, kt)                         \
  MMVQ_KT_LAUNCHER(tag, kt, bf16, __nv_bfloat16)      \
  MMVQ_KT_LAUNCHER(tag, kt, f16, half)                \
  MMVQ_KT_LAUNCHER(tag, kt, f32, float)               \
  DEQUANTIZE_KT_LAUNCHER(tag, bf16, __nv_bfloat16)    \
  DEQUANTIZE_KT_LAUNCHER(tag, f16, half)              \
  DEQUANTIZE_KT_LAUNCHER(tag, f32, float)

KT_LAUNCHERS(iq1_kt, kt_iq1)
KT_LAUNCHERS(iq2_kt, kt_iq2)
KT_LAUNCHERS(iq3_kt, kt_iq3)
KT_LAUNCHERS(iq4_kt, kt_iq4)
