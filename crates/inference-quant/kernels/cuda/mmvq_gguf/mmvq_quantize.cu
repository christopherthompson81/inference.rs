#include "mmvq_gguf.cuh"

// Padding-aware BF16/F16/F32 -> Q8_1 quantize kernels.

// input rows mmvq_tc.cuh's activation layout holds (the mma's B columns)
#define MMVQ_TC_QUANTIZE_ROWS 8

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_bf16(const __nv_bfloat16 *__restrict__ x,
                             void *__restrict__ vy, const int kx,
                             const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? __bfloat162float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_f16(const half *__restrict__ x, void *__restrict__ vy,
                            const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? __half2float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

extern "C" __global__ void
mmvq_gguf_quantize_q8_1_f32(const float *__restrict__ x, void *__restrict__ vy,
                            const int kx, const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? x[iy * kx + ix] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<half &>(y[ib].ds.x) = (half)d;
  reinterpret_cast<half &>(y[ib].ds.y) = (half)sum;
}

// Quantize launchers.

extern "C" void launch_mmvq_gguf_quantize_q8_1_bf16(const void *x, void *vy,
                                                    int kx, int kx_padded,
                                                    int num_rows,
                                                    void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_bf16<<<grid, block, 0, s>>>((const __nv_bfloat16 *)x,
                                                      vy, kx, kx_padded);
}

extern "C" void launch_mmvq_gguf_quantize_q8_1_f16(const void *x, void *vy,
                                                   int kx, int kx_padded,
                                                   int num_rows, void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_f16<<<grid, block, 0, s>>>((const half *)x, vy, kx,
                                                     kx_padded);
}

extern "C" void launch_mmvq_gguf_quantize_q8_1_f32(const void *x, void *vy,
                                                   int kx, int kx_padded,
                                                   int num_rows, void *stream) {
  const int num_blocks_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  dim3 grid(num_blocks_x, num_rows, 1);
  dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mmvq_gguf_quantize_q8_1_f32<<<grid, block, 0, s>>>((const float *)x, vy, kx,
                                                     kx_padded);
}

// mmvq_tc.cuh's activation layout, the mma's B side ready to load: for each 32-value block, an int2 per lane (lane
// 4r + t holds input row r's values 4t..4t+3 and 16+4t..16+4t+3), then after every block the 8 rows' scales, so a
// warp reads a block's fragments as 256 contiguous bytes. Rows past num_rows are zero.
template <typename src_t> static __device__ __forceinline__ float tc_quantize_load(src_t v) { return (float)v; }
template <> __device__ __forceinline__ float tc_quantize_load<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <> __device__ __forceinline__ float tc_quantize_load<half>(half v) { return __half2float(v); }

template <typename src_t>
static __global__ void mmvq_tc_quantize(const src_t *__restrict__ x, void *__restrict__ vy, const int kx,
                                        const int num_rows) {
  const int ib = blockIdx.x;
  const int row = threadIdx.y;
  const int k = threadIdx.x;
  const int nb = gridDim.x;
  const int col = ib * QK8_1 + k;
  const float xi = (row < num_rows && col < kx) ? tc_quantize_load(x[(int64_t)row * kx + col]) : 0.0f;
  const float amax = warp_reduce_max_f32(fabsf(xi));
  const float d = amax / 127.0f;
  __shared__ int8_t q[MMVQ_TC_QUANTIZE_ROWS][QK8_1];
  q[row][k] = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
  int2 *qs = (int2 *)vy;
  float *ds = (float *)(qs + (int64_t)nb * WARP_SIZE);
  if (k == 0) {
    ds[ib * MMVQ_TC_QUANTIZE_ROWS + row] = d;
  }
  __syncthreads();
  if (row == 0) {
    const int r = k >> 2;
    const int t = k & 3;
    qs[(int64_t)ib * WARP_SIZE + k] = make_int2(*(const int *)&q[r][4 * t], *(const int *)&q[r][16 + 4 * t]);
  }
}

template <typename src_t>
static void launch_mmvq_tc_quantize(const void *x, void *vy, int kx, int num_rows, void *stream) {
  const dim3 grid((kx + QK8_1 - 1) / QK8_1, 1, 1);
  const dim3 block(QK8_1, MMVQ_TC_QUANTIZE_ROWS, 1);
  mmvq_tc_quantize<src_t><<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>((const src_t *)x, vy, kx,
                                                                                 num_rows);
}

extern "C" void launch_mmvq_tc_quantize_bf16(const void *x, void *vy, int kx, int num_rows, void *stream) {
  launch_mmvq_tc_quantize<__nv_bfloat16>(x, vy, kx, num_rows, stream);
}
extern "C" void launch_mmvq_tc_quantize_f16(const void *x, void *vy, int kx, int num_rows, void *stream) {
  launch_mmvq_tc_quantize<half>(x, vy, kx, num_rows, stream);
}
extern "C" void launch_mmvq_tc_quantize_f32(const void *x, void *vy, int kx, int num_rows, void *stream) {
  launch_mmvq_tc_quantize<float>(x, vy, kx, num_rows, stream);
}
