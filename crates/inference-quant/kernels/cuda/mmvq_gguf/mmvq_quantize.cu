#include "mmvq_gguf.cuh"

// Padding-aware BF16/F16/F32 -> Q8_1 quantize kernels.

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
