#include "gdn_common.cuh"

template <typename T>
__global__ void
gdn_rmsnorm_gated_kernel(const T *__restrict__ x, const T *__restrict__ gate,
                         const T *__restrict__ weight, T *__restrict__ output,
                         int rows, int hidden_dim, int outer_dim_1,
                         int outer_dim_2, int64_t x_stride_0,
                         int64_t x_stride_1, int64_t x_stride_2,
                         int64_t x_stride_3, int64_t gate_stride_0,
                         int64_t gate_stride_1, int64_t gate_stride_2,
                         int64_t gate_stride_3, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_0 = row / outer_plane;
  const int outer_1 = (row / outer_dim_2) % outer_dim_1;
  const int outer_2 = row % outer_dim_2;
  const size_t x_row_offset = (size_t)outer_0 * x_stride_0 +
                              (size_t)outer_1 * x_stride_1 +
                              (size_t)outer_2 * x_stride_2;
  const size_t gate_row_offset = (size_t)outer_0 * gate_stride_0 +
                                 (size_t)outer_1 * gate_stride_1 +
                                 (size_t)outer_2 * gate_stride_2;
  T *out_row = output + (size_t)row * hidden_dim;

  float sum = 0.0f;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float x_val = (float)x[x_row_offset + (size_t)i * x_stride_3];
    sum = __fmaf_rn(x_val, x_val, sum);
  }

  __shared__ float smem[256];
  smem[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(smem[0] / (float)hidden_dim + eps);
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    const float gate_val =
        (float)gate[gate_row_offset + (size_t)i * gate_stride_3];
    const float out =
        (float)x[x_row_offset + (size_t)i * x_stride_3] * inv_rms *
        (float)weight[i] * gdn_silu(gate_val);
    out_row[i] = (T)out;
  }
}

template <typename T, int HIDDEN_DIM, int ROWS_PER_BLOCK>
__global__ void gdn_rmsnorm_gated_warp_kernel(
    const T *__restrict__ x, const T *__restrict__ gate,
    const T *__restrict__ weight, T *__restrict__ output, int rows,
    int outer_dim_1, int outer_dim_2, int64_t x_stride_0,
    int64_t x_stride_1, int64_t x_stride_2, int64_t gate_stride_0,
    int64_t gate_stride_1, int64_t gate_stride_2, float eps) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int row = blockIdx.x * ROWS_PER_BLOCK + warp;
  if (row >= rows) {
    return;
  }

  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_0 = row / outer_plane;
  const int outer_1 = (row / outer_dim_2) % outer_dim_1;
  const int outer_2 = row % outer_dim_2;
  const T *x_row = x + (size_t)outer_0 * x_stride_0 +
                   (size_t)outer_1 * x_stride_1 +
                   (size_t)outer_2 * x_stride_2;
  const T *gate_row = gate + (size_t)outer_0 * gate_stride_0 +
                      (size_t)outer_1 * gate_stride_1 +
                      (size_t)outer_2 * gate_stride_2;
  T *out_row = output + (size_t)row * HIDDEN_DIM;

  float values[HIDDEN_DIM / 32];
  float sum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    const float value = (float)x_row[i];
    values[i / 32] = value;
    sum = __fmaf_rn(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  const float inv_rms =
      rsqrtf(__shfl_sync(0xffffffff, sum, 0) / (float)HIDDEN_DIM + eps);

#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    out_row[i] = (T)(values[i / 32] * inv_rms * (float)weight[i] *
                     gdn_silu((float)gate_row[i]));
  }
}

template <int ROWS_PER_BLOCK>
__global__ void gdn_rmsnorm_gated_quantized_warp_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ weight,
    gdn_fp8_e4m3 *__restrict__ output, float *__restrict__ scales, int rows,
    int groups, int scale_stride_m, int scale_layout, int outer_dim_1,
    int outer_dim_2, int64_t x_stride_0, int64_t x_stride_1,
    int64_t x_stride_2, int64_t x_stride_3, int64_t gate_stride_0,
    int64_t gate_stride_1, int64_t gate_stride_2, int64_t gate_stride_3,
    float eps) {
  constexpr int HIDDEN_DIM = GDN_RMSNORM_FAST_HIDDEN;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int row = blockIdx.x * ROWS_PER_BLOCK + warp;
  if (row >= rows) {
    return;
  }

  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_0 = row / outer_plane;
  const int outer_1 = (row / outer_dim_2) % outer_dim_1;
  const int outer_2 = row % outer_dim_2;
  const __nv_bfloat16 *x_row = x + (size_t)outer_0 * x_stride_0 +
                               (size_t)outer_1 * x_stride_1 +
                               (size_t)outer_2 * x_stride_2;
  const __nv_bfloat16 *gate_row = gate + (size_t)outer_0 * gate_stride_0 +
                                  (size_t)outer_1 * gate_stride_1 +
                                  (size_t)outer_2 * gate_stride_2;

  float values[HIDDEN_DIM / 32];
  float sum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    const float value = (float)x_row[(size_t)i * x_stride_3];
    values[i / 32] = value;
    sum = __fmaf_rn(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  const float inv_rms =
      rsqrtf(__shfl_sync(0xffffffff, sum, 0) / (float)HIDDEN_DIM + eps);

  __nv_bfloat16 rounded[HIDDEN_DIM / 32];
  float maximum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    rounded[i / 32] = __float2bfloat16_rn(
        values[i / 32] * inv_rms * (float)weight[i] *
        gdn_silu((float)gate_row[(size_t)i * gate_stride_3]));
    maximum = fmaxf(maximum, fabsf((float)rounded[i / 32]));
  }
  maximum = gdn_warp_max(maximum);
  float quant_scale;
  float inverse_quant_scale;
  gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                       inverse_quant_scale);
  if (lane == 0) {
    scales[gdn_fp8_scale_offset(row, groups, scale_stride_m)] =
        inverse_quant_scale;
  }
  gdn_fp8_e4m3 *out_row = output + (size_t)row * HIDDEN_DIM;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    out_row[i] = gdn_fp8_quantize((float)rounded[i / 32], quant_scale,
                                  inverse_quant_scale, scale_layout);
  }
}

template <typename T>
__global__ void gdn_rmsnorm_gated_rows4_kernel(
    const T *__restrict__ x, const T *__restrict__ gate,
    const T *__restrict__ weight, T *__restrict__ output, int rows,
    float eps) {
  using Vec = gdn_rmsnorm_vec8<T>;
  constexpr int vecs_per_row =
      GDN_RMSNORM_FAST_HIDDEN / GDN_RMSNORM_TILED_VALUES_PER_LANE;
  constexpr unsigned warp_mask = 0xffffffffu;

  const int lane = threadIdx.x;
  const int half_warp = lane / GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int lane_in_half = lane % GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int first_row =
      blockIdx.x * GDN_RMSNORM_TILED_ROWS_PER_BLOCK + half_warp;
  const int second_row = first_row + GDN_RMSNORM_TILED_ROW_PAIR_OFFSET;
  const Vec weight_value =
      reinterpret_cast<const Vec *>(weight)[lane_in_half];

  Vec first_value{};
  Vec second_value{};
  if (first_row < rows) {
    first_value = reinterpret_cast<const Vec *>(x)[first_row * vecs_per_row +
                                                   lane_in_half];
  }
  if (second_row < rows) {
    second_value = reinterpret_cast<const Vec *>(x)[second_row * vecs_per_row +
                                                    lane_in_half];
  }

  float first_sum = 0.0f;
  float second_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
    const float first = (float)first_value.data[i];
    const float second = (float)second_value.data[i];
    first_sum = __fmaf_rn(first, first, first_sum);
    second_sum = __fmaf_rn(second, second, second_sum);
  }
#pragma unroll
  for (int offset = GDN_RMSNORM_TILED_LANES_PER_ROW / 2; offset > 0;
       offset >>= 1) {
    first_sum += __shfl_down_sync(warp_mask, first_sum, offset,
                                  GDN_RMSNORM_TILED_LANES_PER_ROW);
    second_sum += __shfl_down_sync(warp_mask, second_sum, offset,
                                   GDN_RMSNORM_TILED_LANES_PER_ROW);
  }
  const float first_inv_rms =
      rsqrtf(__shfl_sync(warp_mask, first_sum, 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                 (float)GDN_RMSNORM_FAST_HIDDEN +
             eps);
  const float second_inv_rms =
      rsqrtf(__shfl_sync(warp_mask, second_sum, 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                 (float)GDN_RMSNORM_FAST_HIDDEN +
             eps);

  if (first_row < rows) {
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[first_row * vecs_per_row + lane_in_half];
    Vec result;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      result.data[i] =
          (T)((float)first_value.data[i] * first_inv_rms *
              (float)weight_value.data[i] *
              gdn_silu((float)gate_value.data[i]));
    }
    reinterpret_cast<Vec *>(output)[first_row * vecs_per_row + lane_in_half] =
        result;
  }
  if (second_row < rows) {
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[second_row * vecs_per_row + lane_in_half];
    Vec result;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      result.data[i] =
          (T)((float)second_value.data[i] * second_inv_rms *
              (float)weight_value.data[i] *
              gdn_silu((float)gate_value.data[i]));
    }
    reinterpret_cast<Vec *>(
        output)[second_row * vecs_per_row + lane_in_half] = result;
  }
}

__global__ void gdn_rmsnorm_gated_quantized_rows4_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ weight,
    gdn_fp8_e4m3 *__restrict__ output, float *__restrict__ scales, int rows,
    int groups, int scale_stride_m, int scale_layout, float eps) {
  using Vec = gdn_rmsnorm_vec8<__nv_bfloat16>;
  constexpr int vecs_per_row =
      GDN_RMSNORM_FAST_HIDDEN / GDN_RMSNORM_TILED_VALUES_PER_LANE;
  constexpr unsigned warp_mask = 0xffffffffu;

  const int lane = threadIdx.x;
  const int half_warp = lane / GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int lane_in_half = lane % GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int first_row =
      blockIdx.x * GDN_RMSNORM_TILED_ROWS_PER_BLOCK + half_warp;
  const int second_row = first_row + GDN_RMSNORM_TILED_ROW_PAIR_OFFSET;
  const Vec weight_value =
      reinterpret_cast<const Vec *>(weight)[lane_in_half];

  Vec values[2]{};
  if (first_row < rows) {
    values[0] = reinterpret_cast<const Vec *>(x)[first_row * vecs_per_row +
                                                  lane_in_half];
  }
  if (second_row < rows) {
    values[1] = reinterpret_cast<const Vec *>(x)[second_row * vecs_per_row +
                                                  lane_in_half];
  }

  float sums[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
    const float first = (float)values[0].data[i];
    const float second = (float)values[1].data[i];
    sums[0] = __fmaf_rn(first, first, sums[0]);
    sums[1] = __fmaf_rn(second, second, sums[1]);
  }
#pragma unroll
  for (int offset = GDN_RMSNORM_TILED_LANES_PER_ROW / 2; offset > 0;
       offset >>= 1) {
    sums[0] += __shfl_down_sync(warp_mask, sums[0], offset,
                                GDN_RMSNORM_TILED_LANES_PER_ROW);
    sums[1] += __shfl_down_sync(warp_mask, sums[1], offset,
                                GDN_RMSNORM_TILED_LANES_PER_ROW);
  }
  const float inv_rms[2] = {
      rsqrtf(__shfl_sync(warp_mask, sums[0], 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                     (float)GDN_RMSNORM_FAST_HIDDEN +
                 eps),
      rsqrtf(__shfl_sync(warp_mask, sums[1], 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                     (float)GDN_RMSNORM_FAST_HIDDEN +
                 eps)};

#pragma unroll
  for (int row_index = 0; row_index < 2; ++row_index) {
    const int row = row_index == 0 ? first_row : second_row;
    if (row >= rows) {
      continue;
    }
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[row * vecs_per_row + lane_in_half];
    __nv_bfloat16 rounded[GDN_RMSNORM_TILED_VALUES_PER_LANE];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      rounded[i] = __float2bfloat16_rn(
          (float)values[row_index].data[i] * inv_rms[row_index] *
          (float)weight_value.data[i] * gdn_silu((float)gate_value.data[i]));
      maximum = fmaxf(maximum, fabsf((float)rounded[i]));
    }
    maximum = gdn_warp_max(maximum, GDN_RMSNORM_TILED_LANES_PER_ROW);
    float quant_scale;
    float inverse_quant_scale;
    gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                         inverse_quant_scale);
    if (lane_in_half == 0) {
      scales[gdn_fp8_scale_offset(row, groups, scale_stride_m)] =
          inverse_quant_scale;
    }
    gdn_fp8_e4m3 *out = output + (size_t)row * GDN_RMSNORM_FAST_HIDDEN +
                          lane_in_half * GDN_RMSNORM_TILED_VALUES_PER_LANE;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      out[i] = gdn_fp8_quantize((float)rounded[i], quant_scale,
                                inverse_quant_scale, scale_layout);
    }
  }
}

template <typename T>
__host__ __forceinline__ bool gdn_rmsnorm_vec8_aligned(const void *ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(gdn_rmsnorm_vec8<T>) == 0;
}

__host__ __forceinline__ bool gdn_rmsnorm_dense_rows(
    int rows, int outer_dim_1, int outer_dim_2, int64_t stride_0,
    int64_t stride_1, int64_t stride_2) {
  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_dim_0 = rows / outer_plane;
  return (outer_dim_2 <= 1 || stride_2 == GDN_RMSNORM_FAST_HIDDEN) &&
         (outer_dim_1 <= 1 ||
          stride_1 == (int64_t)outer_dim_2 * GDN_RMSNORM_FAST_HIDDEN) &&
         (outer_dim_0 <= 1 ||
          stride_0 == (int64_t)outer_plane * GDN_RMSNORM_FAST_HIDDEN);
}

extern "C" void gdn_rmsnorm_gated(const void *x, const void *gate,
                                  const void *weight, void *output, int rows,
                                  int hidden_dim, int outer_dim_1,
                                  int outer_dim_2, int64_t x_stride_0,
                                  int64_t x_stride_1, int64_t x_stride_2,
                                  int64_t x_stride_3, int64_t gate_stride_0,
                                  int64_t gate_stride_1, int64_t gate_stride_2,
                                  int64_t gate_stride_3, float eps, int dtype,
                                  int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  const bool use_warp_kernel =
      hidden_dim == GDN_RMSNORM_FAST_HIDDEN && x_stride_3 == 1 &&
      gate_stride_3 == 1;
  if (use_warp_kernel) {
    const bool use_rows4_kernel =
        rows >= GDN_RMSNORM_TILED_MIN_ROWS &&
        gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, x_stride_0,
                               x_stride_1, x_stride_2) &&
        gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, gate_stride_0,
                               gate_stride_1, gate_stride_2);
    if (dtype == 0 && use_rows4_kernel &&
        gdn_rmsnorm_vec8_aligned<__half>(x) &&
        gdn_rmsnorm_vec8_aligned<__half>(gate) &&
        gdn_rmsnorm_vec8_aligned<__half>(weight) &&
        gdn_rmsnorm_vec8_aligned<__half>(output)) {
      const dim3 rows4_block(32);
      const dim3 rows4_grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                            GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
      gdn_rmsnorm_gated_rows4_kernel<<<rows4_grid, rows4_block, 0, custream>>>(
          (const __half *)x, (const __half *)gate, (const __half *)weight,
          (__half *)output, rows, eps);
      return;
    }
    if (dtype != 0 && use_rows4_kernel &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(x) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(gate) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(weight) &&
        gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(output)) {
      const dim3 rows4_block(32);
      const dim3 rows4_grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                            GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
      gdn_rmsnorm_gated_rows4_kernel<<<rows4_grid, rows4_block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
          (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows, eps);
      return;
    }

    const dim3 warp_block(32, GDN_RMSNORM_ROWS_PER_BLOCK);
    const dim3 warp_grid((rows + GDN_RMSNORM_ROWS_PER_BLOCK - 1) /
                         GDN_RMSNORM_ROWS_PER_BLOCK);
    if (dtype == 0) {
      gdn_rmsnorm_gated_warp_kernel<__half, GDN_RMSNORM_FAST_HIDDEN,
                                     GDN_RMSNORM_ROWS_PER_BLOCK>
          <<<warp_grid, warp_block, 0, custream>>>(
              (const __half *)x, (const __half *)gate,
              (const __half *)weight, (__half *)output, rows, outer_dim_1,
              outer_dim_2, x_stride_0, x_stride_1, x_stride_2, gate_stride_0,
              gate_stride_1, gate_stride_2, eps);
    } else {
      gdn_rmsnorm_gated_warp_kernel<
          __nv_bfloat16, GDN_RMSNORM_FAST_HIDDEN,
          GDN_RMSNORM_ROWS_PER_BLOCK>
          <<<warp_grid, warp_block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
              (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows,
              outer_dim_1, outer_dim_2, x_stride_0, x_stride_1, x_stride_2,
              gate_stride_0, gate_stride_1, gate_stride_2, eps);
    }
    return;
  }

  dim3 block(128);
  dim3 grid(rows);

  if (dtype == 0) {
    gdn_rmsnorm_gated_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)gate, (const __half *)weight,
        (__half *)output, rows, hidden_dim, outer_dim_1, outer_dim_2,
        x_stride_0, x_stride_1, x_stride_2, x_stride_3, gate_stride_0,
        gate_stride_1, gate_stride_2, gate_stride_3, eps);
  } else {
    gdn_rmsnorm_gated_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
        (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, rows,
        hidden_dim, outer_dim_1, outer_dim_2, x_stride_0, x_stride_1,
        x_stride_2, x_stride_3, gate_stride_0, gate_stride_1, gate_stride_2,
        gate_stride_3, eps);
  }
}

extern "C" void gdn_rmsnorm_gated_quantized_bf16(
    const void *x, const void *gate, const void *weight, void *output,
    float *scales, int rows, int groups, int scale_stride_m,
    int scale_layout, int outer_dim_1, int outer_dim_2, int64_t x_stride_0,
    int64_t x_stride_1, int64_t x_stride_2, int64_t x_stride_3,
    int64_t gate_stride_0, int64_t gate_stride_1, int64_t gate_stride_2,
    int64_t gate_stride_3, float eps, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  const bool use_rows4_kernel =
      rows >= GDN_RMSNORM_TILED_MIN_ROWS &&
      x_stride_3 == 1 && gate_stride_3 == 1 &&
      gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, x_stride_0,
                             x_stride_1, x_stride_2) &&
      gdn_rmsnorm_dense_rows(rows, outer_dim_1, outer_dim_2, gate_stride_0,
                             gate_stride_1, gate_stride_2) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(x) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(gate) &&
      gdn_rmsnorm_vec8_aligned<__nv_bfloat16>(weight);
  if (use_rows4_kernel) {
    const dim3 block(32);
    const dim3 grid((rows + GDN_RMSNORM_TILED_ROWS_PER_BLOCK - 1) /
                    GDN_RMSNORM_TILED_ROWS_PER_BLOCK);
    gdn_rmsnorm_gated_quantized_rows4_kernel<<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
        (const __nv_bfloat16 *)weight, (gdn_fp8_e4m3 *)output, scales, rows,
        groups, scale_stride_m, scale_layout, eps);
    return;
  }

  const dim3 block(32, GDN_RMSNORM_ROWS_PER_BLOCK);
  const dim3 grid((rows + GDN_RMSNORM_ROWS_PER_BLOCK - 1) /
                  GDN_RMSNORM_ROWS_PER_BLOCK);
  gdn_rmsnorm_gated_quantized_warp_kernel<GDN_RMSNORM_ROWS_PER_BLOCK>
      <<<grid, block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)gate,
          (const __nv_bfloat16 *)weight, (gdn_fp8_e4m3 *)output, scales,
          rows, groups, scale_stride_m, scale_layout, outer_dim_1,
          outer_dim_2, x_stride_0, x_stride_1, x_stride_2, x_stride_3,
          gate_stride_0, gate_stride_1, gate_stride_2, gate_stride_3, eps);
}

// ============================================================================
// Kernel 3: fused_gdn_gating
//
// Fuses: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
// a_log and dt_bias are per-head (broadcast over batch*seq).
//
// b, a: [total]  a_log, dt_bias: [num_heads]
// beta_out, g_out: [total]
// ============================================================================

template <typename T>
__global__ void
fused_gdn_gating_kernel(const T *__restrict__ b,           // [total]
                        const T *__restrict__ a,           // [total]
                        const float *__restrict__ a_log,   // [num_heads]
                        const float *__restrict__ dt_bias, // [num_heads]
                        T *__restrict__ beta_out,          // [total]
                        T *__restrict__ g_out,             // [total]
                        int total_elements, int num_heads) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  // Head index: elements are laid out as [..., num_heads]
  int head_idx = idx % num_heads;

  // beta = sigmoid(b)
  float b_val = (float)b[idx];
  float beta = 1.0f / (1.0f + expf(-b_val));

  // g = -exp(a_log) * softplus(a + dt_bias)
  float a_val = (float)a[idx];
  float a_log_val = a_log[head_idx];
  float dt_bias_val = dt_bias[head_idx];

  float sp_input = a_val + dt_bias_val;
  float softplus_val = logf(1.0f + expf(sp_input));
  float g_val = -expf(a_log_val) * softplus_val;

  beta_out[idx] = (T)beta;
  g_out[idx] = (T)g_val;
}

extern "C" void fused_gdn_gating(const void *b, const void *a,
                                 const float *a_log, const float *dt_bias,
                                 void *beta_out, void *g_out,
                                 int total_elements, int num_heads, int dtype,
                                 int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block(256);
  dim3 grid((total_elements + 255) / 256);

  if (dtype == 0) {
    fused_gdn_gating_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)b, (const __half *)a, a_log, dt_bias,
        (__half *)beta_out, (__half *)g_out, total_elements, num_heads);
  } else {
    fused_gdn_gating_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)b, (const __nv_bfloat16 *)a, a_log, dt_bias,
        (__nv_bfloat16 *)beta_out, (__nv_bfloat16 *)g_out, total_elements,
        num_heads);
  }
}
