#include "gdn_common.cuh"

template <typename T>
__global__ void gdn_packed_to_padded_kernel(
    const T *__restrict__ source, T *__restrict__ output,
    const uint32_t *__restrict__ cu_seqlens, size_t total_elements,
    int padded_len, int width, int64_t source_token_stride,
    float padding_value) {
  for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < total_elements;
       index += (size_t)gridDim.x * blockDim.x) {
    const size_t token = index / width;
    const int feature = index % width;
    const int row = token / padded_len;
    const int position = token % padded_len;
    const uint32_t start = cu_seqlens[row];
    const uint32_t row_len = cu_seqlens[row + 1] - start;
    output[index] = position < row_len
                        ? source[(size_t)(start + position) *
                                     source_token_stride +
                                 feature]
                        : gdn_ragged_from_float<T>(padding_value);
  }
}

template <typename T>
__global__ void gdn_padded_to_packed_kernel(
    const T *__restrict__ source, T *__restrict__ output,
    const uint32_t *__restrict__ cu_seqlens, int width,
    int64_t source_batch_stride, int64_t source_token_stride,
    int64_t source_feature_stride, int feature_inner_width) {
  const int row = blockIdx.x;
  const uint32_t start = cu_seqlens[row];
  const uint32_t row_len = cu_seqlens[row + 1] - start;
  const size_t row_elements = (size_t)row_len * width;
  for (size_t index = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
       index < row_elements;
       index += (size_t)gridDim.y * blockDim.x) {
    const int position = index / width;
    const int feature = index % width;
    const int feature_outer = feature / feature_inner_width;
    const int feature_inner = feature % feature_inner_width;
    output[(size_t)(start + position) * width + feature] =
        source[(size_t)row * source_batch_stride +
               (size_t)position * source_token_stride +
               (size_t)feature_outer * source_feature_stride + feature_inner];
  }
}

template <typename T>
__global__ void gdn_extract_ragged_conv_state_kernel(
    const T *__restrict__ padded_input, const T *__restrict__ initial_state,
    T *__restrict__ output, const uint32_t *__restrict__ cu_seqlens,
    size_t total_elements, int padded_len, int channels, int state_width,
    int64_t input_batch_stride, int64_t input_token_stride) {
  for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < total_elements;
       index += (size_t)gridDim.x * blockDim.x) {
    const int state_position = index % state_width;
    const size_t channel_row = index / state_width;
    const int channel = channel_row % channels;
    const int row = channel_row / channels;
    const int row_len = cu_seqlens[row + 1] - cu_seqlens[row];
    if (row_len >= state_width) {
      const int input_position = row_len - state_width + state_position;
      output[index] =
          padded_input[(size_t)row * input_batch_stride +
                       (size_t)input_position * input_token_stride +
                       channel];
    } else {
      const int retained = state_width - row_len;
      output[index] = state_position < retained
                          ? initial_state[((size_t)row * channels + channel) *
                                              state_width +
                                          state_position + row_len]
                          : padded_input[(size_t)row * input_batch_stride +
                                         (size_t)(state_position - retained) *
                                             input_token_stride +
                                         channel];
    }
  }
}

extern "C" void gdn_packed_to_padded(
    const void *source, void *output, const uint32_t *cu_seqlens,
    int batch_size, int padded_len, int width, int64_t source_token_stride,
    float padding_value, int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t total_elements =
      (size_t)batch_size * padded_len * width;
  const int blocks =
      min((size_t)65535, (total_elements + THREADS - 1) / THREADS);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_packed_to_padded_kernel<<<blocks, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)source, (__nv_bfloat16 *)output, cu_seqlens,
        total_elements, padded_len, width, source_token_stride, padding_value);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_packed_to_padded_kernel<<<blocks, THREADS, 0, custream>>>(
        (const float *)source, (float *)output, cu_seqlens, total_elements,
        padded_len, width, source_token_stride, padding_value);
  }
}

extern "C" void gdn_padded_to_packed(
    const void *source, void *output, const uint32_t *cu_seqlens,
    int batch_size, int padded_len, int width, int64_t source_batch_stride,
    int64_t source_token_stride, int64_t source_feature_stride,
    int feature_inner_width, int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t max_row_elements = (size_t)padded_len * width;
  const int row_blocks =
      min((size_t)65535, (max_row_elements + THREADS - 1) / THREADS);
  const dim3 grid(batch_size, row_blocks);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_padded_to_packed_kernel<<<grid, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)source, (__nv_bfloat16 *)output, cu_seqlens,
        width, source_batch_stride, source_token_stride, source_feature_stride,
        feature_inner_width);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_padded_to_packed_kernel<<<grid, THREADS, 0, custream>>>(
        (const float *)source, (float *)output, cu_seqlens, width,
        source_batch_stride, source_token_stride, source_feature_stride,
        feature_inner_width);
  }
}

extern "C" void gdn_extract_ragged_conv_state(
    const void *padded_input, const void *initial_state, void *output,
    const uint32_t *cu_seqlens, int batch_size, int padded_len, int channels,
    int state_width, int64_t input_batch_stride, int64_t input_token_stride,
    int dtype, int64_t stream) {
  constexpr int THREADS = 256;
  const size_t total_elements =
      (size_t)batch_size * channels * state_width;
  const int blocks =
      min((size_t)65535, (total_elements + THREADS - 1) / THREADS);
  const cudaStream_t custream = (cudaStream_t)stream;
  if (dtype == GDN_STATE_DTYPE_BF16) {
    gdn_extract_ragged_conv_state_kernel<<<blocks, THREADS, 0, custream>>>(
        (const __nv_bfloat16 *)padded_input,
        (const __nv_bfloat16 *)initial_state, (__nv_bfloat16 *)output,
        cu_seqlens, total_elements, padded_len, channels, state_width,
        input_batch_stride, input_token_stride);
  } else if (dtype == GDN_STATE_DTYPE_F32) {
    gdn_extract_ragged_conv_state_kernel<<<blocks, THREADS, 0, custream>>>(
        (const float *)padded_input, (const float *)initial_state,
        (float *)output, cu_seqlens, total_elements, padded_len, channels,
        state_width, input_batch_stride, input_token_stride);
  }
}
