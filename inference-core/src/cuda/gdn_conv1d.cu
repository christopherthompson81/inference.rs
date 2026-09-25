#include "gdn_common.cuh"

// ============================================================================
// Kernel 2a: causal_conv1d_update (decode path, single step)
//
// Each thread handles one channel: shift conv_state left by 1,
// insert new value, dot product with weight, apply SiLU.
//
// x: [B, 1, conv_dim]  weight: [conv_dim, kernel_size]
// conv_state: [B, conv_dim, kernel_size] (in/out)
// output: [B, 1, conv_dim]
// ============================================================================

template <typename T>
__global__ void causal_conv1d_update_kernel(
    const T *__restrict__ x,      // [B, 1, conv_dim]
    const T *__restrict__ weight, // [conv_dim, kernel_size]
    T *__restrict__ conv_state,   // [B, conv_dim, kernel_size]
    T *__restrict__ output,       // [B, 1, conv_dim]
    int batch_size, int conv_dim, int kernel_size, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  if (gdn_is_padding_row(slot_indices, b)) {
    output[(size_t)b * conv_dim + ch] = (T)0.0f;
    return;
  }

  // Pointer to this batch/channel's conv state
  T *cs = conv_state + (gdn_state_row(slot_indices, b, 0, 1) * conv_dim + ch) * kernel_size;
  const T *w = weight + ch * kernel_size;

  // Shift state left by 1
  for (int i = 0; i < kernel_size - 1; i++) {
    cs[i] = cs[i + 1];
  }
  // Insert new value
  cs[kernel_size - 1] =
      x[(size_t)b * x_stride_b + (size_t)ch * x_stride_c];

  // Dot product with weight
  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    acc += (float)cs[i] * (float)w[i];
  }

  // SiLU activation: x * sigmoid(x)
  float sig = 1.0f / (1.0f + expf(-acc));
  float result = acc * sig;

  output[b * conv_dim + ch] = (T)result;
}

template <typename T>
__global__ void causal_conv1d_update_width4_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ conv_state, T *__restrict__ output, int batch_size,
    int conv_dim, int64_t x_stride_b, int64_t x_stride_s,
    int64_t x_stride_c, const int32_t *__restrict__ slot_indices) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  const size_t input_idx = (size_t)b * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    output[input_idx] = (T)0.0f;
    return;
  }

  const size_t state_row = gdn_state_row(slot_indices, b, 0, 1);
  const size_t state_idx = state_row * conv_dim + ch;
  const size_t x_idx = (size_t)b * x_stride_b + (size_t)ch * x_stride_c;
  auto *state = reinterpret_cast<GdnConvWidth4<T> *>(conv_state);
  const auto *weights = reinterpret_cast<const GdnConvWidth4<T> *>(weight);
  output[input_idx] = gdn_conv_width4_update(
      x[x_idx], weights[ch], &state[state_idx]);
}

extern "C" void causal_conv1d_update(const void *x, const void *weight,
                                     void *conv_state, void *output,
                                     int batch_size, int conv_dim,
                                     int kernel_size, int64_t x_stride_b,
                                     int64_t x_stride_s, int64_t x_stride_c,
                                     const int32_t *slot_indices, int dtype,
                                     int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block(GDN_CHANNEL_BLOCK_SIZE);
  dim3 grid((conv_dim + GDN_CHANNEL_BLOCK_SIZE - 1) / GDN_CHANNEL_BLOCK_SIZE,
            batch_size);

  if (kernel_size == GDN_PACKED_CONV_WIDTH) {
    if (dtype == 0) {
      causal_conv1d_update_width4_kernel<__half><<<grid, block, 0, custream>>>(
          (const __half *)x, (const __half *)weight, (__half *)conv_state,
          (__half *)output, batch_size, conv_dim, x_stride_b, x_stride_s,
          x_stride_c, slot_indices);
    } else {
      causal_conv1d_update_width4_kernel<__nv_bfloat16>
          <<<grid, block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
              (__nv_bfloat16 *)conv_state, (__nv_bfloat16 *)output, batch_size,
              conv_dim, x_stride_b, x_stride_s, x_stride_c, slot_indices);
    }
  } else if (dtype == 0) {
    causal_conv1d_update_kernel<__half><<<grid, block, 0, custream>>>(
        (const __half *)x, (const __half *)weight, (__half *)conv_state,
        (__half *)output, batch_size, conv_dim, kernel_size, x_stride_b,
        x_stride_s, x_stride_c, slot_indices);
  } else {
    causal_conv1d_update_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
        (__nv_bfloat16 *)conv_state, (__nv_bfloat16 *)output, batch_size,
        conv_dim, kernel_size, x_stride_b, x_stride_s, x_stride_c,
        slot_indices);
  }
}

// ============================================================================
// Kernel 2b: causal_conv1d_full (prefill path)
//
// Each thread handles one (channel, position), seeded from the prior state.
// A second pass retains the last kernel_size positions.
//
// x: [B, S, conv_dim]  weight: [conv_dim, kernel_size]
// conv_state_out: [B, conv_dim, kernel_size]  output: [B, S, conv_dim]
// ============================================================================

template <typename T>
__global__ void causal_conv1d_full_kernel(
    const T *__restrict__ x,      // [B, S, conv_dim]
    const T *__restrict__ weight, // [conv_dim, kernel_size]
    const T *__restrict__ conv_state,
    T *__restrict__ output, // [B, S, conv_dim]
    int batch_size, int conv_dim, int seq_len, int kernel_size,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;
  const size_t plane = (size_t)conv_dim * seq_len;

  if (idx >= plane || b >= batch_size)
    return;

  const int pos = (int)(idx / conv_dim);
  const int ch = (int)(idx % conv_dim);

  const size_t output_idx = ((size_t)b * seq_len + pos) * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    output[output_idx] = (T)0.0f;
    return;
  }

  const T *w = weight + (size_t)ch * kernel_size;
  const T *cs =
      conv_state + (gdn_state_row(slot_indices, b, 0, 1) * conv_dim + ch) * kernel_size;

  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    int src_pos = pos - (kernel_size - 1) + i;
    float x_val = src_pos >= 0
                      ? (float)x[(size_t)b * x_stride_b +
                                 (size_t)src_pos * x_stride_s +
                                 (size_t)ch * x_stride_c]
                      : (float)cs[kernel_size + src_pos];
    acc += x_val * (float)w[i];
  }

  // SiLU
  float sig = 1.0f / (1.0f + expf(-acc));
  float result = acc * sig;

  output[output_idx] = (T)result;
}

template <typename T, int TOKEN_TILE>
__global__ void causal_conv1d_full_width4_tiled_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    const T *__restrict__ conv_state, T *__restrict__ output, int batch_size,
    int conv_dim, int seq_len, int64_t x_stride_b, int64_t x_stride_s,
    int64_t x_stride_c, const int32_t *__restrict__ slot_indices) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int start = blockIdx.y * TOKEN_TILE;
  const int b = blockIdx.z;
  if (ch >= conv_dim || start >= seq_len || b >= batch_size) {
    return;
  }

  const int end = min(start + TOKEN_TILE, seq_len);
  T *out = output + ((size_t)b * seq_len + start) * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    for (int pos = start; pos < end; ++pos) {
      *out = (T)0.0f;
      out += conv_dim;
    }
    return;
  }

  const size_t state_row = gdn_state_row(slot_indices, b, 0, 1);
  const T *state = conv_state +
                   (state_row * conv_dim + ch) * GDN_PACKED_CONV_WIDTH;
  const T *w = weight + (size_t)ch * GDN_PACKED_CONV_WIDTH;
  const size_t x_batch_offset = (size_t)b * x_stride_b;
  float x0 = causal_conv1d_width4_load(
      x, state, start - 3, x_batch_offset, x_stride_s, x_stride_c, ch);
  float x1 = causal_conv1d_width4_load(
      x, state, start - 2, x_batch_offset, x_stride_s, x_stride_c, ch);
  float x2 = causal_conv1d_width4_load(
      x, state, start - 1, x_batch_offset, x_stride_s, x_stride_c, ch);
  const float w0 = (float)w[0];
  const float w1 = (float)w[1];
  const float w2 = (float)w[2];
  const float w3 = (float)w[3];

  for (int pos = start; pos < end; ++pos) {
    const float x3 = (float)x[x_batch_offset + (size_t)pos * x_stride_s +
                              (size_t)ch * x_stride_c];
    const float acc = __fmaf_rn(x0, w0, __fmaf_rn(x1, w1, __fmaf_rn(x2, w2, x3 * w3)));
    const float result = acc / (1.0f + expf(-acc));
    *out = (T)result;
    out += conv_dim;
    x0 = x1;
    x1 = x2;
    x2 = x3;
  }
}

template <typename T>
__global__ void save_conv_state_kernel(
    const T *__restrict__ x, // [B, S, conv_dim]
    // May alias conv_state_out (pooled in-place update): every read is ahead of the write position
    const T *conv_state_in,
    T *conv_state_out, // [B, conv_dim, kernel_size]
    int batch_size, int conv_dim, int seq_len, int kernel_size,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  if (gdn_is_padding_row(slot_indices, b)) {
    return;
  }

  const size_t row = gdn_state_row(slot_indices, b, 0, 1);
  const T *prior = conv_state_in + (row * conv_dim + ch) * kernel_size;
  T *cs = conv_state_out + (row * conv_dim + ch) * kernel_size;

  int pad = kernel_size - seq_len;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      cs[i] = prior[i + seq_len];
    } else {
      const int pos = seq_len - kernel_size + i;
      cs[i] = x[(size_t)b * x_stride_b + (size_t)pos * x_stride_s +
                (size_t)ch * x_stride_c];
    }
  }
}

extern "C" void causal_conv1d_full(const void *x, const void *weight,
                                   const void *conv_state_in,
                                   void *conv_state_out, void *output,
                                   int batch_size, int conv_dim, int seq_len,
                                   int kernel_size, int64_t x_stride_b,
                                   int64_t x_stride_s, int64_t x_stride_c,
                                   const int32_t *slot_indices, int dtype,
                                   int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  const dim3 state_block(256);
  dim3 block = state_block;
  const size_t plane = (size_t)conv_dim * seq_len;
  dim3 grid((unsigned int)((plane + 255) / 256), batch_size);

  const bool use_width4_tiled =
      kernel_size == GDN_PACKED_CONV_WIDTH && x_stride_c == 1;
  if (use_width4_tiled) {
    block = dim3(GDN_PREFILL_CONV_THREADS);
    grid = dim3((conv_dim + GDN_PREFILL_CONV_THREADS - 1) /
                    GDN_PREFILL_CONV_THREADS,
                (seq_len + GDN_PREFILL_CONV_TOKEN_TILE - 1) /
                    GDN_PREFILL_CONV_TOKEN_TILE,
                batch_size);
  }

  if (dtype == 0) {
    if (use_width4_tiled) {
      causal_conv1d_full_width4_tiled_kernel<__half,
                                             GDN_PREFILL_CONV_TOKEN_TILE>
          <<<grid, block, 0, custream>>>(
              (const __half *)x, (const __half *)weight,
              (const __half *)conv_state_in, (__half *)output, batch_size,
              conv_dim, seq_len, x_stride_b, x_stride_s, x_stride_c,
              slot_indices);
    } else {
      causal_conv1d_full_kernel<__half><<<grid, block, 0, custream>>>(
          (const __half *)x, (const __half *)weight,
          (const __half *)conv_state_in, (__half *)output, batch_size,
          conv_dim, seq_len, kernel_size, x_stride_b, x_stride_s, x_stride_c,
          slot_indices);
    }
    dim3 grid2((conv_dim + state_block.x - 1) / state_block.x, batch_size);
    save_conv_state_kernel<__half><<<grid2, state_block, 0, custream>>>(
        (const __half *)x, (const __half *)conv_state_in,
        (__half *)conv_state_out, batch_size, conv_dim, seq_len, kernel_size,
        x_stride_b, x_stride_s, x_stride_c, slot_indices);
  } else {
    if (use_width4_tiled) {
      causal_conv1d_full_width4_tiled_kernel<
          __nv_bfloat16, GDN_PREFILL_CONV_TOKEN_TILE>
          <<<grid, block, 0, custream>>>(
              (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
              (const __nv_bfloat16 *)conv_state_in,
              (__nv_bfloat16 *)output, batch_size, conv_dim, seq_len,
              x_stride_b, x_stride_s, x_stride_c, slot_indices);
    } else {
      causal_conv1d_full_kernel<__nv_bfloat16><<<grid, block, 0, custream>>>(
          (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)weight,
          (const __nv_bfloat16 *)conv_state_in, (__nv_bfloat16 *)output,
          batch_size, conv_dim, seq_len, kernel_size, x_stride_b, x_stride_s,
          x_stride_c, slot_indices);
    }
    dim3 grid2((conv_dim + state_block.x - 1) / state_block.x, batch_size);
    save_conv_state_kernel<__nv_bfloat16>
        <<<grid2, state_block, 0, custream>>>(
        (const __nv_bfloat16 *)x, (const __nv_bfloat16 *)conv_state_in,
        (__nv_bfloat16 *)conv_state_out, batch_size, conv_dim, seq_len,
        kernel_size, x_stride_b, x_stride_s, x_stride_c, slot_indices);
  }
}
