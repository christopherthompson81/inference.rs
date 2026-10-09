#include "float8.metal"
#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

// ============================================================================
// FP8 E4M3 to other dtypes (per-element conversion)
// ============================================================================

template <typename OutT>
kernel void fp8_to_dtype_kernel(device const uchar *input [[buffer(0)]],
                                device OutT *output [[buffer(1)]],
                                constant uint &num_elements [[buffer(2)]],
                                uint idx [[thread_position_in_grid]]) {
  if (idx >= num_elements)
    return;
  float val = fp8_e4m3_to_float(input[idx]);
  output[idx] = OutT(val);
}

// ============================================================================
// Other dtypes to FP8 E4M3 (per-element conversion with clamping)
// ============================================================================

template <typename InT>
kernel void dtype_to_fp8_kernel(device const InT *input [[buffer(0)]],
                                device uchar *output [[buffer(1)]],
                                constant uint &num_elements [[buffer(2)]],
                                uint idx [[thread_position_in_grid]]) {
  if (idx >= num_elements)
    return;
  float val = float(input[idx]);
  // Clamp to FP8 E4M3 range: [-448, 448]
  val = clamp(val, -448.0f, 448.0f);
  output[idx] = float_to_fp8_e4m3(val);
}

// ============================================================================
// Instantiate kernels for all supported output types
// ============================================================================

#define instantiate_fp8_to_dtype(type)                                         \
  template [[host_name("fp8_to_dtype_" #type)]] [[kernel]] void                \
  fp8_to_dtype_kernel<type>(device const uchar *input [[buffer(0)]],           \
                            device type *output [[buffer(1)]],                 \
                            constant uint &num_elements [[buffer(2)]],         \
                            uint idx [[thread_position_in_grid]]);

instantiate_fp8_to_dtype(float);
instantiate_fp8_to_dtype(half);
instantiate_fp8_to_dtype(bfloat16_t);

#define instantiate_dtype_to_fp8(type)                                         \
  template [[host_name("dtype_to_fp8_" #type)]] [[kernel]] void                \
  dtype_to_fp8_kernel<type>(device const type *input [[buffer(0)]],            \
                            device uchar *output [[buffer(1)]],                \
                            constant uint &num_elements [[buffer(2)]],         \
                            uint idx [[thread_position_in_grid]]);

instantiate_dtype_to_fp8(float);
instantiate_dtype_to_fp8(half);
instantiate_dtype_to_fp8(bfloat16_t);
