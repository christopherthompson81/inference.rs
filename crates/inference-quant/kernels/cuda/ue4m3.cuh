// NVFP4's UE4M3 block scales, shared by the matvec and mmq kernels.
#pragma once

#include <stdint.h>

// ggml-cuda's software decode, which matches the CPU's for every byte (the FP8 intrinsic reads bit 7 as a sign), halved:
// a normal value's f32 bits built directly (exponent exp - 8, the 3 mantissa bits on top), a subnormal man * 2^-10
static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
  const unsigned exp = (x >> 3) & 0xF;
  const unsigned man = x & 0x7;
  if (x == 0x7F) {
    return 0.0f;
  }
  if (exp == 0) {
    return (float)man * (1.0f / 1024.0f);
  }
  return __uint_as_float(((exp + 119u) << 23) | (man << 20));
}
