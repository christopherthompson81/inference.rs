// mmvq_tc_quantize's activation layout, which the tensor-core matvecs read: per 8-row half and 32-value block, 32 lanes'
// int2 B fragments, then the 8 rows' f16 scales (as q8_1's), then their int16 sums (32 int8 values fit).
#pragma once
#include "cuda_fp16.h"
#include <stdint.h>

#define MMVQ_TC_ACT_HALF_BLOCK_BYTES (32 * 8 + 8 * 2 + 8 * 2)

// rows 2t and 2t + 1's block sums, packed low then high
static __device__ __forceinline__ int2 tc_unpack_sums(const int packed) {
  return make_int2((int)(int16_t)(packed & 0xffff), packed >> 16);
}
