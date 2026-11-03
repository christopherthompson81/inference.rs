#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(nvfp4, block_nvfp4, QK_NVFP4, QI_NVFP4, VDR_NVFP4_Q8_1_MMVQ,
                     vec_dot_nvfp4_q8_1)
MMVQ_LAUNCHER_PLAIN(nvfp4, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(nvfp4, f16, half)
MMVQ_LAUNCHER_PLAIN(nvfp4, f32, float)
DEQUANTIZE_NVFP4_LAUNCHER(bf16, __nv_bfloat16)
DEQUANTIZE_NVFP4_LAUNCHER(f16, half)
DEQUANTIZE_NVFP4_LAUNCHER(f32, float)
