#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq3_xxs, block_iq3_xxs, QK_K, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ,
                     vec_dot_iq3_xxs_q8_1)
MMVQ_LAUNCHER_PLAIN(iq3_xxs, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq3_xxs, f16, half)
MMVQ_LAUNCHER_PLAIN(iq3_xxs, f32, float)
DEQUANTIZE_IQ_LAUNCHER(iq3_xxs, bf16, __nv_bfloat16)
DEQUANTIZE_IQ_LAUNCHER(iq3_xxs, f16, half)
DEQUANTIZE_IQ_LAUNCHER(iq3_xxs, f32, float)
