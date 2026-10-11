#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq2_xxs, block_iq2_xxs, QK_K, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ,
                     vec_dot_iq2_xxs_q8_1)
MMVQ_LAUNCHER_PLAIN(iq2_xxs, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq2_xxs, f16, half)
MMVQ_LAUNCHER_PLAIN(iq2_xxs, f32, float)
DEQUANTIZE_IQ_LAUNCHER(iq2_xxs, bf16, __nv_bfloat16)
DEQUANTIZE_IQ_LAUNCHER(iq2_xxs, f16, half)
DEQUANTIZE_IQ_LAUNCHER(iq2_xxs, f32, float)
