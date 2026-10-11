#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq2_s, block_iq2_s, QK_K, QI2_S, VDR_IQ2_S_Q8_1_MMVQ,
                     vec_dot_iq2_s_q8_1)
MMVQ_LAUNCHER_PLAIN(iq2_s, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq2_s, f16, half)
MMVQ_LAUNCHER_PLAIN(iq2_s, f32, float)
DEQUANTIZE_IQ_LAUNCHER(iq2_s, bf16, __nv_bfloat16)
DEQUANTIZE_IQ_LAUNCHER(iq2_s, f16, half)
DEQUANTIZE_IQ_LAUNCHER(iq2_s, f32, float)
