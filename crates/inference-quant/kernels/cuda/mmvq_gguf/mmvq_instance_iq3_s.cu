#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq3_s, block_iq3_s, QK_K, QI3_S, VDR_IQ3_S_Q8_1_MMVQ,
                     vec_dot_iq3_s_q8_1)
MMVQ_LAUNCHER_PLAIN(iq3_s, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq3_s, f16, half)
MMVQ_LAUNCHER_PLAIN(iq3_s, f32, float)
DEQUANTIZE_IQ_LAUNCHER(iq3_s, bf16, __nv_bfloat16)
DEQUANTIZE_IQ_LAUNCHER(iq3_s, f16, half)
DEQUANTIZE_IQ_LAUNCHER(iq3_s, f32, float)
