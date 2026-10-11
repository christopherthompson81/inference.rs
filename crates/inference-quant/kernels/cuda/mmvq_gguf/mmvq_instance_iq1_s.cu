#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq1_s, block_iq1_s, QK_K, QI1_S, VDR_IQ1_S_Q8_1_MMVQ,
                     vec_dot_iq1_s_q8_1)
MMVQ_LAUNCHER_PLAIN(iq1_s, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq1_s, f16, half)
MMVQ_LAUNCHER_PLAIN(iq1_s, f32, float)
DEQUANTIZE_IQ_LAUNCHER(iq1_s, bf16, __nv_bfloat16)
DEQUANTIZE_IQ_LAUNCHER(iq1_s, f16, half)
DEQUANTIZE_IQ_LAUNCHER(iq1_s, f32, float)
