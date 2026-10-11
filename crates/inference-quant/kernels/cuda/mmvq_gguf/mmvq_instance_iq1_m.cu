#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq1_m, block_iq1_m, QK_K, QI1_M, VDR_IQ1_M_Q8_1_MMVQ,
                     vec_dot_iq1_m_q8_1)
MMVQ_LAUNCHER_PLAIN(iq1_m, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq1_m, f16, half)
MMVQ_LAUNCHER_PLAIN(iq1_m, f32, float)
DEQUANTIZE_IQ1_M_LAUNCHER(bf16, __nv_bfloat16)
DEQUANTIZE_IQ1_M_LAUNCHER(f16, half)
DEQUANTIZE_IQ1_M_LAUNCHER(f32, float)
