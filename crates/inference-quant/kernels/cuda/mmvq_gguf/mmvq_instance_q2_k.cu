#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                     vec_dot_q2_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                        vec_dot_q2_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ,
                        vec_dot_q2_K_q8_1)
MMVQ_LAUNCHER_PLAIN(q2_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q2_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q2_k, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q2_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q2_k)
