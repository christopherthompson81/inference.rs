#include "mmvq_tc_k.cuh"

MMVQ_PLAIN_BATCH_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                     vec_dot_q4_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                        vec_dot_q4_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ,
                        vec_dot_q4_K_q8_1)
MMVQ_LAUNCHER_PLAIN(q4_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q4_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q4_k, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_k)
MMVQ_TC_K_LAUNCHERS(q4_k, TckQ4K)
