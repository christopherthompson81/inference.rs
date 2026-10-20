#include "mmvq_tc_k.cuh"

MMVQ_PLAIN_BATCH_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                     vec_dot_q6_K_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                        vec_dot_q6_K_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ,
                        vec_dot_q6_K_q8_1)
MMVQ_LAUNCHER_PLAIN(q6_k, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q6_k, f16, half)
MMVQ_LAUNCHER_PLAIN(q6_k, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q6_k)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q6_k)
MMVQ_TC_K_LAUNCHERS(q6_k, TckQ6K)
