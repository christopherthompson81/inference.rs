#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                     vec_dot_q4_1_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                        vec_dot_q4_1_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ,
                        vec_dot_q4_1_q8_1)
MMVQ_LAUNCHER_PLAIN(q4_1, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q4_1, f16, half)
MMVQ_LAUNCHER_PLAIN(q4_1, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_1)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_1)
