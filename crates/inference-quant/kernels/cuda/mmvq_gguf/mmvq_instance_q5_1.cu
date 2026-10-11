#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                     vec_dot_q5_1_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                        vec_dot_q5_1_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ,
                        vec_dot_q5_1_q8_1)
MMVQ_LAUNCHER_PLAIN(q5_1, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q5_1, f16, half)
MMVQ_LAUNCHER_PLAIN(q5_1, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q5_1)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q5_1)
