#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                     vec_dot_q4_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                        vec_dot_q4_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ,
                        vec_dot_q4_0_q8_1)
MMVQ_LAUNCHER_PLAIN(q4_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q4_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q4_0, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q4_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q4_0)
