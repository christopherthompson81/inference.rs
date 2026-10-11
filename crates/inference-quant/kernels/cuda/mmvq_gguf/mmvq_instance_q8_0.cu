#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                     vec_dot_q8_0_q8_1)
MMVQ_FUSED_GLU_TYPE_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                        vec_dot_q8_0_q8_1)
MMVQ_FUSED_QKV_TYPE_SET(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ,
                        vec_dot_q8_0_q8_1)
MMVQ_LAUNCHER_PLAIN(q8_0, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(q8_0, f16, half)
MMVQ_LAUNCHER_PLAIN(q8_0, f32, float)
MMVQ_LAUNCHER_FUSED_GLU_TYPE_SET(q8_0)
MMVQ_LAUNCHER_FUSED_QKV_TYPE_SET(q8_0)
