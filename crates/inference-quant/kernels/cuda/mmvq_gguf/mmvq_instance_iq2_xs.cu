#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq2_xs, block_iq2_xs, QK_K, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ,
                     vec_dot_iq2_xs_q8_1)
MMVQ_LAUNCHER_PLAIN(iq2_xs, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq2_xs, f16, half)
MMVQ_LAUNCHER_PLAIN(iq2_xs, f32, float)
DEQUANTIZE_IQ_LAUNCHER(iq2_xs, bf16, __nv_bfloat16)
DEQUANTIZE_IQ_LAUNCHER(iq2_xs, f16, half)
DEQUANTIZE_IQ_LAUNCHER(iq2_xs, f32, float)
