#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq4_xs, block_iq4_xs, QK_K, QI4_XS, VDR_IQ4_XS_Q8_1_MMVQ,
                     vec_dot_iq4_xs_q8_1)
MMVQ_LAUNCHER_PLAIN(iq4_xs, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq4_xs, f16, half)
MMVQ_LAUNCHER_PLAIN(iq4_xs, f32, float)
DEQUANTIZE_IQ_LAUNCHER(iq4_xs, bf16, __nv_bfloat16)
DEQUANTIZE_IQ_LAUNCHER(iq4_xs, f16, half)
DEQUANTIZE_IQ_LAUNCHER(iq4_xs, f32, float)
