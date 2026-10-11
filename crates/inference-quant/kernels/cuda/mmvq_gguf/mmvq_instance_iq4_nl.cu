#include "mmvq_gguf.cuh"

MMVQ_PLAIN_BATCH_SET(iq4_nl, block_iq4_nl, QK4_NL, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ,
                     vec_dot_iq4_nl_q8_1)
MMVQ_LAUNCHER_PLAIN(iq4_nl, bf16, __nv_bfloat16)
MMVQ_LAUNCHER_PLAIN(iq4_nl, f16, half)
MMVQ_LAUNCHER_PLAIN(iq4_nl, f32, float)
DEQUANTIZE_IQ_LAUNCHER(iq4_nl, bf16, __nv_bfloat16)
DEQUANTIZE_IQ_LAUNCHER(iq4_nl, f16, half)
DEQUANTIZE_IQ_LAUNCHER(iq4_nl, f32, float)
