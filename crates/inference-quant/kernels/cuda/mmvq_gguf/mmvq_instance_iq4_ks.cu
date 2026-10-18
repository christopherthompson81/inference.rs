#include "mmvq_iqk.cuh"
#include "mmvq_tc.cuh"

IQK_LAUNCHERS(iq4_ks, DEQUANTIZE_IQK_ROWS_LAUNCHER)
MMVQ_TC_IQ4_KS_LAUNCHER(bf16, __nv_bfloat16)
MMVQ_TC_IQ4_KS_LAUNCHER(f16, half)
MMVQ_TC_IQ4_KS_LAUNCHER(f32, float)
