#include "mmvq_iqk.cuh"
#include "mmvq_tc.cuh"

IQK_LAUNCHERS(iq4_ks, DEQUANTIZE_IQK_ROWS_LAUNCHER)
MMVQ_TC_LAUNCHERS(iq4_ks, iqk_iq4_ks)
