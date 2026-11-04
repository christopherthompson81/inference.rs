// 128 columns for a quantized cache's prompt rows at head 128 (fattn.cu picks it); no generated head sizes beside it

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(128, 128, 16, 8);
