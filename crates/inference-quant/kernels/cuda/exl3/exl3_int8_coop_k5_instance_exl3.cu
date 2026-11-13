// exllamav3's cooperative int8-activation GEMV at 5 bits per weight: only the f32-output, plain int8 form we launch
// (third_party/exllamav3)
#include "quant/comp_units/exl3_gemv_int8_instances.cuh"
#include "quant/exl3_gemv_int8_kernel.cuh"

void* exl3_gemv_int8_coop_sel_k5(bool c_fp32, bool residual)
{
    if (!c_fp32 || residual) return nullptr;
    return (void*) exl3_gemv_int8_coop_kernel<5, true, false>;
}
