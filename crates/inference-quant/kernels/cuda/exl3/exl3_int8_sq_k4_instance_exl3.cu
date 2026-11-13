// exllamav3's int8-activation GEMV, one or two rows, at 4 bits per weight: only the f32-output, plain int8 forms we
// launch (third_party/exllamav3)
#include "quant/comp_units/exl3_gemv_int8_instances.cuh"
#include "quant/exl3_gemv_int8_kernel.cuh"

void* exl3_gemv_int8_sq_sel_k4(int M, bool c_fp32, bool residual)
{
    if (!c_fp32 || residual) return nullptr;
    switch (M)
    {
        case 1: return (void*) exl3_gemv_int8_sq_kernel<4, 1, true, false>;
        case 2: return (void*) exl3_gemv_int8_sq_kernel<4, 2, true, false>;
    }
    return nullptr;
}
