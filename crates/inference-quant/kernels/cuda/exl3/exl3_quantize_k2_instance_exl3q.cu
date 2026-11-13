// exllamav3's trellis quantizer (Viterbi over each tile's states) at 2 bits per weight, 256-weight tiles, both its
// generic and dense forms (third_party/exllamav3)
#include "quant/comp_units/quantize_tiles_instances.cuh"
#include "quant/quantize_tiles_optimized.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb0(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<2, 0> : quantize_tiles_kernel<2, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb1(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<2, 1> : quantize_tiles_kernel<2, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb2(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<2, 2> : quantize_tiles_kernel<2, 2>; }
