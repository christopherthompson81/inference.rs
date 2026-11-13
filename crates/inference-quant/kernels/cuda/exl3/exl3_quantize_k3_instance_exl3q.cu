// exllamav3's trellis quantizer (Viterbi over each tile's states) at 3 bits per weight, 256-weight tiles, both its
// generic and dense forms (third_party/exllamav3)
#include "quant/comp_units/quantize_tiles_instances.cuh"
#include "quant/quantize_tiles_optimized.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb0(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<3, 0> : quantize_tiles_kernel<3, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb1(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<3, 1> : quantize_tiles_kernel<3, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb2(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<3, 2> : quantize_tiles_kernel<3, 2>; }
