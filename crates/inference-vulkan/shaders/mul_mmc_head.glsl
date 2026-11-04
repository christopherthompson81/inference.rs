// Shared by the mul_mmc_* shaders: y = W x for a prompt's tokens on matrix cores (VK_KHR_cooperative_matrix),
// against a quantized W (`rows x cols`) and the activations as f16 (convert_f16's, `m x cols`). A workgroup takes a
// BN x BM tile (weight rows x tokens), stepping BLOCKS 32-value blocks at a time: its threads dequantize the step's
// weight blocks to f16 in shared memory (through the type's unpack_sub), then each subgroup multiplies its 32 x 32
// quarter of the tile as 16 x 16 x 16 f16 products into f32 accumulators, loading the activations from memory. The
// including shader declares the push constants (rows, cols, w_base, x_base and y_base among them), then includes a
// `*_int8.glsl` and defines the tile's hooks before mul_mmc_main.glsl: `begin_tile` (false to leave) with its tokens,
// weight matrix and first activation row (whose 64 rows from there `x` holds, padded), and `y_row`, a token's output row.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer W { uint16_t w[]; };
layout(std430, binding = 1) readonly buffer X { float16_t x[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };


#include "int8_dot.glsl"
