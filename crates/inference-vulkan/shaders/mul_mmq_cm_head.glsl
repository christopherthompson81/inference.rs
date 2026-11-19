// Shared by the mul_mmq_cm_* shaders: y = W x for many activation rows (a prompt's tokens) against a quantized W
// (`rows x cols`) on int8 matrix cores (VK_KHR_cooperative_matrix, 16 x 16 x 16 int8 into int32), the rows as
// quantize_q8_1's int8 blocks. A workgroup of 8 subgroups of 32 takes a BM x BN tile (weight rows x tokens), stepping
// BK blocks at a time through shared memory; each block's int32 product is scaled into f32 sums element by element,
// which reads the accumulator's lane layout (RDNA3's: lane l holds column l % 16, rows l / 16 + 2 e). The including
// shader declares the push constants (with rows, cols, y_base and s_off), a `*_int8.glsl` whose blocks hold one
// scale and offset (no SPLIT_HALVES), and the tile's hooks as mat_mul_q_head.glsl's take them.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer W { uint16_t w[]; };
layout(std430, binding = 1) readonly buffer Q { uvec4 xq[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };
layout(std430, binding = 3) readonly buffer S { vec2 xs[]; };

#include "int8_dot.glsl"
