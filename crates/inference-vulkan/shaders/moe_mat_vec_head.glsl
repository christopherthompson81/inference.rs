// Shared by the moe_mat_vecs_* shaders: y[slot] = W[ids[slot]] x[slot / x_div] for a stack of experts W (each
// `rows x cols`), the activation rows as quantize_q8_1's int8 blocks: grid z is a slot, and the slot leading a run of
// up to MAX_M of its expert's slots computes them all (each weight block packed to int8 once for every member) while
// the rest leave. A workgroup takes ROWS weight rows. The including shader includes its `*_int8.glsl` between this
// and moe_mat_vec_main.glsl.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 32) in;

layout(std430, binding = 0) readonly buffer W { uint16_t w[]; };
layout(std430, binding = 1) readonly buffer Q { uvec4 xq[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };
layout(std430, binding = 3) readonly buffer Ids { uint ids[]; };
layout(std430, binding = 4) readonly buffer S { vec2 xs[]; };

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
    // element offsets into the bound buffers: `w` in u16s, `y` in floats
    uint w_base;
    uint y_base;
    // workgroups per grid row, the grid's second dimension carrying what one dimension's limit cannot
    uint groups_x;
    // slots an activation row serves (a projection of the token's input: its routes), or 1 (of each route's own)
    uint x_div;
    uint ids_off;
    // u16s from one expert's matrix to the next
    uint expert_u16;
    uint slots;
    // where the blocks' (d, d * sum q) start, in vec2s
    uint s_off;
} pc;

const uint THREADS = 32;
const uint QK = 32;
const uint ROWS = 4;
const uint MAX_M = 8;
const uint MAX_SUBGROUPS = THREADS / 8;
shared float partial[ROWS * MAX_M][MAX_SUBGROUPS];

#include "int8_dot.glsl"
