// The head of the moe_mm_cm2_* builds (extensions, bindings, push constants), before a cm2_<type>.glsl (the weights'
// block, BLOCK_U16 u16s of 32 values, and its decode functions `decode_w` and `decode_w_v`) and moe_mm_cm2_main.glsl.
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_NV_cooperative_matrix2 : require
#extension GL_EXT_buffer_reference : require
#extension GL_NV_cooperative_matrix_decode_vector : require

// 4 subgroups: a 128 x 16 matrix needs no more (8 get it wrong on the 3090's driver), and a tile costs less
layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer W { uint16_t w[]; };
layout(std430, binding = 1) readonly buffer H { float16_t h[]; };
layout(std430, binding = 1) readonly buffer H4 { f16vec4 h4[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };
layout(std430, binding = 3) readonly buffer Routing { uint routing[]; };

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
    // element offsets: `w` in u16s, `h` in halves, `y` in floats
    uint w_base;
    uint h_base;
    uint y_base;
    // u16s from one expert's matrix to the next
    uint expert_u16;
    // where moe_sort's tile count and tiles start in `routing` (its `order` at 0)
    uint tiles_off;
    // the gathered rows, padding included
    uint h_rows;
} pc;

const uint BM = 128;
const uint BN_QUARTER = 16;
const uint BK = 64;
const uint QK = 32;
