// Shared by the embed_* shaders: the dequantized rows of a quantized `vocab x cols` matrix at u32 `ids`, row-major
// floats, a thread a 32-value block. The including shader includes a `*_int8.glsl` between this and embed_main.glsl.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_integer_dot_product : require

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer W { uint16_t w[]; };
layout(std430, binding = 1) readonly buffer I { uint ids[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform PushConstants {
    uint vocab;
    // 32-value blocks a row
    uint blocks;
    // in u16s
    uint w_base;
    uint ids_off;
    uint count;
} pc;

#include "int8_dot.glsl"
