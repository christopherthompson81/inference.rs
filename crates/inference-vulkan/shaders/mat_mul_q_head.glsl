// Shared by the mul_mm_* and moe_mm_* shaders: y = W x for many activation rows (a prompt's tokens) against a
// quantized W (`rows x cols`), the rows as quantize_q8_1's int8 blocks. A workgroup takes a BN x BM tile (weight rows
// x tokens), stepping BK blocks at a time: its threads unpack one weight block and one token block each into shared
// memory as packed int8, then each takes TN x TM outputs with packed dot products. The including shader declares the
// push constants (with rows, cols, y_base and s_off) and, before mat_mul_q_main.glsl, a `*_int8.glsl` and the tile's
// hooks: `begin_tile` (false to leave), and the activation and output rows of its token j. It may define TN and TM (a
// thread's weight rows and tokens, so the tile's BN and BM) and BK (the blocks a step) first.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer W { uint16_t w[]; };
layout(std430, binding = 1) readonly buffer Q { uvec4 xq[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };
layout(std430, binding = 3) readonly buffer S { vec4 xs[]; };

#include "int8_dot.glsl"


const uint THREADS = 256;
const uint QK = 32;

#ifndef BK
#define BK 4
#endif
#ifndef TN
#define TN 4
#endif
#ifndef TM
#define TM 4
#endif
// threads across the tile's weight rows; a thread's outputs sit THREADS_N weight rows and THREADS_M tokens apart
const uint THREADS_N = 16;
const uint THREADS_M = THREADS / THREADS_N;
const uint BN = TN * THREADS_N;
const uint BM = TM * THREADS_M;

// a block's 8 packed words as two uvec4s, and its coefficients (a token block's scales); a row padded to spread banks
shared uvec4 w_lo[BK][BN + 1];
shared uvec4 w_hi[BK][BN + 1];
shared vec4 w_c[BK][BN + 1];
shared uvec4 x_lo[BK][BM + 1];
shared uvec4 x_hi[BK][BM + 1];
shared vec4 x_s[BK][BM + 1];
