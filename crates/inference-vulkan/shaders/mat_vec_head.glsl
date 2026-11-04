// Shared by the multi-row mul_mat_vecs_* shaders (mul_mat_vec_* take one row): y[m] = W x[m] for up to MAX_M
// activation rows against a quantized W (`rows x cols`), with the rows as quantize_q8_1's int8 blocks (8 packed words
// and (d, d * sum q) a block, row-major over the rows' blocks). A workgroup takes ROWS weight rows; its threads stride
// over the 32-value blocks, each weight block's values packed to int8 once and dotted four at a time with every row's,
// then sum by subgroup. The including shader includes a `*_int8.glsl` between this and mat_vec_main.glsl: its
// SUPER_VALUES (values a stored block holds, 32 or 256) and SUPER_U16 (its size), and `unpack_sub`, which turns 32
// values of a stored block into int8 halves q (values 0..15 and 16..31) and coefficients (s_lo, s_hi, o_lo, o_hi)
// with w = s q + o over each half.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer W { uint16_t w[]; };
layout(std430, binding = 1) readonly buffer Q { uvec4 xq[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };
layout(std430, binding = 3) readonly buffer S { vec2 xs[]; };

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
    // element offsets into the bound buffers: `w` in u16s, `y` in floats
    uint w_base;
    uint y_base;
    // workgroups per grid row, the grid's second dimension carrying what one dimension's limit cannot
    uint groups_x;
    // activation rows; output row m is `rows` long
    uint m;
    // where the blocks' (d, d * sum q) start, in vec2s (the packed words come first in the same buffer)
    uint s_off;
} pc;

const uint THREADS = 64;
const uint QK = 32;
const uint ROWS = 4;
const uint MAX_M = 8;
// a subgroup's sums, combined across the workgroup's subgroups by thread 0
const uint MAX_SUBGROUPS = THREADS / 8;
shared float partial[ROWS * MAX_M][MAX_SUBGROUPS];
#include "int8_dot.glsl"
