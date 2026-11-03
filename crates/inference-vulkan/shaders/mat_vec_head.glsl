// Shared by the multi-row mul_mat_vecs_* shaders (mul_mat_vec_* take one row): y[m] = W x[m] for up to MAX_M rows of
// `cols` f32s (contiguous from x_base) against a quantized W (`rows x cols`), each weight block decoded once for every
// row. A workgroup takes ROWS weight rows; its threads stride over the blocks, then sum by subgroup. The including
// shader defines BLOCK_U16 (a block's u16s) and `decode` (a block's 32 values, scaled) before mat_vec_main.glsl.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer W { uint16_t w[]; };
layout(std430, binding = 1) readonly buffer X { float x[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
    // element offsets into the bound buffers: `w` in u16s, `x` and `y` in floats
    uint w_base;
    uint x_base;
    uint y_base;
    // workgroups per grid row, the grid's second dimension carrying what one dimension's limit cannot
    uint groups_x;
    // activation rows, each `cols` long; output row m is `rows` long
    uint m;
} pc;

const uint THREADS = 64;
const uint QK = 32;
// the main loop's two sums assume two rows a workgroup
const uint ROWS = 2;
const uint MAX_M = 8;
