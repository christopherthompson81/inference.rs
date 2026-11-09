// A direct 2D convolution, a thread an output: x `(b, groups * c_in, i_h, i_w)` strided, the kernel contiguous
// `(groups * c_out, c_in, k_h, k_w)`, the output contiguous `(b, groups * c_out, o_h, o_w)`; zero padding, as the
// CPU's conv2d (a conv1d is one row, padded along w alone). With BF16 every operand is bf16, summed in f32.
#ifdef BF16
#extension GL_EXT_shader_16bit_storage : require
#include "bf16.glsl"
#define ELEM uint16_t
#define LOAD(v) bf16_bits_to_f32(uint(v))
#define STORE(v) uint16_t(f32_to_bf16_bits(v))
#else
#define ELEM float
#define LOAD(v) (v)
#define STORE(v) (v)
#endif

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer X { ELEM x[]; };
layout(std430, binding = 1) readonly buffer W { ELEM w[]; };
layout(std430, binding = 2) writeonly buffer O { ELEM o[]; };

layout(push_constant) uniform PushConstants {
    uint batch;
    // a group's channels
    uint c_in;
    uint c_out;
    uint groups;
    uint i_h;
    uint i_w;
    uint k_h;
    uint k_w;
    uint o_h;
    uint o_w;
    uint pad_h;
    uint pad_w;
    uint stride;
    uint dilation;
    uint x_off;
    uint x_sb;
    uint x_sc;
    uint x_sh;
    uint x_sw;
    uint w_off;
} pc;

void main() {
    uint total = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    uint channels = pc.groups * pc.c_out;
    uint count = pc.batch * channels * pc.o_h * pc.o_w;
    for (uint e = gl_GlobalInvocationID.y * total + gl_GlobalInvocationID.x; e < count; e += total * gl_NumWorkGroups.y) {
        uint ow = e % pc.o_w;
        uint oh = (e / pc.o_w) % pc.o_h;
        uint co = (e / (pc.o_w * pc.o_h)) % channels;
        uint b = e / (pc.o_w * pc.o_h * channels);
        uint g = co / pc.c_out;
        // the window's top-left, padding included
        int h0 = int(oh * pc.stride) - int(pc.pad_h);
        int w0 = int(ow * pc.stride) - int(pc.pad_w);
        float s = 0.0;
        for (uint ci = 0; ci < pc.c_in; ci++) {
            uint xc = pc.x_off + b * pc.x_sb + (g * pc.c_in + ci) * pc.x_sc;
            uint wc = pc.w_off + (co * pc.c_in + ci) * pc.k_h * pc.k_w;
            for (uint kh = 0; kh < pc.k_h; kh++) {
                int ih = h0 + int(kh * pc.dilation);
                if (ih < 0 || ih >= int(pc.i_h)) {
                    continue;
                }
                for (uint kw = 0; kw < pc.k_w; kw++) {
                    int iw = w0 + int(kw * pc.dilation);
                    if (iw < 0 || iw >= int(pc.i_w)) {
                        continue;
                    }
                    s += LOAD(x[xc + uint(ih) * pc.x_sh + uint(iw) * pc.x_sw]) * LOAD(w[wc + kh * pc.k_w + kw]);
                }
            }
        }
        o[e] = STORE(s);
    }
}
