// Per-row ops over `rows` contiguous rows of `cols` f32s, a workgroup a row: RMS norm (times `alpha`), layer norm (times
// `alpha`, plus `beta`), softmax, and the reductions; an arg reduction writes the index's bits. With BF16 every operand
// is bf16 (f16 under HALF_F16), computed in f32.
#ifdef BF16
#extension GL_EXT_shader_16bit_storage : require
#include "half.glsl"
#define ELEM uint16_t
#define LOAD(v) half_to_f32(uint(v))
#define STORE(v) uint16_t(f32_to_half(v))
#else
#define ELEM float
#define LOAD(v) (v)
#define STORE(v) (v)
#endif

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer X { ELEM x[]; };
layout(std430, binding = 1) readonly buffer Alpha { ELEM alpha[]; };
layout(std430, binding = 2) buffer O { ELEM o[]; };
layout(std430, binding = 3) readonly buffer Beta { ELEM beta[]; };

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
    uint op;
    float eps;
    uint x_off;
    uint alpha_off;
    uint o_off;
    uint beta_off;
} pc;

const uint THREADS = 256;
// op codes, as `RowOp` numbers them
const uint RMS_NORM = 0;
const uint SOFTMAX = 1;
const uint SUM = 2;
const uint MAX = 3;
const uint MIN = 4;
const uint ARGMAX = 5;
const uint ARGMIN = 6;
const uint LAYER_NORM = 7;

shared float red[THREADS];
shared uint arg[THREADS];

// `v` over the workgroup, by sum, max or min; the arg reductions carry the winning index in `arg`
float reduce(float v, uint idx, uint kind) {
    uint t = gl_LocalInvocationID.x;
    red[t] = v;
    arg[t] = idx;
    barrier();
    for (uint s = THREADS / 2; s > 0; s >>= 1) {
        if (t < s) {
            float a = red[t];
            float b = red[t + s];
            if (kind == SUM) {
                red[t] = a + b;
            } else if (kind == MAX || kind == ARGMAX) {
                // ties to the lower index, as the CPU's scan
                if (b > a || (b == a && arg[t + s] < arg[t])) {
                    red[t] = b;
                    arg[t] = arg[t + s];
                }
            } else if (b < a || (b == a && arg[t + s] < arg[t])) {
                red[t] = b;
                arg[t] = arg[t + s];
            }
        }
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}

void main() {
    uint row = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    if (row >= pc.rows) {
        return;
    }
    uint t = gl_LocalInvocationID.x;
    uint base = pc.x_off + row * pc.cols;
    if (pc.op == RMS_NORM) {
        float s = 0.0;
        for (uint c = t; c < pc.cols; c += THREADS) {
            float v = LOAD(x[base + c]);
            s += v * v;
        }
        float scale = 1.0 / sqrt(reduce(s, 0, SUM) / float(pc.cols) + pc.eps);
        for (uint c = t; c < pc.cols; c += THREADS) {
            o[pc.o_off + row * pc.cols + c] = STORE(LOAD(x[base + c]) * scale * LOAD(alpha[pc.alpha_off + c]));
        }
    } else if (pc.op == LAYER_NORM) {
        // the variance as E[x^2] - mean^2, as the CPU's
        float s = 0.0;
        float s2 = 0.0;
        for (uint c = t; c < pc.cols; c += THREADS) {
            float v = LOAD(x[base + c]);
            s += v;
            s2 += v * v;
        }
        float mean = reduce(s, 0, SUM) / float(pc.cols);
        float var = reduce(s2, 0, SUM) / float(pc.cols) - mean * mean;
        float inv_std = 1.0 / sqrt(var + pc.eps);
        for (uint c = t; c < pc.cols; c += THREADS) {
            uint o_at = pc.o_off + row * pc.cols + c;
            o[o_at] = STORE((LOAD(x[base + c]) - mean) * inv_std * LOAD(alpha[pc.alpha_off + c]) + LOAD(beta[pc.beta_off + c]));
        }
    } else if (pc.op == SOFTMAX) {
        float m = -1.0 / 0.0;
        for (uint c = t; c < pc.cols; c += THREADS) {
            m = max(m, LOAD(x[base + c]));
        }
        m = reduce(m, 0, MAX);
        float s = 0.0;
        for (uint c = t; c < pc.cols; c += THREADS) {
            s += exp(LOAD(x[base + c]) - m);
        }
        s = reduce(s, 0, SUM);
        for (uint c = t; c < pc.cols; c += THREADS) {
            o[pc.o_off + row * pc.cols + c] = STORE(exp(LOAD(x[base + c]) - m) / s);
        }
    } else {
        bool is_max = pc.op == MAX || pc.op == ARGMAX;
        float v = pc.op == SUM ? 0.0 : (is_max ? -1.0 / 0.0 : 1.0 / 0.0);
        uint best = 0xffffffffu;
        for (uint c = t; c < pc.cols; c += THREADS) {
            float e = LOAD(x[base + c]);
            if (pc.op == SUM) {
                v += e;
            } else if ((is_max && e > v) || (!is_max && e < v) || best == 0xffffffffu) {
                v = e;
                best = c;
            }
        }
        float r = reduce(v, best, pc.op);
        if (t == 0) {
#ifdef BF16
            // an index's u32 as its two halves
            if (pc.op == ARGMAX || pc.op == ARGMIN) {
                o[2 * (pc.o_off + row)] = uint16_t(arg[0] & 0xffffu);
                o[2 * (pc.o_off + row) + 1] = uint16_t(arg[0] >> 16);
            } else {
                o[pc.o_off + row] = STORE(r);
            }
#else
            o[pc.o_off + row] = pc.op == ARGMAX || pc.op == ARGMIN ? uintBitsToFloat(arg[0]) : r;
#endif
        }
    }
}
