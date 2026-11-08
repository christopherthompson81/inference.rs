// An IQ3_S block (55 u16s, 256 values; ggml's dequantize_row_iq3_s): f16 d, 64 bytes of grid indices' low bits, 8
// bytes of their ninth bits, 32 bytes of signs, 4 bytes of scale nibbles. Sub-block i (32 values): scale d (1 + 2 n),
// n scale nibble i; its 8 values at 8l takes grid entries qs[8i + 2l] and qs[8i + 2l + 1] (ninth bits from qh[i]
// shifted 8 - 2l and 7 - 2l), each four unsigned values, negated where bit j of signs[4i + l] is set.
#include "iq3s_grid.glsl"

// the grid in shared memory: a constant array indexed per lane serializes on NVIDIA
shared uint IQ3S_GRID_SH[512];

#define TYPE_SHARED_INIT
void type_shared_init() {
    uint threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    for (uint i = gl_LocalInvocationIndex; i < 512; i += threads) {
        IQ3S_GRID_SH[i] = IQ3S_GRID[i];
    }
    barrier();
}

const uint SUPER_VALUES = 256;
#define SPLIT_HALVES 0
#define NO_OFFSET
const uint SUPER_U16 = 55;

uint iq3s_byte(uint base, uint i) {
    return (uint(w[base + i / 2]) >> (8 * (i % 2))) & 0xffu;
}

// a grid entry's four values as int8, negated where the low four bits of `signs` say
uint iq3s_signed(uint grid, uint signs) {
    uint packed = 0u;
    for (uint k = 0; k < 4; k++) {
        uint v = (grid >> (8 * k)) & 0xffu;
        if (((signs >> k) & 1u) != 0u) {
            v = (256u - v) & 0xffu;
        }
        packed |= v << (8 * k);
    }
    return packed;
}

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint qh = iq3s_byte(base, 66 + sub);
    uint words[8];
    for (uint l = 0; l < 4; l++) {
        uint q = 2 + 8 * sub + 2 * l;
        uint i1 = iq3s_byte(base, q) | ((qh << (8 - 2 * l)) & 256u);
        uint i2 = iq3s_byte(base, q + 1) | ((qh << (7 - 2 * l)) & 256u);
        uint signs = iq3s_byte(base, 74 + 4 * sub + l);
        words[2 * l] = iq3s_signed(IQ3S_GRID_SH[i1], signs);
        words[2 * l + 1] = iq3s_signed(IQ3S_GRID_SH[i2], signs >> 4);
    }
    lo = uvec4(words[0], words[1], words[2], words[3]);
    hi = uvec4(words[4], words[5], words[6], words[7]);
    uint n = (iq3s_byte(base, 106 + sub / 2) >> (4 * (sub % 2))) & 0xfu;
    float s = half_at(base) * float(1 + 2 * n);
    return vec4(s, s, 0.0, 0.0);
}
