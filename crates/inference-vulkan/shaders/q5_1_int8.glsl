// A Q5_1 block (12 u16s): f16 scale and min, Q5_0's fifth bits and nibbles; a value is d q + m.
#include "nibbles.glsl"

const uint SUPER_VALUES = 32;
#define SPLIT_HALVES 0
const uint SUPER_U16 = 12;

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint qh = word(base + 2);
    uvec4 q = uvec4(word(base + 4), word(base + 6), word(base + 8), word(base + 10));
    lo = (q & 0x0f0f0f0fu) | (fifth_bits(qh) << 4);
    hi = ((q >> 4) & 0x0f0f0f0fu) | (fifth_bits(qh >> 16) << 4);
    float d = half_at(base), m = half_at(base + 1);
    return vec4(d, d, m, m);
}
