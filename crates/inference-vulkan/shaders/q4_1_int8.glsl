// A Q4_1 block (10 u16s): f16 scale and min, then Q4_0's 16 bytes of nibbles; a value is d q + m.

const uint SUPER_VALUES = 32;
#define SPLIT_HALVES 0
const uint SUPER_U16 = 10;

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uvec4 q = uvec4(word(base + 2), word(base + 4), word(base + 6), word(base + 8));
    lo = q & 0x0f0f0f0fu;
    hi = (q >> 4) & 0x0f0f0f0fu;
    float d = half_at(base), m = half_at(base + 1);
    return vec4(d, d, m, m);
}
