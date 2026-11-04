// A Q5_K block (88 u16s, 256 values): Q4_K's d, dmin and scales, 32 bytes of fifth bits (value l of sub-block j at
// bit j of byte l), then Q4_K's 128 bytes of nibbles; a value is d sc q - dmin m.
#include "k_scales.glsl"
#include "nibbles.glsl"

const uint SUPER_VALUES = 256;
#define SPLIT_HALVES 0
const uint SUPER_U16 = 88;

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint at = base + 24 + 16 * (sub / 2);
    uint shift = 4 * (sub % 2);
    uvec4 a = uvec4(word(at), word(at + 2), word(at + 4), word(at + 6));
    uvec4 b = uvec4(word(at + 8), word(at + 10), word(at + 12), word(at + 14));
    // the fifth bits of values 0..31, bit `sub` of each byte, gathered to the low bit
    uvec4 ha = (uvec4(word(base + 8), word(base + 10), word(base + 12), word(base + 14)) >> sub) & 0x01010101u;
    uvec4 hb = (uvec4(word(base + 16), word(base + 18), word(base + 20), word(base + 22)) >> sub) & 0x01010101u;
    lo = ((a >> shift) & 0x0f0f0f0fu) | (ha << 4);
    hi = ((b >> shift) & 0x0f0f0f0fu) | (hb << 4);
    uvec2 sm = scale_min(base + 2, sub);
    float s = half_at(base) * float(sm.x), o = -half_at(base + 1) * float(sm.y);
    return vec4(s, s, o, o);
}
