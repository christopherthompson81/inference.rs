// A Q4_K block (72 u16s, 256 values): f16 d and dmin, 12 bytes of 6-bit sub-block scales and mins, then 128 bytes
// of nibbles, sub-blocks 2i and 2i + 1 the low and high halves of bytes 32i..32i + 31; a value is d sc q - dmin m.
#include "k_scales.glsl"

const uint SUPER_VALUES = 256;
const uint SUPER_U16 = 72;

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint at = base + 8 + 16 * (sub / 2);
    uint shift = 4 * (sub % 2);
    uvec4 a = uvec4(word(at), word(at + 2), word(at + 4), word(at + 6));
    uvec4 b = uvec4(word(at + 8), word(at + 10), word(at + 12), word(at + 14));
    lo = (a >> shift) & 0x0f0f0f0fu;
    hi = (b >> shift) & 0x0f0f0f0fu;
    uvec2 sm = scale_min(base + 2, sub);
    float s = half_at(base) * float(sm.x), o = -half_at(base + 1) * float(sm.y);
    return vec4(s, s, o, o);
}
