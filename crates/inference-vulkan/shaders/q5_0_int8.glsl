// A Q5_0 block (11 u16s): an f16 scale, the 32 values' fifth bits, then their low nibbles, value j in byte j % 16's
// low (j < 16) or high half; a value is its 5 bits less 16.
#include "nibbles.glsl"

const uint SUPER_VALUES = 32;
const uint SUPER_U16 = 11;

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint qh = word(base + 1);
    uvec4 q = uvec4(word(base + 3), word(base + 5), word(base + 7), word(base + 9));
    lo = (q & 0x0f0f0f0fu) | (fifth_bits(qh) << 4);
    hi = ((q >> 4) & 0x0f0f0f0fu) | (fifth_bits(qh >> 16) << 4);
    float d = half_at(base);
    return vec4(d, d, -16.0 * d, -16.0 * d);
}
