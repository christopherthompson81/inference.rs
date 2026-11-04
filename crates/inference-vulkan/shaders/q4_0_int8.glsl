// A Q4_0 block (9 u16s): an f16 scale, then 16 bytes, value j in byte j % 16's low (j < 16) or high half; a value is
// its 4 bits less 8.

const uint SUPER_VALUES = 32;
const uint SUPER_U16 = 9;

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uvec4 q = uvec4(word(base + 1), word(base + 3), word(base + 5), word(base + 7));
    lo = q & 0x0f0f0f0fu;
    hi = (q >> 4) & 0x0f0f0f0fu;
    float d = half_at(base);
    return vec4(d, d, -8.0 * d, -8.0 * d);
}
