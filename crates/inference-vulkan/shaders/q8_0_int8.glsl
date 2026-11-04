// A Q8_0 block (17 u16s): an f16 scale, then 32 int8 values.

const uint SUPER_VALUES = 32;
const uint SUPER_U16 = 17;

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    lo = uvec4(word(base + 1), word(base + 3), word(base + 5), word(base + 7));
    hi = uvec4(word(base + 9), word(base + 11), word(base + 13), word(base + 15));
    float d = half_at(base);
    return vec4(d, d, 0.0, 0.0);
}
