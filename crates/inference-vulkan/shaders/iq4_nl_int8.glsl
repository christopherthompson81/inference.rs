// An IQ4_NL block (9 u16s): an f16 d, then 16 bytes, value j in byte j % 16's low (j < 16) or high half, an index
// into kvalues_iq4nl; a value is d kvalues_iq4nl[q].

const uint SUPER_VALUES = 32;
#define SPLIT_HALVES 0
#define NO_OFFSET
const uint SUPER_U16 = 9;

#include "iq4_nl_table.glsl"

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uvec4 q = uvec4(word(base + 1), word(base + 3), word(base + 5), word(base + 7));
    for (uint i = 0; i < 4; i++) {
        uint l, h;
        iq4_nl_words(q[i], l, h);
        lo[i] = l;
        hi[i] = h;
    }
    float d = half_at(base);
    return vec4(d, d, 0.0, 0.0);
}
