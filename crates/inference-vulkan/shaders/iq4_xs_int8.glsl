// An IQ4_XS block (68 u16s, 256 values): f16 d, 16 bits of scales' high pairs, 4 bytes of their low nibbles, then
// 128 bytes of nibbles; sub-block i's 16 bytes 16i..16i + 15 hold its values 0..15 low and 16..31 high, each an index
// into kvalues_iq4nl; a value is d (ls - 32) kvalues_iq4nl[q], ls = scales_l nibble i | scales_h pair i << 4.

const uint SUPER_VALUES = 256;
#define SPLIT_HALVES 0
#define NO_OFFSET
const uint SUPER_U16 = 68;

#include "iq4_nl_table.glsl"

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint at = base + 4 + 8 * sub;
    uvec4 q = uvec4(word(at), word(at + 2), word(at + 4), word(at + 6));
    for (uint i = 0; i < 4; i++) {
        uint l, h;
        iq4_nl_words(q[i], l, h);
        lo[i] = l;
        hi[i] = h;
    }
    uint scale_byte = (uint(W16(base + 2 + sub / 4)) >> (8 * ((sub / 2) % 2))) & 0xffu;
    uint ls = ((scale_byte >> (4 * (sub % 2))) & 0xfu) | (((uint(W16(base + 1)) >> (2 * sub)) & 3u) << 4);
    float s = half_at(base) * float(int(ls) - 32);
    return vec4(s, s, 0.0, 0.0);
}
