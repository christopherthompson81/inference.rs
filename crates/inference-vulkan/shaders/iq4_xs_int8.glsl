// An IQ4_XS block (68 u16s, 256 values): f16 d, 16 bits of scales' high pairs, 4 bytes of their low nibbles, then
// 128 bytes of nibbles; sub-block i's 16 bytes 16i..16i + 15 hold its values 0..15 low and 16..31 high, each an index
// into kvalues_iq4nl; a value is d (ls - 32) kvalues_iq4nl[q], ls = scales_l nibble i | scales_h pair i << 4.

const uint SUPER_VALUES = 256;
#define SPLIT_HALVES 0
#define NO_OFFSET
const uint SUPER_U16 = 68;

// kvalues_iq4nl's 16 int8 values, four a word, so a lookup is selects and a shift rather than an indexed array
// (which NVIDIA's compiler puts in local memory)
const uint IQ4_NL_0 = 0xbfad9881u;
const uint IQ4_NL_1 = 0xf6eaddcfu;
const uint IQ4_NL_2 = 0x26190d01u;
const uint IQ4_NL_3 = 0x71594535u;

// four nibble indices (a byte each) as their four int8 table values
uint iq4_nl_bytes(uint nibbles) {
    uint packed = 0u;
    for (uint k = 0; k < 4; k++) {
        uint idx = (nibbles >> (8 * k)) & 0xfu;
        uint quad = idx >> 2;
        uint table = quad == 0u ? IQ4_NL_0 : (quad == 1u ? IQ4_NL_1 : (quad == 2u ? IQ4_NL_2 : IQ4_NL_3));
        packed |= ((table >> (8 * (idx & 3u))) & 0xffu) << (8 * k);
    }
    return packed;
}

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint at = base + 4 + 8 * sub;
    uvec4 q = uvec4(word(at), word(at + 2), word(at + 4), word(at + 6));
    uvec4 low = q & 0x0f0f0f0fu, high = (q >> 4) & 0x0f0f0f0fu;
    lo = uvec4(iq4_nl_bytes(low.x), iq4_nl_bytes(low.y), iq4_nl_bytes(low.z), iq4_nl_bytes(low.w));
    hi = uvec4(iq4_nl_bytes(high.x), iq4_nl_bytes(high.y), iq4_nl_bytes(high.z), iq4_nl_bytes(high.w));
    uint scale_byte = (uint(w[base + 2 + sub / 4]) >> (8 * ((sub / 2) % 2))) & 0xffu;
    uint ls = ((scale_byte >> (4 * (sub % 2))) & 0xfu) | (((uint(w[base + 1]) >> (2 * sub)) & 3u) << 4);
    float s = half_at(base) * float(int(ls) - 32);
    return vec4(s, s, 0.0, 0.0);
}
