// An IQ4_XS block (68 u16s, 256 values): f16 d, 16 bits of scales' high pairs, 4 bytes of their low nibbles, then
// 128 bytes of nibbles; sub-block i's 16 bytes 16i..16i + 15 hold its values 0..15 low and 16..31 high, each an index
// into kvalues_iq4nl; a value is d (ls - 32) kvalues_iq4nl[q], ls = scales_l nibble i | scales_h pair i << 4.

const uint SUPER_VALUES = 256;
#define SPLIT_HALVES 0
#define NO_OFFSET
const uint SUPER_U16 = 68;

// kvalues_iq4nl's 16 int8 values, four a word (an indexed array goes to local memory on NVIDIA), for the shared table
const uint IQ4_NL_0 = 0xbfad9881u;
const uint IQ4_NL_1 = 0xf6eaddcfu;
const uint IQ4_NL_2 = 0x26190d01u;
const uint IQ4_NL_3 = 0x71594535u;

uint iq4_nl_value(uint idx) {
    uint quad = idx >> 2;
    uint table = quad == 0u ? IQ4_NL_0 : (quad == 1u ? IQ4_NL_1 : (quad == 2u ? IQ4_NL_2 : IQ4_NL_3));
    return (table >> (8 * (idx & 3u))) & 0xffu;
}

// byte b's two table values, b & 15's in bits 0..7 and b >> 4's in bits 16..23: a byte's nibbles in one read
shared uint IQ4_PAIRS[256];

#define TYPE_SHARED_INIT
void type_shared_init() {
    uint threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    for (uint i = gl_LocalInvocationIndex; i < 256; i += threads) {
        IQ4_PAIRS[i] = iq4_nl_value(i & 0xfu) | (iq4_nl_value(i >> 4) << 16);
    }
    barrier();
}

// a word of four index bytes as the four low nibbles' table values and the four high ones'
void iq4_nl_words(uint q, out uint low, out uint high) {
    low = 0u;
    high = 0u;
    for (uint k = 0; k < 4; k++) {
        uint pair = IQ4_PAIRS[(q >> (8 * k)) & 0xffu];
        low |= (pair & 0xffu) << (8 * k);
        high |= (pair >> 16) << (8 * k);
    }
}

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint at = base + 4 + 8 * sub;
    uvec4 q = uvec4(word(at), word(at + 2), word(at + 4), word(at + 6));
    for (uint i = 0; i < 4; i++) {
        uint l, h;
        iq4_nl_words(q[i], l, h);
        lo[i] = l;
        hi[i] = h;
    }
    uint scale_byte = (uint(w[base + 2 + sub / 4]) >> (8 * ((sub / 2) % 2))) & 0xffu;
    uint ls = ((scale_byte >> (4 * (sub % 2))) & 0xfu) | (((uint(w[base + 1]) >> (2 * sub)) & 3u) << 4);
    float s = half_at(base) * float(int(ls) - 32);
    return vec4(s, s, 0.0, 0.0);
}
