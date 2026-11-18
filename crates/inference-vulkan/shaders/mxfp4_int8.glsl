// An MXFP4 block as uploaded (9 u16s, 32 values): the E8M0 scale byte and a zero byte, then the file's 16 bytes, value
// j in byte j % 16's low (j < 16) or high half, each an index into kvalues_fp4 (twice E2M1); a value is
// 2^(e - 128) kvalues_fp4[q].

const uint SUPER_VALUES = 32;
#define SPLIT_HALVES 0
#define NO_OFFSET
const uint SUPER_U16 = 9;

// kvalues_fp4's 16 int8 values, four a word (an indexed array goes to local memory on NVIDIA), for the shared table
const uint FP4_0 = 0x03020100u;
const uint FP4_1 = 0x0c080604u;
const uint FP4_2 = 0xfdfeff00u;
const uint FP4_3 = 0xf4f8fafcu;

uint fp4_value(uint idx) {
    uint quad = idx >> 2;
    uint table = quad == 0u ? FP4_0 : (quad == 1u ? FP4_1 : (quad == 2u ? FP4_2 : FP4_3));
    return (table >> (8 * (idx & 3u))) & 0xffu;
}

// byte b's two table values, b & 15's in bits 0..7 and b >> 4's in bits 16..23: a byte's nibbles in one read
shared uint FP4_PAIRS[256];

#define TYPE_SHARED_INIT
void type_shared_init() {
    uint threads = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    for (uint i = gl_LocalInvocationIndex; i < 256; i += threads) {
        FP4_PAIRS[i] = fp4_value(i & 0xfu) | (fp4_value(i >> 4) << 16);
    }
    barrier();
}

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uvec4 q = uvec4(word(base + 1), word(base + 3), word(base + 5), word(base + 7));
    for (uint i = 0; i < 4; i++) {
        uint l = 0u, h = 0u;
        for (uint k = 0; k < 4; k++) {
            uint pair = FP4_PAIRS[(q[i] >> (8 * k)) & 0xffu];
            l |= (pair & 0xffu) << (8 * k);
            h |= (pair >> 16) << (8 * k);
        }
        lo[i] = l;
        hi[i] = h;
    }
    float d = ldexp(1.0, int(uint(W16(base)) & 0xffu) - 128);
    return vec4(d, d, 0.0, 0.0);
}
