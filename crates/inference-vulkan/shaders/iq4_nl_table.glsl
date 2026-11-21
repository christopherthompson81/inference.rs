// kvalues_iq4nl, the 16-entry nonlinear table IQ4_NL and IQ4_XS index, and the shared pairs their unpacking reads.

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
