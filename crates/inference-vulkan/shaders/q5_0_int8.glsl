// A Q5_0 block (11 u16s): an f16 scale, the 32 values' fifth bits, then their low nibbles, value j in byte j % 16's
// low (j < 16) or high half; a value is its 5 bits less BIAS, which the matvec takes off through the activations' sum.

const uint BLOCK_U16 = 11;
const float BIAS = 16.0;

// a block's words as loaded, so the loads can be issued well before the unpacking
struct Raw {
    uint d;
    uint qh;
    uvec4 q;
};

Raw read_raw(uint base) {
    return Raw(
        uint(w[base]),
        uint(w[base + 1]) | (uint(w[base + 2]) << 16),
        uvec4(
            uint(w[base + 3]) | (uint(w[base + 4]) << 16),
            uint(w[base + 5]) | (uint(w[base + 6]) << 16),
            uint(w[base + 7]) | (uint(w[base + 8]) << 16),
            uint(w[base + 9]) | (uint(w[base + 10]) << 16)));
}

// four bits to the low bit of each of four bytes
uint spread(uint bits) {
    return ((bits & 0xfu) * 0x00204081u) & 0x01010101u;
}

// the scale, and values 0..15 (lo) and 16..31 (hi) as unsigned bytes 0..31, four a word
float unpack_raw(Raw r, out uvec4 lo, out uvec4 hi) {
    lo = (r.q & 0x0f0f0f0fu)
        | (uvec4(spread(r.qh), spread(r.qh >> 4), spread(r.qh >> 8), spread(r.qh >> 12)) << 4);
    hi = ((r.q >> 4) & 0x0f0f0f0fu)
        | (uvec4(spread(r.qh >> 16), spread(r.qh >> 20), spread(r.qh >> 24), spread(r.qh >> 28)) << 4);
    return unpackHalf2x16(r.d).x;
}

float unpack_block(uint base, out uvec4 lo, out uvec4 hi) {
    return unpack_raw(read_raw(base), lo, hi);
}
