// A Q8_0 block (17 u16s): an f16 scale, then 32 int8 values.

const uint BLOCK_U16 = 17;
const float BIAS = 0.0;

// a block's words as loaded, so the loads can be issued well before the unpacking
struct Raw {
    uint d;
    uvec4 lo;
    uvec4 hi;
};

Raw read_raw(uint base) {
    uint v[8];
    [[unroll]] for (uint i = 0; i < 8; i++) {
        v[i] = uint(w[base + 1 + 2 * i]) | (uint(w[base + 2 + 2 * i]) << 16);
    }
    return Raw(uint(w[base]), uvec4(v[0], v[1], v[2], v[3]), uvec4(v[4], v[5], v[6], v[7]));
}

// the scale, and values 0..15 (lo) and 16..31 (hi), four a word
float unpack_raw(Raw r, out uvec4 lo, out uvec4 hi) {
    lo = r.lo;
    hi = r.hi;
    return unpackHalf2x16(r.d).x;
}

float unpack_block(uint base, out uvec4 lo, out uvec4 hi) {
    return unpack_raw(read_raw(base), lo, hi);
}
