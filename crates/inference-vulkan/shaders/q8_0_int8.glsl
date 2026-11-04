// A Q8_0 block (17 u16s): an f16 scale, then 32 int8 values.

const uint BLOCK_U16 = 17;
const float BIAS = 0.0;

// the scale, and values 0..15 (lo) and 16..31 (hi), four a word
float unpack_block(uint base, out uvec4 lo, out uvec4 hi) {
    uint v[8];
    [[unroll]] for (uint i = 0; i < 8; i++) {
        v[i] = uint(w[base + 1 + 2 * i]) | (uint(w[base + 2 + 2 * i]) << 16);
    }
    lo = uvec4(v[0], v[1], v[2], v[3]);
    hi = uvec4(v[4], v[5], v[6], v[7]);
    return unpackHalf2x16(uint(w[base])).x;
}
