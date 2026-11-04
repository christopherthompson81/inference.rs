// A Q5_0 block's 32 values, scaled: an f16 scale, the values' fifth bits, then their low nibbles (value j in byte
// j % 16's low or high half); a value is its 5 bits less 16. The including shader declares `w` (u16s) first.

const uint BLOCK_U16 = 11;
const float OFFSET = 16.0;

void decode(uint base, out float v[QK]) {
    float d = unpackHalf2x16(uint(w[base])).x;
    uint qh = uint(w[base + 1]) | (uint(w[base + 2]) << 16);
    [[unroll]] for (uint k = 0; k < QK / 4; k++) {
        // bytes 2k and 2k + 1 of the nibbles: values 2k, 2k + 1 below, 2k + 16, 2k + 17 above
        uint q = uint(w[base + 3 + k]);
        uint j = 2 * k;
        v[j] = d * (float((q & 0xfu) | (((qh >> j) & 1u) << 4)) - OFFSET);
        v[j + 1] = d * (float(((q >> 8) & 0xfu) | (((qh >> (j + 1)) & 1u) << 4)) - OFFSET);
        v[j + 16] = d * (float(((q >> 4) & 0xfu) | (((qh >> (j + 16)) & 1u) << 4)) - OFFSET);
        v[j + 17] = d * (float(((q >> 12) & 0xfu) | (((qh >> (j + 17)) & 1u) << 4)) - OFFSET);
    }
}
