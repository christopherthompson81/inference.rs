// A Q3_K block (55 u16s, 256 values; ggml's dequantize_row_q3_K): 32 bytes of high bits (hmask), 64 bytes of low bit
// pairs, 12 bytes of 6-bit scales (one per 16 values), then an f16 d. Sub-block s (32 values) of half n = s / 4,
// part j = s % 4 takes its low pairs from bits 2j of bytes 32n..32n + 31 and its high bit from bit s of hmask's 32
// bytes; a value is d (scale - 32) (q - 4), q in 0..7, its halves' scales 2s and 2s + 1.

const uint SUPER_VALUES = 256;
#define SPLIT_HALVES 1
const uint SUPER_U16 = 55;

uint q3_byte(uint base, uint i) {
    return (uint(w[base + i / 2]) >> (8 * (i % 2))) & 0xffu;
}

// scale k of 16: its low four bits from bytes 0..7 (low nibble, then high), its top two from bytes 8..11
int q3_scale(uint base, uint k) {
    uint low = k < 8 ? q3_byte(base, 96 + k) & 0xfu : q3_byte(base, 96 + k - 8) >> 4;
    uint high = (q3_byte(base, 104 + k % 4) >> (2 * (k / 4))) & 3u;
    return int(low | (high << 4)) - 32;
}

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint n = sub / 4, j = sub % 4;
    uint at = base + 16 + 16 * n;
    uvec4 a = uvec4(word(at), word(at + 2), word(at + 4), word(at + 6));
    uvec4 b = uvec4(word(at + 8), word(at + 10), word(at + 12), word(at + 14));
    uvec4 ha = uvec4(word(base), word(base + 2), word(base + 4), word(base + 6));
    uvec4 hb = uvec4(word(base + 8), word(base + 10), word(base + 12), word(base + 14));
    lo = ((a >> (2 * j)) & 0x03030303u) | (((ha >> sub) & 0x01010101u) << 2);
    hi = ((b >> (2 * j)) & 0x03030303u) | (((hb >> sub) & 0x01010101u) << 2);
    float d = half_at(base + 54);
    float s_lo = d * float(q3_scale(base, 2 * sub)), s_hi = d * float(q3_scale(base, 2 * sub + 1));
    return vec4(s_lo, s_hi, -4.0 * s_lo, -4.0 * s_hi);
}
