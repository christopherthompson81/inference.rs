// A Q6_K block (105 u16s, 256 values): 128 bytes of low nibbles, 64 bytes of high bit pairs, 16 int8 scales (one per
// 16 values), then an f16 d. Sub-block s of half n = s / 4, part p = s % 4 takes its low 4 bits from bytes
// 64n + 32 (p % 2)..+31 (high nibbles when p >= 2) and its top 2 from bits 2p of bytes 128 + 32n..+31; a value is
// d sc (q - 32).

const uint SUPER_VALUES = 256;
#define SPLIT_HALVES 1
const uint SUPER_U16 = 105;

int scale_at(uint base, uint i) {
    return int(((uint(W16(base + 96 + i / 2)) >> (8 * (i % 2))) & 0xffu) << 24) >> 24;
}

vec4 unpack_sub(uint base, uint sub, out uvec4 lo, out uvec4 hi) {
    uint n = sub / 4, p = sub % 4;
    uint at = base + 32 * n + 16 * (p % 2);
    uint shift = 4 * (p / 2);
    uint qh_at = base + 64 + 16 * n;
    uvec4 a = uvec4(word(at), word(at + 2), word(at + 4), word(at + 6));
    uvec4 b = uvec4(word(at + 8), word(at + 10), word(at + 12), word(at + 14));
    uvec4 ha = uvec4(word(qh_at), word(qh_at + 2), word(qh_at + 4), word(qh_at + 6));
    uvec4 hb = uvec4(word(qh_at + 8), word(qh_at + 10), word(qh_at + 12), word(qh_at + 14));
    lo = ((a >> shift) & 0x0f0f0f0fu) | (((ha >> (2 * p)) & 0x03030303u) << 4);
    hi = ((b >> shift) & 0x0f0f0f0fu) | (((hb >> (2 * p)) & 0x03030303u) << 4);
    float d = half_at(base + 104);
    uint si = 8 * n + 2 * p;
    float s_lo = d * float(scale_at(base, si)), s_hi = d * float(scale_at(base, si + 1));
    return vec4(s_lo, s_hi, -32.0 * s_lo, -32.0 * s_hi);
}
