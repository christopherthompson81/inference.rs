// A Q5_0 block (11 u16s): an f16 scale, 32 fifth bits (bit j value j's), then 16 bytes of low nibbles as Q4_0's;
// a value is its 5 bits less 16.
const uint BLOCK_U16 = 11;

struct BlockW {
    float16_t d;
    uint16_t qh[2];
    uint16_t qs[8];
};

layout(buffer_reference, std430, buffer_reference_align = 2) buffer DecodeW {
    BlockW block;
};

uint high_bits(const in DecodeW bl) {
    return uint(bl.block.qh[0]) | (uint(bl.block.qh[1]) << 16);
}

float16_t decode_w(const in DecodeW bl, const in uint block_coords[2], const in uint coord_in_block[2]) {
    uint idx = coord_in_block[1];
    uint q = (uint(bl.block.qs[(idx & 0xE) >> 1]) >> (8 * (idx & 1) + ((idx & 0x10) >> 2))) & 0xF;
    q |= ((high_bits(bl) >> idx) & 1) << 4;
    return (float16_t(q) - float16_t(16)) * bl.block.d;
}

// four values from coord_in_block[1] (a multiple of 4) on
f16vec4 decode_w_v(const in DecodeW bl, const in uint block_coords[2], const in uint coord_in_block[2]) {
    uint idx = coord_in_block[1];
    uint i = (idx & 0xE) >> 1;
    uint q = (uint(bl.block.qs[i]) | (uint(bl.block.qs[i + 1]) << 16)) >> ((idx & 0x10) >> 2);
    uint h = high_bits(bl) >> idx;
    // each value's fifth bit to bit 4 of its byte
    uint hb = ((h & 1) << 4) | ((h & 2) << 11) | ((h & 4) << 18) | ((h & 8) << 25);
    return (f16vec4(unpack8((q & 0x0F0F0F0Fu) | hb)) - float16_t(16)) * bl.block.d;
}
