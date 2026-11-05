// A Q4_1 block (10 u16s): f16 scale and min, then 16 bytes, value j in byte j % 16's low (j < 16) or high half.
const uint BLOCK_U16 = 10;

struct BlockW {
    float16_t d;
    float16_t m;
    uint16_t qs[8];
};

layout(buffer_reference, std430, buffer_reference_align = 2) buffer DecodeW {
    BlockW block;
};

float16_t decode_w(const in DecodeW bl, const in uint block_coords[2], const in uint coord_in_block[2]) {
    uint idx = coord_in_block[1];
    uint q = (uint(bl.block.qs[(idx & 0xE) >> 1]) >> (8 * (idx & 1) + ((idx & 0x10) >> 2))) & 0xF;
    return float16_t(q) * bl.block.d + bl.block.m;
}

// four values from coord_in_block[1] (a multiple of 4) on
f16vec4 decode_w_v(const in DecodeW bl, const in uint block_coords[2], const in uint coord_in_block[2]) {
    uint idx = coord_in_block[1];
    uint i = (idx & 0xE) >> 1;
    uint q = (uint(bl.block.qs[i]) | (uint(bl.block.qs[i + 1]) << 16)) >> ((idx & 0x10) >> 2);
    return f16vec4(unpack8(q & 0x0F0F0F0Fu)) * bl.block.d + bl.block.m;
}
