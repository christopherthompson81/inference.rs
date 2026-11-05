// A Q8_0 block (17 u16s): an f16 scale, then 32 int8 values.
const uint BLOCK_U16 = 17;

struct BlockW {
    float16_t d;
    uint16_t qs[16];
};

layout(buffer_reference, std430, buffer_reference_align = 2) buffer DecodeW {
    BlockW block;
};

float16_t decode_w(const in DecodeW bl, const in uint block_coords[2], const in uint coord_in_block[2]) {
    uint idx = coord_in_block[1];
    int q = int(uint(bl.block.qs[idx >> 1]) << (24 - 8 * (idx & 1))) >> 24;
    return float16_t(q) * bl.block.d;
}

// four values from coord_in_block[1] (a multiple of 4) on
f16vec4 decode_w_v(const in DecodeW bl, const in uint block_coords[2], const in uint coord_in_block[2]) {
    uint i = coord_in_block[1] >> 1;
    uint q = uint(bl.block.qs[i]) | (uint(bl.block.qs[i + 1]) << 16);
    return f16vec4(unpack8(int(q))) * bl.block.d;
}
