// Bit helpers the 4- and 5-bit block types share.

// four bits to the low bit of each of four bytes
uint spread(uint bits) {
    return ((bits & 0xfu) * 0x00204081u) & 0x01010101u;
}

// sixteen bits, one a value, to the low bit of each of sixteen bytes
uvec4 fifth_bits(uint bits) {
    return uvec4(spread(bits), spread(bits >> 4), spread(bits >> 8), spread(bits >> 12));
}
