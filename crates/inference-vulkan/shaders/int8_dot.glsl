// Reads of the weight buffer `w` (u16s) and the int8 dot of one 32-value block, for the matvecs that include it after
// declaring `w`.

// the 32 bits at u16 index `at`
uint word(uint at) {
    return uint(w[at]) | (uint(w[at + 1]) << 16);
}

float half_at(uint at) {
    return unpackHalf2x16(uint(w[at])).x;
}

int dot4(uvec4 a, uvec4 b) {
    ivec4 x = ivec4(a), y = ivec4(b);
    return dotPacked4x8EXT(x.x, y.x) + dotPacked4x8EXT(x.y, y.y) + dotPacked4x8EXT(x.z, y.z)
        + dotPacked4x8EXT(x.w, y.w);
}
