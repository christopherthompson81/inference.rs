// Reads of the weight buffer `w` (u16s) and the int8 dot of one 32-value block, for the matvecs that include it after
// declaring `w`. A kernel reading two weights defines W16 to pick between them.

#ifndef W16
#define W16(at) w[at]
#endif

// the 32 bits at u16 index `at`
uint word(uint at) {
    return uint(W16(at)) | (uint(W16(at + 1)) << 16);
}

float half_at(uint at) {
    return unpackHalf2x16(uint(W16(at))).x;
}

int dot4(uvec4 a, uvec4 b) {
    ivec4 x = ivec4(a), y = ivec4(b);
    return dotPacked4x8EXT(x.x, y.x) + dotPacked4x8EXT(x.y, y.y) + dotPacked4x8EXT(x.z, y.z)
        + dotPacked4x8EXT(x.w, y.w);
}
