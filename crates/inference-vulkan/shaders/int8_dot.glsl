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

// a weight block (its int8 halves and (s_lo, s_hi, o_lo, o_hi)) dotted with an activation block (its int8 halves
// and quantize_q8_1's (d, d * sum_lo, d * sum_hi, 0))
float block_dot(uvec4 lo, uvec4 hi, vec4 coef, uvec4 x_lo, uvec4 x_hi, vec4 xs) {
    return xs.x * (coef.x * float(dot4(lo, x_lo)) + coef.y * float(dot4(hi, x_hi))) + dot(coef.zw, xs.yz);
}
