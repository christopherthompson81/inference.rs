// Block `b` (of a row's `blocks` 32-value blocks) of weight row `row` in the matrix at u16 `matrix`, through the
// included type's stored blocks of SUPER_VALUES values; see block_dot for the result.

vec4 unpack(uint matrix, uint row, uint b, uint blocks, out uvec4 lo, out uvec4 hi) {
    const uint SUBS = SUPER_VALUES / 32;
    uint base = matrix + (row * (blocks / SUBS) + b / SUBS) * SUPER_U16;
    return unpack_sub(base, b % SUBS, lo, hi);
}

// What a block's dot keeps of its coefficients and of its activation block's (d, d * sum q), as little as the type
// allows (which keeps the prompt matmuls' shared tiles small): s alone where the offset is OFFSET_RATIO * s, and d
// alone with NO_OFFSET; (s, o) where the offset is free; all four where the halves differ (SPLIT_HALVES), whose
// activation half-sums come from the int8 words.
#if defined(NO_OFFSET)
#define XSCALE float
XSCALE pack_xs(vec2 s) {
    return s.x;
}
#else
#define XSCALE vec2
XSCALE pack_xs(vec2 s) {
    return s;
}
#endif
#if SPLIT_HALVES
#define COEF vec4
COEF pack_coef(vec4 c) {
    return c;
}
#elif defined(NO_OFFSET) || defined(OFFSET_RATIO)
#define COEF float
COEF pack_coef(vec4 c) {
    return c.x;
}
#else
#define COEF vec2
COEF pack_coef(vec4 c) {
    return c.xz;
}
#endif

float block_dot(uvec4 lo, uvec4 hi, COEF c, uvec4 x_lo, uvec4 x_hi, XSCALE xs) {
#if SPLIT_HALVES
    const uvec4 ones = uvec4(0x01010101u);
    vec4 q = vec4(dot4(lo, x_lo), dot4(hi, x_hi), dot4(x_lo, ones), dot4(x_hi, ones));
    return xs.x * dot(c, q);
#else
    ivec4 a = ivec4(lo), e = ivec4(hi), p = ivec4(x_lo), o = ivec4(x_hi);
    int q = dotPacked4x8EXT(a.x, p.x) + dotPacked4x8EXT(a.y, p.y) + dotPacked4x8EXT(a.z, p.z)
        + dotPacked4x8EXT(a.w, p.w) + dotPacked4x8EXT(e.x, o.x) + dotPacked4x8EXT(e.y, o.y)
        + dotPacked4x8EXT(e.z, o.z) + dotPacked4x8EXT(e.w, o.w);
#if defined(NO_OFFSET)
    return c * (xs * float(q));
#elif defined(OFFSET_RATIO)
    return c * (xs.x * float(q) + OFFSET_RATIO * xs.y);
#else
    return c.x * (xs.x * float(q)) + c.y * xs.y;
#endif
#endif
}
