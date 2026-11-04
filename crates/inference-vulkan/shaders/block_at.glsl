// Block `b` (of a row's `blocks` 32-value blocks) of weight row `row` in the matrix at u16 `matrix`, through the
// included type's stored blocks of SUPER_VALUES values; see block_dot for the result.

vec4 unpack(uint matrix, uint row, uint b, uint blocks, out uvec4 lo, out uvec4 hi) {
    const uint SUBS = SUPER_VALUES / 32;
    uint base = matrix + (row * (blocks / SUBS) + b / SUBS) * SUPER_U16;
    return unpack_sub(base, b % SUBS, lo, hi);
}

// What a block's dot keeps of its coefficients and its activation block's scales, as little as the type allows (which
// keeps the prompt matmuls' shared tiles small): all four of each where the halves differ (SPLIT_HALVES); (s, o) and
// (d, d * sum q) where the offset is free; s alone where it is OFFSET_RATIO * s; and d alone with NO_OFFSET.
#if SPLIT_HALVES
#define COEF vec4
#define XSCALE vec4
COEF pack_coef(vec4 c) {
    return c;
}
XSCALE pack_xs(vec4 s) {
    return s;
}
float block_dot(uvec4 lo, uvec4 hi, COEF c, uvec4 x_lo, uvec4 x_hi, XSCALE xs) {
    return xs.x * (c.x * float(dot4(lo, x_lo)) + c.y * float(dot4(hi, x_hi))) + dot(c.zw, xs.yz);
}
#else
#if defined(NO_OFFSET)
#define XSCALE float
XSCALE pack_xs(vec4 s) {
    return s.x;
}
#else
#define XSCALE vec2
XSCALE pack_xs(vec4 s) {
    return vec2(s.x, s.y + s.z);
}
#endif
#if defined(NO_OFFSET) || defined(OFFSET_RATIO)
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
}
#endif
