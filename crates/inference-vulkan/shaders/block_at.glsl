// Block `b` (of a row's `blocks` 32-value blocks) of weight row `row` in the matrix at u16 `matrix`, through the
// included type's stored blocks of SUPER_VALUES values; see int8_dot.glsl's block_dot for the result.

vec4 unpack(uint matrix, uint row, uint b, uint blocks, out uvec4 lo, out uvec4 hi) {
    const uint SUBS = SUPER_VALUES / 32;
    uint base = matrix + (row * (blocks / SUBS) + b / SUBS) * SUPER_U16;
    return unpack_sub(base, b % SUBS, lo, hi);
}
