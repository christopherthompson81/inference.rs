// Q4_K and Q5_K's 6-bit sub-block scales and mins, packed in 12 bytes from u16 index `at` (ggml's
// get_scale_min_k4).

uint scale_byte(uint at, uint i) {
    return (uint(w[at + i / 2]) >> (8 * (i % 2))) & 0xffu;
}

// sub-block j's (scale, min)
uvec2 scale_min(uint at, uint j) {
    if (j < 4) {
        return uvec2(scale_byte(at, j) & 63u, scale_byte(at, j + 4) & 63u);
    }
    uint s = scale_byte(at, j + 4);
    return uvec2((s & 0xfu) | ((scale_byte(at, j - 4) >> 6) << 4), (s >> 4) | ((scale_byte(at, j) >> 6) << 4));
}
