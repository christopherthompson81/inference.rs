// bf16 is an f32's top half, carried in a uint here so no 16-bit locals are needed: widening is exact, narrowing
// rounds to nearest even, and a NaN stays a quiet NaN.
float bf16_bits_to_f32(uint h) {
    return uintBitsToFloat(h << 16);
}

uint f32_to_bf16_bits(float v) {
    uint u = floatBitsToUint(v);
    return isnan(v) ? (u >> 16) | 0x40u : (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
