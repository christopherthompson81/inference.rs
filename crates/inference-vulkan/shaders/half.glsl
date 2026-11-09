// A 16-bit float carried as its bits in a uint, bf16 or (HALF_F16, set where the pipeline is made) f16, so one build
// serves both: widening is exact, narrowing rounds to nearest even.
#include "bf16.glsl"

layout(constant_id = 0) const bool HALF_F16 = false;

float half_to_f32(uint h) {
    return HALF_F16 ? unpackHalf2x16(h).x : bf16_bits_to_f32(h);
}

uint f32_to_half(float v) {
    return HALF_F16 ? packHalf2x16(vec2(v, 0.0)) : f32_to_bf16_bits(v);
}
