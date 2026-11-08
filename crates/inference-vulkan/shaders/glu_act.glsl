// A GLU's halves, act(x) * y, under `MapOp`'s numbers for them, for map_f32 and the matvecs that end in a GLU.

const uint SILU_MUL = 31;
const uint GELU_MUL = 32;
const uint RELU_MUL = 33;
const uint GELU_ERF_MUL = 34;
const uint SIGMOID_MUL = 35;

const float GELU_C = 0.7978845608028654;
const float GELU_K = 0.044715;
const float INV_SQRT2 = 0.7071067811865476;

// GLSL has no erf: Abramowitz and Stegun 7.1.26's erfc(z) for z >= 0 (erf within 1.5e-7), taken directly below zero
// where 1 + erf(x) would cancel
float gelu_erf(float x) {
    float z = abs(x) * INV_SQRT2;
    float t = 1.0 / (1.0 + 0.3275911 * z);
    float poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    float erfc = poly * exp(-z * z);
    return x >= 0.0 ? x * (1.0 - 0.5 * erfc) : 0.5 * x * erfc;
}

float gelu_tanh(float x) {
    return 0.5 * x * (1.0 + tanh(GELU_C * (x + GELU_K * x * x * x)));
}

float glu(uint op, float x, float y) {
    switch (op) {
        case SILU_MUL: return x / (1.0 + exp(-x)) * y;
        case GELU_MUL: return gelu_tanh(x) * y;
        case RELU_MUL: return max(x, 0.0) * y;
        case GELU_ERF_MUL: return gelu_erf(x) * y;
        case SIGMOID_MUL: return y / (1.0 + exp(-x));
    }
    return x * y;
}
