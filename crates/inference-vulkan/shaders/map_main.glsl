// out = op(a[, b]) elementwise over a shape of up to MAX_DIMS dims, every operand strided (a stride of 0 broadcasts).
// A grid-stride loop over the output's elements, which are unravelled row-major over `dims`. With BF16 every operand
// is bf16, computed in f32.
#ifdef BF16
#extension GL_EXT_shader_16bit_storage : require
#include "bf16.glsl"
#define ELEM uint16_t
#define LOAD(v) bf16_bits_to_f32(uint(v))
#define STORE(v) uint16_t(f32_to_bf16_bits(v))
#else
#define ELEM float
#define LOAD(v) (v)
#define STORE(v) (v)
#endif

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer A { ELEM a[]; };
layout(std430, binding = 1) readonly buffer B { ELEM b[]; };
layout(std430, binding = 2) buffer O { ELEM o[]; };

const uint MAX_DIMS = 6;

layout(push_constant) uniform PushConstants {
    uint n;
    uint rank;
    uint op;
    float p0;
    float p1;
    uint a_off;
    uint b_off;
    uint o_off;
    uint dims[MAX_DIMS];
    uint a_strides[MAX_DIMS];
    uint b_strides[MAX_DIMS];
    uint o_strides[MAX_DIMS];
} pc;

// op codes, as `MapOp` numbers them
const uint COPY = 0;
const uint ADD = 1;
const uint SUB = 2;
const uint MUL = 3;
const uint DIV = 4;
const uint MAXIMUM = 5;
const uint MINIMUM = 6;
const uint NEG = 7;
const uint EXP = 8;
const uint LOG = 9;
const uint SQR = 10;
const uint SQRT = 11;
const uint RECIP = 12;
const uint SILU = 13;
const uint RELU = 14;
const uint GELU = 15;
const uint TANH = 16;
const uint ABS = 17;
const uint AFFINE = 18;
const uint POWF = 19;
const uint SIN = 20;
const uint COS = 21;
const uint FILL = 22;
const uint FLOOR = 23;
const uint CEIL = 24;
const uint ROUND = 25;
const uint SIGN = 26;
const uint ELU = 27;
const uint SIGMOID = 28;
const uint RELU2 = 29;
const uint GELU_ERF = 30;
#include "glu_act.glsl"

float apply(float x, float y) {
    switch (pc.op) {
        case COPY: return x;
        case ADD: return x + y;
        case SUB: return x - y;
        case MUL: return x * y;
        case DIV: return x / y;
        case MAXIMUM: return max(x, y);
        case MINIMUM: return min(x, y);
        case NEG: return -x;
        case EXP: return exp(x);
        case LOG: return log(x);
        case SQR: return x * x;
        case SQRT: return sqrt(x);
        case RECIP: return 1.0 / x;
        case SILU: return x / (1.0 + exp(-x));
        case RELU: return max(x, 0.0);
        case GELU: return gelu_tanh(x);
        case TANH: return tanh(x);
        case ABS: return abs(x);
        case AFFINE: return x * pc.p0 + pc.p1;
        case POWF: return pow(x, pc.p0);
        case SIN: return sin(x);
        case COS: return cos(x);
        case FILL: return pc.p0;
        case FLOOR: return floor(x);
        case CEIL: return ceil(x);
        // half away from zero, as Rust's f32::round (GLSL's round may go either way at .5)
        case ROUND: return sign(x) * floor(abs(x) + 0.5);
        case SIGN: return sign(x);
        case ELU: return x >= 0.0 ? x : pc.p0 * (exp(x) - 1.0);
        case SIGMOID: return 1.0 / (1.0 + exp(-x));
        case RELU2: { float r = max(x, 0.0); return r * r; }
        case GELU_ERF: return gelu_erf(x);
        case SILU_MUL:
        case GELU_MUL:
        case RELU_MUL:
        case GELU_ERF_MUL:
        case SIGMOID_MUL: return glu(pc.op, x, y);
    }
    return x;
}

void main() {
    uint total = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    for (uint i = gl_GlobalInvocationID.x; i < pc.n; i += total) {
        uint rest = i;
        uint ia = pc.a_off;
        uint ib = pc.b_off;
        uint io = pc.o_off;
        for (uint d = pc.rank; d > 0; d--) {
            uint c = rest % pc.dims[d - 1];
            rest /= pc.dims[d - 1];
            ia += c * pc.a_strides[d - 1];
            ib += c * pc.b_strides[d - 1];
            io += c * pc.o_strides[d - 1];
        }
        float x = pc.op == FILL ? 0.0 : LOAD(a[ia]);
        float y = (pc.op >= ADD && pc.op <= MINIMUM) || pc.op >= SILU_MUL ? LOAD(b[ib]) : 0.0;
        o[io] = STORE(apply(x, y));
    }
}
