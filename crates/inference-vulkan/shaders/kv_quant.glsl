// The quantized paged KV cache's rows (inference-paged-attn's kv_quant, CUDA's quant_cache_kernel.cu): a head's values
// rotated by 32-wide Hadamard transforms, every 32-value group scaled by its absmax and quantized to `bits` (2..8) on
// a midpoint grid, its 32 values a little-endian bitstring of `bits` words; after the head_size * bits / 8 data
// bytes, one f16 scale a group. Rows are whole words (head sizes a multiple of 64).

const uint QGROUP = 32;
// keeps an all-zero group's scale positive
const float QMIN_SCALE = 1e-10;

uint quant_row_words(uint head_size, uint bits) {
    return (head_size * bits / 8 + head_size / 16) / 4;
}

// value k of a group's bitstring, already loaded as `bits` words
uint quant_value(uint words[8], uint k, uint bits) {
    uint o = k * bits;
    uint i = o / 32;
    uint sh = o % 32;
    uint v = words[i] >> sh;
    if (sh + bits > 32) {
        v |= words[i + 1] << (32 - sh);
    }
    return v & ((1u << bits) - 1u);
}

float quant_dequantize(uint q, uint bits, float scale) {
    float mid = float(1u << (bits - 1));
    return (float(q) + 0.5 - mid) / mid * scale;
}

// the normalized 32-wide Hadamard transform of `v`, its own inverse
void hadamard32(inout float v[32]) {
    [[unroll]] for (uint s = 1; s < 32; s <<= 1) {
        [[unroll]] for (uint i = 0; i < 32; i++) {
            if ((i & s) == 0) {
                float a = v[i];
                float b = v[i | s];
                v[i] = a + b;
                v[i | s] = a - b;
            }
        }
    }
    float norm = inversesqrt(32.0);
    [[unroll]] for (uint i = 0; i < 32; i++) {
        v[i] *= norm;
    }
}

// NAME(row, d, bits, head_size): dimension d of the (still rotated) row at word `row` of BUF
#define QUANT_READER(NAME, BUF)                                                                                      \
    float NAME(uint row, uint d, uint bits, uint head_size) {                                                        \
        uint o = d * bits;                                                                                           \
        uint sh = o % 32;                                                                                            \
        uint w = BUF[row + o / 32] >> sh;                                                                            \
        if (sh + bits > 32) {                                                                                        \
            w |= BUF[row + o / 32 + 1] << (32 - sh);                                                                 \
        }                                                                                                            \
        uint g = d / QGROUP;                                                                                         \
        float scale = unpackHalf2x16(BUF[row + head_size * bits / 32 + g / 2])[g % 2];                               \
        return quant_dequantize(w & ((1u << bits) - 1u), bits, scale);                                               \
    }
