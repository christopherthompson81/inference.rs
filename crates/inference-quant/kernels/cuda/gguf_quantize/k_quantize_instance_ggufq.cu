// Q6_K (ik_llama.cpp's `quantize_row_q6_K` with an imatrix, as its gcc build rounds it) and Q8_0 on the GPU, byte for
// byte the CPU ports in k_quants.rs and utils.rs (`make_qx_quants`, `quantize_block32`): every product, sum and fused
// multiply-add is an explicit rounding intrinsic. Each writes the blocks and their decoded values (as the CPU's
// `to_float` computes them), which GPTQ's feedback carries on. Kernels: a thread per 16-value Q6_K sub-block (its
// scale search and level refinement), a thread per 256-value super-block (the f16 and 8-bit scales, the levels
// against them, the packing); a thread per Q8_0 block.
#include "iq4k_common.cuh"

#define Q6K_SUB 16
#define Q6K_NMAX 32
#define Q6K_BYTES (QK_K / 2 + QK_K / 4 + QK_K / 16 + 2)
#define Q6K_SCALE_RANGE 128.0f
#define Q6K_SCALE_MAX 127
#define QX_TRIES 9
#define GROUP_MAX_EPS 1e-15f
#define Q8_0_BLOCK 32
#define Q8_0_BYTES (2 + Q8_0_BLOCK)
#define Q8_0_MAX 127.0f
// the NaN x86's FMA returns for inf * 0 (the "indefinite"), whose bits the scale rounding reads
#define X86_NAN_BITS 0xffc00000u

// ggml's nearest_int(a * b), the product fused into the rounding add as ik_llama.cpp's build fuses it
static __device__ __forceinline__ int nearest_int_fma(float a, float b) {
    const float v = __fmaf_rn(a, b, ROUNDING_MAGIC);
    const unsigned bits = isnan(v) ? X86_NAN_BITS : __float_as_uint(v);
    return (int)(bits & 0x007fffffu) - 0x00400000;
}

static __device__ __forceinline__ int q6k_level(float iscale, float v) {
    const int l = nearest_int_fma(iscale, v);
    return l < -Q6K_NMAX ? -Q6K_NMAX : (l > Q6K_NMAX - 1 ? Q6K_NMAX - 1 : l);
}

// a scale's weighted sums of l * x and l^2: fused as the first pass's are, or each product rounded as the tries' are
template <bool FUSED>
static __device__ void q6k_sums(const float* x, const float* w, float iscale, float* sumlx, float* suml2) {
    float sx = 0.0f, s2 = 0.0f;
    for (int i = 0; i < Q6K_SUB; ++i) {
        const float l = (float)q6k_level(iscale, x[i]);
        if (FUSED) {
            sx = __fmaf_rn(__fmul_rn(w[i], x[i]), l, sx);
            s2 = __fmaf_rn(__fmul_rn(w[i], l), l, s2);
        } else {
            sx = __fadd_rn(__fmul_rn(__fmul_rn(w[i], x[i]), l), sx);
            s2 = __fadd_rn(__fmul_rn(__fmul_rn(w[i], l), l), s2);
        }
    }
    *sumlx = sx;
    *suml2 = s2;
}

// ik_llama.cpp's make_qx_quants(16, 32, x, L, 1, qw): the max scale and its nudges from both ends, then single levels
// moved a step along the largest weighted gradient while the fit improves
static __device__ float q6k_sub_scale(const float* x, const float* w, int8_t* ls) {
    float max = 0.0f, amax = 0.0f;
    for (int i = 0; i < Q6K_SUB; ++i) {
        if (fabsf(x[i]) > amax) {
            amax = fabsf(x[i]);
            max = x[i];
        }
    }
    if (amax < GROUP_MAX_EPS) {
        for (int i = 0; i < Q6K_SUB; ++i) ls[i] = 0;
        return 0.0f;
    }
    float iscale = __fdiv_rn(-(float)Q6K_NMAX, max);
    float sumlx, suml2;
    q6k_sums<true>(x, w, iscale, &sumlx, &suml2);
    for (int i = 0; i < Q6K_SUB; ++i) ls[i] = (int8_t)(Q6K_NMAX + q6k_level(iscale, x[i]));
    float scale = suml2 != 0.0f ? __fdiv_rn(sumlx, suml2) : 0.0f;
    float best = __fmul_rn(scale, sumlx);
    float best_sumlx = sumlx, best_suml2 = suml2;
    for (int is = -QX_TRIES; is <= QX_TRIES; ++is) {
        const float tenths = (float)is;
        const float tries[2] = {__fdiv_rn(-__fmaf_rn(0.1f, tenths, (float)Q6K_NMAX), max),
                                __fdiv_rn(__fmaf_rn(0.1f, tenths, (float)(Q6K_NMAX - 1)), max)};
        for (int c = 0; c < 2; ++c) {
            iscale = tries[c];
            q6k_sums<false>(x, w, iscale, &sumlx, &suml2);
            if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
                for (int i = 0; i < Q6K_SUB; ++i) ls[i] = (int8_t)(Q6K_NMAX + q6k_level(iscale, x[i]));
                scale = __fdiv_rn(sumlx, suml2);
                best = __fmul_rn(scale, sumlx);
                best_sumlx = sumlx;
                best_suml2 = suml2;
            }
        }
    }
    sumlx = best_sumlx;
    suml2 = best_suml2;
    for (int step = 0; step < Q6K_SUB * (2 * Q6K_NMAX - 1); ++step) {
        float abs_gmax = 0.0f, gmax = 0.0f;
        int best_j = -1;
        for (int j = 0; j < Q6K_SUB; ++j) {
            const int l = ls[j] - Q6K_NMAX;
            const float g = __fmul_rn(__fmul_rn(scale, w[j]), __fmaf_rn(-scale, (float)l, x[j]));
            const bool movable = (g > 0.0f && l < Q6K_NMAX - 1) || (g < 0.0f && l > -Q6K_NMAX);
            if (movable && fabsf(g) > abs_gmax) {
                abs_gmax = fabsf(g);
                gmax = g;
                best_j = j;
            }
        }
        if (best_j < 0) break;
        const float wj = w[best_j];
        int l = ls[best_j] - Q6K_NMAX;
        float new_sumlx, new_suml2;
        if (gmax > 0.0f) {
            ++l;
            new_sumlx = __fadd_rn(sumlx, __fmul_rn(wj, x[best_j]));
            new_suml2 = __fmaf_rn(wj, (float)(2 * l - 1), suml2);
        } else {
            --l;
            new_sumlx = __fsub_rn(sumlx, __fmul_rn(wj, x[best_j]));
            new_suml2 = __fmaf_rn(-wj, (float)(2 * l + 1), suml2);
        }
        if (new_suml2 > 0.0f && __fmul_rn(new_sumlx, new_sumlx) > __fmul_rn(best, new_suml2)) {
            sumlx = new_sumlx;
            suml2 = new_suml2;
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            ls[best_j] = (int8_t)(l + Q6K_NMAX);
        } else {
            break;
        }
    }
    return scale;
}

static __global__ void q6k_search(const float* x, const float* importance, int64_t rows_per_importance, int64_t subs,
                                  int cols, int8_t* ls, float* scales) {
    const int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= subs) return;
    const int per_row = cols / Q6K_SUB;
    const int64_t row = s / per_row;
    const int col = (int)(s % per_row) * Q6K_SUB;
    float w[Q6K_SUB];
    const float* xs = x + s * Q6K_SUB;
    // without an importance, make_qx_quants weighs each value by its square
    const float* qw = importance ? importance + (row / rows_per_importance) * cols + col : nullptr;
    for (int i = 0; i < Q6K_SUB; ++i) w[i] = qw ? qw[i] : __fmul_rn(xs[i], xs[i]);
    scales[s] = q6k_sub_scale(xs, w, ls + s * Q6K_SUB);
}

static __device__ __forceinline__ float f16_bits_to_float(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

static __global__ void q6k_super(const float* x, int64_t supers, const float* scales, int8_t* ls, uint8_t* out,
                                 float* decoded) {
    const int64_t sb = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= supers) return;
    const float* xs = x + sb * QK_K;
    const float* sc = scales + sb * (QK_K / Q6K_SUB);
    int8_t* l = ls + sb * QK_K;
    uint8_t* o = out + sb * Q6K_BYTES;
    float max_scale = 0.0f, max_abs_scale = 0.0f;
    for (int ib = 0; ib < QK_K / Q6K_SUB; ++ib) {
        if (fabsf(sc[ib]) > max_abs_scale) {
            max_abs_scale = fabsf(sc[ib]);
            max_scale = sc[ib];
        }
    }
    const float iscale = __fdiv_rn(-Q6K_SCALE_RANGE, max_scale);
    const uint16_t dh = to_f16_bits(__fdiv_rn(1.0f, iscale));
    const float d = f16_bits_to_float(dh);
    int8_t q_scales[QK_K / Q6K_SUB];
    for (int ib = 0; ib < QK_K / Q6K_SUB; ++ib) {
        const int q = nearest_int_fma(iscale, sc[ib]);
        q_scales[ib] = (int8_t)(q < Q6K_SCALE_MAX ? q : Q6K_SCALE_MAX);
    }
    for (int ib = 0; ib < QK_K / Q6K_SUB; ++ib) {
        const float dl = __fmul_rn(d, (float)q_scales[ib]);
        // a scale rounded to 0 keeps the search's levels
        if (dl == 0.0f) continue;
        for (int i = 0; i < Q6K_SUB; ++i) {
            int q = nearest_int(__fdiv_rn(xs[ib * Q6K_SUB + i], dl));
            q = q < -Q6K_NMAX ? -Q6K_NMAX : (q > Q6K_NMAX - 1 ? Q6K_NMAX - 1 : q);
            l[ib * Q6K_SUB + i] = (int8_t)(q + Q6K_NMAX);
        }
    }
    for (int h = 0; h < 2; ++h) {
        const int j = 128 * h;
        uint8_t* ql = o + 64 * h;
        uint8_t* qh = o + QK_K / 2 + 32 * h;
        for (int i = 0; i < 32; ++i) {
            const int q1 = l[j + i], q2 = l[j + i + 32], q3 = l[j + i + 64], q4 = l[j + i + 96];
            ql[i] = (uint8_t)((q1 & 0xF) | ((q3 & 0xF) << 4));
            ql[i + 32] = (uint8_t)((q2 & 0xF) | ((q4 & 0xF) << 4));
            qh[i] = (uint8_t)((q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6));
        }
    }
    for (int ib = 0; ib < QK_K / Q6K_SUB; ++ib) o[QK_K / 2 + QK_K / 4 + ib] = (uint8_t)q_scales[ib];
    o[Q6K_BYTES - 2] = (uint8_t)dh;
    o[Q6K_BYTES - 1] = (uint8_t)(dh >> 8);
    for (int k = 0; k < QK_K; ++k)
        decoded[sb * QK_K + k] = __fmul_rn(__fmul_rn(d, (float)q_scales[k / Q6K_SUB]), (float)(l[k] - Q6K_NMAX));
}

// `rows` rows of `cols` f32 on the device (`cols` a multiple of 256) as Q6_K in `out`, each `rows_per_importance` rows
// weighed by the next `cols` of `importance` (or null), their decoded values in `decoded`. Scratch: `ls` a byte a value, `scales`
// a float per 16. Returns 0, or a CUDA error code.
extern "C" int launch_q6_k_encode(const float* x, const float* importance, int64_t rows, int64_t rows_per_importance,
                                  int cols, int8_t* ls, float* scales, uint8_t* out, float* decoded,
                                  cudaStream_t stream) {
    const int64_t subs = rows * (cols / Q6K_SUB);
    q6k_search<<<grid(subs), THREADS, 0, stream>>>(x, importance, rows_per_importance, subs, cols, ls, scales);
    q6k_super<<<grid(rows * (cols / QK_K)), THREADS, 0, stream>>>(x, rows * (cols / QK_K), scales, ls, out, decoded);
    return (int)cudaGetLastError();
}

// one Q8_0 block a thread (`quantize_block32`): d = amax / 127, each value times 1 / d rounded half away from zero
static __global__ void q8_0_block(const float* x, int64_t blocks, uint8_t* out, float* decoded) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const float* xb = x + b * Q8_0_BLOCK;
    float amax = 0.0f;
    for (int j = 0; j < Q8_0_BLOCK; ++j) amax = fmaxf(amax, fabsf(xb[j]));
    const float d = __fdiv_rn(amax, Q8_0_MAX);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    const uint16_t dh = to_f16_bits(d);
    const float dd = f16_bits_to_float(dh);
    uint8_t* o = out + b * Q8_0_BYTES;
    o[0] = (uint8_t)dh;
    o[1] = (uint8_t)(dh >> 8);
    for (int j = 0; j < Q8_0_BLOCK; ++j) {
        float r = roundf(__fmul_rn(xb[j], id));
        r = r < -128.0f ? -128.0f : (r > 127.0f ? 127.0f : r);
        // through the integer level, as the CPU decodes it (roundf keeps -0's sign)
        const int q = (int)r;
        o[2 + j] = (uint8_t)(int8_t)q;
        decoded[b * Q8_0_BLOCK + j] = __fmul_rn((float)q, dd);
    }
}

// as launch_q6_k_encode, for Q8_0 (`cols` a multiple of 32; no importance, as ggml's Q8_0 takes none)
extern "C" int launch_q8_0_encode(const float* x, int64_t rows, int cols, uint8_t* out, float* decoded,
                                  cudaStream_t stream) {
    const int64_t blocks = rows * (cols / Q8_0_BLOCK);
    q8_0_block<<<grid(blocks), THREADS, 0, stream>>>(x, blocks, out, decoded);
    return (int)cudaGetLastError();
}
