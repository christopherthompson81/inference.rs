// ik_llama.cpp's IQ4_KSS quantizer (`quantize_row_iq4_kss_impl`) on the GPU, byte for byte the CPU port in
// iq_quantize.rs: IQ4_KS's search with every four indices pruned to even parity, each word four indices scrambled
// into 15 bits beside one bit of the block's 8-bit scale. Every product, sum and fused multiply-add is an explicit
// rounding intrinsic. Kernels: each block's weights (unfused sigma2, as gcc builds this function); a thread per block
// searching its scale; the row scale; a thread per block for its words and levels; a thread per row refitting the row
// scale in the CPU's order (a fused chain across the row).
#include "iq4k_common.cuh"

#define WORDS (BLOCK / 4)
#define BLOCK_BYTES (QK_K / 2)
#define FUDGE 1.01f
#define PRUNE_STEPS 2

// word k packs x[2k], x[2k + 16], x[2k + 1], x[2k + 17]
static __device__ __forceinline__ int group_at(int k, int j) {
    return (j & 1) * (BLOCK / 2) + 2 * k + (j >> 1);
}

// ik's `prune_iq4ks`: four indices of odd parity get the cheapest move (up to two steps) of one of them that evens
// it; the current error measured on the unshifted codebook whichever is in use, as ik does
static __device__ uint16_t prune(uint16_t v, const int8_t* unshifted, const int8_t* values, const float* x,
                                 const float* w, float dl) {
    if (__popc(v) % 2 == 0) return v;
    uint8_t q4[4];
    float best = 3.402823466e38f;
    int jbest = -1;
    uint8_t bestq = 0;
    for (int j = 0; j < 4; ++j) {
        const uint8_t q = (v >> (4 * j)) & 0xf;
        q4[j] = q;
        const int pc = __popc(q);
        const float diff0 = __fmaf_rn(dl, (float)unshifted[q], -x[j]);
        const int lo = max((int)q - PRUNE_STEPS, 0), hi = min((int)q + PRUNE_STEPS, 15);
        for (int qq = lo; qq <= hi; ++qq) {
            if (qq == q || abs(__popc(qq) - pc) % 2 == 0) continue;
            const float diff1 = __fmaf_rn(dl, (float)values[qq], -x[j]);
            const float score = __fmul_rn(w[j], __fmaf_rn(diff1, diff1, -__fmul_rn(diff0, diff0)));
            if (score < best) {
                best = score;
                jbest = j;
                bestq = (uint8_t)qq;
            }
        }
    }
    q4[jbest] = bestq;
    return (uint16_t)(q4[0] | (q4[1] << 4) | (q4[2] << 8) | (q4[3] << 12));
}

// a thread per block: its best scale over both codebooks and signs, each try's indices pruned
static __global__ void iq4_kss_search(const float* x, const float* weights, int64_t blocks, float* scales) {
    __shared__ Tables t;
    load_tables(&t);
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const float* xb = x + b * BLOCK;
    const float* wb = weights + b * BLOCK;
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < BLOCK; ++j) {
        if (fabsf(xb[j]) > amax) {
            amax = fabsf(xb[j]);
            max = xb[j];
        }
    }
    if (amax < EPS) {
        scales[b] = 0.0f;
        return;
    }
    float best = 0.0f, d = __fdiv_rn(-max, (float)t.values[0]);
    for (int itry = -TRIES; itry <= TRIES; ++itry) {
        for (int book = 0; book < 2; ++book) {
            const int8_t* values = t.values + 16 * book;
            const float inverse = __fdiv_rn(__fadd_rn((float)itry, (float)values[0]), max);
            const float this_d = __fdiv_rn(1.0f, inverse);
            for (int negated = 0; negated < 2; ++negated) {
                float sumqx = 0.0f, sumq2 = 0.0f;
                for (int k = 0; k < WORDS; ++k) {
                    float xv[4], wv[4];
                    uint16_t v = 0;
                    for (int j = 0; j < 4; ++j) {
                        xv[j] = xb[group_at(k, j)];
                        wv[j] = wb[group_at(k, j)];
                        float al = __fmul_rn(inverse, xv[j]);
                        if (negated) al = -al;
                        v |= (uint16_t)(best_index(t.index, values, al) << (4 * j));
                    }
                    v = prune(v, t.values, values, xv, wv, this_d);
                    for (int j = 0; j < 4; ++j) {
                        const float q = values[(v >> (4 * j)) & 0xf];
                        const float wq = __fmul_rn(wv[j], q);
                        sumqx = __fmaf_rn(wq, xv[j], sumqx);
                        sumq2 = __fmaf_rn(wq, q, sumq2);
                    }
                }
                if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
                    d = __fdiv_rn(sumqx, sumq2);
                    best = __fmul_rn(d, sumqx);
                }
            }
        }
    }
    scales[b] = d;
}

// ik's scramble table entry for 15 bits: the word whose `j ^ (j << 1)` gives them with their parity as bit 15
static __device__ __forceinline__ uint16_t scramble(uint32_t i) {
    const uint32_t val = __popc(i) % 2 ? i | (1u << 15) : i;
    uint32_t j = 0;
    for (int k = 0; k < 15; ++k) {
        const uint32_t prev = k ? (j >> (k - 1)) & 1 : 0;
        j |= (((val >> k) & 1) ^ prev) << k;
    }
    return (uint16_t)j;
}

// a thread per block: its 8-bit scale, codebook (by the lower weighted error) and words; each value's level (index
// value times the block scale, 0 where nothing is refit) for the row's refit
static __global__ void iq4_kss_blocks(const float* x, const float* weights, int64_t blocks, int cols,
                                      const float* scales, const float* row_d, uint8_t* out, float* levels) {
    __shared__ Tables t;
    load_tables(&t);
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const int per_row = cols / BLOCK;
    const int64_t row = b / per_row;
    const int ib_row = (int)(b % per_row);
    uint16_t* words = (uint16_t*)(out + row * (ROW_HEADER + (int64_t)(cols / QK_K) * BLOCK_BYTES) + ROW_HEADER +
                                  (int64_t)ib_row * (BLOCK / 2));
    float* lv = levels + b * BLOCK;
    for (int j = 0; j < BLOCK; ++j) lv[j] = 0.0f;
    const float d = row_d[row];
    if (d == 0.0f) {
        for (int k = 0; k < WORDS; ++k) words[k] = 0;
        return;
    }
    const float id = __fdiv_rn(1.0f, d);
    int l = nearest_int(__fmul_rn(0.5f, __fadd_rn(__fmul_rn(id, scales[b]), (float)SCALE_MAX)));
    l = ((l < 0 ? 0 : (l > SCALE_MAX ? SCALE_MAX : l)) << 1) - SCALE_MAX;
    if (l == 0) {
        l += SCALE_MAX;
        for (int k = 0; k < WORDS; ++k) words[k] = (uint16_t)((l >> k) & 1);
        return;
    }
    const float* xb = x + b * BLOCK;
    const float* wb = weights + b * BLOCK;
    const float dl = __fmul_rn(d, (float)l);
    const float idl = __fdiv_rn(1.0f, dl);
    const int8_t* values = t.values;
    const int8_t* shifted = t.values + 16;
    uint16_t vp[WORDS], vm[WORDS];
    float mse_p = 0.0f, mse_m = 0.0f;
    for (int k = 0; k < WORDS; ++k) {
        float xv[4], wv[4];
        uint16_t p = 0, m = 0;
        for (int j = 0; j < 4; ++j) {
            xv[j] = xb[group_at(k, j)];
            wv[j] = wb[group_at(k, j)];
            const float al = __fmul_rn(idl, xv[j]);
            p |= (uint16_t)(best_index(t.index, values, al) << (4 * j));
            m |= (uint16_t)(best_index(t.index, shifted, al) << (4 * j));
        }
        vp[k] = prune(p, values, values, xv, wv, dl);
        vm[k] = prune(m, values, shifted, xv, wv, dl);
        for (int j = 0; j < 4; ++j) {
            float q = values[(vp[k] >> (4 * j)) & 0xf];
            float e = __fmaf_rn(-dl, q, xv[j]);
            mse_p = __fmaf_rn(__fmul_rn(wv[j], e), e, mse_p);
            q = shifted[(vm[k] >> (4 * j)) & 0xf];
            e = __fmaf_rn(-dl, q, xv[j]);
            mse_m = __fmaf_rn(__fmul_rn(wv[j], e), e, mse_m);
        }
    }
    const bool use_shifted = mse_m < mse_p;
    const uint16_t* v = use_shifted ? vm : vp;
    const int8_t* codebook = use_shifted ? shifted : values;
    for (int k = 0; k < WORDS; ++k)
        for (int j = 0; j < 4; ++j) lv[group_at(k, j)] = (float)(codebook[(v[k] >> (4 * j)) & 0xf] * l);
    l += SCALE_MAX;
    if (use_shifted) l |= 1;
    for (int k = 0; k < WORDS; ++k) words[k] = (uint16_t)((scramble(v[k] & 0x7fff) << 1) | ((l >> k) & 1));
}

// a thread per row: the levels' fit chained across the row in the CPU's order, the row scale refit to it
static __global__ void iq4_kss_refit(const float* x, const float* weights, int64_t rows, int cols, const float* row_d,
                                     const float* levels, uint8_t* out) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || row_d[row] == 0.0f) return;
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int ib = 0; ib < cols / BLOCK; ++ib) {
        const int64_t base = row * cols + (int64_t)ib * BLOCK;
        for (int k = 0; k < WORDS; ++k) {
            for (int j = 0; j < 4; ++j) {
                const int64_t i = base + group_at(k, j);
                const float q = levels[i];
                if (q == 0.0f) continue;
                const float wq = __fmul_rn(weights[i], q);
                sumqx = __fmaf_rn(wq, x[i], sumqx);
                sumq2 = __fmaf_rn(wq, q, sumq2);
            }
        }
    }
    if (!(sumq2 > 0.0f)) return;
    const uint32_t bits = __float_as_uint(__fdiv_rn(__fmul_rn(FUDGE, sumqx), sumq2));
    uint8_t* orow = out + row * (ROW_HEADER + (int64_t)(cols / QK_K) * BLOCK_BYTES);
    for (int k = 0; k < 4; ++k) orow[k] = (uint8_t)(bits >> (8 * k));
}

// as launch_iq4_ks_quantize, for IQ4_KSS; `levels` holds `rows * cols` floats (`terms` there), `shifted` is unused
extern "C" int launch_iq4_kss_quantize(const void* source, int dtype, float* x, const float* imatrix,
                                       int64_t rows_per_imatrix, int64_t rows, int cols, float* weights, float* levels,
                                       float* scales, uint8_t* shifted, float* row_d, uint8_t* out,
                                       cudaStream_t stream) {
    (void)shifted;
    const int64_t blocks = rows * (cols / BLOCK);
    to_f32<<<grid(rows * cols), THREADS, 0, stream>>>(source, dtype, rows * cols, x);
    iq4k_weights<false><<<grid(blocks), THREADS, 0, stream>>>(x, imatrix, rows_per_imatrix, blocks, cols, weights);
    iq4_kss_search<<<grid(blocks), THREADS, 0, stream>>>(x, weights, blocks, scales);
    iq4k_row_scale<<<grid(rows), THREADS, 0, stream>>>(scales, rows, cols, ROW_HEADER + (cols / QK_K) * BLOCK_BYTES,
                                                       FUDGE, row_d, out);
    iq4_kss_blocks<<<grid(blocks), THREADS, 0, stream>>>(x, weights, blocks, cols, scales, row_d, out, levels);
    iq4_kss_refit<<<grid(rows), THREADS, 0, stream>>>(x, weights, rows, cols, row_d, levels, out);
    return (int)cudaGetLastError();
}
