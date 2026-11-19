// ik_llama.cpp's IQ4_KS quantizer (`quantize_row_iq4_k_impl_bs32`, as its AVX2 build writes the bytes) on the GPU,
// byte for byte the CPU port in iq_quantize.rs: every product, sum and fused multiply-add is an explicit rounding
// intrinsic, so nvcc contracts nothing the CPU build does not.
// Kernels: each block's weights; a warp per block searching its scale (lane j holds value j, and the CPU's eight
// interleaved partial sums are chained lane j - 8 to lane j); a thread per row for the row scale; a thread per block
// for its indices and refit terms; a thread per row adding the terms in the CPU's order.
#include "iq4k_common.cuh"

#define BLOCK_BYTES (QK_K / BLOCK + QK_K / 2)

// lane j's `fma(a, b, acc)` chained through lanes j % 8, j % 8 + 8, ... as the CPU's eight partial sums run, then
// those summed in `hsum_float_8`'s order; every lane gets the total
static __device__ __forceinline__ float chained_sum(float a, float b, int lane) {
    float acc = __fmaf_rn(a, b, 0.0f);
    for (int step = 1; step < BLOCK / LANES; ++step) {
        const float prev = __shfl_up_sync(0xffffffffu, acc, LANES);
        if (lane / LANES == step) acc = __fmaf_rn(a, b, prev);
    }
    float l[LANES];
    for (int k = 0; k < LANES; ++k) l[k] = __shfl_sync(0xffffffffu, acc, BLOCK - LANES + k);
    return __fadd_rn(__fadd_rn(__fadd_rn(l[0], l[4]), __fadd_rn(l[2], l[6])),
                     __fadd_rn(__fadd_rn(l[1], l[5]), __fadd_rn(l[3], l[7])));
}

// one scale try (`evaluate_block_iq4k`) by the warp; true when it improved (d, best), the same in every lane
static __device__ bool evaluate(const uint8_t* simple, float inverse, const int8_t* values, float x, float w, int lane, float* d,
                         float* best) {
    const float xi = __fmul_rn(inverse, x);
    const float qp = candidate(simple, values, xi), qm = candidate(simple, values, -xi);
    const float wp = __fmul_rn(w, qp), wm = __fmul_rn(w, qm);
    const float sqx_p = chained_sum(wp, x, lane), sq2_p = chained_sum(wp, qp, lane);
    const float sqx_m = chained_sum(wm, x, lane), sq2_m = chained_sum(wm, qm, lane);
    bool updated = false;
    if (sq2_p > 0.0f && __fmul_rn(sqx_p, sqx_p) > __fmul_rn(sq2_p, *best)) {
        *d = __fdiv_rn(sqx_p, sq2_p);
        *best = __fmul_rn(*d, sqx_p);
        updated = true;
    }
    if (sq2_m > 0.0f && __fmul_rn(sqx_m, sqx_m) > __fmul_rn(sq2_m, *best)) {
        *d = __fdiv_rn(sqx_m, sq2_m);
        *best = __fmul_rn(*d, sqx_m);
        updated = true;
    }
    return updated;
}

// a warp per block: its best scale and whether it takes the shifted codebook
static __global__ void iq4_ks_search(const float* x, const float* weights, int64_t blocks, float* scales, uint8_t* shifted) {
    __shared__ Tables t;
    load_tables(&t);
    const int64_t b = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / BLOCK;
    const int lane = threadIdx.x % BLOCK;
    if (b >= blocks) return;
    const float v = x[b * BLOCK + lane], w = weights[b * BLOCK + lane];
    // the first index holding the largest magnitude, as the CPU's strict `>` scan finds it
    float amax = fabsf(v);
    int at = lane;
    for (int off = BLOCK / 2; off > 0; off /= 2) {
        const float oa = __shfl_down_sync(0xffffffffu, amax, off);
        const int oat = __shfl_down_sync(0xffffffffu, at, off);
        if (oa > amax || (oa == amax && oat < at)) {
            amax = oa;
            at = oat;
        }
    }
    amax = __shfl_sync(0xffffffffu, amax, 0);
    at = __shfl_sync(0xffffffffu, at, 0);
    const float max = __shfl_sync(0xffffffffu, v, at);
    if (amax < EPS) {
        if (lane == 0) {
            scales[b] = 0.0f;
            shifted[b] = 0;
        }
        return;
    }
    const int8_t* values = t.values;
    const int8_t* shift = t.values + 16;
    float d = __fdiv_rn(-max, (float)values[0]), best = 0.0f;
    bool is_shifted = false;
    evaluate(t.simple, __fdiv_rn(1.0f, d), values, v, w, lane, &d, &best);
    for (int itry = -TRIES; itry <= TRIES; ++itry) {
        if (evaluate(t.simple, __fdiv_rn(__fadd_rn((float)itry, (float)values[0]), max), values, v, w, lane, &d, &best))
            is_shifted = false;
        if (evaluate(t.simple, __fdiv_rn(__fadd_rn((float)itry, (float)shift[0]), max), shift, v, w, lane, &d, &best))
            is_shifted = true;
    }
    if (lane == 0) {
        scales[b] = d;
        shifted[b] = is_shifted;
    }
}

// a thread per block: its 7-bit scale and indices, and its refit terms (the CPU adds them to the row's sums in order)
static __global__ void iq4_ks_blocks(const float* x, const float* weights, int64_t blocks, int cols, const float* scales,
                              const uint8_t* shifted, const float* row_d, uint8_t* out, float* terms_qx,
                              float* terms_q2, float* levels) {
    __shared__ Tables t;
    load_tables(&t);
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const int per_row = cols / BLOCK;
    const int64_t row = b / per_row;
    const int ib_row = (int)(b % per_row);
    const int ibl = ib_row / (QK_K / BLOCK), ib = ib_row % (QK_K / BLOCK);
    uint8_t* sb = out + row * (ROW_HEADER + (int64_t)(cols / QK_K) * BLOCK_BYTES) + ROW_HEADER + ibl * BLOCK_BYTES;
    uint8_t* qs = sb + QK_K / BLOCK + ib * (BLOCK / 2);
    float* tx = terms_qx + b * (BLOCK / 2);
    float* t2 = terms_q2 + b * (BLOCK / 2);
    const float d = row_d[row];
    float* lv = levels ? levels + b * BLOCK : nullptr;
    if (d == 0.0f) {
        sb[ib] = 0;
        for (int j = 0; j < BLOCK / 2; ++j) {
            qs[j] = 0;
            tx[j] = t2[j] = 0.0f;
        }
        if (lv)
            for (int j = 0; j < BLOCK; ++j) lv[j] = 0.0f;
        return;
    }
    const float id = __fdiv_rn(1.0f, d);
    const int8_t* codebook = shifted[b] ? t.values + 16 : t.values;
    int l = nearest_int(__fmul_rn(0.5f, __fadd_rn(__fmul_rn(id, scales[b]), (float)SCALE_MAX)));
    l = (l < 0 ? 0 : (l > SCALE_MAX ? SCALE_MAX : l)) << 1;
    sb[ib] = (uint8_t)(shifted[b] | l);
    l -= SCALE_MAX;
    const float dl = __fmul_rn(d, (float)l);
    const float idl = dl != 0.0f ? __fdiv_rn(1.0f, dl) : 0.0f;
    const float* xb = x + b * BLOCK;
    const float* w = weights + b * BLOCK;
    for (int j = 0; j < BLOCK / 2; ++j) {
        const int i1 = best_index(t.index, codebook, __fmul_rn(idl, xb[j]));
        const int i2 = best_index(t.index, codebook, __fmul_rn(idl, xb[j + BLOCK / 2]));
        qs[j] = (uint8_t)(i1 | (i2 << 4));
        const float q1 = __fmul_rn((float)codebook[i1], (float)l);
        const float q2 = __fmul_rn((float)codebook[i2], (float)l);
        if (lv) {
            lv[j] = q1;
            lv[j + BLOCK / 2] = q2;
        }
        const float w1 = __fmul_rn(w[j], q1), w2 = __fmul_rn(w[j + BLOCK / 2], q2);
        tx[j] = __fmaf_rn(w1, xb[j], __fmul_rn(w2, xb[j + BLOCK / 2]));
        t2[j] = __fmaf_rn(w1, q1, __fmul_rn(w2, q2));
    }
}

// per row: the terms added in the CPU's order, and the row scale refit to them
static __global__ void iq4_ks_refit(int64_t rows, int cols, const float* row_d, const float* terms_qx, const float* terms_q2,
                             uint8_t* out) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || row_d[row] == 0.0f) return;
    const int n = cols / 2;
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int k = 0; k < n; ++k) {
        sumqx = __fadd_rn(sumqx, terms_qx[row * n + k]);
        sumq2 = __fadd_rn(sumq2, terms_q2[row * n + k]);
    }
    if (!(sumq2 > 0.0f)) return;
    const uint32_t bits = __float_as_uint(__fdiv_rn(sumqx, sumq2));
    uint8_t* orow = out + row * (ROW_HEADER + (int64_t)(cols / QK_K) * BLOCK_BYTES);
    for (int k = 0; k < 4; ++k) orow[k] = (uint8_t)(bits >> (8 * k));
}

// `rows` rows of `cols` values of ggml type `dtype` (f32, f16 or bf16; `x` then holds them as f32) to IQ4_KS rows in
// `out`, `imatrix` holding `cols` importances per `rows_per_imatrix` rows (an expert stack's experts one after
// another) or null. Scratch: `weights` and
// `terms` hold `rows * cols` floats, `scales` and `shifted` one entry per 32 values, `row_d` one per row. Returns 0,
// or a CUDA error code.
extern "C" int launch_iq4_ks_quantize(const void* source, int dtype, float* x, const float* imatrix,
                                      int64_t rows_per_imatrix, int64_t rows, int cols, float* weights,
                                      float* terms, float* scales, uint8_t* shifted, float* row_d, uint8_t* out,
                                      cudaStream_t stream) {
    const int64_t blocks = rows * (cols / BLOCK);
    const int64_t half = rows * cols / 2;
    to_f32<<<grid(rows * cols), THREADS, 0, stream>>>(source, dtype, rows * cols, x);
    iq4k_weights<true><<<grid(blocks), THREADS, 0, stream>>>(x, imatrix, rows_per_imatrix, blocks, cols, weights);
    iq4_ks_search<<<grid(blocks * BLOCK), THREADS, 0, stream>>>(x, weights, blocks, scales, shifted);
    iq4k_row_scale<<<grid(rows), THREADS, 0, stream>>>(scales, rows, cols, ROW_HEADER + (cols / QK_K) * BLOCK_BYTES,
                                                       1.0f, row_d, out);
    iq4_ks_blocks<<<grid(blocks), THREADS, 0, stream>>>(x, weights, blocks, cols, scales, shifted, row_d, out, terms,
                                                        terms + half, nullptr);
    iq4_ks_refit<<<grid(rows), THREADS, 0, stream>>>(rows, cols, row_d, terms, terms + half, out);
    return (int)cudaGetLastError();
}

// GPTQ's pieces (a row scale fixed first, then a super-block at a time): each row's running fit sums continued with a
// super-block's terms in the CPU's order
static __global__ void iq4_ks_accumulate(int64_t rows, const float* terms_qx, const float* terms_q2, float* sums) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sqx = sums[2 * row], sq2 = sums[2 * row + 1];
    for (int k = 0; k < QK_K / 2; ++k) {
        sqx = __fadd_rn(sqx, terms_qx[row * (QK_K / 2) + k]);
        sq2 = __fadd_rn(sq2, terms_q2[row * (QK_K / 2) + k]);
    }
    sums[2 * row] = sqx;
    sums[2 * row + 1] = sq2;
}

// each row's scale from an ordinary first pass (the largest block scale over 127), `importance` its `cols` weights
extern "C" int launch_iq4_ks_row_scale(const float* x, const float* importance, int64_t rows, int cols, float* weights,
                                       float* scales, uint8_t* shifted, float* row_d, cudaStream_t stream) {
    const int64_t blocks = rows * (cols / BLOCK);
    iq4k_weights<true><<<grid(blocks), THREADS, 0, stream>>>(x, importance, rows, blocks, cols, weights);
    iq4_ks_search<<<grid(blocks * BLOCK), THREADS, 0, stream>>>(x, weights, blocks, scales, shifted);
    iq4k_row_scale<<<grid(rows), THREADS, 0, stream>>>(scales, rows, cols, 0, 1.0f, row_d, nullptr);
    return (int)cudaGetLastError();
}

// one super-block of each row (`x`: rows of 256) against its fixed row scale: `out` rows of the header's width then the
// super-block's bytes, `levels` each value's level, `sums` (2 a row) continued
extern "C" int launch_iq4_ks_super_fixed(const float* x, const float* importance, int64_t rows, const float* row_d,
                                         float* weights, float* scales, uint8_t* shifted, uint8_t* out, float* terms,
                                         float* levels, float* sums, cudaStream_t stream) {
    const int64_t blocks = rows * (QK_K / BLOCK);
    const int64_t half = rows * QK_K / 2;
    iq4k_weights<true><<<grid(blocks), THREADS, 0, stream>>>(x, importance, rows, blocks, QK_K, weights);
    iq4_ks_search<<<grid(blocks * BLOCK), THREADS, 0, stream>>>(x, weights, blocks, scales, shifted);
    iq4_ks_blocks<<<grid(blocks), THREADS, 0, stream>>>(x, weights, blocks, QK_K, scales, shifted, row_d, out, terms,
                                                        terms + half, levels);
    iq4_ks_accumulate<<<grid(rows), THREADS, 0, stream>>>(rows, terms, terms + half, sums);
    return (int)cudaGetLastError();
}
