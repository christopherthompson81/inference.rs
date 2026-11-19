// ik_llama.cpp's IQ3_KS quantizer (`quantize_row_iq3_ks_impl`, AVX2 search) on the GPU, byte for byte the CPU port in
// iqk_quantize.rs: every product, sum and fused multiply-add is an explicit rounding intrinsic. Kernels: each block's
// weights; a warp per 32-value block searching its scale over both 3-bit codebooks (the CPU's eight interleaved
// partial sums chained between lanes); the row scale (the block scale of largest magnitude over 16, as f16); a thread
// per super-block for its packed scales and indices (the eight blocks share bytes) and each value's level; a thread
// per row refitting the row scale in the CPU's order.
#include "iq4k_common.cuh"

#define BLOCK_BYTES 102
#define F16_HEADER 2
#define KS_TRIES 5
#define K_EPS 1e-16f
#define SCALE_LO (-16)
#define SCALE_HI 15

static __constant__ int8_t kIq3Values[16] = {-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51};
static __constant__ int8_t kIq3Index[111] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
    9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 12, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 13, 13, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 14, 14, 7, 7, 7, 7, 7, 7, 7, 7, 7};
static __constant__ uint8_t kIq3Simple[128] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};

struct Iq3Tables {
    int8_t values[16];
    int8_t index[111];
    uint8_t simple[128];
};

static __device__ __forceinline__ void load_iq3_tables(Iq3Tables* t) {
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        if (i < 16) t->values[i] = kIq3Values[i];
        if (i < 111) t->index[i] = kIq3Index[i];
        t->simple[i] = kIq3Simple[i];
    }
    __syncthreads();
}

static __device__ __forceinline__ int best_index_iq3(const int8_t* index, const int8_t* values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = index[ix];
    if (ix < 8) return ix;
    return __fsub_rn(x, (float)values[ix - 8]) < __fsub_rn((float)values[ix - 7], x) ? ix - 8 : ix - 7;
}

// each block's weights: the imatrix (a block weighing all but zero replaced by IMATRIX_EPS, its sums fused) times
// sqrt(sigma2 + x^2), or x^2 without one
static __global__ void iq3_ks_weights(const float* x, const float* imatrix, int64_t rows_per_imatrix, int64_t blocks,
                                      int cols, float* weights) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const int per_row = cols / BLOCK;
    const int ib_row = (int)(b % per_row);
    const float* xb = x + b * BLOCK;
    float* w = weights + b * BLOCK;
    if (!imatrix) {
        for (int j = 0; j < BLOCK; ++j) w[j] = __fmul_rn(xb[j], xb[j]);
        return;
    }
    const float s2 = sigma2<true>(x + (b / per_row) * cols + (ib_row / (QK_K / BLOCK)) * QK_K);
    const float* qw = imatrix + (b / per_row / rows_per_imatrix) * cols + ib_row * BLOCK;
    const float floor = __fmul_rn(__fmul_rn((float)BLOCK, IMATRIX_EPS), IMATRIX_EPS);
    float sumw2 = 0.0f, sumx2 = 0.0f, sumwx = 0.0f;
    for (int j = 0; j < BLOCK; ++j) {
        sumw2 = __fmaf_rn(qw[j], qw[j], sumw2);
        sumx2 = __fmaf_rn(xb[j], xb[j], sumx2);
        sumwx = __fmaf_rn(qw[j], fabsf(xb[j]), sumwx);
    }
    const bool keep = sumw2 > floor && sumx2 > floor && sumwx > floor;
    for (int j = 0; j < BLOCK; ++j)
        w[j] = __fmul_rn(keep ? qw[j] : IMATRIX_EPS, __fsqrt_rn(__fmaf_rn(xb[j], xb[j], s2)));
}

static __device__ __forceinline__ float pick(const uint8_t* simple, const int8_t* values, float v) {
    int ix = (int)v - values[0];
    ix = ix < 0 ? 0 : (ix > 127 ? 127 : ix);
    const int t0 = simple[ix];
    const float c0 = values[t0], c1 = values[t0 + 1];
    return __fsub_rn(v, c0) < __fsub_rn(c1, v) ? c0 : c1;
}

// lane j's `fma(a, b, acc)` chained through lanes j % 8, j % 8 + 8, ... then summed in `hsum_float_8`'s order
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

// one scale try (`evaluate_block_avx2`) by the warp, at `id` and `-id`; true when it improved (d, best)
static __device__ bool evaluate(const uint8_t* simple, float id, const int8_t* values, float x, float w, int lane,
                                float* d, float* best) {
    const float v = __fmul_rn(id, x);
    const float qp = pick(simple, values, v), qm = pick(simple, values, -v);
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

// a warp per block (`find_best_scale_avx2`, step 1): its scale and whether it took the shifted codebook
static __global__ void iq3_ks_search(const float* x, const float* weights, int64_t blocks, float* scales,
                                     uint8_t* shifted) {
    __shared__ Iq3Tables t;
    load_iq3_tables(&t);
    const int64_t b = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / BLOCK;
    const int lane = threadIdx.x % BLOCK;
    if (b >= blocks) return;
    const float v = x[b * BLOCK + lane], w = weights[b * BLOCK + lane];
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
    if (amax < K_EPS) {
        if (lane == 0) {
            scales[b] = 0.0f;
            shifted[b] = 0;
        }
        return;
    }
    const int8_t* values = t.values;
    const int8_t* shift = t.values + 8;
    const float v0 = values[0];
    float d = __fdiv_rn(-max, v0), best = 0.0f;
    bool is_shifted = false;
    evaluate(t.simple, __fdiv_rn(1.0f, d), values, v, w, lane, &d, &best);
    for (int itry = -KS_TRIES; itry <= KS_TRIES; ++itry) {
        if (evaluate(t.simple, __fdiv_rn(__fadd_rn(__fmul_rn(1.0f, (float)itry), v0), max), values, v, w, lane, &d,
                     &best))
            is_shifted = false;
        if (evaluate(t.simple, __fdiv_rn(__fadd_rn(__fmul_rn(1.0f, (float)itry), (float)shift[0]), max), shift, v, w,
                     lane, &d, &best))
            is_shifted = true;
    }
    if (lane == 0) {
        scales[b] = d;
        shifted[b] = is_shifted;
    }
}

// per row: the block scale of largest magnitude (the first, in order) over -16, and the f16 header
static __global__ void iq3_ks_row_scale(const float* scales, int64_t rows, int cols, float* row_d, uint8_t* out) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int per_row = cols / BLOCK;
    float amax_scale = 0.0f, max_scale = 0.0f;
    for (int ib = 0; ib < per_row; ++ib) {
        const float s = scales[row * per_row + ib];
        if (fabsf(s) > amax_scale) {
            amax_scale = fabsf(s);
            max_scale = s;
        }
    }
    const float d = __fdiv_rn(-max_scale, 16.0f);
    row_d[row] = d;
    const uint16_t h = to_f16_bits(d);
    uint8_t* orow = out + row * (F16_HEADER + (int64_t)(cols / QK_K) * BLOCK_BYTES);
    orow[0] = (uint8_t)h;
    orow[1] = (uint8_t)(h >> 8);
}

// a thread per super-block: its codebook bits, 5-bit block scales and 3-bit indices, and each value's level
static __global__ void iq3_ks_super(const float* x, int64_t supers, int cols, const float* scales,
                                    const uint8_t* shifted, const float* row_d, uint8_t* out, float* levels) {
    __shared__ Iq3Tables t;
    load_iq3_tables(&t);
    const int64_t sb = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= supers) return;
    const int per_row = cols / QK_K;
    const int64_t row = sb / per_row;
    uint8_t* block = out + row * (F16_HEADER + (int64_t)per_row * BLOCK_BYTES) + F16_HEADER +
                     (sb % per_row) * BLOCK_BYTES;
    for (int i = 0; i < BLOCK_BYTES; ++i) block[i] = 0;
    float* lv = levels + sb * QK_K;
    const float d = row_d[row];
    uint16_t extra = 0;
    for (int ib = 0; ib < QK_K / BLOCK; ++ib)
        if (shifted[sb * (QK_K / BLOCK) + ib]) extra |= (uint16_t)(1u << (8 + ib));
    if (d == 0.0f) {
        block[0] = (uint8_t)extra;
        block[1] = (uint8_t)(extra >> 8);
        for (int j = 0; j < QK_K; ++j) lv[j] = 0.0f;
        return;
    }
    const float id = __fdiv_rn(1.0f, d);
    for (int ib = 0; ib < QK_K / BLOCK; ++ib) {
        const int64_t b = sb * (QK_K / BLOCK) + ib;
        const int8_t* codebook = shifted[b] ? t.values + 8 : t.values;
        int l = nearest_int(__fmul_rn(id, scales[b]));
        l = l < SCALE_LO ? SCALE_LO : (l > SCALE_HI ? SCALE_HI : l);
        const uint8_t ul = (uint8_t)(l - SCALE_LO);
        block[2 + ib % 4] |= (uint8_t)((ul & 0xf) << (4 * (ib / 4)));
        extra |= (uint16_t)((ul >> 4) << ib);
        const float dl = __fmul_rn(d, (float)l);
        const float idl = dl != 0.0f ? __fdiv_rn(1.0f, dl) : 0.0f;
        const float* xb = x + b * BLOCK;
        for (int j = 0; j < BLOCK; ++j) {
            const int i = best_index_iq3(t.index, codebook, __fmul_rn(idl, xb[j]));
            block[6 + (ib / 4) * BLOCK + j] |= (uint8_t)((i & 3) << (2 * (ib % 4)));
            block[70 + j] |= (uint8_t)((i >> 2) << (ib % 8));
            lv[ib * BLOCK + j] = (float)(codebook[i] * l);
        }
    }
    block[0] = (uint8_t)extra;
    block[1] = (uint8_t)(extra >> 8);
}

// a thread per row: the levels' fit chained across the row in the CPU's order, the row scale refit to it
static __global__ void iq3_ks_refit(const float* x, const float* weights, int64_t rows, int cols, const float* row_d,
                                    const float* levels, uint8_t* out) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || row_d[row] == 0.0f) return;
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int64_t i = row * cols; i < (row + 1) * cols; ++i) {
        const float q = levels[i];
        const float wq = __fmul_rn(weights[i], q);
        sumqx = __fmaf_rn(wq, x[i], sumqx);
        sumq2 = __fmaf_rn(wq, q, sumq2);
    }
    if (!(sumq2 > 0.0f)) return;
    const uint16_t h = to_f16_bits(__fdiv_rn(sumqx, sumq2));
    uint8_t* orow = out + row * (F16_HEADER + (int64_t)(cols / QK_K) * BLOCK_BYTES);
    orow[0] = (uint8_t)h;
    orow[1] = (uint8_t)(h >> 8);
}

// as launch_iq4_ks_quantize, for IQ3_KS (`terms` its per-value levels)
extern "C" int launch_iq3_ks_quantize(const void* source, int dtype, float* x, const float* imatrix,
                                      int64_t rows_per_imatrix, int64_t rows, int cols, float* weights, float* levels,
                                      float* scales, uint8_t* shifted, float* row_d, uint8_t* out,
                                      cudaStream_t stream) {
    const int64_t blocks = rows * (cols / BLOCK);
    const int64_t supers = rows * (cols / QK_K);
    to_f32<<<grid(rows * cols), THREADS, 0, stream>>>(source, dtype, rows * cols, x);
    iq3_ks_weights<<<grid(blocks), THREADS, 0, stream>>>(x, imatrix, rows_per_imatrix, blocks, cols, weights);
    iq3_ks_search<<<grid(blocks * BLOCK), THREADS, 0, stream>>>(x, weights, blocks, scales, shifted);
    iq3_ks_row_scale<<<grid(rows), THREADS, 0, stream>>>(scales, rows, cols, row_d, out);
    iq3_ks_super<<<grid(supers), THREADS, 0, stream>>>(x, supers, cols, scales, shifted, row_d, out, levels);
    iq3_ks_refit<<<grid(rows), THREADS, 0, stream>>>(x, weights, rows, cols, row_d, levels, out);
    return (int)cudaGetLastError();
}
