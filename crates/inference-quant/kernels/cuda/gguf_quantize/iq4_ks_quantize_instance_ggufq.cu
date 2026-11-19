// ik_llama.cpp's IQ4_KS quantizer (`quantize_row_iq4_k_impl_bs32`, as its AVX2 build writes the bytes) on the GPU,
// byte for byte the CPU port in iq_quantize.rs: every product, sum and fused multiply-add is an explicit rounding
// intrinsic, so nvcc contracts nothing the CPU build does not. One thread searches each 32-value block's scale; one
// thread per row then picks the 7-bit scales and indices and refits the row scale, summing in the CPU's order.
// Kernels: each block's weights; a warp per block searching its scale (lane j holds value j, and the CPU's eight
// interleaved partial sums are chained lane j - 8 to lane j); a thread per row for the row scale; a thread per block
// for its indices and refit terms; a thread per row adding the terms in the CPU's order.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define QK_K 256
#define BLOCK 32
#define LANES 8
#define TRIES 7
#define SCALE_MAX 127
#define BLOCK_BYTES (QK_K / BLOCK + QK_K / 2)
#define ROW_HEADER 4
#define EPS 1e-16f
#define IMATRIX_EPS 1e-9f
#define ROUNDING_MAGIC 12582912.0f
#define THREADS 128
// ggml's type ids for the float sources
#define GGML_F32 0
#define GGML_F16 1
#define GGML_BF16 30

__constant__ int8_t kValues[32] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117};
__constant__ int8_t kIndex[241] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 17, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 18, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 19, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 21, 21, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 22, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 23, 23, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 24, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
__constant__ uint8_t kSimple[256] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14};

// ggml's nearest_int: the float added to 1.5 * 2^23, ties to even
__device__ __forceinline__ int nearest_int(float x) {
    return (__float_as_int(__fadd_rn(x, ROUNDING_MAGIC)) & 0x007fffff) - 0x00400000;
}

__device__ __forceinline__ int best_index(const int8_t* index, const int8_t* values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = index[ix];
    if (ix < 16) return ix;
    return __fsub_rn(x, (float)values[ix - 16]) < __fsub_rn((float)values[ix - 15], x) ? ix - 16 : ix - 15;
}

// the 256 values' sum of squares (fused, in order) times 2 / 256
__device__ float sigma2(const float* xbl) {
    float s = 0.0f;
    for (int j = 0; j < QK_K; ++j) s = __fmaf_rn(xbl[j], xbl[j], s);
    return __fmul_rn(s, 2.0f / QK_K);
}

// each block's weights: the imatrix (a block weighing all but zero replaced by IMATRIX_EPS) times sqrt(sigma2 + x^2),
// or x^2 without one
__global__ void iq4_ks_weights(const float* x, const float* imatrix, int64_t rows_per_imatrix, int64_t blocks, int cols,
                               float* weights) {
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
    const float s2 = sigma2(x + (b / per_row) * cols + (ib_row / (QK_K / BLOCK)) * QK_K);
    const float* qw = imatrix + (b / per_row / rows_per_imatrix) * cols + ib_row * BLOCK;
    const float floor = __fmul_rn(__fmul_rn((float)BLOCK, IMATRIX_EPS), IMATRIX_EPS);
    float sumw2 = 0.0f, sumx2 = 0.0f, sumwx = 0.0f;
    for (int j = 0; j < BLOCK; ++j) sumw2 = __fadd_rn(sumw2, __fmul_rn(qw[j], qw[j]));
    for (int j = 0; j < BLOCK; ++j) sumx2 = __fadd_rn(sumx2, __fmul_rn(xb[j], xb[j]));
    for (int j = 0; j < BLOCK; ++j) sumwx = __fadd_rn(sumwx, __fmul_rn(qw[j], fabsf(xb[j])));
    const bool keep = sumw2 > floor && sumx2 > floor && sumwx > floor;
    for (int j = 0; j < BLOCK; ++j)
        w[j] = __fmul_rn(keep ? qw[j] : IMATRIX_EPS, __fsqrt_rn(__fmaf_rn(xb[j], xb[j], s2)));
}

// the tables as each kernel copies them to shared memory: constant memory serializes a warp's differing indices
struct Tables {
    int8_t values[32];
    int8_t index[241];
    uint8_t simple[256];
};

__device__ __forceinline__ void load_tables(Tables* t) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        if (i < 32) t->values[i] = kValues[i];
        if (i < 241) t->index[i] = kIndex[i];
        t->simple[i] = kSimple[i];
    }
    __syncthreads();
}

__device__ __forceinline__ float candidate(const uint8_t* simple, const int8_t* values, float xi) {
    int ix = (int)xi - values[0];
    ix = ix < 0 ? 0 : (ix > 240 ? 240 : ix);
    const int lo = simple[ix];
    const float c0 = values[lo], c1 = values[lo + 1];
    return __fsub_rn(xi, c0) < __fsub_rn(c1, xi) ? c0 : c1;
}

// lane j's `fma(a, b, acc)` chained through lanes j % 8, j % 8 + 8, ... as the CPU's eight partial sums run, then
// those summed in `hsum_float_8`'s order; every lane gets the total
__device__ __forceinline__ float chained_sum(float a, float b, int lane) {
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
__device__ bool evaluate(const uint8_t* simple, float inverse, const int8_t* values, float x, float w, int lane, float* d,
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
__global__ void iq4_ks_search(const float* x, const float* weights, int64_t blocks, float* scales, uint8_t* shifted) {
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

// per row: the row scale, the largest block scale over 127, also the header unless the refit replaces it
__global__ void iq4_ks_row_scale(const float* scales, int64_t rows, int cols, float* row_d, uint8_t* out) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int per_row = cols / BLOCK;
    float amax_scale = 0.0f;
    for (int ib = 0; ib < per_row; ++ib) amax_scale = fmaxf(amax_scale, fabsf(scales[row * per_row + ib]));
    const float d = __fdiv_rn(amax_scale, (float)SCALE_MAX);
    row_d[row] = d;
    const uint32_t bits = __float_as_uint(d);
    uint8_t* orow = out + row * (ROW_HEADER + (int64_t)(cols / QK_K) * BLOCK_BYTES);
    for (int k = 0; k < 4; ++k) orow[k] = (uint8_t)(bits >> (8 * k));
}

// a thread per block: its 7-bit scale and indices, and its refit terms (the CPU adds them to the row's sums in order)
__global__ void iq4_ks_blocks(const float* x, const float* weights, int64_t blocks, int cols, const float* scales,
                              const uint8_t* shifted, const float* row_d, uint8_t* out, float* terms_qx,
                              float* terms_q2) {
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
    if (d == 0.0f) {
        sb[ib] = 0;
        for (int j = 0; j < BLOCK / 2; ++j) qs[j] = 0;
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
        const float w1 = __fmul_rn(w[j], q1), w2 = __fmul_rn(w[j + BLOCK / 2], q2);
        tx[j] = __fmaf_rn(w1, xb[j], __fmul_rn(w2, xb[j + BLOCK / 2]));
        t2[j] = __fmaf_rn(w1, q1, __fmul_rn(w2, q2));
    }
}

// per row: the terms added in the CPU's order, and the row scale refit to them
__global__ void iq4_ks_refit(int64_t rows, int cols, const float* row_d, const float* terms_qx, const float* terms_q2,
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

// the source rows as f32 (f16 and bf16 widen exactly)
__global__ void to_f32(const void* src, int dtype, int64_t n, float* dst) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (dtype == GGML_BF16) dst[i] = __bfloat162float(((const __nv_bfloat16*)src)[i]);
    else if (dtype == GGML_F16) dst[i] = __half2float(((const __half*)src)[i]);
    else dst[i] = ((const float*)src)[i];
}

static unsigned grid(int64_t n) { return (unsigned)((n + THREADS - 1) / THREADS); }

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
    iq4_ks_weights<<<grid(blocks), THREADS, 0, stream>>>(x, imatrix, rows_per_imatrix, blocks, cols, weights);
    iq4_ks_search<<<grid(blocks * BLOCK), THREADS, 0, stream>>>(x, weights, blocks, scales, shifted);
    iq4_ks_row_scale<<<grid(rows), THREADS, 0, stream>>>(scales, rows, cols, row_d, out);
    iq4_ks_blocks<<<grid(blocks), THREADS, 0, stream>>>(x, weights, blocks, cols, scales, shifted, row_d, out, terms,
                                                        terms + half);
    iq4_ks_refit<<<grid(rows), THREADS, 0, stream>>>(rows, cols, row_d, terms, terms + half, out);
    return (int)cudaGetLastError();
}
