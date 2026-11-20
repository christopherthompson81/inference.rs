// What ik_llama.cpp's quantizers share on the GPU (the *_quantize_instance_ggufq.cu files): the IQ4 codebooks and
// index tables, ggml's rounding, f16 conversion, the IQ4_K block weights and row scale. Everything is internal to each
// instance file that includes it.
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define QK_K 256
#define BLOCK 32
#define LANES 8
#define TRIES 7
#define SCALE_MAX 127
#define ROW_HEADER 4
#define EPS 1e-16f
#define IMATRIX_EPS 1e-9f
#define ROUNDING_MAGIC 12582912.0f
#define THREADS 128
// ggml's type ids for the float sources
#define GGML_F32 0
#define GGML_F16 1
#define GGML_BF16 30

static __constant__ int8_t kValues[32] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117};
static __constant__ int8_t kIndex[241] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 17, 17, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 18, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 19, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 20, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 21, 21, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 22, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 23, 23, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 24, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};
static __constant__ uint8_t kSimple[256] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14};

// ggml's nearest_int: the float added to 1.5 * 2^23, ties to even
static __device__ __forceinline__ int nearest_int(float x) {
    return (__float_as_int(__fadd_rn(x, ROUNDING_MAGIC)) & 0x007fffff) - 0x00400000;
}

static __device__ __forceinline__ int best_index(const int8_t* index, const int8_t* values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = index[ix];
    if (ix < 16) return ix;
    return __fsub_rn(x, (float)values[ix - 16]) < __fsub_rn((float)values[ix - 15], x) ? ix - 16 : ix - 15;
}

// the 256 values' sum of squares in order (fused, or not: gcc fuses it in IQ4_KS's function, not IQ4_KSS's) times
// 2 / 256
template <bool FUSED>
static __device__ float sigma2(const float* xbl) {
    float s = 0.0f;
    for (int j = 0; j < QK_K; ++j) s = FUSED ? __fmaf_rn(xbl[j], xbl[j], s) : __fadd_rn(s, __fmul_rn(xbl[j], xbl[j]));
    return __fmul_rn(s, 2.0f / QK_K);
}

// each block's weights: the imatrix (a block weighing all but zero replaced by IMATRIX_EPS) times sqrt(sigma2 + x^2),
// or x^2 without one
template <bool FUSED>
static __global__ void iq4k_weights(const float* x, const float* imatrix, int64_t rows_per_imatrix, int64_t blocks, int cols,
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
    const float s2 = sigma2<FUSED>(x + (b / per_row) * cols + (ib_row / (QK_K / BLOCK)) * QK_K);
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

static __device__ __forceinline__ void load_tables(Tables* t) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        if (i < 32) t->values[i] = kValues[i];
        if (i < 241) t->index[i] = kIndex[i];
        t->simple[i] = kSimple[i];
    }
    __syncthreads();
}

// a block's values per lane when LANES lanes share it
#define SLOTS (BLOCK / LANES)

// the mask of this lane's aligned group of LANES lanes
static __device__ __forceinline__ unsigned group_mask() {
    return ((1u << LANES) - 1u) << ((threadIdx.x % 32) / LANES * LANES);
}

// the group's eight partial sums added in `hsum_float_8`'s order, ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7)):
// the xor steps pair the same operands (addition commutes exactly), so every lane holds the same total
static __device__ __forceinline__ float group_sum(float p, unsigned mask) {
    p = __fadd_rn(p, __shfl_xor_sync(mask, p, 4, LANES));
    p = __fadd_rn(p, __shfl_xor_sync(mask, p, 2, LANES));
    return __fadd_rn(p, __shfl_xor_sync(mask, p, 1, LANES));
}

// the group's largest magnitude and its first index (the CPU's strict `>` scan), in every lane
static __device__ __forceinline__ void group_first_max(float* amax, int* at, unsigned mask) {
    for (int off = LANES / 2; off > 0; off /= 2) {
        const float oa = __shfl_xor_sync(mask, *amax, off, LANES);
        const int oat = __shfl_xor_sync(mask, *at, off, LANES);
        if (oa > *amax || (oa == *amax && oat < *at)) {
            *amax = oa;
            *at = oat;
        }
    }
}

static __device__ __forceinline__ float candidate(const uint8_t* simple, const int8_t* values, float xi) {
    int ix = (int)xi - values[0];
    ix = ix < 0 ? 0 : (ix > 240 ? 240 : ix);
    const int lo = simple[ix];
    const float c0 = values[lo], c1 = values[lo + 1];
    return __fsub_rn(xi, c0) < __fsub_rn(c1, xi) ? c0 : c1;
}

// per row: the row scale, the largest block scale over 127, and the header (times `fudge`) unless the refit replaces it
static __global__ void iq4k_row_scale(const float* scales, int64_t rows, int cols, int row_bytes, float fudge,
                                      float* row_d, uint8_t* out) {
    const int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int per_row = cols / BLOCK;
    float amax_scale = 0.0f;
    for (int ib = 0; ib < per_row; ++ib) amax_scale = fmaxf(amax_scale, fabsf(scales[row * per_row + ib]));
    const float d = __fdiv_rn(amax_scale, (float)SCALE_MAX);
    row_d[row] = d;
    if (!out) return;
    const uint32_t bits = __float_as_uint(__fmul_rn(fudge, d));
    uint8_t* orow = out + row * (int64_t)row_bytes;
    for (int k = 0; k < 4; ++k) orow[k] = (uint8_t)(bits >> (8 * k));
}

// f32 to f16 bits, ties to even, keeping subnormals (the `half` crate's conversion): the library builds with
// `--use_fast_math`, whose flush-to-zero would turn `__float2half_rn`'s subnormal results to 0
static __device__ uint16_t to_f16_bits(float f) {
    const uint32_t x = __float_as_uint(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = x & 0x7f800000u;
    const uint32_t man = x & 0x007fffffu;
    if (exp == 0x7f800000u) return (uint16_t)(sign | 0x7c00u | (man ? 0x0200u | (man >> 13) : 0u));
    const int unbiased = (int)(exp >> 23) - 127;
    if (unbiased >= 16) return (uint16_t)(sign | 0x7c00u);
    const int half_exp = unbiased + 15;
    if (half_exp <= 0) {
        if (14 - half_exp > 24) return (uint16_t)sign;
        const uint32_t m = man | 0x00800000u;
        uint32_t half_man = m >> (14 - half_exp);
        const uint32_t round_bit = 1u << (13 - half_exp);
        if ((m & round_bit) && (m & (3u * round_bit - 1u))) half_man += 1;
        return (uint16_t)(sign | half_man);
    }
    const uint32_t h = sign | ((uint32_t)half_exp << 10) | (man >> 13);
    return (uint16_t)((man & 0x1000u) && (man & 0x2fffu) ? h + 1 : h);
}

// the source rows as f32 (f16 and bf16 widen exactly)
static __global__ void to_f32(const void* src, int dtype, int64_t n, float* dst) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (dtype == GGML_BF16) dst[i] = __bfloat162float(((const __nv_bfloat16*)src)[i]);
    else if (dtype == GGML_F16) dst[i] = __half2float(((const __half*)src)[i]);
    else dst[i] = ((const float*)src)[i];
}

static unsigned grid(int64_t n) { return (unsigned)((n + THREADS - 1) / THREADS); }

