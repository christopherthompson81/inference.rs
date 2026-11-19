// ik_llama.cpp's IQ4_XS quantizer (`quantize_row_iq4_nl_impl` as its AVX2 build runs it) on the GPU, byte for byte the
// CPU port in iq_quantize.rs (`quantize_super_block_ik`): every product, sum and fused multiply-add is an explicit
// rounding intrinsic. Kernels: a thread per 32-value block (its weights, scale search over both codebook ends and the
// step-by-step level refinement); a thread per 256-value super-block (the shared f16 scale, 6-bit block scales and the
// indices).
#include "iq4k_common.cuh"

#define BLOCK_BYTES (2 + 2 + QK_K / 64 + QK_K / 2)
#define GROUP_MAX_EPS 1e-15f
#define SCALE_OFFSET 32
#define REFINE_STEPS (32 * BLOCK)

// the codebook index nearest `x` as the AVX2 search finds it: the lower entry around trunc(x) unless the upper is
// strictly closer
static __device__ __forceinline__ int simple_index(const uint8_t* simple, const int8_t* values, float x) {
    int ix = (int)x - values[0];
    ix = ix < 0 ? 0 : (ix > 240 ? 240 : ix);
    const int lo = simple[ix];
    return __fsub_rn((float)values[lo + 1], x) < __fsub_rn(x, (float)values[lo]) ? lo + 1 : lo;
}

// `hsum_float_8`'s order
static __device__ __forceinline__ float lanes_sum(const float* l) {
    return __fadd_rn(__fadd_rn(__fadd_rn(l[0], l[4]), __fadd_rn(l[2], l[6])),
                     __fadd_rn(__fadd_rn(l[1], l[5]), __fadd_rn(l[3], l[7])));
}

// one scale try (`evaluate_quant_block_avx2`): the fit sums kept per lane, the indices into `lb` when given
static __device__ void evaluate(const uint8_t* simple, const int8_t* values, float inverse, const float* xb,
                                const float* w, uint8_t* lb, float* sumqx, float* sumq2) {
    float sqx[LANES], sq2[LANES];
    for (int k = 0; k < LANES; ++k) sqx[k] = sq2[k] = 0.0f;
    for (int j = 0; j < BLOCK; ++j) {
        const int l = simple_index(simple, values, __fmul_rn(inverse, xb[j]));
        if (lb) lb[j] = (uint8_t)l;
        const float q = values[l];
        const float wq = __fmul_rn(w[j], q);
        sqx[j % LANES] = __fmaf_rn(wq, xb[j], sqx[j % LANES]);
        sq2[j % LANES] = __fmaf_rn(wq, q, sq2[j % LANES]);
    }
    *sumqx = lanes_sum(sqx);
    *sumq2 = lanes_sum(sq2);
}

// a thread per block: its scale, searched from both codebook ends, then its levels moved a step at a time while the
// fit improves
static __global__ void iq4_xs_search(const float* x, const float* imatrix, int64_t rows_per_imatrix, int64_t blocks,
                                     int cols, float* scales) {
    __shared__ Tables t;
    load_tables(&t);
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const int per_row = cols / BLOCK;
    const int ib_row = (int)(b % per_row);
    const float* xb = x + b * BLOCK;
    // sigma2 over the super-block, its squares summed per lane as the AVX2 build does
    const float* xs = x + (b / per_row) * cols + (ib_row / (QK_K / BLOCK)) * QK_K;
    float lanes[LANES];
    for (int k = 0; k < LANES; ++k) lanes[k] = 0.0f;
    for (int j = 0; j < QK_K; ++j) lanes[j % LANES] = __fmaf_rn(xs[j], xs[j], lanes[j % LANES]);
    const float s2 = __fmul_rn(lanes_sum(lanes), 2.0f / QK_K);
    float w[BLOCK];
    const float* qw = imatrix ? imatrix + (b / per_row / rows_per_imatrix) * cols + ib_row * BLOCK : nullptr;
    for (int j = 0; j < BLOCK; ++j)
        w[j] = qw ? __fmul_rn(qw[j], __fsqrt_rn(__fmaf_rn(xb[j], xb[j], s2))) : __fmul_rn(xb[j], xb[j]);
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < BLOCK; ++j) {
        if (fabsf(xb[j]) > amax) {
            amax = fabsf(xb[j]);
            max = xb[j];
        }
    }
    if (amax < GROUP_MAX_EPS) {
        scales[b] = 0.0f;
        return;
    }
    const int8_t* values = t.values;
    uint8_t lb[BLOCK];
    float sumqx, sumq2;
    const float id0 = __fdiv_rn(1.0f, __fdiv_rn(-max, (float)values[0]));
    evaluate(t.simple, values, id0, xb, w, lb, &sumqx, &sumq2);
    float d = __fdiv_rn(sumqx, sumq2);
    float best = __fmul_rn(d, sumqx);
    float best_sumqx = sumqx, best_sumq2 = sumq2, best_id = id0;
    const float imax = __fdiv_rn(1.0f, max);
    for (int itry = -TRIES; itry <= TRIES; ++itry) {
        for (int e = 0; e < 2; ++e) {
            const float id = __fmul_rn((float)(itry + values[e ? 15 : 0]), imax);
            evaluate(t.simple, values, id, xb, w, nullptr, &sumqx, &sumq2);
            if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
                d = __fdiv_rn(sumqx, sumq2);
                best = __fmul_rn(d, sumqx);
                best_sumqx = sumqx;
                best_sumq2 = sumq2;
                best_id = id;
            }
        }
    }
    evaluate(t.simple, values, best_id, xb, w, lb, &sumqx, &sumq2);
    sumqx = best_sumqx;
    sumq2 = best_sumq2;
    for (int step = 0; step < REFINE_STEPS; ++step) {
        float min_step = __int_as_float(0x7f800000);
        int best_j = -1, dir = 0;
        for (int j = 0; j < BLOCK; ++j) {
            const int l = lb[j];
            const float g = __fmul_rn(__fmul_rn(d, w[j]), __fmaf_rn(-d, (float)values[l], xb[j]));
            if (g > 0.0f && l < 15) {
                const float s = __fdiv_rn(__fsub_rn((float)values[l + 1], (float)values[l]), g);
                if (s < min_step) {
                    min_step = s;
                    best_j = j;
                    dir = 1;
                }
            } else if (g < 0.0f && l > 0) {
                const float s = __fdiv_rn(__fsub_rn((float)values[l - 1], (float)values[l]), g);
                if (s < min_step) {
                    min_step = s;
                    best_j = j;
                    dir = -1;
                }
            }
        }
        if (best_j < 0) break;
        const int l = lb[best_j], next = l + dir;
        const int a = values[next], c = values[l];
        const float new_sumqx = __fmaf_rn(__fmul_rn(w[best_j], xb[best_j]), (float)(a - c), sumqx);
        const float new_sumq2 = __fmaf_rn(w[best_j], (float)(a * a - c * c), sumq2);
        if (new_sumq2 > 0.0f && __fmul_rn(new_sumqx, new_sumqx) > __fmul_rn(best, new_sumq2)) {
            sumqx = new_sumqx;
            sumq2 = new_sumq2;
            d = __fdiv_rn(sumqx, sumq2);
            best = __fmul_rn(d, sumqx);
            lb[best_j] = (uint8_t)next;
        } else {
            break;
        }
    }
    scales[b] = d;
}

// a thread per super-block: the f16 scale from its largest block scale, the 6-bit block scales and the indices
static __global__ void iq4_xs_super(const float* x, int64_t supers, const float* scales, uint8_t* out) {
    __shared__ Tables t;
    load_tables(&t);
    const int64_t sb = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= supers) return;
    const float* xs = x + sb * QK_K;
    const float* sc = scales + sb * (QK_K / BLOCK);
    uint8_t* o = out + sb * BLOCK_BYTES;
    float max_scale = 0.0f, amax_scale = 0.0f;
    for (int ib = 0; ib < QK_K / BLOCK; ++ib) {
        if (fabsf(sc[ib]) > amax_scale) {
            amax_scale = fabsf(sc[ib]);
            max_scale = sc[ib];
        }
    }
    const float d = __fdiv_rn(-max_scale, (float)SCALE_OFFSET);
    const float inverse = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    uint16_t scales_h = 0;
    uint8_t scales_l[QK_K / 64] = {0, 0, 0, 0};
    for (int ib = 0; ib < QK_K / BLOCK; ++ib) {
        int l = (__float_as_int(__fmaf_rn(inverse, sc[ib], ROUNDING_MAGIC)) & 0x007fffff) - 0x00400000;
        l = l < -SCALE_OFFSET ? -SCALE_OFFSET : (l > SCALE_OFFSET - 1 ? SCALE_OFFSET - 1 : l);
        const float dl = __fmul_rn(d, (float)l);
        const float idl = dl != 0.0f ? __fdiv_rn(1.0f, dl) : 0.0f;
        const float* xb = xs + ib * BLOCK;
        uint8_t* qs = o + 8 + ib * (BLOCK / 2);
        for (int j = 0; j < BLOCK / 2; ++j) {
            const int lo = simple_index(t.simple, t.values, __fmul_rn(idl, xb[j]));
            const int hi = simple_index(t.simple, t.values, __fmul_rn(idl, xb[j + BLOCK / 2]));
            qs[j] = (uint8_t)(lo | (hi << 4));
        }
        const uint8_t code = (uint8_t)(l + SCALE_OFFSET);
        scales_l[ib / 2] |= (code & 0x0f) << (4 * (ib % 2));
        scales_h |= (uint16_t)(code >> 4) << (2 * ib);
    }
    const uint16_t dh = to_f16_bits(d);
    o[0] = (uint8_t)dh;
    o[1] = (uint8_t)(dh >> 8);
    o[2] = (uint8_t)scales_h;
    o[3] = (uint8_t)(scales_h >> 8);
    for (int k = 0; k < QK_K / 64; ++k) o[4 + k] = scales_l[k];
}

// as launch_iq4_ks_quantize, for IQ4_XS (no row scale; `weights`, `terms`, `shifted` and `row_d` unused)
extern "C" int launch_iq4_xs_quantize(const void* source, int dtype, float* x, const float* imatrix,
                                      int64_t rows_per_imatrix, int64_t rows, int cols, float* weights, float* terms,
                                      float* scales, uint8_t* shifted, float* row_d, uint8_t* out,
                                      cudaStream_t stream) {
    (void)weights;
    (void)terms;
    (void)shifted;
    (void)row_d;
    const int64_t blocks = rows * (cols / BLOCK);
    to_f32<<<grid(rows * cols), THREADS, 0, stream>>>(source, dtype, rows * cols, x);
    iq4_xs_search<<<grid(blocks), THREADS, 0, stream>>>(x, imatrix, rows_per_imatrix, blocks, cols, scales);
    iq4_xs_super<<<grid(rows * (cols / QK_K)), THREADS, 0, stream>>>(x, rows * (cols / QK_K), scales, out);
    return (int)cudaGetLastError();
}
