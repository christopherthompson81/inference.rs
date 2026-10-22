#pragma once

// Decode and speculative-verify attention over a quantized paged K/V cache: few Q rows a sequence, every Q head of a
// K/V head in one block, the cache split across blocks. K and V are widened to f16 in smem, scale included, for f16
// MMA (an int8 Q.K pays a rescale per score and 32-value group, which costs more than it saves). Small tiles keep two
// blocks an SM, so one block's copies and unpacks overlap the other's math.

// included after fattn-mma-f16.cuh, whose cp.async, mma and row helpers it uses
#include "fattn-mma-f16.cuh"

#define FQ_TILE        32   // keys a tile
#define FQ_WARPS       4
#define FQ_GROUP       32   // values a quantization scale covers
#define FQ_MAX_ROWS    8
#define FQ_MAX_COLS    64   // Q rows x Q heads of one K/V head
// the transposed kernel's 8-column n tiles, and how many a warp holds (each costs 64 registers at head 256)
#define FQ_T_TILES       6
#define FQ_T_TILES_WARP  3
// tiles a split takes at least: short caches use fewer splits than the grid has, decided on the device (graph capture
// fixes the grid, the lengths stay device-side)
#define FQ_MIN_SPLIT_TILES 8
// copies whose block-table lookups a thread issues before any copy
#define FQ_LOOKUPS     4

static __device__ __forceinline__ int fq_live_splits(const int kv_len, const int splits) {
    const int ntiles = (kv_len + FQ_TILE - 1) / FQ_TILE;
    return max(1, min(splits, (ntiles + FQ_MIN_SPLIT_TILES - 1) / FQ_MIN_SPLIT_TILES));
}

// a staged quantized row in smem: its bytes rounded up to the 16-byte cp.async unit
static __host__ __device__ __forceinline__ int fq_raw_stride(const int D, const int bits) {
    return (D*bits/8 + D/16 + 15) & ~15;
}

// smem ahead of the raw slots; two raw slots follow, each a tile's K rows then its V rows at their bits
template<int D>
struct fq_smem {
    static constexpr int f16_stride = 2*D + 16;    // bytes: 16 of padding keep ldmatrix rows on distinct banks
    static constexpr int f16_bytes  = FQ_TILE*f16_stride;
    static constexpr int fixed      = 2*f16_bytes; // K then V
};

static __host__ __device__ __forceinline__ int fq_raw_slot(const int D, const int k_bits, const int v_bits) {
    return FQ_TILE*(fq_raw_stride(D, k_bits) + fq_raw_stride(D, v_bits));
}

static __device__ __forceinline__ void fq_cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// waits for all but the newest committed group
static __device__ __forceinline__ void fq_cp_async_wait_prev() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group 1;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void fq_cp_async_4(const unsigned int dst, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dst), "l"(src));
#else
    GGML_UNUSED_VARS(dst, src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Starts the async copies of tile rows [k0, k0 + FQ_TILE) of K and V into a raw slot. Rows past the sequence are
// masked: the general path reads the first row for them, the contiguous one the block's own (stale) rows.
template<int D>
static __device__ __forceinline__ void fq_stage_tile(
        const uint8_t * K, const uint8_t * V, const fattn_kv_rows & rk, const fattn_kv_rows & rv,
        const int stride_K, const int stride_V, uint8_t * slot, const int k0) {
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int nthreads = blockDim.y*WARP_SIZE;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const fattn_kv_rows & rows = part ? rv : rk;
        const uint8_t * src_base = part ? V : K;
        const int stride_src = part ? stride_V : stride_K;
        const int row_bytes = rows.data_bytes + D/16;
        const int stride = fq_raw_stride(D, rows.bits);
        uint8_t * dst_base = slot + part*FQ_TILE*fq_raw_stride(D, rk.bits);
        const unsigned int dst_32 = ggml_cuda_cvta_generic_to_shared(dst_base);
        // a tile inside one paged block whose rows sit back to back, staged as packed: one contiguous copy
        if ((1 << rows.block_size_log2) >= FQ_TILE && stride == row_bytes && 4*stride_src == row_bytes) {
            const uint8_t * src = src_base + 4*fattn_kv_row(rows, k0, stride_src);
            for (int c = 16*tid; c < FQ_TILE*row_bytes; c += 16*nthreads) {
                cp_async_cg_16<0>(dst_32 + c, src + c);
            }
            continue;
        }
        const int unit  = row_bytes % 16 == 0 ? 16 : 4;
        const int units = row_bytes / unit;
        const int items = FQ_TILE*units;
        for (int item0 = tid; item0 < items; item0 += nthreads*FQ_LOOKUPS) {
            int64_t row[FQ_LOOKUPS];
#pragma unroll
            for (int k = 0; k < FQ_LOOKUPS; ++k) {
                const int item = item0 + k*nthreads;
                row[k] = item < items ? 4*fattn_kv_row(rows, k0 + item/units, stride_src) : 0;
            }
#pragma unroll
            for (int k = 0; k < FQ_LOOKUPS; ++k) {
                const int item = item0 + k*nthreads;
                if (item >= items) {
                    break;
                }
                const int i = item / units;
                const int u = item - i*units;
                if (unit == 16) {
                    cp_async_cg_16<0>(dst_32 + i*stride + u*16, src_base + row[k] + u*16);
                } else {
                    fq_cp_async_4(dst_32 + i*stride + u*4, src_base + row[k] + u*4);
                }
            }
        }
    }
}

// K or V as f16 rows, (2c + 1 - 2^bits) * scale / 2^bits: 0x6400 | (2c + 1) is the half 1024 + 2c + 1, so one exact
// subtract and one multiply convert two values. Rows from `valid` on are zeroed: stale bytes there could hold a NaN,
// which P = 0 would not cancel.
template<int D, int bits>
static __device__ __forceinline__ void fq_unpack_f16(const uint8_t * raw, const int data_bytes, uint8_t * f16, const int valid) {
    constexpr int groups = D/FQ_GROUP;
    constexpr uint32_t mask = (1u << bits) - 1;
    constexpr int stride = (D*bits/8 + D/16 + 15) & ~15;
    const half2 bias = __float2half2_rn(1024.0f + float(1 << bits));
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    for (int item = tid; item < FQ_TILE*groups; item += blockDim.y*WARP_SIZE) {
        const int i = item / groups;
        const int g = item - i*groups;
        const uint8_t * row = raw + i*stride;
        const uint32_t * src = (const uint32_t *) row + g*bits;
        uint32_t w[bits];
#pragma unroll
        for (int k = 0; k < bits; ++k) {
            w[k] = src[k];
        }
        const half2 q2 = __float2half2_rn(__half2float(((const half *) (row + data_bytes))[g]) / float(1 << bits));
        __align__(16) half2 vals[FQ_GROUP/2];
#pragma unroll
        for (int j = 0; j < FQ_GROUP; j += 2) {
            uint32_t c[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int p = (j + e)*bits;
                const int wi = p/32;
                const int off = p%32;
                c[e] = (off + bits <= 32 ? w[wi] >> off : __funnelshift_r(w[wi], w[wi + 1], off)) & mask;
            }
            const uint32_t u = 0x64016401u | (c[0] << 1) | (c[1] << 17);
            vals[j/2] = i < valid ? __hmul2(__hsub2(*(const half2 *) &u, bias), q2) : make_half2(0.0f, 0.0f);
        }
        uint8_t * dst = f16 + i*(2*D + 16) + g*2*FQ_GROUP;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            ggml_cuda_memcpy_1<16>(dst + 16*c, vals + 4*c);
        }
    }
}

template<int D>
static __device__ __forceinline__ void fq_unpack_rows(const int bits, const uint8_t * raw, const int data_bytes,
        uint8_t * f16, const int valid) {
    switch (bits) {
        case 2: fq_unpack_f16<D, 2>(raw, data_bytes, f16, valid); break;
        case 3: fq_unpack_f16<D, 3>(raw, data_bytes, f16, valid); break;
        case 4: fq_unpack_f16<D, 4>(raw, data_bytes, f16, valid); break;
        case 5: fq_unpack_f16<D, 5>(raw, data_bytes, f16, valid); break;
        case 6: fq_unpack_f16<D, 6>(raw, data_bytes, f16, valid); break;
        case 7: fq_unpack_f16<D, 7>(raw, data_bytes, f16, valid); break;
        default: fq_unpack_f16<D, 8>(raw, data_bytes, f16, valid); break;
    }
}

// the transposed kernel's unpack, kept out of line so the kernel's epilogue cannot reshape its schedule
template<int D>
static __device__ __noinline__ void fq_unpack_rows_ool(const int bits, const uint8_t * raw, const int data_bytes,
        uint8_t * f16, const int valid) {
    fq_unpack_rows<D>(bits, raw, data_bytes, f16, valid);
}

static __device__ __forceinline__ void fq_mma_f16_k8(half2 * d, const half2 a0, const half2 a1, const uint32_t b) {
#ifdef TURING_MMA_AVAILABLE
    uint32_t * di = (uint32_t *) d;
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(di[0]), "+r"(di[1])
        : "r"(*(const uint32_t *) &a0), "r"(*(const uint32_t *) &a1), "r"(b));
#else
    GGML_UNUSED_VARS(d, a0, a1, b);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

static __device__ __forceinline__ void fq_ldmatrix_x4_trans(uint32_t * r, const void * smem_row) {
#ifdef TURING_MMA_AVAILABLE
    const unsigned int addr = ggml_cuda_cvta_generic_to_shared((void *) smem_row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
#else
    GGML_UNUSED_VARS(r, smem_row);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

// Each warp holds 16 Q columns (one m tile) against FQ_TILE/ksplit keys of every tile; its partial result is one
// split of the workspace: per Q column, D unnormalized output values, then the running max (log2) and sum.
template<int D, int ncols_pad>
static __global__ void __launch_bounds__(FQ_WARPS*WARP_SIZE, 2) flash_attn_quant_decode(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, float * __restrict__ part,
        const float scale_log2, const int rows, const int gqa,
        const int nb01, const int nb02, const int64_t nb03,
        const int nb11, const int nb12, const int64_t nb13,
        const int nb21, const int nb22, const int64_t nb23,
        const bool Q_bf16, const fattn_layout lay) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    constexpr int m_tiles   = ncols_pad/16;
    constexpr int ksplit    = FQ_WARPS/m_tiles;
    constexpr int keys_warp = FQ_TILE/ksplit;      // 8, 16 or 32
    constexpr int n_key     = keys_warp/8;         // 8-key n tiles a warp scores
    using S = fq_smem<D>;
    const int raw_slot = fq_raw_slot(D, lay.k_bits, lay.v_bits);

    const int split    = blockIdx.x;
    const int splits   = gridDim.x;
    const int kv_head  = blockIdx.y;
    const int sequence = blockIdx.z;
    const int warp     = threadIdx.y;
    const int lane     = threadIdx.x;
    const int mt       = warp / ksplit;
    const int ks_i     = warp % ksplit;
    const int ncols    = rows*gqa;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t * k16 = smem;
    uint8_t * v16 = k16 + S::f16_bytes;
    uint8_t * raw = v16 + S::f16_bytes;

    const fattn_kv_rows rk = fattn_sequence_rows(lay, sequence, false);
    const fattn_kv_rows rv = fattn_sequence_rows(lay, sequence, true);
    const int kv_len = rk.len;
    const int ntiles = (kv_len + FQ_TILE - 1) / FQ_TILE;
    const int live = fq_live_splits(kv_len, splits);
    if (split >= live) {
        return;
    }
    const int t0 = int(int64_t(ntiles)*split/live);
    const int t1 = int(int64_t(ntiles)*(split + 1)/live);

    const uint8_t * Kh = (const uint8_t *) K + nb13*sequence + int64_t(nb12)*kv_head;
    const uint8_t * Vh = (const uint8_t *) V + nb23*sequence + int64_t(nb22)*kv_head;
    const int stride_K = nb11/4;
    const int stride_V = nb21/4;

    if (t0 < t1) {
        fq_stage_tile<D>(Kh, Vh, rk, rv, stride_K, stride_V, raw, t0*FQ_TILE);
    }
    fq_cp_async_commit();

    // Q columns g and g + 8 of this warp's m tile as f16 A fragments per 16 dims (m16n8k16: a0/a2 row g, a1/a3 row
    // g + 8, k = 2 t + 0..1 and 8 + 2 t + 0..1), scaled for exp2
    const int g_row = lane / 4;
    const int t_col = lane % 4;
    half2 q[D/16][4];
#pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
        const int j = mt*16 + g_row + 8*half_row;
        const bool j_live = j < ncols;
        const int r = j_live ? j / gqa : 0;
        const int h = j_live ? j - r*gqa : 0;
        const char * q_row = Q + nb03*sequence + int64_t(nb02)*(kv_head*gqa + h) + int64_t(nb01)*r;
#pragma unroll
        for (int kb = 0; kb < D/16; ++kb) {
#pragma unroll
            for (int part_k = 0; part_k < 2; ++part_k) {
                const int d = kb*16 + 8*part_k + 2*t_col;
                float2 x = make_float2(0.0f, 0.0f);
                if (j_live) {
                    x = Q_bf16 ? __bfloat1622float2(*(const nv_bfloat162 *) (q_row + 2*d)) : *(const float2 *) (q_row + 4*d);
                }
                q[kb][2*part_k + half_row] = __floats2half2_rn(x.x*scale_log2, x.y*scale_log2);
            }
        }
    }

    // O for rows g and g + 8 over D/8 n tiles, f16 accumulators as the mma kernel's
    half2 O[D/8][2];
#pragma unroll
    for (int n = 0; n < D/8; ++n) {
        O[n][0] = make_half2(0.0f, 0.0f);
        O[n][1] = make_half2(0.0f, 0.0f);
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};

    const int qp[2] = {
        kv_len - rows + min((mt*16 + g_row    ) / gqa, rows - 1),
        kv_len - rows + min((mt*16 + g_row + 8) / gqa, rows - 1),
    };

    for (int t = t0; t < t1; ++t) {
        uint8_t * slot = raw + (t - t0)%2*raw_slot;
        if (t + 1 < t1) {
            fq_stage_tile<D>(Kh, Vh, rk, rv, stride_K, stride_V, raw + (t + 1 - t0)%2*raw_slot, (t + 1)*FQ_TILE);
        }
        fq_cp_async_commit();
        fq_cp_async_wait_prev();
        __syncthreads();
        fq_unpack_rows<D>(rk.bits, slot, rk.data_bytes, k16, kv_len - t*FQ_TILE);
        fq_unpack_rows<D>(rv.bits, slot + FQ_TILE*fq_raw_stride(D, rk.bits), rv.data_bytes, v16, kv_len - t*FQ_TILE);
        __syncthreads();

        const int key0 = ks_i*keys_warp;
        float s[n_key][4];
#pragma unroll
        for (int n = 0; n < n_key; ++n) {
            tile<16, 8, float> C;
#pragma unroll
            for (int kb = 0; kb < D/16; ++kb) {
                tile<16, 8, half2> A;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    A.x[k] = q[kb][k];
                }
                tile<8, 8, half2> B;
                load_ldmatrix(B, (const half2 *) (k16 + (key0 + 8*n)*S::f16_stride) + 8*kb, S::f16_stride/4);
                mma(C, A, B);
            }
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                s[n][l] = C.x[l];
            }
        }

        // mask, then the online softmax per Q column (rows g and g + 8), reduced over the 4 lanes sharing a row
        float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < n_key; ++n) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int hr = l / 2;
                const int kp = t*FQ_TILE + key0 + 8*n + 2*t_col + (l % 2);
                const bool visible = kp < kv_len && (!lay.causal || kp <= qp[hr]) &&
                    (lay.window_left < 0 || qp[hr] - kp <= lay.window_left);
                s[n][l] = visible ? s[n][l] : -INFINITY;
                tile_max[hr] = fmaxf(tile_max[hr], s[n][l]);
            }
        }
        float corr[2];
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            tile_max[hr] = fmaxf(tile_max[hr], __shfl_xor_sync(0xffffffff, tile_max[hr], 1));
            tile_max[hr] = fmaxf(tile_max[hr], __shfl_xor_sync(0xffffffff, tile_max[hr], 2));
            const float m_new = fmaxf(m_run[hr], tile_max[hr]);
            corr[hr] = m_new == -INFINITY ? 1.0f : exp2f(m_run[hr] - m_new);
            m_run[hr] = m_new;
        }
        float tile_sum[2] = {0.0f, 0.0f};
        half2 P[n_key][2];
#pragma unroll
        for (int n = 0; n < n_key; ++n) {
            float p[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int hr = l / 2;
                p[l] = m_run[hr] == -INFINITY ? 0.0f : exp2f(s[n][l] - m_run[hr]);
                tile_sum[hr] += p[l];
            }
            P[n][0] = __floats2half2_rn(p[0], p[1]);
            P[n][1] = __floats2half2_rn(p[2], p[3]);
        }
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            tile_sum[hr] += __shfl_xor_sync(0xffffffff, tile_sum[hr], 1);
            tile_sum[hr] += __shfl_xor_sync(0xffffffff, tile_sum[hr], 2);
            l_run[hr] = l_run[hr]*corr[hr] + tile_sum[hr];
        }
        const half2 c0 = __float2half2_rn(corr[0]);
        const half2 c1 = __float2half2_rn(corr[1]);
#pragma unroll
        for (int n = 0; n < D/8; ++n) {
            O[n][0] = __hmul2(O[n][0], c0);
            O[n][1] = __hmul2(O[n][1], c1);
        }

        // P (16 columns x 8 keys, the score fragment as is) times V (8 keys x 8 dims, ldmatrix.trans of its rows)
#pragma unroll
        for (int n = 0; n < n_key; ++n) {
            const uint8_t * v_rows = v16 + (key0 + 8*n + lane % 8)*S::f16_stride;
#pragma unroll
            for (int d4 = 0; d4 < D/32; ++d4) {
                uint32_t b[4];
                fq_ldmatrix_x4_trans(b, v_rows + 2*(32*d4 + 8*(lane/8)));
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    fq_mma_f16_k8(O[4*d4 + k], P[n][0], P[n][1], b[k]);
                }
            }
        }
    }
    cp_async_wait_all();

    // this warp's split: rows g and g + 8, dims 8 n + 2 t + 0..1
    const int split_w = split*ksplit + ks_i;
    const int splits_w = splits*ksplit;
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const int j = mt*16 + g_row + 8*hr;
        if (j >= ncols) {
            continue;
        }
        float * dst = part + ((int64_t(sequence)*gridDim.y + kv_head)*ncols + j)*int64_t(splits_w)*(D + 2) +
            int64_t(split_w)*(D + 2);
#pragma unroll
        for (int n = 0; n < D/8; ++n) {
            const float2 o = __half22float2(O[n][hr]);
            *(float2 *) (dst + 8*n + 2*t_col) = o;
        }
        if (t_col == 0) {
            dst[D + 0] = m_run[hr];
            dst[D + 1] = l_run[hr];
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, part, scale_log2, rows, gqa, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23,
        Q_bf16, lay);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

static __device__ __forceinline__ void fq_ldmatrix_x4(uint32_t * r, const void * smem_row) {
#ifdef TURING_MMA_AVAILABLE
    const unsigned int addr = ggml_cuda_cvta_generic_to_shared((void *) smem_row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
#else
    GGML_UNUSED_VARS(r, smem_row);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

static __device__ __forceinline__ uint32_t fq_movmatrix_trans(const uint32_t x) {
#ifdef AMPERE_MMA_AVAILABLE
    uint32_t y;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
#else
    GGML_UNUSED(x);
    NO_DEVICE_CODE;
    return 0;
#endif // AMPERE_MMA_AVAILABLE
}

// m16n8k16, f16 inputs; f32 or f16 accumulators
static __device__ __forceinline__ void fq_mma_f32_k16(float * c, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void fq_mma_f16_k16(uint32_t * c, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// Transposed: K and V are the 16-row operand and Q the 8-wide one, so 48 columns pad to 48, not 64. A warp holds 16
// keys of every tile (km) against nw n tiles: two key halves x n_tiles/nw groups spread the columns over all schedulers.
template<int D, int n_tiles, int nw>
static __global__ void __launch_bounds__(2*n_tiles/nw*WARP_SIZE, 2) flash_attn_quant_decode_t(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, float * __restrict__ part,
        const float scale_log2, const int rows, const int gqa,
        const int nb01, const int nb02, const int64_t nb03,
        const int nb11, const int nb12, const int64_t nb13,
        const int nb21, const int nb22, const int64_t nb23,
        const bool Q_bf16, const fattn_layout lay) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    static_assert(n_tiles % nw == 0, "n tiles split evenly over a key half's warps");
    constexpr int groups = n_tiles/nw;
    using S = fq_smem<D>;
    const int raw_slot = fq_raw_slot(D, lay.k_bits, lay.v_bits);

    const int split    = blockIdx.x;
    const int splits   = gridDim.x;
    const int kv_head  = blockIdx.y;
    const int sequence = blockIdx.z;
    const int warp     = threadIdx.y;
    const int lane     = threadIdx.x;
    const int km       = warp / groups;
    const int nt0      = (warp % groups)*nw;
    const int ncols    = rows*gqa;
    const int g        = lane / 4;
    const int tq       = lane % 4;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t * raw = smem + S::fixed;

    const fattn_kv_rows rk = fattn_sequence_rows(lay, sequence, false);
    const fattn_kv_rows rv = fattn_sequence_rows(lay, sequence, true);
    const int kv_len = rk.len;
    const int ntiles = (kv_len + FQ_TILE - 1) / FQ_TILE;
    const int live = fq_live_splits(kv_len, splits);
    if (split >= live) {
        return;
    }
    const int t0 = int(int64_t(ntiles)*split/live);
    const int t1 = int(int64_t(ntiles)*(split + 1)/live);

    const uint8_t * Kh = (const uint8_t *) K + nb13*sequence + int64_t(nb12)*kv_head;
    const uint8_t * Vh = (const uint8_t *) V + nb23*sequence + int64_t(nb22)*kv_head;
    const int stride_K = nb11/4;
    const int stride_V = nb21/4;
    const int v_raw = FQ_TILE*fq_raw_stride(D, rk.bits);

    if (t0 < t1) {
        fq_stage_tile<D>(Kh, Vh, rk, rv, stride_K, stride_V, raw, t0*FQ_TILE);
    }
    fq_cp_async_commit();

    // Q column g of each n tile as B fragments per 16 dims: b0 dims 2 tq + 0..1, b1 dims 8 + 2 tq + 0..1
    uint32_t q[nw][D/16][2];
#pragma unroll
    for (int i = 0; i < nw; ++i) {
        const int j = (nt0 + i)*8 + g;
        const bool j_live = j < ncols;
        const int r = j_live ? j / gqa : 0;
        const int h = j_live ? j - r*gqa : 0;
        const char * q_row = Q + nb03*sequence + int64_t(nb02)*(kv_head*gqa + h) + int64_t(nb01)*r;
#pragma unroll
        for (int kb = 0; kb < D/16; ++kb) {
#pragma unroll
            for (int part_k = 0; part_k < 2; ++part_k) {
                const int d = kb*16 + 8*part_k + 2*tq;
                float2 x = make_float2(0.0f, 0.0f);
                if (j_live) {
                    x = Q_bf16 ? __bfloat1622float2(*(const nv_bfloat162 *) (q_row + 2*d)) : *(const float2 *) (q_row + 4*d);
                }
                const half2 v = __floats2half2_rn(x.x*scale_log2, x.y*scale_log2);
                q[i][kb][part_k] = *(const uint32_t *) &v;
            }
        }
    }

    // O^T for dims 16 md + g (+ 8) and columns 2 tq + 0..1 of each n tile, f16 accumulators
    uint32_t O[nw][D/16][2];
    float m_run[nw][2];
    float l_run[nw][2];
    int qp[nw][2];
#pragma unroll
    for (int i = 0; i < nw; ++i) {
#pragma unroll
        for (int md = 0; md < D/16; ++md) {
            O[i][md][0] = 0;
            O[i][md][1] = 0;
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            m_run[i][e] = -INFINITY;
            l_run[i][e] = 0.0f;
            qp[i][e] = kv_len - rows + min(((nt0 + i)*8 + 2*tq + e) / gqa, rows - 1);
        }
    }

    const int key0 = 16*km;
    // ldmatrix row addresses: K as A (x4: keys 0-7 / 8-15 by lane bit 3, dims 0-7 / 8-15 by bit 4), V^T as A
    // (.trans: keys 0-7 / 8-15 by bit 4, dims 0-7 / 8-15 by bit 3)
    const int li = lane / 8;
    const int lr = lane % 8;
    const int k_off = (key0 + lr + 8*(li % 2))*S::f16_stride + 2*8*(li / 2);
    const int v_off = S::f16_bytes + (key0 + lr + 8*(li / 2))*S::f16_stride + 2*8*(li % 2);

    for (int t = t0; t < t1; ++t) {
        uint8_t * slot = raw + (t - t0)%2*raw_slot;
        if (t + 1 < t1) {
            fq_stage_tile<D>(Kh, Vh, rk, rv, stride_K, stride_V, raw + (t + 1 - t0)%2*raw_slot, (t + 1)*FQ_TILE);
        }
        fq_cp_async_commit();
        fq_cp_async_wait_prev();
        __syncthreads();
        fq_unpack_rows_ool<D>(rk.bits, slot, rk.data_bytes, smem, kv_len - t*FQ_TILE);
        fq_unpack_rows_ool<D>(rv.bits, slot + v_raw, rv.data_bytes, smem + S::f16_bytes, kv_len - t*FQ_TILE);
        __syncthreads();
        const uint8_t * k_addr = smem + k_off;
        const uint8_t * v_addr = smem + v_off;

        // S^T: keys g, g + 8 against columns 2 tq + 0..1
        float s[nw][4];
#pragma unroll
        for (int i = 0; i < nw; ++i) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                s[i][l] = 0.0f;
            }
        }
#pragma unroll
        for (int kb = 0; kb < D/16; ++kb) {
            uint32_t a[4];
            fq_ldmatrix_x4(a, k_addr + 2*16*kb);
#pragma unroll
            for (int i = 0; i < nw; ++i) {
                fq_mma_f32_k16(s[i], a, q[i][kb][0], q[i][kb][1]);
            }
        }

        uint32_t pb[nw][2];
#pragma unroll
        for (int i = 0; i < nw; ++i) {
            float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int e = l % 2;
                const int kp = t*FQ_TILE + key0 + g + 8*(l / 2);
                const bool visible = kp < kv_len && (!lay.causal || kp <= qp[i][e]) &&
                    (lay.window_left < 0 || qp[i][e] - kp <= lay.window_left);
                s[i][l] = visible ? s[i][l] : -INFINITY;
                tile_max[e] = fmaxf(tile_max[e], s[i][l]);
            }
            float corr[2];
            float tile_sum[2] = {0.0f, 0.0f};
#pragma unroll
            for (int e = 0; e < 2; ++e) {
#pragma unroll
                for (int x = 4; x < WARP_SIZE; x *= 2) {
                    tile_max[e] = fmaxf(tile_max[e], __shfl_xor_sync(0xffffffff, tile_max[e], x));
                }
                const float m_new = fmaxf(m_run[i][e], tile_max[e]);
                corr[e] = m_new == -INFINITY ? 1.0f : exp2f(m_run[i][e] - m_new);
                m_run[i][e] = m_new;
            }
            float p[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int e = l % 2;
                p[l] = m_run[i][e] == -INFINITY ? 0.0f : exp2f(s[i][l] - m_run[i][e]);
                tile_sum[e] += p[l];
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
#pragma unroll
                for (int x = 4; x < WARP_SIZE; x *= 2) {
                    tile_sum[e] += __shfl_xor_sync(0xffffffff, tile_sum[e], x);
                }
                l_run[i][e] = l_run[i][e]*corr[e] + tile_sum[e];
            }
            const half2 c2 = __floats2half2_rn(corr[0], corr[1]);
#pragma unroll
            for (int md = 0; md < D/16; ++md) {
                *(half2 *) &O[i][md][0] = __hmul2(*(const half2 *) &O[i][md][0], c2);
                *(half2 *) &O[i][md][1] = __hmul2(*(const half2 *) &O[i][md][1], c2);
            }
            // P^T as B: each 8x8 score block (keys g, columns 2 tq) transposed to (keys 2 tq, column g)
            const half2 p0 = __floats2half2_rn(p[0], p[1]);
            const half2 p1 = __floats2half2_rn(p[2], p[3]);
            pb[i][0] = fq_movmatrix_trans(*(const uint32_t *) &p0);
            pb[i][1] = fq_movmatrix_trans(*(const uint32_t *) &p1);
        }

#pragma unroll
        for (int md = 0; md < D/16; ++md) {
            uint32_t a[4];
            fq_ldmatrix_x4_trans(a, v_addr + 2*16*md);
#pragma unroll
            for (int i = 0; i < nw; ++i) {
                fq_mma_f16_k16(O[i][md], a, pb[i][0], pb[i][1]);
            }
        }
    }
    cp_async_wait_all();

    // the key halves merge through smem (the tiles are done), so each block writes one split
    constexpr int words = nw*(D/16)*2 + nw*4;
    uint32_t * xchg = (uint32_t *) smem + ((warp % groups)*WARP_SIZE + lane)*words;
    __syncthreads();
    if (km == 1) {
#pragma unroll
        for (int i = 0; i < nw; ++i) {
#pragma unroll
            for (int md = 0; md < D/16; ++md) {
                xchg[(i*(D/16) + md)*2 + 0] = O[i][md][0];
                xchg[(i*(D/16) + md)*2 + 1] = O[i][md][1];
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                xchg[nw*(D/16)*2 + i*4 + e    ] = __float_as_uint(m_run[i][e]);
                xchg[nw*(D/16)*2 + i*4 + e + 2] = __float_as_uint(l_run[i][e]);
            }
        }
    }
    __syncthreads();
    if (km == 1) {
        return;
    }

    // this block's split: columns 2 tq + 0..1 of each n tile, dims 16 md + g (+ 8)
#pragma unroll
    for (int i = 0; i < nw; ++i) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int j = (nt0 + i)*8 + 2*tq + e;
            if (j >= ncols) {
                continue;
            }
            const float m1 = __uint_as_float(xchg[nw*(D/16)*2 + i*4 + e]);
            const float l1 = __uint_as_float(xchg[nw*(D/16)*2 + i*4 + e + 2]);
            const float m = fmaxf(m_run[i][e], m1);
            const float w0 = m == -INFINITY ? 0.0f : exp2f(m_run[i][e] - m);
            const float w1 = m == -INFINITY ? 0.0f : exp2f(m1 - m);
            float * dst = part + ((int64_t(sequence)*gridDim.y + kv_head)*ncols + j)*int64_t(splits)*(D + 2) +
                int64_t(split)*(D + 2);
#pragma unroll
            for (int md = 0; md < D/16; ++md) {
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    const uint32_t o1 = xchg[(i*(D/16) + md)*2 + hr];
                    dst[16*md + g + 8*hr] = w0*__half2float(((const half *) &O[i][md][hr])[e]) +
                        w1*__half2float(((const half *) &o1)[e]);
                }
            }
            if (g == 0) {
                dst[D + 0] = m;
                dst[D + 1] = l_run[i][e]*w0 + l1*w1;
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, part, scale_log2, rows, gqa, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23,
        Q_bf16, lay);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// Merges a Q column's splits and writes its row of dst ([sequence*rows + row][head][D], f32 or bf16).
template<int D>
static __global__ void flash_attn_quant_combine(
        const float * __restrict__ part, void * __restrict__ dst, const int splits, const int ksplit, const int rows,
        const int gqa, const int kv_heads, const int n_head, const bool dst_bf16, const fattn_layout lay) {
    const int col = blockIdx.x;                     // ((sequence*kv_heads + kv_head)*ncols + j)
    const int ncols = rows*gqa;
    const int j = col % ncols;
    const int kv_head = (col / ncols) % kv_heads;
    const int sequence = col / (ncols*kv_heads);
    const float * p = part + int64_t(col)*splits*ksplit*(D + 2);
    const int live = fq_live_splits(fattn_sequence_rows(lay, sequence, false).len, splits)*ksplit;
    float m = -INFINITY;
    for (int s = 0; s < live; ++s) {
        m = fmaxf(m, p[int64_t(s)*(D + 2) + D]);
    }
    const int r = j / gqa;
    const int h = j - r*gqa;
    const int64_t out = ((int64_t(sequence)*rows + r)*n_head + kv_head*gqa + h)*D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        float l = 0.0f;
        if (m != -INFINITY) {
            for (int s = 0; s < live; ++s) {
                const float * ps = p + int64_t(s)*(D + 2);
                const float w = exp2f(ps[D] - m);
                acc += w*ps[d];
                l += w*ps[D + 1];
            }
        }
        fattn_store_dst(dst, dst_bf16, out + d, l > 0.0f ? acc/l : 0.0f);
    }
}

// Runs the quantized decode kernel when the call is one it serves; false leaves the call to the mma kernel.
static bool ggml_cuda_flash_attn_ext_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const fattn_layout lay = fattn_get_layout(dst);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));

    const int D = Q->ne[0];
    const int rows = Q->ne[1];
    const int gqa = Q->ne[2] / K->ne[2];
    if (!lay.k_bits || !lay.block_table || lay.cu_q || lay.cu_kv || lay.chunk > 0 || sinks || softcap != 0.0f ||
            max_bias != 0.0f || (mask && !lay.implicit_mask) || cc < GGML_CUDA_CC_AMPERE || (D != 128 && D != 256) ||
            lay.quant_head_dim != D || rows > FQ_MAX_ROWS || rows*gqa > FQ_MAX_COLS) {
        return false;
    }

    const int n_seq    = Q->ne[3];
    const int kv_heads = K->ne[2];
    const int ncols    = rows*gqa;
    // above 32 columns the m-tile kernel pads to 64, the transposed one to 8*FQ_T_TILES
    const bool transposed = ncols > 32 && ncols <= 8*FQ_T_TILES;
    const int ncols_pad = ncols <= 16 ? 16 : ncols <= 32 ? 32 : 64;
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const float log2e = 1.4426950408889634f;
    const bool Q_bf16 = Q->type == GGML_TYPE_BF16;
    const bool dst_bf16 = dst->type == GGML_TYPE_BF16;
    cudaStream_t stream = ctx.stream();

    // NC: the m-tile kernel's padded columns; T: the transposed kernel
    auto launch = [&](auto d_c, auto ncols_c, auto t_c) {
        constexpr int DD = decltype(d_c)::value;
        constexpr int NC = decltype(ncols_c)::value;
        constexpr bool T = decltype(t_c)::value;
        constexpr int warps  = T ? 2*FQ_T_TILES/FQ_T_TILES_WARP : FQ_WARPS;
        constexpr int ksplit = T ? 1 : FQ_WARPS/(NC/16);
        auto kernel = [] {
            if constexpr (T) {
                return flash_attn_quant_decode_t<DD, FQ_T_TILES, FQ_T_TILES_WARP>;
            } else {
                return flash_attn_quant_decode<DD, NC>;
            }
        }();
        const int smem = fq_smem<DD>::fixed + 2*fq_raw_slot(DD, lay.k_bits, lay.v_bits);
        // the raw slots follow the bits, so the limit follows the largest launch seen
        static int raised[GGML_CUDA_MAX_DEVICES] = {0};
        if (raised[ctx.device] < smem) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            raised[ctx.device] = smem;
        }
        // one whole wave: device-side lengths hide the work per block, and a partial second wave idles most of the GPU
        static int occ_smem[GGML_CUDA_MAX_DEVICES] = {0};
        static int per_sm[GGML_CUDA_MAX_DEVICES] = {0};
        if (occ_smem[ctx.device] != smem) {
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm[ctx.device], kernel, WARP_SIZE*warps, smem));
            occ_smem[ctx.device] = smem;
        }
        const int splits = std::max(1, std::max(1, per_sm[ctx.device])*nsm / std::max(1, kv_heads*n_seq));
        ggml_cuda_pool_alloc<float> part(ctx.pool(), size_t(n_seq)*kv_heads*ncols*splits*ksplit*(D + 2));
        const dim3 grid(splits, kv_heads, n_seq);
        const dim3 block(WARP_SIZE, warps);
        kernel<<<grid, block, smem, stream>>>((const char *) Q->data, (const char *) K->data, (const char *) V->data,
            part.ptr, scale*log2e, rows, gqa, Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3],
            V->nb[1], V->nb[2], V->nb[3], Q_bf16, lay);
        CUDA_CHECK(cudaGetLastError());
        flash_attn_quant_combine<DD><<<n_seq*kv_heads*ncols, DD/2, 0, stream>>>(
            part.ptr, dst->data, splits, ksplit, rows, gqa, kv_heads, Q->ne[2], dst_bf16, lay);
        CUDA_CHECK(cudaGetLastError());
    };
    using c128 = std::integral_constant<int, 128>;
    using c256 = std::integral_constant<int, 256>;
    using n16 = std::integral_constant<int, 16>;
    using n32 = std::integral_constant<int, 32>;
    using n64 = std::integral_constant<int, 64>;
    using mtile = std::false_type;
    using trans = std::true_type;
    if (D == 128) {
        transposed ? launch(c128{}, n64{}, trans{}) : ncols_pad == 16 ? launch(c128{}, n16{}, mtile{}) :
            ncols_pad == 32 ? launch(c128{}, n32{}, mtile{}) : launch(c128{}, n64{}, mtile{});
    } else {
        transposed ? launch(c256{}, n64{}, trans{}) : ncols_pad == 16 ? launch(c256{}, n16{}, mtile{}) :
            ncols_pad == 32 ? launch(c256{}, n32{}, mtile{}) : launch(c256{}, n64{}, mtile{});
    }
    return true;
}
