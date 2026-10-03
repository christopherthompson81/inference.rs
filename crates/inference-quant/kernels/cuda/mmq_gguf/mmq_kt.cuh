// mmq tile loaders for ik_llama.cpp's trellis types; they expand trellis values into Q8_0-style int8 tiles.
// Copied from ik_llama.cpp (MIT, Iwan Kawrakow) template-instances/mmq-instance-iq*_kt_id.cu; rows use byte strides.
#pragma once

#include "mmq_ik_common.cuh"

typedef struct { uint8_t sh[QK_K / 32]; uint8_t ql[QK_K / 8]; uint8_t qh[QK_K / 16]; } block_iq1_kt;
typedef struct { uint8_t scales[QK_K / 64]; uint8_t ql[QK_K / 4]; } block_iq2_kt;
typedef struct { uint8_t scales[QK_K / 64]; uint8_t ql[QK_K / 4]; uint8_t qh[QK_K / 8]; } block_iq3_kt;
typedef struct { uint32_t qs[QK_K / 8]; } block_iq4_kt;

// mmq-instance-iq1_kt_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq1_kt(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq1_kt * bxi = (const block_iq1_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        uint32_t val = bxi->ql[kqsx] + ((bxi->qh[kqsx%16] << (8 - 4*(kqsx/16))) & 0xf00) + ((bxi->sh[kqsx/4] << (8 - (kqsx%4))) & 0x1000) + 4096;
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.x |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.y |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0];
        const block_iq1_kt * bxi = (const block_iq1_kt *)(dptr + 1) + kbx0;
        const int ls = iq4k_values[bxi->sh[threadIdx.x % 8] & 0xf];

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ1_KT> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq1_kt<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

// mmq-instance-iq2_kt_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_kt(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_kt * bxi = (const block_iq2_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto ql = (const uint16_t *)bxi->ql;
        uint32_t val = ql[4*ib32+j] + 4096;
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.x |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.y |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0] * 1.05f;
        const block_iq2_kt * bxi = (const block_iq2_kt *)(dptr + 1) + kbx0;
        int ib32 = threadIdx.x % 8;
        const int ls = iq4k_values[(bxi->scales[ib32%4] >> 4*(ib32/4)) & 0xf];

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ2_KT> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_kt<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

// mmq-instance-iq3_kt_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_kt(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_kt * bxi = (const block_iq3_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto ql = (const uint16_t *)bxi->ql;
        const auto qh = (const uint32_t *)bxi->qh;
        uint32_t mask = 0x01010101 << ib32;
        uint32_t val = ql[4*ib32+j] + 4096;
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.x |= abs(ggml_cuda_dp4a(val & km, 0x01010101, -126)) << 8*k;
        }
        auto signs = __vcmpne4(qh[2*j+0] & mask, 0);
        v.x = __vsub4(v.x ^ signs, signs);
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.y |= abs(ggml_cuda_dp4a(val & km, 0x01010101, -126)) << 8*k;
        }
        signs = __vcmpne4(qh[2*j+1] & mask, 0);
        v.y = __vsub4(v.y ^ signs, signs);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0] * 1.01f;
        const block_iq3_kt * bxi = (const block_iq3_kt *)(dptr + 1) + kbx0;
        int ib32 = threadIdx.x % 8;
        const int ls = (bxi->scales[ib32%4] >> 4*(ib32/4)) & 0xf;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_KT> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_kt<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

// mmq-instance-iq4_kt_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_kt(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_kt * bxi = (const block_iq4_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto shb = bxi->qs;
        const auto ql  = (const uint8_t *)(shb + 8);
        const auto qh  = ql + 64;
        const uint32_t sh = shb[ib32] >> (8 + 6*j);
        uint32_t offset = 4096 + ((shb[ib32] & 1) << 15);
        uint32_t val1 = offset + ql[8*ib32+2*j+0] + ((qh[8*(ib32%4)+2*j+0] << (8 - 4*(ib32/4))) & 0xf00) + ((sh & 7) << 12);
        uint32_t val2 = offset + ql[8*ib32+2*j+1] + ((qh[8*(ib32%4)+2*j+1] << (8 - 4*(ib32/4))) & 0xf00) + ((sh & 56) << 9);
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val1 *= ka;
            val2 *= ka;
            v.x |= (ggml_cuda_dp4a(val1 & km, 0x01010101, -126) & 0xff) << 8*k;
            v.y |= (ggml_cuda_dp4a(val2 & km, 0x01010101, -126) & 0xff) << 8*k;
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_kt * bxi = (const block_iq4_kt *)(dptr + 1) + kbx0;
        const int ls = (bxi->qs[threadIdx.x % 8] & 0xff) >> 1;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = dptr[0] * (ls - 64);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = dptr[0] * (ls - 64);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_KT> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_kt<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};
