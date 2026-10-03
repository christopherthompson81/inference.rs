// mmq tile loaders for ik_llama.cpp's IQK types (block-16 tiles for IQ*_K, Q8_0 tiles for the _KS / _KSS / _KL types).
// Copied from ik_llama.cpp (MIT, Iwan Kawrakow) template-instances/mmq-instance-iq*_k*_id.cu; rows use byte strides.
#pragma once

#include "mmq_ik_common.cuh"

typedef struct { half d; uint16_t extra; uint8_t scales[QK_K/32]; uint8_t qs[QK_K/4]; } block_iq2_k;
typedef struct { half d; uint16_t extra; uint16_t scales_h; uint8_t scales_l[QK_K/32]; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/8]; } block_iq3_k;
typedef struct { half d; uint16_t extra; uint8_t scales_h[QK_K/64]; uint8_t scales_l[QK_K/32]; uint8_t qs[QK_K/2]; } block_iq4_k;
typedef struct { half d; uint16_t extra; uint8_t scales_h[QK_K/64]; uint8_t scales_l[QK_K/32]; uint8_t qs[QK_K/2]; uint8_t qh[QK_K/8]; } block_iq5_k;
typedef struct { half d; uint16_t extra; int8_t scales[QK_K/16]; uint8_t qs[QK_K/2]; uint8_t qh[QK_K/4]; } block_iq6_k;
typedef struct { uint16_t extra; uint8_t scales[QK_K/64]; uint8_t qs[QK_K/4]; } block_iq2_ks;
typedef struct { uint16_t extra; uint8_t scales[QK_K/64]; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/8]; } block_iq3_ks;
typedef struct { uint8_t scales[QK_K/32]; uint8_t qs[QK_K/2]; } block_iq4_ks;
typedef struct { uint32_t qs[QK_K/8]; } block_iq4_kss;
typedef struct { uint8_t scales[QK_K/32]; uint8_t qs[QK_K/2]; uint8_t qh[QK_K/8]; } block_iq5_ks;
typedef struct { uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/16]; } block_iq2_kl;

// mmq-instance-iq2_k_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = mmq_get_nwarps_device();
    //constexpr int warp_size = ggml_cuda_get_physical_warp_size();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_k * bxi = (const block_iq2_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

#ifdef __CUDA_ARCH__

        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {
            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 2) & 0x44444444);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 0) & 0x44444444);
            int2 v1 = get_int_from_table_8(val1, iq2nl_values);
            int2 v2 = get_int_from_table_8(val2, iq2nl_values);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#else

        auto all_values = (const int *)iq2k_table;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  0] = int_from_table_4((ql >> 0) & 0x03030303, all_values + ((extra & 0x01) << 8));
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  8] = int_from_table_4((ql >> 2) & 0x03030303, all_values + ((extra & 0x04) << 6));
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 16] = int_from_table_4((ql >> 4) & 0x03030303, all_values + ((extra & 0x10) << 4));
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 24] = int_from_table_4((ql >> 6) & 0x03030303, all_values + ((extra & 0x40) << 2));
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = int_from_table_4((ql >> 0) & 0x03030303, all_values + ((extra & 0x01) << 8));
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = int_from_table_4((ql >> 2) & 0x03030303, all_values + ((extra & 0x04) << 6));
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = int_from_table_4((ql >> 4) & 0x03030303, all_values + ((extra & 0x10) << 4));
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = int_from_table_4((ql >> 6) & 0x03030303, all_values + ((extra & 0x40) << 2));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

            extra >>= 8;
        }
#endif // __CUDA_ARCH__

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * (((bxi->scales[kqsx] >> 0) & 0xf) - 8);
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * (((bxi->scales[kqsx] >> 4) & 0xf) - 8);
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * (((bxi->scales[kqsx] >> 0) & 0xf) - 8);
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * (((bxi->scales[kqsx] >> 4) & 0xf) - 8);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mmq-instance-iq3_k_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq3_k * bxi = (const block_iq3_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;

        uint16_t extra = bxi->extra >> (kqsx/4);
        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        int qh = get_int_b2(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            //extra << 3, extra << 1, extra >> 1, extra >> 3
            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 3) & 0x88888888)
                          | ((qh << 2) & 0x04040404) | ((qh << 4) & 0x40404040);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 1) & 0x88888888)
                          | ((qh << 1) & 0x04040404) | ((qh << 3) & 0x40404040);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            qh >>= 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        uint16_t sh = bxi->scales_h >> 2*kqsx;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mmq-instance-iq4_k_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq4_k * bxi = (const block_iq4_k *)(x + i*stride) + kbx0;
        const uint16_t extra = bxi->extra >> 2*kqsx;

        // iq4k_values high half is the low half + 4, so the half select is a packed byte add.
        const int add_l = 0x04040404 & -( extra       & 1);
        const int add_h = 0x04040404 & -((extra >> 1) & 1);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int q4 = get_int_b4(bxi->qs, (qstep/2)*kqsx + l);

            const int2 v = get_int_from_table_16(q4, iq4k_values);
            const int val0 = v.x + add_l;
            const int val1 = v.y + add_h;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + l + 0] = val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + l + 4] = val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + l + 0] = val0;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + l + 4] = val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const uint8_t sh = bxi->scales_h[kqsx/2] >> 4*(kqsx%2);
        const int ls1 = ((bxi->scales_l[kqsx] & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = ((bxi->scales_l[kqsx] >>  4) | ((sh << 2) & 0x30)) - 32;

        const float d = bxi->d;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * ls1;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * ls2;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ls1;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ls2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mmq-instance-iq5_k_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq5_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq5nl_values;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq5_k * bxi = (const block_iq5_k *)(x + i*stride) + kbx0;

        int qh = get_int_b4(bxi->qh, kqsx);
        uint16_t extra = bxi->extra >> (kqsx/4);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((extra & 1) * 0x20202020); // this is very slightly faster
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((extra & 4) * 0x08080808); // then the version below
            //aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((extra & 1) ? 0x20202020 : 0);
            //aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((extra & 4) ? 0x20202020 : 0);
            qh    >>= 2;
            extra >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const uint8_t sh = bxi->scales_h[kqsx/2] >> 4*(kqsx%2);
        const int ls1 = ((bxi->scales_l[kqsx] & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = ((bxi->scales_l[kqsx] >>  4) | ((sh << 2) & 0x30)) - 32;

        const float d = bxi->d;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * ls1;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * ls2;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ls1;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ls2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mmq-instance-iq6_k_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq6_k(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
    int   * x_sc = (int   *) (x_df + WARP_SIZE/QI6_K);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq6nl_values;
    int qh[2];

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq6_k * bxi = (const block_iq6_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

        qh[0] = get_int_b4(bxi->qh, kqsx+0);
        qh[1] = get_int_b4(bxi->qh, kqsx+8);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh[l/2] & 0x03030303) << 4) | ((extra & 1) * 0x40404040);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh[l/2] & 0x0c0c0c0c) << 2) | ((extra & 4) * 0x10101010);
            qh[l/2] >>= 4;
            extra   >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q6_K + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }


#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        ((uint16_t *)(x_sc + i*MMQ_MMA_TILE_X_K_Q6_K))[kqsx] = ((const uint16_t *) bxi->scales)[kqsx];
        x_df[i*MMQ_MMA_TILE_X_K_Q6_K] = d;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mmq-instance-iq2_ks_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x%16;

#ifdef __CUDA_ARCH__
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = (const block_iq2_ks *)(x + i*stride + sizeof(half)) + kbx0;

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

        uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;
        uint32_t val1 = ((q2 >> 0) & 0x33333333) | ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
        uint32_t val2 = ((q2 >> 2) & 0x33333333) | ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
        int2 v1 = get_int_from_table_8(val1, iq2nl_values);
        int2 v2 = get_int_from_table_8(val2, iq2nl_values);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#else // __CUDA_ARCH__


    const int * all_values = (const int *)iq2k_table;
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = (const block_iq2_ks *)(x + i*stride + sizeof(half)) + kbx0;

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
#endif // __CUDA_ARCH__

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = i0 + threadIdx.y * 8 + threadIdx.x / 4;

        if (need_check) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = dptr[0];
        const block_iq2_ks * bxi = (const block_iq2_ks *)(dptr + 1) + kbx0;
        const int ls1 = ((bxi->scales[threadIdx.x % 4] >> 0) & 0xf) | ((bxi->extra >> (4 + 2*(threadIdx.x % 4))) & 0x10);
        const int ls2 = ((bxi->scales[threadIdx.x % 4] >> 4) & 0xf) | ((bxi->extra >> (5 + 2*(threadIdx.x % 4))) & 0x10);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*(WARP_SIZE/4) + i/4   + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mmq-instance-iq3_ks_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = __half2float(dptr[0]);
        const block_iq3_ks * bxi = (const block_iq3_ks *)(dptr + 1) + kbx0;

        //uint16_t extra = bxi->extra >> 8;
        int qh = get_int_b2(bxi->qh, kqsx);

        uint32_t extra32 = uint32_t(bxi->extra >> 8) * 0x01010101;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((qh << 2) & 0x04040404) | ((extra32 << 3) & 0x08080808)
                                                     | ((qh << 4) & 0x40404040) | ((extra32 << 5) & 0x80808080);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((qh << 1) & 0x04040404) | ((extra32 << 2) & 0x08080808)
                                                     | ((qh << 3) & 0x40404040) | ((extra32 << 4) & 0x80808080);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            extra32 >>= 4;
            qh      >>= 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mmq-instance-iq4_ks_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_kss(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x / 4;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_kss * bxi = (const block_iq4_kss *)(dptr + 1) + kbx0;
        const uint32_t * q4 = bxi->qs + 4*kqsx;
        uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
        uint8_t ls = (s32 | (s32 >> 15)) & 0xff;

        auto values = iq4k_values + ((ls & 1) << 4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t val = q4[j] & 0xfffefffe;
            val = val ^ (val >> 1);
            auto v = get_int_from_table_16(val, values);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 4] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[0] * ((ls & 254) - 127);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[0] * ((ls & 254) - 127);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

}

// mmq-instance-iq4_ks_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x / 4;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_ks * bxi = (const block_iq4_ks *)(dptr + 1) + kbx0;
        const int ls = (bxi->scales[kqsx] & 254) - 127;

        auto values = iq4k_values + ((bxi->scales[kqsx] & 1) << 4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q4 = get_int_b4(bxi->qs, 4*kqsx+j);
            const int2 v = get_int_from_table_16(q4, values);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 4] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[0] * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[0] * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

}

// mmq-instance-iq5_ks_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq5_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ5_KS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq5nl_values;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0];
        const block_iq5_ks * bxi = (const block_iq5_ks *)(dptr + 1) + kbx0;

        int qh = get_int_b4(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((bxi->scales[2*l+0] & 1) * 0x20202020);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((bxi->scales[2*l+1] & 1) * 0x20202020);
            qh    >>= 2;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0               + kqsx] = d * ((bxi->scales[kqsx] & 254) - 127);
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + kqsx] = d * ((bxi->scales[kqsx] & 254) - 127);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// mmq-instance-iq2_kl_id.cu
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_kl(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x/4;

    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = *dptr;
        const block_iq2_kl * bxi = (const block_iq2_kl *)(dptr + 1) + kbx0;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            auto ql = get_int_b2(bxi->qs, 4*(kqsx/2) + 2*(kqsx%2) + j);
            auto qh = get_int_b2(bxi->qh, 2*(kqsx%2) + j) >> 2*(kqsx/2);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh << 4) & 0x10101010);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh << 3) & 0x10101010);
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int val1 = iq2kl_values[a8[2*l+0]] | (iq2kl_values[a8[2*l+1]] << 16);
                int val2 = iq2kl_values[a8[2*l+4]] | (iq2kl_values[a8[2*l+5]] << 16);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
                x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 0] = val1;
                x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 8] = val2;
#else
                x_qs[i*(2*WARP_SIZE + 1)     + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 0] = val1;
                x_qs[i*(2*WARP_SIZE + 1)     + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 8] = val2;
#endif
            }
        }

        int ls = int(((bxi->scales_l[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->scales_h >> 2*kqsx) & 3) << 4)) - 32;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = d * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = d * ls;
#endif
    }

}


template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ2_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_k<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_k<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_k<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ5_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq5_k<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ6_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq6_k<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q6_K_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ2_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_ks<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_ks<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_ks<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_KSS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_kss<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ5_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq5_ks<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_IQ2_KL> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_kl<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};
