// Value tables and helpers shared by the mmq loaders for ik_llama.cpp's types.
// Copied from ik_llama.cpp (MIT, Iwan Kawrakow): ggml-common.h and ggml-cuda/iqk_cuda_common.h.
#pragma once

#include "mmq_gguf.cuh"

static const __device__ int8_t iq2nl_values[8] = {-31, -13, 1, 17, -26, -8, 6, 22};
static const __device__ uint16_t iq2kl_values[32] = {0xe9c1, 0x0dc1, 0xc1d8, 0xf6d8, 0x0dd8, 0x2fd8, 0xd8e9, 0xe9e9, 0x01e9, 0x0de9, 0x1ce9, 0xc1f6, 0x01f6, 0x0df6, 0x2ff6, 0xe901, 0xf601, 0x0101, 0x0d01, 0x1c01, 0xd80d, 0xe90d, 0xf60d, 0x010d, 0x0d0d, 0xc11c, 0xe91c, 0x011c, 0x1c1c, 0x2f1c, 0xe92f, 0x0d2f,};
static const __device__ int8_t iq3nl_values[16] = {-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,};
static const __device__ int8_t iq4k_values[32] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, -123, -100, -79, -61, -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117};
static const __device__ int8_t iq5nl_values[64] = {-126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6, -1, 5, 11, 17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121, -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10, -4, 1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89, 99, 111, 123,};
static const __device__ int8_t iq6nl_values[128] = {-127, -121, -115, -109, -104, -98, -93, -88, -84, -79, -74, -70, -66, -62, -58, -54, -51, -47, -44, -40, -37, -34, -31, -28, -25, -22, -19, -16, -13, -11, -8, -5, -2, 0, 3, 6, 9, 12, 14, 17, 20, 23, 27, 30, 33, 36, 40, 44, 47, 51, 55, 59, 63, 68, 72, 77, 82, 87, 92, 98, 103, 109, 115, 121, -126, -120, -114, -108, -103, -97, -92, -87, -83, -78, -73, -69, -65, -61, -57, -53, -50, -46, -43, -39, -36, -33, -30, -27, -24, -21, -18, -15, -12, -10, -7, -4, -1, 1, 4, 7, 10, 13, 15, 18, 21, 24, 28, 31, 34, 37, 41, 45, 48, 52, 56, 60, 64, 69, 73, 78, 83, 88, 93, 99, 104, 110, 116, 122,};

static const __device__ uint32_t iq2k_table[512] = {
    0xe1e1e1e1, 0xe1e1e1f3, 0xe1e1e101, 0xe1e1e111, 0xe1e1f3e1, 0xe1e1f3f3, 0xe1e1f301, 0xe1e1f311,
    0xe1e101e1, 0xe1e101f3, 0xe1e10101, 0xe1e10111, 0xe1e111e1, 0xe1e111f3, 0xe1e11101, 0xe1e11111,
    0xe1f3e1e1, 0xe1f3e1f3, 0xe1f3e101, 0xe1f3e111, 0xe1f3f3e1, 0xe1f3f3f3, 0xe1f3f301, 0xe1f3f311,
    0xe1f301e1, 0xe1f301f3, 0xe1f30101, 0xe1f30111, 0xe1f311e1, 0xe1f311f3, 0xe1f31101, 0xe1f31111,
    0xe101e1e1, 0xe101e1f3, 0xe101e101, 0xe101e111, 0xe101f3e1, 0xe101f3f3, 0xe101f301, 0xe101f311,
    0xe10101e1, 0xe10101f3, 0xe1010101, 0xe1010111, 0xe10111e1, 0xe10111f3, 0xe1011101, 0xe1011111,
    0xe111e1e1, 0xe111e1f3, 0xe111e101, 0xe111e111, 0xe111f3e1, 0xe111f3f3, 0xe111f301, 0xe111f311,
    0xe11101e1, 0xe11101f3, 0xe1110101, 0xe1110111, 0xe11111e1, 0xe11111f3, 0xe1111101, 0xe1111111,
    0xf3e1e1e1, 0xf3e1e1f3, 0xf3e1e101, 0xf3e1e111, 0xf3e1f3e1, 0xf3e1f3f3, 0xf3e1f301, 0xf3e1f311,
    0xf3e101e1, 0xf3e101f3, 0xf3e10101, 0xf3e10111, 0xf3e111e1, 0xf3e111f3, 0xf3e11101, 0xf3e11111,
    0xf3f3e1e1, 0xf3f3e1f3, 0xf3f3e101, 0xf3f3e111, 0xf3f3f3e1, 0xf3f3f3f3, 0xf3f3f301, 0xf3f3f311,
    0xf3f301e1, 0xf3f301f3, 0xf3f30101, 0xf3f30111, 0xf3f311e1, 0xf3f311f3, 0xf3f31101, 0xf3f31111,
    0xf301e1e1, 0xf301e1f3, 0xf301e101, 0xf301e111, 0xf301f3e1, 0xf301f3f3, 0xf301f301, 0xf301f311,
    0xf30101e1, 0xf30101f3, 0xf3010101, 0xf3010111, 0xf30111e1, 0xf30111f3, 0xf3011101, 0xf3011111,
    0xf311e1e1, 0xf311e1f3, 0xf311e101, 0xf311e111, 0xf311f3e1, 0xf311f3f3, 0xf311f301, 0xf311f311,
    0xf31101e1, 0xf31101f3, 0xf3110101, 0xf3110111, 0xf31111e1, 0xf31111f3, 0xf3111101, 0xf3111111,
    0x01e1e1e1, 0x01e1e1f3, 0x01e1e101, 0x01e1e111, 0x01e1f3e1, 0x01e1f3f3, 0x01e1f301, 0x01e1f311,
    0x01e101e1, 0x01e101f3, 0x01e10101, 0x01e10111, 0x01e111e1, 0x01e111f3, 0x01e11101, 0x01e11111,
    0x01f3e1e1, 0x01f3e1f3, 0x01f3e101, 0x01f3e111, 0x01f3f3e1, 0x01f3f3f3, 0x01f3f301, 0x01f3f311,
    0x01f301e1, 0x01f301f3, 0x01f30101, 0x01f30111, 0x01f311e1, 0x01f311f3, 0x01f31101, 0x01f31111,
    0x0101e1e1, 0x0101e1f3, 0x0101e101, 0x0101e111, 0x0101f3e1, 0x0101f3f3, 0x0101f301, 0x0101f311,
    0x010101e1, 0x010101f3, 0x01010101, 0x01010111, 0x010111e1, 0x010111f3, 0x01011101, 0x01011111,
    0x0111e1e1, 0x0111e1f3, 0x0111e101, 0x0111e111, 0x0111f3e1, 0x0111f3f3, 0x0111f301, 0x0111f311,
    0x011101e1, 0x011101f3, 0x01110101, 0x01110111, 0x011111e1, 0x011111f3, 0x01111101, 0x01111111,
    0x11e1e1e1, 0x11e1e1f3, 0x11e1e101, 0x11e1e111, 0x11e1f3e1, 0x11e1f3f3, 0x11e1f301, 0x11e1f311,
    0x11e101e1, 0x11e101f3, 0x11e10101, 0x11e10111, 0x11e111e1, 0x11e111f3, 0x11e11101, 0x11e11111,
    0x11f3e1e1, 0x11f3e1f3, 0x11f3e101, 0x11f3e111, 0x11f3f3e1, 0x11f3f3f3, 0x11f3f301, 0x11f3f311,
    0x11f301e1, 0x11f301f3, 0x11f30101, 0x11f30111, 0x11f311e1, 0x11f311f3, 0x11f31101, 0x11f31111,
    0x1101e1e1, 0x1101e1f3, 0x1101e101, 0x1101e111, 0x1101f3e1, 0x1101f3f3, 0x1101f301, 0x1101f311,
    0x110101e1, 0x110101f3, 0x11010101, 0x11010111, 0x110111e1, 0x110111f3, 0x11011101, 0x11011111,
    0x1111e1e1, 0x1111e1f3, 0x1111e101, 0x1111e111, 0x1111f3e1, 0x1111f3f3, 0x1111f301, 0x1111f311,
    0x111101e1, 0x111101f3, 0x11110101, 0x11110111, 0x111111e1, 0x111111f3, 0x11111101, 0x11111111,
    0xe6e6e6e6, 0xe6e6e6f8, 0xe6e6e606, 0xe6e6e616, 0xe6e6f8e6, 0xe6e6f8f8, 0xe6e6f806, 0xe6e6f816,
    0xe6e606e6, 0xe6e606f8, 0xe6e60606, 0xe6e60616, 0xe6e616e6, 0xe6e616f8, 0xe6e61606, 0xe6e61616,
    0xe6f8e6e6, 0xe6f8e6f8, 0xe6f8e606, 0xe6f8e616, 0xe6f8f8e6, 0xe6f8f8f8, 0xe6f8f806, 0xe6f8f816,
    0xe6f806e6, 0xe6f806f8, 0xe6f80606, 0xe6f80616, 0xe6f816e6, 0xe6f816f8, 0xe6f81606, 0xe6f81616,
    0xe606e6e6, 0xe606e6f8, 0xe606e606, 0xe606e616, 0xe606f8e6, 0xe606f8f8, 0xe606f806, 0xe606f816,
    0xe60606e6, 0xe60606f8, 0xe6060606, 0xe6060616, 0xe60616e6, 0xe60616f8, 0xe6061606, 0xe6061616,
    0xe616e6e6, 0xe616e6f8, 0xe616e606, 0xe616e616, 0xe616f8e6, 0xe616f8f8, 0xe616f806, 0xe616f816,
    0xe61606e6, 0xe61606f8, 0xe6160606, 0xe6160616, 0xe61616e6, 0xe61616f8, 0xe6161606, 0xe6161616,
    0xf8e6e6e6, 0xf8e6e6f8, 0xf8e6e606, 0xf8e6e616, 0xf8e6f8e6, 0xf8e6f8f8, 0xf8e6f806, 0xf8e6f816,
    0xf8e606e6, 0xf8e606f8, 0xf8e60606, 0xf8e60616, 0xf8e616e6, 0xf8e616f8, 0xf8e61606, 0xf8e61616,
    0xf8f8e6e6, 0xf8f8e6f8, 0xf8f8e606, 0xf8f8e616, 0xf8f8f8e6, 0xf8f8f8f8, 0xf8f8f806, 0xf8f8f816,
    0xf8f806e6, 0xf8f806f8, 0xf8f80606, 0xf8f80616, 0xf8f816e6, 0xf8f816f8, 0xf8f81606, 0xf8f81616,
    0xf806e6e6, 0xf806e6f8, 0xf806e606, 0xf806e616, 0xf806f8e6, 0xf806f8f8, 0xf806f806, 0xf806f816,
    0xf80606e6, 0xf80606f8, 0xf8060606, 0xf8060616, 0xf80616e6, 0xf80616f8, 0xf8061606, 0xf8061616,
    0xf816e6e6, 0xf816e6f8, 0xf816e606, 0xf816e616, 0xf816f8e6, 0xf816f8f8, 0xf816f806, 0xf816f816,
    0xf81606e6, 0xf81606f8, 0xf8160606, 0xf8160616, 0xf81616e6, 0xf81616f8, 0xf8161606, 0xf8161616,
    0x06e6e6e6, 0x06e6e6f8, 0x06e6e606, 0x06e6e616, 0x06e6f8e6, 0x06e6f8f8, 0x06e6f806, 0x06e6f816,
    0x06e606e6, 0x06e606f8, 0x06e60606, 0x06e60616, 0x06e616e6, 0x06e616f8, 0x06e61606, 0x06e61616,
    0x06f8e6e6, 0x06f8e6f8, 0x06f8e606, 0x06f8e616, 0x06f8f8e6, 0x06f8f8f8, 0x06f8f806, 0x06f8f816,
    0x06f806e6, 0x06f806f8, 0x06f80606, 0x06f80616, 0x06f816e6, 0x06f816f8, 0x06f81606, 0x06f81616,
    0x0606e6e6, 0x0606e6f8, 0x0606e606, 0x0606e616, 0x0606f8e6, 0x0606f8f8, 0x0606f806, 0x0606f816,
    0x060606e6, 0x060606f8, 0x06060606, 0x06060616, 0x060616e6, 0x060616f8, 0x06061606, 0x06061616,
    0x0616e6e6, 0x0616e6f8, 0x0616e606, 0x0616e616, 0x0616f8e6, 0x0616f8f8, 0x0616f806, 0x0616f816,
    0x061606e6, 0x061606f8, 0x06160606, 0x06160616, 0x061616e6, 0x061616f8, 0x06161606, 0x06161616,
    0x16e6e6e6, 0x16e6e6f8, 0x16e6e606, 0x16e6e616, 0x16e6f8e6, 0x16e6f8f8, 0x16e6f806, 0x16e6f816,
    0x16e606e6, 0x16e606f8, 0x16e60606, 0x16e60616, 0x16e616e6, 0x16e616f8, 0x16e61606, 0x16e61616,
    0x16f8e6e6, 0x16f8e6f8, 0x16f8e606, 0x16f8e616, 0x16f8f8e6, 0x16f8f8f8, 0x16f8f806, 0x16f8f816,
    0x16f806e6, 0x16f806f8, 0x16f80606, 0x16f80616, 0x16f816e6, 0x16f816f8, 0x16f81606, 0x16f81616,
    0x1606e6e6, 0x1606e6f8, 0x1606e606, 0x1606e616, 0x1606f8e6, 0x1606f8f8, 0x1606f806, 0x1606f816,
    0x160606e6, 0x160606f8, 0x16060606, 0x16060616, 0x160616e6, 0x160616f8, 0x16061606, 0x16061616,
    0x1616e6e6, 0x1616e6f8, 0x1616e606, 0x1616e616, 0x1616f8e6, 0x1616f8f8, 0x1616f806, 0x1616f816,
    0x161606e6, 0x161606f8, 0x16160606, 0x16160616, 0x161616e6, 0x161616f8, 0x16161606, 0x16161616,
};

static __device__ __forceinline__ int int_from_table_4(const uint32_t idx, const int * values) {
    return values[ggml_cuda_dp4a(idx, 0x40100401, 0)];
}

#ifdef __CUDA_ARCH__
static __device__ __forceinline__ int2 get_int_from_table_8(const int & q4, const int8_t * values) {
    const uint32_t * values32 = (const uint32_t *)values;
    uint32_t v1 = __byte_perm(values32[0], values32[1], q4);
    uint32_t v2 = __byte_perm(values32[0], values32[1], q4 >> 16);
    return make_int2(__byte_perm(v1, v2, 0x6420), __byte_perm(v1, v2, 0x7531));
}
#endif

// ik_llama.cpp's _q8 loaders for IQ2_K..IQ5_K re-quantize block-16 values into Q8_0 tiles; tensor-core MMA only
static constexpr __host__ __device__ bool mmq_requant(const int mmq_x, const int min_x) {
#ifdef TURING_MMA_AVAILABLE
    return mmq_x >= min_x;
#else
    GGML_UNUSED_VARS(mmq_x, min_x);
    return false;
#endif // TURING_MMA_AVAILABLE
}

#ifdef TURING_MMA_AVAILABLE
// Table lookup requantized to round((v + 1152 + bias)*r) via the half 1280 + q, whose ulp is 1 and low byte is q
static __device__ __forceinline__ int2 get_int_from_table_16_q8(const int & q4, const int8_t * values,
        const uint32_t & bias_x, const half2 & r_x, const uint32_t & bias_y, const half2 & r_y) {
    const uint32_t * values32 = (const uint32_t *)values;
    const uint32_t t0 = values32[0] ^ 0x80808080;
    const uint32_t t1 = values32[1] ^ 0x80808080;
    const uint32_t t2 = values32[2] ^ 0x80808080;
    const uint32_t t3 = values32[3] ^ 0x80808080;
    const half2 magic = __float2half2_rn(1280.0f);

    const uint32_t mask = 0x32103210 | ((q4 & 0x88888888) >> 1);
    // bytes of v3: low nibble 0, high nibble 0, low nibble 1, high nibble 1
    const uint32_t v3 = __byte_perm(__byte_perm(t0, t1, q4),       __byte_perm(t2, t3, q4),       mask);
    const uint32_t v4 = __byte_perm(__byte_perm(t0, t1, q4 >> 16), __byte_perm(t2, t3, q4 >> 16), mask >> 16);

    uint32_t x01 = __byte_perm(v3, 0x64646464, 0x4240);
    uint32_t y01 = __byte_perm(v3, 0x64646464, 0x4341);
    uint32_t x23 = __byte_perm(v4, 0x64646464, 0x4240);
    uint32_t y23 = __byte_perm(v4, 0x64646464, 0x4341);

    *(half2 *)&x01 = __hfma2(__hadd2(*(half2 *)&x01, *(const half2 *)&bias_x), r_x, magic);
    *(half2 *)&x23 = __hfma2(__hadd2(*(half2 *)&x23, *(const half2 *)&bias_x), r_x, magic);
    *(half2 *)&y01 = __hfma2(__hadd2(*(half2 *)&y01, *(const half2 *)&bias_y), r_y, magic);
    *(half2 *)&y23 = __hfma2(__hadd2(*(half2 *)&y23, *(const half2 *)&bias_y), r_y, magic);

    return make_int2(__byte_perm(x01, x23, 0x6420), __byte_perm(y01, y23, 0x6420));
}

// round(v*r) for 4 int8 values the same way, r equal in both halves, results must fit int8
static __device__ __forceinline__ int requant_int_q8(const int & v, const half2 & r) {
    const uint32_t u = v ^ 0x80808080;
    const uint32_t bias = 0xe480e480; // -1152
    const half2 magic = __float2half2_rn(1280.0f);

    uint32_t x02 = __byte_perm(u, 0x64646464, 0x4240);
    uint32_t x13 = __byte_perm(u, 0x64646464, 0x4341);

    *(half2 *)&x02 = __hfma2(__hadd2(*(half2 *)&x02, *(const half2 *)&bias), r, magic);
    *(half2 *)&x13 = __hfma2(__hadd2(*(half2 *)&x13, *(const half2 *)&bias), r, magic);

    return __byte_perm(x02, x13, 0x6240);
}
#endif // TURING_MMA_AVAILABLE
