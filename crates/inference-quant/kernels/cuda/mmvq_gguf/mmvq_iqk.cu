// Matvec and dequantize kernels for ik_llama.cpp's IQ*_K GGUF types, with Q8_1 activations.
// Vec dots, dequantizers, value tables and helpers are copied from ik_llama.cpp (MIT, Iwan Kawrakow):
// mmvq-instance-iq*_k*.cu, convert.cu, ggml-common.h, iqk_cuda_common.h and iqk_mmvq_templates.cuh.
#include "mmvq_rows.cuh"
#include <type_traits>

typedef half ggml_half;


typedef struct { ggml_half d; uint16_t extra; uint8_t scales[QK_K/32]; uint8_t qs[QK_K/4]; } block_iq2_k;

typedef struct { ggml_half d; uint16_t extra; uint16_t scales_h; uint8_t scales_l[QK_K/32]; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/8]; } block_iq3_k;

typedef struct { ggml_half d; uint16_t extra; uint8_t scales_h[QK_K/64]; uint8_t scales_l[QK_K/32]; uint8_t qs[QK_K/2]; } block_iq4_k;

typedef struct { ggml_half d; uint16_t extra; uint8_t scales_h[QK_K/64]; uint8_t scales_l[QK_K/32]; uint8_t qs[QK_K/2]; uint8_t qh[QK_K/8]; } block_iq5_k;

typedef struct { ggml_half d; uint16_t extra; int8_t scales[QK_K/16]; uint8_t qs[QK_K/2]; uint8_t qh[QK_K/4]; } block_iq6_k;

typedef struct { uint8_t scales[QK_K/32]; uint8_t qs[QK_K/2]; } block_iq4_ks;

typedef struct { uint16_t extra; uint8_t scales[QK_K/64]; uint8_t qs[QK_K/4]; } block_iq2_ks;

typedef struct { uint32_t qs[QK_K/8]; } block_iq4_kss;

typedef struct { uint8_t scales[QK_K/32]; uint8_t qs[QK_K/2]; uint8_t qh[QK_K/8]; } block_iq5_ks;

typedef struct { uint16_t extra; uint8_t scales[QK_K/64]; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/8]; } block_iq3_ks;

typedef struct { uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/16]; } block_iq2_kl;

static const __device__ int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

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

__device__ __forceinline__ int int_from_table_4(const uint32_t idx, const int * values) {
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

static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * values) {
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    return ggml_cuda_perm_table_16(q4, values);
#elif defined(__CUDA_ARCH__)
    uint32_t v1, v2, v3, v4, mask;
    const uint32_t * values32 = (const uint32_t *)values;

    mask = (0x32103210 | ((q4 & 0x88888888) >> 1));
    // Perform lookups in the lower half of the table (indices 0-7).
    v1 = __byte_perm(values32[0], values32[1], q4);
    // Perform lookups in the upper half of the table (indices 8-15).
    v2 = __byte_perm(values32[2], values32[3], q4);
    // Select between the low and high results based on the MSB of each index nibble.
    v3 = __byte_perm(v1, v2, mask);
    // Same for the upper part of q4.
    v1 = __byte_perm(values32[0], values32[1], q4 >> 16);
    v2 = __byte_perm(values32[2], values32[3], q4 >> 16);
    v4 = __byte_perm(v1, v2, mask >> 16);

    // Mix the results to get the final int2.
    return make_int2(__byte_perm(v3, v4, 0x6420), __byte_perm(v3, v4, 0x7531));
#else
    const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t * q0_8   = (const int8_t *) &q0_32;
    const char4    val0_8 = make_char4(values[q0_8[0]], values[q0_8[1]], values[q0_8[2]], values[q0_8[3]]);

    const int      q1_32  = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t * q1_8   = (const int8_t *) &q1_32;
    const char4    val1_8 = make_char4(values[q1_8[0]], values[q1_8[1]], values[q1_8[2]], values[q1_8[3]]);

    return make_int2(*((const int *) &val0_8), *((const int *) &val1_8));
#endif
}

static __device__ __forceinline__ int int_from_table(const uint8_t * a8, const uint8_t * values) {
    uint16_t v1 = values[a8[0]] | (values[a8[1]] << 8);
    uint16_t v2 = values[a8[2]] | (values[a8[3]] << 8);
    return v1 | (v2 << 16);
}



// mmvq-instance-iq2_k.cu
__device__ __forceinline__ void vec_dot_iq2_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    // iqs is 0, 4, 8, 12, 16, 20, 24, 28
    // we have 16 packed quants (when cast to int)

    int i4 = iqs/4;  // 0...7. We will process q8 blocks 4*(i4/4), 4*(i4/4)+1, 4*(i4/4)+2, 4*(i4/4)+3
    const int32_t  * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t  * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t  * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t  * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const block_iq2_k * bq2 = (const block_iq2_k *) vbq + kbx;
    const uint32_t * q2 = (const uint32_t *)bq2->qs + 8*(i4/4) + 2*(i4%4);
    const uint16_t extra = bq2->extra >> (8*(i4/4) + (i4%4)/2);

    const uint32_t * scales = (const uint32_t *)bq2->scales;
    uint32_t s32 = __vsub4((scales[i4/4] >> 4*(((i4%4)/2)%2)) & 0x0f0f0f0f, 0x08080808);
    const int8_t * s8 = (const int8_t *)&s32;

    // Block of 16: (32*(4*(i4/4)+k)+8*(i4%4))/16 = 8*(i4/4) + 2*k + (i4%4)/2
    // -> scales_l[4*(i4/4) + k] >> 4*(((i4%4)/2)%2)

#ifdef __CUDA_ARCH__
    uint32_t extra32 = uint32_t(extra & 0xff) * 0x01010101;
    uint32_t extra32_1 = (extra32 << 2) & 0x44444444;
    uint32_t extra32_2 = (extra32 << 0) & 0x44444444;

    uint32_t val1, val2;

    val1 = ((q2[0] >> 0) & 0x33333333) | extra32_1; val2 = ((q2[1] >> 0) & 0x33333333) | extra32_1;
    int2 v1 = get_int_from_table_8(val1, iq2nl_values);
    int2 v2 = get_int_from_table_8(val2, iq2nl_values);
    int sumi1 = ggml_cuda_dp4a(v2.x, q8_1[1], ggml_cuda_dp4a(v1.x, q8_1[0], 0)) * s8[0];
    int sumi3 = ggml_cuda_dp4a(v2.y, q8_3[1], ggml_cuda_dp4a(v1.y, q8_3[0], 0)) * s8[2];

    val1 = ((q2[0] >> 2) & 0x33333333) | extra32_2; val2 = ((q2[1] >> 2) & 0x33333333) | extra32_2;
    v1 = get_int_from_table_8(val1, iq2nl_values);
    v2 = get_int_from_table_8(val2, iq2nl_values);
    int sumi2 = ggml_cuda_dp4a(v2.x, q8_2[1], ggml_cuda_dp4a(v1.x, q8_2[0], 0)) * s8[1];
    int sumi4 = ggml_cuda_dp4a(v2.y, q8_4[1], ggml_cuda_dp4a(v1.y, q8_4[0], 0)) * s8[3];

#else

    const int * all_values = (const int *)iq2k_table;
    const int * values;

    uint32_t val1 = q2[0], val2 = q2[1];

    uint32_t aux32[2];
    int v1, v2;

    aux32[0] = ((val1 >> 0) & 0x03030303); aux32[1] = ((val2 >> 0) & 0x03030303); values = all_values + ((extra & 0x01) << 8);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi1 = ggml_cuda_dp4a(v2, q8_1[1], ggml_cuda_dp4a(v1, q8_1[0], 0)) * s8[0];

    aux32[0] = ((val1 >> 2) & 0x03030303); aux32[1] = ((val2 >> 2) & 0x03030303); values = all_values + ((extra & 0x04) << 6);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi2 = ggml_cuda_dp4a(v2, q8_2[1], ggml_cuda_dp4a(v1, q8_2[0], 0)) * s8[1];

    aux32[0] = ((val1 >> 4) & 0x03030303); aux32[1] = ((val2 >> 4) & 0x03030303); values = all_values + ((extra & 0x10) << 4);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi3 = ggml_cuda_dp4a(v2, q8_3[1], ggml_cuda_dp4a(v1, q8_3[0], 0)) * s8[2];

    aux32[0] = ((val1 >> 6) & 0x03030303); aux32[1] = ((val2 >> 6) & 0x03030303); values = all_values + ((extra & 0x40) << 2);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi4 = ggml_cuda_dp4a(v2, q8_4[1], ggml_cuda_dp4a(v1, q8_4[0], 0)) * s8[3];
#endif

    *result += __half2float(bq2->d) * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                                    +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                                    +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                                    +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

// mmvq-instance-iq3_k.cu
__device__ __forceinline__ void vec_dot_iq3_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {
    const block_iq3_k * bq3 = (const block_iq3_k *) vbq + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1. 0 works on quants 0...127, 1 on quants 128...255
                              // Each thread processes 8 quants in each of the 4 32-blocks
    const int il8   = iqs%4;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31
    const int shift = 4*(il8/2);

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int hshift = 4*(1-ib128);
    const uint16_t sh = bq3->scales_h >> (8*ib128 + il8/2);

    const uint8_t extra = bq3->extra >> (8*ib128 + il8/2);
    uint32_t extra32 = uint32_t(extra) * 0x01010101;
    uint32_t extra32_1 = (extra32 << 3) & 0x88888888;
    uint32_t extra32_2 = (extra32 << 1) & 0x88888888;

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) << hshift);

        uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh >> 2) & 0x04040404) | ((vh >> 0) & 0x40404040);
        uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh >> 3) & 0x04040404) | ((vh >> 1) & 0x40404040);
        int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const float d = __half2float(bq3->d);
    const uint16_t * sl16 = (const uint16_t *)bq3->scales_l + 2*ib128;
    aux32 = ((((sl16[0] | (sl16[1] << 16)) >> shift) & 0x0f0f0f0f) << 1) | 0x01010101;
    *result += d * (__low2float(bq8_1[4*ib128+0].ds) * aux8[0] * (sh & 0x01 ? -1 : 1) * sumi[0] +
                    __low2float(bq8_1[4*ib128+1].ds) * aux8[1] * (sh & 0x04 ? -1 : 1) * sumi[1] +
                    __low2float(bq8_1[4*ib128+2].ds) * aux8[2] * (sh & 0x10 ? -1 : 1) * sumi[2] +
                    __low2float(bq8_1[4*ib128+3].ds) * aux8[3] * (sh & 0x40 ? -1 : 1) * sumi[3]);

}

// mmvq-instance-iq4_k.cu
__device__ __forceinline__ void vec_dot_iq4_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq4_k * bq4 = (const block_iq4_k *) vbq + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4;
    // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint16_t * q4 = (const uint16_t *)bq4->qs + 8*ib32;
    const uint16_t extra = bq4->extra >> 2*ib32;
    // iq4k_values high half is the low half + 4, and no entry reaches 0xFC, so the half select is a packed byte add.
    const int add1 = 0x04040404 & -( extra       & 1);
    const int add2 = 0x04040404 & -((extra >> 1) & 1);
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        const int aux32 = q4[2*j+0] | (q4[2*j+1] << 16);
        const int2 v = get_int_from_table_16(aux32, kvalues_iq4nl);
        sumi1 = ggml_cuda_dp4a(v.x + add1, q8[j+0], sumi1);
        sumi2 = ggml_cuda_dp4a(v.y + add2, q8[j+4], sumi2);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1[ib32].ds);
    const uint8_t sh = bq4->scales_h[ib32/2] >> 4*(ib32%2);
    const int ls1 = ((bq4->scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = ((bq4->scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
    *result += d * (sumi1 * ls1 + sumi2 * ls2);
}

// mmvq-instance-iq5_k.cu
__device__ __forceinline__ void vec_dot_iq5_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq5_k * bq5 = (const block_iq5_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq5nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq5->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq5->qh + 4*(i4%2);
    const uint16_t extra = bq5->extra >> (4*(i4/2) + (i4%2));
    const uint8_t * values1 = all_values + 32*(extra & 1);
    const uint8_t * values2 = all_values +  8*(extra & 4);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 2*(i4/2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const float d5 = __half2float(bq5->d);
    const uint8_t sh = bq5->scales_h[i4/2] >> 2*(i4%2);
    const int ls1 = (((bq5->scales_l[2*(i4/2)+0] >> 4*(i4%2)) & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = (((bq5->scales_l[2*(i4/2)+1] >> 4*(i4%2)) & 0xf) | ((sh << 0) & 0x30)) - 32;
    *result += d5 * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * ls1 + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * ls2);
}

// mmvq-instance-iq6_k.cu
__device__ __forceinline__ void vec_dot_iq6_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq6_k * bq6 = (const block_iq6_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq6nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)
                     //         Blocks of 32 index is 2*(i4/2) + 0 or 1

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq6->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq6->qh + 8*(i4/4) + 4*(i4%2);
    const uint16_t extra = bq6->extra >> (4*(i4/2) + (i4%2));
    const uint8_t * values1 = all_values + 64*(extra & 1);
    const uint8_t * values2 = all_values + 16*(extra & 4);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 4*((i4/2)%2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x30303030);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 2) & 0x30303030);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const float d6 = __half2float(bq6->d);
    *result += d6 * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * bq6->scales[4*(i4/2)+(i4%2)] + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * bq6->scales[4*(i4/2)+(i4%2)+2]);
}

// mmvq-instance-iq4_ks.cu
__device__ __forceinline__ void vec_dot_iq4_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq4_ks * bq4 = (const block_iq4_ks *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    const float dl = scale * ((bq4->scales[ib32] & 254) - 127);
    auto values = iq4k_values + ((bq4->scales[ib32] & 1) << 4);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        auto v = get_int_from_table_16(q4[j], values);
        sumi = ggml_cuda_dp4a(v.x, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[j+4], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

// mmvq-instance-iq2_ks.cu
__device__ __forceinline__ void vec_dot_iq2_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const half *)vbq;
    const block_iq2_ks * bq2 = (const block_iq2_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int i4 = iqs/4;  // 0...7. We will process q8 blocks 4*(i4/4), 4*(i4/4)+1, 4*(i4/4)+2, 4*(i4/4)+3
    const int32_t  * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t  * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t  * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t  * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const uint16_t * q2 = (const uint16_t *)bq2->qs + 16*(i4/4) + 4*(i4%4);
    const uint16_t extra = bq2->extra >> 4*(i4/4);

    uint32_t val1 = q2[0] | (q2[1] << 16), val2 = q2[2] | (q2[3] << 16);

    int32_t scales32;
    const uint16_t * scales16 = (const uint16_t *)bq2->scales;
    scales32 = __vsub4((scales16[i4/4] | (scales16[i4/4] << 12)) & 0x0f0f0f0f, 0x10101010);
    int8_t * s8 = (int8_t *)&scales32;
    s8[0] += ((extra >> 4) & 0x10);
    s8[1] += ((extra >> 6) & 0x10);
    s8[2] += ((extra >> 5) & 0x10);
    s8[3] += ((extra >> 7) & 0x10);

#ifdef __CUDA_ARCH__

    uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;

    uint32_t this_extra = ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
    uint32_t idx1 = ((val1 >> 0) & 0x33333333) | this_extra;
    uint32_t idx2 = ((val2 >> 0) & 0x33333333) | this_extra;
    int2 v1 = get_int_from_table_8(idx1, iq2nl_values);
    int2 v2 = get_int_from_table_8(idx2, iq2nl_values);

    int sumi1 = ggml_cuda_dp4a(v2.x, q8_1[1], ggml_cuda_dp4a(v1.x, q8_1[0], 0)) * s8[0];
    int sumi3 = ggml_cuda_dp4a(v2.y, q8_3[1], ggml_cuda_dp4a(v1.y, q8_3[0], 0)) * s8[1];

    this_extra = ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
    idx1 = ((val1 >> 2) & 0x33333333) | this_extra;
    idx2 = ((val2 >> 2) & 0x33333333) | this_extra;
    v1 = get_int_from_table_8(idx1, iq2nl_values);
    v2 = get_int_from_table_8(idx2, iq2nl_values);

    int sumi2 = ggml_cuda_dp4a(v2.x, q8_2[1], ggml_cuda_dp4a(v1.x, q8_2[0], 0)) * s8[2];
    int sumi4 = ggml_cuda_dp4a(v2.y, q8_4[1], ggml_cuda_dp4a(v1.y, q8_4[0], 0)) * s8[3];

#else
    uint32_t aux32[2];
    int v1, v2;
    const int * all_values = (const int *)iq2k_table;
    const int * values;

    aux32[0] = ((val1 >> 0) & 0x03030303); aux32[1] = ((val2 >> 0) & 0x03030303); values = all_values + ((extra & 0x01) << 8);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi1 = ggml_cuda_dp4a(v2, q8_1[1], ggml_cuda_dp4a(v1, q8_1[0], 0)) * s8[0];

    aux32[0] = ((val1 >> 2) & 0x03030303); aux32[1] = ((val2 >> 2) & 0x03030303); values = all_values + ((extra & 0x02) << 7);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi2 = ggml_cuda_dp4a(v2, q8_2[1], ggml_cuda_dp4a(v1, q8_2[0], 0)) * s8[2];

    aux32[0] = ((val1 >> 4) & 0x03030303); aux32[1] = ((val2 >> 4) & 0x03030303); values = all_values + ((extra & 0x04) << 6);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi3 = ggml_cuda_dp4a(v2, q8_3[1], ggml_cuda_dp4a(v1, q8_3[0], 0)) * s8[1];

    aux32[0] = ((val1 >> 6) & 0x03030303); aux32[1] = ((val2 >> 6) & 0x03030303); values = all_values + ((extra & 0x08) << 5);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi4 = ggml_cuda_dp4a(v2, q8_4[1], ggml_cuda_dp4a(v1, q8_4[0], 0)) * s8[3];
#endif

    *result += scale * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                     +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                     +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                     +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

// mmvq-instance-iq4_kss.cu
__device__ __forceinline__ void vec_dot_iq4_kss_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq4_kss * bq4 = (const block_iq4_kss *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
    uint8_t ls = (s32 | (s32 >> 15)) & 0xff;
    const float dl = scale * ((ls & 254) - 127);
    auto values = iq4k_values + ((ls & 1) << 4);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t aux32 = q4[j] & 0xfffefffe;
        aux32 ^= (aux32 >> 1);
        auto v = get_int_from_table_16(aux32, values);
        sumi = ggml_cuda_dp4a(v.x, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[j+4], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

// mmvq-instance-iq5_ks.cu
__device__ __forceinline__ void vec_dot_iq5_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq5_ks * bq5 = (const block_iq5_ks *)((const char *)vbq + sizeof(float)) + kbx;
    const uint8_t * all_values = (const uint8_t *)iq5nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq5->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq5->qh + 4*(i4%2);
    const uint8_t * values1 = all_values + ((bq5->scales[2*(i4/2)+0] & 1) << 5);
    const uint8_t * values2 = all_values + ((bq5->scales[2*(i4/2)+1] & 1) << 5);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 2*(i4/2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const int ls1 = (bq5->scales[2*(i4/2)+0] & 254) - 127;
    const int ls2 = (bq5->scales[2*(i4/2)+1] & 254) - 127;
    *result += scale * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * ls1 + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * ls2);
}

// mmvq-instance-iq3_ks.cu
__device__ __forceinline__ void vec_dot_iq3_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {

    float d = __half2float(*(const half *)vbq);
    const block_iq3_ks * bq3 = (const block_iq3_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1. 0 works on quants 0...127, 1 on quants 128...255
                              // Each thread processes 8 quants in each of the 4 32-blocks
    const int il8   = iqs%4;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint16_t extra = bq3->extra >> 4*ib128;
    uint32_t extra_v = uint32_t(extra >> 8) * 0x01010101;

    uint32_t extra32_1 = ((extra_v << 3) & 0x08080808) | ((extra_v << 5) & 0x80808080);
    uint32_t extra32_2 = ((extra_v << 2) & 0x08080808) | ((extra_v << 4) & 0x80808080);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) >> 4*ib128);

        uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh << 2) & 0x04040404) | ((vh << 4) & 0x40404040);
        uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh << 1) & 0x04040404) | ((vh << 3) & 0x40404040);
        int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const uint16_t * sl16 = (const uint16_t *)bq3->scales;
    int32_t aux32 = __vsub4(((sl16[0] | (sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f, 0x10101010);
    const int8_t * a8 = (const int8_t *)&aux32;
    *result += d * (__low2float(bq8_1[4*ib128+0].ds) * (a8[0] + ((extra << 4) & 0x10)) * sumi[0] +
                    __low2float(bq8_1[4*ib128+1].ds) * (a8[1] + ((extra << 3) & 0x10)) * sumi[1] +
                    __low2float(bq8_1[4*ib128+2].ds) * (a8[2] + ((extra << 2) & 0x10)) * sumi[2] +
                    __low2float(bq8_1[4*ib128+3].ds) * (a8[3] + ((extra << 1) & 0x10)) * sumi[3]);

}

// mmvq-instance-iq2_kl.cu
__device__ __forceinline__ void vec_dot_iq2_kl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {

    float d = __half2float(*(const half *)vbq);
    const block_iq2_kl * bq2 = (const block_iq2_kl *)((const char *)vbq + sizeof(half)) + kbx;

    int iqs = iiqs/4;
    const int ib64 = iqs/2;  // 0...3. 0 works on quants 0...63, 1 on quants 64...127, etc.
                             // Each thread processes 16 quants in each of the 2 32-blocks
    const int il16 = iqs%2;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31

    const uint16_t * ql = (const uint16_t *)bq2->qs + 8*ib64 + 4*il16;
    const uint16_t * qh = (const uint16_t *)bq2->qh + 4*il16;

    int32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int * q8l = (const int *)bq8_1[2*ib64+0].qs + 4*il16;
    const int * q8h = (const int *)bq8_1[2*ib64+1].qs + 4*il16;

    int sumi1 = 0, sumi2 = 0;
    int v1, v2;
    for (int i = 0; i < 2; ++i) {
        uint32_t vl =  ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = (qh[2*i+0] | (qh[2*i+1] << 16)) >> 2*ib64;

        aux32 = (vl & 0x0f0f0f0f) | ((vh << 4) & 0x10101010);
        v1 = iq2kl_values[aux8[0]] | (iq2kl_values[aux8[1]] << 16);
        v2 = iq2kl_values[aux8[2]] | (iq2kl_values[aux8[3]] << 16);
        sumi1 = ggml_cuda_dp4a(v1, q8l[2*i+0], ggml_cuda_dp4a(v2, q8l[2*i+1], sumi1));

        aux32 = ((vl >> 4) & 0x0f0f0f0f) | ((vh << 3) & 0x10101010);
        v1 = iq2kl_values[aux8[0]] | (iq2kl_values[aux8[1]] << 16);
        v2 = iq2kl_values[aux8[2]] | (iq2kl_values[aux8[3]] << 16);
        sumi2 = ggml_cuda_dp4a(v1, q8h[2*i+0], ggml_cuda_dp4a(v2, q8h[2*i+1], sumi2));
    }

    auto sh = bq2->scales_h >> 4*ib64;
    int ls1 = int(((bq2->scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 4) & 0x30)) - 32;
    int ls2 = int(((bq2->scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 2) & 0x30)) - 32;

    *result += d * (__low2float(bq8_1[2*ib64+0].ds) * ls1 * sumi1 + __low2float(bq8_1[2*ib64+1].ds) * ls2 * sumi2);

}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq2_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_iq2_k * x = (const block_iq2_k *) vx;

    const int tid = threadIdx.x;
    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + i*QK_K + 128*ib128 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * (((x[i].scales[4*ib128+0] >> 4*(il/8)) & 0xf) - 8);
    const float dl2 = d * (((x[i].scales[4*ib128+1] >> 4*(il/8)) & 0xf) - 8);
    const float dl3 = d * (((x[i].scales[4*ib128+2] >> 4*(il/8)) & 0xf) - 8);
    const float dl4 = d * (((x[i].scales[4*ib128+3] >> 4*(il/8)) & 0xf) - 8);
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    const int16_t extra = x[i].extra >> (8*ib128 + (il/8));
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = __float2bfloat16(dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)]);
            y[j+32] = __float2bfloat16(dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 0) & 4)]);
            y[j+64] = __float2bfloat16(dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 2) & 4)]);
            y[j+96] = __float2bfloat16(dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 4) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)];
            y[j+32] = dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 0) & 4)];
            y[j+64] = dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 2) & 4)];
            y[j+96] = dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 4) & 4)];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq3_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_iq3_k * x = (const block_iq3_k *) vx;

    const int tid = threadIdx.x;
    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + i*QK_K + 128*ib128 + 2*il;
    const float d = (float)x[i].d;
    const uint16_t sh = x[i].scales_h >> (8*ib128 + (il/8));
    const float dl1 = d * ((2*((x[i].scales_l[4*ib128+0] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x01) ? -1 : 1));
    const float dl2 = d * ((2*((x[i].scales_l[4*ib128+1] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x04) ? -1 : 1));
    const float dl3 = d * ((2*((x[i].scales_l[4*ib128+2] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x10) ? -1 : 1));
    const float dl4 = d * ((2*((x[i].scales_l[4*ib128+3] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x40) ? -1 : 1));
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    const int16_t extra = x[i].extra >> (8*ib128 + (il/8));
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h = qh[j] >> (4*(ib128%2));
            y[j+ 0] = __float2bfloat16(dl1 * iq3nl_values[(((qs[j] >> 0) & 0x03) | ((h & 0x01) << 2)) + ((extra << 3) & 8)]);
            y[j+32] = __float2bfloat16(dl2 * iq3nl_values[(((qs[j] >> 2) & 0x03) | ((h & 0x02) << 1)) + ((extra << 1) & 8)]);
            y[j+64] = __float2bfloat16(dl3 * iq3nl_values[(((qs[j] >> 4) & 0x03) | ((h & 0x04) >> 0)) + ((extra >> 1) & 8)]);
            y[j+96] = __float2bfloat16(dl4 * iq3nl_values[(((qs[j] >> 6) & 0x03) | ((h & 0x08) >> 1)) + ((extra >> 3) & 8)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h = qh[j] >> (4*(ib128%2));
            y[j+ 0] = dl1 * iq3nl_values[(((qs[j] >> 0) & 0x03) | ((h & 0x01) << 2)) + ((extra << 3) & 8)];
            y[j+32] = dl2 * iq3nl_values[(((qs[j] >> 2) & 0x03) | ((h & 0x02) << 1)) + ((extra << 1) & 8)];
            y[j+64] = dl3 * iq3nl_values[(((qs[j] >> 4) & 0x03) | ((h & 0x04) >> 0)) + ((extra >> 1) & 8)];
            y[j+96] = dl4 * iq3nl_values[(((qs[j] >> 6) & 0x03) | ((h & 0x08) >> 1)) + ((extra >> 3) & 8)];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq4_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_k * x = (const block_iq4_k *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d;
    const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
    const float d1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
    const float d2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
    const int8_t * values1 = iq4k_values + 16*((x[i].extra >> (2*ib+0)) & 1);
    const int8_t * values2 = iq4k_values + 16*((x[i].extra >> (2*ib+1)) & 1);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = __float2bfloat16(d1 * values1[q4[j] & 0xf]);
            y[j+16] = __float2bfloat16(d2 * values2[q4[j] >>  4]);
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = d1 * values1[q4[j] & 0xf];
            y[j+16] = d2 * values2[q4[j] >>  4];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq5_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_iq5_k * x = (const block_iq5_k *) vx;

    const int tid = threadIdx.x;
    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 64*ib64 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * (((x[i].scales_l[2*ib64+0] & 0xf) | ((x[i].scales_h[ib64] << 4) & 0x30)) - 32);
    const float dl2 = d * (((x[i].scales_l[2*ib64+0] >>  4) | ((x[i].scales_h[ib64] << 2) & 0x30)) - 32);
    const float dl3 = d * (((x[i].scales_l[2*ib64+1] & 0xf) | ((x[i].scales_h[ib64] >> 0) & 0x30)) - 32);
    const float dl4 = d * (((x[i].scales_l[2*ib64+1] >>  4) | ((x[i].scales_h[ib64] >> 2) & 0x30)) - 32);
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    const uint8_t extra = x[i].extra >> 4*(ib64%4);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = __float2bfloat16(dl1 * iq5nl_values[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4) | ((extra << 5) & 0x20)]);
            y[j+16] = __float2bfloat16(dl2 * iq5nl_values[(qs[j+16] & 0xf) | ((h2 & 1) << 4) | ((extra << 4) & 0x20)]);
            y[j+32] = __float2bfloat16(dl3 * iq5nl_values[(qs[j+ 0] >>  4) | ((h1 & 2) << 3) | ((extra << 3) & 0x20)]);
            y[j+48] = __float2bfloat16(dl4 * iq5nl_values[(qs[j+16] >>  4) | ((h2 & 2) << 3) | ((extra << 2) & 0x20)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = dl1 * iq5nl_values[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4) | ((extra << 5) & 0x20)];
            y[j+16] = dl2 * iq5nl_values[(qs[j+16] & 0xf) | ((h2 & 1) << 4) | ((extra << 4) & 0x20)];
            y[j+32] = dl3 * iq5nl_values[(qs[j+ 0] >>  4) | ((h1 & 2) << 3) | ((extra << 3) & 0x20)];
            y[j+48] = dl4 * iq5nl_values[(qs[j+16] >>  4) | ((h2 & 2) << 3) | ((extra << 2) & 0x20)];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq6_k(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_iq6_k * x = (const block_iq6_k *) vx;

    const int tid = threadIdx.x;
    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 64*ib64 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * x[i].scales[4*ib64+0];
    const float dl2 = d * x[i].scales[4*ib64+1];
    const float dl3 = d * x[i].scales[4*ib64+2];
    const float dl4 = d * x[i].scales[4*ib64+3];
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 32*(ib64/2) + 2*il;
    const uint8_t extra = x[i].extra >> 4*(ib64%4);
    for (int j = 0; j < 2; ++j) {
        const uint8_t h1 = qh[j] >> 4*(ib64%2), h2 = qh[j+16] >> 4*(ib64%2);
        uint8_t q1 = (qs[j+ 0] & 0xf) | ((h1 & 0x03) << 4);
        uint8_t q2 = (qs[j+16] & 0xf) | ((h2 & 0x03) << 4);
        uint8_t q3 = (qs[j+ 0] >>  4) | ((h1 & 0x0c) << 2);
        uint8_t q4 = (qs[j+16] >>  4) | ((h2 & 0x0c) << 2);
        if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
            y[j+ 0] = __float2bfloat16(dl1 * (iq6nl_values[q1] + (extra & 1 ? 1 : 0)));
            y[j+16] = __float2bfloat16(dl2 * (iq6nl_values[q2] + (extra & 2 ? 1 : 0)));
            y[j+32] = __float2bfloat16(dl3 * (iq6nl_values[q3] + (extra & 4 ? 1 : 0)));
            y[j+48] = __float2bfloat16(dl4 * (iq6nl_values[q4] + (extra & 8 ? 1 : 0)));
        } else {
            y[j+ 0] = dl1 * (iq6nl_values[q1] + (extra & 1 ? 1 : 0));
            y[j+16] = dl2 * (iq6nl_values[q2] + (extra & 2 ? 1 : 0));
            y[j+32] = dl3 * (iq6nl_values[q3] + (extra & 4 ? 1 : 0));
            y[j+48] = dl4 * (iq6nl_values[q4] + (extra & 8 ? 1 : 0));
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq4_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq4_ks * x = (const block_iq4_ks *)(cx + sizeof(float));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + ii*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = scale * ((x[i].scales[ib] & 254) - 127);
    const int8_t * values = iq4k_values + ((x[i].scales[ib] & 1) << 4);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = __float2bfloat16(d * values[q4[j] & 0xf]);
            y[j+16] = __float2bfloat16(d * values[q4[j] >>  4]);
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = d * values[q4[j] & 0xf];
            y[j+16] = d * values[q4[j] >>  4];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq2_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float d = (float)*(const half *)cx;
    const block_iq2_ks * x = (const block_iq2_ks *)(cx + sizeof(half));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int tid = threadIdx.x;
    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + ii*QK_K + 128*ib128 + 2*il;
    const int16_t extra = x[i].extra >> 4*ib128;
    const float dl1 = d * (((x[i].scales[2*ib128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16);
    const float dl2 = d * (((x[i].scales[2*ib128+0] >>  4) | ((extra >> 5) & 0x10)) - 16);
    const float dl3 = d * (((x[i].scales[2*ib128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16);
    const float dl4 = d * (((x[i].scales[2*ib128+1] >>  4) | ((extra >> 7) & 0x10)) - 16);
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = __float2bfloat16(dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)]);
            y[j+32] = __float2bfloat16(dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 1) & 4)]);
            y[j+64] = __float2bfloat16(dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 0) & 4)]);
            y[j+96] = __float2bfloat16(dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 1) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)];
            y[j+32] = dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 1) & 4)];
            y[j+64] = dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 0) & 4)];
            y[j+96] = dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 1) & 4)];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq4_kss(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq4_kss * x = (const block_iq4_kss *)(cx + sizeof(float));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + ii*QK_K + 32*ib + 4*il;
    const uint32_t * q4 = x[i].qs + 4*ib;
    uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
    uint8_t ls = (s32 | (s32 >> 15)) & 0xff;
    const float d = scale * ((ls & 254) - 127);
    const int8_t * values = iq4k_values + ((ls & 1) << 4);
    uint32_t aux32[2];
    aux32[0] = q4[il] & 0xfffefffe;
    aux32[0] ^= (aux32[0] >> 1);
    aux32[1] = ((aux32[0] >> 4) & 0x0f0f0f0f);
    aux32[0] &= 0x0f0f0f0f;
    const uint8_t * aux8 = (const uint8_t *)aux32;
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = __float2bfloat16(d * values[aux8[j+0]]);
            y[j+16] = __float2bfloat16(d * values[aux8[j+4]]);
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = d * values[aux8[j+0]];
            y[j+16] = d * values[aux8[j+4]];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq5_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float d = *(const float *)cx;
    const block_iq5_ks * x = (const block_iq5_ks *)(cx + sizeof(float));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int tid = threadIdx.x;
    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + ii*QK_K + 64*ib64 + 2*il;
    const float dl1 = d * ((int)(x[i].scales[2*ib64+0] & 254) - 127);
    const float dl2 = d * ((int)(x[i].scales[2*ib64+1] & 254) - 127);
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    auto values1 = iq5nl_values + ((x[i].scales[2*ib64+0] & 1) << 5);
    auto values2 = iq5nl_values + ((x[i].scales[2*ib64+1] & 1) << 5);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = __float2bfloat16(dl1 * values1[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4)]);
            y[j+16] = __float2bfloat16(dl1 * values1[(qs[j+16] & 0xf) | ((h2 & 1) << 4)]);
            y[j+32] = __float2bfloat16(dl2 * values2[(qs[j+ 0] >>  4) | ((h1 & 2) << 3)]);
            y[j+48] = __float2bfloat16(dl2 * values2[(qs[j+16] >>  4) | ((h2 & 2) << 3)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = dl1 * values1[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4)];
            y[j+16] = dl1 * values1[(qs[j+16] & 0xf) | ((h2 & 1) << 4)];
            y[j+32] = dl2 * values2[(qs[j+ 0] >>  4) | ((h1 & 2) << 3)];
            y[j+48] = dl2 * values2[(qs[j+16] >>  4) | ((h2 & 2) << 3)];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq3_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const ggml_half *)cx;
    const block_iq3_ks * x = (const block_iq3_ks *)(cx + sizeof(ggml_half));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t is = tid/16;
    const int64_t il = tid%16;
    dst_t * y = yy + ii*QK_K + 128*is + 2*il;
    const uint8_t  * qs = x[i].qs + 32*is + 2*il;
    const uint8_t  * qh = x[i].qh + 2*il;
    uint16_t extra = x[i].extra >> 4*is;
    const float d0 = scale * (int(((x[i].scales[0] >> 4*is) & 0xf) | ((extra << 4) & 0x10)) - 16);
    const float d1 = scale * (int(((x[i].scales[1] >> 4*is) & 0xf) | ((extra << 3) & 0x10)) - 16);
    const float d2 = scale * (int(((x[i].scales[2] >> 4*is) & 0xf) | ((extra << 2) & 0x10)) - 16);
    const float d3 = scale * (int(((x[i].scales[3] >> 4*is) & 0xf) | ((extra << 1) & 0x10)) - 16);
    extra >>= 8;
    const int8_t * values0 = iq3nl_values + ((extra & 1) << 3);
    const int8_t * values1 = iq3nl_values + ((extra & 2) << 2);
    const int8_t * values2 = iq3nl_values + ((extra & 4) << 1);
    const int8_t * values3 = iq3nl_values + ((extra & 8) << 0);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 4*is;
            y[j+ 0] = __float2bfloat16(d0 * values0[((qs[j] >> 0) & 3) | ((h << 2) & 4)]);
            y[j+32] = __float2bfloat16(d1 * values1[((qs[j] >> 2) & 3) | ((h << 1) & 4)]);
            y[j+64] = __float2bfloat16(d2 * values2[((qs[j] >> 4) & 3) | ((h >> 0) & 4)]);
            y[j+96] = __float2bfloat16(d3 * values3[((qs[j] >> 6) & 3) | ((h >> 1) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 4*is;
            y[j+ 0] = d0 * values0[((qs[j] >> 0) & 3) | ((h << 2) & 4)];
            y[j+32] = d1 * values1[((qs[j] >> 2) & 3) | ((h << 1) & 4)];
            y[j+64] = d2 * values2[((qs[j] >> 4) & 3) | ((h >> 0) & 4)];
            y[j+96] = d3 * values3[((qs[j] >> 6) & 3) | ((h >> 1) & 4)];
        }
    }
}

// convert.cu
template<typename dst_t>
static __global__ void dequantize_block_iq2_kl(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {

    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = (float)*(const ggml_half *)cx;
    const block_iq2_kl * x = (const block_iq2_kl *)(cx + sizeof(ggml_half));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t ib64 = tid/8;
    const int64_t il   = tid%8;
    dst_t * y = yy + ii*QK_K + 64*ib64 + 4*il;
    const uint8_t  * qs = x[i].qs + 16*ib64 + 2*il;
    const uint8_t  * qh = x[i].qh + 2*il;
    auto sh = x[i].scales_h >> 4*ib64;
    const float d1 = scale * (int(((x[i].scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 4) & 0x30)) - 32);
    const float d2 = scale * (int(((x[i].scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 2) & 0x30)) - 32);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 2*ib64;
            auto val1 = (const int8_t *)(iq2kl_values + ((qs[j] & 0xf) | ((h & 1) << 4)));
            auto val2 = (const int8_t *)(iq2kl_values + ((qs[j] >>  4) | ((h & 2) << 3)));
            y[2*j+ 0] = __float2bfloat16(d1 * val1[0]);
            y[2*j+ 1] = __float2bfloat16(d1 * val1[1]);
            y[2*j+32] = __float2bfloat16(d2 * val2[0]);
            y[2*j+33] = __float2bfloat16(d2 * val2[1]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 2*ib64;
            auto val1 = (const int8_t *)(iq2kl_values + ((qs[j] & 0xf) | ((h & 1) << 4)));
            auto val2 = (const int8_t *)(iq2kl_values + ((qs[j] >>  4) | ((h & 2) << 3)));
            y[2*j+ 0] = d1 * val1[0];
            y[2*j+ 1] = d1 * val1[1];
            y[2*j+32] = d2 * val2[0];
            y[2*j+33] = d2 * val2[1];
        }
    }
}


struct iqk_iq2_k {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 0 + (int64_t)sizeof(block_iq2_k) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq2_k_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq3_k {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 0 + (int64_t)sizeof(block_iq3_k) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq3_k_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq4_k {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 0 + (int64_t)sizeof(block_iq4_k) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq4_k_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq5_k {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 0 + (int64_t)sizeof(block_iq5_k) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq5_k_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq6_k {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 0 + (int64_t)sizeof(block_iq6_k) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq6_k_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq4_ks {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 4 + (int64_t)sizeof(block_iq4_ks) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq4_ks_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq2_ks {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 2 + (int64_t)sizeof(block_iq2_ks) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq2_ks_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq4_kss {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 4 + (int64_t)sizeof(block_iq4_kss) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq4_kss_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq5_ks {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 4 + (int64_t)sizeof(block_iq5_ks) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq5_ks_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq3_ks {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 2 + (int64_t)sizeof(block_iq3_ks) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq3_ks_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

struct iqk_iq2_kl {
  static constexpr bool has_tail = false;
  static __host__ __device__ int64_t row_size(int ncols) { return 2 + (int64_t)sizeof(block_iq2_kl) * (ncols / QK_K); }
  static __device__ __forceinline__ void vec_dot(const void *vbq, const block_q8_1 *bq8_1, int kbx, int iqs, float *result) {
    vec_dot_iq2_kl_q8_1(vbq, bq8_1, kbx, iqs, result);
  }
  static __device__ __forceinline__ void vec_dot_tail(const void *, const block_q8_1 *, int, int, int, float *) {}
};

#define DEQUANTIZE_IQK_LAUNCHER(tag, dst_tag, dst_c_type)                                                       \
  extern "C" void launch_dequantize_##tag##_##dst_tag(const void *vx, void *dst, int64_t nrows, int64_t ncols,  \
                                                      void *stream) {                                          \
    dequantize_block_##tag<dst_c_type><<<(unsigned int)(nrows * ncols / QK_K), WARP_SIZE, 0,                   \
                                         static_cast<cudaStream_t>(stream)>>>(vx, (dst_c_type *)dst);           \
  }

#define DEQUANTIZE_IQK_ROWS_LAUNCHER(tag, dst_tag, dst_c_type)                                                  \
  extern "C" void launch_dequantize_##tag##_##dst_tag(const void *vx, void *dst, int64_t nrows, int64_t ncols,  \
                                                      void *stream) {                                          \
    dequantize_block_##tag<dst_c_type><<<(unsigned int)(nrows * ncols / QK_K), WARP_SIZE, 0,                   \
                                         static_cast<cudaStream_t>(stream)>>>(vx, (dst_c_type *)dst, ncols,     \
                                                                              iqk_##tag::row_size(ncols));      \
  }

#define IQK_LAUNCHERS(tag, dequantize)                    \
  MMVQ_ROWS_LAUNCHER(tag, iqk_##tag, bf16, __nv_bfloat16) \
  MMVQ_ROWS_LAUNCHER(tag, iqk_##tag, f16, half)           \
  MMVQ_ROWS_LAUNCHER(tag, iqk_##tag, f32, float)          \
  MMVQ_ROWS_MOE_LAUNCHER(tag, iqk_##tag)                  \
  dequantize(tag, bf16, __nv_bfloat16)                    \
  dequantize(tag, f16, half)                              \
  dequantize(tag, f32, float)


IQK_LAUNCHERS(iq2_k, DEQUANTIZE_IQK_LAUNCHER)

IQK_LAUNCHERS(iq3_k, DEQUANTIZE_IQK_LAUNCHER)

IQK_LAUNCHERS(iq4_k, DEQUANTIZE_IQK_LAUNCHER)

IQK_LAUNCHERS(iq5_k, DEQUANTIZE_IQK_LAUNCHER)

IQK_LAUNCHERS(iq6_k, DEQUANTIZE_IQK_LAUNCHER)

IQK_LAUNCHERS(iq4_ks, DEQUANTIZE_IQK_ROWS_LAUNCHER)

IQK_LAUNCHERS(iq2_ks, DEQUANTIZE_IQK_ROWS_LAUNCHER)

IQK_LAUNCHERS(iq4_kss, DEQUANTIZE_IQK_ROWS_LAUNCHER)

IQK_LAUNCHERS(iq5_ks, DEQUANTIZE_IQK_ROWS_LAUNCHER)

IQK_LAUNCHERS(iq3_ks, DEQUANTIZE_IQK_ROWS_LAUNCHER)

IQK_LAUNCHERS(iq2_kl, DEQUANTIZE_IQK_ROWS_LAUNCHER)
