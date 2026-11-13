// The C ABI over exllamav3's trellis quantizer: quantize.cu's quantize_tiles and quantize_tiles_frac without torch
// (third_party/exllamav3/quant/quantize.cu). Each tile (256 weights) is encoded by a Viterbi search over its states:
// its quantized values and the state index at each position come back, the scratch sized by exl3_quantize_scratch.
#include <cuda_fp16.h>
#include <array>
#include <map>
#include <mutex>
#include <cstdint>
#include "util.h"
#include "util.cuh"
#include "quant/codebook.cuh"
#include "quant/comp_units/quantize_tiles_instances.cuh"
#include "quant/quantize_tiles_kernel.cuh"

#define CC_TILE_LEN 256

#define __(i, cb) quantize_tiles_kernel_k##i##_cb##cb
static const std::array<fp_quantize_tiles_kernel (*)(bool), 24> instances
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

// quantize.cu's choice of the dense specialization per arch, K and codebook (EXL3_QT_OPTIMIZED overrides)
static bool use_optimized(int major, int minor, int K, int cb)
{
    if (const char* env = std::getenv("EXL3_QT_OPTIMIZED")) return env[0] == '1';
    if (major == 12) return true;
    if (major == 8 && minor == 9) return K == 1 || K == 2 || K == 4 || K == 5 || (K == 7 && cb == 0) || K == 8;
    if (major == 8 && minor == 6) return K == 1 || K == 2 || K == 4 || K == 5 || (K == 7 && cb != 2) || K == 8;
    return false;
}

struct Launch
{
    bool optimized;
    fp_quantize_tiles_kernel kernel;
    int num_threads;
    int shmem;
    int wave;
};

static Launch launch_for(int K, int cb)
{
    int device, major, minor, sms;
    cuda_check(cudaGetDevice(&device));
    cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    cuda_check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const bool optimized = use_optimized(major, minor, K, cb);
    const int edges = 65536 >> K;
    const int cost_arrays = optimized ? (K == 1 ? 1 : 2) : (K >= QT_SHARED_COSTS_MIN_K ? 2 : 0);
    const int shmem = cost_arrays * edges * sizeof(half) + (optimized && QT_OPTIMIZED_COSTS_ONLY ? 0 : CC_TILE_LEN * sizeof(half) + 64 + 128);
    auto kernel = instances[K - 1 + 8 * cb](optimized);
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    cudaFuncAttributes attr;
    cuda_check(cudaFuncGetAttributes(&attr, kernel));
    int blocks_per_sm;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, attr.maxThreadsPerBlock, shmem));
    return {optimized, kernel, attr.maxThreadsPerBlock, shmem, blocks_per_sm * sms};
}

template <int cb>
__global__ void codebook_table_kernel(half* table)
{
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    table[state] = decode_3inst<cb>(state);
}

// the dense K = 6 kernel reads every state's value from a table, built once per device and codebook (never freed)
static const half2* codebook_table(int cb, cudaStream_t stream)
{
    static std::mutex mutex;
    static std::map<std::pair<int, int>, half*> tables;
    int device;
    cuda_check(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    half*& table = tables[{device, cb}];
    if (!table)
    {
        cuda_check(cudaMalloc(&table, 65536 * sizeof(half)));
        if (cb == 0) codebook_table_kernel<0><<<256, 256, 0, stream>>>(table);
        if (cb == 1) codebook_table_kernel<1><<<256, 256, 0, stream>>>(table);
        if (cb == 2) codebook_table_kernel<2><<<256, 256, 0, stream>>>(table);
        cuda_check(cudaPeekAtLastError());
    }
    return (const half2*) table;
}

// The scratch one wave of tiles needs, per tile: cost bytes (half) and history bytes. Returns the tiles a wave
// keeps resident, -1 for a K outside 1..8. A half-integer rate (KA + 0.5, mul1) passes half_k.
extern "C" int64_t exl3_quantize_scratch(int K, int cb, bool half_k, int64_t* cost_bytes, int64_t* edge_bytes)
{
    if (K < 1 || K > 8 || (half_k && (K > 3 || cb != 2))) return -1;
    const int64_t edges = 65536 >> K;
    if (half_k)
    {
        // quantize_tiles_frac: costs (2, edges) half, history (256, edges) short; batches are the caller's choice
        *cost_bytes = 2 * edges * 2;
        *edge_bytes = CC_TILE_LEN * edges * 2;
        int device, sms;
        cuda_check(cudaGetDevice(&device));
        cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return sms;
    }
    const Launch launch = launch_for(K, cb);
    *cost_bytes = 2 * edges * 2;
    *edge_bytes = launch.optimized ? CC_TILE_LEN * edges / (K == 1 ? 8 : 1) : CC_TILE_LEN * edges * 2;
    return launch.wave;
}

// `tiles` tiles of 256 f32 values to their encodings: `out` the quantized values, `idx` each position's state.
// The scratch holds `scratch_tiles` tiles of exl3_quantize_scratch's sizes. Returns 0, or -1 when nothing fits.
extern "C" int launch_exl3_quantize_tiles(
    const float* in,
    float* out,
    uint16_t* idx,
    half* costs,
    uint16_t* edges,
    int64_t tiles,
    int64_t scratch_tiles,
    int K,
    int cb,
    bool half_k,
    cudaStream_t stream)
{
    if (!tiles) return 0;
    if (K < 1 || K > 8 || scratch_tiles < 1) return -1;
    if (half_k)
    {
        fp_quantize_tiles_kernel kernel = nullptr;
        if (cb == 2 && K == 1) kernel = quantize_tiles_frac_kernel_a1_maaaa();
        if (cb == 2 && K == 2) kernel = quantize_tiles_frac_kernel_a2_maaaa();
        if (cb == 2 && K == 3) kernel = quantize_tiles_frac_kernel_a3_maaaa();
        if (!kernel) return -1;
        const int shmem = CC_TILE_LEN * sizeof(half) + 32 * sizeof(int) + 128;
        cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        for (int64_t i = 0; i < tiles; i += scratch_tiles)
        {
            const int bsz = (int) MIN(scratch_tiles, tiles - i);
            kernel<<<bsz, 512, shmem, stream>>>(in + CC_TILE_LEN * i, out + CC_TILE_LEN * i, idx + CC_TILE_LEN * i,
                                                 costs, edges, nullptr);
            cuda_check(cudaPeekAtLastError());
        }
        return 0;
    }
    const Launch launch = launch_for(K, cb);
    const int64_t batch = MIN(scratch_tiles, (int64_t) launch.wave);
    const half2* lut = launch.optimized && K == 6 ? codebook_table(cb, stream) : nullptr;
    for (int64_t i = 0; i < tiles; i += batch)
    {
        const int bsz = (int) MIN(batch, tiles - i);
        launch.kernel<<<bsz, launch.num_threads, launch.shmem, stream>>>(in + CC_TILE_LEN * i, out + CC_TILE_LEN * i,
                                                                         idx + CC_TILE_LEN * i, costs, edges, lut);
        cuda_check(cudaPeekAtLastError());
    }
    return 0;
}
