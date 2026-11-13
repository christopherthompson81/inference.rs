// The C ABI over exllamav3's EXL3 matmul: exl3_gemm_gr's CUDA path (third_party/exllamav3/quant/exl3_gemm.cu) without
// torch, graph recording or the ROCm paths. A shape not yet tuned takes the default kernel while its stream captures. C = A @ B, A rotated by suh and the 128-wide Hadamard on the
// way in and C by the Hadamard and svh on the way out, inside the kernels.
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <set>
#include <vector>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemm_kernel.cuh"
#include "quant/exl3_devctx.cuh"
#include "quant/exl3_gemv.cuh"
#include "quant/exl3_gemv_int8.cuh"
#include "quant/coop_autotune.cuh"

int exl3_gemm_tilesize_k_g[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n_g[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim_g[] = {EXL3_GEMM_BLOCKDIM};

static std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

static uint64_t roundup_pow2(uint64_t x)
{
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

// exl3_gemm.cu's key, so a shape tunes once per m bucket
static uint64_t gemm_autotune_hash(int size_m, int size_k, int size_n, int K, bool c_fp32, int device, int cc,
                                   int max_num_sms, int cb, bool half_k)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) (half_k ? 1 : 0));
    mix((uint64_t) (size_m <= 16 ? roundup_pow2(size_m) : MIN(CEIL_DIVIDE(size_m, 16) * 16, 64)));
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) K);
    mix(c_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) max_num_sms);
    mix((uint64_t) cb);
    return h;
}

// A (m x k, f16), B the trellis ((k / 16) x (n / 16) x tile_u16 uint16), C (m x n, f16 or f32 with c_fp32); A_had
// scratch of A's size; cb 0 3INST, 1 mcg, 2 mul1. Returns which kernel ran (89 the int8 GEMV, 90 the GEMV, else the
// GEMM shape), 0 when nothing needed to run, -1 when no kernel fits.
extern "C" int launch_exl3_gemm(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int tile_u16,
    int cb,
    bool c_fp32,
    const half* suh,
    half* A_had,
    const half* svh,
    cudaStream_t stream)
{
    if (!size_m || !size_n) return 0;
    const bool half_k = (tile_u16 % 16) != 0;
    int K = tile_u16 / 16;
    if ((half_k && (tile_u16 % 16 != 8 || cb != 2)) || K < 1 || K > 8) return -1;
    if (!size_k)
    {
        cuda_check(cudaMemsetAsync(C, 0, (size_t) size_m * size_n * (c_fp32 ? 4 : 2), stream));
        return 0;
    }

    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);
    int smem_max = DevCtx::instance().get_smem_request(device);

    void* kernelArgs[] =
    {
        (void*)& A,
        (void*)& B,
        (void*)& C,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh,
        (void*)& A_had,
        (void*)& svh
    };

    // mul1 at small m: activations quantized to int8 (EXL3_INT8_GEMV, exllamav3's default)
    if (cb == 2 && exl3_gemv_int8_enabled() &&
        exl3_gemv_int8(A, B, C, size_m, size_k, size_n, tile_u16, c_fp32, suh, A_had, svh, stream))
    {
        cuda_check(cudaPeekAtLastError());
        return 89;
    }

    void* gemv_kernel = nullptr;
    if (exl3_gemv_try_launch(kernelArgs, size_m, size_k, size_n, K, half_k, cb, c_fp32, suh && A_had && svh,
                             device, stream, &gemv_kernel, false))
    {
        cuda_check(cudaPeekAtLastError());
        return 90;
    }

    uint64_t key = gemm_autotune_hash(MAX(size_m, 2), size_k, size_n, K, c_fp32, device, cc, num_sms, cb, half_k);
    CoopAutotuneLaunch tuned;
    if (CoopKernelAutotuner::launch_locked(key, kernelArgs, smem_max, stream, &tuned))
    {
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }
    cudaStreamCaptureStatus capture;
    cuda_check(cudaStreamIsCapturing(stream, &capture));
    if (capture == cudaStreamCaptureStatusNone)
    {
        std::vector<CoopAutotuneCandidate> candidates;
        for (int shape = 1; shape <= EXL3_GEMM_NUM_SHAPES; ++shape)
        {
            if (!exl3_gemm_shape_compat(shape, size_m, size_k, size_n, K, half_k)) continue;
            fp_exl3_gemm_kernel candidate = get_gemm_kernel_ptr(K, shape, c_fp32, cb, half_k);
            if (!candidate) continue;
            int max_slices = MAX(size_k / exl3_gemm_tilesize_k_g[shape] * size_n / exl3_gemm_tilesize_n_g[shape], 1);
            int max_sms = MAX(MIN(max_slices, num_sms), 1);
            candidates.push_back({(void*) candidate, exl3_gemm_blockdim_g[shape], max_sms, 1, max_sms, shape});
        }
        if (candidates.empty()) return -1;
        tuned = CoopKernelAutotuner::launch(key, candidates, kernelArgs, smem_max, stream, (size_t) size_k * size_n);
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }

    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel = select_exl3_gemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        0, &block_dim, &shape_idx,
        &num_sms, cb, half_k
    );
    if (!kernel) return -1;
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
        kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }
    EXL3_COOP_LAUNCH((void*) kernel, num_sms, block_dim, kernelArgs, smem_max, stream);
    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}
