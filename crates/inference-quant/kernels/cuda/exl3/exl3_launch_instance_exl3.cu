// The C ABI over exllamav3's EXL3 matmul: exl3_gemm_gr's CUDA path (third_party/exllamav3/quant/exl3_gemm.cu) without
// torch, graph recording, autotuning or the ROCm paths. C = A @ B, A rotated by suh and the 128-wide Hadamard on the
// way in and C by the Hadamard and svh on the way out, inside the kernels.
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <set>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemm_kernel.cuh"
#include "quant/exl3_devctx.cuh"
#include "quant/exl3_gemv.cuh"
#include "quant/coop_autotune.cuh"

int exl3_gemm_tilesize_k_g[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n_g[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim_g[] = {EXL3_GEMM_BLOCKDIM};

static std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

// A (m x k, f16), B the trellis ((k / 16) x (n / 16) x tile_u16 uint16), C (m x n, f16 or f32 with c_fp32); A_had
// scratch of A's size; cb 0 3INST, 1 mcg, 2 mul1. Returns which kernel ran (90 the GEMV, else the GEMM shape), 0 when
// nothing needed to run, -1 when no kernel fits.
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

    void* gemv_kernel = nullptr;
    if (exl3_gemv_try_launch(kernelArgs, size_m, size_k, size_n, K, half_k, cb, c_fp32, suh && A_had && svh,
                             device, stream, &gemv_kernel, false))
    {
        cuda_check(cudaPeekAtLastError());
        return 90;
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
