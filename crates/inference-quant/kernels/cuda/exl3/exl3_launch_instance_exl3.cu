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

// exl3_gemm.cu's multi-matrix key: the gemm key plus the slot counts
static uint64_t mgemm_autotune_hash(int size_m, int size_k, int size_n, int K, bool c_fp32, int device, int cc,
                                    int max_num_sms, int cb, int bszm_in, int bszm_out, bool half_k)
{
    uint64_t h = gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb, half_k);
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) MIN(bszm_in, 24));
    mix((uint64_t) MIN(bszm_out, 24));
    return h;
}

// exl3_mgemm_gr's indexed mode without torch, graph recording, weights or expert-range filtering: slot j multiplies
// A[j] (m x k, f16) by expert indices[j] (B_list, suh_list, svh_list hold each expert's device addresses) into C[j]
// (m x n, f32 or f16). A_had is scratch for slots x m x k. Returns the kernel shape, 0 when nothing ran, -1 when no
// kernel fits.
extern "C" int launch_exl3_mgemm(
    const half* A,
    const uint16_t* const* B_list,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int tile_u16,
    int cb,
    bool c_fp32,
    const half* const* suh_list,
    half* A_had,
    const half* const* svh_list,
    const int64_t* indices,
    int slots,
    cudaStream_t stream)
{
    if (!slots || !size_m || !size_n) return 0;
    const bool half_k = (tile_u16 % 16) != 0;
    int K = tile_u16 / 16;
    if ((half_k && (tile_u16 % 16 != 8 || cb != 2)) || K < 1 || K > 8 || !size_k) return -1;

    int device;
    cudaGetDevice(&device);
    int total_sms = DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);
    int smem_max = DevCtx::instance().get_smem_request(device);

    int bszm_in = slots;
    int bszm_out = slots;
    int min_index = -1;
    int max_index = -1;
    int num_tokens = 1;
    half* weights = nullptr;
    const int* size_n_list = nullptr;
    void** c_list = nullptr;
    const int* n_stride_list = nullptr;
    const int* had_src_list = nullptr;
    int num_had_src = 0;
    void* kernelArgs[] =
    {
        (void*)& A,
        (void*)& B_list,
        (void*)& C,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_list,
        (void*)& A_had,
        (void*)& svh_list,
        (void*)& indices,
        (void*)& weights,
        (void*)& bszm_in,
        (void*)& bszm_out,
        (void*)& min_index,
        (void*)& max_index,
        (void*)& num_tokens,
        (void*)& size_n_list,
        (void*)& c_list,
        (void*)& n_stride_list,
        (void*)& had_src_list,
        (void*)& num_had_src
    };

    uint64_t key = mgemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, total_sms, cb, bszm_in,
                                       bszm_out, half_k);
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
            fp_exl3_mgemm_kernel candidate = get_mgemm_kernel_ptr(K, shape, c_fp32, cb, half_k);
            if (!candidate) continue;
            int max_slices = MAX(size_k / exl3_gemm_tilesize_k_g[shape] * size_n / exl3_gemm_tilesize_n_g[shape], 1);
            int max_sms = MAX(MIN(max_slices, total_sms), 1);
            candidates.push_back({(void*) candidate, exl3_gemm_blockdim_g[shape], max_sms, slots, total_sms, shape});
        }
        if (candidates.empty()) return -1;
        tuned = CoopKernelAutotuner::launch(key, candidates, kernelArgs, smem_max, stream,
                                            (size_t) size_k * size_n * slots);
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }

    int block_dim;
    int shape_idx;
    int num_sms = total_sms;
    fp_exl3_mgemm_kernel kernel = select_exl3_mgemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        0, &block_dim, &shape_idx,
        &num_sms, cb, bszm_in, bszm_out, half_k
    );
    if (!kernel) return -1;
    int tiles = MAX(size_k / exl3_gemm_tilesize_k_g[shape_idx] * size_n / exl3_gemm_tilesize_n_g[shape_idx], 1);
    num_sms = tiles;
    if (num_sms * slots > total_sms) num_sms = MAX(total_sms / slots, 1);
    if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
    int concurrency = MIN(total_sms / num_sms, slots);
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
        kernel_attr_set[device].insert((void*) kernel);
    }
    EXL3_COOP_LAUNCH((void*) kernel, dim3(num_sms, 1, concurrency), block_dim, kernelArgs, smem_max, stream);
    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

// c_ptrs[j] = slice j's first output: its source's block (sources end to end from `base`, each m x its width) at the
// slice's column; `prefix` holds each slice's source offset in columns (summed widths of the sources before it)
__global__ void exl3_fill_slice_ptrs(void** c_ptrs, char* base, const int* prefix, const int* col, int slices, int m,
                                     int elem)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < slices) c_ptrs[j] = base + ((size_t) m * prefix[j] + col[j]) * elem;
}

// One input through `num_src` matrices (q, k and v), each with its own rotations, cut into `slices` equal column
// slices for an even spread: exl3_mgemm_gr's sliced mode. Device tables: B_list and svh_list per slice (pre-offset
// to the slice's first column), suh_list per source, and the ints [size_n (the slice width), n_stride (its source's
// width), had_src (its source), col (its first column), prefix (its source's offset)], `slices` apiece. The outputs
// land end to end from C, each source m x its width; c_ptrs is device scratch for `slices` addresses, filled here.
// A_had holds num_src x m x k. Returns as launch_exl3_mgemm.
extern "C" int launch_exl3_mgemm_sliced(
    const half* A,
    const uint16_t* const* B_list,
    const half* const* suh_list,
    const half* const* svh_list,
    const int* ints,
    int num_src,
    int slices,
    int slice_width,
    void* C,
    void** c_ptrs,
    int size_m,
    int size_k,
    int tile_u16,
    int cb,
    bool c_fp32,
    half* A_had,
    cudaStream_t stream)
{
    if (!slices || !size_m) return 0;
    const bool half_k = (tile_u16 % 16) != 0;
    int K = tile_u16 / 16;
    if ((half_k && (tile_u16 % 16 != 8 || cb != 2)) || K < 1 || K > 8 || !size_k) return -1;
    const int* size_n_list = ints;
    const int* n_stride_list = ints + slices;
    const int* had_src_list = ints + 2 * slices;
    exl3_fill_slice_ptrs<<<CEIL_DIVIDE(slices, 128), 128, 0, stream>>>(c_ptrs, (char*) C, ints + 4 * slices,
                                                                      ints + 3 * slices, slices, size_m,
                                                                      c_fp32 ? 4 : 2);
    cuda_check(cudaPeekAtLastError());

    int device;
    cudaGetDevice(&device);
    int total_sms = DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);
    int smem_max = DevCtx::instance().get_smem_request(device);

    int size_n = slice_width;
    int bszm_in = 1;
    int bszm_out = slices;
    const int64_t* indices = nullptr;
    half* weights = nullptr;
    int min_index = -1;
    int max_index = -1;
    int num_tokens = 1;
    void* kernelArgs[] =
    {
        (void*)& A,
        (void*)& B_list,
        (void*)& C,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_list,
        (void*)& A_had,
        (void*)& svh_list,
        (void*)& indices,
        (void*)& weights,
        (void*)& bszm_in,
        (void*)& bszm_out,
        (void*)& min_index,
        (void*)& max_index,
        (void*)& num_tokens,
        (void*)& size_n_list,
        (void*)& c_ptrs,
        (void*)& n_stride_list,
        (void*)& had_src_list,
        (void*)& num_src
    };

    // exl3_gemm.cu keys sliced launches apart from plain ones
    uint64_t key = mgemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, total_sms, cb, bszm_in,
                                       bszm_out, half_k) ^ 0x9e3779b97f4a7c15ull;
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
            fp_exl3_mgemm_kernel candidate = get_mgemm_kernel_ptr(K, shape, c_fp32, cb, half_k);
            if (!candidate) continue;
            int max_slices = MAX(size_k / exl3_gemm_tilesize_k_g[shape] * size_n / exl3_gemm_tilesize_n_g[shape], 1);
            int max_sms = MAX(MIN(max_slices, total_sms), 1);
            candidates.push_back({(void*) candidate, exl3_gemm_blockdim_g[shape], max_sms, slices, total_sms, shape});
        }
        if (candidates.empty()) return -1;
        tuned = CoopKernelAutotuner::launch(key, candidates, kernelArgs, smem_max, stream,
                                            (size_t) size_k * size_n * slices);
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }

    int block_dim;
    int shape_idx;
    int num_sms = total_sms;
    fp_exl3_mgemm_kernel kernel = select_exl3_mgemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        0, &block_dim, &shape_idx,
        &num_sms, cb, bszm_in, bszm_out, half_k
    );
    if (!kernel) return -1;
    int tiles = MAX(size_k / exl3_gemm_tilesize_k_g[shape_idx] * size_n / exl3_gemm_tilesize_n_g[shape_idx], 1);
    num_sms = tiles;
    if (num_sms * slices > total_sms) num_sms = MAX(total_sms / slices, 1);
    if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
    int concurrency = MIN(total_sms / num_sms, slices);
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
        kernel_attr_set[device].insert((void*) kernel);
    }
    EXL3_COOP_LAUNCH((void*) kernel, dim3(num_sms, 1, concurrency), block_dim, kernelArgs, smem_max, stream);
    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}
