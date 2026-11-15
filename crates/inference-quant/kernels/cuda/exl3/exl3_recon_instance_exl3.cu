// exllamav3's reconstruct kernel and 128-wide row Hadamards (third_party/exllamav3), and our C ABI over them: a
// prefill matmul as the rotated input times the reconstructed f16 weight (cuBLAS, by the caller), rotated out
#include "quant/reconstruct.cu"
#include "quant/hadamard_inner.cuh"

// 1 / sqrt(128)
#define HAD_R_SCALE 0.088388347648f

// one warp per (row, 128-column group): x (f16, or bf16 with BF16_IN) times suh, rotated, f16 out
template <bool BF16_IN>
__global__ __launch_bounds__(32) void exl3_had_in_kernel(const half* x, half* out, const half* suh)
{
    size_t off = (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<true, false, BF16_IN>(x + off, out + off, suh, HAD_R_SCALE);
}

// as above for the output: f32 rotated, times svh, in place or to bf16
__global__ __launch_bounds__(32) void exl3_had_out_kernel(float* y, const half* svh)
{
    size_t off = (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner<false, true>(y + off, y + off, svh, HAD_R_SCALE);
}

__global__ __launch_bounds__(32) void exl3_had_out_bf16_kernel(const float* y, __nv_bfloat16* out, const half* svh)
{
    size_t off = (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_fb_r_128_inner<false, true>(y + off, out + off, svh, HAD_R_SCALE);
}

// `rows x k` x (f16, or bf16 with `x_bf16`) times `suh`, rotated, into `out` (f16); k a multiple of 128
extern "C" int launch_exl3_had_in(const void* x, bool x_bf16, half* out, const half* suh, int rows, int k,
                                  cudaStream_t stream)
{
    if (k % 128 || rows < 1) return -1;
    dim3 grid(rows, k / 128);
    if (x_bf16) exl3_had_in_kernel<true><<<grid, 32, 0, stream>>>((const half*) x, out, suh);
    else exl3_had_in_kernel<false><<<grid, 32, 0, stream>>>((const half*) x, out, suh);
    return 0;
}

// `rows x n` f32 y rotated, times `svh`: in place, or into `out_bf16` when it is not null
extern "C" int launch_exl3_had_out(float* y, __nv_bfloat16* out_bf16, const half* svh, int rows, int n,
                                   cudaStream_t stream)
{
    if (n % 128 || rows < 1) return -1;
    dim3 grid(rows, n / 128);
    if (out_bf16) exl3_had_out_bf16_kernel<<<grid, 32, 0, stream>>>(y, out_bf16, svh);
    else exl3_had_out_kernel<<<grid, 32, 0, stream>>>(y, svh);
    return 0;
}

// the trellis (`k / 16 x n / 16` tiles of `words` uint16) as its f16 weight `k x n` in the rotated basis (no suh, svh
// or Hadamards); cb as exllamav3's (0 3inst, 1 mcg, 2 mul1); -1 for a rate or shape it does not take
extern "C" int launch_exl3_reconstruct(half* out, const uint16_t* trellis, int k, int n, int words, int cb,
                                       cudaStream_t stream)
{
    if (k % 16 || n % 128 || cb < 0 || cb > 2) return -1;
    bool half_k = words % 16 == 8;
    int bits = words / 16;
    if (words % 16 != 0 && !half_k) return -1;
    if (bits < 1 || bits > 8 || (half_k && (cb != 2 || bits > 3))) return -1;
    dim3 grid(n / 128, k / 16);
    int tiles_n = n / 16;
    #define RECON(K, CB) reconstruct_kernel<K, CB><<<grid, 256, 0, stream>>>(out, trellis, tiles_n, 0)
    #define RECON_CB(K) \
        if (cb == 0) RECON(K, 0); else if (cb == 1) RECON(K, 1); else RECON(K, 2)
    if (half_k)
    {
        if (bits == 1) reconstruct_kernel<1, 2, true><<<grid, 256, 0, stream>>>(out, trellis, tiles_n, 0);
        else if (bits == 2) reconstruct_kernel<2, 2, true><<<grid, 256, 0, stream>>>(out, trellis, tiles_n, 0);
        else reconstruct_kernel<3, 2, true><<<grid, 256, 0, stream>>>(out, trellis, tiles_n, 0);
        return 0;
    }
    switch (bits)
    {
        case 1: RECON_CB(1); break;
        case 2: RECON_CB(2); break;
        case 3: RECON_CB(3); break;
        case 4: RECON_CB(4); break;
        case 5: RECON_CB(5); break;
        case 6: RECON_CB(6); break;
        case 7: RECON_CB(7); break;
        case 8: RECON_CB(8); break;
    }
    #undef RECON_CB
    #undef RECON
    return 0;
}
