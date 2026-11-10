// Sparse convolution helpers: rows gathered by index into an im2col table, 16 bytes a thread.
#include <stdint.h>

// out[r] = src[index[r]], or zeros where index[r] == missing; rows are row_vecs 16-byte vectors
extern "C" __global__ void gather_rows(const uint4 *src, const uint32_t *index, uint64_t rows, uint32_t row_vecs,
                                       uint32_t missing, uint4 *out) {
    const uint64_t total = rows * row_vecs;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t r = i / row_vecs;
        const uint32_t v = (uint32_t)(i - r * row_vecs);
        const uint32_t idx = index[r];
        out[i] = idx == missing ? make_uint4(0, 0, 0, 0) : src[(uint64_t)idx * row_vecs + v];
    }
}

#define TAPS 27
#define CHILDREN 8

// a subdivided level's neighbour table from its parents': child k (parent rows[k] / 8, slot rows[k] % 8, bit 0 for x,
// 1 for y, 2 for z) finds its tap's neighbour in the parent, or the parent's neighbour, that holds it; slots[p * 8 + j]
// is parent p's child j or `children` when it was not kept, as `parent_missing` marks a missing parent neighbour
extern "C" __global__ void child_neighbors(const uint32_t *parent_nb, uint32_t parent_missing, const uint32_t *rows,
                                           const uint32_t *slots, uint32_t children, uint32_t *out) {
    const uint64_t total = (uint64_t)children * TAPS;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const uint32_t k = (uint32_t)(i / TAPS);
        const int v = (int)(i - (uint64_t)k * TAPS);
        const uint32_t p = rows[k] / CHILDREN, j = rows[k] % CHILDREN;
        const int d[3] = {v / 9 - 1, v / 3 % 3 - 1, v % 3 - 1};
        int parent_tap = 0, slot = 0;
        for (int a = 0; a < 3; a++) {
            const int t = (int)((j >> a) & 1) + d[a];
            const int up = t < 0 ? -1 : t >> 1;
            parent_tap = parent_tap * 3 + up + 1;
            slot |= (t & 1) << a;
        }
        const uint32_t q = parent_nb[(uint64_t)p * TAPS + parent_tap];
        out[i] = q == parent_missing ? children : slots[(uint64_t)q * CHILDREN + slot];
    }
}

// The decoder's elementwise passes fused, each rounding as the separate f16 ops do. The tensor kernels build with fused
// multiply-adds and this file without, so the products they fuse are written as fmaf here.
#include <cuda_fp16.h>

#define WARP 32

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float2 warp_sum(float2 a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    }
    return a;
}

// a row of x (plus bias, rounded, when given) LayerNorm'd in f32 as the tensor kernel does (block_size threads, sums and
// squares), times alpha plus beta, rounded, then silu'd and rounded again when `act` is set
extern "C" __global__ void ln_f16(const __half *x, const __half *bias, const float *alpha, const float *beta, int ncols,
                                  int block_size, float eps, int act, __half *out) {
    const uint64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half *xr = x + row * ncols;
    float2 mv = make_float2(0.f, 0.f);
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = __half2float(bias ? __hadd(xr[col], bias[col]) : xr[col]);
        mv.x += xi;
        mv.y = fmaf(xi, xi, mv.y);
    }
    mv = warp_sum(mv);
    if (block_size > WARP) {
        __shared__ float2 s_sum[32];
        const int warp_id = tid / WARP, lane_id = tid % WARP;
        if (lane_id == 0) s_sum[warp_id] = mv;
        __syncthreads();
        mv = s_sum[lane_id];
        mv = warp_sum(mv);
    }
    const float mean = mv.x / ncols;
    const float var = fmaf(-mean, mean, mv.y / ncols);
    const float inv_std = rsqrtf(var + eps);
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = __half2float(bias ? __hadd(xr[col], bias[col]) : xr[col]);
        const float lhs = (xi - mean) * inv_std;
        __half y = __float2half(fmaf(lhs, alpha[col], beta[col]));
        if (act) y = __float2half(silu(__half2float(y)));
        out[row * ncols + col] = y;
    }
}

// silu(x + bias), each rounded
extern "C" __global__ void bias_silu_f16(const __half *x, const __half *bias, uint64_t n, int c, __half *out) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        out[i] = __float2half(silu(__half2float(__hadd(x[i], bias[i % c]))));
    }
}

// residual + (x + bias), each rounded
extern "C" __global__ void bias_residual_f16(const __half *x, const __half *bias, const __half *residual, uint64_t n,
                                             int c, __half *out) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        out[i] = __hadd(residual[i], __hadd(x[i], bias[i % c]));
    }
}
