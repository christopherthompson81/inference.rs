// Fused elementwise passes of TRELLIS.2's DiT blocks in bf16, keeping the unfused path's arithmetic: statistics and
// rotations in f32, with a bf16 rounding wherever that path casts back.
#include <cuda_bf16.h>
#include <stdint.h>
#include <cmath>

#define LN_BLOCK 256

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 b(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float block_sum(float v, float* shared) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < (int)(blockDim.x >> 5); w++) total += shared[w];
    __syncthreads();
    return total;
}

// one block a row: LayerNorm (affine when `w` is given) rounded to bf16, then when `mod` is given the AdaLN
// modulation h * (1 + scale) + shift in bf16, scale and shift read from row `row / rows_per_batch` of `mod` (stride
// `mod_stride`) at `scale_at` / `shift_at`
extern "C" __global__ void ln_mod_bf16(const bf16* x, const float* w, const float* bias, const bf16* mod, int mod_stride,
                                       int scale_at, int shift_at, bf16* out, int c, int rows_per_batch, float eps) {
    __shared__ float shared[32];
    const long row = blockIdx.x;
    const bf16* xr = x + row * c;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < c; i += blockDim.x) sum += f(xr[i]);
    const float mean = block_sum(sum, shared) / (float)c;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float d = f(xr[i]) - mean;
        sq += d * d;
    }
    const float denom = sqrtf(block_sum(sq, shared) / (float)c + eps);
    const bf16* m = mod ? mod + (row / rows_per_batch) * (long)mod_stride : nullptr;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        float y = (f(xr[i]) - mean) / denom;
        if (w) y = y * w[i] + bias[i];
        bf16 r = b(y);
        if (m) {
            const bf16 one_plus = b(1.0f + f(m[scale_at + i]));
            r = b(f(r) * f(one_plus));
            r = b(f(r) + f(m[shift_at + i]));
        }
        out[row * c + i] = r;
    }
}

// one warp a (batch, token, head) row of `d` (a multiple of 64): read at `src + token_row * token_stride + offset +
// head * d` plus the linear's `bias` there (rounded), F.normalize'd times `gain` [heads, d] (when given) and rounded, then rotated by `cos` / `sin` [tokens, d/2]
// (when given, adjacent channel pairs) and rounded, written to `out` [batch, heads, tokens, d]
extern "C" __global__ void head_norm_rope_bf16(const bf16* src, const bf16* bias, long token_stride, long offset,
                                               const float* gain,
                                               const float* cos_t, const float* sin_t, bf16* out, int batch,
                                               int tokens, int heads, int d, float eps) {
    const long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= (long)batch * tokens * heads) return;
    const int h = warp % heads;
    const long bt = warp / heads;
    const int t = bt % tokens;
    const int bi = bt / tokens;
    const bf16* s = src + bt * token_stride + offset + (long)h * d;
    bf16* o = out + (((long)bi * heads + h) * tokens + t) * d;
    const int per = d / 32;
    // each lane owns adjacent pairs, so a rotation stays within the lane
    float v[8];
    const bf16* bh = bias + offset + (long)h * d;
    for (int k = 0; k < per; k++) v[k] = f(__hadd(s[lane * per + k], bh[lane * per + k]));
    if (gain) {
        float sq = 0.0f;
        for (int k = 0; k < per; k++) sq += v[k] * v[k];
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, off);
        const float norm = fmaxf(sqrtf(sq), eps);
        for (int k = 0; k < per; k++) v[k] = f(b(v[k] / norm * gain[h * d + lane * per + k]));
    }
    if (cos_t) {
        for (int k = 0; k < per; k += 2) {
            const int pair = (lane * per + k) / 2;
            const float c = cos_t[(long)t * (d / 2) + pair], sn = sin_t[(long)t * (d / 2) + pair];
            const float e = v[k], od = v[k + 1];
            v[k] = e * c - od * sn;
            v[k + 1] = e * sn + od * c;
        }
    }
    for (int k = 0; k < per; k++) o[lane * per + k] = b(v[k]);
}

// x + (y + bias) * gate in bf16 (each product and sum rounded), the gate read from row `row / rows_per_batch` of `mod` at
// `gate_at`; x + y when `mod` is null
extern "C" __global__ void gate_residual_bf16(const bf16* x, const bf16* y, const bf16* bias, const bf16* mod,
                                              int mod_stride, int gate_at,
                                              bf16* out, long n, int c, int rows_per_batch) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float yv = f(__hadd(y[i], bias[i % c]));
    if (mod) {
        const long row = i / c;
        yv = f(b(yv * f(mod[(row / rows_per_batch) * (long)mod_stride + gate_at + i % c])));
    }
    out[i] = b(f(x[i]) + yv);
}

// gelu(x + bias) (tanh form), the sum rounded and the activation from it in f32 as the unary kernel computes it; that
// kernel builds with FMA contraction and this file without, so its contracted product is an fmaf here
extern "C" __global__ void bias_gelu_bf16(const bf16* x, const bf16* bias, long n, int c, bf16* out) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = f(__hadd(x[i], bias[i % c]));
    const float cube = v * v * v;
    const float alpha = fmaf(0.044715f, cube, v);
    out[i] = b(0.5f * v * (1.0f + tanhf((float)(M_2_SQRTPI * M_SQRT1_2) * alpha)));
}
