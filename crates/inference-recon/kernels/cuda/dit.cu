// Fused elementwise passes of the DiT blocks (TRELLIS.2 in bf16, Idol in f16), keeping the unfused path's arithmetic:
// statistics and rotations in f32, with a rounding to the element type wherever that path casts back.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cmath>

#define LN_BLOCK 256

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float f(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T to(float x);
template <> __device__ __forceinline__ bf16 to<bf16>(float x) { return __float2bfloat16_rn(x); }
template <> __device__ __forceinline__ __half to<__half>(float x) { return __float2half_rn(x); }

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

// one block a row: LayerNorm (affine when `w` is given) rounded, then when `mod` is given the AdaLN modulation
// h * (1 + scale) + shift in the element type, scale and shift read from row `row / rows_per_batch` of `mod` (stride
// `mod_stride`) at `scale_at` / `shift_at`
template <typename T>
__device__ void ln_mod(const T* x, const float* w, const float* bias, const T* mod, int mod_stride, int scale_at,
                       int shift_at, T* out, int c, int rows_per_batch, float eps) {
    __shared__ float shared[32];
    const long row = blockIdx.x;
    const T* xr = x + row * c;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < c; i += blockDim.x) sum += f(xr[i]);
    const float mean = block_sum(sum, shared) / (float)c;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        const float d = f(xr[i]) - mean;
        sq += d * d;
    }
    const float denom = sqrtf(block_sum(sq, shared) / (float)c + eps);
    const T* m = mod ? mod + (row / rows_per_batch) * (long)mod_stride : nullptr;
    for (int i = threadIdx.x; i < c; i += blockDim.x) {
        float y = (f(xr[i]) - mean) / denom;
        if (w) y = y * w[i] + bias[i];
        T r = to<T>(y);
        if (m) {
            const T one_plus = to<T>(1.0f + f(m[scale_at + i]));
            r = to<T>(f(r) * f(one_plus));
            r = to<T>(f(r) + f(m[shift_at + i]));
        }
        out[row * c + i] = r;
    }
}

// one warp a (batch, token, head) row of `d` (even, up to 256; each lane an even share, the last lanes idle when
// they would be empty): read at `src + token_row * token_stride + offset +
// head * d` plus the linear's `bias` there (rounded); F.normalize'd times `gain` [heads, d] (when given) and rounded,
// and rotated by `cos` / `sin` (when given, adjacent channel pairs, pair p of (b, t, h) at `b * cos_b + t * cos_t +
// h * cos_h + p`) and rounded, the rotation first when `rope_first`; written to `out` [batch, heads, tokens, d]
template <typename T>
__device__ void head_norm_rope(const T* src, const T* bias, long token_stride, long offset, const float* gain,
                               const float* cos_t, const float* sin_t, long cos_b, long cos_tok, long cos_h,
                               int rope_first, T* out, int batch, int tokens, int heads, int d, float eps) {
    const long warp = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= (long)batch * tokens * heads) return;
    const int h = warp % heads;
    const long bt = warp / heads;
    const int t = bt % tokens;
    const int bi = bt / tokens;
    const T* s = src + bt * token_stride + offset + (long)h * d;
    T* o = out + (((long)bi * heads + h) * tokens + t) * d;
    // each lane owns adjacent pairs, so a rotation stays within the lane: d = 96 takes 4 a lane over 24 lanes
    int per = (d + 31) / 32;
    per += per & 1;
    const bool active = lane * per < d;
    float v[8];
    const T* bh = bias + offset + (long)h * d;
    for (int k = 0; k < per; k++) v[k] = active ? f(__hadd(s[lane * per + k], bh[lane * per + k])) : 0.0f;
    const float* cs = cos_t ? cos_t + bi * cos_b + t * cos_tok + h * cos_h : nullptr;
    const float* sn = sin_t ? sin_t + bi * cos_b + t * cos_tok + h * cos_h : nullptr;
    auto rotate = [&]() {
        if (!active) return;
        for (int k = 0; k < per; k += 2) {
            const int pair = (lane * per + k) / 2;
            const float c = cs[pair], s1 = sn[pair];
            const float e = v[k], od = v[k + 1];
            v[k] = f(to<T>(e * c - od * s1));
            v[k + 1] = f(to<T>(e * s1 + od * c));
        }
    };
    if (cs && rope_first) rotate();
    if (gain) {
        float sq = 0.0f;
        for (int k = 0; k < per; k++) sq += v[k] * v[k];
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, off);
        const float norm = fmaxf(sqrtf(sq), eps);
        if (active)
            for (int k = 0; k < per; k++) v[k] = f(to<T>(v[k] / norm * gain[h * d + lane * per + k]));
    }
    if (cs && !rope_first) rotate();
    if (active)
        for (int k = 0; k < per; k++) o[lane * per + k] = to<T>(v[k]);
}

// x + (y + bias) * gate (each product and sum rounded), the gate read from row `row / rows_per_batch` of `mod` at
// `gate_at`; x + y when `mod` is null
template <typename T>
__device__ void gate_residual(const T* x, const T* y, const T* bias, const T* mod, int mod_stride, int gate_at, T* out,
                              long n, int c, int rows_per_batch) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float yv = f(__hadd(y[i], bias[i % c]));
    if (mod) {
        const long row = i / c;
        yv = f(to<T>(yv * f(mod[(row / rows_per_batch) * (long)mod_stride + gate_at + i % c])));
    }
    out[i] = to<T>(f(x[i]) + yv);
}

// gelu(x + bias) (tanh form), the sum rounded and the activation from it in f32 as the unary kernel computes it; that
// kernel builds with FMA contraction and this file without, so its contracted product is an fmaf here
template <typename T> __device__ void bias_gelu(const T* x, const T* bias, long n, int c, T* out) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = f(__hadd(x[i], bias[i % c]));
    const float cube = v * v * v;
    const float alpha = fmaf(0.044715f, cube, v);
    out[i] = to<T>(0.5f * v * (1.0f + tanhf((float)(M_2_SQRTPI * M_SQRT1_2) * alpha)));
}

#define DIT_KERNELS(T, SUFFIX)                                                                                        \
    extern "C" __global__ void ln_mod_##SUFFIX(const T* x, const float* w, const float* bias, const T* mod,           \
                                               int mod_stride, int scale_at, int shift_at, T* out, int c,             \
                                               int rows_per_batch, float eps) {                                       \
        ln_mod<T>(x, w, bias, mod, mod_stride, scale_at, shift_at, out, c, rows_per_batch, eps);                      \
    }                                                                                                                 \
    extern "C" __global__ void head_norm_rope_##SUFFIX(const T* src, const T* bias, long token_stride, long offset,   \
                                                       const float* gain, const float* cos_t, const float* sin_t,     \
                                                       long cos_b, long cos_tok, long cos_h, int rope_first, T* out,  \
                                                       int batch, int tokens, int heads, int d, float eps) {          \
        head_norm_rope<T>(src, bias, token_stride, offset, gain, cos_t, sin_t, cos_b, cos_tok, cos_h, rope_first,   \
                          out, batch, tokens, heads, d, eps);                                                         \
    }                                                                                                                 \
    extern "C" __global__ void gate_residual_##SUFFIX(const T* x, const T* y, const T* bias, const T* mod,           \
                                                      int mod_stride, int gate_at, T* out, long n, int c,             \
                                                      int rows_per_batch) {                                           \
        gate_residual<T>(x, y, bias, mod, mod_stride, gate_at, out, n, c, rows_per_batch);                            \
    }                                                                                                                 \
    extern "C" __global__ void bias_gelu_##SUFFIX(const T* x, const T* bias, long n, int c, T* out) {                \
        bias_gelu<T>(x, bias, n, c, out);                                                                            \
    }

DIT_KERNELS(bf16, bf16)
DIT_KERNELS(__half, f16)
