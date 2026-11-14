// Our decode GEMV over the EXL3 trellis (m = 1), no grid sync: a block per (128-column group, k-split of 128 or 256
// rows) of up to four matrices sharing x, each warp one 16-column tile, its trellis words loaded up front while the
// block rotates its slice of x. Partial rows meet in the stream's workspace and the last block of a column group sums
// them in split order (the same bits every run), rotates them out (Hadamard, svh) and resets its counter.
// exllamav3's in-register window extraction and codebook decode (exl3_gemv_kernel.cuh).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_gemv_kernel.cuh"
#include "quant/exl3_devctx.cuh"

namespace exl3_dgemv {

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int GROUP_COLS = WARPS * 16;
constexpr float HAD_SCALE = 0.088388347648f;  // 1/sqrt(128)

enum OutKind { OUT_F32 = 0, OUT_F16 = 1, OUT_BF16 = 2 };

// 32 lanes x 4 values = one 128-wide Hadamard, scaled
__device__ __forceinline__ void had128(float& v0, float& v1, float& v2, float& v3, int lane)
{
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    v0 = s0 + s1; v1 = d0 + d1; v2 = s0 - s1; v3 = d0 - d1;
    shuffle_had_f2x32(v0, v1, lane);
    shuffle_had_f2x32(v2, v3, lane);
    v0 *= HAD_SCALE; v1 *= HAD_SCALE; v2 *= HAD_SCALE; v3 *= HAD_SCALE;
}

__device__ __forceinline__ void load4(const void* x, bool bf16, int i, float& v0, float& v1, float& v2, float& v3)
{
    if (bf16)
    {
        uint2 u = *((const uint2*) ((const __nv_bfloat16*) x + i));
        float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u.x));
        float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u.y));
        // rounded through f16 as the other kernels' inputs are
        v0 = __half2float(__float2half_rn(a.x)); v1 = __half2float(__float2half_rn(a.y));
        v2 = __half2float(__float2half_rn(b.x)); v3 = __half2float(__float2half_rn(b.y));
    }
    else
    {
        uint2 u = *((const uint2*) ((const half*) x + i));
        float2 a = __half22float2(*reinterpret_cast<half2*>(&u.x));
        float2 b = __half22float2(*reinterpret_cast<half2*>(&u.y));
        v0 = a.x; v1 = a.y; v2 = b.x; v3 = b.y;
    }
}

// m16n8k16, f16 inputs, f32 accumulate; only row 0 of A is real
__device__ __forceinline__ void mma_row0(half2 a0, half2 a2, const FragB& b, float* c)
{
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t r0 = *reinterpret_cast<uint32_t*>(&a0), r2 = *reinterpret_cast<uint32_t*>(&a2), z = 0;
    asm volatile
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(r0), "r"(z), "r"(r2), "r"(z), "r"(bb[0]), "r"(bb[1])
    );
}

// up to MAX_JOBS matrices of one k in a launch (q, k and v: one x, each its own rotations and width), their column
// groups end to end
constexpr int MAX_JOBS = 4;

struct Job
{
    const uint32_t* B;      // (k / 16) x (n / 16) tiles
    const half* suh;        // k
    const half* svh;        // n
    void* out;              // n, of out_kind
    int n;
    int group0;             // first column group, of all the launch's
    int col0;               // first column, of all the launch's
};

struct Params
{
    const void* x;          // k, f16 or bf16 (x_bf16)
    bool x_bf16;
    int size_k;
    int splits;
    int groups;             // all jobs' column groups
    int total_n;            // all jobs' columns
    int jobs;
    int out_kind;
    float* partial;         // splits x total_n
    int* counters;          // a counter per column group, left zeroed
    Job job[MAX_JOBS];
};

template <int bits, int cb, int KR>
__global__ __launch_bounds__(THREADS)
void dgemv_kernel(const __grid_constant__ Params p)
{
    constexpr int TWORDS = 8 * bits;    // uint32 a 16x16 tile
    constexpr int SLICES = KR / 16;     // 16-row k-slices a warp
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = blockIdx.x % p.groups;
    const int kz = blockIdx.x / p.groups;
    int j = 0;
    #pragma unroll
    for (int i = 1; i < MAX_JOBS; ++i)
        if (i < p.jobs && g >= p.job[i].group0) j = i;
    const Job& job = p.job[j];
    const int group = g - job.group0;
    const int size_n = job.n;
    const int splits = p.splits;
    const int ntiles = size_n / 16;
    const int tile = group * WARPS + warp;
    const int ks0 = kz * SLICES;

    // every trellis word of this warp's tile column in flight first
    uint32_t w[SLICES];
    const uint32_t* bp = job.B + ((size_t) ks0 * ntiles + tile) * TWORDS + lane;
    #pragma unroll
    for (int s = 0; s < SLICES; ++s)
        w[s] = lane < TWORDS ? __ldcs(bp + (size_t) s * ntiles * TWORDS) : 0;

    // this block's rows of x, * suh and rotated, as f16
    __shared__ half2 sx[KR / 2];
    __shared__ float red[GROUP_COLS];
    __shared__ float sh_out[GROUP_COLS];
    __shared__ bool last;
    for (int wc = warp; wc < KR / 128; wc += WARPS)
    {
        const int k0 = kz * KR + wc * 128 + lane * 4;
        float v0, v1, v2, v3;
        load4(p.x, p.x_bf16, k0, v0, v1, v2, v3);
        uint2 su = *((const uint2*) (job.suh + k0));
        float2 s01 = __half22float2(*reinterpret_cast<half2*>(&su.x));
        float2 s23 = __half22float2(*reinterpret_cast<half2*>(&su.y));
        v0 *= s01.x; v1 *= s01.y; v2 *= s23.x; v3 *= s23.y;
        had128(v0, v1, v2, v3, lane);
        sx[(wc * 128 + lane * 4) / 2] = __floats2half2_rn(v0, v1);
        sx[(wc * 128 + lane * 4) / 2 + 1] = __floats2half2_rn(v2, v3);
    }
    __syncthreads();

    // per-lane window sources (exl3_gemv_kernel's)
    [[maybe_unused]] int src_a = 0, src_b = 0, s2 = 0;
    if constexpr (bits == 2)
    {
        src_b = lane >> 1;
        src_a = (src_b + 15) & 15;
    }
    else if constexpr (bits == 3)
    {
        int b1 = ((lane << 3) + 257) * 3;
        int b2 = b1 + 21;
        int i2 = (b2 - 1) / 32;
        s2 = (i2 + 1) * 32 - b2;
        src_a = ((b1 - 16) / 32) % 24;
        src_b = i2 % 24;
    }

    float c0[4] = {}, c1[4] = {};
    const half2 hz = __half2half2(__ushort_as_half(0));
    #pragma unroll
    for (int s = 0; s < SLICES; ++s)
    {
        FragB f0, f1;
        if constexpr (bits == 4)
        {
            uint32_t aw = __shfl_sync(0xffffffffu, w[s], (lane + 31) & 31);
            exl3_gemv_ns::dq8_regs_4bits<cb>(aw, w[s], f0, f1);
        }
        else if constexpr (bits == 2)
        {
            uint32_t bw = __shfl_sync(0xffffffffu, w[s], src_b);
            uint32_t aw = __shfl_sync(0xffffffffu, w[s], src_a);
            exl3_gemv_ns::dq8_regs_2bits<cb>(aw, bw, lane << 3, f0, f1);
        }
        else
        {
            uint32_t aw = __shfl_sync(0xffffffffu, w[s], src_a);
            uint32_t bw = __shfl_sync(0xffffffffu, w[s], src_b);
            exl3_gemv_ns::dq8_regs_3bits<cb>(aw, bw, s2, f0, f1);
        }
        // A row 0: k pairs (2 (lane % 4), +1) and (+8, +9) of this slice
        half2 a0 = lane < 4 ? sx[s * 8 + lane] : hz;
        half2 a2 = lane < 4 ? sx[s * 8 + lane + 4] : hz;
        mma_row0(a0, a2, f0, c0);
        mma_row0(a0, a2, f1, c1);
    }

    // lanes 0..3 hold row 0: columns 2 (lane % 4) (+1) of each n8 half
    if (lane < 4)
    {
        red[warp * 16 + 2 * lane] = c0[0];
        red[warp * 16 + 2 * lane + 1] = c0[1];
        red[warp * 16 + 8 + 2 * lane] = c1[0];
        red[warp * 16 + 8 + 2 * lane + 1] = c1[1];
    }
    __syncthreads();
    const int col0 = group * GROUP_COLS;
    if (splits == 1)
    {
        if (threadIdx.x < GROUP_COLS) sh_out[threadIdx.x] = red[threadIdx.x];
    }
    else
    {
        // a row of partials per split, summed in split order by the last to arrive: the same sum every run
        float* part = p.partial + job.col0 + col0 + threadIdx.x;
        if (threadIdx.x < GROUP_COLS) __stcg(part + (size_t) kz * p.total_n, red[threadIdx.x]);
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(p.counters + g, 1) == splits - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
        if (threadIdx.x < GROUP_COLS)
        {
            float sum = 0.0f;
            for (int z = 0; z < splits; ++z) sum += __ldcg(part + (size_t) z * p.total_n);
            sh_out[threadIdx.x] = sum;
        }
        if (threadIdx.x == 0) p.counters[g] = 0;
    }
    __syncthreads();

    // output rotation, 128 columns a warp, * svh
    if (warp == 0)
    {
        float v0 = sh_out[lane * 4], v1 = sh_out[lane * 4 + 1], v2 = sh_out[lane * 4 + 2], v3 = sh_out[lane * 4 + 3];
        had128(v0, v1, v2, v3, lane);
        const int c = col0 + lane * 4;
        const int out_kind = p.out_kind;
        void* out = job.out;
        uint2 sv = *((const uint2*) (job.svh + c));
        float2 s01 = __half22float2(*reinterpret_cast<half2*>(&sv.x));
        float2 s23 = __half22float2(*reinterpret_cast<half2*>(&sv.y));
        v0 *= s01.x; v1 *= s01.y; v2 *= s23.x; v3 *= s23.y;
        if (out_kind == OUT_F32)
            *((float4*) ((float*) out + c)) = make_float4(v0, v1, v2, v3);
        else if (out_kind == OUT_BF16)
        {
            __nv_bfloat162 a = __floats2bfloat162_rn(v0, v1), b = __floats2bfloat162_rn(v2, v3);
            uint2 u;
            u.x = *reinterpret_cast<uint32_t*>(&a);
            u.y = *reinterpret_cast<uint32_t*>(&b);
            *((uint2*) ((__nv_bfloat16*) out + c)) = u;
        }
        else
        {
            half2 a = __floats2half2_rn(v0, v1), b = __floats2half2_rn(v2, v3);
            uint2 u;
            u.x = *reinterpret_cast<uint32_t*>(&a);
            u.y = *reinterpret_cast<uint32_t*>(&b);
            *((uint2*) ((half*) out + c)) = u;
        }
    }
}

typedef void (*dgemv_fn)(const Params);

static dgemv_fn select(int bits, int cb, int kr)
{
    #define SEL(b_, c_, k_) if (bits == b_ && cb == c_ && kr == k_) return dgemv_kernel<b_, c_, k_>;
    #define SEL_K(b_, c_) SEL(b_, c_, 128) SEL(b_, c_, 256)
    SEL_K(2, 0) SEL_K(2, 1) SEL_K(2, 2) SEL_K(3, 0) SEL_K(3, 1) SEL_K(3, 2) SEL_K(4, 0) SEL_K(4, 1) SEL_K(4, 2)
    #undef SEL_K
    #undef SEL
    return nullptr;
}

// k rows a block: 128 (more blocks, shorter serial chains), 256 past this many blocks a multiprocessor (fewer partials)
constexpr int KR_256_BLOCKS_PER_SM = 16;
// the workspace's leading words: a completion counter per column group
constexpr int DGEMV_COUNTERS = 4096;

}  // namespace exl3_dgemv

// One row x (f16, or bf16 with x_bf16) through `jobs` trellises of one k (B, suh, svh, out, n apiece; tiles of
// tile_u16 words): each out (f32, f16 or bf16 by out_kind) = had(had(x * suh) @ W) * svh. ws is the stream's workspace
// of ws_floats words, zeroed when made: counters first (left zeroed), then the partial rows. Returns 0 when launched,
// -1 for a shape or rate it does not take or a workspace too small.
extern "C" int launch_exl3_dgemv(const void* x, bool x_bf16, int jobs, const uint16_t* const* B, const half* const* suh,
                                 const half* const* svh, void* const* out, const int* n, int size_k, int tile_u16,
                                 int cb, float* ws, int64_t ws_floats, int out_kind, cudaStream_t stream)
{
    using namespace exl3_dgemv;
    if (jobs < 1 || jobs > MAX_JOBS || tile_u16 % 16 || size_k % 128) return -1;
    Params p = {};
    p.x = x;
    p.x_bf16 = x_bf16;
    p.size_k = size_k;
    p.jobs = jobs;
    p.out_kind = out_kind;
    for (int j = 0; j < jobs; ++j)
    {
        if (n[j] % GROUP_COLS) return -1;
        p.job[j] = {(const uint32_t*) B[j], suh[j], svh[j], out[j], n[j], p.groups, p.total_n};
        p.groups += n[j] / GROUP_COLS;
        p.total_n += n[j];
    }
    int device;
    cudaGetDevice(&device);
    const int sms = DevCtx::instance().get_num_sms(device);
    static int forced = -2;
    if (forced == -2)
    {
        const char* env = std::getenv("EXL3_DGEMV_KR");
        forced = env ? atoi(env) : 0;
    }
    int kr = forced ? forced : p.groups * (size_k / 128) > KR_256_BLOCKS_PER_SM * sms && size_k % 256 == 0 ? 256 : 128;
    if (size_k % kr) return -1;
    dgemv_fn kernel = select(tile_u16 / 16, cb, kr);
    if (!kernel) return -1;
    p.splits = size_k / kr;
    if (p.groups > DGEMV_COUNTERS || DGEMV_COUNTERS + (int64_t) p.splits * p.total_n > ws_floats) return -1;
    p.counters = (int*) ws;
    p.partial = ws + DGEMV_COUNTERS;
    kernel<<<p.groups * p.splits, THREADS, 0, stream>>>(p);
    cuda_check(cudaPeekAtLastError());
    return 0;
}
