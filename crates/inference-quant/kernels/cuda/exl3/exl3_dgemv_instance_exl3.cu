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

// m16n8k16, f16 inputs, f32 accumulate; rows 8..15 of A zero
constexpr int MAX_ROWS = 8;
// half2s between rows of the rotated input: KR / 2 is a multiple of 32 banks, so 4 more spread a warp's 8 rows over them
__host__ __device__ constexpr int sx_stride(int kr) { return kr / 2 + 4; }
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

// uint32 loads a lane makes for one 16x16 tile: a tile is 8 x bits words
__host__ __device__ constexpr int tile_loads(int bits) { return (8 * bits + 31) / 32; }

// every trellis word of a warp's tile column over one k-split, issued at once (lanes past the tile's words idle)
template <int bits, int KR>
__device__ __forceinline__ void load_tiles(uint32_t (&w)[KR / 16 * tile_loads(bits)], const uint32_t* B, int ntiles,
                                           int tile, int kz, int lane)
{
    constexpr int TILE_WORDS = 8 * bits;
    constexpr int L = tile_loads(bits);
    const uint32_t* bp = B + ((size_t) kz * (KR / 16) * ntiles + tile) * TILE_WORDS + lane;
    #pragma unroll
    for (int s = 0; s < KR / 16; ++s)
        #pragma unroll
        for (int l = 0; l < L; ++l)
            w[s * L + l] = lane + 32 * l < TILE_WORDS ? __ldcs(bp + (size_t) s * ntiles * TILE_WORDS + 32 * l) : 0;
}

// `rows` rows (up to 8; row r of sx at r * sx_stride(KR): the split's KR rotated f16 values) times the warp's loaded tiles:
// row r's 16 columns into red[r * GROUP_COLS ..]. Past 4 bpw the words go through `stage` (the warp's KR / 16 tiles
// of shared memory) to exllamav3's generic decode
template <int bits, int cb, int KR>
__device__ __forceinline__ void dot_tiles(const uint32_t (&w)[KR / 16 * tile_loads(bits)], const half2* sx, int rows,
                                          int lane, float* red, uint32_t* stage)
{
    constexpr int TILE_WORDS = 8 * bits;
    constexpr int L = tile_loads(bits);
    if constexpr (bits > 4)
    {
        #pragma unroll
        for (int s = 0; s < KR / 16; ++s)
            #pragma unroll
            for (int l = 0; l < L; ++l)
                if (lane + 32 * l < TILE_WORDS) stage[s * TILE_WORDS + lane + 32 * l] = w[s * L + l];
        __syncwarp();
    }
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
    for (int s = 0; s < KR / 16; ++s)
    {
        FragB f0, f1;
        if constexpr (bits > 4)
            dq_dispatch<bits, cb>(stage + s * TILE_WORDS, lane << 3, f0, f1);
        else if constexpr (bits == 4)
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
        // A row lane / 4: k pairs (2 (lane % 4), +1) and (+8, +9) of this slice; rows 8..15 stay zero
        const half2* row = sx + (lane / 4) * sx_stride(KR);
        half2 a0 = lane / 4 < rows ? row[s * 8 + lane % 4] : hz;
        half2 a2 = lane / 4 < rows ? row[s * 8 + lane % 4 + 4] : hz;
        mma_row0(a0, a2, f0, c0);
        mma_row0(a0, a2, f1, c1);
    }

    // lane holds row lane / 4: columns 2 (lane % 4) (+1) of each n8 half
    if (lane / 4 < rows)
    {
        float* r = red + (lane / 4) * GROUP_COLS;
        r[2 * (lane % 4)] = c0[0];
        r[2 * (lane % 4) + 1] = c0[1];
        r[8 + 2 * (lane % 4)] = c1[0];
        r[8 + 2 * (lane % 4) + 1] = c1[1];
    }
}

// x[k0 .. k0 + 4) of one lane * suh, then the warp's 128 values rotated into sx (f16)
__device__ __forceinline__ void rotate_in(const void* x, bool x_bf16, const half* suh, int k0, half2* sx, int lane,
                                          size_t row0 = 0)
{
    float v0, v1, v2, v3;
    load4(x, x_bf16, row0 + k0, v0, v1, v2, v3);
    uint2 su = *((const uint2*) (suh + k0));
    float2 s01 = __half22float2(*reinterpret_cast<half2*>(&su.x));
    float2 s23 = __half22float2(*reinterpret_cast<half2*>(&su.y));
    v0 *= s01.x; v1 *= s01.y; v2 *= s23.x; v3 *= s23.y;
    had128(v0, v1, v2, v3, lane);
    sx[lane * 2] = __floats2half2_rn(v0, v1);
    sx[lane * 2 + 1] = __floats2half2_rn(v2, v3);
}

// the 128 values of v (shared) rotated out (Hadamard, * svh[0..128)) by one warp: lane's 4 into o
__device__ __forceinline__ void rotate_out(const float* v, const half* svh, int lane, float (&o)[4])
{
    float v0 = v[lane * 4], v1 = v[lane * 4 + 1], v2 = v[lane * 4 + 2], v3 = v[lane * 4 + 3];
    had128(v0, v1, v2, v3, lane);
    uint2 sv = *((const uint2*) (svh + lane * 4));
    float2 s01 = __half22float2(*reinterpret_cast<half2*>(&sv.x));
    float2 s23 = __half22float2(*reinterpret_cast<half2*>(&sv.y));
    o[0] = v0 * s01.x; o[1] = v1 * s01.y; o[2] = v2 * s23.x; o[3] = v3 * s23.y;
}

__device__ __forceinline__ void store4(void* out, int out_kind, size_t i, const float (&o)[4])
{
    if (out_kind == OUT_F32)
        *((float4*) ((float*) out + i)) = make_float4(o[0], o[1], o[2], o[3]);
    else if (out_kind == OUT_BF16)
    {
        __nv_bfloat162 a = __floats2bfloat162_rn(o[0], o[1]), b = __floats2bfloat162_rn(o[2], o[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *((uint2*) ((__nv_bfloat16*) out + i)) = u;
    }
    else
    {
        half2 a = __floats2half2_rn(o[0], o[1]), b = __floats2half2_rn(o[2], o[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *((uint2*) ((half*) out + i)) = u;
    }
}

// the block's last arrival for its counter, after publishing its stores; resets the counter when it is
__device__ __forceinline__ bool arrive_last(int* counter, int arrivals, bool* flag)
{
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *flag = atomicAdd(counter, 1) == arrivals - 1;
        if (*flag) *counter = 0;
    }
    __syncthreads();
    if (*flag) __threadfence();
    return *flag;
}

// up to MAX_JOBS matrices of one k in a launch (q, k and v: one x, each its own rotations and width), their column
// groups end to end
constexpr int MAX_JOBS = 4;

struct Job
{
    const uint32_t* B;      // (k / 16) x (n / 16) tiles
    const half* suh;        // k
    const half* svh;        // n
    void* out;              // rows x n, of out_kind
    int n;
    int group0;             // first column group, of all the launch's
    int col0;               // first column, of all the launch's
};

struct Params
{
    const void* x;          // rows x k, f16 or bf16 (x_bf16)
    bool x_bf16;
    int size_k;
    int rows;               // 1 to MAX_ROWS
    int splits;
    int groups;             // all jobs' column groups
    int total_n;            // all jobs' columns
    int jobs;
    int out_kind;
    float* partial;         // splits x rows x total_n
    int* counters;          // a counter per column group, left zeroed
    const half2* xr;        // past one row: jobs x rows x k / 2, each job's rows rotated once (null: blocks rotate)
    Job job[MAX_JOBS];
};

// each job's rows of x rotated (x * suh, then 128-wide Hadamards) into p.xr, a warp a 128-wide piece: the same values
// the GEMV's blocks would each rotate for themselves
__global__ __launch_bounds__(THREADS)
void rotate_rows_kernel(const __grid_constant__ Params p, half2* xr)
{
    const int lane = threadIdx.x % 32;
    const int pieces = p.size_k / 128;
    const int wc = blockIdx.x * WARPS + threadIdx.x / 32;
    if (wc >= p.jobs * p.rows * pieces) return;
    const int piece = wc % pieces, jr = wc / pieces;
    const int j = jr / p.rows, r = jr % p.rows;
    rotate_in(p.x, p.x_bf16, p.job[j].suh, piece * 128 + lane * 4, xr + (size_t) jr * (p.size_k / 2) + piece * 64,
              lane, (size_t) r * p.size_k);
}

template <int bits, int cb, int KR>
__global__ __launch_bounds__(THREADS)
void dgemv_kernel(const __grid_constant__ Params p)
{
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
    const int splits = p.splits;
    const int ntiles = job.n / 16;
    const int tile = group * WARPS + warp;

    uint32_t w[KR / 16 * tile_loads(bits)];
    load_tiles<bits, KR>(w, job.B, ntiles, tile, kz, lane);

    const int rows = p.rows;
    __shared__ half2 sx[MAX_ROWS * sx_stride(KR)];
    __shared__ float red[MAX_ROWS * GROUP_COLS];
    __shared__ bool last;
    if (p.xr)
    {
        // 16-byte copies of the split's slice of each pre-rotated row (sx_stride(KR) keeps rows 16-byte aligned)
        constexpr int ROW_U4 = KR / 8;
        const uint4* src = (const uint4*) (p.xr + (size_t) j * rows * (p.size_k / 2) + kz * (KR / 2));
        for (int i = threadIdx.x; i < rows * ROW_U4; i += THREADS)
        {
            const int r = i / ROW_U4, q = i % ROW_U4;
            *((uint4*) (sx + r * sx_stride(KR)) + q) = __ldcg(src + (size_t) r * (p.size_k / 8) + q);
        }
    }
    else
        for (int wc = warp; wc < rows * (KR / 128); wc += WARPS)
        {
            const int r = wc / (KR / 128), c = wc % (KR / 128);
            rotate_in(p.x, p.x_bf16, job.suh, kz * KR + c * 128 + lane * 4, sx + r * sx_stride(KR) + c * 64, lane,
                      (size_t) r * p.size_k);
        }
    __syncthreads();
    __shared__ uint32_t stage[bits > 4 ? WARPS * KR / 16 * 8 * bits : 1];
    dot_tiles<bits, cb, KR>(w, sx, rows, lane, red + warp * 16, stage + (bits > 4 ? warp * KR / 16 * 8 * bits : 0));
    __syncthreads();

    const int col0 = group * GROUP_COLS;
    if (splits > 1)
    {
        // a row of partials per split and row, summed in split order by the last to arrive: the same sum every run
        float* part = p.partial + job.col0 + col0;
        for (int i = threadIdx.x; i < rows * GROUP_COLS; i += THREADS)
            __stcg(part + ((size_t) kz * rows + i / GROUP_COLS) * p.total_n + i % GROUP_COLS, red[i]);
        if (!arrive_last(p.counters + g, splits, &last)) return;
        for (int i = threadIdx.x; i < rows * GROUP_COLS; i += THREADS)
        {
            float sum = 0.0f;
            for (int z = 0; z < splits; ++z)
                sum += __ldcg(part + ((size_t) z * rows + i / GROUP_COLS) * p.total_n + i % GROUP_COLS);
            red[i] = sum;
        }
        __syncthreads();
    }
    if (warp < rows)
    {
        float o[4];
        rotate_out(red + warp * GROUP_COLS, job.svh + col0, lane, o);
        store4(job.out, p.out_kind, (size_t) warp * job.n + col0 + lane * 4, o);
    }
}

// A MoE layer's routed experts for one token: gate and up (kernel A, whose last block of each slot's column group
// rotates both out, applies silu(gate) * up and rotates that into the down projection's input), then down (kernel B,
// whose last block of a slot's column group rotates out and weights it, and whose last slot of a column group sums
// the slots in order into the token's row)
struct MoeParams
{
    const void* x;          // tokens x H, f16 or bf16 (x_bf16)
    const void* sel;        // tokens x topk, int64 or uint32 (sel_u32)
    const void* rw;         // tokens x topk, f16 or f32 (rw_f32)
    bool x_bf16;
    bool sel_u32;
    bool rw_f32;
    int out_kind;
    const int64_t* gate;    // each projection's table: experts trellis, then suh, then svh addresses
    const int64_t* up;
    const int64_t* down;
    int experts;
    int topk;
    int tokens;             // 1 to MAX_ROWS; slot j is token j / topk's
    int H;
    int I;
    int splits_a;
    int splits_b;
    float* part_a;          // splits_a x slots x 2 x I
    half* act;              // slots x I, the down projection's rotated input
    float* part_b;          // splits_b x slots x H
    float* slot_out;        // slots x H, each slot's weighted output
    int* ctr_a;             // slots x I / 128
    int* ctr_b;             // slots x H / 128
    int* ctr_t;             // tokens x H / 128
    void* out;              // tokens x H
};

__device__ __forceinline__ int moe_expert(const MoeParams& p, int slot)
{
    return p.sel_u32 ? (int) ((const uint32_t*) p.sel)[slot] : (int) ((const int64_t*) p.sel)[slot];
}

template <int bits, int cb, int KR>
__global__ __launch_bounds__(THREADS)
void moe_a_kernel(const __grid_constant__ MoeParams p)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ig = p.I / GROUP_COLS;
    const int slots = p.tokens * p.topk;
    const int groups = slots * 2 * ig;
    const int g = blockIdx.x % groups;
    const int kz = blockIdx.x / groups;
    const int slot = g / (2 * ig);
    const int proj = g / ig % 2;
    const int group = g % ig;
    const int e = moe_expert(p, slot);
    const int64_t* table = proj ? p.up : p.gate;

    uint32_t w[KR / 16 * tile_loads(bits)];
    load_tiles<bits, KR>(w, (const uint32_t*) table[e], p.I / 16, group * WARPS + warp, kz, lane);

    __shared__ half2 sx[KR / 2];
    __shared__ float red[2][GROUP_COLS];
    __shared__ bool last;
    for (int wc = warp; wc < KR / 128; wc += WARPS)
        rotate_in(p.x, p.x_bf16, (const half*) table[p.experts + e], kz * KR + wc * 128 + lane * 4, sx + wc * 64, lane,
                  (size_t) (slot / p.topk) * p.H);
    __syncthreads();
    dot_tiles<bits, cb, KR>(w, sx, 1, lane, red[0] + warp * 16, nullptr);
    __syncthreads();

    const int stride = slots * 2 * p.I;
    const int col0 = group * GROUP_COLS;
    float* part = p.part_a + (size_t) (slot * 2 + proj) * p.I + col0 + threadIdx.x;
    if (threadIdx.x < GROUP_COLS) __stcg(part + (size_t) kz * stride, red[0][threadIdx.x]);
    if (!arrive_last(p.ctr_a + slot * ig + group, 2 * p.splits_a, &last)) return;
    if (threadIdx.x < 2 * GROUP_COLS)
    {
        // gate's columns in red[0], up's in red[1]
        const int q = threadIdx.x / GROUP_COLS;
        const int c = threadIdx.x % GROUP_COLS;
        const float* src = p.part_a + (size_t) (slot * 2 + q) * p.I + col0 + c;
        float sum = 0.0f;
        for (int z = 0; z < p.splits_a; ++z) sum += __ldcg(src + (size_t) z * stride);
        red[q][c] = sum;
    }
    __syncthreads();
    if (warp == 0)
    {
        float og[4], ou[4];
        rotate_out(red[0], (const half*) p.gate[2 * p.experts + e] + col0, lane, og);
        rotate_out(red[1], (const half*) p.up[2 * p.experts + e] + col0, lane, ou);
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = og[i] / (1.0f + __expf(-og[i])) * ou[i];
        // into the down projection's input: * its suh, rotated, as f16
        const int c = col0 + lane * 4;
        uint2 su = *((const uint2*) ((const half*) p.down[p.experts + e] + c));
        float2 s01 = __half22float2(*reinterpret_cast<half2*>(&su.x));
        float2 s23 = __half22float2(*reinterpret_cast<half2*>(&su.y));
        v[0] *= s01.x; v[1] *= s01.y; v[2] *= s23.x; v[3] *= s23.y;
        had128(v[0], v[1], v[2], v[3], lane);
        half2* dst = (half2*) (p.act + (size_t) slot * p.I + c);
        dst[0] = __floats2half2_rn(v[0], v[1]);
        dst[1] = __floats2half2_rn(v[2], v[3]);
    }
}

template <int bits, int cb, int KR>
__global__ __launch_bounds__(THREADS)
void moe_b_kernel(const __grid_constant__ MoeParams p)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int hg = p.H / GROUP_COLS;
    const int slots = p.tokens * p.topk;
    const int groups = slots * hg;
    const int g = blockIdx.x % groups;
    const int kz = blockIdx.x / groups;
    const int slot = g / hg;
    const int group = g % hg;
    const int e = moe_expert(p, slot);

    uint32_t w[KR / 16 * tile_loads(bits)];
    load_tiles<bits, KR>(w, (const uint32_t*) p.down[e], p.H / 16, group * WARPS + warp, kz, lane);

    __shared__ half2 sx[KR / 2];
    __shared__ float red[GROUP_COLS];
    __shared__ bool last;
    const half2* act = (const half2*) (p.act + (size_t) slot * p.I + kz * KR);
    for (int i = threadIdx.x; i < KR / 2; i += THREADS) sx[i] = act[i];
    __syncthreads();
    dot_tiles<bits, cb, KR>(w, sx, 1, lane, red + warp * 16, nullptr);
    __syncthreads();

    const int stride = slots * p.H;
    const int col0 = group * GROUP_COLS;
    float* part = p.part_b + (size_t) slot * p.H + col0 + threadIdx.x;
    if (threadIdx.x < GROUP_COLS) __stcg(part + (size_t) kz * stride, red[threadIdx.x]);
    if (!arrive_last(p.ctr_b + slot * hg + group, p.splits_b, &last)) return;
    if (threadIdx.x < GROUP_COLS)
    {
        float sum = 0.0f;
        for (int z = 0; z < p.splits_b; ++z) sum += __ldcg(part + (size_t) z * stride);
        red[threadIdx.x] = sum;
    }
    __syncthreads();
    if (warp == 0)
    {
        float o[4];
        rotate_out(red, (const half*) p.down[2 * p.experts + e] + col0, lane, o);
        const float rw = p.rw_f32 ? ((const float*) p.rw)[slot] : __half2float(((const half*) p.rw)[slot]);
        float* dst = p.slot_out + (size_t) slot * p.H + col0 + lane * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) __stcg(dst + i, rw * o[i]);
    }
    const int token = slot / p.topk;
    if (!arrive_last(p.ctr_t + token * hg + group, p.topk, &last)) return;
    if (warp == 0)
    {
        // the token's slots in order: the same sum every run
        float o[4] = {};
        for (int j = token * p.topk; j < (token + 1) * p.topk; ++j)
        {
            const float* src = p.slot_out + (size_t) j * p.H + col0 + lane * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[i] += __ldcg(src + i);
        }
        store4(p.out, p.out_kind, (size_t) token * p.H + col0 + lane * 4, o);
    }
}

typedef void (*moe_fn)(const MoeParams);

typedef void (*dgemv_fn)(const Params);

static dgemv_fn select(int bits, int cb, int kr)
{
    #define SEL(b_, c_, k_) if (bits == b_ && cb == c_ && kr == k_) return dgemv_kernel<b_, c_, k_>;
    #define SEL_K(b_, c_) SEL(b_, c_, 128) SEL(b_, c_, 256)
    SEL_K(2, 0) SEL_K(2, 1) SEL_K(2, 2) SEL_K(3, 0) SEL_K(3, 1) SEL_K(3, 2) SEL_K(4, 0) SEL_K(4, 1) SEL_K(4, 2)
    // past 4 bpw (heads, mostly): 128 rows only
    SEL(5, 0, 128) SEL(5, 1, 128) SEL(5, 2, 128) SEL(6, 0, 128) SEL(6, 1, 128) SEL(6, 2, 128)
    SEL(8, 0, 128) SEL(8, 1, 128) SEL(8, 2, 128)
    #undef SEL_K
    #undef SEL
    return nullptr;
}

static moe_fn select_moe(bool a, int bits, int cb, int kr)
{
    #define SEL(b_, c_, k_) if (bits == b_ && cb == c_ && kr == k_) return a ? moe_a_kernel<b_, c_, k_> : moe_b_kernel<b_, c_, k_>;
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

// Prefill MoE in groups of 16 rows (an expert's tokens), all 16 rows of each MMA real: a block per (group, 128-column
// group) over every k, so no workspace. Gate and up (with silu and the rotation into down's input) in one kernel, down
// in another. 2 to 4 bpw; other rates take exllamav3's kernels
constexpr int GROUP_M = 16;
// k a step: one Hadamard span
constexpr int KC = 128;
// shared rows of A padded past KC: 68 words apart, so the fragment loads of a warp's eight row pairs miss each other's
// banks
constexpr int SA_STRIDE = KC + 8;

// m16n8k16, f16 inputs, f32 accumulate, all 16 rows of A: a0/a1 rows r and r + 8 at k pair 2t, a2/a3 at 2t + 8
__device__ __forceinline__ void mma16(half2 a0, half2 a1, half2 a2, half2 a3, const FragB& b, float* c)
{
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t r0 = *reinterpret_cast<uint32_t*>(&a0), r1 = *reinterpret_cast<uint32_t*>(&a1);
    uint32_t r2 = *reinterpret_cast<uint32_t*>(&a2), r3 = *reinterpret_cast<uint32_t*>(&a3);
    asm volatile
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(bb[0]), "r"(bb[1])
    );
}

// 16 rows of A (sa: GROUP_M x SA_STRIDE f16, one k step) times the warp's loaded tiles, into c0 (its columns 0..7)
// and c1 (8..15)
template <int bits, int cb>
__device__ __forceinline__ void mma_tiles16(const uint32_t (&w)[KC / 16 * tile_loads(bits)], const half* sa, int lane,
                                            float* c0, float* c1)
{
    static_assert(bits >= 2 && bits <= 4, "the grouped kernels decode 2 to 4 bpw in registers");
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
    const int r = lane / 4, t = lane % 4;
    const half2* lo = reinterpret_cast<const half2*>(sa + r * SA_STRIDE);
    const half2* hi = reinterpret_cast<const half2*>(sa + (r + 8) * SA_STRIDE);
    #pragma unroll
    for (int s = 0; s < KC / 16; ++s)
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
        half2 a0 = lo[s * 8 + t], a1 = hi[s * 8 + t], a2 = lo[s * 8 + t + 4], a3 = hi[s * 8 + t + 4];
        mma16(a0, a1, a2, a3, f0, c0);
        mma16(a0, a1, a2, a3, f1, c1);
    }
}

// a warp's accumulators into its 16 columns of so (GROUP_M x SO_STRIDE f32)
constexpr int SO_STRIDE = GROUP_COLS + 4;
__device__ __forceinline__ void store_acc(float* so, int warp, int lane, const float* c0, const float* c1)
{
    const int r = lane / 4, c = warp * 16 + 2 * (lane % 4);
    so[r * SO_STRIDE + c] = c0[0];
    so[r * SO_STRIDE + c + 1] = c0[1];
    so[(r + 8) * SO_STRIDE + c] = c0[2];
    so[(r + 8) * SO_STRIDE + c + 1] = c0[3];
    so[r * SO_STRIDE + c + 8] = c1[0];
    so[r * SO_STRIDE + c + 9] = c1[1];
    so[(r + 8) * SO_STRIDE + c + 8] = c1[2];
    so[(r + 8) * SO_STRIDE + c + 9] = c1[3];
}

struct GroupedParams
{
    const void* x;              // tokens x H, f16 or bf16 (x_bf16)
    bool x_bf16;
    const uint32_t* row_token;  // groups x GROUP_M, each row's token (tokens for padding: a zero row)
    const int64_t* group_expert;// groups
    int tokens;
    const int64_t* gate;        // each projection's table: experts trellis, then suh, then svh addresses
    const int64_t* up;
    const int64_t* down;
    int experts;
    int H;
    int I;
    half* act;                  // groups x GROUP_M x I, down's rotated input
    float* out;                 // groups x GROUP_M x H
};

// gate and up for one group's 16 rows and 128 columns of I: each k step's rows gathered and rotated by both
// projections' suh, then at the end both rotated out, silu(gate) * up, and that rotated by down's suh into act
template <int bits, int cb>
__global__ __launch_bounds__(THREADS)
void grouped_gu_kernel(const __grid_constant__ GroupedParams p)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int cgroup = blockIdx.x, grp = blockIdx.y;
    const int e = (int) p.group_expert[grp];
    const uint32_t* bg = (const uint32_t*) p.gate[e];
    const uint32_t* bu = (const uint32_t*) p.up[e];
    const half* suh_g = (const half*) p.gate[p.experts + e];
    const half* suh_u = (const half*) p.up[p.experts + e];
    const half* svh_g = (const half*) p.gate[2 * p.experts + e] + cgroup * GROUP_COLS;
    const half* svh_u = (const half*) p.up[2 * p.experts + e] + cgroup * GROUP_COLS;
    const half* suh_d = (const half*) p.down[p.experts + e] + cgroup * GROUP_COLS;
    const int ntiles = p.I / 16, tile = cgroup * WARPS + warp;

    __shared__ half sa[2][GROUP_M * SA_STRIDE];
    __shared__ float so[2][GROUP_M * SO_STRIDE];
    float g0[4] = {}, g1[4] = {}, u0[4] = {}, u1[4] = {};
    for (int kz = 0; kz < p.H / KC; ++kz)
    {
        uint32_t wg[KC / 16 * tile_loads(bits)], wu[KC / 16 * tile_loads(bits)];
        load_tiles<bits, KC>(wg, bg, ntiles, tile, kz, lane);
        load_tiles<bits, KC>(wu, bu, ntiles, tile, kz, lane);
        for (int r = warp; r < GROUP_M; r += WARPS)
        {
            const uint32_t tok = p.row_token[grp * GROUP_M + r];
            const int k0 = kz * KC + lane * 4;
            float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
            if (tok < (uint32_t) p.tokens) load4(p.x, p.x_bf16, (size_t) tok * p.H + k0, x0, x1, x2, x3);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
            {
                uint2 su = *((const uint2*) ((j ? suh_u : suh_g) + k0));
                float2 s01 = __half22float2(*reinterpret_cast<half2*>(&su.x));
                float2 s23 = __half22float2(*reinterpret_cast<half2*>(&su.y));
                float v0 = x0 * s01.x, v1 = x1 * s01.y, v2 = x2 * s23.x, v3 = x3 * s23.y;
                had128(v0, v1, v2, v3, lane);
                half2* row = reinterpret_cast<half2*>(sa[j] + r * SA_STRIDE);
                row[lane * 2] = __floats2half2_rn(v0, v1);
                row[lane * 2 + 1] = __floats2half2_rn(v2, v3);
            }
        }
        __syncthreads();
        mma_tiles16<bits, cb>(wg, sa[0], lane, g0, g1);
        mma_tiles16<bits, cb>(wu, sa[1], lane, u0, u1);
        __syncthreads();
    }
    store_acc(so[0], warp, lane, g0, g1);
    store_acc(so[1], warp, lane, u0, u1);
    __syncthreads();
    for (int r = warp; r < GROUP_M; r += WARPS)
    {
        float og[4], ou[4];
        rotate_out(so[0] + r * SO_STRIDE, svh_g, lane, og);
        rotate_out(so[1] + r * SO_STRIDE, svh_u, lane, ou);
        uint2 su = *((const uint2*) (suh_d + lane * 4));
        float2 s01 = __half22float2(*reinterpret_cast<half2*>(&su.x));
        float2 s23 = __half22float2(*reinterpret_cast<half2*>(&su.y));
        float a[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = og[i] / (1.0f + __expf(-og[i])) * ou[i];
        a[0] *= s01.x; a[1] *= s01.y; a[2] *= s23.x; a[3] *= s23.y;
        had128(a[0], a[1], a[2], a[3], lane);
        half2 h01 = __floats2half2_rn(a[0], a[1]), h23 = __floats2half2_rn(a[2], a[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&h01);
        u.y = *reinterpret_cast<uint32_t*>(&h23);
        *((uint2*) (p.act + ((size_t) grp * GROUP_M + r) * p.I + cgroup * GROUP_COLS + lane * 4)) = u;
    }
}

// down for one group's 16 rows (act, already rotated) and 128 columns of H, rotated out by svh into out
template <int bits, int cb>
__global__ __launch_bounds__(THREADS)
void grouped_down_kernel(const __grid_constant__ GroupedParams p)
{
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int cgroup = blockIdx.x, grp = blockIdx.y;
    const int e = (int) p.group_expert[grp];
    const uint32_t* bd = (const uint32_t*) p.down[e];
    const half* svh_d = (const half*) p.down[2 * p.experts + e] + cgroup * GROUP_COLS;
    const int ntiles = p.H / 16, tile = cgroup * WARPS + warp;

    __shared__ half sa[GROUP_M * SA_STRIDE];
    __shared__ float so[GROUP_M * SO_STRIDE];
    float c0[4] = {}, c1[4] = {};
    for (int kz = 0; kz < p.I / KC; ++kz)
    {
        uint32_t w[KC / 16 * tile_loads(bits)];
        load_tiles<bits, KC>(w, bd, ntiles, tile, kz, lane);
        for (int r = warp; r < GROUP_M; r += WARPS)
            *((uint2*) (sa + r * SA_STRIDE) + lane) =
                *((const uint2*) (p.act + ((size_t) grp * GROUP_M + r) * p.I + kz * KC) + lane);
        __syncthreads();
        mma_tiles16<bits, cb>(w, sa, lane, c0, c1);
        __syncthreads();
    }
    store_acc(so, warp, lane, c0, c1);
    __syncthreads();
    for (int r = warp; r < GROUP_M; r += WARPS)
    {
        float o[4];
        rotate_out(so + r * SO_STRIDE, svh_d, lane, o);
        *((float4*) (p.out + ((size_t) grp * GROUP_M + r) * p.H + cgroup * GROUP_COLS + lane * 4)) =
            make_float4(o[0], o[1], o[2], o[3]);
    }
}

typedef void (*grouped_fn)(const GroupedParams);

template <int bits>
static grouped_fn select_grouped_cb(bool gu, int cb)
{
    if (cb == 0) return gu ? grouped_gu_kernel<bits, 0> : grouped_down_kernel<bits, 0>;
    if (cb == 1) return gu ? grouped_gu_kernel<bits, 1> : grouped_down_kernel<bits, 1>;
    if (cb == 2) return gu ? grouped_gu_kernel<bits, 2> : grouped_down_kernel<bits, 2>;
    return nullptr;
}

static grouped_fn select_grouped(bool gu, int bits, int cb)
{
    if (bits == 2) return select_grouped_cb<2>(gu, cb);
    if (bits == 3) return select_grouped_cb<3>(gu, cb);
    if (bits == 4) return select_grouped_cb<4>(gu, cb);
    return nullptr;
}

}  // namespace exl3_dgemv

// `rows` (1 to 8) rows x (f16, or bf16 with x_bf16) through `jobs` trellises of one k (B, suh, svh, out, n apiece;
// tiles of tile_u16 words): each out (rows x n, f32, f16 or bf16 by out_kind) = had(had(x * suh) @ W) * svh. ws is the
// stream's workspace of ws_floats words, zeroed when made: counters first (left zeroed), then the partial rows.
// Returns 0 when launched, -1 for a shape or rate it does not take or a workspace too small.
extern "C" int launch_exl3_dgemv(const void* x, bool x_bf16, int rows, int jobs, const uint16_t* const* B,
                                 const half* const* suh, const half* const* svh, void* const* out, const int* n,
                                 int size_k, int tile_u16, int cb, float* ws, int64_t ws_floats, int out_kind,
                                 cudaStream_t stream)
{
    using namespace exl3_dgemv;
    if (jobs < 1 || jobs > MAX_JOBS || tile_u16 % 16 || size_k % 128 || rows < 1 || rows > MAX_ROWS) return -1;
    Params p = {};
    p.x = x;
    p.x_bf16 = x_bf16;
    p.size_k = size_k;
    p.rows = rows;
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
    if (tile_u16 / 16 > 4) kr = 128;
    if (size_k % kr) return -1;
    dgemv_fn kernel = select(tile_u16 / 16, cb, kr);
    if (!kernel) return -1;
    p.splits = size_k / kr;
    // past one row, the rows rotated once at the workspace's end (a half2 a word)
    const int64_t rotated = rows > 1 ? (int64_t) jobs * rows * size_k / 2 : 0;
    if (p.groups > DGEMV_COUNTERS || DGEMV_COUNTERS + (int64_t) p.splits * rows * p.total_n + rotated > ws_floats)
        return -1;
    p.counters = (int*) ws;
    p.partial = ws + DGEMV_COUNTERS;
    if (rotated)
    {
        half2* xr = (half2*) (ws + ws_floats - rotated);
        const int warps = jobs * rows * (size_k / 128);
        rotate_rows_kernel<<<(warps + WARPS - 1) / WARPS, THREADS, 0, stream>>>(p, xr);
        p.xr = xr;
    }
    kernel<<<p.groups * p.splits, THREADS, 0, stream>>>(p);
    cuda_check(cudaPeekAtLastError());
    return 0;
}

// rows a block for `groups` column groups over k: 128, or 256 past KR_256_BLOCKS_PER_SM blocks a multiprocessor
static int pick_kr(int groups, int size_k, int sms)
{
    return groups * (size_k / 128) > exl3_dgemv::KR_256_BLOCKS_PER_SM * sms && size_k % 256 == 0 ? 256 : 128;
}

// `tokens` (1 to 8) tokens' routed experts (sel, rw: tokens x topk) through gate, up (silu) and down into out (tokens
// x H, of out_kind). Each table holds `experts` trellis, then suh, then svh addresses. ws as launch_exl3_dgemv's.
// Returns 0 when launched, -1 for shapes, rates or a workspace it does not take.
extern "C" int launch_exl3_dmoe(const void* x, bool x_bf16, int tokens, const void* sel, bool sel_u32, const void* rw,
                                bool rw_f32, const int64_t* gate, const int64_t* up, const int64_t* down, int experts,
                                int topk, int hidden, int inter, int gu_tile_u16, int d_tile_u16, int cb, float* ws,
                                int64_t ws_floats, void* out, int out_kind, cudaStream_t stream)
{
    using namespace exl3_dgemv;
    if (gu_tile_u16 % 16 || d_tile_u16 % 16 || hidden % 128 || inter % 128 || topk < 1) return -1;
    if (tokens < 1 || tokens > MAX_ROWS) return -1;
    int device;
    cudaGetDevice(&device);
    const int sms = DevCtx::instance().get_num_sms(device);
    const int ig = inter / GROUP_COLS, hg = hidden / GROUP_COLS;
    const int slots = tokens * topk;
    const int kr_a = pick_kr(slots * 2 * ig, hidden, sms);
    const int kr_b = pick_kr(slots * hg, inter, sms);
    if (hidden % kr_a || inter % kr_b) return -1;
    moe_fn ka = select_moe(true, gu_tile_u16 / 16, cb, kr_a);
    moe_fn kb = select_moe(false, d_tile_u16 / 16, cb, kr_b);
    if (!ka || !kb) return -1;
    MoeParams p = {};
    p.x = x; p.x_bf16 = x_bf16;
    p.sel = sel; p.sel_u32 = sel_u32;
    p.rw = rw; p.rw_f32 = rw_f32;
    p.out_kind = out_kind;
    p.gate = gate; p.up = up; p.down = down;
    p.experts = experts; p.topk = topk; p.tokens = tokens; p.H = hidden; p.I = inter;
    p.splits_a = hidden / kr_a;
    p.splits_b = inter / kr_b;
    // counters: three regions of the workspace's leading words
    constexpr int CTR_REGION = DGEMV_COUNTERS / 4;
    if (slots * ig > CTR_REGION || slots * hg > CTR_REGION || tokens * hg > CTR_REGION) return -1;
    p.ctr_a = (int*) ws;
    p.ctr_b = p.ctr_a + CTR_REGION;
    p.ctr_t = p.ctr_b + CTR_REGION;
    float* f = ws + DGEMV_COUNTERS;
    p.part_a = f;
    f += (size_t) p.splits_a * slots * 2 * inter;
    p.act = (half*) f;
    f += (size_t) slots * inter / 2;
    p.part_b = f;
    f += (size_t) p.splits_b * slots * hidden;
    p.slot_out = f;
    f += (size_t) slots * hidden;
    if (f - ws > ws_floats) return -1;
    p.out = out;
    ka<<<slots * 2 * ig * p.splits_a, THREADS, 0, stream>>>(p);
    cuda_check(cudaPeekAtLastError());
    kb<<<slots * hg * p.splits_b, THREADS, 0, stream>>>(p);
    cuda_check(cudaPeekAtLastError());
    return 0;
}

// A prompt's routed experts in groups of 16 rows (row_token, group_expert: `groups` of them, host-built), gate, up and
// silu into act (groups x 16 x inter f16, down's rotated input), down into out (groups x 16 x hidden f32). Each table
// holds `experts` trellis, then suh, then svh addresses. Returns 0 when launched, -1 for shapes or rates it does not
// take.
extern "C" int launch_exl3_grouped_moe(const void* x, bool x_bf16, const uint32_t* row_token,
                                       const int64_t* group_expert, int groups, int tokens, const int64_t* gate,
                                       const int64_t* up, const int64_t* down, int experts, int hidden, int inter,
                                       int gu_tile_u16, int d_tile_u16, int cb, half* act, float* out,
                                       cudaStream_t stream)
{
    using namespace exl3_dgemv;
    if (gu_tile_u16 % 16 || d_tile_u16 % 16 || hidden % 128 || inter % 128 || groups < 1) return -1;
    grouped_fn gu = select_grouped(true, gu_tile_u16 / 16, cb);
    grouped_fn dn = select_grouped(false, d_tile_u16 / 16, cb);
    if (!gu || !dn) return -1;
    GroupedParams p = {};
    p.x = x; p.x_bf16 = x_bf16;
    p.row_token = row_token; p.group_expert = group_expert; p.tokens = tokens;
    p.gate = gate; p.up = up; p.down = down; p.experts = experts;
    p.H = hidden; p.I = inter;
    p.act = act; p.out = out;
    gu<<<dim3(inter / GROUP_COLS, groups), THREADS, 0, stream>>>(p);
    cuda_check(cudaPeekAtLastError());
    dn<<<dim3(hidden / GROUP_COLS, groups), THREADS, 0, stream>>>(p);
    cuda_check(cudaPeekAtLastError());
    return 0;
}
