// matmul_f32_tiled's product on matrix cores (VK_KHR_cooperative_matrix): c[b] = a[b] (m x k) times b[b] (k x n),
// every operand strided, c contiguous. A workgroup takes a BM x BN tile of c, stepping k by BK: its threads load the
// step's a and b tiles four values at a time (vec4s where an operand's unit stride and alignment allow) into shared
// memory as f16, the next step's loads in flight while each subgroup multiplies its 32 x 64 part as 16 x 16 x 16 f16
// products into f32 accumulators. a's tile is k-contiguous; b's follows its unit stride, k (a transposed weight) or n.
// Inputs past f16's range overflow, and each is rounded to f16, whose products f32 holds exactly. With B_HALF, b is
// stored as f16 (a weight kept in f16), read as the f32 build reads it rounded. With BF16 a, b and c are bf16: four
// values ride in a uvec2 (as few registers as an f16vec4) until the tile store rounds them to f16, and c goes out
// through the staging tile.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require

layout(local_size_x = 256) in;

#ifdef BF16
#extension GL_GOOGLE_include_directive : require
#include "bf16.glsl"
#define AVEC uvec2
#define BVEC uvec2
layout(std430, binding = 0) readonly buffer A { uint16_t a[]; };
layout(std430, binding = 0) readonly buffer A4 { uvec2 a4[]; };
layout(std430, binding = 1) readonly buffer B { uint16_t bm[]; };
layout(std430, binding = 1) readonly buffer B4 { uvec2 b4[]; };
layout(std430, binding = 2) buffer C { uint16_t c[]; };
// a scalar read, as the bits it packs into a uvec2
#define A_AT(i) uint(a[i])
#define B_AT(i) uint(bm[i])

f16vec4 tile4(uvec2 v) {
    return f16vec4(bf16_bits_to_f32(v.x & 0xffffu), bf16_bits_to_f32(v.x >> 16),
                   bf16_bits_to_f32(v.y & 0xffffu), bf16_bits_to_f32(v.y >> 16));
}

void pack(inout uvec2 v, uint i, uint bits) {
    v[i / 2] |= bits << (16 * (i % 2));
}
#else
#define AVEC vec4
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 0) readonly buffer A4 { vec4 a4[]; };
#define A_AT(i) a[i]
#define B_AT(i) bm[i]
#define tile4(v) f16vec4(v)
#define pack(v, i, x) v[i] = x
#ifdef B_HALF
// b's values stay f16 from the load to the shared tile (widened to f32 between, the kernel ran ~40% slower)
#define BVEC f16vec4
layout(std430, binding = 1) readonly buffer B { float16_t bm[]; };
layout(std430, binding = 1) readonly buffer B4 { f16vec4 b4[]; };
#else
#define BVEC vec4
layout(std430, binding = 1) readonly buffer B { float bm[]; };
layout(std430, binding = 1) readonly buffer B4 { vec4 b4[]; };
#endif
layout(std430, binding = 2) buffer C { float c[]; };
#endif

layout(push_constant) uniform PushConstants {
    uint batch;
    uint m;
    uint n;
    uint k;
    uint a_off;
    uint a_bs;
    uint a_ms;
    uint a_ks;
    uint b_off;
    uint b_bs;
    uint b_ks;
    uint b_ns;
} pc;

const uint THREADS = 256;
// row tiles that sweep the columns together, so the workgroups in flight share their a and b panels in the cache (a
// 780M at 7500 x 4096 x 1024: 1.2 TFLOPS a row of tiles at a time, 1.8 four or eight)
const uint GROUP_M = 8;
const uint BM = 128;
const uint BN = 128;
const uint BK = 64;
const uint TILE = 16;
// a subgroup's part of the tile: SG_M x SG_N, as SG_M / TILE x SG_N / TILE products
const uint SG_M = 32;
const uint SG_N = 64;
const uint TM = SG_M / TILE;
const uint TN = SG_N / TILE;
// the tile's vec4s a thread loads a step, of each operand
const uint LOADS = BM * BK / 4 / THREADS;
// shared rows, padded against bank conflicts: a k-contiguous row and an n-contiguous one, in halves
const uint K_STRIDE = BK + 8;
const uint N_STRIDE = BN + 8;
const uint A_HALVES = BM * K_STRIDE;
const uint B_HALVES = BN * K_STRIDE > BK * N_STRIDE ? BN * K_STRIDE : BK * N_STRIDE;

shared f16vec4 a_tile[A_HALVES / 4];
shared f16vec4 b_tile[B_HALVES / 4];
// a subgroup's 16 x 16 block on its way to a partial tile's edge
shared float stage[THREADS / 32][TILE * TILE];

uint m0, n0, a_base, b_base;
bool a_vec, b_along_k, b_vec;

// a's 4 values from (row, kq), zero past its edges
AVEC load_a(uint row, uint kq) {
    uint r = m0 + row;
    if (a_vec && r < pc.m && kq < pc.k) {
        return a4[(a_base + r * pc.a_ms + kq) / 4];
    }
    AVEC v = AVEC(0);
    [[unroll]] for (uint i = 0; i < 4; i++) {
        if (r < pc.m && kq + i < pc.k) {
            pack(v, i, A_AT(a_base + r * pc.a_ms + (kq + i) * pc.a_ks));
        }
    }
    return v;
}

// b's 4 values along k from (kq, col), or along n from (kk, nq)
BVEC load_b(uint kq, uint col, bool along_k) {
    uint kk_step = along_k ? 1 : 0;
    uint nn_step = along_k ? 0 : 1;
    if (b_vec && kq < pc.k && col < pc.n) {
        return b4[(b_base + kq * pc.b_ks + col * pc.b_ns) / 4];
    }
    BVEC v = BVEC(0);
    [[unroll]] for (uint i = 0; i < 4; i++) {
        uint kk = kq + i * kk_step;
        uint nn = col + i * nn_step;
        if (kk < pc.k && nn < pc.n) {
            pack(v, i, B_AT(b_base + kk * pc.b_ks + nn * pc.b_ns));
        }
    }
    return v;
}

// thread `u`'s vec4 of a step: a (row, k quad) of a, and of b a (col, k quad) or a (k, n quad)
void fetch(uint k0, uint u, out AVEC va, out BVEC vb) {
    va = load_a(u / (BK / 4), k0 + (u % (BK / 4)) * 4);
    if (b_along_k) {
        vb = load_b(k0 + (u % (BK / 4)) * 4, n0 + u / (BK / 4), true);
    } else {
        vb = load_b(k0 + u / (BN / 4), n0 + (u % (BN / 4)) * 4, false);
    }
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint tiles_n = gl_NumWorkGroups.x;
    uint tiles_m = gl_NumWorkGroups.y;
    uint id = gl_WorkGroupID.y * tiles_n + gl_WorkGroupID.x;
    uint group = id / (GROUP_M * tiles_n);
    uint first_m = group * GROUP_M;
    uint group_m = min(tiles_m - first_m, GROUP_M);
    uint in_group = id % (GROUP_M * tiles_n);
    m0 = (first_m + in_group % group_m) * BM;
    n0 = (in_group / group_m) * BN;
    uint z = gl_WorkGroupID.z;
    a_base = pc.a_off + z * pc.a_bs;
    b_base = pc.b_off + z * pc.b_bs;
    a_vec = pc.a_ks == 1 && (pc.a_off | pc.a_bs | pc.a_ms | pc.k) % 4 == 0;
    b_along_k = pc.b_ks == 1;
    bool b_along_n = !b_along_k && pc.b_ns == 1;
    b_vec = (b_along_k && (pc.b_off | pc.b_bs | pc.b_ns | pc.k) % 4 == 0)
        || (b_along_n && (pc.b_off | pc.b_bs | pc.b_ks | pc.n) % 4 == 0);
    uint sg = gl_SubgroupID;
    uint sm = (sg / (BN / SG_N)) * SG_M;
    uint sn = (sg % (BN / SG_N)) * SG_N;
    coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator> acc[TM][TN];
    [[unroll]] for (uint i = 0; i < TM; i++) {
        [[unroll]] for (uint j = 0; j < TN; j++) {
            acc[i][j] = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0);
        }
    }
    AVEC ra[LOADS];
    BVEC rb[LOADS];
    [[unroll]] for (uint l = 0; l < LOADS; l++) {
        fetch(0, t + l * THREADS, ra[l], rb[l]);
    }
    for (uint k0 = 0; k0 < pc.k; k0 += BK) {
        [[unroll]] for (uint l = 0; l < LOADS; l++) {
            uint u = t + l * THREADS;
            a_tile[(u / (BK / 4)) * (K_STRIDE / 4) + u % (BK / 4)] = tile4(ra[l]);
            if (b_along_k) {
                b_tile[(u / (BK / 4)) * (K_STRIDE / 4) + u % (BK / 4)] = tile4(rb[l]);
            } else {
                b_tile[(u / (BN / 4)) * (N_STRIDE / 4) + u % (BN / 4)] = tile4(rb[l]);
            }
        }
        barrier();
        if (k0 + BK < pc.k) {
            [[unroll]] for (uint l = 0; l < LOADS; l++) {
                fetch(k0 + BK, t + l * THREADS, ra[l], rb[l]);
            }
        }
        [[unroll]] for (uint kk = 0; kk < BK; kk += TILE) {
            coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseA> am[TM];
            coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseB> bt[TN];
            [[unroll]] for (uint i = 0; i < TM; i++) {
                coopMatLoad(am[i], a_tile, ((sm + TILE * i) * K_STRIDE + kk) / 4, K_STRIDE / 4, gl_CooperativeMatrixLayoutRowMajor);
            }
            [[unroll]] for (uint j = 0; j < TN; j++) {
                if (b_along_k) {
                    coopMatLoad(bt[j], b_tile, ((sn + TILE * j) * K_STRIDE + kk) / 4, K_STRIDE / 4, gl_CooperativeMatrixLayoutColumnMajor);
                } else {
                    coopMatLoad(bt[j], b_tile, (kk * N_STRIDE + sn + TILE * j) / 4, N_STRIDE / 4, gl_CooperativeMatrixLayoutRowMajor);
                }
            }
            [[unroll]] for (uint i = 0; i < TM; i++) {
                [[unroll]] for (uint j = 0; j < TN; j++) {
                    acc[i][j] = coopMatMulAdd(am[i], bt[j], acc[i][j]);
                }
            }
        }
        barrier();
    }
    uint c_base = z * pc.m * pc.n;
    // a tile stores straight to c only on 16-byte rows: NVIDIA's driver stores misaligned ones wrong (odd widths)
#ifdef BF16
    bool whole = false;
#else
    bool whole = m0 + BM <= pc.m && n0 + BN <= pc.n && pc.n % 4 == 0;
#endif
    uint lane = gl_SubgroupInvocationID;
    [[unroll]] for (uint i = 0; i < TM; i++) {
        [[unroll]] for (uint j = 0; j < TN; j++) {
            uint row = m0 + sm + TILE * i;
            uint col = n0 + sn + TILE * j;
            if (whole) {
#ifndef BF16
                coopMatStore(acc[i][j], c, c_base + row * pc.n + col, pc.n, gl_CooperativeMatrixLayoutRowMajor);
#endif
            } else {
                coopMatStore(acc[i][j], stage[sg], 0, TILE, gl_CooperativeMatrixLayoutRowMajor);
                subgroupMemoryBarrierShared();
                subgroupBarrier();
                for (uint e = lane; e < TILE * TILE; e += gl_SubgroupSize) {
                    uint r = row + e / TILE, cc = col + e % TILE;
                    if (r < pc.m && cc < pc.n) {
#ifdef BF16
                        c[c_base + r * pc.n + cc] = uint16_t(f32_to_bf16_bits(stage[sg][e]));
#else
                        c[c_base + r * pc.n + cc] = stage[sg][e];
#endif
                    }
                }
                subgroupBarrier();
            }
        }
    }
}
