// c[b] = a[b] (m x k) times b[b] (k x n), every operand strided, c contiguous: a workgroup a BM x BN tile of c (grid x
// over n, y over m, z over the batch), stepping k by BK through shared tiles; each thread a TM x TN block of it.

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float bm[]; };
layout(std430, binding = 2) buffer C { float c[]; };

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
// the tile and a thread's block of it; an includer may set a smaller one first
#ifndef TILE
#define TILE 64
#endif
const uint BM = TILE;
const uint BN = TILE;
const uint BK = 16;
const uint TM = TILE / 16;
const uint TN = TILE / 16;
// threads across a tile's columns; a thread's outputs sit THREADS_N columns and THREADS / THREADS_N rows apart
const uint THREADS_N = BN / TN;
const uint THREADS_M = THREADS / THREADS_N;
const uint LOADS = BM * BK / THREADS;

shared float as_[BK][BM + 1];
shared float bs_[BK][BN + 1];

void main() {
    uint t = gl_LocalInvocationID.x;
    uint col0 = gl_WorkGroupID.x * BN;
    uint row0 = gl_WorkGroupID.y * BM;
    uint batch = gl_WorkGroupID.z;
    uint a_base = pc.a_off + batch * pc.a_bs;
    uint b_base = pc.b_off + batch * pc.b_bs;
    // load along whichever of b's dimensions is unit-stride, so a tile's reads coalesce
    bool b_along_n = pc.b_ns == 1;
    uint tx = t % THREADS_N;
    uint ty = t / THREADS_N;
    float acc[TM][TN];
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) {
            acc[i][j] = 0.0;
        }
    }
    for (uint k0 = 0; k0 < pc.k; k0 += BK) {
        for (uint l = 0; l < LOADS; l++) {
            uint i = t + l * THREADS;
            uint kk = i % BK;
            uint r = i / BK;
            float v = 0.0;
            if (row0 + r < pc.m && k0 + kk < pc.k) {
                v = a[a_base + (row0 + r) * pc.a_ms + (k0 + kk) * pc.a_ks];
            }
            as_[kk][r] = v;
            uint bk = b_along_n ? i / BN : i % BK;
            uint bc = b_along_n ? i % BN : i / BK;
            float w = 0.0;
            if (col0 + bc < pc.n && k0 + bk < pc.k) {
                w = bm[b_base + (k0 + bk) * pc.b_ks + (col0 + bc) * pc.b_ns];
            }
            bs_[bk][bc] = w;
        }
        barrier();
        for (uint kk = 0; kk < BK; kk++) {
            float av[TM];
            float bv[TN];
            for (uint i = 0; i < TM; i++) {
                av[i] = as_[kk][ty + i * THREADS_M];
            }
            for (uint j = 0; j < TN; j++) {
                bv[j] = bs_[kk][tx + j * THREADS_N];
            }
            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j++) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }
        barrier();
    }
    for (uint i = 0; i < TM; i++) {
        uint row = row0 + ty + i * THREADS_M;
        if (row >= pc.m) {
            continue;
        }
        for (uint j = 0; j < TN; j++) {
            uint col = col0 + tx + j * THREADS_N;
            if (col < pc.n) {
                c[(batch * pc.m + row) * pc.n + col] = acc[i][j];
            }
        }
    }
}
