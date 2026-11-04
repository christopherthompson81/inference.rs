#include "block_at.glsl"

const uint THREADS = 128;
const uint QK = 32;
const uint BN = 64;
const uint BM = 64;
// 32-value blocks a step, and the step's values (the tiles' row length, padded a 16-bit pair against bank conflicts)
const uint BLOCKS = 4;
const uint BK = BLOCKS * QK;
const uint STRIDE = BK + 8;
const uint TILE = 16;

shared float16_t w_tile[BN * STRIDE];
shared float c_tile[BN * BM];

// a block's 32 int8 values (two uvec4s of packed bytes) as w = s q + o, into a tile row from `at`
void dequantize(uvec4 lo, uvec4 hi, vec4 c, uint at) {
    [[unroll]] for (uint i = 0; i < 8; i++) {
        uint word = i < 4 ? lo[i] : hi[i - 4];
        float s = i < 4 ? c.x : c.y;
        float o = i < 4 ? c.z : c.w;
        [[unroll]] for (uint k = 0; k < 4; k++) {
            int q = int(word << (24 - 8 * k)) >> 24;
            w_tile[at + 4 * i + k] = float16_t(s * float(q) + o);
        }
    }
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint n0 = gl_WorkGroupID.x * BN;
    uint tokens, w_matrix, x_row0;
    // uniform across the workgroup, so all leave before any barrier
    if (!begin_tile(tokens, w_matrix, x_row0)) {
        return;
    }
    uint blocks = pc.cols / QK;
    // the subgroup's 32 x 32 quarter of the tile
    uint sg = gl_SubgroupID;
    uint sn = (sg % 2) * 32;
    uint sm = (sg / 2) * 32;
    coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator> acc[2][2];
    [[unroll]] for (uint i = 0; i < 2; i++) {
        [[unroll]] for (uint j = 0; j < 2; j++) {
            acc[i][j] = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0);
        }
    }
    for (uint k0 = 0; k0 < blocks; k0 += BLOCKS) {
        // the step's weight blocks, BN * BLOCKS of them over the threads
        for (uint u = t; u < BN * BLOCKS; u += THREADS) {
            uint w_row = u / BLOCKS;
            uint w_block = u % BLOCKS;
            uint b = k0 + w_block;
            uint at = w_row * STRIDE + w_block * QK;
            if (n0 + w_row < pc.rows && b < blocks) {
                uvec4 lo, hi;
                vec4 c = unpack(w_matrix, n0 + w_row, b, blocks, lo, hi);
                dequantize(lo, hi, c, at);
            } else {
                [[unroll]] for (uint j = 0; j < QK; j++) {
                    w_tile[at + j] = float16_t(0.0);
                }
            }
        }
        barrier();
        [[unroll]] for (uint kk = 0; kk < BK; kk += TILE) {
            coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseA> a[2];
            coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseB> bm[2];
            [[unroll]] for (uint i = 0; i < 2; i++) {
                coopMatLoad(a[i], w_tile, (sn + TILE * i) * STRIDE + kk, STRIDE, gl_CooperativeMatrixLayoutRowMajor);
                // B is (values x tokens), a token's values a row of x; rows past the tile's tokens are padding or
                // another tile's, their outputs dropped
                uint row = x_row0 + sm + TILE * i;
                coopMatLoad(bm[i], x, pc.x_base + row * pc.cols + k0 * QK + kk, pc.cols, gl_CooperativeMatrixLayoutColumnMajor);
            }
            [[unroll]] for (uint i = 0; i < 2; i++) {
                [[unroll]] for (uint j = 0; j < 2; j++) {
                    acc[i][j] = coopMatMulAdd(a[i], bm[j], acc[i][j]);
                }
            }
        }
        barrier();
    }
    [[unroll]] for (uint i = 0; i < 2; i++) {
        [[unroll]] for (uint j = 0; j < 2; j++) {
            coopMatStore(acc[i][j], c_tile, (sn + TILE * i) * BM + sm + TILE * j, BM, gl_CooperativeMatrixLayoutRowMajor);
        }
    }
    barrier();
    for (uint i = t; i < BN * BM; i += THREADS) {
        uint n = i % BN, mm = i / BN;
        if (n0 + n < pc.rows && mm < tokens) {
            y[pc.y_base + y_row(mm) * pc.rows + n0 + n] = c_tile[n * BM + mm];
        }
    }
}
