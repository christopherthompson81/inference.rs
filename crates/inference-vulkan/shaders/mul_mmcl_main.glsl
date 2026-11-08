// mul_mmc_main.glsl's prompt matmul in 128 x 128 tiles (weight rows x tokens), for products that fill the device with
// them: each of the 4 subgroups a 64 x 64 quarter as 4 x 4 accumulators, both operands staged in shared memory a step
// (the weights dequantized to f16, the tokens' f16 rows copied), so a staged byte serves four times the products the
// 64 x 64 tile's did (llama.cpp's large matrix-core tile). Each 16 x 16 result leaves through a subgroup's stage.
#include "block_at.glsl"

const uint THREADS = 128;
const uint QK = 32;
const uint BN = 128;
const uint BM = 128;
// 32-value blocks a step, and the step's values (the tiles' row length, padded against bank conflicts)
const uint BLOCKS = 2;
const uint BK = BLOCKS * QK;
const uint STRIDE = BK + 8;
const uint TILE = 16;
// a subgroup's quarter, in TILE x TILE accumulators a side
const uint SUB = 4;

// the tokens' f16 rows 8 halves a load (a row's values and its tile offsets are whole eights)
layout(std430, binding = 1) readonly buffer X8 { uvec4 x8[]; };

shared f16vec4 w_tile[BN * STRIDE / 4];
shared f16vec4 x_tile[BM * STRIDE / 4];
// a subgroup's 16 x 16 result on its way out, rows of tokens
shared float stage[4 * TILE * TILE];

// a block's 32 int8 values (two uvec4s of packed bytes) as w = s q + o, into a tile row from `at` (in halves)
void dequantize(uvec4 lo, uvec4 hi, vec4 c, uint at) {
    [[unroll]] for (uint i = 0; i < 8; i++) {
        uint word = i < 4 ? lo[i] : hi[i - 4];
        float s = i < 4 ? c.x : c.y;
        float o = i < 4 ? c.z : c.w;
        vec4 q = vec4(ivec4(int(word << 24) >> 24, int(word << 16) >> 24, int(word << 8) >> 24, int(word) >> 24));
        w_tile[at / 4 + i] = f16vec4(s * q + o);
    }
}

void main() {
#ifdef TYPE_SHARED_INIT
    // the block type's lookup tables into shared memory, before any thread can leave
    type_shared_init();
#endif
    uint t = gl_LocalInvocationID.x;
    uint n0 = gl_WorkGroupID.x * BN;
    uint tokens, w_matrix, x_row0;
    // uniform across the workgroup, so all leave before any barrier
    if (!begin_tile(tokens, w_matrix, x_row0)) {
        return;
    }
    uint blocks = pc.cols / QK;
    uint sg = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint sn = (sg % 2) * (SUB * TILE);
    uint sm = (sg / 2) * (SUB * TILE);
    coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator> acc[SUB][SUB];
    [[unroll]] for (uint i = 0; i < SUB; i++) {
        [[unroll]] for (uint j = 0; j < SUB; j++) {
            acc[i][j] = coopmat<float, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseAccumulator>(0.0);
        }
    }
    for (uint k0 = 0; k0 < blocks; k0 += BLOCKS) {
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
                [[unroll]] for (uint j = 0; j < QK / 4; j++) {
                    w_tile[at / 4 + j] = f16vec4(0.0);
                }
            }
        }
        // the tokens' values; past a row's last block the next row's (or the scratch's zero padding), against zero
        // weights
        for (uint i = t; i < BM * (BK / 8); i += THREADS) {
            uint r = i / (BK / 8);
            uint c8 = i % (BK / 8);
            uvec4 v = x8[(pc.x_base + (x_row0 + r) * pc.cols + k0 * QK) / 8 + c8];
            uint at = r * (STRIDE / 4) + 2 * c8;
            x_tile[at] = f16vec4(unpackFloat2x16(v.x), unpackFloat2x16(v.y));
            x_tile[at + 1] = f16vec4(unpackFloat2x16(v.z), unpackFloat2x16(v.w));
        }
        barrier();
        [[unroll]] for (uint kk = 0; kk < BK; kk += TILE) {
            coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseA> a[SUB];
            [[unroll]] for (uint i = 0; i < SUB; i++) {
                coopMatLoad(a[i], w_tile, ((sn + TILE * i) * STRIDE + kk) / 4, STRIDE / 4, gl_CooperativeMatrixLayoutRowMajor);
            }
            [[unroll]] for (uint j = 0; j < SUB; j++) {
                // B is (values x tokens), a token's values a row of the tile
                coopmat<float16_t, gl_ScopeSubgroup, TILE, TILE, gl_MatrixUseB> bm;
                coopMatLoad(bm, x_tile, ((sm + TILE * j) * STRIDE + kk) / 4, STRIDE / 4, gl_CooperativeMatrixLayoutColumnMajor);
                [[unroll]] for (uint i = 0; i < SUB; i++) {
                    acc[i][j] = coopMatMulAdd(a[i], bm, acc[i][j]);
                }
            }
        }
        barrier();
    }
    // each result as (rows x tokens), a token's 16 rows a stretch of its output row
    [[unroll]] for (uint i = 0; i < SUB; i++) {
        [[unroll]] for (uint j = 0; j < SUB; j++) {
            coopMatStore(acc[i][j], stage, sg * TILE * TILE, TILE, gl_CooperativeMatrixLayoutColumnMajor);
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            for (uint e = lane; e < TILE * TILE; e += gl_SubgroupSize) {
                uint n = e % TILE, mm = e / TILE;
                uint row = n0 + sn + TILE * i + n;
                uint token = sm + TILE * j + mm;
                if (row < pc.rows && token < tokens) {
                    y[pc.y_base + y_row(token) * pc.rows + row] = stage[sg * TILE * TILE + mm * TILE + n];
                }
            }
            subgroupBarrier();
        }
    }
}
