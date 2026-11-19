#include "block_at.glsl"

#if SPLIT_HALVES
#error "mul_mmq_cm takes a scale and offset a block"
#endif

const uint THREADS = 256;
const uint SUBGROUP = 32;
const uint QK = 32;
#ifndef BN
#define BN 128
#endif
#ifndef WN
#define WN 64
#endif
const uint BM = 128;
const uint BK = 2;
// a subgroup's tile: the workgroup's subgroups split BM into BM / WM and BN into the rest
const uint WM = BM / (THREADS / SUBGROUP / (BN / WN));
const uint CM = 16;
const uint CM_ROWS = WM / CM;
const uint CM_COLS = WN / CM;
const uint ELEMS = CM * CM / SUBGROUP;
// a row's packed words in shared memory, BK blocks of 8 and a pad that spreads banks
const uint PITCH = BK * 8 + 4;

shared uint a_q[BM * PITCH];
// a weight block's (scale, offset)
shared vec2 a_c[BK * BM];
shared uint b_q[BN * PITCH];
// a token block's (d, d * sum q)
shared vec2 b_s[BK * BN];

// a thread's share of a step's weight and token blocks, read a step ahead so the reads overlap the products
const uint A_LOADS = (BM * BK + THREADS - 1) / THREADS;
const uint B_LOADS = (BN * BK + THREADS - 1) / THREADS;
uvec4 pa_lo[A_LOADS];
uvec4 pa_hi[A_LOADS];
vec2 pa_c[A_LOADS];
uvec4 pb_lo[B_LOADS];
uvec4 pb_hi[B_LOADS];
vec2 pb_s[B_LOADS];

void prefetch(uint k0, uint n0, uint tokens, uint w_matrix, uint blocks) {
    uint t = gl_LocalInvocationID.x;
    [[unroll]] for (uint l = 0; l < A_LOADS; l++) {
        uint i = t + l * THREADS;
        uint lr = i / BK;
        uint b = k0 + i % BK;
        pa_lo[l] = uvec4(0);
        pa_hi[l] = uvec4(0);
        vec4 c = vec4(0.0);
        if (i < BM * BK && n0 + lr < pc.rows && b < blocks) {
            c = unpack(w_matrix, n0 + lr, b, blocks, pa_lo[l], pa_hi[l]);
        }
        pa_c[l] = c.xz;
    }
    [[unroll]] for (uint l = 0; l < B_LOADS; l++) {
        uint i = t + l * THREADS;
        uint xr = i / BK;
        uint b = k0 + i % BK;
        pb_lo[l] = uvec4(0);
        pb_hi[l] = uvec4(0);
        pb_s[l] = vec2(0.0);
        if (i < BN * BK && xr < tokens && b < blocks) {
            uint xb = x_row(xr) * blocks + b;
            pb_lo[l] = xq[2 * xb];
            pb_hi[l] = xq[2 * xb + 1];
            pb_s[l] = xs[pc.s_off + xb];
        }
    }
}

void main() {
#ifdef TYPE_SHARED_INIT
    // the block type's lookup tables into shared memory, before any thread can leave
    type_shared_init();
#endif
    uint t = gl_LocalInvocationID.x;
    uint n0 = gl_WorkGroupID.x * BM;
    uint tokens;
    uint w_matrix;
    // uniform across the workgroup, so all leave before any barrier
    if (!begin_tile(tokens, w_matrix)) {
        return;
    }
    uint blocks = pc.cols / QK;
    uint lane = gl_SubgroupInvocationID;
    uint a_row0 = (gl_SubgroupID % (BM / WM)) * WM;
    uint b_col0 = (gl_SubgroupID / (BM / WM)) * WN;
    uint col_lane = lane % CM;
    uint row_lane = lane / CM;

    float sums[CM_ROWS * CM_COLS * ELEMS];
    [[unroll]] for (uint i = 0; i < CM_ROWS * CM_COLS * ELEMS; i++) {
        sums[i] = 0.0;
    }
    prefetch(0, n0, tokens, w_matrix, blocks);
    for (uint k0 = 0; k0 < blocks; k0 += BK) {
        [[unroll]] for (uint l = 0; l < A_LOADS; l++) {
            uint i = t + l * THREADS;
            if (i >= BM * BK) {
                break;
            }
            uint at = (i / BK) * PITCH + (i % BK) * 8;
            [[unroll]] for (uint j = 0; j < 4; j++) {
                a_q[at + j] = pa_lo[l][j];
                a_q[at + 4 + j] = pa_hi[l][j];
            }
            a_c[(i % BK) * BM + i / BK] = pa_c[l];
        }
        [[unroll]] for (uint l = 0; l < B_LOADS; l++) {
            uint i = t + l * THREADS;
            if (i >= BN * BK) {
                break;
            }
            uint at = (i / BK) * PITCH + (i % BK) * 8;
            [[unroll]] for (uint j = 0; j < 4; j++) {
                b_q[at + j] = pb_lo[l][j];
                b_q[at + 4 + j] = pb_hi[l][j];
            }
            b_s[(i % BK) * BN + i / BK] = pb_s[l];
        }
        barrier();
        if (k0 + BK < blocks) {
            prefetch(k0 + BK, n0, tokens, w_matrix, blocks);
        }
        // uniform across the subgroup: its tokens past the tile's are all padding
        if (b_col0 < tokens) {
            [[unroll]] for (uint kb = 0; kb < BK; kb++) {
                coopmat<int8_t, gl_ScopeSubgroup, CM, CM, gl_MatrixUseA> a[CM_ROWS * 2];
                coopmat<int8_t, gl_ScopeSubgroup, CM, CM, gl_MatrixUseB> bm[CM_COLS * 2];
                [[unroll]] for (uint r = 0; r < CM_ROWS; r++) {
                    [[unroll]] for (uint h = 0; h < 2; h++) {
                        coopMatLoad(a[r * 2 + h], a_q, (a_row0 + r * CM) * PITCH + kb * 8 + h * 4, PITCH,
                            gl_CooperativeMatrixLayoutRowMajor);
                    }
                }
                [[unroll]] for (uint c = 0; c < CM_COLS; c++) {
                    [[unroll]] for (uint h = 0; h < 2; h++) {
                        coopMatLoad(bm[c * 2 + h], b_q, (b_col0 + c * CM) * PITCH + kb * 8 + h * 4, PITCH,
                            gl_CooperativeMatrixLayoutColumnMajor);
                    }
                }
                vec2 xd[CM_COLS];
                [[unroll]] for (uint c = 0; c < CM_COLS; c++) {
                    xd[c] = b_s[kb * BN + b_col0 + c * CM + col_lane];
                }
                // every tile's products first, independent of each other, then their scaling
                coopmat<int32_t, gl_ScopeSubgroup, CM, CM, gl_MatrixUseAccumulator> acc[CM_ROWS * CM_COLS];
                [[unroll]] for (uint i = 0; i < CM_ROWS * CM_COLS; i++) {
                    acc[i] = coopmat<int32_t, gl_ScopeSubgroup, CM, CM, gl_MatrixUseAccumulator>(0);
                }
                [[unroll]] for (uint h = 0; h < 2; h++) {
                    [[unroll]] for (uint r = 0; r < CM_ROWS; r++) {
                        [[unroll]] for (uint c = 0; c < CM_COLS; c++) {
                            uint i = r * CM_COLS + c;
                            acc[i] = coopMatMulAdd(a[r * 2 + h], bm[c * 2 + h], acc[i]);
                        }
                    }
                }
                [[unroll]] for (uint r = 0; r < CM_ROWS; r++) {
                    vec2 wc[ELEMS];
                    [[unroll]] for (uint e = 0; e < ELEMS; e++) {
                        wc[e] = a_c[kb * BM + a_row0 + r * CM + row_lane + 2 * e];
                    }
                    [[unroll]] for (uint c = 0; c < CM_COLS; c++) {
                        uint tile = (r * CM_COLS + c) * ELEMS;
                        [[unroll]] for (uint e = 0; e < ELEMS; e++) {
                            float q = float(acc[r * CM_COLS + c][e]);
                            sums[tile + e] += q * (wc[e].x * xd[c].x) + wc[e].y * xd[c].y;
                        }
                    }
                }
            }
        }
        barrier();
    }
    [[unroll]] for (uint r = 0; r < CM_ROWS; r++) {
        [[unroll]] for (uint c = 0; c < CM_COLS; c++) {
            uint token = b_col0 + c * CM + col_lane;
            if (token >= tokens) {
                continue;
            }
            uint out_row = y_row(token);
            [[unroll]] for (uint e = 0; e < ELEMS; e++) {
                uint row = n0 + a_row0 + r * CM + row_lane + 2 * e;
                if (row < pc.rows) {
                    y[pc.y_base + out_row * pc.rows + row] = sums[(r * CM_COLS + c) * ELEMS + e];
                }
            }
        }
    }
}
