// The row-streaming variant of mat_mul_q_main.glsl, with the same head hooks: a workgroup takes THREADS weight rows,
// a thread each, and up to BM tokens. Each thread unpacks each of its row's blocks once into registers, as a matvec
// does (no weight tile in shared memory), and dots it with every token's block, which the step stages in shared
// memory and every thread reads at once (a broadcast).
#include "block_at.glsl"

void main() {
    uint t = gl_LocalInvocationID.x;
    uint row = gl_WorkGroupID.x * THREADS + t;
    uint tokens;
    uint w_matrix;
    // uniform across the workgroup, so all leave before any barrier
    if (!begin_tile(tokens, w_matrix)) {
        return;
    }
    uint blocks = pc.cols / QK;
    // a row past the matrix decodes the last one, its sums never written
    uint w_row = min(row, pc.rows - 1);
    float acc[BM];
    [[unroll]] for (uint j = 0; j < BM; j++) {
        acc[j] = 0.0;
    }
    for (uint k0 = 0; k0 < blocks; k0 += BK) {
        for (uint i = t; i < BM * BK; i += THREADS) {
            uint xr = i % BM;
            uint xk = i / BM;
            uint b = k0 + xk;
            uvec4 qlo = uvec4(0);
            uvec4 qhi = uvec4(0);
            vec4 s = vec4(0.0);
            if (xr < tokens && b < blocks) {
                uint xb = x_row(xr) * blocks + b;
                qlo = xq[2 * xb];
                qhi = xq[2 * xb + 1];
                s = xs[pc.s_off + xb];
            }
            x_lo[xk][xr] = qlo;
            x_hi[xk][xr] = qhi;
            x_s[xk][xr] = s;
        }
        barrier();
        [[unroll]] for (uint kk = 0; kk < BK; kk++) {
            if (k0 + kk < blocks) {
                uvec4 lo;
                uvec4 hi;
                vec4 c = unpack(w_matrix, w_row, k0 + kk, blocks, lo, hi);
                [[unroll]] for (uint j = 0; j < BM; j++) {
                    // uniform: the tile's real tokens
                    if (j >= tokens) {
                        break;
                    }
                    acc[j] += block_dot(lo, hi, c, x_lo[kk][j], x_hi[kk][j], x_s[kk][j]);
                }
            }
        }
        barrier();
    }
    if (row < pc.rows) {
        [[unroll]] for (uint j = 0; j < BM; j++) {
            if (j < tokens) {
                y[pc.y_base + y_row(j) * pc.rows + row] = acc[j];
            }
        }
    }
}
