#include "block_at.glsl"

void main() {
    uint t = gl_LocalInvocationID.x;
    uint n0 = gl_WorkGroupID.x * BN;
    uint tokens;
    uint w_matrix;
    // uniform across the workgroup, so all leave before any barrier
    if (!begin_tile(tokens, w_matrix)) {
        return;
    }
    uint blocks = pc.cols / QK;
    uint tx = t % THREADS_N;
    uint ty = t / THREADS_N;

    float acc[TN][TM];
    [[unroll]] for (uint i = 0; i < TN; i++) {
        [[unroll]] for (uint j = 0; j < TM; j++) {
            acc[i][j] = 0.0;
        }
    }
    for (uint k0 = 0; k0 < blocks; k0 += BK) {
        // the step's weight blocks (block i % BK of row i / BK, so neighbours read a row's adjacent blocks), then its
        // token blocks, a few a thread
        for (uint i = t; i < BN * BK; i += THREADS) {
            uint lr = i / BK;
            uint lk = i % BK;
            uint b = k0 + lk;
            uvec4 lo = uvec4(0);
            uvec4 hi = uvec4(0);
            vec4 c = vec4(0.0);
            if (n0 + lr < pc.rows && b < blocks) {
                c = unpack(w_matrix, n0 + lr, b, blocks, lo, hi);
            }
            w_lo[lk][lr] = lo;
            w_hi[lk][lr] = hi;
            w_c[lk][lr] = c;
        }
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
            uvec4 wl[TN];
            uvec4 wh[TN];
            vec4 wc[TN];
            [[unroll]] for (uint i = 0; i < TN; i++) {
                uint r = tx + i * THREADS_N;
                wl[i] = w_lo[kk][r];
                wh[i] = w_hi[kk][r];
                wc[i] = w_c[kk][r];
            }
            [[unroll]] for (uint j = 0; j < TM; j++) {
                uint c = ty + j * THREADS_M;
                uvec4 ql = x_lo[kk][c];
                uvec4 qh = x_hi[kk][c];
                vec4 sx = x_s[kk][c];
                [[unroll]] for (uint i = 0; i < TN; i++) {
                    acc[i][j] += block_dot(wl[i], wh[i], wc[i], ql, qh, sx);
                }
            }
        }
        barrier();
    }
    [[unroll]] for (uint j = 0; j < TM; j++) {
        uint local = ty + j * THREADS_M;
        if (local >= tokens) {
            continue;
        }
        uint out_row = y_row(local);
        [[unroll]] for (uint i = 0; i < TN; i++) {
            uint row = n0 + tx + i * THREADS_N;
            if (row < pc.rows) {
                y[pc.y_base + out_row * pc.rows + row] = acc[i][j];
            }
        }
    }
}
