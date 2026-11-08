#include "block_at.glsl"

void main() {
#ifdef TYPE_SHARED_INIT
    // the block type's lookup tables into shared memory, before any thread can leave
    type_shared_init();
#endif
    uint row0 = (gl_WorkGroupID.y * pc.groups_x + gl_WorkGroupID.x) * ROWS;
    uint tid = gl_LocalInvocationID.x;
    uint blocks = pc.cols / QK;
    float acc[ROWS * MAX_M];
    [[unroll]] for (uint i = 0; i < ROWS * MAX_M; i++) {
        acc[i] = 0.0;
    }
#ifdef GLU
    float acc2[ROWS * MAX_M];
    [[unroll]] for (uint i = 0; i < ROWS * MAX_M; i++) {
        acc2[i] = 0.0;
    }
#endif
    for (uint b = tid; b < blocks; b += THREADS) {
        COEF coef[ROWS];
        uvec4 lo[ROWS];
        uvec4 hi[ROWS];
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            // a row past the matrix decodes the last one, its sums never written
            coef[r] = pack_coef(unpack(pc.w_base, min(row0 + r, pc.rows - 1), b, blocks, lo[r], hi[r]));
        }
#ifdef GLU
        COEF coef2[ROWS];
        uvec4 lo2[ROWS];
        uvec4 hi2[ROWS];
        wsel = 1u;
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            coef2[r] = pack_coef(unpack(pc.w2_base, min(row0 + r, pc.rows - 1), b, blocks, lo2[r], hi2[r]));
        }
        wsel = 0u;
#endif
        // constant bounds, so the accumulators stay in registers
        [[unroll]] for (uint mi = 0; mi < MAX_M; mi++) {
            if (mi >= pc.m) {
                break;
            }
            uint xb = mi * blocks + b;
            uvec4 x_lo = xq[2 * xb], x_hi = xq[2 * xb + 1];
            XSCALE sx = pack_xs(xs[pc.s_off + xb]);
            [[unroll]] for (uint r = 0; r < ROWS; r++) {
                acc[r * MAX_M + mi] += block_dot(lo[r], hi[r], coef[r], x_lo, x_hi, sx);
#ifdef GLU
                acc2[r * MAX_M + mi] += block_dot(lo2[r], hi2[r], coef2[r], x_lo, x_hi, sx);
#endif
            }
        }
    }
    [[unroll]] for (uint mi = 0; mi < MAX_M; mi++) {
        if (mi >= pc.m) {
            break;
        }
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            float sum = subgroupAdd(acc[r * MAX_M + mi]);
#ifdef GLU
            float sum2 = subgroupAdd(acc2[r * MAX_M + mi]);
#endif
            if (subgroupElect()) {
                partial[r * MAX_M + mi][gl_SubgroupID] = sum;
#ifdef GLU
                partial2[r * MAX_M + mi][gl_SubgroupID] = sum2;
#endif
            }
        }
    }
    barrier();
    if (tid == 0) {
        for (uint r = 0; r < ROWS; r++) {
            if (row0 + r >= pc.rows) {
                break;
            }
            for (uint mi = 0; mi < pc.m; mi++) {
                float sum = 0.0;
                for (uint g = 0; g < gl_NumSubgroups; g++) {
                    sum += partial[r * MAX_M + mi][g];
                }
#ifdef GLU
                float sum2 = 0.0;
                for (uint g = 0; g < gl_NumSubgroups; g++) {
                    sum2 += partial2[r * MAX_M + mi][g];
                }
                sum = glu(pc.act, sum, sum2);
#endif
                y[pc.y_base + mi * pc.rows + row0 + r] = sum;
            }
        }
    }
}
