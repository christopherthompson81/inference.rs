void main() {
    uint row0 = (gl_WorkGroupID.y * pc.groups_x + gl_WorkGroupID.x) * ROWS;
    uint tid = gl_LocalInvocationID.x;
    uint blocks = pc.cols / QK;
    float acc[ROWS * MAX_M];
    [[unroll]] for (uint i = 0; i < ROWS * MAX_M; i++) {
        acc[i] = 0.0;
    }
    for (uint b = tid; b < blocks; b += THREADS) {
        float d[ROWS];
        uvec4 lo[ROWS];
        uvec4 hi[ROWS];
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            // a row past the matrix decodes the last one, its sums never written
            uint base = (min(row0 + r, pc.rows - 1) * blocks + b) * BLOCK_U16 + pc.w_base;
            d[r] = unpack_block(base, lo[r], hi[r]);
        }
        // constant bounds, so the accumulators stay in registers
        [[unroll]] for (uint mi = 0; mi < MAX_M; mi++) {
            if (mi >= pc.m) {
                break;
            }
            uint xb = mi * blocks + b;
            uvec4 ql = xq[2 * xb];
            uvec4 qh = xq[2 * xb + 1];
            vec2 sx = xs[pc.s_off + xb];
            [[unroll]] for (uint r = 0; r < ROWS; r++) {
                // signed bytes both sides: Q5_0's 0..31 fit
                ivec4 a = ivec4(lo[r]);
                ivec4 c = ivec4(hi[r]);
                ivec4 p = ivec4(ql);
                ivec4 o = ivec4(qh);
                int dot = dotPacked4x8EXT(a.x, p.x) + dotPacked4x8EXT(a.y, p.y) + dotPacked4x8EXT(a.z, p.z)
                    + dotPacked4x8EXT(a.w, p.w) + dotPacked4x8EXT(c.x, o.x) + dotPacked4x8EXT(c.y, o.y)
                    + dotPacked4x8EXT(c.z, o.z) + dotPacked4x8EXT(c.w, o.w);
                acc[r * MAX_M + mi] += d[r] * (sx.x * float(dot) - BIAS * sx.y);
            }
        }
    }
    [[unroll]] for (uint mi = 0; mi < MAX_M; mi++) {
        if (mi >= pc.m) {
            break;
        }
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            float sum = subgroupAdd(acc[r * MAX_M + mi]);
            if (subgroupElect()) {
                partial[r * MAX_M + mi][gl_SubgroupID] = sum;
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
                y[pc.y_base + mi * pc.rows + row0 + r] = sum;
            }
        }
    }
}
