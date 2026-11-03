// a subgroup's sums, combined across the workgroup's subgroups by thread 0
const uint MAX_SUBGROUPS = THREADS / 8;
shared float partial[ROWS * MAX_M][MAX_SUBGROUPS];

void main() {
    uint row0 = (gl_WorkGroupID.y * pc.groups_x + gl_WorkGroupID.x) * ROWS;
    uint tid = gl_LocalInvocationID.x;
    uint blocks = pc.cols / QK;
    float acc[ROWS * MAX_M];
    for (uint i = 0; i < ROWS * MAX_M; i++) {
        acc[i] = 0.0;
    }
    for (uint b = tid; b < blocks; b += THREADS) {
        float v[ROWS][QK];
        [[unroll]] for (uint r = 0; r < ROWS; r++) {
            // a row past the matrix decodes the last one, its sums never written
            decode((min(row0 + r, pc.rows - 1) * blocks + b) * BLOCK_U16 + pc.w_base, v[r]);
        }
        // constant bounds, so the accumulators stay in registers
        [[unroll]] for (uint mi = 0; mi < MAX_M; mi++) {
            if (mi >= pc.m) {
                break;
            }
            uint xb = pc.x_base + mi * pc.cols + b * QK;
            float s0 = 0.0;
            float s1 = 0.0;
            [[unroll]] for (uint j = 0; j < QK; j++) {
                float xv = x[xb + j];
                s0 += v[0][j] * xv;
                s1 += v[1][j] * xv;
            }
            acc[mi] += s0;
            acc[MAX_M + mi] += s1;
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
