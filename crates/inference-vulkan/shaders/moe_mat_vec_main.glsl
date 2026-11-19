#include "block_at.glsl"

void main() {
#ifdef TYPE_SHARED_INIT
    // the block type's lookup tables into shared memory, before any thread can leave
    type_shared_init();
#endif
    uint slot = gl_WorkGroupID.z;
    uint expert = ids[pc.ids_off + slot];
    uint before = 0;
    for (uint s = 0; s < slot; s++) {
        before += uint(ids[pc.ids_off + s] == expert);
    }
    // the whole workgroup leaves together, before any barrier
    if (before % MAX_M != 0) {
        return;
    }
    uint members[MAX_M];
    uint m = 0;
    for (uint s = slot; s < pc.slots && m < MAX_M; s++) {
        if (ids[pc.ids_off + s] == expert) {
            members[m] = s;
            m++;
        }
    }
    uint row0 = (gl_WorkGroupID.y * pc.groups_x + gl_WorkGroupID.x) * ROWS;
    uint w_expert = pc.w_base + expert * pc.expert_u16;
    uint tid = gl_LocalInvocationID.x;
    uint blocks = pc.cols / QK;
    float acc[ROWS * MAX_M];
    [[unroll]] for (uint i = 0; i < ROWS * MAX_M; i++) {
        acc[i] = 0.0;
    }
    // lanes stride over (row, block) pairs, so rows shorter than the workgroup keep every lane busy
    for (uint i = tid; i < ROWS * blocks; i += THREADS) {
        uint r = i / blocks, b = i - r * blocks;
        uvec4 lo, hi;
        COEF coef = pack_coef(unpack(w_expert, min(row0 + r, pc.rows - 1), b, blocks, lo, hi));
        [[unroll]] for (uint mi = 0; mi < MAX_M; mi++) {
            if (mi >= m) {
                break;
            }
            uint xb = (members[mi] / pc.x_div) * blocks + b;
            float v = block_dot(lo, hi, coef, xq[2 * xb], xq[2 * xb + 1], pack_xs(xs[pc.s_off + xb]));
            [[unroll]] for (uint rr = 0; rr < ROWS; rr++) {
                acc[rr * MAX_M + mi] += rr == r ? v : 0.0;
            }
        }
    }
    [[unroll]] for (uint mi = 0; mi < MAX_M; mi++) {
        if (mi >= m) {
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
            for (uint mi = 0; mi < m; mi++) {
                float sum = 0.0;
                for (uint g = 0; g < gl_NumSubgroups; g++) {
                    sum += partial[r * MAX_M + mi][g];
                }
                y[pc.y_base + members[mi] * pc.rows + row0 + r] = sum;
            }
        }
    }
}
