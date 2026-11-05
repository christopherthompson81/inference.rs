// ranked_topk's shared parts: the workgroup's reductions and select_top, which leaves the K largest of `n` candidates
// in cand_v and cand_i, sorted descending (ties to the earlier candidate, which the includer orders by vocabulary
// index). The includer declares its buffers and defines VALUE(i) and INDEX(i), candidate i's value and vocabulary
// index, before including this.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 1024) in;

const uint THREADS = 1024;
const uint K = 128;
const uint BINS = 256;
const uint MAX_SUBGROUPS = 64;

shared uint hist[BINS];
shared float reduce_f[MAX_SUBGROUPS];
shared uint reduce_u[2 * MAX_SUBGROUPS];
shared uint sel_prefix;
shared uint sel_mask;
shared uint sel_want;
shared float cand_v[K];
shared uint cand_i[K];

// larger float, larger key; NaN and -inf the smallest
uint key_of(float f) {
    if (isnan(f) || isinf(f) && f < 0.0) {
        return 0u;
    }
    uint u = floatBitsToUint(f);
    return (u & 0x80000000u) != 0u ? ~u : u | 0x80000000u;
}

bool live(float f) {
    return !isnan(f) && !(isinf(f) && f < 0.0);
}

float workgroup_max(float v) {
    v = subgroupMax(v);
    if (gl_SubgroupInvocationID == 0) {
        reduce_f[gl_SubgroupID] = v;
    }
    barrier();
    float m = reduce_f[0];
    for (uint s = 1; s < gl_NumSubgroups; s++) {
        m = max(m, reduce_f[s]);
    }
    barrier();
    return m;
}

float workgroup_sum(float v) {
    v = subgroupAdd(v);
    if (gl_SubgroupInvocationID == 0) {
        reduce_f[gl_SubgroupID] = v;
    }
    barrier();
    float s = 0.0;
    for (uint i = 0; i < gl_NumSubgroups; i++) {
        s += reduce_f[i];
    }
    barrier();
    return s;
}

// exclusive prefix sums of two counts over the workgroup, in thread order, and their totals
uvec2 workgroup_scan(uvec2 v, out uvec2 total) {
    uvec2 within = uvec2(subgroupExclusiveAdd(v.x), subgroupExclusiveAdd(v.y));
    uvec2 sums = uvec2(subgroupAdd(v.x), subgroupAdd(v.y));
    if (gl_SubgroupInvocationID == 0) {
        reduce_u[2 * gl_SubgroupID] = sums.x;
        reduce_u[2 * gl_SubgroupID + 1] = sums.y;
    }
    barrier();
    uvec2 before = uvec2(0);
    total = uvec2(0);
    for (uint s = 0; s < gl_NumSubgroups; s++) {
        uvec2 here = uvec2(reduce_u[2 * s], reduce_u[2 * s + 1]);
        if (s < gl_SubgroupID) {
            before += here;
        }
        total += here;
    }
    barrier();
    return before + within;
}


void select_top(uint n) {
    uint t = gl_LocalInvocationID.x;
    // the K-th largest key, a byte at a time from the top
    if (t == 0) {
        sel_prefix = 0u;
        sel_mask = 0u;
        sel_want = K;
    }
    [[unroll]] for (uint pass = 0; pass < 4; pass++) {
        uint shift = 24 - 8 * pass;
        for (uint b = t; b < BINS; b += THREADS) {
            hist[b] = 0u;
        }
        barrier();
        uint prefix = sel_prefix;
        uint mask = sel_mask;
        for (uint i = t; i < n; i += THREADS) {
            uint k = key_of(VALUE(i));
            if ((k & mask) == prefix) {
                atomicAdd(hist[(k >> shift) & 0xffu], 1u);
            }
        }
        barrier();
        if (t == 0) {
            uint want = sel_want;
            uint above = 0u;
            uint b = BINS - 1;
            while (b > 0 && above + hist[b] < want) {
                above += hist[b];
                b--;
            }
            sel_want = want - above;
            sel_prefix = prefix | (b << shift);
            sel_mask = mask | (0xffu << shift);
        }
        barrier();
    }
    uint pivot = sel_prefix;
    uint ties = sel_want;

    // keys above the pivot, then the first `ties` equal to it in candidate order, each thread a contiguous span
    uint span = (n + THREADS - 1) / THREADS;
    uint from = min(t * span, n);
    uint to = min(from + span, n);
    uvec2 counts = uvec2(0);
    for (uint i = from; i < to; i++) {
        uint k = key_of(VALUE(i));
        counts += uvec2(k > pivot ? 1u : 0u, k == pivot ? 1u : 0u);
    }
    uvec2 total;
    uvec2 at = workgroup_scan(counts, total);
    uint above = total.x;
    for (uint i = from; i < to; i++) {
        float v = VALUE(i);
        uint k = key_of(v);
        if (k > pivot) {
            cand_v[at.x] = v;
            cand_i[at.x] = INDEX(i);
            at.x++;
        } else if (k == pivot) {
            if (at.y < ties) {
                cand_v[above + at.y] = v;
                cand_i[above + at.y] = INDEX(i);
            }
            at.y++;
        }
    }
    barrier();

    // bitonic sort of the K candidates: larger value first, lower index among equals
    for (uint size = 2; size <= K; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            if (t < K) {
                uint p = t ^ stride;
                if (p > t) {
                    float va = cand_v[t], vb = cand_v[p];
                    uint ia = cand_i[t], ib = cand_i[p];
                    uint ka = key_of(va), kb = key_of(vb);
                    bool a_first = ka > kb || (ka == kb && ia < ib);
                    bool descending = (t & size) == 0;
                    if (a_first != descending) {
                        cand_v[t] = vb;
                        cand_v[p] = va;
                        cand_i[t] = ib;
                        cand_i[p] = ia;
                    }
                }
            }
            barrier();
        }
    }
}
