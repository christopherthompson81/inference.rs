// Mesh simplification on the GPU, one round of parallel edge collapses at a time: the same arithmetic as the host port
// in trellis2/cleanup.rs (built with --fmad=false so every product rounds as it does there).
#include <stdint.h>

#define SCAN_BLOCK 1024

typedef unsigned int u32;
typedef unsigned long long u64;

struct V3 { float x, y, z; };

__device__ __forceinline__ V3 load3(const float* p, u32 i) { return V3{p[3 * i], p[3 * i + 1], p[3 * i + 2]}; }
__device__ __forceinline__ V3 sub3(V3 a, V3 b) { return V3{a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ V3 add3(V3 a, V3 b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ V3 scale3(V3 a, float s) { return V3{a.x * s, a.y * s, a.z * s}; }
__device__ __forceinline__ float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return V3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// Rust's f32::clamp: NaN stays NaN
__device__ __forceinline__ float clamp01(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

// exclusive scan of each block of SCAN_BLOCK values; the block's total goes to `sums`
extern "C" __global__ void scan_block_u32(const u32* in, u32* out, u32* sums, u32 n) {
    __shared__ u32 warp_sums[32];
    const u32 i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    const u32 lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const u32 x = i < n ? in[i] : 0;
    u32 v = x;
    for (u32 o = 1; o < 32; o <<= 1) {
        const u32 y = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += y;
    }
    if (lane == 31) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        u32 w = warp_sums[lane];
        for (u32 o = 1; o < 32; o <<= 1) {
            const u32 y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        warp_sums[lane] = w;
    }
    __syncthreads();
    if (warp > 0) v += warp_sums[warp - 1];
    if (i < n) out[i] = v - x;
    if (threadIdx.x == SCAN_BLOCK - 1) sums[blockIdx.x] = v;
}

extern "C" __global__ void scan_add_u32(u32* out, const u32* offsets, u32 n) {
    const u32 i = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    if (i < n) out[i] += offsets[blockIdx.x];
}

extern "C" __global__ void fill_u32(u32* p, u32 value, u32 n) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = value;
}

extern "C" __global__ void fill_u64(u64* p, u64 value, u32 n) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = value;
}

extern "C" __global__ void vf_count(const u32* faces, u32 f_count, u32* count) {
    const u32 f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= f_count) return;
    for (u32 k = 0; k < 3; k++) atomicAdd(&count[faces[3 * f + k]], 1u);
}

extern "C" __global__ void vf_fill(const u32* faces, u32 f_count, u32* fill, u32* list) {
    const u32 f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= f_count) return;
    for (u32 k = 0; k < 3; k++) list[atomicAdd(&fill[faces[3 * f + k]], 1u)] = f;
}

// each vertex's faces ascending, the order the host fills them in
extern "C" __global__ void vf_sort(const u32* offsets, u32* list, u32 v_count) {
    const u32 v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= v_count) return;
    const u32 s = offsets[v], e = offsets[v + 1];
    for (u32 i = s + 1; i < e; i++) {
        const u32 x = list[i];
        u32 j = i;
        while (j > s && list[j - 1] > x) { list[j] = list[j - 1]; j--; }
        list[j] = x;
    }
}

// the higher neighbours of v through its faces: entry t is slot t % 3 of face t / 3 of v's list
__device__ __forceinline__ u32 entry(const u32* faces, const u32* list, u32 s, u32 t) {
    return faces[3 * list[s + t / 3] + t % 3];
}

extern "C" __global__ void edge_count(const u32* offsets, const u32* list, const u32* faces, u32 v_count, u32* count) {
    const u32 v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= v_count) return;
    const u32 s = offsets[v], k = 3 * (offsets[v + 1] - s);
    u32 unique = 0;
    for (u32 t = 0; t < k; t++) {
        const u32 n = entry(faces, list, s, t);
        if (n <= v) continue;
        bool first = true;
        for (u32 q = 0; q < t && first; q++) first = entry(faces, list, s, q) != n;
        unique += first;
    }
    count[v] = unique;
}

// edges (v, n) with n ascending per v, and how many faces use each
extern "C" __global__ void edge_fill(const u32* offsets, const u32* list, const u32* faces, u32 v_count,
                                     const u32* edge_offsets, u32* edges, u32* face_counts) {
    const u32 v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= v_count) return;
    const u32 s = offsets[v], k = 3 * (offsets[v + 1] - s), base = edge_offsets[v];
    for (u32 t = 0; t < k; t++) {
        const u32 n = entry(faces, list, s, t);
        if (n <= v) continue;
        bool first = true;
        for (u32 q = 0; q < t && first; q++) first = entry(faces, list, s, q) != n;
        if (!first) continue;
        u32 rank = 0, uses = 0;
        for (u32 q = 0; q < k; q++) {
            const u32 m = entry(faces, list, s, q);
            if (m == n) uses++;
            if (m <= v || m >= n) continue;
            bool mfirst = true;
            for (u32 r = 0; r < q && mfirst; r++) mfirst = entry(faces, list, s, r) != m;
            rank += mfirst;
        }
        edges[2 * (base + rank)] = v;
        edges[2 * (base + rank) + 1] = n;
        face_counts[base + rank] = uses;
    }
}

extern "C" __global__ void boundary_mark(const u32* edges, const u32* face_counts, u32 e_count, u32* boundary) {
    const u32 e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= e_count || face_counts[e] != 1) return;
    boundary[edges[2 * e]] = 1;
    boundary[edges[2 * e + 1]] = 1;
}

extern "C" __global__ void vertex_qem(const u32* offsets, const u32* list, const u32* faces, const float* verts,
                                      u32 v_count, float* qems) {
    const u32 v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= v_count) return;
    float q[10] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 i = offsets[v]; i < offsets[v + 1]; i++) {
        const u32 f = list[i];
        const V3 a = load3(verts, faces[3 * f]), b = load3(verts, faces[3 * f + 1]), c = load3(verts, faces[3 * f + 2]);
        V3 n = cross3(sub3(b, a), sub3(c, a));
        n = scale3(n, 1.0f / sqrtf(dot3(n, n)));
        const float p[4] = {n.x, n.y, n.z, -dot3(n, a)};
        q[0] += p[0] * p[0]; q[1] += p[0] * p[1]; q[2] += p[0] * p[2]; q[3] += p[0] * p[3];
        q[4] += p[1] * p[1]; q[5] += p[1] * p[2]; q[6] += p[1] * p[3];
        q[7] += p[2] * p[2]; q[8] += p[2] * p[3];
        q[9] += p[3] * p[3];
    }
    for (u32 k = 0; k < 10; k++) qems[10 * v + k] = q[k];
}

__device__ __forceinline__ V3 collapse_point(V3 v0, V3 v1, bool b0, bool b1) {
    const float w0 = (b0 && !b1) ? 1.0f : ((!b0 && b1) ? 0.0f : 0.5f);
    return add3(scale3(v0, w0), scale3(v1, 1.0f - w0));
}

extern "C" __global__ void edge_cost(const u32* edges, u32 e_count, const float* verts, const u32* faces,
                                     const u32* offsets, const u32* list, const u32* boundary, const float* qems,
                                     float threshold, float lambda_edge_length, float lambda_skinny,
                                     float skinny_eps, float* costs) {
    const u32 e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= e_count) return;
    const u32 e0 = edges[2 * e], e1 = edges[2 * e + 1];
    const V3 v0 = load3(verts, e0), v1 = load3(verts, e1);
    const V3 v = collapse_point(v0, v1, boundary[e0], boundary[e1]);
    const float length2 = dot3(sub3(v1, v0), sub3(v1, v0));
    float q[10];
    for (u32 k = 0; k < 10; k++) q[k] = qems[10 * e0 + k] + qems[10 * e1 + k];
    float r = 0.0f;
    r += q[0] * v.x * v.x;
    r += 2.0f * q[1] * v.x * v.y;
    r += 2.0f * q[2] * v.x * v.z;
    r += 2.0f * q[3] * v.x;
    r += q[4] * v.y * v.y;
    r += 2.0f * q[5] * v.y * v.z;
    r += 2.0f * q[6] * v.y;
    r += q[7] * v.z * v.z;
    r += 2.0f * q[8] * v.z;
    r += q[9];
    float cost = r;
    cost += lambda_edge_length * length2;
    if (cost > threshold) { costs[e] = cost; return; }
    float skinny = 0.0f;
    u32 tris = 0;
    for (u32 side = 0; side < 2; side++) {
        const u32 keep = side == 0 ? e0 : e1, other = side == 0 ? e1 : e0;
        for (u32 i = offsets[keep]; i < offsets[keep + 1]; i++) {
            const u32 f = list[i];
            const u32 ids[3] = {faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]};
            if (ids[0] == other || ids[1] == other || ids[2] == other) continue;
            const V3 abc[3] = {load3(verts, ids[0]), load3(verts, ids[1]), load3(verts, ids[2])};
            const V3 na = ids[0] == keep ? v : abc[0];
            const V3 nb = ids[1] == keep ? v : abc[1];
            const V3 nc = ids[2] == keep ? v : abc[2];
            const V3 old_n = cross3(sub3(abc[1], abc[0]), sub3(abc[2], abc[0]));
            const V3 e1n = sub3(nb, na), e2n = sub3(nc, na);
            const V3 new_n = cross3(e1n, e2n);
            if (dot3(old_n, new_n) < 0.0f) { costs[e] = __int_as_float(0x7f800000); return; }
            const float area = 0.5f * sqrtf(dot3(new_n, new_n));
            const V3 cb = sub3(nc, nb);
            const float denom = fmaxf(dot3(cb, cb) + dot3(e1n, e1n) + dot3(e2n, e2n), skinny_eps);
            const float shape = 4.0f * sqrtf(3.0f) * area / denom;
            skinny += 1.0f - clamp01(shape);
            tris++;
        }
    }
    if (tris > 0) skinny /= (float)tris;
    costs[e] = cost + lambda_skinny * skinny * length2;
}

__device__ __forceinline__ u64 pack(float cost, u32 e) {
    return ((u64)__float_as_uint(cost) << 32) | (u64)e;
}

extern "C" __global__ void edge_best(const u32* edges, u32 e_count, const float* costs, const u32* offsets,
                                     const u32* list, u64* best) {
    const u32 e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= e_count) return;
    const u64 p = pack(costs[e], e);
    for (u32 side = 0; side < 2; side++) {
        const u32 v = edges[2 * e + side];
        for (u32 i = offsets[v]; i < offsets[v + 1]; i++) atomicMin(&best[list[i]], p);
    }
}

// a winner is under the threshold (NaN passes, as on the host) and the cheapest edge at every face around it; winners
// share no face and no vertex, so they collapse side by side
extern "C" __global__ void edge_collapse(const u32* edges, u32 e_count, const float* costs, float threshold,
                                         const u32* offsets, const u32* list, const u64* best, const u32* boundary,
                                         float* verts, u32* faces, u32* vertex_kept, u32* face_kept) {
    const u32 e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= e_count || costs[e] > threshold) return;
    const u64 p = pack(costs[e], e);
    const u32 e0 = edges[2 * e], e1 = edges[2 * e + 1];
    for (u32 side = 0; side < 2; side++) {
        const u32 v = side == 0 ? e0 : e1;
        for (u32 i = offsets[v]; i < offsets[v + 1]; i++)
            if (best[list[i]] != p) return;
    }
    const V3 c = collapse_point(load3(verts, e0), load3(verts, e1), boundary[e0], boundary[e1]);
    verts[3 * e0] = c.x;
    verts[3 * e0 + 1] = c.y;
    verts[3 * e0 + 2] = c.z;
    vertex_kept[e1] = 0;
    for (u32 i = offsets[e0]; i < offsets[e0 + 1]; i++) {
        const u32 f = list[i];
        if (faces[3 * f] == e1 || faces[3 * f + 1] == e1 || faces[3 * f + 2] == e1) face_kept[f] = 0;
    }
    for (u32 i = offsets[e1]; i < offsets[e1 + 1]; i++) {
        const u32 f = list[i];
        for (u32 k = 0; k < 3; k++)
            if (faces[3 * f + k] == e1) { faces[3 * f + k] = e0; break; }
    }
}

extern "C" __global__ void compact_vertices(const float* verts, const u32* kept, const u32* position, u32 v_count,
                                            float* out) {
    const u32 v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= v_count || !kept[v]) return;
    const u32 o = position[v];
    out[3 * o] = verts[3 * v];
    out[3 * o + 1] = verts[3 * v + 1];
    out[3 * o + 2] = verts[3 * v + 2];
}

extern "C" __global__ void compact_faces(const u32* faces, const u32* kept, const u32* position, const u32* remap,
                                         u32 f_count, u32* out) {
    const u32 f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= f_count || !kept[f]) return;
    const u32 o = position[f];
    for (u32 k = 0; k < 3; k++) out[3 * o + k] = remap[faces[3 * f + k]];
}

// closest points over a bounding volume hierarchy, the host's traversal (nearer child first, a box skipped when no
// nearer than the best so far) and Ericson's closest point on a triangle
#define BVH_STACK 128

__device__ __forceinline__ float box_distance2(const float* box, V3 p) {
    const float d0 = fmaxf(fmaxf(box[0] - p.x, p.x - box[3]), 0.0f);
    const float d1 = fmaxf(fmaxf(box[1] - p.y, p.y - box[4]), 0.0f);
    const float d2 = fmaxf(fmaxf(box[2] - p.z, p.z - box[5]), 0.0f);
    return d0 * d0 + d1 * d1 + d2 * d2;
}

__device__ __forceinline__ V3 mad3(V3 a, V3 b, float s) { return V3{a.x + b.x * s, a.y + b.y * s, a.z + b.z * s}; }

__device__ V3 closest_on_triangle(V3 p, V3 a, V3 b, V3 c) {
    const V3 ab = sub3(b, a), ac = sub3(c, a), ap = sub3(p, a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const V3 bp = sub3(p, b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return mad3(a, ab, d1 / (d1 - d3));
    const V3 cp = sub3(p, c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return mad3(a, ac, d2 / (d2 - d6));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
        return mad3(b, sub3(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    const float denom = 1.0f / (va + vb + vc);
    return mad3(mad3(a, ab, vb * denom), ac, vc * denom);
}

// boxes: min xyz, max xyz per node; links: (first triangle, count) for a leaf, (right child, 0) for an inner node
extern "C" __global__ void bvh_closest(const float* boxes, const u32* links, const float* tris, const u32* tri_faces,
                                       const float* points, u32 n, float* distance, u32* face, float* closest) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const V3 p = load3(points, i);
    float best = __int_as_float(0x7f800000);
    V3 best_q = p;
    u32 best_face = 0;
    u32 stack[BVH_STACK];
    u32 top = 1;
    stack[0] = 0;
    while (top > 0) {
        const u32 node = stack[--top];
        if (box_distance2(boxes + 6 * node, p) >= best) continue;
        const u32 start = links[2 * node], count = links[2 * node + 1];
        if (count > 0) {
            for (u32 t = start; t < start + count; t++) {
                const float* tri = tris + 9 * t;
                const V3 q = closest_on_triangle(p, V3{tri[0], tri[1], tri[2]}, V3{tri[3], tri[4], tri[5]},
                                                 V3{tri[6], tri[7], tri[8]});
                const V3 d = sub3(q, p);
                const float d2 = dot3(d, d);
                if (d2 < best) { best = d2; best_q = q; best_face = tri_faces[t]; }
            }
            continue;
        }
        const u32 l = node + 1, r = start;
        const float dl = box_distance2(boxes + 6 * l, p), dr = box_distance2(boxes + 6 * r, p);
        stack[top] = dl < dr ? r : l;
        stack[top + 1] = dl < dr ? l : r;
        top += 2;
    }
    const V3 d = sub3(best_q, p);
    distance[i] = sqrtf(dot3(d, d));
    face[i] = best_face;
    closest[3 * i] = best_q.x;
    closest[3 * i + 1] = best_q.y;
    closest[3 * i + 2] = best_q.z;
}
