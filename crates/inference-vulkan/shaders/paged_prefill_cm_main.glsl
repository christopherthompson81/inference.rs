// paged_prefill_main.glsl's prompt attention on matrix cores (VK_KHR_cooperative_matrix, f16 16 x 16 x 16 into f32),
// with its bindings, push constants and partials: a workgroup (split, kv head, BR query rows) takes positions x the kv
// head's query heads as its rows, stages BC positions of keys and values as f16 a tile, and each of its 4 subgroups
// (of 32) scores its 16 rows against the tile, takes their softmax in pairs of threads and adds P V into its output
// accumulators. A row's running max moves only when a tile passes it by RESCALE nats, the one time its accumulators
// round-trip shared memory to be rescaled; the max written is that reference, which paged_decode_reduce takes as is.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_vote : require

layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer Q { vec4 q[]; };
#ifdef CACHE_QUANT
layout(std430, binding = 1) readonly buffer KC { uint key_cache[]; };
layout(std430, binding = 2) readonly buffer VC { uint value_cache[]; };
#include "kv_quant.glsl"
#elif defined(CACHE_F16)
layout(std430, binding = 1) readonly buffer KC { f16vec4 key_cache[]; };
layout(std430, binding = 2) readonly buffer VC { f16vec4 value_cache[]; };
#else
layout(std430, binding = 1) readonly buffer KC { vec4 key_cache[]; };
layout(std430, binding = 2) readonly buffer VC { vec4 value_cache[]; };
#endif
layout(std430, binding = 3) readonly buffer BT { uint block_tables[]; };
layout(std430, binding = 4) writeonly buffer P { float part[]; };

layout(push_constant) uniform PushConstants {
    uint heads;
    uint kv_heads;
    uint head_size;
    uint block_size;
    float scale;
    // in vec4s
    uint q_off;
    // the sequence's block-table row
    uint table_off;
    // the sequence's cached positions, the prompt rows' included, and its prompt rows
    uint context;
    uint rows;
    uint splits;
    // positions a split covers, a multiple of BC
    uint span;
    // a quantized cache's K and V bits
    uint k_bits;
    uint v_bits;
} pc;

const uint THREADS = 128;
const uint SUBGROUPS = 4;
const uint M = 16;
// query rows a workgroup (M a subgroup), and positions a tile
const uint BR = SUBGROUPS * M;
const uint BC = 32;
const uint MAX_HEAD = 128;
const uint MAX_CHUNKS = MAX_HEAD / M;
// tile row strides, padded against bank conflicts
const uint KV_STRIDE = MAX_HEAD + 8;
const uint S_STRIDE = BC + 4;
const uint P_STRIDE = BC + 8;
// a masked score: finite, as NVIDIA's exp of -inf came out wrong (see paged_prefill_main.glsl)
const float MASKED = -1e30;
// how far a tile's max may pass a row's reference before the row is rescaled (P stays below e^8 in f16)
const float RESCALE = 8.0;

shared float16_t keys[BC * KV_STRIDE];
shared float16_t vals[BC * KV_STRIDE];
// a subgroup's 16 x BC scores, then its scratch for rescaling and writing out one 16 x 16 accumulator
shared float scores[BR * S_STRIDE];
shared float16_t probs[BR * P_STRIDE];
shared float corr_sh[BR];

#ifdef CACHE_QUANT
// the first word of an HND row: block `block`'s position `p` of kv head `kvh`
uint quant_row(uint block, uint kvh, uint p, uint words) {
    return ((block * pc.kv_heads + kvh) * pc.block_size + p) * words;
}

// NAME(p0, end, kvh, bits): the tile's rows of BUF into DST as (position, dims) f16, a thread a 32-value group, which
// reads its row and scale once; positions from `end` are zeros
#define QUANT_TILE(NAME, BUF, DST)                                                                                   \
    void NAME(uint p0, uint end, uint kvh, uint bits) {                                                              \
        uint groups = pc.head_size / QGROUP;                                                                         \
        uint words = quant_row_words(pc.head_size, bits);                                                            \
        float mid = float(1u << (bits - 1));                                                                         \
        uint mask = (1u << bits) - 1u;                                                                               \
        for (uint i = gl_LocalInvocationID.x; i < BC * groups; i += THREADS) {                                       \
            uint p = i / groups;                                                                                     \
            uint g = i % groups;                                                                                     \
            uint at = p * KV_STRIDE + g * QGROUP;                                                                    \
            if (p0 + p >= end) {                                                                                     \
                [[unroll]] for (uint k = 0; k < QGROUP; k++) {                                                       \
                    DST[at + k] = float16_t(0.0);                                                                    \
                }                                                                                                    \
                continue;                                                                                            \
            }                                                                                                        \
            uint pos = p0 + p;                                                                                       \
            uint block = block_tables[pc.table_off + pos / pc.block_size];                                           \
            uint row = quant_row(block, kvh, pos % pc.block_size, words);                                            \
            float s = unpackHalf2x16(BUF[row + pc.head_size * bits / 32 + g / 2])[g % 2] / mid;                      \
            float base = 0.5 - mid;                                                                                  \
            uint w0 = row + g * bits;                                                                                \
            [[unroll]] for (uint k = 0; k < QGROUP; k++) {                                                           \
                uint o = k * bits;                                                                                   \
                uint sh = o % 32;                                                                                    \
                uint v = BUF[w0 + o / 32] >> sh;                                                                     \
                if (sh + bits > 32) {                                                                                \
                    v |= BUF[w0 + o / 32 + 1] << (32 - sh);                                                          \
                }                                                                                                    \
                DST[at + k] = float16_t((float(v & mask) + base) * s);                                               \
            }                                                                                                        \
        }                                                                                                            \
    }
QUANT_TILE(key_tile, key_cache, keys)
QUANT_TILE(value_tile, value_cache, vals)
#endif

#ifndef CACHE_QUANT
// position `pos`'s 4 key values from dimension d (a multiple of 4)
vec4 key4(uint pos, uint kvh, uint d) {
    uint block = block_tables[pc.table_off + pos / pc.block_size];
    uint off = pos % pc.block_size;
#if defined(CACHE_F16)
    // [blocks, kv_heads, head_size / 8, block_size, 8] halves, two f16vec4s a (dims, position)
    uint at = ((block * pc.kv_heads + kvh) * (pc.head_size / 8) + d / 8) * pc.block_size + off;
    return vec4(key_cache[at * 2 + (d / 4) % 2]);
#else
    // [blocks, kv_heads, head_size / 4, block_size, 4]
    return key_cache[((block * pc.kv_heads + kvh) * (pc.head_size / 4) + d / 4) * pc.block_size + off];
#endif
}

// dimension d's values at positions pos..pos+3 (pos a multiple of 4, within one block): [blocks, kv_heads, head_size,
// block_size]
vec4 value4(uint pos, uint kvh, uint d) {
    uint block = block_tables[pc.table_off + pos / pc.block_size];
    uint off = pos % pc.block_size;
    return vec4(value_cache[(((block * pc.kv_heads + kvh) * pc.head_size + d) * pc.block_size + off) / 4]);
}

#endif

void main() {
    uint split = gl_WorkGroupID.x;
    uint kvh = gl_WorkGroupID.y;
    uint t = gl_LocalInvocationID.x;
    uint sg = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint group = pc.heads / pc.kv_heads;
    uint positions = BR / group;
    uint pos0 = gl_WorkGroupID.z * positions;
    uint hs = pc.head_size;
    uint hs4 = hs / 4;
    uint chunks = hs / M;

    // this thread's row of the softmax (a pair of lanes a row, half the tile each), and its query
    uint srow = sg * M + lane / 2;
    uint side = lane % 2;
    uint spos = pos0 + srow / group;
    bool live = srow < positions * group && spos < pc.rows;
    uint len = live ? pc.context - pc.rows + spos + 1 : 0;

    // the workgroup's last position sees the most
    uint last = min(pos0 + positions, pc.rows);
    uint tile_len = pc.context - pc.rows + last;
    uint start = split * pc.span;
    uint end = min(start + pc.span, tile_len);

    // Q's rows as f16 A matrices, staged through `keys` half a workgroup at a time
    coopmat<float16_t, gl_ScopeSubgroup, M, M, gl_MatrixUseA> qa[MAX_CHUNKS];
    [[unroll]] for (uint half_ = 0; half_ < 2; half_++) {
        for (uint i = t; i < BC * hs4; i += THREADS) {
            uint r = half_ * BC + i / hs4;
            uint d = (i % hs4) * 4;
            uint p = pos0 + r / group;
            vec4 v = vec4(0.0);
            if (r < positions * group && p < pc.rows) {
                v = q[pc.q_off + (p * pc.heads + kvh * group + r % group) * hs4 + d / 4];
            }
            uint at = (i / hs4) * KV_STRIDE + d;
            keys[at] = float16_t(v.x);
            keys[at + 1] = float16_t(v.y);
            keys[at + 2] = float16_t(v.z);
            keys[at + 3] = float16_t(v.w);
        }
        barrier();
        if (sg / 2 == half_) {
            [[unroll]] for (uint k = 0; k < MAX_CHUNKS; k++) {
                if (k < chunks) {
                    coopMatLoad(qa[k], keys, ((sg % 2) * M) * KV_STRIDE + k * M, KV_STRIDE,
                                gl_CooperativeMatrixLayoutRowMajor);
                }
            }
        }
        barrier();
    }

    coopmat<float, gl_ScopeSubgroup, M, M, gl_MatrixUseAccumulator> acc[MAX_CHUNKS];
    [[unroll]] for (uint k = 0; k < MAX_CHUNKS; k++) {
        acc[k] = coopmat<float, gl_ScopeSubgroup, M, M, gl_MatrixUseAccumulator>(0.0);
    }
    float m = MASKED;
    float l = 0.0;

    for (uint p0 = start; p0 < end; p0 += BC) {
        // the tile's keys and values as (position, dims) rows; past the workgroup's longest context both are zeros,
        // so a stale NaN there cannot reach P V
#ifdef CACHE_QUANT
        key_tile(p0, end, kvh, pc.k_bits);
        value_tile(p0, end, kvh, pc.v_bits);
#else
        for (uint i = t; i < BC * hs4; i += THREADS) {
            uint p = i / hs4;
            uint d = (i % hs4) * 4;
            vec4 v = p0 + p < end ? key4(p0 + p, kvh, d) : vec4(0.0);
            uint at = p * KV_STRIDE + d;
            keys[at] = float16_t(v.x);
            keys[at + 1] = float16_t(v.y);
            keys[at + 2] = float16_t(v.z);
            keys[at + 3] = float16_t(v.w);
        }
        for (uint i = t; i < (BC / 4) * hs; i += THREADS) {
            uint d = i / (BC / 4);
            uint p = (i % (BC / 4)) * 4;
            vec4 v = p0 + p < end ? value4(p0 + p, kvh, d) : vec4(0.0);
            [[unroll]] for (uint c = 0; c < 4; c++) {
                vals[(p + c) * KV_STRIDE + d] = float16_t(p0 + p + c < end ? v[c] : 0.0);
            }
        }
#endif
        barrier();

        // the subgroup's 16 x BC scores
        [[unroll]] for (uint c = 0; c < BC / M; c++) {
            coopmat<float, gl_ScopeSubgroup, M, M, gl_MatrixUseAccumulator> s =
                coopmat<float, gl_ScopeSubgroup, M, M, gl_MatrixUseAccumulator>(0.0);
            [[unroll]] for (uint k = 0; k < MAX_CHUNKS; k++) {
                if (k < chunks) {
                    coopmat<float16_t, gl_ScopeSubgroup, M, M, gl_MatrixUseB> kb;
                    coopMatLoad(kb, keys, (c * M) * KV_STRIDE + k * M, KV_STRIDE,
                                gl_CooperativeMatrixLayoutColumnMajor);
                    s = coopMatMulAdd(qa[k], kb, s);
                }
            }
            coopMatStore(s, scores, (sg * M) * S_STRIDE + c * M, S_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();

        uint limit = min(len, end);
        float sv[BC / 2];
        float tile_max = MASKED;
        [[unroll]] for (uint j = 0; j < BC / 2; j++) {
            uint col = side * (BC / 2) + j;
            bool valid = p0 + col < limit;
            sv[j] = valid ? scores[srow * S_STRIDE + col] * pc.scale : MASKED;
            tile_max = max(tile_max, sv[j]);
        }
        tile_max = max(tile_max, subgroupShuffleXor(tile_max, 1));
        float corr = 1.0;
        bool moved = tile_max > m + RESCALE || (m == MASKED && tile_max > MASKED);
        if (moved) {
            corr = m == MASKED ? 0.0 : exp(m - tile_max);
            m = tile_max;
            l *= corr;
        }
        [[unroll]] for (uint j = 0; j < BC / 2; j++) {
            float e = sv[j] > MASKED ? exp(sv[j] - m) : 0.0;
            l += e;
            probs[srow * P_STRIDE + side * (BC / 2) + j] = float16_t(e);
        }
        if (side == 0) {
            corr_sh[srow] = corr;
        }
        // a moved row with output so far scales its accumulators: each 16 x 16 through the subgroup's score rows
        if (subgroupAny(moved && corr != 0.0)) {
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            [[unroll]] for (uint k = 0; k < MAX_CHUNKS; k++) {
                if (k < chunks) {
                    coopMatStore(acc[k], scores, (sg * M) * S_STRIDE, S_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
                    subgroupMemoryBarrierShared();
                    subgroupBarrier();
                    [[unroll]] for (uint e = lane; e < M * M; e += 32) {
                        uint r = e / M;
                        scores[(sg * M + r) * S_STRIDE + e % M] *= corr_sh[sg * M + r];
                    }
                    subgroupMemoryBarrierShared();
                    subgroupBarrier();
                    coopMatLoad(acc[k], scores, (sg * M) * S_STRIDE, S_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
                    subgroupBarrier();
                }
            }
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();

        // P V into the accumulators
        [[unroll]] for (uint c = 0; c < BC / M; c++) {
            coopmat<float16_t, gl_ScopeSubgroup, M, M, gl_MatrixUseA> pa;
            coopMatLoad(pa, probs, (sg * M) * P_STRIDE + c * M, P_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
            [[unroll]] for (uint k = 0; k < MAX_CHUNKS; k++) {
                if (k < chunks) {
                    coopmat<float16_t, gl_ScopeSubgroup, M, M, gl_MatrixUseB> vb;
                    coopMatLoad(vb, vals, (c * M) * KV_STRIDE + k * M, KV_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
                    acc[k] = coopMatMulAdd(pa, vb, acc[k]);
                }
            }
        }
        barrier();
    }

    l += subgroupShuffleXor(l, 1);
    uint head = kvh * group + srow % group;
    uint stride = hs + 2;
    uint at = ((spos * pc.heads + head) * pc.splits + split) * stride;
    if (live && side == 0) {
        part[at] = m;
        part[at + 1] = l;
    }
    // each accumulator through the subgroup's score rows: a lane pair writes its row's 16 values
    [[unroll]] for (uint k = 0; k < MAX_CHUNKS; k++) {
        if (k < chunks) {
            coopMatStore(acc[k], scores, (sg * M) * S_STRIDE, S_STRIDE, gl_CooperativeMatrixLayoutRowMajor);
            subgroupMemoryBarrierShared();
            subgroupBarrier();
            if (live) {
                [[unroll]] for (uint j = 0; j < M / 2; j++) {
                    uint col = side * (M / 2) + j;
                    part[at + 2 + k * M + col] = scores[srow * S_STRIDE + col];
                }
            }
            subgroupBarrier();
        }
    }
}
