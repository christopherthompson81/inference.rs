// One query a sequence against vLLM-layout caches (x = 4; f32, or f16 with CACHE_F16) through its block table, split
// over the context: a workgroup a (CHUNK positions, kv head, sequence) scores the chunk for every query head of its kv
// head, a TILE of keys
// at a time staged in shared memory, then sums the values a TILE at a time staged too (as vec4s of 4 positions of a
// dimension, each serving several heads' accumulators). It writes each head's max score, its exponentials' sum and the
// unnormalized value sum (`head_size + 2` floats a (sequence, head, chunk)); paged_decode_reduce combines the chunks.
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Q { vec4 q[]; };
#ifdef CACHE_F16
#define CACHE_VEC4 f16vec4
#else
#define CACHE_VEC4 vec4
#endif

// where vec4 `r` of a block's keys sits: its position, and its four dimensions' index (in fours). A key row holds
// 16 bytes of dimensions a position (x): 4 f32s, so a vec4 a (dims, position); or 8 f16s, two vec4s a position
uint key_position(uint r, uint bs) {
#ifdef CACHE_F16
    return (r / 2) % bs;
#else
    return r % bs;
#endif
}

uint key_dims(uint r, uint bs) {
#ifdef CACHE_F16
    return (r / (2 * bs)) * 2 + r % 2;
#else
    return r / bs;
#endif
}
#ifdef CACHE_QUANT
layout(std430, binding = 1) readonly buffer KC { uint key_cache[]; };
layout(std430, binding = 2) readonly buffer VC { uint value_cache[]; };
#include "kv_quant.glsl"
QUANT_READER(key_at, key_cache)
QUANT_READER(value_at, value_cache)
#else
layout(std430, binding = 1) readonly buffer KC { CACHE_VEC4 key_cache[]; };
layout(std430, binding = 2) readonly buffer VC { CACHE_VEC4 value_cache[]; };
#endif
layout(std430, binding = 3) readonly buffer BT { uint block_tables[]; };
layout(std430, binding = 4) readonly buffer CL { uint context_lens[]; };
layout(std430, binding = 5) writeonly buffer P { float part[]; };

layout(push_constant) uniform PushConstants {
    uint heads;
    uint kv_heads;
    uint head_size;
    uint block_size;
    uint max_blocks;
    float scale;
    // in vec4s
    uint q_off;
    uint tables_off;
    uint lens_off;
    uint chunks;
    // a quantized cache's K and V bits
    uint k_bits;
    uint v_bits;
} pc;
#ifdef CACHE_QUANT
// the first word of an HND row: block `block`'s position `p` of kv head `kvh`
uint quant_row(uint block, uint kvh, uint p, uint words) {
    return ((block * pc.kv_heads + kvh) * pc.block_size + p) * words;
}
#endif

const uint THREADS = 256;
const uint CHUNK = 64;
// a multiple of the block size, so a tile holds whole blocks
const uint TILE = 32;
// a wide build (head size 256) defines both, its group bounded by the value heads a thread sums anyway
#ifndef DECODE_MAX_HEAD4
#define DECODE_MAX_HEAD4 32
#define DECODE_MAX_GROUP 16
#endif
const uint MAX_HEAD4 = DECODE_MAX_HEAD4;
const uint MAX_GROUP = DECODE_MAX_GROUP;
// the heads a thread scores a position for
const uint SCORE_HEADS = MAX_GROUP / (THREADS / TILE);
// the heads a value thread sums for (its dimension's vec4 read once for them all)
const uint MAX_OUT = 8;
const uint LANES = THREADS / MAX_GROUP;

// a kv head's query heads times the head size, at most 2048 floats (the op checks it)
const uint MAX_Q4 = 512;
shared vec4 qs[MAX_Q4];
// a tile's keys (a position a padded row of vec4s), then its values (a dimension a padded row of 4-position vec4s)
const uint VALUE_ROW = TILE / 4 + 1;
const uint KV_VEC4S = max(TILE * (MAX_HEAD4 + 1), 4 * MAX_HEAD4 * VALUE_ROW);
shared vec4 keys[KV_VEC4S];
shared vec4 scores[MAX_GROUP * CHUNK / 4];
shared float stat_m[MAX_GROUP];
shared float stat_l[MAX_GROUP];

void main() {
    uint c = gl_WorkGroupID.x;
    uint kvh = gl_WorkGroupID.y;
    uint seq = gl_WorkGroupID.z;
    uint t = gl_LocalInvocationID.x;
    uint group = pc.heads / pc.kv_heads;
    uint hs4 = pc.head_size / 4;
    uint hp4 = hs4 + 1;
    uint bs = pc.block_size;
    uint block_vec4s = hs4 * bs;
    uint len = context_lens[pc.lens_off + seq];
    uint start = c * CHUNK;
    uint n = start < len ? min(CHUNK, len - start) : 0;
    uint table = pc.tables_off + seq * pc.max_blocks;
    uint kv_slab = kvh * block_vec4s;
#ifdef CACHE_QUANT
    uint k_words = quant_row_words(pc.head_size, pc.k_bits);
    uint v_words = quant_row_words(pc.head_size, pc.v_bits);
#endif

    for (uint i = t; i < group * hs4; i += THREADS) {
        qs[i] = q[pc.q_off + (seq * pc.heads + kvh * group) * hs4 + i];
    }
    for (uint i = t; i < MAX_GROUP * CHUNK / 4; i += THREADS) {
        scores[i] = vec4(0.0);
    }
    barrier();
    for (uint tile0 = 0; tile0 < n; tile0 += TILE) {
        uint first_block = (start + tile0) / bs;
#ifdef CACHE_QUANT
        for (uint i = t; i < TILE * hs4; i += THREADS) {
            uint p = i / hs4;
            uint d = (i % hs4) * 4;
            vec4 v = vec4(0.0);
            if (start + tile0 + p < len) {
                uint row = quant_row(block_tables[table + first_block + p / bs], kvh, p % bs, k_words);
                v = vec4(key_at(row, d, pc.k_bits, pc.head_size), key_at(row, d + 1, pc.k_bits, pc.head_size),
                         key_at(row, d + 2, pc.k_bits, pc.head_size), key_at(row, d + 3, pc.k_bits, pc.head_size));
            }
            keys[p * hp4 + d / 4] = v;
        }
#else
        // a block's keys are contiguous as (d / 4, position) vec4s
        for (uint i = t; i < TILE * hs4; i += THREADS) {
            uint bt = i / block_vec4s;
            uint r = i % block_vec4s;
            uint p = bt * bs + key_position(r, bs);
            vec4 v = vec4(0.0);
            if (start + tile0 + p < len) {
                uint block = block_tables[table + first_block + bt];
                v = vec4(key_cache[block * pc.kv_heads * block_vec4s + kv_slab + r]);
            }
            keys[p * hp4 + key_dims(r, bs)] = v;
        }
#endif
        barrier();
        uint p = t % TILE;
        uint g0 = t / TILE;
        if (tile0 + p < n) {
            float s[SCORE_HEADS];
            [[unroll]] for (uint k = 0; k < SCORE_HEADS; k++) {
                s[k] = 0.0;
            }
            for (uint d4 = 0; d4 < hs4; d4++) {
                vec4 kv = keys[p * hp4 + d4];
                [[unroll]] for (uint k = 0; k < SCORE_HEADS; k++) {
                    uint g = g0 + k * (THREADS / TILE);
                    if (g < group) {
                        s[k] += dot(qs[g * hs4 + d4], kv);
                    }
                }
            }
            [[unroll]] for (uint k = 0; k < SCORE_HEADS; k++) {
                uint g = g0 + k * (THREADS / TILE);
                if (g < group) {
                    uint at = g * CHUNK + tile0 + p;
                    scores[at / 4][at % 4] = s[k] * pc.scale;
                }
            }
        }
        barrier();
    }
    {
        uint g = t / LANES;
        uint lane = t % LANES;
        // a head's lanes are one cluster, all in or all out
        if (g < group) {
            float m = -1.0 / 0.0;
            for (uint p = lane; p < n; p += LANES) {
                uint at = g * CHUNK + p;
                m = max(m, scores[at / 4][at % 4]);
            }
            m = subgroupClusteredMax(m, LANES);
            float l = 0.0;
            for (uint p = lane; p < n; p += LANES) {
                uint at = g * CHUNK + p;
                float e = exp(scores[at / 4][at % 4] - m);
                scores[at / 4][at % 4] = e;
                l += e;
            }
            l = subgroupClusteredAdd(l, LANES);
            if (lane == 0) {
                stat_m[g] = m;
                stat_l[g] = l;
            }
        }
    }
    barrier();
    // thread t sums dimension t % head_size for heads t / head_size, + heads_step, ...
    uint d = t % pc.head_size;
    uint g0 = t / pc.head_size;
    uint heads_step = THREADS / pc.head_size;
    float acc[MAX_OUT];
    [[unroll]] for (uint k = 0; k < MAX_OUT; k++) {
        acc[k] = 0.0;
    }
    for (uint tile0 = 0; tile0 < n; tile0 += TILE) {
        uint first_block = (start + tile0) / bs;
#ifdef CACHE_QUANT
        for (uint i = t; i < TILE * hs4; i += THREADS) {
            uint dd = i / (TILE / 4);
            uint p4 = i % (TILE / 4);
            vec4 v = vec4(0.0);
            [[unroll]] for (uint c = 0; c < 4; c++) {
                uint p = p4 * 4 + c;
                if (start + tile0 + p < len) {
                    uint row = quant_row(block_tables[table + first_block + p / bs], kvh, p % bs, v_words);
                    v[c] = value_at(row, dd, pc.v_bits, pc.head_size);
                }
            }
            keys[dd * VALUE_ROW + p4] = v;
        }
#else
        // a block's values are contiguous as (d, position / 4) vec4s
        for (uint i = t; i < TILE * hs4; i += THREADS) {
            uint bt = i / block_vec4s;
            uint r = i % block_vec4s;
            uint dd = r / (bs / 4);
            uint p4 = bt * (bs / 4) + r % (bs / 4);
            vec4 v = vec4(0.0);
            if (start + tile0 + p4 * 4 < len) {
                uint block = block_tables[table + first_block + bt];
                v = vec4(value_cache[block * pc.kv_heads * block_vec4s + kv_slab + r]);
                // past the context, the cache's stale values meet zero scores; zeroed so a NaN there cannot leak
                uint valid = len - (start + tile0 + p4 * 4);
                if (valid < 4) {
                    v = vec4(v.x, valid > 1 ? v.y : 0.0, valid > 2 ? v.z : 0.0, 0.0);
                }
            }
            keys[dd * VALUE_ROW + p4] = v;
        }
#endif
        barrier();
        uint lim4 = (min(TILE, n - tile0) + 3) / 4;
        for (uint p4 = 0; p4 < lim4; p4++) {
            vec4 v = keys[d * VALUE_ROW + p4];
            [[unroll]] for (uint k = 0; k < MAX_OUT; k++) {
                uint g = g0 + k * heads_step;
                if (g < group) {
                    acc[k] += dot(scores[(g * CHUNK + tile0) / 4 + p4], v);
                }
            }
        }
        barrier();
    }
    uint stride = pc.head_size + 2;
    [[unroll]] for (uint k = 0; k < MAX_OUT; k++) {
        uint g = g0 + k * heads_step;
        if (g < group) {
            uint row = (seq * pc.heads + kvh * group + g) * pc.chunks + c;
            part[row * stride + 2 + d] = acc[k];
        }
    }
    if (t < group) {
        uint row = (seq * pc.heads + kvh * group + t) * pc.chunks + c;
        part[row * stride] = stat_m[t];
        part[row * stride + 1] = stat_l[t];
    }
}
