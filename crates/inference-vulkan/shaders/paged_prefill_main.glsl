// One sequence's prompt rows against vLLM-layout caches (x = 4; f32, or f16 with CACHE_F16) through its block table,
// row r of `rows` causal over the first `context - rows + r + 1` positions, split over the context: a workgroup a
// (split, kv head, ROWS query rows) holds each (row, query head of the kv head) in a pair of threads (half the head
// each), stages TILE positions of keys and values in shared memory for every row to read, and keeps a flash-style
// running max, exponential sum and value sum over its split. It writes paged_decode's partials (max, sum, value sum
// a (row, head, split)), which paged_decode_reduce combines.
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
layout(std430, binding = 1) readonly buffer KC { CACHE_VEC4 key_cache[]; };
layout(std430, binding = 2) readonly buffer VC { CACHE_VEC4 value_cache[]; };
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
    // positions a split covers, a multiple of TILE
    uint span;
} pc;

const uint THREADS = 256;
const uint ROWS = 8;
const uint MAX_GROUP = 16;
const uint TILE = 32;
const uint MAX_HEAD4 = 32;
// a thread's half of a head, in vec4s
const uint HALF4 = MAX_HEAD4 / 2;
// a masked score: finite, as -inf scores came out wrong through exp on NVIDIA's driver (half the masked positions'
// weight leaked into the sums); a tile of nothing but these leaves the running sums alone
const float MASKED = -1e30;

// a tile's keys and values, a position a padded row of vec4s
const uint ROW4 = MAX_HEAD4 + 1;
shared vec4 keys[TILE * ROW4];
// values as floats: the transposing loader writes single components, which a vec4 array would turn into racing
// whole-vector stores
shared float vals[TILE * ROW4 * 4];

void main() {
    uint split = gl_WorkGroupID.x;
    uint kvh = gl_WorkGroupID.y;
    uint row0 = gl_WorkGroupID.z * ROWS;
    uint t = gl_LocalInvocationID.x;
    uint group = pc.heads / pc.kv_heads;
    uint hs4 = pc.head_size / 4;
    uint half4 = hs4 / 2;
    uint bs = pc.block_size;
    uint block_vec4s = hs4 * bs;
    uint kv_slab = kvh * block_vec4s;

    // a pair of threads a (row, head): lane `side` holds vec4s side * half4..
    uint pair = t / 2;
    uint side = t % 2;
    uint r = pair / MAX_GROUP;
    uint g = pair % MAX_GROUP;
    uint row = row0 + r;
    bool live = row < pc.rows && g < group;
    uint len = live ? pc.context - pc.rows + row + 1 : 0;
    uint head = kvh * group + g;

    uint table = pc.table_off;
    // the workgroup's last row sees the most positions
    uint tile_len = pc.context - pc.rows + min(row0 + ROWS, pc.rows);
    uint start = split * pc.span;
    uint end = min(start + pc.span, tile_len);

    vec4 qv[HALF4];
    vec4 acc[HALF4];
    [[unroll]] for (uint i = 0; i < HALF4; i++) {
        qv[i] = live && i < half4 ? q[pc.q_off + (row * pc.heads + head) * hs4 + side * half4 + i] : vec4(0.0);
        acc[i] = vec4(0.0);
    }
    float m = MASKED;
    float l = 0.0;

    for (uint p0 = start; p0 < end; p0 += TILE) {
        uint first_block = p0 / bs;
        // keys are contiguous a block as (d / 4, position) vec4s, values as (d, position / 4) vec4s; past the
        // workgroup's longest context both are zeros, so a stale NaN there cannot leak
        for (uint i = t; i < TILE * hs4; i += THREADS) {
            uint bt = i / block_vec4s;
            uint rr = i % block_vec4s;
            uint p = bt * bs + key_position(rr, bs);
            vec4 v = vec4(0.0);
            if (p0 + p < end) {
                uint block = block_tables[table + first_block + bt];
                v = vec4(key_cache[block * pc.kv_heads * block_vec4s + kv_slab + rr]);
            }
            keys[p * ROW4 + key_dims(rr, bs)] = v;
        }
        for (uint i = t; i < TILE * hs4; i += THREADS) {
            uint bt = i / block_vec4s;
            uint rr = i % block_vec4s;
            uint d = rr / (bs / 4);
            uint p4 = bt * (bs / 4) + rr % (bs / 4);
            vec4 v = vec4(0.0);
            if (p0 + p4 * 4 < end) {
                uint block = block_tables[table + first_block + bt];
                v = vec4(value_cache[block * pc.kv_heads * block_vec4s + kv_slab + rr]);
                uint valid = end - (p0 + p4 * 4);
                if (valid < 4) {
                    v = vec4(v.x, valid > 1 ? v.y : 0.0, valid > 2 ? v.z : 0.0, 0.0);
                }
            }
            [[unroll]] for (uint c = 0; c < 4; c++) {
                vals[(p4 * 4 + c) * ROW4 * 4 + d] = v[c];
            }
        }
        barrier();
        float s[TILE];
        float tile_max = MASKED;
        [[unroll]] for (uint p = 0; p < TILE; p++) {
            float dot_half = 0.0;
            [[unroll]] for (uint i = 0; i < HALF4; i++) {
                if (i < half4) {
                    dot_half += dot(qv[i], keys[p * ROW4 + side * half4 + i]);
                }
            }
            float sc = subgroupClusteredAdd(dot_half, 2) * pc.scale;
            s[p] = p0 + p < min(len, end) ? sc : MASKED;
            tile_max = max(tile_max, s[p]);
        }
        if (tile_max > MASKED) {
            float m_new = max(m, tile_max);
            float corr = exp(m - m_new);
            l *= corr;
            [[unroll]] for (uint i = 0; i < HALF4; i++) {
                acc[i] *= corr;
            }
            [[unroll]] for (uint p = 0; p < TILE; p++) {
                float e = exp(s[p] - m_new);
                l += e;
                [[unroll]] for (uint i = 0; i < HALF4; i++) {
                    if (i < half4) {
                        uint at = (p * ROW4 + side * half4 + i) * 4;
                        acc[i] += e * vec4(vals[at], vals[at + 1], vals[at + 2], vals[at + 3]);
                    }
                }
            }
            m = m_new;
        }
        barrier();
    }
    if (live) {
        uint stride = pc.head_size + 2;
        uint at = ((row * pc.heads + head) * pc.splits + split) * stride;
        if (side == 0) {
            part[at] = m;
            part[at + 1] = l;
        }
        [[unroll]] for (uint i = 0; i < HALF4; i++) {
            if (i < half4) {
                uint d = (side * half4 + i) * 4;
                part[at + 2 + d] = acc[i].x;
                part[at + 3 + d] = acc[i].y;
                part[at + 4 + d] = acc[i].z;
                part[at + 5 + d] = acc[i].w;
            }
        }
    }
}
