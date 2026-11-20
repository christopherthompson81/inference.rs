// paged_decode_main.glsl's decode over an f16 cache for up to MAX_GROUP query heads a kv head, a head to MAX_HEAD:
// a workgroup a (STREAM_CHUNK positions, kv head, sequence), KEY_PARTS threads scoring each position for every query
// head over their share of its key, read straight from the cache (its 8-dimension rows coalesce across neighbouring
// positions), then each thread summing the chunk's values for its dimensions. No tile is staged, so a workgroup's
// shared memory is its query and scores. It writes paged_decode_main.glsl's partials, which paged_decode_reduce combines.
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_clustered : require

layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer Q { vec4 q[]; };
layout(std430, binding = 1) readonly buffer KC { f16vec4 key_cache[]; };
layout(std430, binding = 2) readonly buffer VC { f16vec4 value_cache[]; };
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
    // a quantized cache's K and V bits (unused here)
    uint k_bits;
    uint v_bits;
    // a sliding window's positions (the sequence's last `window`), or 0 for the whole context
    uint window;
} pc;

const uint THREADS = 128;
const uint CHUNK = STREAM_CHUNK;
// the threads splitting a position's key, each over a contiguous share of its 8-dimension rows
const uint KEY_PARTS = THREADS / CHUNK;
const uint MAX_GROUP = 8;
const uint MAX_HEAD = 512;
// the dimensions a value thread sums
const uint DIM_STEPS = MAX_HEAD / THREADS;
// a head's lanes in the softmax
const uint LANES = THREADS / MAX_GROUP;

shared vec4 qs[MAX_GROUP * MAX_HEAD / 4];
shared vec4 probs[MAX_GROUP * CHUNK / 4];
shared float part_scores[KEY_PARTS > 1 ? KEY_PARTS * MAX_GROUP * CHUNK : 1];
shared float stat_m[MAX_GROUP];
shared float stat_l[MAX_GROUP];

void main() {
    uint c = gl_WorkGroupID.x;
    uint kvh = gl_WorkGroupID.y;
    uint seq = gl_WorkGroupID.z;
    uint t = gl_LocalInvocationID.x;
    uint group = pc.heads / pc.kv_heads;
    uint hs = pc.head_size;
    uint hs4 = hs / 4;
    uint bs = pc.block_size;
    uint len = context_lens[pc.lens_off + seq];
    // positions before `lo` are outside the window; the chunks start at the one holding it
    uint lo = pc.window != 0 && len > pc.window ? len - pc.window : 0;
    uint start = lo / CHUNK * CHUNK + c * CHUNK;
    uint end = min(start + CHUNK, len);
    uint table = pc.tables_off + seq * pc.max_blocks;

    for (uint i = t; i < group * hs4; i += THREADS) {
        qs[i] = q[pc.q_off + (seq * pc.heads + kvh * group) * hs4 + i];
    }
    barrier();

    uint p = start + t % CHUNK;
    uint key_part = t / CHUNK;
    bool valid = p < end && p >= lo;
    float s[MAX_GROUP];
    [[unroll]] for (uint g = 0; g < MAX_GROUP; g++) {
        s[g] = 0.0;
    }
    if (valid) {
        // [blocks, kv_heads, head_size / 8, block_size, 8] halves: two f16vec4s a (8 dimensions, position)
        uint block = block_tables[table + p / bs];
        uint row = ((block * pc.kv_heads + kvh) * (hs / 8)) * bs + p % bs;
        uint rows = hs / 8 / KEY_PARTS;
        for (uint d8 = key_part * rows; d8 < (key_part + 1) * rows; d8++) {
            uint at = (row + d8 * bs) * 2;
            vec4 k0 = vec4(key_cache[at]);
            vec4 k1 = vec4(key_cache[at + 1]);
            [[unroll]] for (uint g = 0; g < MAX_GROUP; g++) {
                if (g < group) {
                    s[g] += dot(qs[g * hs4 + 2 * d8], k0) + dot(qs[g * hs4 + 2 * d8 + 1], k1);
                }
            }
        }
    }
    if (KEY_PARTS == 1) {
        [[unroll]] for (uint g = 0; g < MAX_GROUP; g++) {
            if (g < group) {
                uint at = g * CHUNK + t;
                probs[at / 4][at % 4] = s[g] * pc.scale;
            }
        }
    } else {
        [[unroll]] for (uint g = 0; g < MAX_GROUP; g++) {
            if (g < group) {
                part_scores[(key_part * MAX_GROUP + g) * CHUNK + t % CHUNK] = s[g];
            }
        }
        barrier();
        for (uint i = t; i < group * CHUNK; i += THREADS) {
            float sum = 0.0;
            [[unroll]] for (uint k = 0; k < KEY_PARTS; k++) {
                sum += part_scores[k * MAX_GROUP * CHUNK + i];
            }
            probs[i / 4][i % 4] = sum * pc.scale;
        }
    }
    barrier();
    {
        uint g = t / LANES;
        uint lane = t % LANES;
        // a head's lanes are one cluster, all in or all out
        if (g < group) {
            float m = -1.0 / 0.0;
            for (uint i = lane; i < CHUNK; i += LANES) {
                if (start + i < end && start + i >= lo) {
                    uint at = g * CHUNK + i;
                    m = max(m, probs[at / 4][at % 4]);
                }
            }
            m = subgroupClusteredMax(m, LANES);
            float l = 0.0;
            for (uint i = lane; i < CHUNK; i += LANES) {
                uint at = g * CHUNK + i;
                float e = start + i < end && start + i >= lo ? exp(probs[at / 4][at % 4] - m) : 0.0;
                probs[at / 4][at % 4] = e;
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

    float acc[DIM_STEPS * MAX_GROUP];
    [[unroll]] for (uint k = 0; k < DIM_STEPS * MAX_GROUP; k++) {
        acc[k] = 0.0;
    }
    // the chunk's positions in fours from the window's start: a block holds whole fours
    uint first = max(start, lo / 4 * 4);
    for (uint p4 = first; p4 < end; p4 += 4) {
        uint block = block_tables[table + p4 / bs];
        uint off = p4 % bs;
        uint i4 = (p4 - start) / 4;
        [[unroll]] for (uint j = 0; j < DIM_STEPS; j++) {
            uint d = t + j * THREADS;
            if (d < hs) {
                // [blocks, kv_heads, head_size, block_size] halves, 4 positions a f16vec4
                vec4 v = vec4(value_cache[(((block * pc.kv_heads + kvh) * hs + d) * bs + off) / 4]);
                // past the context, the cache's stale values meet zero scores; zeroed so a NaN there cannot leak
                uint live = end - p4;
                if (live < 4) {
                    v = vec4(v.x, live > 1 ? v.y : 0.0, live > 2 ? v.z : 0.0, 0.0);
                }
                [[unroll]] for (uint g = 0; g < MAX_GROUP; g++) {
                    if (g < group) {
                        acc[j * MAX_GROUP + g] += dot(probs[g * CHUNK / 4 + i4], v);
                    }
                }
            }
        }
    }
    uint stride = hs + 2;
    [[unroll]] for (uint j = 0; j < DIM_STEPS; j++) {
        uint d = t + j * THREADS;
        [[unroll]] for (uint g = 0; g < MAX_GROUP; g++) {
            if (g < group && d < hs) {
                uint row = (seq * pc.heads + kvh * group + g) * pc.chunks + c;
                part[row * stride + 2 + d] = acc[j * MAX_GROUP + g];
            }
        }
    }
    if (t < group) {
        uint row = (seq * pc.heads + kvh * group + t) * pc.chunks + c;
        part[row * stride] = stat_m[t];
        part[row * stride + 1] = stat_l[t];
    }
}
