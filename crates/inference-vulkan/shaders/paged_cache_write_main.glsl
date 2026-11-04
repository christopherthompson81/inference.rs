// reshape_and_cache over vLLM-layout caches (f32, or f16 with CACHE_F16): token t's key and value rows go to slot
// slot_mapping[t] (i64, a negative slot skipped); keys as [blocks, kv_heads, head_size / X, block_size, X], values as
// [blocks, kv_heads, head_size, block_size].

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer K { float key[]; };
layout(std430, binding = 1) readonly buffer V { float value[]; };
#ifdef CACHE_F16
#define CACHE_T float16_t
#else
#define CACHE_T float
#endif
layout(std430, binding = 2) buffer KC { CACHE_T key_cache[]; };
layout(std430, binding = 3) buffer VC { CACHE_T value_cache[]; };
layout(std430, binding = 4) readonly buffer S { uint slots[]; };

layout(push_constant) uniform PushConstants {
    uint tokens;
    uint kv_heads;
    uint head_size;
    uint block_size;
    uint x;
    uint key_off;
    uint value_off;
    uint slots_off;
} pc;

void main() {
    uint total = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    uint row = pc.kv_heads * pc.head_size;
    for (uint e = gl_GlobalInvocationID.x; e < pc.tokens * row; e += total) {
        uint t = e / row;
        uint h = (e / pc.head_size) % pc.kv_heads;
        uint d = e % pc.head_size;
        // the i64 slot's low and high words
        uint lo = slots[pc.slots_off + 2 * t];
        uint hi = slots[pc.slots_off + 2 * t + 1];
        if ((hi & 0x80000000u) != 0) {
            continue;
        }
        uint block = lo / pc.block_size;
        uint off = lo % pc.block_size;
        uint kidx = (((block * pc.kv_heads + h) * (pc.head_size / pc.x) + d / pc.x) * pc.block_size + off) * pc.x + d % pc.x;
        uint vidx = ((block * pc.kv_heads + h) * pc.head_size + d) * pc.block_size + off;
        key_cache[kidx] = CACHE_T(key[pc.key_off + e]);
        value_cache[vidx] = CACHE_T(value[pc.value_off + e]);
    }
}
