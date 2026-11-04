// Mamba-2's prompt scan as chunked SSD, in three passes over CHUNK-position chunks of each batch row: ssd_chunk_state
// (each chunk's own state, from zero), ssd_state_pass (the state each chunk starts from, carried in order, and the
// pool row advanced in place) and ssd_chunk_out (y). The step's math is ssm_scan_step's: dt = clamp(softplus(dt +
// dt_bias)), state = state * exp(dt * a) + dt * x * B, y = state . C + d * x. Scratch: cs (batch, chunks, heads, CHUNK)
// floats, the inclusive sums of dt * a within the chunk, then ds (batch, chunks, heads, head_dim, STATE), a chunk's
// own state, which the state pass overwrites with the state it starts from.

layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) readonly buffer Dt { float dt[]; };
layout(std430, binding = 2) readonly buffer A { float a[]; };
layout(std430, binding = 3) readonly buffer B { float bm[]; };
layout(std430, binding = 4) readonly buffer C { float cm[]; };
layout(std430, binding = 5) readonly buffer D { float d[]; };
layout(std430, binding = 6) readonly buffer DtBias { float dt_bias[]; };
layout(std430, binding = 7) buffer Pool { float pool[]; };
layout(std430, binding = 8) readonly buffer Slots { uint slots[]; };
layout(std430, binding = 9) buffer Y { float y[]; };
layout(std430, binding = 10) buffer Scratch { float scratch[]; };

layout(push_constant) uniform PushConstants {
    uint batch;
    uint heads;
    uint head_dim;
    uint groups;
    uint state;
    float dt_min;
    float dt_max;
    uint pool_off;
    uint slots_off;
    uint x_off;
    uint dt_off;
    uint b_off;
    uint c_off;
    uint seq;
} pc;

const uint PAD_SLOT = 0x7fffffffu;
const uint CHUNK = 32;
const uint STATE = 128;
const uint MAX_HEAD_DIM = 128;
const uint THREADS = 256;

uint chunks() {
    return (pc.seq + CHUNK - 1) / CHUNK;
}

// scratch index of cs[b][k][h][t]
uint cs_at(uint b, uint k, uint h, uint t) {
    return ((b * chunks() + k) * pc.heads + h) * CHUNK + t;
}

// scratch index of ds[b][k][h][p][0]
uint ds_at(uint b, uint k, uint h, uint p) {
    uint cs_len = pc.batch * chunks() * pc.heads * CHUNK;
    return cs_len + (((b * chunks() + k) * pc.heads + h) * pc.head_dim + p) * STATE;
}

// position t of chunk k's dt after the bias, softplus and clamp, 0 past the sequence (so it neither decays nor adds)
float step_dt(uint b, uint k, uint h, uint t) {
    uint pos = k * CHUNK + t;
    if (pos >= pc.seq) {
        return 0.0;
    }
    float raw = dt[pc.dt_off + (b * pc.seq + pos) * pc.heads + h] + dt_bias[h];
    return clamp(log(1.0 + exp(raw)), pc.dt_min, pc.dt_max);
}

// the first element of x / y at (b, pos, h)
uint x_at(uint b, uint pos, uint h) {
    return ((b * pc.seq + pos) * pc.heads + h) * pc.head_dim;
}

// the first element of B / C at (b, pos, the group of h)
uint bc_at(uint b, uint pos, uint h) {
    return ((b * pc.seq + pos) * pc.groups + h / (pc.heads / pc.groups)) * STATE;
}
