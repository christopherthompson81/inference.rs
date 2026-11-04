#include "block_at.glsl"

void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= pc.count * pc.blocks) {
        return;
    }
    uint i = g / pc.blocks;
    uint b = g % pc.blocks;
    // an id past the vocabulary reads the last row
    uint row = min(ids[pc.ids_off + i], pc.vocab - 1);
    uvec4 lo, hi;
    vec4 c = unpack(pc.w_base, row, b, pc.blocks, lo, hi);
    uint at = g * 32;
    for (uint j = 0; j < 4; j++) {
        for (uint k = 0; k < 4; k++) {
            y[at + 4 * j + k] = c.x * float(bitfieldExtract(int(lo[j]), int(8 * k), 8)) + c.z;
            y[at + 16 + 4 * j + k] = c.y * float(bitfieldExtract(int(hi[j]), int(8 * k), 8)) + c.w;
        }
    }
}
