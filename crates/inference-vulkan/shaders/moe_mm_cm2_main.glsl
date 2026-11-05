// The body of the moe_mm_cm2_* builds, after moe_mm_cm2_head.glsl and a cm2_<type>.glsl: the including shader
// defines WIDTH, its tiles' routes.
// the routes' rows through decode functions, as llama.cpp's MUL_MAT_ID loads B: a value, or four, of row
// block_coords[0] (h's rows are whole BK blocks), read from `h` directly (the buffer reference is unused)
layout(buffer_reference, std430, buffer_reference_align = 2) buffer DecodeH {
    float16_t v;
};

float16_t decode_h(const in DecodeH bl, const in uint block_coords[2], const in uint coord_in_block[2]) {
    return h[pc.h_base + block_coords[0] * pc.cols + block_coords[1] * BK + coord_in_block[1]];
}

f16vec4 decode_h_v(const in DecodeH bl, const in uint block_coords[2], const in uint coord_in_block[2]) {
    return h4[(pc.h_base + block_coords[0] * pc.cols + block_coords[1] * BK + coord_in_block[1]) / 4];
}

uint first;
uint tokens;
uint n0;

float store_y(const in uint r, const in uint c, const in float v) {
    uint row = n0 + r;
    if (row < pc.rows && c < tokens) {
#ifdef DENSE
        y[pc.y_base + (first + c) * pc.rows + row] = v;
#else
        y[pc.y_base + routing[first + c] * pc.rows + row] = v;
#endif
    }
    return v;
}

// the tile's K loop over NC routes' columns, then its scatter; the whole tile in bounds loads unclamped
#define COMMA ,
// f16 sums run the 3090's tensor cores at twice f32's rate (llama.cpp's default for quantized weights on coopmat2),
// held inside f16's range
#ifdef F32_ACC
#define ACC float
#define ACC_CLAMP(SUM)
#else
#define ACC float16_t
const float16_t ACC_MAX = float16_t(65504.0);
#define ACC_CLAMP(SUM) \
    [[unroll]] for (uint i = 0; i < SUM.length(); ++i) { \
        SUM[i] = clamp(SUM[i], -ACC_MAX, ACC_MAX); \
    }
#endif
#define TILE_LOOP(NC, KS, STORE, DECODE) \
    coopmat<ACC, gl_ScopeWorkgroup, BM, NC, gl_MatrixUseAccumulator> sum = \
        coopmat<ACC, gl_ScopeWorkgroup, BM, NC, gl_MatrixUseAccumulator>(ACC(0.0)); \
    [[dont_unroll]] for (uint k = 0; k < pc.cols; k += KS) { \
        coopmat<float16_t, gl_ScopeWorkgroup, BM, KS, gl_MatrixUseA> a; \
        coopmat<float16_t, gl_ScopeWorkgroup, KS, NC, gl_MatrixUseB> b; \
        if (inside) { \
            coopMatLoadTensorNV(a, w, w_at, sliceTensorLayoutNV(lu, n0, BM, k, KS), DECODE); \
        } else { \
            coopMatLoadTensorNV(a, w, w_at, sliceTensorLayoutNV(la, n0, BM, k, KS), DECODE); \
        } \
        coopMatLoadTensorNV(b, h, 0, sliceTensorLayoutNV(lb, first, NC, k, KS), transpose, decode_h, decode_h_v); \
        sum = coopMatMulAdd(a, b, sum); \
    } \
    ACC_CLAMP(sum) \
    coopmat<float, gl_ScopeWorkgroup, BM, NC, gl_MatrixUseAccumulator> result = \
        coopmat<float, gl_ScopeWorkgroup, BM, NC, gl_MatrixUseAccumulator>(sum); \
    STORE(result, NC)

#define STORE_EACH(SUM, NC) coopMatPerElementNV(SUM, SUM, store_y);
void main() {
#ifdef DENSE
    // one matrix over the tokens in order, WIDTH a tile (tiles_off carries their count); the f16 rows are padded to
    // whole tiles with zeros
    uint expert = 0;
    first = gl_WorkGroupID.y * WIDTH;
    if (first >= pc.tiles_off) {
        return;
    }
    tokens = min(WIDTH, pc.tiles_off - first);
#else
    uint tile = gl_WorkGroupID.y;
    if (tile >= routing[pc.tiles_off]) {
        return;
    }
    uint at = pc.tiles_off + 1 + 3 * tile;
    uint expert = routing[at];
    first = routing[at + 1];
    tokens = routing[at + 2];
    // each build takes the tiles of its width, (WIDTH / 2, WIDTH] routes (the narrowest from 1), the builds dispatched
    // side by side
    if (tokens > WIDTH || (WIDTH > BN_QUARTER && tokens <= WIDTH / 2)) {
        return;
    }
#endif
    n0 = gl_WorkGroupID.x * BM;

    tensorLayoutNV<2, gl_CooperativeMatrixClampModeConstantNV> la =
        createTensorLayoutNV(2, gl_CooperativeMatrixClampModeConstantNV);
    la = setTensorLayoutBlockSizeNV(la, 1, QK);
    la = setTensorLayoutDimensionNV(la, pc.rows, pc.cols);
    la = setTensorLayoutStrideNV(la, pc.cols / QK, 1);
    tensorLayoutNV<2> lb = createTensorLayoutNV(2);
    lb = setTensorLayoutBlockSizeNV(lb, 1, BK);
    lb = setTensorLayoutDimensionNV(lb, pc.h_rows, pc.cols);
    lb = setTensorLayoutStrideNV(lb, pc.cols / BK, 1);
    tensorViewNV<2, false, 1, 0> transpose = createTensorViewNV(2, false, 1, 0);

    uint w_at = pc.w_base + expert * pc.expert_u16;
    tensorLayoutNV<2> lu = createTensorLayoutNV(2);
    lu = setTensorLayoutBlockSizeNV(lu, 1, QK);
    // whole row blocks: a last block past the matrix reads the next expert's rows (its sums never stored)
    lu = setTensorLayoutDimensionNV(lu, (pc.rows + BM - 1) / BM * BM, pc.cols);
    lu = setTensorLayoutStrideNV(lu, pc.cols / QK, 1);
    // clamped only where a block's rows would run past the weight buffer (the last expert's last block)
    bool inside = w_at + (n0 + BM) * (pc.cols / QK) * BLOCK_U16 <= w.length();
    // a tile of few routes multiplies narrower matrices (the rest of the 64 empty)
    TILE_LOOP(WIDTH, BK, STORE_EACH, decode_w COMMA decode_w_v)
}
