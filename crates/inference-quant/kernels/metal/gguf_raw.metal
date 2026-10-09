// Dequantize, row-gather and matvec kernels for the GGUF types held as raw blocks (mainline IQ, ik_llama.cpp's
// trellis and IQK types). The decoders are ports of the CUDA ones in kernels/cuda/mmvq_gguf (ggml-cuda
// dequantize.cuh, ik_llama.cpp convert.cu; MIT): each of a simdgroup's 32 lanes decodes 8 values of a 256-value block.
#include <metal_stdlib>
#include "utils.metal"
#include "gguf_iq_tables.metal"

using namespace metal;

#define QK_K 256
#define IQ1_DELTA 0.125f
#define KT_ROW_META 4
#define KT_TAIL_BLOCK 32
#define KT_INDEX_OFFSET 4096u
#define MV_SIMDGROUPS 4
#define MV_MAX_BATCH 8

typedef device const uchar *bytes_t;

constant int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constant int8_t iq2nl_values[8] = {-31, -13, 1, 17, -26, -8, 6, 22};
constant uint16_t iq2kl_values[32] = {0xe9c1, 0x0dc1, 0xc1d8, 0xf6d8, 0x0dd8, 0x2fd8, 0xd8e9, 0xe9e9,
                                      0x01e9, 0x0de9, 0x1ce9, 0xc1f6, 0x01f6, 0x0df6, 0x2ff6, 0xe901,
                                      0xf601, 0x0101, 0x0d01, 0x1c01, 0xd80d, 0xe90d, 0xf60d, 0x010d,
                                      0x0d0d, 0xc11c, 0xe91c, 0x011c, 0x1c1c, 0x2f1c, 0xe92f, 0x0d2f};
constant int8_t iq3nl_values[16] = {-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51};
constant int8_t iq4k_values[32] = {-127, -104, -83, -65, -49, -35, -22, -10, 1,  13, 25, 38, 53, 69, 89, 113,
                                   -123, -100, -79, -61, -45, -31, -18, -6,  5,  17, 29, 42, 57, 73, 93, 117};
constant int8_t iq5nl_values[64] = {-126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6,
                                    -1,   5,    11,   17,  23,  29,  36,  43,  51,  59,  68,  77,  87,  97,  109, 121,
                                    -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10, -4,
                                    1,    7,    13,   19,  25,  31,  38,  45,  53,  61,  70,  79,  89,  99,  111, 123};
constant int8_t iq6nl_values[128] = {
    -127, -121, -115, -109, -104, -98, -93, -88, -84, -79, -74, -70, -66, -62, -58, -54, -51, -47, -44, -40, -37, -34,
    -31,  -28,  -25,  -22,  -19,  -16, -13, -11, -8,  -5,  -2,  0,   3,   6,   9,   12,  14,  17,  20,  23,  27,  30,
    33,   36,   40,   44,   47,   51,  55,  59,  63,  68,  72,  77,  82,  87,  92,  98,  103, 109, 115, 121, -126, -120,
    -114, -108, -103, -97,  -92,  -87, -83, -78, -73, -69, -65, -61, -57, -53, -50, -46, -43, -39, -36, -33, -30, -27,
    -24,  -21,  -18,  -15,  -12,  -10, -7,  -4,  -1,  1,   4,   7,   10,  13,  15,  18,  21,  24,  28,  31,  34,  37,
    41,   45,   48,   52,   56,   60,  64,  69,  73,  78,  83,  88,  93,  99,  104, 110, 116, 122};

// Byte loads: row-scaled rows put blocks at offsets no wider load can assume aligned
static inline uint ld16(bytes_t p) { return uint(p[0]) | (uint(p[1]) << 8); }
static inline uint ld32(bytes_t p) { return ld16(p) | (ld16(p + 2) << 16); }
static inline float ldh(bytes_t p) { return float(as_type<half>(ushort(ld16(p)))); }
static inline float ldf(bytes_t p) { return as_type<float>(ld32(p)); }
static inline float byte_of(ulong v, int j) { return float((v >> (8 * j)) & 0xff); }
static inline float sign_of(uint signs, int j) { return signs & kmask_iq2xs[j] ? -1.f : 1.f; }

// Eight consecutive values from 32*ib + 8*il, the layout of most 2/3-bit decoders
static inline void runs_of_8(ushort tid, thread ushort *pos) {
  const ushort base = 32 * (tid % 8) + 8 * (tid / 8);
  for (int j = 0; j < 8; ++j) pos[j] = base + j;
}

// Four values from 32*ib + 4*il and four 16 further on, the 4-bit nibble layout
static inline void nibble_pairs(ushort tid, thread ushort *pos) {
  const ushort base = 32 * (tid % 8) + 4 * (tid / 8);
  for (int j = 0; j < 4; ++j) {
    pos[j] = base + j;
    pos[j + 4] = base + 16 + j;
  }
}

// Two values each at offsets 0, step, 2*step and 3*step from `base`
static inline void quads_of_2(ushort base, ushort step, thread ushort *pos) {
  for (int k = 0; k < 4; ++k) {
    pos[2 * k] = base + k * step;
    pos[2 * k + 1] = base + k * step + 1;
  }
}

static inline int trellis_next(thread uint &val) {
  val *= 0xCBAC1FEDu;
  const uint m = val & 0x3f3f3f3fu;
  return int((m & 0xff) + ((m >> 8) & 0xff) + ((m >> 16) & 0xff) + (m >> 24)) - 126;
}

// `row` is the row's first byte, `i` the 256-value block within it; false where the lane has nothing to decode
struct iq2_xxs {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 66 * i;
    const int il = tid / 8, ib = tid % 8;
    bytes_t q2 = x + 2 + 8 * ib;
    const ulong grid = iq2xxs_grid[q2[il]];
    const uint aux32 = ld32(q2 + 4);
    const float d = ldh(x) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) v[j] = d * byte_of(grid, j) * sign_of(signs, j);
    runs_of_8(tid, pos);
    return true;
  }
};

struct iq2_xs {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 74 * i;
    const int il = tid / 8, ib = tid % 8;
    const uint q = ld16(x + 2 + 2 * (4 * ib + il));
    const ulong grid = iq2xs_grid[q & 511];
    const float d = ldh(x) * (0.5f + ((x[66 + ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint signs = ksigns_iq2xs[q >> 9];
    for (int j = 0; j < 8; ++j) v[j] = d * byte_of(grid, j) * sign_of(signs, j);
    runs_of_8(tid, pos);
    return true;
  }
};

struct iq2_s {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 82 * i;
    const int il = tid / 8, ib = tid % 8;
    const ulong grid = iq2s_grid[x[2 + 4 * ib + il] | ((x[66 + ib] << (8 - 2 * il)) & 0x300)];
    const float d = ldh(x) * (0.5f + ((x[74 + ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint signs = x[2 + QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) v[j] = d * byte_of(grid, j) * sign_of(signs, j);
    runs_of_8(tid, pos);
    return true;
  }
};

struct iq3_xxs {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 98 * i;
    const int il = tid / 8, ib = tid % 8;
    bytes_t q3 = x + 2 + 8 * ib;
    const uint grid1 = iq3xxs_grid[q3[2 * il]];
    const uint grid2 = iq3xxs_grid[q3[2 * il + 1]];
    const uint aux32 = ld32(x + 2 + QK_K / 4 + 4 * ib);
    const float d = ldh(x) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
      v[j] = d * byte_of(grid1, j) * sign_of(signs, j);
      v[j + 4] = d * byte_of(grid2, j) * sign_of(signs, j + 4);
    }
    runs_of_8(tid, pos);
    return true;
  }
};

struct iq3_s {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 110 * i;
    const int il = tid / 8, ib = tid % 8;
    bytes_t qs = x + 2 + 8 * ib;
    const uint qh = x[66 + ib];
    const uint grid1 = iq3s_grid[qs[2 * il] | ((qh << (8 - 2 * il)) & 256)];
    const uint grid2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 256)];
    const float d = ldh(x) * (1 + 2 * ((x[106 + ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint signs = x[74 + 4 * ib + il];
    for (int j = 0; j < 4; ++j) {
      v[j] = d * byte_of(grid1, j) * sign_of(signs, j);
      v[j + 4] = d * byte_of(grid2, j) * sign_of(signs, j + 4);
    }
    runs_of_8(tid, pos);
    return true;
  }
};

static inline void iq1_grid(uint index, float d, float delta, thread float *v) {
  const uint grid = iq1s_grid_gpu[index];
  for (int j = 0; j < 4; ++j) {
    v[j] = d * (float((grid >> (8 * j)) & 0x0f) + delta);
    v[j + 4] = d * (float((grid >> (8 * j + 4)) & 0x0f) + delta);
  }
}

struct iq1_s {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 50 * i;
    const int il = tid / 8, ib = tid % 8;
    const uint qh = ld16(x + 34 + 2 * ib);
    const float delta = qh & 0x8000 ? -1 - IQ1_DELTA : -1 + IQ1_DELTA;
    const float d = ldh(x) * (2 * ((qh >> 12) & 7) + 1);
    iq1_grid(x[2 + 4 * ib + il] | (((qh >> 3 * il) & 7) << 8), d, delta, v);
    runs_of_8(tid, pos);
    return true;
  }
};

struct iq1_m {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 56 * i;
    const int il = tid / 8, ib = tid % 8;
    uint sc[4];
    for (int k = 0; k < 4; ++k) sc[k] = ld16(x + 48 + 2 * k);
    const uint scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int ib16 = 2 * ib + il / 2;
    const float d = float(as_type<half>(ushort(scale))) * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const uint qh = x[32 + 2 * ib + il / 2];
    const float delta = qh & (0x08 << 4 * (il % 2)) ? -1 - IQ1_DELTA : -1 + IQ1_DELTA;
    iq1_grid(x[4 * ib + il] | (((qh >> 4 * (il % 2)) & 7) << 8), d, delta, v);
    runs_of_8(tid, pos);
    return true;
  }
};

// IQ4_NL blocks are 32 values; a 256-value step takes eight of them, the last ones possibly past the row
struct iq4_nl {
  static bool dq(bytes_t row, uint i, uint ncols, ushort tid, thread float *v, thread ushort *pos) {
    const int il = tid / 8, ib = tid % 8;
    if (QK_K * i + 32 * ib >= ncols) return false;
    bytes_t x = row + 18 * (8 * i + ib);
    bytes_t q4 = x + 2 + 4 * il;
    const float d = ldh(x);
    for (int j = 0; j < 4; ++j) {
      v[j] = d * kvalues_iq4nl[q4[j] & 0xf];
      v[j + 4] = d * kvalues_iq4nl[q4[j] >> 4];
    }
    nibble_pairs(tid, pos);
    return true;
  }
};

struct iq4_xs {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 136 * i;
    const int il = tid / 8, ib = tid % 8;
    bytes_t q4 = x + 8 + 16 * ib + 4 * il;
    const int ls = ((x[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((ld16(x + 2) >> 2 * ib) & 3) << 4);
    const float d = ldh(x) * (ls - 32);
    for (int j = 0; j < 4; ++j) {
      v[j] = d * kvalues_iq4nl[q4[j] & 0xf];
      v[j + 4] = d * kvalues_iq4nl[q4[j] >> 4];
    }
    nibble_pairs(tid, pos);
    return true;
  }
};

// Trellis rows: an f32 scale, whole blocks, then (IQ3_KT / IQ4_KT) 32-value tail sub-blocks
struct iq1_kt {
  static bool dq(bytes_t row, uint i, uint ncols, ushort ib, thread float *v, thread ushort *pos) {
    if (i >= ncols / QK_K) return false;
    const float scale = ldf(row);
    bytes_t x = row + KT_ROW_META + 56 * i;
    const uint sh = x[ib / 4];
    uint idx = (x[8 + ib] | ((x[40 + ib % 16] << (8 - 4 * (ib / 16))) & 0xf00) | ((sh << (8 - (ib % 4))) & 0x1000)) +
               KT_INDEX_OFFSET;
    const float dl = scale * iq4k_values[sh & 0xf];
    for (int j = 0; j < 8; ++j) {
      v[j] = dl * trellis_next(idx);
      pos[j] = 8 * ib + j;
    }
    return true;
  }
};

struct iq2_kt {
  static bool dq(bytes_t row, uint i, uint ncols, ushort ib, thread float *v, thread ushort *pos) {
    if (i >= ncols / QK_K) return false;
    const float scale = ldf(row);
    bytes_t x = row + KT_ROW_META + 68 * i;
    uint idx = ld16(x + 4 + 2 * ib) + KT_INDEX_OFFSET;
    const float dl = scale * iq4k_values[(x[(ib / 4) % 4] >> 4 * (ib / 16)) & 0xf] * 1.05f;
    for (int j = 0; j < 8; ++j) {
      v[j] = dl * trellis_next(idx);
      pos[j] = 8 * ib + j;
    }
    return true;
  }
};

struct iq3_kt {
  static bool dq(bytes_t row, uint i, uint ncols, ushort ib, thread float *v, thread ushort *pos) {
    const float scale = ldf(row);
    bytes_t x = row + KT_ROW_META + 100 * i;
    for (int j = 0; j < 8; ++j) pos[j] = 8 * ib + j;
    if (i < ncols / QK_K) {
      uint idx = ld16(x + 4 + 2 * ib) + KT_INDEX_OFFSET;
      const float dl = scale * ((x[(ib / 4) % 4] >> 4 * (ib / 16)) & 0xf) * 1.01f;
      const uint mask = 1u << (ib / 4);
      for (int j = 0; j < 8; ++j) {
        const float val = dl * abs(trellis_next(idx));
        v[j] = x[68 + (8 * ib + j) % 32] & mask ? -val : val;
      }
      return true;
    }
    const uint nt = (ncols % QK_K) / KT_TAIL_BLOCK;
    if (ib / 4 >= nt) return false;
    bytes_t qh = x + 8 * nt;
    bytes_t scales = qh + 4 * nt;
    uint idx = ld16(x + 2 * ib) + KT_INDEX_OFFSET;
    const float dl = scale * ((scales[ib / 8] >> 4 * ((ib / 4) & 1)) & 0xf) * 1.01f;
    const uint sgn = qh[ib];
    for (int j = 0; j < 8; ++j) {
      const float val = dl * abs(trellis_next(idx));
      v[j] = sgn & (1u << j) ? -val : val;
    }
    return true;
  }
};

struct iq4_kt {
  static bool dq(bytes_t row, uint i, uint ncols, ushort ib, thread float *v, thread ushort *pos) {
    const float scale = ldf(row);
    bytes_t x = row + KT_ROW_META + 128 * i;
    const int ib32 = ib / 4, ig = ib % 4;
    bytes_t ql;
    uint qh1, qh2, sh;
    int qj;
    if (i < ncols / QK_K) {
      const int jj = ib32 * 8 + 2 * ig;
      ql = x + 32;
      bytes_t qh = ql + 64;
      qj = jj;
      qh1 = (uint(qh[jj % 32]) << (8 - 4 * (jj / 32))) & 0xf00;
      qh2 = (uint(qh[(jj + 1) % 32]) << (8 - 4 * ((jj + 1) / 32))) & 0xf00;
      sh = ld32(x + 4 * ib32);
    } else {
      const uint nt = (ncols % QK_K) / KT_TAIL_BLOCK;
      if (uint(ib32) >= nt) return false;
      ql = x + 16 * ib32 + 4;
      bytes_t qh = x + 16 * ib32 + 12;
      qj = 2 * ig;
      qh1 = (qh[qj / 2] & 0x0f) << 8;
      qh2 = (qh[qj / 2] & 0xf0) << 4;
      sh = ld32(x + 16 * ib32);
    }
    const uint offset = sh & 1 ? KT_INDEX_OFFSET + 32768 : KT_INDEX_OFFSET;
    uint idx1 = ql[qj] + qh1 + (((sh >> (8 + 6 * ig)) & 7) << 12) + offset;
    uint idx2 = ql[qj + 1] + qh2 + (((sh >> (8 + 6 * ig + 3)) & 7) << 12) + offset;
    const float dl = scale * (int((sh & 0xff) >> 1) - 64);
    for (int j = 0; j < 4; ++j) {
      v[j] = dl * trellis_next(idx1);
      v[j + 4] = dl * trellis_next(idx2);
    }
    for (int j = 0; j < 8; ++j) pos[j] = 8 * ib + j;
    return true;
  }
};

struct iq2_k {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 76 * i;
    const int ib128 = tid / 16, il = tid % 16;
    const float d = ldh(x);
    const uint e = ld16(x + 2) >> (8 * ib128 + il / 8);
    bytes_t qs = x + 12 + 32 * ib128 + 2 * il;
    float dl[4];
    for (int k = 0; k < 4; ++k) dl[k] = d * (int((x[4 + 4 * ib128 + k] >> 4 * (il / 8)) & 0xf) - 8);
    for (int j = 0; j < 2; ++j) {
      v[0 + j] = dl[0] * iq2nl_values[((qs[j] >> 0) & 3) + ((e << 2) & 4)];
      v[2 + j] = dl[1] * iq2nl_values[((qs[j] >> 2) & 3) + ((e << 0) & 4)];
      v[4 + j] = dl[2] * iq2nl_values[((qs[j] >> 4) & 3) + ((e >> 2) & 4)];
      v[6 + j] = dl[3] * iq2nl_values[((qs[j] >> 6) & 3) + ((e >> 4) & 4)];
    }
    quads_of_2(128 * ib128 + 2 * il, 32, pos);
    return true;
  }
};

struct iq3_k {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 110 * i;
    const int ib128 = tid / 16, il = tid % 16;
    const float d = ldh(x);
    const uint sh = ld16(x + 4) >> (8 * ib128 + il / 8);
    float dl[4];
    for (int k = 0; k < 4; ++k) {
      const float sign = sh & (1u << (2 * k)) ? -1.f : 1.f;
      dl[k] = d * (2 * ((x[6 + 4 * ib128 + k] >> 4 * (il / 8)) & 0xf) + 1) * sign;
    }
    bytes_t qs = x + 14 + 32 * ib128 + 2 * il;
    bytes_t qh = x + 78 + 2 * il;
    const uint e = ld16(x + 2) >> (8 * ib128 + il / 8);
    for (int j = 0; j < 2; ++j) {
      const uint h = qh[j] >> (4 * (ib128 % 2));
      v[0 + j] = dl[0] * iq3nl_values[(((qs[j] >> 0) & 3) | ((h & 1) << 2)) + ((e << 3) & 8)];
      v[2 + j] = dl[1] * iq3nl_values[(((qs[j] >> 2) & 3) | ((h & 2) << 1)) + ((e << 1) & 8)];
      v[4 + j] = dl[2] * iq3nl_values[(((qs[j] >> 4) & 3) | ((h & 4) >> 0)) + ((e >> 1) & 8)];
      v[6 + j] = dl[3] * iq3nl_values[(((qs[j] >> 6) & 3) | ((h & 8) >> 1)) + ((e >> 3) & 8)];
    }
    quads_of_2(128 * ib128 + 2 * il, 32, pos);
    return true;
  }
};

struct iq4_k {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 144 * i;
    const int il = tid / 8, ib = tid % 8;
    bytes_t q4 = x + 16 + 16 * ib + 4 * il;
    const float d = ldh(x);
    const uint extra = ld16(x + 2);
    const uint sh = x[4 + ib / 2] >> 4 * (ib % 2);
    const uint sl = x[8 + ib];
    const float d1 = d * (int((sl & 0xf) | ((sh << 4) & 0x30)) - 32);
    const float d2 = d * (int((sl >> 4) | ((sh << 2) & 0x30)) - 32);
    constant int8_t *values1 = iq4k_values + 16 * ((extra >> (2 * ib)) & 1);
    constant int8_t *values2 = iq4k_values + 16 * ((extra >> (2 * ib + 1)) & 1);
    for (int j = 0; j < 4; ++j) {
      v[j] = d1 * values1[q4[j] & 0xf];
      v[j + 4] = d2 * values2[q4[j] >> 4];
    }
    nibble_pairs(tid, pos);
    return true;
  }
};

struct iq5_k {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 176 * i;
    const int ib64 = tid / 8, il = tid % 8;
    const float d = ldh(x);
    const uint sh = x[4 + ib64];
    const uint sl0 = x[8 + 2 * ib64], sl1 = x[8 + 2 * ib64 + 1];
    const float dl1 = d * (int((sl0 & 0xf) | ((sh << 4) & 0x30)) - 32);
    const float dl2 = d * (int((sl0 >> 4) | ((sh << 2) & 0x30)) - 32);
    const float dl3 = d * (int((sl1 & 0xf) | ((sh >> 0) & 0x30)) - 32);
    const float dl4 = d * (int((sl1 >> 4) | ((sh >> 2) & 0x30)) - 32);
    bytes_t qs = x + 16 + 32 * ib64 + 2 * il;
    bytes_t qh = x + 144 + 2 * il;
    const uint e = (ld16(x + 2) >> 4 * (ib64 % 4)) & 0xff;
    for (int j = 0; j < 2; ++j) {
      const uint h1 = qh[j] >> 2 * (ib64 % 4), h2 = qh[j + 16] >> 2 * (ib64 % 4);
      v[0 + j] = dl1 * iq5nl_values[(qs[j] & 0xf) | ((h1 & 1) << 4) | ((e << 5) & 0x20)];
      v[2 + j] = dl2 * iq5nl_values[(qs[j + 16] & 0xf) | ((h2 & 1) << 4) | ((e << 4) & 0x20)];
      v[4 + j] = dl3 * iq5nl_values[(qs[j] >> 4) | ((h1 & 2) << 3) | ((e << 3) & 0x20)];
      v[6 + j] = dl4 * iq5nl_values[(qs[j + 16] >> 4) | ((h2 & 2) << 3) | ((e << 2) & 0x20)];
    }
    quads_of_2(64 * ib64 + 2 * il, 16, pos);
    return true;
  }
};

struct iq6_k {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    bytes_t x = row + 212 * i;
    const int ib64 = tid / 8, il = tid % 8;
    const float d = ldh(x);
    float dl[4];
    for (int k = 0; k < 4; ++k) dl[k] = d * float(as_type<char>(x[4 + 4 * ib64 + k]));
    bytes_t qs = x + 20 + 32 * ib64 + 2 * il;
    bytes_t qh = x + 148 + 32 * (ib64 / 2) + 2 * il;
    const uint e = (ld16(x + 2) >> 4 * (ib64 % 4)) & 0xff;
    for (int j = 0; j < 2; ++j) {
      const uint h1 = qh[j] >> 4 * (ib64 % 2), h2 = qh[j + 16] >> 4 * (ib64 % 2);
      const uint q1 = (qs[j] & 0xf) | ((h1 & 0x03) << 4);
      const uint q2 = (qs[j + 16] & 0xf) | ((h2 & 0x03) << 4);
      const uint q3 = (qs[j] >> 4) | ((h1 & 0x0c) << 2);
      const uint q4 = (qs[j + 16] >> 4) | ((h2 & 0x0c) << 2);
      v[0 + j] = dl[0] * (iq6nl_values[q1] + (e & 1 ? 1 : 0));
      v[2 + j] = dl[1] * (iq6nl_values[q2] + (e & 2 ? 1 : 0));
      v[4 + j] = dl[2] * (iq6nl_values[q3] + (e & 4 ? 1 : 0));
      v[6 + j] = dl[3] * (iq6nl_values[q4] + (e & 8 ? 1 : 0));
    }
    quads_of_2(64 * ib64 + 2 * il, 16, pos);
    return true;
  }
};

// Row-scaled IQK rows: an f32 or f16 scale, then whole blocks
struct iq4_ks {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    const float scale = ldf(row);
    bytes_t x = row + 4 + 136 * i;
    const int il = tid / 8, ib = tid % 8;
    bytes_t q4 = x + 8 + 16 * ib + 4 * il;
    const float d = scale * (int(x[ib] & 254) - 127);
    constant int8_t *values = iq4k_values + ((x[ib] & 1) << 4);
    for (int j = 0; j < 4; ++j) {
      v[j] = d * values[q4[j] & 0xf];
      v[j + 4] = d * values[q4[j] >> 4];
    }
    nibble_pairs(tid, pos);
    return true;
  }
};

struct iq2_ks {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    const float d = ldh(row);
    bytes_t x = row + 2 + 70 * i;
    const int ib128 = tid / 16, il = tid % 16;
    const uint e = ld16(x) >> 4 * ib128;
    const uint s0 = x[2 + 2 * ib128], s1 = x[2 + 2 * ib128 + 1];
    const float dl1 = d * (int((s0 & 0xf) | ((e >> 4) & 0x10)) - 16);
    const float dl2 = d * (int((s0 >> 4) | ((e >> 5) & 0x10)) - 16);
    const float dl3 = d * (int((s1 & 0xf) | ((e >> 6) & 0x10)) - 16);
    const float dl4 = d * (int((s1 >> 4) | ((e >> 7) & 0x10)) - 16);
    bytes_t qs = x + 6 + 32 * ib128 + 2 * il;
    for (int j = 0; j < 2; ++j) {
      v[0 + j] = dl1 * iq2nl_values[((qs[j] >> 0) & 3) + ((e << 2) & 4)];
      v[2 + j] = dl2 * iq2nl_values[((qs[j] >> 2) & 3) + ((e << 1) & 4)];
      v[4 + j] = dl3 * iq2nl_values[((qs[j] >> 4) & 3) + ((e >> 0) & 4)];
      v[6 + j] = dl4 * iq2nl_values[((qs[j] >> 6) & 3) + ((e >> 1) & 4)];
    }
    quads_of_2(128 * ib128 + 2 * il, 32, pos);
    return true;
  }
};

struct iq4_kss {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    const float scale = ldf(row);
    bytes_t x = row + 4 + 128 * i;
    const int il = tid / 8, ib = tid % 8;
    bytes_t q4 = x + 16 * ib;
    const uint s32 = (ld32(q4) & 0x00010001) | ((ld32(q4 + 4) & 0x00010001) << 2) |
                     ((ld32(q4 + 8) & 0x00010001) << 4) | ((ld32(q4 + 12) & 0x00010001) << 6);
    const uint ls = (s32 | (s32 >> 15)) & 0xff;
    const float d = scale * (int(ls & 254) - 127);
    constant int8_t *values = iq4k_values + ((ls & 1) << 4);
    uint a0 = ld32(q4 + 4 * il) & 0xfffefffe;
    a0 ^= a0 >> 1;
    const uint a1 = (a0 >> 4) & 0x0f0f0f0f;
    a0 &= 0x0f0f0f0f;
    for (int j = 0; j < 4; ++j) {
      v[j] = d * values[(a0 >> (8 * j)) & 0xff];
      v[j + 4] = d * values[(a1 >> (8 * j)) & 0xff];
    }
    nibble_pairs(tid, pos);
    return true;
  }
};

struct iq5_ks {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    const float d = ldf(row);
    bytes_t x = row + 4 + 168 * i;
    const int ib64 = tid / 8, il = tid % 8;
    const uint s0 = x[2 * ib64], s1 = x[2 * ib64 + 1];
    const float dl1 = d * (int(s0 & 254) - 127);
    const float dl2 = d * (int(s1 & 254) - 127);
    bytes_t qs = x + 8 + 32 * ib64 + 2 * il;
    bytes_t qh = x + 136 + 2 * il;
    constant int8_t *values1 = iq5nl_values + ((s0 & 1) << 5);
    constant int8_t *values2 = iq5nl_values + ((s1 & 1) << 5);
    for (int j = 0; j < 2; ++j) {
      const uint h1 = qh[j] >> 2 * (ib64 % 4), h2 = qh[j + 16] >> 2 * (ib64 % 4);
      v[0 + j] = dl1 * values1[(qs[j] & 0xf) | ((h1 & 1) << 4)];
      v[2 + j] = dl1 * values1[(qs[j + 16] & 0xf) | ((h2 & 1) << 4)];
      v[4 + j] = dl2 * values2[(qs[j] >> 4) | ((h1 & 2) << 3)];
      v[6 + j] = dl2 * values2[(qs[j + 16] >> 4) | ((h2 & 2) << 3)];
    }
    quads_of_2(64 * ib64 + 2 * il, 16, pos);
    return true;
  }
};

struct iq3_ks {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    const float scale = ldh(row);
    bytes_t x = row + 2 + 102 * i;
    const int is = tid / 16, il = tid % 16;
    bytes_t qs = x + 6 + 32 * is + 2 * il;
    bytes_t qh = x + 70 + 2 * il;
    uint e = ld16(x) >> 4 * is;
    float dl[4];
    for (int k = 0; k < 4; ++k) dl[k] = scale * (int(((x[2 + k] >> 4 * is) & 0xf) | ((e << (4 - k)) & 0x10)) - 16);
    e >>= 8;
    constant int8_t *values[4];
    for (int k = 0; k < 4; ++k) values[k] = iq3nl_values + (((e >> k) & 1) << 3);
    for (int j = 0; j < 2; ++j) {
      const uint h = qh[j] >> 4 * is;
      v[0 + j] = dl[0] * values[0][((qs[j] >> 0) & 3) | ((h << 2) & 4)];
      v[2 + j] = dl[1] * values[1][((qs[j] >> 2) & 3) | ((h << 1) & 4)];
      v[4 + j] = dl[2] * values[2][((qs[j] >> 4) & 3) | ((h >> 0) & 4)];
      v[6 + j] = dl[3] * values[3][((qs[j] >> 6) & 3) | ((h >> 1) & 4)];
    }
    quads_of_2(128 * is + 2 * il, 32, pos);
    return true;
  }
};

struct iq2_kl {
  static bool dq(bytes_t row, uint i, uint, ushort tid, thread float *v, thread ushort *pos) {
    const float scale = ldh(row);
    bytes_t x = row + 2 + 86 * i;
    const int ib64 = tid / 8, il = tid % 8;
    bytes_t qs = x + 6 + 16 * ib64 + 2 * il;
    bytes_t qh = x + 70 + 2 * il;
    const uint sh = ld16(x) >> 4 * ib64;
    const float d1 = scale * (int(((x[2 + (2 * ib64) % 4] >> 4 * (ib64 / 2)) & 0xf) | ((sh << 4) & 0x30)) - 32);
    const float d2 = scale * (int(((x[2 + (2 * ib64 + 1) % 4] >> 4 * (ib64 / 2)) & 0xf) | ((sh << 2) & 0x30)) - 32);
    const ushort base = 64 * ib64 + 4 * il;
    for (int j = 0; j < 2; ++j) {
      const uint h = qh[j] >> 2 * ib64;
      const uint val1 = iq2kl_values[(qs[j] & 0xf) | ((h & 1) << 4)];
      const uint val2 = iq2kl_values[(qs[j] >> 4) | ((h & 2) << 3)];
      v[4 * j + 0] = d1 * float(as_type<char>(uchar(val1 & 0xff)));
      v[4 * j + 1] = d1 * float(as_type<char>(uchar(val1 >> 8)));
      v[4 * j + 2] = d2 * float(as_type<char>(uchar(val2 & 0xff)));
      v[4 * j + 3] = d2 * float(as_type<char>(uchar(val2 >> 8)));
      pos[4 * j + 0] = base + 2 * j;
      pos[4 * j + 1] = base + 2 * j + 1;
      pos[4 * j + 2] = base + 2 * j + 32;
      pos[4 * j + 3] = base + 2 * j + 33;
    }
    return true;
  }
};

// One 32-thread threadgroup per 256-value step of a row: (step, row) over the grid
template <typename Q, typename T>
kernel void gguf_raw_dequant(bytes_t src [[buffer(0)]], device T *dst [[buffer(1)]], constant uint &ncols [[buffer(2)]],
                             constant ulong &row_bytes [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]]) {
  float v[8];
  ushort pos[8];
  if (!Q::dq(src + tg.y * row_bytes, tg.x, ncols, tid, v, pos)) return;
  device T *y = dst + ulong(tg.y) * ncols + QK_K * tg.x;
  for (int j = 0; j < 8; ++j) y[pos[j]] = T(v[j]);
}

// As gguf_raw_dequant, over the rows `ids` names; an id past `nrows` reads as a row of zeros
template <typename Q, typename T>
kernel void gguf_raw_get_rows(bytes_t src [[buffer(0)]], device const uint *ids [[buffer(1)]],
                              device T *dst [[buffer(2)]], constant uint &ncols [[buffer(3)]],
                              constant ulong &row_bytes [[buffer(4)]], constant uint &nrows [[buffer(5)]],
                              uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  const uint id = ids[tg.y];
  device T *y = dst + ulong(tg.y) * ncols + QK_K * tg.x;
  if (id >= nrows) {
    for (uint j = 8 * tid; j < 8 * tid + 8 && QK_K * tg.x + j < ncols; ++j) y[j] = T(0.f);
    return;
  }
  float v[8];
  ushort pos[8];
  if (!Q::dq(src + id * row_bytes, tg.x, ncols, tid, v, pos)) return;
  for (int j = 0; j < 8; ++j) y[pos[j]] = T(v[j]);
}

// dst[b, r] = x[b] . w[r] for up to MV_MAX_BATCH activation rows; one simdgroup per weight row
template <typename Q, typename T>
kernel void gguf_raw_mv(bytes_t src [[buffer(0)]], device const T *x [[buffer(1)]], device T *dst [[buffer(2)]],
                        constant uint &ncols [[buffer(3)]], constant ulong &row_bytes [[buffer(4)]],
                        constant uint &nrows [[buffer(5)]], constant uint &batch [[buffer(6)]],
                        uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
  const uint r = tg * MV_SIMDGROUPS + sg;
  if (r >= nrows) return;
  bytes_t row = src + r * row_bytes;
  float acc[MV_MAX_BATCH] = {0};
  const uint steps = (ncols + QK_K - 1) / QK_K;
  for (uint i = 0; i < steps; ++i) {
    float v[8];
    ushort pos[8];
    if (!Q::dq(row, i, ncols, lane, v, pos)) continue;
    for (uint b = 0; b < batch; ++b) {
      device const T *xb = x + ulong(b) * ncols + QK_K * i;
      float sum = 0.f;
      for (int j = 0; j < 8; ++j) sum += v[j] * float(xb[pos[j]]);
      acc[b] += sum;
    }
  }
  for (uint b = 0; b < batch; ++b) {
    const float total = simd_sum(acc[b]);
    if (lane == 0) dst[ulong(b) * nrows + r] = T(total);
  }
}

#define instantiate_gguf_raw_t(name, tname, T)                                                                       \
  template [[host_name("gguf_raw_dequant_" #name "_" #tname)]] [[kernel]] void gguf_raw_dequant<name, T>(            \
      bytes_t, device T *, constant uint &, constant ulong &, uint2, ushort);                                          \
  template [[host_name("gguf_raw_get_rows_" #name "_" #tname)]] [[kernel]] void gguf_raw_get_rows<name, T>(          \
      bytes_t, device const uint *, device T *, constant uint &, constant ulong &, constant uint &, uint2, ushort);     \
  template [[host_name("gguf_raw_mv_" #name "_" #tname)]] [[kernel]] void gguf_raw_mv<name, T>(                      \
      bytes_t, device const T *, device T *, constant uint &, constant ulong &, constant uint &, constant uint &,      \
      uint, ushort, ushort);

#define instantiate_gguf_raw(name)                                                                                   \
  instantiate_gguf_raw_t(name, f32, float) instantiate_gguf_raw_t(name, f16, half)                                     \
      instantiate_gguf_raw_t(name, bf16, bfloat16_t)

instantiate_gguf_raw(iq2_xxs);
instantiate_gguf_raw(iq2_xs);
instantiate_gguf_raw(iq2_s);
instantiate_gguf_raw(iq3_xxs);
instantiate_gguf_raw(iq3_s);
instantiate_gguf_raw(iq1_s);
instantiate_gguf_raw(iq1_m);
instantiate_gguf_raw(iq4_nl);
instantiate_gguf_raw(iq4_xs);
instantiate_gguf_raw(iq1_kt);
instantiate_gguf_raw(iq2_kt);
instantiate_gguf_raw(iq3_kt);
instantiate_gguf_raw(iq4_kt);
instantiate_gguf_raw(iq2_k);
instantiate_gguf_raw(iq3_k);
instantiate_gguf_raw(iq4_k);
instantiate_gguf_raw(iq5_k);
instantiate_gguf_raw(iq6_k);
instantiate_gguf_raw(iq4_ks);
instantiate_gguf_raw(iq2_ks);
instantiate_gguf_raw(iq4_kss);
instantiate_gguf_raw(iq5_ks);
instantiate_gguf_raw(iq3_ks);
instantiate_gguf_raw(iq2_kl);
