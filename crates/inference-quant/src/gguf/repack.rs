//! Inverses of ik_llama.cpp's `repack_*`: its `_R4` / `_R8` types interleave 4 or 8 rows of a base type's blocks.

use inference_tensor::Result;

const QK: usize = 32;
const QK_K: usize = 256;
const HALF: usize = 2;
const SUB_BLOCKS_K: usize = QK_K / QK;

const Q4_0: u32 = 2;
const Q5_0: u32 = 6;
const Q8_0: u32 = 8;
const Q2_K: u32 = 10;
const Q3_K: u32 = 11;
const Q4_K: u32 = 12;
const Q5_K: u32 = 13;
const Q6_K: u32 = 14;
const IQ2_XXS: u32 = 16;
const IQ2_XS: u32 = 17;
const IQ3_XXS: u32 = 18;
const IQ4_NL: u32 = 20;
const IQ3_S: u32 = 21;
const IQ2_S: u32 = 22;
const IQ4_XS: u32 = 23;
const MXFP4: u32 = 39;
const IQ2_K: u32 = 137;
const IQ3_K: u32 = 138;
const IQ4_K: u32 = 139;
const IQ5_K: u32 = 140;
const IQ4_KS: u32 = 144;
const IQ5_KS: u32 = 152;
// IQ4_KS / IQ5_KS rows lead with an f32 scale; a repacked group leads with its rows' scales
const ROW_SCALE_BYTES: usize = 4;

/// A repacked ggml type: its id, the base type it interleaves, and the rows per interleaved group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Repacked {
    pub id: u32,
    pub base: u32,
    pub rows: usize,
}

const fn r4(id: u32, base: u32) -> Repacked {
    Repacked { id, base, rows: 4 }
}

const fn r8(id: u32, base: u32) -> Repacked {
    Repacked { id, base, rows: 8 }
}

const REPACKED: [Repacked; 22] = [
    r8(202, Q4_0),
    r4(206, Q5_0),
    r8(208, Q8_0),
    r4(210, Q2_K),
    r4(211, Q3_K),
    r4(212, Q4_K),
    r4(213, Q5_K),
    r4(214, Q6_K),
    r4(216, IQ2_XXS),
    r4(217, IQ2_XS),
    r4(218, IQ3_XXS),
    r4(220, IQ4_NL),
    r4(221, IQ3_S),
    r4(222, IQ2_S),
    r8(223, IQ4_XS),
    r4(337, IQ2_K),
    r4(338, IQ3_K),
    r4(339, IQ4_K),
    r4(340, IQ5_K),
    r4(344, IQ4_KS),
    r4(352, IQ5_KS),
    r8(353, MXFP4),
];

pub const fn repacked(id: u32) -> Option<Repacked> {
    let mut i = 0;
    while i < REPACKED.len() {
        if REPACKED[i].id == id {
            return Some(REPACKED[i]);
        }
        i += 1;
    }
    None
}

struct Layout {
    block_elems: usize,
    block_bytes: usize,
    row_meta: usize,
    unpack_block: UnpackBlock,
}

fn layout(base: u32) -> Layout {
    let (block_elems, block_bytes, row_meta, unpack_block): (usize, usize, usize, UnpackBlock) =
        match base {
            Q4_0 => (QK, Q4_0_BYTES, 0, q4_0),
            Q5_0 => (QK, Q5_0_BYTES, 0, q5_0),
            Q8_0 => (QK, Q8_0_BYTES, 0, q8_0),
            Q2_K => (QK_K, Q2_K_BYTES, 0, q2_k),
            Q3_K => (QK_K, Q3_K_BYTES, 0, q3_k),
            Q4_K => (QK_K, Q4_K_BYTES, 0, q4_k),
            Q5_K => (QK_K, Q5_K_BYTES, 0, q5_k),
            Q6_K => (QK_K, Q6_K_BYTES, 0, q6_k),
            IQ2_XXS => (QK_K, IQ2_XXS_BYTES, 0, iq2_xxs),
            IQ2_XS => (QK_K, IQ2_XS_BYTES, 0, iq2_xs),
            IQ3_XXS => (QK_K, IQ3_XXS_BYTES, 0, iq3_xxs),
            IQ4_NL => (QK, IQ4_NL_BYTES, 0, iq4_nl),
            IQ3_S => (QK_K, IQ3_S_BYTES, 0, iq3_s),
            IQ2_S => (QK_K, IQ2_S_BYTES, 0, iq2_s),
            IQ4_XS => (QK_K, IQ4_XS_BYTES, 0, iq4_xs),
            MXFP4 => (QK, MXFP4_BYTES, 0, mxfp4),
            IQ2_K => (QK_K, IQ2_K_BYTES, 0, iq2_k),
            IQ3_K => (QK_K, IQ3_K_BYTES, 0, iq3_k),
            IQ4_K => (QK_K, IQ4_K_BYTES, 0, iq4_k),
            IQ5_K => (QK_K, IQ5_K_BYTES, 0, iq5_k),
            IQ4_KS => (QK_K, IQ4_KS_BYTES, ROW_SCALE_BYTES, iq4_ks),
            _ => (QK_K, IQ5_KS_BYTES, ROW_SCALE_BYTES, iq5_ks),
        };
    Layout {
        block_elems,
        block_bytes,
        row_meta,
        unpack_block,
    }
}

/// The base type's bytes for `bytes` of repacked type `id` with rows of `cols` elements.
pub fn unpack(id: u32, cols: usize, bytes: &[u8]) -> Result<Vec<u8>> {
    let Some(ty) = repacked(id) else {
        inference_tensor::bail!("ggml type {id} is not a repacked type");
    };
    let Layout {
        block_elems,
        block_bytes,
        row_meta,
        unpack_block,
    } = layout(ty.base);
    if cols == 0 || !cols.is_multiple_of(block_elems) {
        inference_tensor::bail!(
            "repacked type {id} needs rows of whole {block_elems}-element blocks, got {cols}"
        );
    }
    let row_bytes = row_meta + cols / block_elems * block_bytes;
    let group_bytes = ty.rows * row_bytes;
    if !bytes.len().is_multiple_of(group_bytes) {
        inference_tensor::bail!(
            "repacked type {id} holds {} bytes, not whole groups of {} rows of {cols} elements",
            bytes.len(),
            ty.rows
        );
    }
    let mut out = vec![0u8; bytes.len()];
    for (src, dst) in bytes
        .chunks_exact(group_bytes)
        .zip(out.chunks_exact_mut(group_bytes))
    {
        let (metas, blocks) = src.split_at(ty.rows * row_meta);
        for (k, row) in dst.chunks_exact_mut(row_bytes).enumerate() {
            row[..row_meta].copy_from_slice(&metas[k * row_meta..(k + 1) * row_meta]);
        }
        // repacked block ib holds block ib of every row in the group
        for (ib, packed) in blocks.chunks_exact(ty.rows * block_bytes).enumerate() {
            for (k, row) in dst.chunks_exact_mut(row_bytes).enumerate() {
                let at = row_meta + ib * block_bytes;
                unpack_block(packed, k, &mut row[at..at + block_bytes]);
            }
        }
    }
    Ok(out)
}

// Writes row `k`'s base block (zeroed on entry) from one interleaved block
type UnpackBlock = fn(&[u8], usize, &mut [u8]);

const Q4_0_BYTES: usize = HALF + QK / 2;
const Q4_0_R8_QS: usize = 8 * HALF;

fn q4_0(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    for l in 0..4 {
        for i in 0..4 {
            dst[HALF + 4 * l + i] = src[Q4_0_R8_QS + 32 * l + 4 * k + i];
        }
    }
}

const Q8_0_BYTES: usize = HALF + QK;
const Q8_0_R8_QS: usize = 8 * HALF;
const Q8_0_R8_UPPER: usize = 128;

fn q8_0(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    let qs = &src[Q8_0_R8_QS..];
    for l in 0..4 {
        for i in 0..4 {
            dst[HALF + 4 * l + i] = qs[32 * l + 4 * k + i];
            dst[HALF + QK / 2 + 4 * l + i] = qs[Q8_0_R8_UPPER + 32 * l + 4 * k + i];
        }
    }
}

const Q5_0_BYTES: usize = HALF + 4 + QK / 2;
const Q5_0_QS: usize = HALF + 4;
const Q5_0_R4_QH: usize = 4 * HALF;
const Q5_0_R4_QS: usize = Q5_0_R4_QH + QK / 2;

fn q5_0(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    let mut q = [0u8; QK];
    for l in 0..4 {
        let l1 = 4 * (l / 2) + 16 * (l % 2);
        let l2 = l1 + 8;
        for i in 0..4 {
            let qs = src[Q5_0_R4_QS + 4 * k + i + 16 * l];
            let qh = src[Q5_0_R4_QH + 4 * k + i];
            q[i + l1] = (qs & 0xf) | (((qh >> l) & 1) << 4);
            q[i + l2] = (qs >> 4) | (((qh >> (l + 4)) & 1) << 4);
        }
    }
    let mut qh = 0u32;
    for j in 0..QK / 2 {
        dst[Q5_0_QS + j] = (q[j] & 0xf) | ((q[j + QK / 2] & 0xf) << 4);
        qh |= (u32::from(q[j] >> 4) << j) | (u32::from(q[j + QK / 2] >> 4) << (j + QK / 2));
    }
    dst[HALF..Q5_0_QS].copy_from_slice(&qh.to_le_bytes());
}

// Q2_K_R4 / Q3_K_R4 pack sub-block ib's 2-bit quants for row k into bytes 32*ib+4*k+i (+16 for the second half)
fn crumbs_r4(qs: &[u8], k: usize, q: &mut [u8; QK_K]) {
    for ib in 0..SUB_BLOCKS_K {
        for i in 0..4 {
            let lo = qs[32 * ib + 4 * k + i];
            let hi = qs[32 * ib + 4 * k + i + 16];
            for s in 0..4 {
                q[32 * ib + i + 4 * s] = (lo >> (2 * s)) & 3;
                q[32 * ib + i + 16 + 4 * s] = (hi >> (2 * s)) & 3;
            }
        }
    }
}

// Q2_K / Q3_K hold each 128-element half as 32 bytes of four 2-bit planes
fn encode_crumbs(q: &[u8; QK_K], qs: &mut [u8]) {
    for h in 0..2 {
        for s in 0..4 {
            for l in 0..32 {
                qs[32 * h + l] |= (q[128 * h + 32 * s + l] & 3) << (2 * s);
            }
        }
    }
}

const Q2_K_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 * HALF;
const Q2_K_QS: usize = QK_K / 16;
const Q2_K_D: usize = Q2_K_QS + QK_K / 4;
const Q2_K_R4_SCALES: usize = 8 * HALF;
const Q2_K_R4_QS: usize = Q2_K_R4_SCALES + QK_K / 4;

fn q2_k(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[Q2_K_D..Q2_K_D + HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    dst[Q2_K_D + HALF..].copy_from_slice(&src[HALF * (k + 4)..HALF * (k + 5)]);
    for ib in 0..QK_K / 16 {
        dst[ib] = src[Q2_K_R4_SCALES + 4 * ib + k];
    }
    let mut q = [0u8; QK_K];
    crumbs_r4(&src[Q2_K_R4_QS..], k, &mut q);
    encode_crumbs(&q, &mut dst[Q2_K_QS..Q2_K_D]);
}

const Q3_K_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + HALF;
const Q3_K_QS: usize = QK_K / 8;
const Q3_K_SCALES: usize = Q3_K_QS + QK_K / 4;
const Q3_K_D: usize = Q3_K_SCALES + 12;
const Q3_K_R4_SCALES_H: usize = 4 * HALF;
const Q3_K_R4_SCALES_L: usize = Q3_K_R4_SCALES_H + QK_K / 16;
const Q3_K_R4_QH: usize = Q3_K_R4_SCALES_L + QK_K / 8;
const Q3_K_R4_QS: usize = Q3_K_R4_QH + QK_K / 2;

fn q3_k(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[Q3_K_D..].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    let mut scales = [0u8; 16];
    for ib in 0..SUB_BLOCKS_K {
        for (half, is) in [8 * ib + k, 8 * ib + k + 4].into_iter().enumerate() {
            let lo = nibble_at(&src[Q3_K_R4_SCALES_L..Q3_K_R4_QH], is);
            let hi = crumb_at(&src[Q3_K_R4_SCALES_H..Q3_K_R4_SCALES_L], is);
            scales[2 * ib + half] = lo | (hi << 4);
        }
    }
    // ggml packs sixteen 6-bit scales as two low-nibble planes and one plane of 2-bit high parts
    for b in 0..4 {
        let s = &mut dst[Q3_K_SCALES..Q3_K_D];
        s[b] = (scales[b] & 0xf) | ((scales[8 + b] & 0xf) << 4);
        s[4 + b] = (scales[4 + b] & 0xf) | ((scales[12 + b] & 0xf) << 4);
        s[8 + b] = (scales[b] >> 4)
            | ((scales[4 + b] >> 4) << 2)
            | ((scales[8 + b] >> 4) << 4)
            | ((scales[12 + b] >> 4) << 6);
    }
    let mut q = [0u8; QK_K];
    crumbs_r4(&src[Q3_K_R4_QS..], k, &mut q);
    high_bits_r4(&src[Q3_K_R4_QH..], k, CRUMB_HIGH_BITS, CRUMB_BITS, &mut q);
    encode_crumbs(&q, &mut dst[Q3_K_QS..Q3_K_SCALES]);
    encode_bit_planes(&q, CRUMB_BITS, &mut dst[..Q3_K_QS]);
}

const CRUMB_BITS: u32 = 2;
const NIBBLE_BITS: u32 = 4;
// Element position within a sub-block of each bit of a repacked high-bit byte, per layout
const CRUMB_HIGH_BITS: [usize; 8] = [0, 4, 8, 12, 16, 20, 24, 28];
const Q5_K_HIGH_BITS: [usize; 8] = [0, 8, 4, 12, 16, 24, 20, 28];
const IQ5_K_HIGH_BITS: [usize; 8] = [0, 8, 16, 24, 4, 12, 20, 28];

// _R4 types with one high bit per element keep a sub-block's for row k in byte 16*ib+4*k+i
fn high_bits_r4(qh: &[u8], k: usize, order: [usize; 8], shift: u32, q: &mut [u8; QK_K]) {
    for ib in 0..SUB_BLOCKS_K {
        for i in 0..4 {
            let h = qh[16 * ib + 4 * k + i];
            for (b, at) in order.into_iter().enumerate() {
                q[32 * ib + i + at] |= ((h >> b) & 1) << shift;
            }
        }
    }
}

// Q3_K hmask, Q5_K / IQ5_K qh and IQ3_K qh: bit ib of byte j is the high bit of element 32*ib+j
fn encode_bit_planes(q: &[u8; QK_K], bit: u32, out: &mut [u8]) {
    for ib in 0..SUB_BLOCKS_K {
        for j in 0..QK {
            out[j] |= ((q[QK * ib + j] >> bit) & 1) << ib;
        }
    }
}

// Byte offset within a 64-byte run and the sub-block positions of its low and high nibbles
const NIBBLE_LANES: [(usize, usize, usize); 4] =
    [(0, 0, 8), (16, 16, 24), (32, 4, 12), (48, 20, 28)];

// Q4_K_R4 / Q5_K_R4 / Q6_K_R4 pack sub-block ib's low nibbles for row k into bytes 64*ib+4*k+i+lane
fn nibbles_r4(qs: &[u8], k: usize, q: &mut [u8; QK_K]) {
    for ib in 0..SUB_BLOCKS_K {
        for i in 0..4 {
            for (lane, lo, hi) in NIBBLE_LANES {
                let b = qs[64 * ib + 4 * k + i + lane];
                q[32 * ib + i + lo] = b & 0xf;
                q[32 * ib + i + hi] = b >> 4;
            }
        }
    }
}

const SCALES_K_BYTES: usize = 12;
const Q4_K_BYTES: usize = 2 * HALF + SCALES_K_BYTES + QK_K / 2;
const Q4_K_SCALES: usize = 2 * HALF;
const Q4_K_QS: usize = Q4_K_SCALES + SCALES_K_BYTES;
const Q4_K_R4_SCALES_H: usize = 8 * HALF;
const Q4_K_R4_SCALES_L: usize = Q4_K_R4_SCALES_H + QK_K / 16;
const Q4_K_R4_QS: usize = Q4_K_R4_SCALES_L + QK_K / 8;

// Q4_K_R4 / Q5_K_R4 head: d and dmin per row, then each sub-block's 6-bit scale and min split into nibble planes
fn k4_head(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    dst[HALF..2 * HALF].copy_from_slice(&src[HALF * (k + 4)..HALF * (k + 5)]);
    let (mut d, mut m) = ([0u8; SUB_BLOCKS_K], [0u8; SUB_BLOCKS_K]);
    for ib in 0..SUB_BLOCKS_K {
        let is = 4 * ib + k;
        let lo = src[Q4_K_R4_SCALES_L + is];
        let hi = nibble_at(&src[Q4_K_R4_SCALES_H..Q4_K_R4_SCALES_L], is);
        d[ib] = (lo & 0xf) | ((hi & 3) << 4);
        m[ib] = (lo >> 4) | ((hi >> 2) << 4);
    }
    // inverse of ggml's get_scale_min_k4
    let s = &mut dst[Q4_K_SCALES..Q4_K_QS];
    for j in 0..4 {
        s[j] = d[j] | ((d[j + 4] >> 4) << 6);
        s[j + 4] = m[j] | ((m[j + 4] >> 4) << 6);
        s[j + 8] = (d[j + 4] & 0xf) | ((m[j + 4] & 0xf) << 4);
    }
}

// Q4_K / Q5_K store each 64-element run as 32 bytes: elements j in the low nibble, j+32 in the high
fn encode_nibbles_k(q: &[u8; QK_K], qs: &mut [u8]) {
    for ib64 in 0..QK_K / 64 {
        for j in 0..32 {
            qs[32 * ib64 + j] = (q[64 * ib64 + j] & 0xf) | ((q[64 * ib64 + j + 32] & 0xf) << 4);
        }
    }
}

fn q4_k(src: &[u8], k: usize, dst: &mut [u8]) {
    k4_head(src, k, dst);
    let mut q = [0u8; QK_K];
    nibbles_r4(&src[Q4_K_R4_QS..], k, &mut q);
    encode_nibbles_k(&q, &mut dst[Q4_K_QS..]);
}

const Q5_K_BYTES: usize = Q4_K_BYTES + QK_K / 8;
const Q5_K_QH: usize = Q4_K_QS;
const Q5_K_QS: usize = Q5_K_QH + QK_K / 8;
const Q5_K_R4_QH: usize = Q4_K_R4_QS;
const Q5_K_R4_QS: usize = Q5_K_R4_QH + QK_K / 2;

fn q5_k(src: &[u8], k: usize, dst: &mut [u8]) {
    k4_head(src, k, dst);
    let mut q = [0u8; QK_K];
    nibbles_r4(&src[Q5_K_R4_QS..], k, &mut q);
    high_bits_r4(&src[Q5_K_R4_QH..], k, Q5_K_HIGH_BITS, NIBBLE_BITS, &mut q);
    encode_nibbles_k(&q, &mut dst[Q5_K_QS..]);
    encode_bit_planes(&q, NIBBLE_BITS, &mut dst[Q5_K_QH..Q5_K_QS]);
}

const Q6_K_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + HALF;
const Q6_K_QH: usize = QK_K / 2;
const Q6_K_SCALES: usize = Q6_K_QH + QK_K / 4;
const Q6_K_D: usize = Q6_K_SCALES + QK_K / 16;
const Q6_K_R4_SCALES: usize = 4 * HALF;
const Q6_K_R4_QH: usize = Q6_K_R4_SCALES + QK_K / 4;
const Q6_K_R4_QL: usize = Q6_K_R4_QH + QK_K;
// Sub-block positions of the four 2-bit high parts in each of a Q6_K_R4 row's two qh bytes per sub-block
const Q6_K_R4_HIGH_PAIRS: [[usize; 4]; 2] = [[0, 8, 4, 12], [16, 24, 20, 28]];

fn q6_k(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[Q6_K_D..].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    for ib in 0..SUB_BLOCKS_K {
        dst[Q6_K_SCALES + 2 * ib] = src[Q6_K_R4_SCALES + 8 * ib + k];
        dst[Q6_K_SCALES + 2 * ib + 1] = src[Q6_K_R4_SCALES + 8 * ib + k + 4];
    }
    let mut q = [0u8; QK_K];
    nibbles_r4(&src[Q6_K_R4_QL..], k, &mut q);
    for ib in 0..SUB_BLOCKS_K {
        for i in 0..4 {
            for (half, positions) in Q6_K_R4_HIGH_PAIRS.iter().enumerate() {
                let qh = src[Q6_K_R4_QH + 32 * ib + 4 * k + i + 16 * half];
                for (b, at) in positions.iter().enumerate() {
                    q[32 * ib + i + at] |= ((qh >> (2 * b)) & 3) << 4;
                }
            }
        }
    }
    // each 128-element half: ql bytes l and l+32 carry elements l, l+32 (low nibbles) and l+64, l+96 (high)
    for h in 0..2 {
        let q = &q[128 * h..128 * (h + 1)];
        for l in 0..32 {
            dst[64 * h + l] = (q[l] & 0xf) | ((q[l + 64] & 0xf) << 4);
            dst[64 * h + l + 32] = (q[l + 32] & 0xf) | ((q[l + 96] & 0xf) << 4);
            dst[Q6_K_QH + 32 * h + l] = (q[l] >> 4)
                | ((q[l + 32] >> 4) << 2)
                | ((q[l + 64] >> 4) << 4)
                | ((q[l + 96] >> 4) << 6);
        }
    }
}

// A row's nibble-scale or 2-bit field `i` in the repacked scale planes (field i in byte i%len, slot i/len)
fn nibble_at(plane: &[u8], i: usize) -> u8 {
    (plane[i % plane.len()] >> (4 * (i / plane.len()))) & 0xf
}

fn crumb_at(plane: &[u8], i: usize) -> u8 {
    (plane[i % plane.len()] >> (2 * (i / plane.len()))) & 3
}

fn bit_at(plane: &[u8], i: usize) -> u8 {
    (plane[i % plane.len()] >> (i / plane.len())) & 1
}

// IQ4_NL_R4 / IQ4_KS_R4 / IQ4_K_R4: 16 bytes of a row's nibbles, spread over 64 bytes at 4*k+i (+16, +32, +48)
fn nl_nibbles_r4(r: &[u8], k: usize, qs: &mut [u8]) {
    for i in 0..4 {
        let (r0, r16) = (r[4 * k + i], r[4 * k + i + 16]);
        let (r32, r48) = (r[4 * k + i + 32], r[4 * k + i + 48]);
        qs[i] = (r0 & 0xf) | (r16 << 4);
        qs[i + 8] = (r0 >> 4) | (r16 & 0xf0);
        qs[i + 4] = (r32 & 0xf) | (r48 << 4);
        qs[i + 12] = (r32 >> 4) | (r48 & 0xf0);
    }
}

// IQ*_K_R4: row k's 2-bit `extra` per sub-block is split into bit ib of bytes k and k+4
fn extra_r4(extra: &[u8], k: usize, dst: &mut [u8]) {
    let mut out = 0u16;
    for ib in 0..SUB_BLOCKS_K {
        out |= u16::from((extra[k] >> ib) & 1) << (2 * ib);
        out |= u16::from((extra[k + 4] >> ib) & 1) << (2 * ib + 1);
    }
    dst.copy_from_slice(&out.to_le_bytes());
}

// ik's `scrambled_sign` permutes 7-bit sign indices; repacked IQ2_XXS / IQ2_XS / IQ3_XXS store them permuted
const SCRAMBLED_SIGNS: [u8; 128] = [
    0x00, 0x7f, 0x7e, 0x01, 0x7c, 0x03, 0x02, 0x7d, 0x78, 0x07, 0x06, 0x79, 0x04, 0x7b, 0x7a, 0x05,
    0x70, 0x0f, 0x0e, 0x71, 0x0c, 0x73, 0x72, 0x0d, 0x08, 0x77, 0x76, 0x09, 0x74, 0x0b, 0x0a, 0x75,
    0x60, 0x1f, 0x1e, 0x61, 0x1c, 0x63, 0x62, 0x1d, 0x18, 0x67, 0x66, 0x19, 0x64, 0x1b, 0x1a, 0x65,
    0x10, 0x6f, 0x6e, 0x11, 0x6c, 0x13, 0x12, 0x6d, 0x68, 0x17, 0x16, 0x69, 0x14, 0x6b, 0x6a, 0x15,
    0x40, 0x3f, 0x3e, 0x41, 0x3c, 0x43, 0x42, 0x3d, 0x38, 0x47, 0x46, 0x39, 0x44, 0x3b, 0x3a, 0x45,
    0x30, 0x4f, 0x4e, 0x31, 0x4c, 0x33, 0x32, 0x4d, 0x48, 0x37, 0x36, 0x49, 0x34, 0x4b, 0x4a, 0x35,
    0x20, 0x5f, 0x5e, 0x21, 0x5c, 0x23, 0x22, 0x5d, 0x58, 0x27, 0x26, 0x59, 0x24, 0x5b, 0x5a, 0x25,
    0x50, 0x2f, 0x2e, 0x51, 0x2c, 0x53, 0x52, 0x2d, 0x28, 0x57, 0x56, 0x29, 0x54, 0x2b, 0x2a, 0x55,
];

const UNSCRAMBLED_SIGNS: [u8; 128] = {
    let mut inverse = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        inverse[SCRAMBLED_SIGNS[i] as usize] = i as u8;
        i += 1;
    }
    inverse
};
const SIGN_BITS: u32 = 7;
const SAS_SCALE_SHIFT: u32 = 28;

// IQ2_XXS / IQ3_XXS: a sub-block's u32 of four 7-bit sign indices and a 4-bit scale, from its repacked bytes
fn unscramble_sas(sas: &[u8]) -> [u8; 4] {
    let mut word = 0u32;
    for (m, s) in sas.iter().enumerate() {
        word |= u32::from(UNSCRAMBLED_SIGNS[usize::from(s >> 1)]) << (SIGN_BITS * m as u32);
        word |= u32::from(s & 1) << (SAS_SCALE_SHIFT + m as u32);
    }
    word.to_le_bytes()
}

const IQ2_XXS_BYTES: usize = HALF + QK_K / 4;
const IQ2_XXS_R4_SAS: usize = 4 * HALF;
const IQ2_XXS_R4_QS: usize = IQ2_XXS_R4_SAS + QK_K / 2;

fn iq2_xxs(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    for ib in 0..SUB_BLOCKS_K {
        let at = HALF + 8 * ib;
        for i in 0..4 {
            dst[at + i] = src[IQ2_XXS_R4_QS + 16 * ib + 4 * k + i];
        }
        let sas = IQ2_XXS_R4_SAS + 16 * ib + 4 * k;
        dst[at + 4..at + 8].copy_from_slice(&unscramble_sas(&src[sas..sas + 4]));
    }
}

const IQ3_XXS_BYTES: usize = HALF + 3 * QK_K / 8;
const IQ3_XXS_SAS: usize = HALF + QK_K / 4;
const IQ3_XXS_R4_SAS: usize = 4 * HALF;
const IQ3_XXS_R4_QS: usize = IQ3_XXS_R4_SAS + QK_K / 2;

fn iq3_xxs(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    for ib in 0..SUB_BLOCKS_K {
        for i in 0..8 {
            dst[HALF + 8 * ib + i] = src[IQ3_XXS_R4_QS + 32 * ib + 8 * k + i];
        }
        let sas = IQ3_XXS_R4_SAS + 16 * ib + 4 * k;
        let at = IQ3_XXS_SAS + 4 * ib;
        dst[at..at + 4].copy_from_slice(&unscramble_sas(&src[sas..sas + 4]));
    }
}

const IQ2_XS_BYTES: usize = HALF + QK_K / 4 + QK_K / 32;
const IQ2_XS_SCALES: usize = HALF + QK_K / 4;
const IQ2_XS_R4_QS: usize = 4 * HALF;
const IQ2_XS_R4_SCALES: usize = IQ2_XS_R4_QS + QK_K;
const IQ2_XS_GRID_MASK: u16 = 511;
const IQ2_XS_SIGN_SHIFT: u32 = 9;

fn iq2_xs(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    for ib in 0..SUB_BLOCKS_K {
        for i in 0..4 {
            let at = IQ2_XS_R4_QS + 2 * (16 * ib + 4 * k + i);
            let v = u16::from_le_bytes([src[at], src[at + 1]]);
            let sign = u16::from(UNSCRAMBLED_SIGNS[usize::from(v >> IQ2_XS_SIGN_SHIFT)]);
            let v = (v & IQ2_XS_GRID_MASK) | (sign << IQ2_XS_SIGN_SHIFT);
            let at = HALF + 2 * (4 * ib + i);
            dst[at..at + 2].copy_from_slice(&v.to_le_bytes());
        }
        dst[IQ2_XS_SCALES + ib] = src[IQ2_XS_R4_SCALES + 4 * ib + k];
    }
}

const IQ2_S_BYTES: usize = HALF + QK_K / 4 + 2 * (QK_K / 32);
const IQ2_S_SIGNS: usize = HALF + QK_K / 8;
const IQ2_S_QH: usize = HALF + QK_K / 4;
const IQ2_S_SCALES: usize = IQ2_S_QH + QK_K / 32;
const IQ2_S_R4_QS: usize = 4 * HALF;
const IQ2_S_R4_QH: usize = IQ2_S_R4_QS + QK_K / 2;
const IQ2_S_R4_SIGNS: usize = IQ2_S_R4_QH + QK_K / 8;
const IQ2_S_R4_SCALES: usize = IQ2_S_R4_SIGNS + QK_K / 2;

fn iq2_s(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    for ib in 0..SUB_BLOCKS_K {
        dst[IQ2_S_SCALES + ib] = src[IQ2_S_R4_SCALES + 4 * ib + k];
        dst[IQ2_S_QH + ib] = src[IQ2_S_R4_QH + 4 * ib + k];
        for i in 0..4 {
            dst[HALF + 4 * ib + i] = src[IQ2_S_R4_QS + 16 * ib + 4 * k + i];
            dst[IQ2_S_SIGNS + 4 * ib + i] = src[IQ2_S_R4_SIGNS + 16 * ib + 4 * k + i];
        }
    }
}

const IQ3_S_SCALE_BYTES: usize = QK_K / 64;
const IQ3_S_BYTES: usize = HALF + QK_K / 4 + QK_K / 32 + QK_K / 8 + IQ3_S_SCALE_BYTES;
const IQ3_S_QH: usize = HALF + QK_K / 4;
const IQ3_S_SIGNS: usize = IQ3_S_QH + QK_K / 32;
const IQ3_S_SCALES: usize = IQ3_S_SIGNS + QK_K / 8;
const IQ3_S_R4_QS: usize = 4 * HALF;
const IQ3_S_R4_QH: usize = IQ3_S_R4_QS + QK_K;
const IQ3_S_R4_SIGNS: usize = IQ3_S_R4_QH + QK_K / 8;
const IQ3_S_R4_SCALES: usize = IQ3_S_R4_SIGNS + QK_K / 2;

fn iq3_s(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    let scales = &src[IQ3_S_R4_SCALES..IQ3_S_R4_SCALES + 4 * IQ3_S_SCALE_BYTES];
    for ib in 0..IQ3_S_SCALE_BYTES {
        let j = 8 * ib + k;
        dst[IQ3_S_SCALES + ib] = nibble_at(scales, j) | (nibble_at(scales, j + 4) << 4);
    }
    for ib in 0..SUB_BLOCKS_K {
        dst[IQ3_S_QH + ib] = src[IQ3_S_R4_QH + 4 * ib + k];
        for i in 0..4 {
            dst[HALF + 8 * ib + i] = src[IQ3_S_R4_QS + 32 * ib + k + 8 * i];
            dst[HALF + 8 * ib + i + 4] = src[IQ3_S_R4_QS + 32 * ib + k + 8 * i + 4];
            // bits 2m and 2m+1 are bits i and 4+i of the row's sign byte m
            let s = src[IQ3_S_R4_SIGNS + 16 * ib + 4 * k + i];
            for m in 0..4 {
                dst[IQ3_S_SIGNS + 4 * ib + m] |=
                    (((s >> (2 * m)) & 1) << i) | (((s >> (2 * m + 1)) & 1) << (4 + i));
            }
        }
    }
}

const IQ4_NL_BYTES: usize = HALF + QK / 2;
const IQ4_NL_R4_QS: usize = 4 * HALF;

fn iq4_nl(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    nl_nibbles_r4(&src[IQ4_NL_R4_QS..], k, &mut dst[HALF..]);
}

const MXFP4_BYTES: usize = 1 + QK / 2;
const MXFP4_R8_QS: usize = 8;

fn mxfp4(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[0] = src[k];
    for l in 0..4 {
        for i in 0..4 {
            dst[1 + 4 * l + i] = src[MXFP4_R8_QS + 32 * l + 4 * k + i];
        }
    }
}

const IQ4_XS_BYTES: usize = 2 * HALF + QK_K / 64 + QK_K / 2;
const IQ4_XS_SCALES_H: usize = HALF;
const IQ4_XS_SCALES_L: usize = 2 * HALF;
const IQ4_XS_QS: usize = IQ4_XS_SCALES_L + QK_K / 64;
const IQ4_XS_R8_SCALES_H: usize = 8 * HALF;
const IQ4_XS_R8_SCALES_L: usize = IQ4_XS_R8_SCALES_H + QK_K / 16;
const IQ4_XS_R8_QS: usize = IQ4_XS_R8_SCALES_L + QK_K / 8;

fn iq4_xs(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    let scales_h = &src[IQ4_XS_R8_SCALES_H..IQ4_XS_R8_SCALES_L];
    let scales_l = &src[IQ4_XS_R8_SCALES_L..IQ4_XS_R8_QS];
    let mut sh = 0u16;
    for ib in 0..SUB_BLOCKS_K {
        let j = 8 * ib + k;
        dst[IQ4_XS_SCALES_L + ib / 2] |= nibble_at(scales_l, j) << (4 * (ib % 2));
        sh |= u16::from(crumb_at(scales_h, j)) << (2 * ib);
        let r = &src[IQ4_XS_R8_QS + 128 * ib..];
        let qs = &mut dst[IQ4_XS_QS + 16 * ib..];
        for i in 0..4 {
            let (r0, r32) = (r[4 * k + i], r[4 * k + i + 32]);
            let (r64, r96) = (r[4 * k + i + 64], r[4 * k + i + 96]);
            qs[i] = (r0 & 0xf) | (r64 << 4);
            qs[i + 4] = (r0 >> 4) | (r64 & 0xf0);
            qs[i + 8] = (r32 & 0xf) | (r96 << 4);
            qs[i + 12] = (r32 >> 4) | (r96 & 0xf0);
        }
    }
    dst[IQ4_XS_SCALES_H..IQ4_XS_SCALES_L].copy_from_slice(&sh.to_le_bytes());
}

const IQK_EXTRA: usize = HALF;
const IQK_R4_EXTRA: usize = 4 * HALF;
const IQK_R4_EXTRA_BYTES: usize = 8;
const IQK_R4_SCALES: usize = IQK_R4_EXTRA + IQK_R4_EXTRA_BYTES;

const IQ2_K_BYTES: usize = 2 * HALF + QK_K / 32 + QK_K / 4;
const IQ2_K_SCALES: usize = 2 * HALF;
const IQ2_K_QS: usize = IQ2_K_SCALES + QK_K / 32;
const IQ2_K_R4_QS: usize = IQK_R4_SCALES + QK_K / 8;

// IQ2_K_R4 / IQ3_K_R4 / IQ4_K_R4 / IQ5_K_R4 head: d per row, then the 2-bit extras
fn iqk_head(src: &[u8], k: usize, dst: &mut [u8]) {
    dst[..HALF].copy_from_slice(&src[HALF * k..HALF * (k + 1)]);
    extra_r4(
        &src[IQK_R4_EXTRA..IQK_R4_SCALES],
        k,
        &mut dst[IQK_EXTRA..IQK_EXTRA + HALF],
    );
}

fn iq2_k(src: &[u8], k: usize, dst: &mut [u8]) {
    iqk_head(src, k, dst);
    let scales = &src[IQK_R4_SCALES..IQ2_K_R4_QS];
    for ib in 0..SUB_BLOCKS_K {
        let j = 8 * ib + k;
        dst[IQ2_K_SCALES + ib] = nibble_at(scales, j) | (nibble_at(scales, j + 4) << 4);
    }
    let mut q = [0u8; QK_K];
    crumbs_r4(&src[IQ2_K_R4_QS..], k, &mut q);
    encode_crumbs(&q, &mut dst[IQ2_K_QS..]);
}

const IQ3_K_BYTES: usize = 3 * HALF + QK_K / 32 + QK_K / 4 + QK_K / 8;
const IQ3_K_SCALES_H: usize = 2 * HALF;
const IQ3_K_SCALES_L: usize = 3 * HALF;
const IQ3_K_QS: usize = IQ3_K_SCALES_L + QK_K / 32;
const IQ3_K_QH: usize = IQ3_K_QS + QK_K / 4;
const IQ3_K_R4_SCALES_L: usize = IQK_R4_SCALES + QK_K / 32;
const IQ3_K_R4_QS: usize = IQ3_K_R4_SCALES_L + QK_K / 8;
const IQ3_K_R4_QH: usize = IQ3_K_R4_QS + QK_K;

fn iq3_k(src: &[u8], k: usize, dst: &mut [u8]) {
    iqk_head(src, k, dst);
    let scales_h = &src[IQK_R4_SCALES..IQ3_K_R4_SCALES_L];
    let scales_l = &src[IQ3_K_R4_SCALES_L..IQ3_K_R4_QS];
    let mut sh = 0u16;
    for ib in 0..SUB_BLOCKS_K {
        let j = 8 * ib + k;
        dst[IQ3_K_SCALES_L + ib] = nibble_at(scales_l, j) | (nibble_at(scales_l, j + 4) << 4);
        sh |= u16::from(bit_at(scales_h, j)) << (2 * ib);
        sh |= u16::from(bit_at(scales_h, j + 4)) << (2 * ib + 1);
    }
    dst[IQ3_K_SCALES_H..IQ3_K_SCALES_L].copy_from_slice(&sh.to_le_bytes());
    let mut q = [0u8; QK_K];
    crumbs_r4(&src[IQ3_K_R4_QS..], k, &mut q);
    high_bits_r4(&src[IQ3_K_R4_QH..], k, CRUMB_HIGH_BITS, CRUMB_BITS, &mut q);
    encode_crumbs(&q, &mut dst[IQ3_K_QS..IQ3_K_QH]);
    encode_bit_planes(&q, CRUMB_BITS, &mut dst[IQ3_K_QH..]);
}

const IQ4_K_BYTES: usize = 2 * HALF + QK_K / 64 + QK_K / 32 + QK_K / 2;
const IQ4_K_SCALES_H: usize = 2 * HALF;
const IQ4_K_SCALES_L: usize = IQ4_K_SCALES_H + QK_K / 64;
const IQ4_K_QS: usize = IQ4_K_SCALES_L + QK_K / 32;
const IQ4_K_R4_SCALES_L: usize = IQK_R4_SCALES + QK_K / 16;
const IQ4_K_R4_QS: usize = IQ4_K_R4_SCALES_L + QK_K / 8;

// IQ4_K_R4 / IQ5_K_R4 scales: each sub-block's two 6-bit scales as nibble and 2-bit planes
fn iq45_k_head(src: &[u8], k: usize, dst: &mut [u8]) {
    iqk_head(src, k, dst);
    let scales_h = &src[IQK_R4_SCALES..IQ4_K_R4_SCALES_L];
    let scales_l = &src[IQ4_K_R4_SCALES_L..IQ4_K_R4_QS];
    for ib in 0..SUB_BLOCKS_K {
        let j = 8 * ib + k;
        dst[IQ4_K_SCALES_L + ib] = nibble_at(scales_l, j) | (nibble_at(scales_l, j + 4) << 4);
        let sh = crumb_at(scales_h, j) | (crumb_at(scales_h, j + 4) << 2);
        dst[IQ4_K_SCALES_H + ib / 2] |= sh << (4 * (ib % 2));
    }
}

fn iq4_k(src: &[u8], k: usize, dst: &mut [u8]) {
    iq45_k_head(src, k, dst);
    for ib in 0..SUB_BLOCKS_K {
        nl_nibbles_r4(
            &src[IQ4_K_R4_QS + 64 * ib..],
            k,
            &mut dst[IQ4_K_QS + 16 * ib..],
        );
    }
}

const IQ5_K_BYTES: usize = IQ4_K_BYTES + QK_K / 8;
const IQ5_K_QH: usize = IQ4_K_BYTES;
const IQ5_K_R4_QH: usize = IQ4_K_R4_QS + 2 * QK_K;

fn iq5_k(src: &[u8], k: usize, dst: &mut [u8]) {
    iq45_k_head(src, k, dst);
    let mut q = [0u8; QK_K];
    nibbles_r4(&src[IQ4_K_R4_QS..], k, &mut q);
    high_bits_r4(&src[IQ5_K_R4_QH..], k, IQ5_K_HIGH_BITS, NIBBLE_BITS, &mut q);
    encode_nibbles_k(&q, &mut dst[IQ4_K_QS..IQ5_K_QH]);
    encode_bit_planes(&q, NIBBLE_BITS, &mut dst[IQ5_K_QH..]);
}

const IQ4_KS_BYTES: usize = QK_K / 32 + QK_K / 2;
const IQ4_KS_QS: usize = QK_K / 32;
const IQ4_KS_R4_QS: usize = QK_K / 8;

fn iq4_ks(src: &[u8], k: usize, dst: &mut [u8]) {
    for ib in 0..SUB_BLOCKS_K {
        dst[ib] = src[4 * ib + k];
        nl_nibbles_r4(
            &src[IQ4_KS_R4_QS + 64 * ib..],
            k,
            &mut dst[IQ4_KS_QS + 16 * ib..],
        );
    }
}

const IQ5_KS_BYTES: usize = IQ4_KS_BYTES + QK_K / 8;
const IQ5_KS_QH: usize = IQ4_KS_BYTES;
const IQ5_KS_R4_QH: usize = IQ4_KS_R4_QS + 2 * QK_K;

fn iq5_ks(src: &[u8], k: usize, dst: &mut [u8]) {
    for ib in 0..SUB_BLOCKS_K {
        dst[ib] = src[4 * ib + k];
    }
    let mut q = [0u8; QK_K];
    nibbles_r4(&src[IQ4_KS_R4_QS..], k, &mut q);
    high_bits_r4(
        &src[IQ5_KS_R4_QH..],
        k,
        IQ5_K_HIGH_BITS,
        NIBBLE_BITS,
        &mut q,
    );
    encode_nibbles_k(&q, &mut dst[IQ4_KS_QS..IQ5_KS_QH]);
    encode_bit_planes(&q, NIBBLE_BITS, &mut dst[IQ5_KS_QH..]);
}

#[cfg(test)]
pub(crate) mod tests {
    use base64::{Engine, engine::general_purpose::STANDARD};

    use super::*;

    // Rows of one tensor quantized by ik_llama.cpp, plain and repacked (tests/fixtures/gguf_ik_repack/make_goldens.py)
    const GOLDENS: &str = include_str!("../../tests/fixtures/gguf_ik_repack/goldens.json");

    #[derive(serde::Deserialize)]
    pub(crate) struct Golden {
        pub ty: String,
        pub id: u32,
        pub base: u32,
        pub rows_per_group: usize,
        pub cols: usize,
        plain: String,
        repacked: String,
    }

    impl Golden {
        pub fn plain(&self) -> Vec<u8> {
            STANDARD.decode(&self.plain).expect("base64")
        }

        pub fn repacked(&self) -> Vec<u8> {
            STANDARD.decode(&self.repacked).expect("base64")
        }
    }

    pub(crate) const GOLDEN_ROWS: usize = 8;

    pub(crate) fn goldens() -> Vec<Golden> {
        serde_json::from_str(GOLDENS).expect("valid goldens")
    }

    #[test]
    fn repack_goldens_unpack_to_plain_rows() -> Result<()> {
        let goldens = goldens();
        assert_eq!(goldens.len(), REPACKED.len());
        for g in goldens {
            let ty = repacked(g.id).expect("known repacked id");
            assert_eq!((ty.base, ty.rows), (g.base, g.rows_per_group), "{}", g.ty);
            let (plain, packed) = (g.plain(), g.repacked());
            assert_ne!(plain, packed, "{}", g.ty);
            assert!(
                unpack(g.id, g.cols, &packed)? == plain,
                "{} does not unpack to the plain rows",
                g.ty
            );
        }
        Ok(())
    }

    #[test]
    fn repack_rejects_partial_groups_and_blocks() {
        for g in goldens() {
            let packed = g.repacked();
            let row_bytes = packed.len() / GOLDEN_ROWS;
            assert!(
                unpack(g.id, g.cols, &packed[..packed.len() - row_bytes]).is_err(),
                "{}",
                g.ty
            );
            assert!(unpack(g.id, g.cols + 16, &packed).is_err(), "{}", g.ty);
            assert!(unpack(g.id, 0, &packed).is_err(), "{}", g.ty);
        }
        assert!(unpack(Q4_K, QK_K, &[0; Q4_K_BYTES * 4]).is_err());
    }
}
