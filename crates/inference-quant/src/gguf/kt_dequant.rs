//! ik_llama.cpp's trellis (`IQ*_KT`) dequantization of one row, ported from its CUDA `dequantize_block_iq*_kt`.

use super::kernel::GgufType;

const QK_K: usize = 256;
const ROW_SCALE_BYTES: usize = 4;
const GROUP: usize = 8;
const IQ4_KT_GROUP: usize = 4;
const TAIL_BLOCK: usize = 32;
const TRELLIS_MUL: u32 = 0xCBAC1FED;
const TRELLIS_MASK: u32 = 0x3f3f3f3f;
const TRELLIS_BIAS: i32 = 126;
const INDEX_OFFSET: u32 = 4096;
const IQ4_KT_HIGH_OFFSET: u32 = 32768;
// The CUDA and CPU-GEMM kernels scale IQ2_KT by 1.05 and IQ3_KT by 1.01; ik's `dequantize_row_iq2_kt` omits its factor
const IQ2_KT_SCALE: f32 = 1.05;
const IQ3_KT_SCALE: f32 = 1.01;
// The first half of ik's `iq4k_values`, the IQ4_NL codebook
const IQ4K_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
// Byte layouts of the 256-element blocks
const IQ1_KT_QL: usize = 8;
const IQ1_KT_QH: usize = 40;
const IQ2_KT_QL: usize = 4;
const IQ3_KT_QL: usize = 4;
const IQ3_KT_QH: usize = 68;
const IQ4_KT_QL: usize = 32;
const IQ4_KT_QH: usize = 96;

// Each step multiplies the state and sums its four 6-bit bytes: a roughly Gaussian value in -126..=126.
fn trellis_next(state: &mut u32) -> i32 {
    *state = state.wrapping_mul(TRELLIS_MUL);
    let s = *state & TRELLIS_MASK;
    s.to_le_bytes().iter().map(|&b| i32::from(b)).sum::<i32>() - TRELLIS_BIAS
}

fn u16_at(bytes: &[u8], at: usize) -> u32 {
    u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn fill(values: &mut [f32], scale: f32, mut state: u32) {
    for value in values {
        *value = scale * trellis_next(&mut state) as f32;
    }
}

/// Dequantizes one row of `y.len()` elements whose bytes (with the leading f32 scale) are `row`.
pub(super) fn dequantize_row(ty: GgufType, row: &[u8], y: &mut [f32]) {
    let (scale_bytes, blocks) = row.split_at(ROW_SCALE_BYTES);
    let row_scale = f32::from_le_bytes(scale_bytes.try_into().expect("four scale bytes"));
    // Folded into the row scale first, as ik's reference does, so values match it to the bit
    let d = match ty {
        GgufType::Iq2Kt => row_scale * IQ2_KT_SCALE,
        GgufType::Iq3Kt => row_scale * IQ3_KT_SCALE,
        _ => row_scale,
    };
    let size = ty.type_size();
    let whole = y.len() / QK_K;
    for (i, out) in y.as_chunks_mut::<QK_K>().0.iter_mut().enumerate() {
        let block = &blocks[i * size..][..size];
        match ty {
            GgufType::Iq1Kt => iq1_kt(block, d, out),
            GgufType::Iq2Kt => iq2_kt(block, d, out),
            GgufType::Iq3Kt => iq3_kt(block, d, out),
            GgufType::Iq4Kt => iq4_kt(block, d, out),
            _ => unreachable!("{ty:?} is not a trellis type"),
        }
    }
    let tails = (y.len() % QK_K) / TAIL_BLOCK;
    if tails > 0 {
        let tail = &blocks[whole * size..];
        let out = &mut y[whole * QK_K..];
        match ty {
            GgufType::Iq3Kt => iq3_kt_tail(tail, tails, d, out),
            GgufType::Iq4Kt => iq4_kt_tail(tail, tails, d, out),
            _ => unreachable!("{ty:?} rows are whole blocks"),
        }
    }
}

fn iq1_kt(block: &[u8], d: f32, y: &mut [f32]) {
    let (sh, ql, qh) = (
        &block[..IQ1_KT_QL],
        &block[IQ1_KT_QL..IQ1_KT_QH],
        &block[IQ1_KT_QH..],
    );
    for (g, out) in y.as_chunks_mut::<GROUP>().0.iter_mut().enumerate() {
        let s = u32::from(sh[g / 4]);
        let idx = u32::from(ql[g])
            | ((u32::from(qh[g % 16]) << (8 - 4 * (g / 16))) & 0xf00)
            | ((s << (8 - (g % 4))) & 0x1000);
        let scale = d * f32::from(IQ4K_VALUES[(s & 0xf) as usize]);
        fill(out, scale, idx + INDEX_OFFSET);
    }
}

fn iq2_kt(block: &[u8], d: f32, y: &mut [f32]) {
    for (g, out) in y.as_chunks_mut::<GROUP>().0.iter_mut().enumerate() {
        let nibble = (block[(g / 4) % 4] >> (4 * (g / 16))) & 0xf;
        let scale = d * f32::from(IQ4K_VALUES[usize::from(nibble)]);
        fill(out, scale, u16_at(block, IQ2_KT_QL + 2 * g) + INDEX_OFFSET);
    }
}

fn signed_abs_fill(
    values: &mut [f32],
    scale: f32,
    mut state: u32,
    negative: impl Fn(usize) -> bool,
) {
    for (j, value) in values.iter_mut().enumerate() {
        let magnitude = scale * trellis_next(&mut state).abs() as f32;
        *value = if negative(j) { -magnitude } else { magnitude };
    }
}

fn iq3_kt(block: &[u8], d: f32, y: &mut [f32]) {
    let qh = &block[IQ3_KT_QH..];
    for (g, out) in y.as_chunks_mut::<GROUP>().0.iter_mut().enumerate() {
        let nibble = (block[(g / 4) % 4] >> (4 * (g / 16))) & 0xf;
        let scale = d * f32::from(nibble);
        let mask = 1u8 << (g / 4);
        signed_abs_fill(
            out,
            scale,
            u16_at(block, IQ3_KT_QL + 2 * g) + INDEX_OFFSET,
            |j| qh[(GROUP * g + j) % 32] & mask != 0,
        );
    }
}

fn iq3_kt_tail(tail: &[u8], tails: usize, d: f32, y: &mut [f32]) {
    let (qh, scales) = (8 * tails, 12 * tails);
    for (g, out) in y
        .as_chunks_mut::<GROUP>()
        .0
        .iter_mut()
        .take(4 * tails)
        .enumerate()
    {
        let nibble = (tail[scales + g / 8] >> (4 * ((g / 4) & 1))) & 0xf;
        let scale = d * f32::from(nibble);
        let signs = tail[qh + g];
        signed_abs_fill(out, scale, u16_at(tail, 2 * g) + INDEX_OFFSET, |j| {
            signs & (1 << j) != 0
        });
    }
}

fn iq4_kt_sub_block(sh: u32, ql: &[u8], qh_bits: impl Fn(usize) -> u32, d: f32, y: &mut [f32]) {
    let offset = INDEX_OFFSET + if sh & 1 != 0 { IQ4_KT_HIGH_OFFSET } else { 0 };
    let scale = d * (((sh & 0xff) >> 1) as i32 - 64) as f32;
    for (ig, out) in y.as_chunks_mut::<IQ4_KT_GROUP>().0.iter_mut().enumerate() {
        let idx = u32::from(ql[ig]) | qh_bits(ig) | (((sh >> (8 + 3 * ig)) & 7) << 12);
        fill(out, scale, idx + offset);
    }
}

fn iq4_kt(block: &[u8], d: f32, y: &mut [f32]) {
    let (ql, qh) = (&block[IQ4_KT_QL..IQ4_KT_QH], &block[IQ4_KT_QH..]);
    for (ib, out) in y.as_chunks_mut::<TAIL_BLOCK>().0.iter_mut().enumerate() {
        let jj = |ig: usize| ib * 8 + ig;
        iq4_kt_sub_block(
            u32_at(block, 4 * ib),
            &ql[8 * ib..],
            |ig| (u32::from(qh[jj(ig) % 32]) << (8 - 4 * (jj(ig) / 32))) & 0xf00,
            d,
            out,
        );
    }
}

fn iq4_kt_tail(tail: &[u8], tails: usize, d: f32, y: &mut [f32]) {
    for (ib, out) in y
        .as_chunks_mut::<TAIL_BLOCK>()
        .0
        .iter_mut()
        .take(tails)
        .enumerate()
    {
        let sub = &tail[16 * ib..];
        let qh = &sub[12..16];
        iq4_kt_sub_block(
            u32_at(sub, 0),
            &sub[4..12],
            |ig| (u32::from((qh[ig / 2] >> (4 * (ig & 1))) & 0xf)) << 8,
            d,
            out,
        );
    }
}
