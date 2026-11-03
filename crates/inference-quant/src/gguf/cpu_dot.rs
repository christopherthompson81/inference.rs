//! One GGUF row dotted with Q8_K activations on the CPU, from ggml-cpu's `ggml_vec_dot_*_q8_K` (generic and AVX2).

use inference_tensor::quantized::GgmlType;
use inference_tensor::quantized::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K,
};

use super::iq_tables::{IQ2XXS_GRID, IQ3S_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use super::kernel::GgufType;
use super::raw::KVALUES_IQ4NL;

const IQ4_XS_BYTES: usize = 136;
const IQ4_XS_SCALES_H: usize = 2;
const IQ4_XS_SCALES_L: usize = 4;
const IQ4_XS_QS: usize = 8;
const IQ3_S_BYTES: usize = 110;
const IQ3_S_QS: usize = 2;
const IQ3_S_QH: usize = 66;
const IQ3_S_SIGNS: usize = 74;
const IQ3_S_SCALES: usize = 106;
// an f16 scale, then per 32 values: four grid bytes and a word of four 7-bit sign codes and a 4-bit scale
const IQ2_XXS_BYTES: usize = 66;
const IQ2_XXS_QS: usize = 2;
// the sub-block scale is (2 * ls + 1) / 8 of the block's
const IQ2_XXS_SCALE: f32 = 0.125;
// each 7-bit sign code's eight signs as bytes of +1 / -1 (ggml's `keven_signs_q2xs`): the eighth makes the parity even
const IQ2_XXS_SIGNS: [u64; 128] = iq2_xxs_signs();
const IQ4_KSS_BYTES: usize = 128;
const IQ4_KSS_ROW_SCALE: usize = 4;
// ik's shifted IQ4_K table is the IQ4_NL one plus this, picked by a sub-block scale code's low bit
const IQ4_KSS_SHIFT: i8 = 4;

const fn iq2_xxs_signs() -> [u64; 128] {
    let mut table = [0u64; 128];
    let mut code = 0;
    while code < 128 {
        let bits = KSIGNS_IQ2XS[code];
        let mut j = 0;
        while j < 8 {
            let byte: u64 = if bits & (1 << j) != 0 { 0xff } else { 0x01 };
            table[code] |= byte << (8 * j);
            j += 1;
        }
        code += 1;
    }
    table
}

/// Whether `vec_dot` serves this type.
pub(crate) fn supports(ty: GgufType) -> bool {
    matches!(
        ty,
        GgufType::Iq2Xxs
            | GgufType::Iq3S
            | GgufType::Iq4Xs
            | GgufType::Iq4Kss
            | GgufType::Q2K
            | GgufType::Q3K
            | GgufType::Q4K
            | GgufType::Q5K
            | GgufType::Q6K
    )
}

/// `row` holds `ys.len()` super-blocks of `ty`, one per Q8_K block, after the row scale of a type that has one.
pub(crate) fn vec_dot(ty: GgufType, row: &[u8], ys: &[BlockQ8K]) -> f32 {
    let n = ys.len() * QK_K;
    match ty {
        GgufType::Iq2Xxs => iq2_xxs(row, ys),
        GgufType::Iq3S => iq3_s(row, ys),
        GgufType::Iq4Xs => iq4_xs(row, ys),
        GgufType::Iq4Kss => iq4_kss(row, ys),
        GgufType::Q2K => BlockQ2K::vec_dot(n, blocks(row), ys),
        GgufType::Q3K => BlockQ3K::vec_dot(n, blocks(row), ys),
        GgufType::Q4K => BlockQ4K::vec_dot(n, blocks(row), ys),
        GgufType::Q5K => BlockQ5K::vec_dot(n, blocks(row), ys),
        GgufType::Q6K => BlockQ6K::vec_dot(n, blocks(row), ys),
        _ => unreachable!("cpu_dot::supports({ty:?}) is false"),
    }
}

/// Values per block of the 32-wide family (IQ4_NL, Q8_0), whose rows need not fill a Q8_K super-block.
pub(crate) const QK32: usize = 32;
const IQ4_NL_BYTES: usize = 18;
const Q4_0_BYTES: usize = 18;
// an f16 scale, the 32 values' fifth bits, their low nibbles
const Q5_0_BYTES: usize = 22;
const Q5_0_QH: usize = 2;
const Q5_0_QS: usize = 6;
const Q8_0_BYTES: usize = 34;
// Q4_0 and Q5_0 store values offset by these
const Q4_0_OFFSET: i32 = 8;
const Q5_0_OFFSET: i32 = 16;
// both blocks open with their f16 scale
const BLOCK32_QS: usize = 2;
// Bytes ahead of the block being dotted that a single-token dot prefetches: expert rows stream from DRAM, and the
// hardware prefetcher alone leaves the dots waiting on them
#[cfg(target_arch = "x86_64")]
const PREFETCH_AHEAD: usize = 1024;

/// Whether `vec_dot_32` serves this type.
pub(crate) fn supports_32(ty: GgufType) -> bool {
    matches!(
        ty,
        GgufType::Iq4Nl | GgufType::Q4_0 | GgufType::Q5_0 | GgufType::Q8_0
    )
}

/// 32 activations quantized to i8 under one scale, as ggml's Q8_0 with an f32 scale.
#[derive(Clone, Copy)]
pub(crate) struct Q8Block32 {
    d: f32,
    // the sum of `qs`, for kernels that dot unsigned weights and take their offset back out
    sum: i32,
    qs: [i8; QK32],
}

impl Q8Block32 {
    pub(crate) const ZERO: Self = Self {
        d: 0.0,
        sum: 0,
        qs: [0; QK32],
    };

    /// `xs` in blocks of 32, into `ys`.
    pub(crate) fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (x, y) in xs.as_chunks::<QK32>().0.iter().zip(ys) {
            y.d = inference_tensor::quantized::quantize_block32(x, &mut y.qs);
            y.sum = y.qs.iter().map(|&q| i32::from(q)).sum();
        }
    }
}

/// [`vec_dot_32`] for four activation rows, each weight block decoded once; every row scores as it would alone.
pub(crate) fn vec_dot_32_x4(ty: GgufType, row: &[u8], ys: [&[Q8Block32]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match ty {
            // SAFETY: AVX2 and FMA were detected above
            GgufType::Iq4Nl => return unsafe { avx2::iq4_nl_32_x4(row, ys) },
            GgufType::Q4_0 => return unsafe { avx2::q4_0_32_x4(row, ys) },
            GgufType::Q5_0 => return unsafe { avx2::q5_0_32_x4(row, ys) },
            GgufType::Q8_0 => return unsafe { avx2::q8_0_32_x4(row, ys) },
            _ => {}
        }
    }
    ys.map(|y| vec_dot_32(ty, row, y))
}

/// Eight rows of a 32-wide `ty` against four activation rows, `[token][row]`: each row's integer block sums land in
/// one lane, so a block's scales apply to all eight rows in one multiply. Within rounding of [`vec_dot_32`].
pub(crate) fn vec_dot_32_8x4(
    ty: GgufType,
    rows: [&[u8]; 8],
    ys: [&[Q8Block32]; 4],
) -> [[f32; 8]; 4] {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match ty {
            // SAFETY: AVX2 and FMA were detected above
            GgufType::Iq4Nl => return unsafe { avx2::iq4_nl_32_8x4(rows, ys) },
            GgufType::Q4_0 => return unsafe { avx2::q4_0_32_8x4(rows, ys) },
            GgufType::Q5_0 => return unsafe { avx2::q5_0_32_8x4(rows, ys) },
            GgufType::Q8_0 => return unsafe { avx2::q8_0_32_8x4(rows, ys) },
            _ => {}
        }
    }
    ys.map(|y| rows.map(|row| vec_dot_32(ty, row, y)))
}

/// [`vec_dot_32_8x4`] for one activation row, as a decode step's single route per expert is.
pub(crate) fn vec_dot_32_8x1(ty: GgufType, rows: [&[u8]; 8], y: &[Q8Block32]) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match ty {
            // SAFETY: AVX2 and FMA were detected above
            GgufType::Iq4Nl => return unsafe { avx2::iq4_nl_32_8x1(rows, y) },
            GgufType::Q4_0 => return unsafe { avx2::q4_0_32_8x1(rows, y) },
            GgufType::Q5_0 => return unsafe { avx2::q5_0_32_8x1(rows, y) },
            GgufType::Q8_0 => return unsafe { avx2::q8_0_32_8x1(rows, y) },
            _ => {}
        }
    }
    rows.map(|row| vec_dot_32(ty, row, y))
}

/// `row` holds `ys.len()` blocks of a 32-wide `ty`.
pub(crate) fn vec_dot_32(ty: GgufType, row: &[u8], ys: &[Q8Block32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match ty {
            // SAFETY: AVX2 and FMA were detected above
            GgufType::Iq4Nl => return unsafe { avx2::iq4_nl_32(row, ys) },
            GgufType::Q4_0 => return unsafe { avx2::q4_0_32(row, ys) },
            GgufType::Q5_0 => return unsafe { avx2::q5_0_32(row, ys) },
            GgufType::Q8_0 => return unsafe { avx2::q8_0_32(row, ys) },
            _ => {}
        }
    }
    vec_dot_32_generic(ty, row, ys)
}

fn vec_dot_32_generic(ty: GgufType, row: &[u8], ys: &[Q8Block32]) -> f32 {
    match ty {
        GgufType::Iq4Nl => row
            .as_chunks::<IQ4_NL_BYTES>()
            .0
            .iter()
            .zip(ys)
            .map(|(block, y)| {
                let qs = &block[BLOCK32_QS..];
                let sum = (0..QK32 / 2)
                    .map(|j| {
                        i32::from(KVALUES_IQ4NL[usize::from(qs[j] & 0x0F)]) * i32::from(y.qs[j])
                            + i32::from(KVALUES_IQ4NL[usize::from(qs[j] >> 4)])
                                * i32::from(y.qs[j + QK32 / 2])
                    })
                    .sum::<i32>();
                f16_at(block, 0) * y.d * sum as f32
            })
            .sum(),
        GgufType::Q4_0 => row
            .as_chunks::<Q4_0_BYTES>()
            .0
            .iter()
            .zip(ys)
            .map(|(block, y)| {
                let qs = &block[BLOCK32_QS..];
                let sum = (0..QK32 / 2)
                    .map(|j| {
                        (i32::from(qs[j] & 0x0F) - Q4_0_OFFSET) * i32::from(y.qs[j])
                            + (i32::from(qs[j] >> 4) - Q4_0_OFFSET) * i32::from(y.qs[j + QK32 / 2])
                    })
                    .sum::<i32>();
                f16_at(block, 0) * y.d * sum as f32
            })
            .sum(),
        GgufType::Q5_0 => row
            .as_chunks::<Q5_0_BYTES>()
            .0
            .iter()
            .zip(ys)
            .map(|(block, y)| {
                let qh = u32::from_le_bytes([
                    block[Q5_0_QH],
                    block[Q5_0_QH + 1],
                    block[Q5_0_QH + 2],
                    block[Q5_0_QH + 3],
                ]);
                let qs = &block[Q5_0_QS..];
                let value =
                    |nibble: u8, bit: usize| i32::from(nibble) | (((qh >> bit) & 1) as i32) << 4;
                let sum = (0..QK32 / 2)
                    .map(|j| {
                        (value(qs[j] & 0x0F, j) - Q5_0_OFFSET) * i32::from(y.qs[j])
                            + (value(qs[j] >> 4, j + QK32 / 2) - Q5_0_OFFSET)
                                * i32::from(y.qs[j + QK32 / 2])
                    })
                    .sum::<i32>();
                f16_at(block, 0) * y.d * sum as f32
            })
            .sum(),
        GgufType::Q8_0 => row
            .as_chunks::<Q8_0_BYTES>()
            .0
            .iter()
            .zip(ys)
            .map(|(block, y)| {
                let sum = block[BLOCK32_QS..]
                    .iter()
                    .zip(&y.qs)
                    .map(|(&w, &x)| i32::from(w as i8) * i32::from(x))
                    .sum::<i32>();
                f16_at(block, 0) * y.d * sum as f32
            })
            .sum(),
        _ => unreachable!("cpu_dot::supports_32({ty:?}) is false"),
    }
}

fn blocks<T>(row: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert!(
        row.len().is_multiple_of(size)
            && (row.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()),
        "GGUF row is not a whole number of aligned blocks"
    );
    // SAFETY: the k-quant blocks are repr(C) plain data, and the length and alignment were checked above
    unsafe { std::slice::from_raw_parts(row.as_ptr().cast(), row.len() / size) }
}

fn f16_at(block: &[u8], at: usize) -> f32 {
    half::f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
}

#[cfg(target_arch = "x86_64")]
fn has_avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
}

fn iq4_xs(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above
        return unsafe { avx2::iq4_xs_one(row, ys) };
    }
    iq4_xs_generic(row, ys)
}

fn iq2_xxs(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above
        return unsafe { avx2::iq2_xxs_one(row, ys) };
    }
    iq2_xxs_generic(row, ys)
}

fn iq3_s(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above
        return unsafe { avx2::iq3_s_one(row, ys) };
    }
    iq3_s_generic(row, ys)
}

fn iq4_kss(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above
        return unsafe { avx2::iq4_kss_one(row, ys) };
    }
    iq4_kss_generic(row, ys)
}

/// `vec_dot` of one row against four activation rows, decoding the row's blocks once for all of them.
pub(crate) fn vec_dot_x4(ty: GgufType, row: &[u8], ys: [&[BlockQ8K]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match ty {
            // SAFETY: AVX2 and FMA were detected above
            GgufType::Iq2Xxs => return unsafe { avx2::iq2_xxs_x4(row, ys) },
            GgufType::Iq3S => return unsafe { avx2::iq3_s_x4(row, ys) },
            GgufType::Iq4Xs => return unsafe { avx2::iq4_xs_x4(row, ys) },
            GgufType::Iq4Kss => return unsafe { avx2::iq4_kss_x4(row, ys) },
            _ => {}
        }
    }
    ys.map(|y| vec_dot(ty, row, y))
}

/// Whether `decode_tile` and `tile_dot` serve this type on this CPU.
pub(crate) fn tile_supports(ty: GgufType) -> bool {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        return matches!(
            ty,
            GgufType::Iq2Xxs | GgufType::Iq3S | GgufType::Iq4Xs | GgufType::Iq4Kss
        );
    }
    let _ = ty;
    false
}

/// Decodes whole rows of `cols` values of `ty` into i16 weights with each sub-block's scale folded in, plus each
/// block's scale (the row scale, for a type that has one).
pub(crate) fn decode_tile(ty: GgufType, rows: &[u8], cols: usize, w: &mut [i16], d: &mut [f32]) {
    let row_bytes = ty.row_scale_bytes() + cols / QK_K * ty.type_size();
    assert!(
        tile_supports(ty)
            && w.len() == d.len() * QK_K
            && w.len().is_multiple_of(cols)
            && rows.len() == w.len() / cols * row_bytes
    );
    #[cfg(target_arch = "x86_64")]
    // SAFETY: tile_supports checked AVX2 and FMA
    unsafe {
        match ty {
            GgufType::Iq2Xxs => avx2::decode_iq2_xxs(rows, w, d),
            GgufType::Iq3S => avx2::decode_iq3_s(rows, w, d),
            GgufType::Iq4Kss => avx2::decode_iq4_kss(rows, row_bytes, cols, w, d),
            _ => avx2::decode_iq4_xs(rows, w, d),
        }
    }
}

/// One activation row for `tile_dot`: its Q8_K blocks and their values widened to i16.
#[derive(Clone, Copy)]
pub(crate) struct TileRow<'a> {
    pub qs: &'a [i16],
    pub blocks: &'a [BlockQ8K],
}

/// Widens Q8_K values to i16, as `TileRow::qs` takes them.
pub(crate) fn widen_into(blocks: &[BlockQ8K], qs: &mut [i16]) {
    for (b, qs) in blocks.iter().zip(qs.as_chunks_mut::<QK_K>().0) {
        for (q, &v) in qs.iter_mut().zip(b.qs()) {
            *q = i16::from(v);
        }
    }
}

/// `out[r * tokens.len() + t]` = decoded row `r` . token `t`, over a tile from `decode_tile` with an even row count.
pub(crate) fn tile_dot(w: &[i16], d: &[f32], tokens: &[TileRow], out: &mut [f32]) {
    let blocks = tokens[0].blocks.len();
    assert!(
        d.len().is_multiple_of(2 * blocks)
            && w.len() == d.len() * QK_K
            && out.len() == d.len() / blocks * tokens.len()
            && tokens
                .iter()
                .all(|t| t.blocks.len() == blocks && t.qs.len() == blocks * QK_K)
    );
    #[cfg(target_arch = "x86_64")]
    // SAFETY: decode_tile only fills a tile when AVX2 and FMA are present, and the lengths were checked above
    unsafe {
        avx2::tile_dot(w, d, tokens, out)
    }
}

fn iq4_xs_sub_scale(block: &[u8], ib: usize) -> i32 {
    let scales_h = u16::from_le_bytes([block[IQ4_XS_SCALES_H], block[IQ4_XS_SCALES_H + 1]]);
    let low = (block[IQ4_XS_SCALES_L + ib / 2] >> (4 * (ib % 2))) & 0xf;
    let high = ((scales_h >> (2 * ib)) & 3) as u8;
    i32::from(low | (high << 4)) - 32
}

fn iq4_xs_generic(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (block, y) in row.as_chunks::<IQ4_XS_BYTES>().0.iter().zip(ys) {
        let d = f16_at(block, 0) * y.d();
        let q8 = y.qs();
        for ib in 0..QK_K / 32 {
            let qs = &block[IQ4_XS_QS + 16 * ib..][..16];
            let q8 = &q8[32 * ib..][..32];
            let sumi = (0..16)
                .map(|j| {
                    i32::from(q8[j]) * i32::from(KVALUES_IQ4NL[usize::from(qs[j] & 0xf)])
                        + i32::from(q8[j + 16]) * i32::from(KVALUES_IQ4NL[usize::from(qs[j] >> 4)])
                })
                .sum::<i32>();
            sumf += d * (iq4_xs_sub_scale(block, ib) * sumi) as f32;
        }
    }
    sumf
}

fn row_scale(row: &[u8]) -> f32 {
    f32::from_le_bytes(row[..IQ4_KSS_ROW_SCALE].try_into().expect("f32 row scale"))
}

// sub-block `ib`'s eight u16 codes: their low bits are its scale code, the rest its nibbles once unmixed
fn iq4_kss_codes(block: &[u8], ib: usize) -> ([u8; 16], u8) {
    let (mut nibbles, mut ls) = ([0u8; 16], 0u8);
    for k in 0..8 {
        let q = u16::from_le_bytes([block[16 * ib + 2 * k], block[16 * ib + 2 * k + 1]]);
        let a = q & 0xfffe;
        nibbles[2 * k..2 * k + 2].copy_from_slice(&(a ^ (a >> 1)).to_le_bytes());
        ls |= ((q & 1) as u8) << k;
    }
    (nibbles, ls)
}

fn iq4_kss_sub_scale(ls: u8) -> i32 {
    i32::from(ls & 254) - 127
}

fn iq4_kss_shift(ls: u8) -> i8 {
    IQ4_KSS_SHIFT * (ls & 1) as i8
}

fn iq4_kss_generic(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    let blocks = row[IQ4_KSS_ROW_SCALE..].as_chunks::<IQ4_KSS_BYTES>().0;
    let mut sumf = 0f32;
    for (block, y) in blocks.iter().zip(ys) {
        let q8 = y.qs();
        let mut bsum = 0i32;
        for ib in 0..QK_K / 32 {
            let (qs, ls) = iq4_kss_codes(block, ib);
            let value = |code: u8| i32::from(KVALUES_IQ4NL[usize::from(code)] + iq4_kss_shift(ls));
            let q8 = &q8[32 * ib..][..32];
            let sumi = (0..16)
                .map(|j| {
                    i32::from(q8[j]) * value(qs[j] & 0xf)
                        + i32::from(q8[j + 16]) * value(qs[j] >> 4)
                })
                .sum::<i32>();
            bsum += iq4_kss_sub_scale(ls) * sumi;
        }
        sumf += y.d() * bsum as f32;
    }
    row_scale(row) * sumf
}

// sub-block `ib32`'s four grid bytes and the word of its sign codes (7 bits each) and scale (the top 4)
fn iq2_xxs_words(block: &[u8], ib32: usize) -> ([u8; 4], u32) {
    let at = IQ2_XXS_QS + 8 * ib32;
    let grid = block[at..at + 4].try_into().expect("4 grid bytes");
    let word = u32::from_le_bytes(block[at + 4..at + 8].try_into().expect("a sign word"));
    (grid, word)
}

fn iq2_xxs_generic(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (block, y) in row.as_chunks::<IQ2_XXS_BYTES>().0.iter().zip(ys) {
        let d = f16_at(block, 0) * y.d();
        let q8 = y.qs();
        let mut bsum = 0i32;
        for ib32 in 0..QK_K / 32 {
            let (grid, word) = iq2_xxs_words(block, ib32);
            let mut sumi = 0i32;
            for l in 0..4 {
                let values = IQ2XXS_GRID[usize::from(grid[l])].to_le_bytes();
                let signs = IQ2_XXS_SIGNS[((word >> (7 * l)) & 127) as usize].to_le_bytes();
                for j in 0..8 {
                    sumi += i32::from(values[j])
                        * i32::from(signs[j] as i8)
                        * i32::from(q8[32 * ib32 + 8 * l + j]);
                }
            }
            bsum += sumi * (2 * (word >> 28) as i32 + 1);
        }
        sumf += d * bsum as f32;
    }
    IQ2_XXS_SCALE * sumf
}

fn iq3_s_grid_index(block: &[u8], ib32: usize, k: usize) -> usize {
    let qh = usize::from(block[IQ3_S_QH + ib32]);
    usize::from(block[IQ3_S_QS + 8 * ib32 + k]) | ((qh << (8 - k)) & 256)
}

fn iq3_s_generic(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (block, y) in row.as_chunks::<IQ3_S_BYTES>().0.iter().zip(ys) {
        let d = f16_at(block, 0) * y.d();
        let q8 = y.qs();
        let mut bsum = 0i32;
        for ib32 in 0..QK_K / 32 {
            let scale = block[IQ3_S_SCALES + ib32 / 2] >> (4 * (ib32 % 2)) & 0xf;
            let mut sumi = 0i32;
            for l in 0..4 {
                let signs = block[IQ3_S_SIGNS + 4 * ib32 + l];
                let q8 = &q8[32 * ib32 + 8 * l..][..8];
                for half in 0..2 {
                    let grid = IQ3S_GRID[iq3_s_grid_index(block, ib32, 2 * l + half)].to_le_bytes();
                    for j in 0..4 {
                        let value = i32::from(grid[j]) * i32::from(q8[4 * half + j]);
                        sumi += if signs & KMASK_IQ2XS[4 * half + j] != 0 {
                            -value
                        } else {
                            value
                        };
                    }
                }
            }
            bsum += sumi * (2 * i32::from(scale) + 1);
        }
        sumf += d * bsum as f32;
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use inference_tensor::quantized::k_quants::{BlockQ8K, QK_K};

    use super::{
        BLOCK32_QS, IQ2_XXS_BYTES, IQ2_XXS_SCALE, IQ2_XXS_SIGNS, IQ2XXS_GRID, IQ3_S_BYTES,
        IQ3_S_QH, IQ3_S_QS, IQ3_S_SCALES, IQ3_S_SIGNS, IQ3S_GRID, IQ4_KSS_BYTES, IQ4_KSS_ROW_SCALE,
        IQ4_KSS_SHIFT, IQ4_NL_BYTES, IQ4_XS_BYTES, IQ4_XS_QS, KVALUES_IQ4NL, PREFETCH_AHEAD,
        Q4_0_BYTES, Q4_0_OFFSET, Q5_0_BYTES, Q5_0_OFFSET, Q5_0_QH, Q5_0_QS, Q8_0_BYTES, Q8Block32,
        TileRow, f16_at, iq2_xxs_words, iq4_xs_sub_scale, row_scale,
    };

    // byte b of each 32-bit sign word spread over lanes 8b..8b+8, then each lane's bit
    const SIGN_BYTES: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    ];
    const SIGN_BITS: [u8; 32] = [
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1,
        2, 4, 8, 16, 32, 64, 128,
    ];

    #[target_feature(enable = "avx2,fma")]
    fn hsum(x: __m256) -> f32 {
        let res = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        _mm_cvtss_f32(_mm_add_ss(res, _mm_movehdup_ps(res)))
    }

    #[target_feature(enable = "avx2,fma")]
    fn load(bytes: &[u8]) -> __m256i {
        assert!(bytes.len() >= 32);
        // SAFETY: 32 readable bytes were checked above, and the load is unaligned
        unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2,fma")]
    fn load_q8(qs: &[i8]) -> __m256i {
        assert!(qs.len() >= 32);
        // SAFETY: as in `load`
        unsafe { _mm256_loadu_si256(qs.as_ptr().cast()) }
    }

    // asks for `lines` cache lines `PREFETCH_AHEAD` bytes past `block`, which may lie past the row (a prefetch never faults)
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn prefetch_ahead(block: &[u8], lines: usize) {
        let ahead = block.as_ptr().wrapping_add(PREFETCH_AHEAD).cast::<i8>();
        for line in 0..lines {
            _mm_prefetch::<_MM_HINT_T0>(ahead.wrapping_add(64 * line));
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq4_xs_one(row: &[u8], ys: &[BlockQ8K]) -> f32 {
        // SAFETY: the codebook holds 16 bytes
        let values = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) };
        let m4 = _mm_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        for (block, y) in row.as_chunks::<IQ4_XS_BYTES>().0.iter().zip(ys) {
            prefetch_ahead(block, 2);
            let mut sumi = _mm256_setzero_si256();
            for ib in 0..QK_K / 32 {
                // SAFETY: a sub-block's 16 nibble bytes lie inside the block
                let bits = unsafe { _mm_loadu_si128(block[IQ4_XS_QS + 16 * ib..].as_ptr().cast()) };
                let q4 = _mm256_set_m128i(
                    _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                    _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)),
                );
                let q8 = load_q8(&y.qs()[32 * ib..]);
                // |q4| * sign(q4)·q8, as maddubs takes one unsigned side
                let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(q4, q4), _mm256_sign_epi8(q8, q4));
                let scale = _mm256_set1_epi16(iq4_xs_sub_scale(block, ib) as i16);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, scale));
            }
            let d = _mm256_set1_ps(f16_at(block, 0) * y.d());
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum(acc)
    }

    // the i32 sums of `w * y` over a block's 32 signed bytes, as maddubs takes one unsigned side
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn signed_dot(w: __m256i, y: __m256i) -> __m256i {
        let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(y, w));
        _mm256_madd_epi16(p16, _mm256_set1_epi16(1))
    }

    // the f16 at `at`, converted in hardware; `half` reaches F16C only through a call per value
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn f16c_at(block: &[u8], at: usize) -> f32 {
        let bits = i32::from(u16::from_le_bytes([block[at], block[at + 1]]));
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)))
    }

    // a 32-value block's scale times the activations' and its integer dot, onto `acc`
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn scaled_32<const N: usize>(
        block: &[u8; N],
        w: __m256i,
        y: &Q8Block32,
        acc: __m256,
    ) -> __m256 {
        let d = _mm256_set1_ps(f16c_at(block, 0) * y.d);
        _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(signed_dot(w, load_q8(&y.qs))), acc)
    }

    // `row`'s blocks decoded by `decode` (each to its 32 signed weights) dotted with `ys`, alternate blocks on two
    // accumulators
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn sum_blocks_32<const N: usize>(
        row: &[u8],
        ys: &[Q8Block32],
        decode: impl Fn(&[u8; N]) -> __m256i,
    ) -> f32 {
        let blocks = &row.as_chunks::<N>().0[..ys.len()];
        let (pairs, rest) = blocks.as_chunks::<2>();
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        for ([b0, b1], y) in pairs.iter().zip(ys.as_chunks::<2>().0) {
            prefetch_ahead(b0, 1);
            a0 = scaled_32(b0, decode(b0), &y[0], a0);
            prefetch_ahead(b1, 1);
            a1 = scaled_32(b1, decode(b1), &y[1], a1);
        }
        if let [b] = rest {
            a0 = scaled_32(b, decode(b), &ys[ys.len() - 1], a0);
        }
        hsum(_mm256_add_ps(a0, a1))
    }

    // `sum_blocks_32` for four activation rows, each block decoded once for all of them; every row's sums run as
    // the single-row kernel runs them, so a row scores the same in either
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn sum_blocks_32_x4<const N: usize>(
        row: &[u8],
        ys: [&[Q8Block32]; 4],
        decode: impl Fn(&[u8; N]) -> __m256i,
    ) -> [f32; 4] {
        let n = ys[0].len();
        let blocks = &row.as_chunks::<N>().0[..n];
        let (pairs, rest) = blocks.as_chunks::<2>();
        let (mut a0, mut a1) = ([_mm256_setzero_ps(); 4], [_mm256_setzero_ps(); 4]);
        for (p, [b0, b1]) in pairs.iter().enumerate() {
            prefetch_ahead(b0, 1);
            prefetch_ahead(b1, 1);
            let (w0, w1) = (decode(b0), decode(b1));
            for t in 0..4 {
                a0[t] = scaled_32(b0, w0, &ys[t][2 * p], a0[t]);
                a1[t] = scaled_32(b1, w1, &ys[t][2 * p + 1], a1[t]);
            }
        }
        if let [b] = rest {
            let w = decode(b);
            for t in 0..4 {
                a0[t] = scaled_32(b, w, &ys[t][n - 1], a0[t]);
            }
        }
        std::array::from_fn(|t| hsum(_mm256_add_ps(a0[t], a1[t])))
    }

    // `v`'s 32-bit word k in every word, for k = 0..4 of each 128-bit lane
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn words(v: __m256i) -> [__m256i; 4] {
        [
            _mm256_shuffle_epi32::<0x00>(v),
            _mm256_shuffle_epi32::<0x55>(v),
            _mm256_shuffle_epi32::<0xaa>(v),
            _mm256_shuffle_epi32::<0xff>(v),
        ]
    }

    // 4x4 transposes of 32-bit words within each 128-bit lane: word j of output k is word k of input j
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn transpose_words([a, b, c, d]: [__m256i; 4]) -> [__m256i; 4] {
        let (ab_lo, ab_hi) = (_mm256_unpacklo_epi32(a, b), _mm256_unpackhi_epi32(a, b));
        let (cd_lo, cd_hi) = (_mm256_unpacklo_epi32(c, d), _mm256_unpackhi_epi32(c, d));
        [
            _mm256_unpacklo_epi64(ab_lo, cd_lo),
            _mm256_unpackhi_epi64(ab_lo, cd_lo),
            _mm256_unpacklo_epi64(ab_hi, cd_hi),
            _mm256_unpackhi_epi64(ab_hi, cd_hi),
        ]
    }

    // two rows' blocks as (values 0..16, values 16..32), row `a` in the low lane and row `b` in the high one
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn split_pair(a: __m256i, b: __m256i) -> [__m256i; 2] {
        [
            _mm256_permute2x128_si256::<0x20>(a, b),
            _mm256_permute2x128_si256::<0x31>(a, b),
        ]
    }

    // two rows' 16 nibble bytes as `split_pair` lays them out, before any codebook
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn nibble_pair(a: &[u8], b: &[u8]) -> [__m256i; 2] {
        assert!(a.len() >= 16 && b.len() >= 16);
        // SAFETY: 16 readable bytes of each were checked above
        let bits = unsafe { _mm256_loadu2_m128i(b.as_ptr().cast(), a.as_ptr().cast()) };
        let m4 = _mm256_set1_epi8(0x0f);
        [
            _mm256_and_si256(bits, m4),
            _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4),
        ]
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn pair_iq4_nl(a: &[u8; IQ4_NL_BYTES], b: &[u8; IQ4_NL_BYTES]) -> [__m256i; 2] {
        // SAFETY: the codebook holds 16 bytes
        let values =
            _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) });
        let [lo, hi] = nibble_pair(&a[BLOCK32_QS..], &b[BLOCK32_QS..]);
        [
            _mm256_shuffle_epi8(values, lo),
            _mm256_shuffle_epi8(values, hi),
        ]
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn pair_q4_0(a: &[u8; Q4_0_BYTES], b: &[u8; Q4_0_BYTES]) -> [__m256i; 2] {
        nibble_pair(&a[BLOCK32_QS..], &b[BLOCK32_QS..])
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn pair_q5_0(a: &[u8; Q5_0_BYTES], b: &[u8; Q5_0_BYTES]) -> [__m256i; 2] {
        split_pair(decode_q5_0_unsigned(a), decode_q5_0_unsigned(b))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn pair_q8_0(a: &[u8; Q8_0_BYTES], b: &[u8; Q8_0_BYTES]) -> [__m256i; 2] {
        split_pair(decode_q8_0(a), decode_q8_0(b))
    }

    // eight rows by `T` activation rows, the pairs' values signed or, for a nonzero `OFFSET`, unsigned and that much
    // above the weights: rows i and i + 4 decode as one pair, and the transposes leave each row's
    // four values of every word position in its own lane, so lane i of each token's accumulator is row i's sum
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn sum_blocks_32_8xt<const N: usize, const OFFSET: i32, const T: usize>(
        rows: [&[u8]; 8],
        ys: [&[Q8Block32]; T],
        pair: impl Fn(&[u8; N], &[u8; N]) -> [__m256i; 2],
    ) -> [[f32; 8]; T] {
        let n = ys[0].len();
        let blocks = rows.map(|row| &row.as_chunks::<N>().0[..n]);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); T];
        for l in 0..n {
            let (mut lo, mut hi) = ([_mm256_setzero_si256(); 4], [_mm256_setzero_si256(); 4]);
            for i in 0..4 {
                prefetch_ahead(&blocks[i][l], 1);
                prefetch_ahead(&blocks[i + 4][l], 1);
                [lo[i], hi[i]] = pair(&blocks[i][l], &blocks[i + 4][l]);
            }
            let w = [transpose_words(lo), transpose_words(hi)];
            let bits = |i: usize| i16::from_le_bytes([blocks[i][l][0], blocks[i][l][1]]);
            let dw = _mm256_cvtph_ps(_mm_set_epi16(
                bits(7),
                bits(6),
                bits(5),
                bits(4),
                bits(3),
                bits(2),
                bits(1),
                bits(0),
            ));
            for (t, y) in ys.iter().enumerate() {
                let y = &y[l];
                // SAFETY: a block holds 32 activation bytes
                let halves = unsafe {
                    [
                        _mm256_broadcastsi128_si256(_mm_loadu_si128(y.qs.as_ptr().cast())),
                        _mm256_broadcastsi128_si256(_mm_loadu_si128(y.qs[16..].as_ptr().cast())),
                    ]
                };
                // array maps and zips of vectors do not inline into this function, so plain index loops
                let mut sum = _mm256_setzero_si256();
                for h in 0..2 {
                    let y = words(halves[h]);
                    if OFFSET == 0 {
                        for k in 0..4 {
                            let w = w[h][k];
                            let p16 = _mm256_maddubs_epi16(
                                _mm256_sign_epi8(w, w),
                                _mm256_sign_epi8(y[k], w),
                            );
                            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p16, ones));
                        }
                    } else {
                        // weights below 32 against |y| <= 127: four pair sums stay within i16
                        let mut p16 = _mm256_maddubs_epi16(w[h][0], y[0]);
                        for k in 1..4 {
                            p16 = _mm256_add_epi16(p16, _mm256_maddubs_epi16(w[h][k], y[k]));
                        }
                        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p16, ones));
                    }
                }
                if OFFSET != 0 {
                    sum = _mm256_sub_epi32(sum, _mm256_set1_epi32(OFFSET * y.sum));
                }
                let d = _mm256_mul_ps(dw, _mm256_set1_ps(y.d));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), d, acc[t]);
            }
        }
        let mut out = [[0f32; 8]; T];
        for t in 0..T {
            // SAFETY: a row of `out` holds eight floats
            unsafe { _mm256_storeu_ps(out[t].as_mut_ptr(), acc[t]) };
        }
        out
    }

    // a block's 16 nibble bytes as 32 values: the low nibbles first, then the high ones
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn nibbles_32(bytes: &[u8]) -> __m256i {
        assert!(bytes.len() >= 16);
        // SAFETY: 16 readable bytes were checked above
        let bits = unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) };
        let m4 = _mm_set1_epi8(0x0f);
        _mm256_set_m128i(
            _mm_and_si128(_mm_srli_epi16(bits, 4), m4),
            _mm_and_si128(bits, m4),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn decode_iq4_nl(block: &[u8; IQ4_NL_BYTES]) -> __m256i {
        // SAFETY: the codebook holds 16 bytes
        let values =
            _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) });
        _mm256_shuffle_epi8(values, nibbles_32(&block[BLOCK32_QS..]))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn decode_q4_0(block: &[u8; Q4_0_BYTES]) -> __m256i {
        _mm256_sub_epi8(
            nibbles_32(&block[BLOCK32_QS..]),
            _mm256_set1_epi8(Q4_0_OFFSET as i8),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn decode_q5_0(block: &[u8; Q5_0_BYTES]) -> __m256i {
        // a clear fifth bit makes the value 16 below its nibble: 0xf0 ORed in is that in two's complement
        _mm256_or_si256(
            nibbles_32(&block[Q5_0_QS..]),
            _mm256_andnot_si256(q5_0_fifth_bits(block), _mm256_set1_epi8(0xf0u8 as i8)),
        )
    }

    // 0xff in each byte whose value has its fifth bit set
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn q5_0_fifth_bits(block: &[u8; Q5_0_BYTES]) -> __m256i {
        // byte i of the broadcast mask picks the byte of `qh` holding bit i, the OR leaves 0xff where the bit is set
        let pick = _mm256_set_epi64x(
            0x0303_0303_0303_0303,
            0x0202_0202_0202_0202,
            0x0101_0101_0101_0101,
            0,
        );
        let others = _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe);
        let qh = i32::from_le_bytes([
            block[Q5_0_QH],
            block[Q5_0_QH + 1],
            block[Q5_0_QH + 2],
            block[Q5_0_QH + 3],
        ]);
        _mm256_cmpeq_epi8(
            _mm256_or_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(qh), pick), others),
            _mm256_set1_epi64x(-1),
        )
    }

    // the block's values plus `Q5_0_OFFSET`, 0..32
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn decode_q5_0_unsigned(block: &[u8; Q5_0_BYTES]) -> __m256i {
        _mm256_or_si256(
            nibbles_32(&block[Q5_0_QS..]),
            _mm256_and_si256(q5_0_fifth_bits(block), _mm256_set1_epi8(0x10)),
        )
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn decode_q8_0(block: &[u8; Q8_0_BYTES]) -> __m256i {
        load(&block[BLOCK32_QS..])
    }

    // each 32-value type's one-row and four-row kernels over its decoder
    macro_rules! kernels_32 {
        ($($single:ident, $quad:ident, $wide:ident, $tall:ident, $bytes:ident, $decode:ident, $pair:ident, $offset:expr;)*) => {$(
            #[target_feature(enable = "avx2,fma,f16c")]
            pub(super) fn $wide(rows: [&[u8]; 8], ys: [&[Q8Block32]; 4]) -> [[f32; 8]; 4] {
                sum_blocks_32_8xt::<$bytes, { $offset }, 4>(rows, ys, |a, b| $pair(a, b))
            }

            #[target_feature(enable = "avx2,fma,f16c")]
            pub(super) fn $tall(rows: [&[u8]; 8], y: &[Q8Block32]) -> [f32; 8] {
                sum_blocks_32_8xt::<$bytes, { $offset }, 1>(rows, [y], |a, b| $pair(a, b))[0]
            }

            #[target_feature(enable = "avx2,fma,f16c")]
            pub(super) fn $single(row: &[u8], ys: &[Q8Block32]) -> f32 {
                sum_blocks_32::<$bytes>(row, ys, |block| $decode(block))
            }

            #[target_feature(enable = "avx2,fma,f16c")]
            pub(super) fn $quad(row: &[u8], ys: [&[Q8Block32]; 4]) -> [f32; 4] {
                sum_blocks_32_x4::<$bytes>(row, ys, |block| $decode(block))
            }
        )*};
    }

    kernels_32! {
        iq4_nl_32, iq4_nl_32_x4, iq4_nl_32_8x4, iq4_nl_32_8x1, IQ4_NL_BYTES, decode_iq4_nl, pair_iq4_nl, 0;
        q4_0_32, q4_0_32_x4, q4_0_32_8x4, q4_0_32_8x1, Q4_0_BYTES, decode_q4_0, pair_q4_0, Q4_0_OFFSET;
        q5_0_32, q5_0_32_x4, q5_0_32_8x4, q5_0_32_8x1, Q5_0_BYTES, decode_q5_0, pair_q5_0, Q5_0_OFFSET;
        q8_0_32, q8_0_32_x4, q8_0_32_8x4, q8_0_32_8x1, Q8_0_BYTES, decode_q8_0, pair_q8_0, 0;
    }

    // the IQ4_NL codebook in both lanes, and each u16 code's place in its sub-block's scale code
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn iq4_kss_consts() -> (__m256i, __m256i) {
        // SAFETY: the codebook holds 16 bytes
        let values = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) };
        let places = _mm256_setr_epi16(1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);
        (_mm256_broadcastsi128_si256(values), places)
    }

    // a block's codes as four loads, each holding sub-blocks 2p and 2p + 1 in its two lanes
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn iq4_kss_codes(block: &[u8; IQ4_KSS_BYTES]) -> [__m256i; 4] {
        [
            load(&block[..]),
            load(&block[32..]),
            load(&block[64..]),
            load(&block[96..]),
        ]
    }

    // the eight sub-blocks' scales as i16 pairs and table shifts as bytes: i32 k of a lane is sub-block 2k + lane
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn iq4_kss_scales(codes: &[__m256i; 4], places: __m256i) -> (__m256i, __m256i) {
        // each code's low bit at its place; the madds' partial sums add up across a lane into its scale code
        let low = _mm256_set1_epi16(1);
        let m0 = _mm256_madd_epi16(_mm256_and_si256(codes[0], low), places);
        let m1 = _mm256_madd_epi16(_mm256_and_si256(codes[1], low), places);
        let m2 = _mm256_madd_epi16(_mm256_and_si256(codes[2], low), places);
        let m3 = _mm256_madd_epi16(_mm256_and_si256(codes[3], low), places);
        let ls = _mm256_hadd_epi32(_mm256_hadd_epi32(m0, m1), _mm256_hadd_epi32(m2, m3));
        let scale = _mm256_sub_epi32(
            _mm256_and_si256(ls, _mm256_set1_epi32(254)),
            _mm256_set1_epi32(127),
        );
        let scale = _mm256_or_si256(
            _mm256_and_si256(scale, _mm256_set1_epi32(0xffff)),
            _mm256_slli_epi32::<16>(scale),
        );
        let one = _mm256_set1_epi32(1);
        let shifted = _mm256_cmpeq_epi32(_mm256_and_si256(ls, one), one);
        (
            scale,
            _mm256_and_si256(shifted, _mm256_set1_epi8(IQ4_KSS_SHIFT)),
        )
    }

    // i32 `p` of each lane in all of that lane
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn lane_broadcast(v: __m256i, p: usize) -> __m256i {
        match p {
            0 => _mm256_shuffle_epi32::<0x00>(v),
            1 => _mm256_shuffle_epi32::<0x55>(v),
            2 => _mm256_shuffle_epi32::<0xaa>(v),
            _ => _mm256_shuffle_epi32::<0xff>(v),
        }
    }

    // pair `p`'s values (lane = sub-block): its low nibbles, then its high, and its scale
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn iq4_kss_pair(
        codes: &[__m256i; 4],
        (scales, shifts): (__m256i, __m256i),
        p: usize,
        values: __m256i,
    ) -> [__m256i; 3] {
        let a = _mm256_and_si256(codes[p], _mm256_set1_epi16(0xfffe_u16 as i16));
        let a = _mm256_xor_si256(a, _mm256_srli_epi16(a, 1));
        let m4 = _mm256_set1_epi8(0x0f);
        let shift = lane_broadcast(shifts, p);
        let lo = _mm256_add_epi8(_mm256_shuffle_epi8(values, _mm256_and_si256(a, m4)), shift);
        let hi = _mm256_add_epi8(
            _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(a, 4), m4)),
            shift,
        );
        [lo, hi, lane_broadcast(scales, p)]
    }

    // pair `p`'s activations laid out as `iq4_kss_pair` lays out its values
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn iq4_kss_q8(qs: &[i8], p: usize) -> [__m256i; 2] {
        let (a, b) = (load_q8(&qs[64 * p..]), load_q8(&qs[64 * p + 32..]));
        [
            _mm256_permute2x128_si256::<0x20>(a, b),
            _mm256_permute2x128_si256::<0x31>(a, b),
        ]
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn iq4_kss_dot([lo, hi, scale]: [__m256i; 3], [qlo, qhi]: [__m256i; 2]) -> __m256i {
        let plo = _mm256_maddubs_epi16(_mm256_sign_epi8(lo, lo), _mm256_sign_epi8(qlo, lo));
        let phi = _mm256_maddubs_epi16(_mm256_sign_epi8(hi, hi), _mm256_sign_epi8(qhi, hi));
        _mm256_add_epi32(_mm256_madd_epi16(plo, scale), _mm256_madd_epi16(phi, scale))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq4_kss_one(row: &[u8], ys: &[BlockQ8K]) -> f32 {
        let (values, places) = iq4_kss_consts();
        let mut acc = _mm256_setzero_ps();
        let blocks = row[IQ4_KSS_ROW_SCALE..].as_chunks::<IQ4_KSS_BYTES>().0;
        for (block, y) in blocks.iter().zip(ys) {
            prefetch_ahead(block, 2);
            let codes = iq4_kss_codes(block);
            let scales = iq4_kss_scales(&codes, places);
            let mut sumi = _mm256_setzero_si256();
            for p in 0..QK_K / 64 {
                let pair = iq4_kss_pair(&codes, scales, p, values);
                sumi = _mm256_add_epi32(sumi, iq4_kss_dot(pair, iq4_kss_q8(y.qs(), p)));
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d()), _mm256_cvtepi32_ps(sumi), acc);
        }
        row_scale(row) * hsum(acc)
    }

    // sub-block `ib32` of an IQ2_XXS block: its 32 unsigned grid values, their signs as +1 / -1 bytes, and its scale
    #[target_feature(enable = "avx2,fma")]
    fn iq2_xxs_sub(block: &[u8], ib32: usize) -> (__m256i, __m256i, __m256i) {
        let (grid, word) = iq2_xxs_words(block, ib32);
        let at = |l: usize| IQ2XXS_GRID[usize::from(grid[l])] as i64;
        let sign = |l: usize| IQ2_XXS_SIGNS[((word >> (7 * l)) & 127) as usize] as i64;
        (
            _mm256_set_epi64x(at(3), at(2), at(1), at(0)),
            _mm256_set_epi64x(sign(3), sign(2), sign(1), sign(0)),
            _mm256_set1_epi16(2 * (word >> 28) as i16 + 1),
        )
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq2_xxs_one(row: &[u8], ys: &[BlockQ8K]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        for (block, y) in row.as_chunks::<IQ2_XXS_BYTES>().0.iter().zip(ys) {
            prefetch_ahead(block, 1);
            let q8 = y.qs();
            let mut sumi = _mm256_setzero_si256();
            for ib32 in 0..QK_K / 32 {
                let (q2, signs, scale) = iq2_xxs_sub(block, ib32);
                let q8 = _mm256_sign_epi8(load_q8(&q8[32 * ib32..]), signs);
                sumi =
                    _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(q2, q8), scale));
            }
            let d = _mm256_set1_ps(f16_at(block, 0) * y.d());
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
        }
        IQ2_XXS_SCALE * hsum(acc)
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn decode_iq2_xxs(rows: &[u8], w: &mut [i16], d: &mut [f32]) {
        let blocks = rows.as_chunks::<IQ2_XXS_BYTES>().0;
        for ((block, w), d) in blocks.iter().zip(w.as_chunks_mut::<QK_K>().0).zip(d) {
            *d = IQ2_XXS_SCALE * f16_at(block, 0);
            for ib32 in 0..QK_K / 32 {
                let (q2, signs, scale) = iq2_xxs_sub(block, ib32);
                // grid values reach 43, so they stay positive as i8 and take their signs there
                let q2 = _mm256_sign_epi8(q2, signs);
                store_scaled(&mut w[32 * ib32..], _mm256_castsi256_si128(q2), scale);
                store_scaled(
                    &mut w[32 * ib32 + 16..],
                    _mm256_extracti128_si256(q2, 1),
                    scale,
                );
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq3_s_one(row: &[u8], ys: &[BlockQ8K]) -> f32 {
        let sign_bytes = load(&SIGN_BYTES);
        let sign_bits = load(&SIGN_BITS);
        // code k's ninth index bit is bit k of its sub-block's qh byte, shifted up to 256
        let qh_shift = _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1);
        let qh_bit = _mm256_set1_epi32(256);
        let mut acc = _mm256_setzero_ps();
        for (block, y) in row.as_chunks::<IQ3_S_BYTES>().0.iter().zip(ys) {
            prefetch_ahead(block, 2);
            let q8 = y.qs();
            let mut sumi = _mm256_setzero_si256();
            for ib32 in 0..QK_K / 32 {
                // SAFETY: the sub-block's 8 grid bytes lie inside the block
                let qs = unsafe { _mm_loadl_epi64(block[IQ3_S_QS + 8 * ib32..].as_ptr().cast()) };
                let qh = _mm256_set1_epi32(i32::from(block[IQ3_S_QH + ib32]));
                let idx = _mm256_or_si256(
                    _mm256_cvtepu8_epi32(qs),
                    _mm256_and_si256(_mm256_sllv_epi32(qh, qh_shift), qh_bit),
                );
                // SAFETY: every index is below 512, the grid's length
                let q3 = unsafe { _mm256_i32gather_epi32::<4>(IQ3S_GRID.as_ptr().cast(), idx) };
                let signs: &[u8; 4] = block[IQ3_S_SIGNS + 4 * ib32..][..4]
                    .try_into()
                    .expect("4 sign bytes");
                let spread = _mm256_and_si256(
                    _mm256_shuffle_epi8(
                        _mm256_set1_epi32(u32::from_le_bytes(*signs) as i32),
                        sign_bytes,
                    ),
                    sign_bits,
                );
                let negate = _mm256_cmpeq_epi8(spread, sign_bits);
                let q8 = load_q8(&q8[32 * ib32..]);
                let q8 = _mm256_sub_epi8(_mm256_xor_si256(negate, q8), negate);
                let scale = block[IQ3_S_SCALES + ib32 / 2] >> (4 * (ib32 % 2)) & 0xf;
                let scale = _mm256_set1_epi16(2 * i16::from(scale) + 1);
                let p16 = _mm256_maddubs_epi16(q3, q8);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, scale));
            }
            let d = _mm256_set1_ps(f16_at(block, 0) * y.d());
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum(acc)
    }

    #[target_feature(enable = "avx2,fma")]
    fn store_scaled(w: &mut [i16], values: __m128i, scale: __m256i) {
        assert!(w.len() >= 16);
        let v = _mm256_mullo_epi16(_mm256_cvtepi8_epi16(values), scale);
        // SAFETY: 16 writable i16 were checked above
        unsafe { _mm256_storeu_si256(w.as_mut_ptr().cast(), v) };
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn decode_iq4_xs(rows: &[u8], w: &mut [i16], d: &mut [f32]) {
        // SAFETY: the codebook holds 16 bytes
        let values = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) };
        let m4 = _mm_set1_epi8(0x0f);
        let blocks = rows.as_chunks::<IQ4_XS_BYTES>().0;
        for ((block, w), d) in blocks.iter().zip(w.as_chunks_mut::<QK_K>().0).zip(d) {
            *d = f16_at(block, 0);
            for ib in 0..QK_K / 32 {
                // SAFETY: a sub-block's 16 nibble bytes lie inside the block
                let bits = unsafe { _mm_loadu_si128(block[IQ4_XS_QS + 16 * ib..].as_ptr().cast()) };
                let scale = _mm256_set1_epi16(iq4_xs_sub_scale(block, ib) as i16);
                let lo = _mm_shuffle_epi8(values, _mm_and_si128(bits, m4));
                let hi = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4));
                store_scaled(&mut w[32 * ib..], lo, scale);
                store_scaled(&mut w[32 * ib + 16..], hi, scale);
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn decode_iq4_kss(
        rows: &[u8],
        row_bytes: usize,
        cols: usize,
        w: &mut [i16],
        d: &mut [f32],
    ) {
        let (values, places) = iq4_kss_consts();
        for ((row, w), d) in rows
            .chunks_exact(row_bytes)
            .zip(w.chunks_exact_mut(cols))
            .zip(d.chunks_exact_mut(cols / QK_K))
        {
            let scale = row_scale(row);
            let blocks = row[IQ4_KSS_ROW_SCALE..].as_chunks::<IQ4_KSS_BYTES>().0;
            for ((block, w), d) in blocks.iter().zip(w.as_chunks_mut::<QK_K>().0).zip(d) {
                *d = scale;
                let codes = iq4_kss_codes(block);
                let scales = iq4_kss_scales(&codes, places);
                for p in 0..QK_K / 64 {
                    let [lo, hi, sub] = iq4_kss_pair(&codes, scales, p, values);
                    let s0 = _mm256_permute2x128_si256::<0x00>(sub, sub);
                    let s1 = _mm256_permute2x128_si256::<0x11>(sub, sub);
                    let w = &mut w[64 * p..];
                    store_scaled(w, _mm256_castsi256_si128(lo), s0);
                    store_scaled(&mut w[16..], _mm256_castsi256_si128(hi), s0);
                    store_scaled(&mut w[32..], _mm256_extracti128_si256::<1>(lo), s1);
                    store_scaled(&mut w[48..], _mm256_extracti128_si256::<1>(hi), s1);
                }
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn decode_iq3_s(rows: &[u8], w: &mut [i16], d: &mut [f32]) {
        let sign_bytes = load(&SIGN_BYTES);
        let sign_bits = load(&SIGN_BITS);
        let qh_shifts = _mm256_setr_epi32(8, 7, 6, 5, 4, 3, 2, 1);
        let bit8 = _mm256_set1_epi32(256);
        let blocks = rows.as_chunks::<IQ3_S_BYTES>().0;
        for ((block, w), d) in blocks.iter().zip(w.as_chunks_mut::<QK_K>().0).zip(d) {
            *d = f16_at(block, 0);
            for ib32 in 0..QK_K / 32 {
                // the 9-bit grid indices: 8 qs bytes, each with its qh bit as bit 8
                // SAFETY: the sub-block's 8 qs bytes lie inside the block
                let qs = unsafe { _mm_loadl_epi64(block[IQ3_S_QS + 8 * ib32..].as_ptr().cast()) };
                let qh = _mm256_set1_epi32(i32::from(block[IQ3_S_QH + ib32]));
                let index = _mm256_or_si256(
                    _mm256_cvtepu8_epi32(qs),
                    _mm256_and_si256(_mm256_sllv_epi32(qh, qh_shifts), bit8),
                );
                // SAFETY: every index is below 512, the grid's length
                let q3 = unsafe { _mm256_i32gather_epi32::<4>(IQ3S_GRID.as_ptr().cast(), index) };
                let signs: &[u8; 4] = block[IQ3_S_SIGNS + 4 * ib32..][..4]
                    .try_into()
                    .expect("4 sign bytes");
                let spread = _mm256_and_si256(
                    _mm256_shuffle_epi8(
                        _mm256_set1_epi32(u32::from_le_bytes(*signs) as i32),
                        sign_bytes,
                    ),
                    sign_bits,
                );
                let negate = _mm256_cmpeq_epi8(spread, sign_bits);
                let q3 = _mm256_sub_epi8(_mm256_xor_si256(negate, q3), negate);
                let scale = block[IQ3_S_SCALES + ib32 / 2] >> (4 * (ib32 % 2)) & 0xf;
                let scale = _mm256_set1_epi16(2 * i16::from(scale) + 1);
                store_scaled(&mut w[32 * ib32..], _mm256_castsi256_si128(q3), scale);
                store_scaled(
                    &mut w[32 * ib32 + 16..],
                    _mm256_extracti128_si256(q3, 1),
                    scale,
                );
            }
        }
    }

    // two decoded rows against four tokens: each activation load feeds both rows' madds
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::needless_range_loop)]
    unsafe fn dot_2x4(w: [*const i16; 2], d: [&[f32]; 2], tokens: [&TileRow; 4]) -> [[f32; 4]; 2] {
        let blocks = d[0].len();
        let mut facc = [_mm256_setzero_ps(); 8];
        for b in 0..blocks {
            let mut acc = [_mm256_setzero_si256(); 8];
            for s in 0..QK_K / 16 {
                let o = b * QK_K + 16 * s;
                // SAFETY: the caller's rows and tokens hold `blocks` whole blocks
                let (w0, w1) = unsafe {
                    (
                        _mm256_loadu_si256(w[0].add(o).cast()),
                        _mm256_loadu_si256(w[1].add(o).cast()),
                    )
                };
                for t in 0..4 {
                    // SAFETY: as above
                    let x = unsafe { _mm256_loadu_si256(tokens[t].qs.as_ptr().add(o).cast()) };
                    acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(w0, x));
                    acc[4 + t] = _mm256_add_epi32(acc[4 + t], _mm256_madd_epi16(w1, x));
                }
            }
            for t in 0..4 {
                let yd = tokens[t].blocks[b].d();
                for r in 0..2 {
                    let i = 4 * r + t;
                    facc[i] = _mm256_fmadd_ps(
                        _mm256_set1_ps(d[r][b] * yd),
                        _mm256_cvtepi32_ps(acc[i]),
                        facc[i],
                    );
                }
            }
        }
        [
            [hsum(facc[0]), hsum(facc[1]), hsum(facc[2]), hsum(facc[3])],
            [hsum(facc[4]), hsum(facc[5]), hsum(facc[6]), hsum(facc[7])],
        ]
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn tile_dot(w: &[i16], d: &[f32], tokens: &[TileRow], out: &mut [f32]) {
        let blocks = tokens[0].blocks.len();
        let n = tokens.len();
        for (pair, (w, d)) in w
            .chunks_exact(2 * blocks * QK_K)
            .zip(d.chunks_exact(2 * blocks))
            .enumerate()
        {
            let rows = [w.as_ptr(), w[blocks * QK_K..].as_ptr()];
            for t0 in (0..n).step_by(4) {
                // a short last quad repeats its final token and drops those results
                let quad = std::array::from_fn(|k| &tokens[(t0 + k).min(n - 1)]);
                // SAFETY: tile_dot's caller checked every row and token holds `blocks` blocks
                let ys = unsafe { dot_2x4(rows, [&d[..blocks], &d[blocks..]], quad) };
                for (r, ys) in ys.iter().enumerate() {
                    let row = 2 * pair + r;
                    for (k, &y) in ys.iter().enumerate().take(n - t0) {
                        out[row * n + t0 + k] = y;
                    }
                }
            }
        }
    }

    // four activation rows at once: the weight sub-block is decoded once and each row adds one maddubs
    macro_rules! four_rows {
        ($acc:ident, $sumi:ident, $yb:ident, $q8_at:expr, $dot:expr) => {
            for t in 0..4 {
                let q8 = load_q8(&$yb[t].qs()[$q8_at..]);
                $sumi[t] = _mm256_add_epi32($sumi[t], $dot(q8));
            }
        };
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq4_xs_x4(row: &[u8], ys: [&[BlockQ8K]; 4]) -> [f32; 4] {
        // SAFETY: the codebook holds 16 bytes
        let values = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) };
        let m4 = _mm_set1_epi8(0x0f);
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        for (i, block) in row.as_chunks::<IQ4_XS_BYTES>().0.iter().enumerate() {
            let yb = [&ys[0][i], &ys[1][i], &ys[2][i], &ys[3][i]];
            let mut sumi = [_mm256_setzero_si256(); 4];
            for ib in 0..QK_K / 32 {
                // SAFETY: a sub-block's 16 nibble bytes lie inside the block
                let bits = unsafe { _mm_loadu_si128(block[IQ4_XS_QS + 16 * ib..].as_ptr().cast()) };
                let q4 = _mm256_set_m128i(
                    _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                    _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)),
                );
                let abs_q4 = _mm256_sign_epi8(q4, q4);
                let scale = _mm256_set1_epi16(iq4_xs_sub_scale(block, ib) as i16);
                four_rows!(a, sumi, yb, 32 * ib, |q8| _mm256_madd_epi16(
                    _mm256_maddubs_epi16(abs_q4, _mm256_sign_epi8(q8, q4)),
                    scale
                ));
            }
            let d = f16_at(block, 0);
            a0 = _mm256_fmadd_ps(
                _mm256_set1_ps(d * yb[0].d()),
                _mm256_cvtepi32_ps(sumi[0]),
                a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_set1_ps(d * yb[1].d()),
                _mm256_cvtepi32_ps(sumi[1]),
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_set1_ps(d * yb[2].d()),
                _mm256_cvtepi32_ps(sumi[2]),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_set1_ps(d * yb[3].d()),
                _mm256_cvtepi32_ps(sumi[3]),
                a3,
            );
        }
        [hsum(a0), hsum(a1), hsum(a2), hsum(a3)]
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq4_kss_x4(row: &[u8], ys: [&[BlockQ8K]; 4]) -> [f32; 4] {
        let (values, places) = iq4_kss_consts();
        let mut acc = [_mm256_setzero_ps(); 4];
        let blocks = row[IQ4_KSS_ROW_SCALE..].as_chunks::<IQ4_KSS_BYTES>().0;
        for (i, block) in blocks.iter().enumerate() {
            let yb = [&ys[0][i], &ys[1][i], &ys[2][i], &ys[3][i]];
            let codes = iq4_kss_codes(block);
            let scales = iq4_kss_scales(&codes, places);
            let mut sumi = [_mm256_setzero_si256(); 4];
            for p in 0..QK_K / 64 {
                // the weights are reordered once into whole sub-blocks, so the four rows load as they lie
                let [lo, hi, scale] = iq4_kss_pair(&codes, scales, p, values);
                let q0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
                let q1 = _mm256_permute2x128_si256::<0x31>(lo, hi);
                let s0 = _mm256_permute2x128_si256::<0x00>(scale, scale);
                let s1 = _mm256_permute2x128_si256::<0x11>(scale, scale);
                let (a0, a1) = (_mm256_sign_epi8(q0, q0), _mm256_sign_epi8(q1, q1));
                for (sumi, y) in sumi.iter_mut().zip(yb) {
                    let x0 = load_q8(&y.qs()[64 * p..]);
                    let x1 = load_q8(&y.qs()[64 * p + 32..]);
                    let p0 =
                        _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(x0, q0)), s0);
                    let p1 =
                        _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(x1, q1)), s1);
                    *sumi = _mm256_add_epi32(*sumi, _mm256_add_epi32(p0, p1));
                }
            }
            for ((acc, sumi), y) in acc.iter_mut().zip(sumi).zip(yb) {
                *acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d()), _mm256_cvtepi32_ps(sumi), *acc);
            }
        }
        let d = row_scale(row);
        acc.map(|a| d * hsum(a))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq2_xxs_x4(row: &[u8], ys: [&[BlockQ8K]; 4]) -> [f32; 4] {
        let mut acc = [_mm256_setzero_ps(); 4];
        for (i, block) in row.as_chunks::<IQ2_XXS_BYTES>().0.iter().enumerate() {
            prefetch_ahead(block, 1);
            let yb = [&ys[0][i], &ys[1][i], &ys[2][i], &ys[3][i]];
            let mut sumi = [_mm256_setzero_si256(); 4];
            for ib32 in 0..QK_K / 32 {
                let (q2, signs, scale) = iq2_xxs_sub(block, ib32);
                four_rows!(a, sumi, yb, 32 * ib32, |q8| _mm256_madd_epi16(
                    _mm256_maddubs_epi16(q2, _mm256_sign_epi8(q8, signs)),
                    scale
                ));
            }
            let d = f16_at(block, 0);
            for t in 0..4 {
                acc[t] = _mm256_fmadd_ps(
                    _mm256_set1_ps(d * yb[t].d()),
                    _mm256_cvtepi32_ps(sumi[t]),
                    acc[t],
                );
            }
        }
        acc.map(|a| IQ2_XXS_SCALE * hsum(a))
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq3_s_x4(row: &[u8], ys: [&[BlockQ8K]; 4]) -> [f32; 4] {
        let sign_bytes = load(&SIGN_BYTES);
        let sign_bits = load(&SIGN_BITS);
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        for (i, block) in row.as_chunks::<IQ3_S_BYTES>().0.iter().enumerate() {
            let yb = [&ys[0][i], &ys[1][i], &ys[2][i], &ys[3][i]];
            let mut sumi = [_mm256_setzero_si256(); 4];
            for ib32 in 0..QK_K / 32 {
                let qs: &[u8; 8] = block[IQ3_S_QS + 8 * ib32..][..8]
                    .try_into()
                    .expect("8 grid bytes");
                let qh = u32::from(block[IQ3_S_QH + ib32]);
                // the index is 9 bits: a qs byte plus one qh bit
                let grid = |k: usize| {
                    IQ3S_GRID[(u32::from(qs[k]) | ((qh << (8 - k)) & 256)) as usize & 511] as i32
                };
                let q3 = _mm256_set_epi32(
                    grid(7),
                    grid(6),
                    grid(5),
                    grid(4),
                    grid(3),
                    grid(2),
                    grid(1),
                    grid(0),
                );
                let signs: &[u8; 4] = block[IQ3_S_SIGNS + 4 * ib32..][..4]
                    .try_into()
                    .expect("4 sign bytes");
                let spread = _mm256_and_si256(
                    _mm256_shuffle_epi8(
                        _mm256_set1_epi32(u32::from_le_bytes(*signs) as i32),
                        sign_bytes,
                    ),
                    sign_bits,
                );
                let negate = _mm256_cmpeq_epi8(spread, sign_bits);
                let scale = block[IQ3_S_SCALES + ib32 / 2] >> (4 * (ib32 % 2)) & 0xf;
                let scale = _mm256_set1_epi16(2 * i16::from(scale) + 1);
                four_rows!(a, sumi, yb, 32 * ib32, |q8| _mm256_madd_epi16(
                    _mm256_maddubs_epi16(q3, _mm256_sub_epi8(_mm256_xor_si256(negate, q8), negate)),
                    scale
                ));
            }
            let d = f16_at(block, 0);
            a0 = _mm256_fmadd_ps(
                _mm256_set1_ps(d * yb[0].d()),
                _mm256_cvtepi32_ps(sumi[0]),
                a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_set1_ps(d * yb[1].d()),
                _mm256_cvtepi32_ps(sumi[1]),
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_set1_ps(d * yb[2].d()),
                _mm256_cvtepi32_ps(sumi[2]),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_set1_ps(d * yb[3].d()),
                _mm256_cvtepi32_ps(sumi[3]),
                a3,
            );
        }
        [hsum(a0), hsum(a1), hsum(a2), hsum(a3)]
    }
}

#[cfg(test)]
mod tests {
    use inference_tensor::quantized::GgmlType;
    use inference_tensor::quantized::k_quants::{BlockQ8K, QK_K};

    use super::*;

    const BLOCKS: usize = 8;

    fn bytes(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    // random blocks with sane f16 scales in front
    fn row(block_bytes: usize, seed: u32) -> Vec<u8> {
        let mut row = bytes(block_bytes * BLOCKS, seed);
        for (i, block) in row.chunks_exact_mut(block_bytes).enumerate() {
            let d = half::f16::from_f32(0.01 + 0.002 * i as f32).to_le_bytes();
            block[..2].copy_from_slice(&d);
        }
        row
    }

    // random blocks behind a sane f32 row scale
    fn row_scaled(block_bytes: usize, seed: u32) -> Vec<u8> {
        let mut row = 0.003f32.to_le_bytes().to_vec();
        row.extend(bytes(block_bytes * BLOCKS, seed));
        row
    }

    fn activations() -> Vec<BlockQ8K> {
        let xs = (0..BLOCKS * QK_K)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect::<Vec<_>>();
        let mut ys = vec![BlockQ8K::zeros(); BLOCKS];
        BlockQ8K::from_float(&xs, &mut ys);
        ys
    }

    fn dequantized_dot(ty: GgufType, row: &[u8], ys: &[BlockQ8K]) -> f32 {
        let weights = super::super::raw::dequantize_rows(ty, BLOCKS * QK_K, row).unwrap();
        let mut xs = vec![0f32; BLOCKS * QK_K];
        BlockQ8K::to_float(ys, &mut xs);
        weights.iter().zip(&xs).map(|(w, x)| w * x).sum()
    }

    #[test]
    fn iq_dots_match_dequantized_rows() {
        let ys = activations();
        for (ty, block_bytes) in [
            (GgufType::Iq4Xs, IQ4_XS_BYTES),
            (GgufType::Iq3S, IQ3_S_BYTES),
            (GgufType::Iq4Kss, IQ4_KSS_BYTES),
            (GgufType::Iq2Xxs, IQ2_XXS_BYTES),
        ] {
            for seed in 0..4 {
                let row = match ty {
                    GgufType::Iq4Kss => row_scaled(block_bytes, seed),
                    _ => row(block_bytes, seed),
                };
                let expected = dequantized_dot(ty, &row, &ys);
                let generic = match ty {
                    GgufType::Iq4Xs => iq4_xs_generic(&row, &ys),
                    GgufType::Iq4Kss => iq4_kss_generic(&row, &ys),
                    GgufType::Iq2Xxs => iq2_xxs_generic(&row, &ys),
                    _ => iq3_s_generic(&row, &ys),
                };
                let tolerance = 1e-3 * expected.abs().max(1.0);
                assert!(
                    (generic - expected).abs() < tolerance,
                    "{ty:?} generic {generic} vs {expected}"
                );
                let dispatched = vec_dot(ty, &row, &ys);
                assert!(
                    (dispatched - expected).abs() < tolerance,
                    "{ty:?} {dispatched} vs {expected}"
                );
                let [many, ..] = vec_dot_x4(ty, &row, [&ys[..]; 4]);
                assert!(
                    (many - dispatched).abs() < 1e-5 * expected.abs().max(1.0),
                    "{ty:?} {many}"
                );
            }
        }
    }

    #[test]
    fn block32_dots_match_the_generic_kernels() {
        let xs = (0..BLOCKS * QK32)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect::<Vec<_>>();
        let mut ys = vec![Q8Block32::ZERO; BLOCKS];
        Q8Block32::quantize(&xs, &mut ys);
        for (ty, block_bytes) in [
            (GgufType::Iq4Nl, IQ4_NL_BYTES),
            (GgufType::Q4_0, Q4_0_BYTES),
            (GgufType::Q5_0, Q5_0_BYTES),
            (GgufType::Q8_0, Q8_0_BYTES),
        ] {
            // an odd block count too, past the kernels' pairs
            for (seed, n) in (0..4).flat_map(|seed| [(seed, BLOCKS), (seed, BLOCKS - 1)]) {
                let row = &row(block_bytes, seed)[..n * block_bytes];
                let expected = vec_dot_32_generic(ty, row, &ys[..n]);
                let dispatched = vec_dot_32(ty, row, &ys[..n]);
                assert!(
                    (dispatched - expected).abs() < 1e-4 * expected.abs().max(1.0),
                    "{ty:?} {n} blocks: {dispatched} vs {expected}"
                );
                // four rows at once score each row as it scores alone, bit for bit
                let quad = vec_dot_32_x4(ty, row, [&ys[..n]; 4]);
                assert!(
                    quad.iter().all(|q| q.to_bits() == dispatched.to_bits()),
                    "{ty:?} {n} blocks: {quad:?}"
                );
                // eight rows a lane each, each its own seed, within rounding of the one-row kernel
                let wide_rows: Vec<Vec<u8>> = (0..8)
                    .map(|i| self::row(block_bytes, seed + 10 * i))
                    .collect();
                let rows: [&[u8]; 8] = std::array::from_fn(|i| &wide_rows[i][..n * block_bytes]);
                let wide = vec_dot_32_8x4(ty, rows, [&ys[..n]; 4]);
                // one activation row scores as the four-row tile's lanes do, bit for bit
                let tall = vec_dot_32_8x1(ty, rows, &ys[..n]);
                assert_eq!(
                    tall.map(f32::to_bits),
                    wide[0].map(f32::to_bits),
                    "{ty:?} {n} blocks"
                );
                for (i, row) in rows.iter().enumerate() {
                    let one = vec_dot_32(ty, row, &ys[..n]);
                    for got in wide.map(|t| t[i]) {
                        assert!(
                            (got - one).abs() <= 1e-4 * one.abs().max(1.0),
                            "{ty:?} {n} blocks, wide row {i}: {got} vs {one}"
                        );
                    }
                }
            }
        }
    }

    // a decoded tile dotted against the activations matches the row dot, two rows so the pair kernel runs
    #[test]
    fn tiles_match_row_dots() {
        let ys = activations();
        let mut qs = vec![0i16; BLOCKS * QK_K];
        widen_into(&ys, &mut qs);
        let token = TileRow {
            qs: &qs,
            blocks: &ys,
        };
        for (ty, block_bytes) in [
            (GgufType::Iq4Xs, IQ4_XS_BYTES),
            (GgufType::Iq3S, IQ3_S_BYTES),
            (GgufType::Iq4Kss, IQ4_KSS_BYTES),
            (GgufType::Iq2Xxs, IQ2_XXS_BYTES),
        ] {
            if !tile_supports(ty) {
                continue;
            }
            let rows = [0, 1].map(|seed| match ty {
                GgufType::Iq4Kss => row_scaled(block_bytes, seed),
                _ => row(block_bytes, seed),
            });
            let (mut w, mut d) = (vec![0i16; 2 * BLOCKS * QK_K], vec![0f32; 2 * BLOCKS]);
            decode_tile(ty, &rows.concat(), BLOCKS * QK_K, &mut w, &mut d);
            let mut out = [0f32; 2];
            tile_dot(&w, &d, &[token], &mut out);
            for (row, out) in rows.iter().zip(out) {
                let expected = vec_dot(ty, row, &ys);
                assert!(
                    (out - expected).abs() < 1e-4 * expected.abs().max(1.0),
                    "{ty:?} tile {out} vs {expected}"
                );
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use inference_tensor::quantized::GgmlType;
    #[test]
    #[ignore = "benchmark"]
    fn single_thread_rate() {
        for (ty, size) in [
            (GgufType::Iq2Xxs, IQ2_XXS_BYTES),
            (GgufType::Iq3S, IQ3_S_BYTES),
            (GgufType::Iq4Xs, IQ4_XS_BYTES),
            (GgufType::Iq4Kss, IQ4_KSS_BYTES),
        ] {
            let rows = 512;
            let row_bytes = ty.row_scale_bytes() + 8 * size;
            let bytes = vec![3u8; rows * row_bytes];
            let xs = vec![0.5f32; 8 * QK_K];
            let mut ys = vec![BlockQ8K::zeros(); 8];
            BlockQ8K::from_float(&xs, &mut ys);
            let start = std::time::Instant::now();
            let mut acc = 0f32;
            for _ in 0..200 {
                for row in bytes.chunks_exact(row_bytes) {
                    acc += vec_dot(ty, row, &ys);
                }
            }
            let secs = start.elapsed().as_secs_f64();
            let start4 = std::time::Instant::now();
            for _ in 0..200 {
                for row in bytes.chunks_exact(row_bytes) {
                    acc += vec_dot_x4(ty, row, [&ys[..]; 4])[3];
                }
            }
            println!(
                "{ty:?} x4: {:.2} GB/s per token one thread",
                (4 * 200 * bytes.len()) as f64 / start4.elapsed().as_secs_f64() / 1e9
            );
            println!(
                "{ty:?}: {:.2} GB/s one thread ({acc})",
                (200 * bytes.len()) as f64 / secs / 1e9
            );
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn single_thread_rate_32() {
        for (ty, size) in [
            (GgufType::Iq4Nl, IQ4_NL_BYTES),
            (GgufType::Q8_0, Q8_0_BYTES),
        ] {
            let (rows, blocks) = (2048, 20);
            let row_bytes = blocks * size;
            let bytes = vec![3u8; rows * row_bytes];
            let xs = vec![0.5f32; blocks * QK32];
            let mut ys = vec![Q8Block32::ZERO; blocks];
            Q8Block32::quantize(&xs, &mut ys);
            let start = std::time::Instant::now();
            let mut acc = 0f32;
            for _ in 0..200 {
                for row in bytes.chunks_exact(row_bytes) {
                    acc += vec_dot_32(ty, row, &ys);
                }
            }
            println!(
                "{ty:?} 32: {:.2} GB/s one thread ({acc})",
                (200 * bytes.len()) as f64 / start.elapsed().as_secs_f64() / 1e9
            );
        }
    }

    // eight rows by 32 tokens, as the experts' items run them: a row at a time, or eight rows a lane each
    #[test]
    #[ignore = "benchmark"]
    fn tile_rate_32() {
        const COLS: usize = 2688;
        const TOKENS: usize = 32;
        const ITERS: usize = 2_000;
        let blocks = COLS / QK32;
        let xs = (0..TOKENS * COLS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect::<Vec<_>>();
        let mut ys = vec![Q8Block32::ZERO; TOKENS * blocks];
        Q8Block32::quantize(&xs, &mut ys);
        let y = |t: usize| &ys[t * blocks..][..blocks];
        for (ty, size) in [
            (GgufType::Iq4Nl, IQ4_NL_BYTES),
            (GgufType::Q4_0, Q4_0_BYTES),
            (GgufType::Q5_0, Q5_0_BYTES),
            (GgufType::Q8_0, Q8_0_BYTES),
        ] {
            let bytes = (0..8 * blocks * size)
                .map(|i| (i * 7 % 251) as u8)
                .collect::<Vec<_>>();
            let row = |i: usize| &bytes[i * blocks * size..][..blocks * size];
            let rate = |label: &str, run: &dyn Fn() -> f32| {
                let start = std::time::Instant::now();
                let mut acc = 0f32;
                for _ in 0..ITERS {
                    acc += run();
                }
                let secs = start.elapsed().as_secs_f64() / ITERS as f64;
                println!(
                    "{ty:?} {label}: {:.1} GMAC/s one thread ({acc})",
                    (8 * COLS * TOKENS) as f64 / secs / 1e9
                );
            };
            rate("1x4", &|| {
                let mut acc = 0f32;
                for r in 0..8 {
                    for t in (0..TOKENS).step_by(4) {
                        acc += vec_dot_32_x4(ty, row(r), std::array::from_fn(|k| y(t + k)))[0];
                    }
                }
                acc
            });
            rate("8x4", &|| {
                let mut acc = 0f32;
                let rows = std::array::from_fn(row);
                for t in (0..TOKENS).step_by(4) {
                    let out = vec_dot_32_8x4(ty, rows, std::array::from_fn(|k| y(t + k)));
                    acc += out[0][0];
                }
                acc
            });
        }
    }

    // one work item's tile: decode 8 rows, then dot them against 16 tokens
    #[test]
    #[ignore = "benchmark"]
    fn tile_rate() {
        const ROWS: usize = 8;
        const TOKENS: usize = 16;
        const ITERS: usize = 20_000;
        for (ty, size, cols) in [
            (GgufType::Iq2Xxs, IQ2_XXS_BYTES, 1024),
            (GgufType::Iq3S, IQ3_S_BYTES, 2048),
            (GgufType::Iq4Xs, IQ4_XS_BYTES, 512),
            (GgufType::Iq4Kss, IQ4_KSS_BYTES, 2048),
        ] {
            let blocks = cols / QK_K;
            let bytes = (0..ROWS * (ty.row_scale_bytes() + blocks * size))
                .map(|i| (i * 7 % 251) as u8)
                .collect::<Vec<_>>();
            let (mut w, mut d) = (vec![0i16; ROWS * cols], vec![0f32; ROWS * blocks]);
            let start = std::time::Instant::now();
            for _ in 0..ITERS {
                decode_tile(ty, std::hint::black_box(&bytes), cols, &mut w, &mut d);
            }
            let decode = start.elapsed().as_secs_f64() / ITERS as f64;
            let mut q8 = vec![BlockQ8K::zeros(); TOKENS * blocks];
            BlockQ8K::from_float(&vec![0.3f32; TOKENS * cols], &mut q8);
            let mut q16 = vec![0i16; q8.len() * QK_K];
            widen_into(&q8, &mut q16);
            let tokens = (0..TOKENS)
                .map(|t| TileRow {
                    qs: &q16[t * cols..][..cols],
                    blocks: &q8[t * blocks..][..blocks],
                })
                .collect::<Vec<_>>();
            let mut out = vec![0f32; ROWS * TOKENS];
            let start = std::time::Instant::now();
            for _ in 0..ITERS {
                tile_dot(&w, &d, std::hint::black_box(&tokens), &mut out);
            }
            let dot = start.elapsed().as_secs_f64() / ITERS as f64;
            println!(
                "{ty:?}: decode {:.2} us, dot {:.2} us ({:.1} GMAC/s one thread)",
                decode * 1e6,
                dot * 1e6,
                (ROWS * cols * TOKENS) as f64 / dot / 1e9
            );
        }
    }
}
