//! One GGUF row dotted with Q8_K activations on the CPU, from ggml-cpu's `ggml_vec_dot_*_q8_K` (generic and AVX2).

use inference_tensor::quantized::GgmlType;
use inference_tensor::quantized::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, QK_K,
};

use super::iq_tables::{IQ3S_GRID, KMASK_IQ2XS};
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
const IQ4_KSS_BYTES: usize = 128;
const IQ4_KSS_ROW_SCALE: usize = 4;
// ik's shifted IQ4_K table is the IQ4_NL one plus this, picked by a sub-block scale code's low bit
const IQ4_KSS_SHIFT: i8 = 4;

/// Whether `vec_dot` serves this type.
pub(crate) fn supports(ty: GgufType) -> bool {
    matches!(
        ty,
        GgufType::Iq3S
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
const Q8_0_BYTES: usize = 34;
// both blocks open with their f16 scale
const BLOCK32_QS: usize = 2;
const Q8_MAX: f32 = 127.0;
// Bytes ahead of the block being dotted that a single-token dot prefetches: expert rows stream from DRAM, and the
// hardware prefetcher alone leaves the dots waiting on them
#[cfg(target_arch = "x86_64")]
const PREFETCH_AHEAD: usize = 1024;

/// Whether `vec_dot_32` serves this type.
pub(crate) fn supports_32(ty: GgufType) -> bool {
    matches!(ty, GgufType::Iq4Nl | GgufType::Q8_0)
}

/// 32 activations quantized to i8 under one scale, as ggml's Q8_0 with an f32 scale.
#[derive(Clone, Copy)]
pub(crate) struct Q8Block32 {
    d: f32,
    qs: [i8; QK32],
}

impl Q8Block32 {
    pub(crate) const ZERO: Self = Self {
        d: 0.0,
        qs: [0; QK32],
    };

    /// `xs` in blocks of 32, into `ys`.
    pub(crate) fn quantize(xs: &[f32], ys: &mut [Self]) {
        for (x, y) in xs.as_chunks::<QK32>().0.iter().zip(ys) {
            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / Q8_MAX;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            y.d = d;
            for (q, v) in y.qs.iter_mut().zip(x) {
                // within [-127, 127] by the scale's choice
                *q = (v * id).round() as i8;
            }
        }
    }
}

/// `row` holds `ys.len()` blocks of a 32-wide `ty`.
pub(crate) fn vec_dot_32(ty: GgufType, row: &[u8], ys: &[Q8Block32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match ty {
            // SAFETY: AVX2 and FMA were detected above
            GgufType::Iq4Nl => return unsafe { avx2::iq4_nl_32(row, ys) },
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
        return matches!(ty, GgufType::Iq3S | GgufType::Iq4Xs | GgufType::Iq4Kss);
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
        BLOCK32_QS, IQ3_S_BYTES, IQ3_S_QH, IQ3_S_QS, IQ3_S_SCALES, IQ3_S_SIGNS, IQ3S_GRID,
        IQ4_KSS_BYTES, IQ4_KSS_ROW_SCALE, IQ4_KSS_SHIFT, IQ4_NL_BYTES, IQ4_XS_BYTES, IQ4_XS_QS,
        KVALUES_IQ4NL, PREFETCH_AHEAD, Q8_0_BYTES, Q8Block32, TileRow, f16_at, iq4_xs_sub_scale,
        row_scale,
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

    // sum over blocks of `block_dot(block, y)` scaled by the block's and y's scales, on two accumulators so
    // consecutive blocks' multiply-adds overlap
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    fn sum_blocks_32<const N: usize>(
        row: &[u8],
        ys: &[Q8Block32],
        block_dot: impl Fn(&[u8; N], &Q8Block32) -> __m256i,
    ) -> f32 {
        let scaled = |block: &[u8; N], y: &Q8Block32, acc: __m256| {
            prefetch_ahead(block, 1);
            let d = _mm256_set1_ps(f16c_at(block, 0) * y.d);
            _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(block_dot(block, y)), acc)
        };
        let blocks = &row.as_chunks::<N>().0[..ys.len()];
        let (pairs, rest) = blocks.as_chunks::<2>();
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        for ([b0, b1], y) in pairs.iter().zip(ys.as_chunks::<2>().0) {
            a0 = scaled(b0, &y[0], a0);
            a1 = scaled(b1, &y[1], a1);
        }
        if let [b] = rest {
            a0 = scaled(b, &ys[ys.len() - 1], a0);
        }
        hsum(_mm256_add_ps(a0, a1))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) fn iq4_nl_32(row: &[u8], ys: &[Q8Block32]) -> f32 {
        // SAFETY: the codebook holds 16 bytes
        let values = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) };
        let m4 = _mm_set1_epi8(0x0f);
        sum_blocks_32::<IQ4_NL_BYTES>(row, ys, |block, y| {
            // SAFETY: the 16 nibble bytes follow the block's scale
            let bits = unsafe { _mm_loadu_si128(block[BLOCK32_QS..].as_ptr().cast()) };
            // low nibbles pair with the first 16 activations, high ones with the last 16
            let q4 = _mm256_set_m128i(
                _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)),
            );
            signed_dot(q4, load_q8(&y.qs))
        })
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) fn q8_0_32(row: &[u8], ys: &[Q8Block32]) -> f32 {
        sum_blocks_32::<Q8_0_BYTES>(row, ys, |block, y| {
            signed_dot(load(&block[BLOCK32_QS..]), load_q8(&y.qs))
        })
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

    // one work item's tile: decode 8 rows, then dot them against 16 tokens
    #[test]
    #[ignore = "benchmark"]
    fn tile_rate() {
        const ROWS: usize = 8;
        const TOKENS: usize = 16;
        const ITERS: usize = 20_000;
        for (ty, size, cols) in [
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
