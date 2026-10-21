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

/// Whether `vec_dot` serves this type.
pub(crate) fn supports(ty: GgufType) -> bool {
    matches!(
        ty,
        GgufType::Iq3S
            | GgufType::Iq4Xs
            | GgufType::Q2K
            | GgufType::Q3K
            | GgufType::Q4K
            | GgufType::Q5K
            | GgufType::Q6K
    )
}

/// `row` holds `ys.len()` super-blocks of `ty`, one per Q8_K block.
pub(crate) fn vec_dot(ty: GgufType, row: &[u8], ys: &[BlockQ8K]) -> f32 {
    let n = ys.len() * QK_K;
    match ty {
        GgufType::Iq3S => iq3_s(row, ys),
        GgufType::Iq4Xs => iq4_xs(row, ys),
        GgufType::Q2K => BlockQ2K::vec_dot(n, blocks(row), ys),
        GgufType::Q3K => BlockQ3K::vec_dot(n, blocks(row), ys),
        GgufType::Q4K => BlockQ4K::vec_dot(n, blocks(row), ys),
        GgufType::Q5K => BlockQ5K::vec_dot(n, blocks(row), ys),
        GgufType::Q6K => BlockQ6K::vec_dot(n, blocks(row), ys),
        _ => unreachable!("cpu_dot::supports({ty:?}) is false"),
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
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
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

/// `vec_dot` of one row against four activation rows, decoding the row's blocks once for all of them.
pub(crate) fn vec_dot_x4(ty: GgufType, row: &[u8], ys: [&[BlockQ8K]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match ty {
            // SAFETY: AVX2 and FMA were detected above
            GgufType::Iq3S => return unsafe { avx2::iq3_s_x4(row, ys) },
            GgufType::Iq4Xs => return unsafe { avx2::iq4_xs_x4(row, ys) },
            _ => {}
        }
    }
    ys.map(|y| vec_dot(ty, row, y))
}

/// Whether `decode_tile` and `tile_dot` serve this type on this CPU.
pub(crate) fn tile_supports(ty: GgufType) -> bool {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        return matches!(ty, GgufType::Iq3S | GgufType::Iq4Xs);
    }
    let _ = ty;
    false
}

/// Decodes whole rows of `ty` into i16 weights with each sub-block's scale folded in, plus each block's scale.
pub(crate) fn decode_tile(ty: GgufType, rows: &[u8], w: &mut [i16], d: &mut [f32]) {
    assert!(
        tile_supports(ty) && w.len() == d.len() * QK_K && rows.len() == d.len() * ty.type_size()
    );
    #[cfg(target_arch = "x86_64")]
    // SAFETY: tile_supports checked AVX2 and FMA
    unsafe {
        match ty {
            GgufType::Iq3S => avx2::decode_iq3_s(rows, w, d),
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
        IQ3_S_BYTES, IQ3_S_QH, IQ3_S_QS, IQ3_S_SCALES, IQ3_S_SIGNS, IQ3S_GRID, IQ4_XS_BYTES,
        IQ4_XS_QS, KVALUES_IQ4NL, TileRow, f16_at, iq4_xs_sub_scale,
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

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq4_xs_one(row: &[u8], ys: &[BlockQ8K]) -> f32 {
        // SAFETY: the codebook holds 16 bytes
        let values = unsafe { _mm_loadu_si128(KVALUES_IQ4NL.as_ptr().cast()) };
        let m4 = _mm_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_ps();
        for (block, y) in row.as_chunks::<IQ4_XS_BYTES>().0.iter().zip(ys) {
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

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn iq3_s_one(row: &[u8], ys: &[BlockQ8K]) -> f32 {
        let sign_bytes = load(&SIGN_BYTES);
        let sign_bits = load(&SIGN_BITS);
        let mut acc = _mm256_setzero_ps();
        for (block, y) in row.as_chunks::<IQ3_S_BYTES>().0.iter().zip(ys) {
            let q8 = y.qs();
            let mut sumi = _mm256_setzero_si256();
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
        ] {
            for seed in 0..4 {
                let row = row(block_bytes, seed);
                let expected = dequantized_dot(ty, &row, &ys);
                let generic = match ty {
                    GgufType::Iq4Xs => iq4_xs_generic(&row, &ys),
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
        ] {
            let rows = 512;
            let bytes = vec![3u8; rows * 8 * size];
            let xs = vec![0.5f32; 8 * QK_K];
            let mut ys = vec![BlockQ8K::zeros(); 8];
            BlockQ8K::from_float(&xs, &mut ys);
            let start = std::time::Instant::now();
            let mut acc = 0f32;
            for _ in 0..200 {
                for row in bytes.chunks_exact(8 * size) {
                    acc += vec_dot(ty, row, &ys);
                }
            }
            let secs = start.elapsed().as_secs_f64();
            let start4 = std::time::Instant::now();
            for _ in 0..200 {
                for row in bytes.chunks_exact(8 * size) {
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
        ] {
            let blocks = cols / QK_K;
            let bytes = (0..ROWS * blocks * size)
                .map(|i| (i * 7 % 251) as u8)
                .collect::<Vec<_>>();
            let (mut w, mut d) = (vec![0i16; ROWS * cols], vec![0f32; ROWS * blocks]);
            let start = std::time::Instant::now();
            for _ in 0..ITERS {
                decode_tile(ty, std::hint::black_box(&bytes), &mut w, &mut d);
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
