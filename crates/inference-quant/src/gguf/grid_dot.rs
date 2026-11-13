//! The 1-3 bit grid i-quants (IQ1_S, IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS) against Q8_K activations: each 32-value
//! sub-block decodes to signed bytes under two 16-value integer scales, and each block to one float factor.

use inference_tensor::quantized::k_quants::{BlockQ8K, QK_K};

use super::super::iq_tables::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS};
use super::GgufType;

const IQ2_XS_QS: usize = 2;
const IQ2_XS_SCALES: usize = 66;
const IQ2_S_QS: usize = 2;
const IQ2_S_SIGNS: usize = 34;
const IQ2_S_QH: usize = 66;
const IQ2_S_SCALES: usize = 74;
const IQ3_XXS_QS: usize = 2;
const IQ3_XXS_SCALES_AND_SIGNS: usize = 66;
const IQ1_S_QS: usize = 2;
const IQ1_S_QH: usize = 34;
const IQ1_M_QH: usize = 32;
const IQ1_M_SCALES: usize = 48;
// a sub-scale code s weighs (2 s + 1) times these of the block's f16 scale
const IQ2_FACTOR: f32 = 0.125;
const IQ3_XXS_FACTOR: f32 = 0.25;
// IQ1 values are grid + or - 1/8, held here as 8 grid + or - 1 under an eighth of the scale
const IQ1_FACTOR: f32 = 0.125;
const IQ1_GRID_STEP: i8 = 8;
// each byte's eight bits as bytes of -1 (bit set) or +1
const BYTE_SIGNS: [u64; 256] = byte_signs();

const fn byte_signs() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut j = 0;
        while j < 8 {
            let sign: u64 = if byte & (1 << j) != 0 { 0xff } else { 0x01 };
            table[byte] |= sign << (8 * j);
            j += 1;
        }
        byte += 1;
    }
    table
}

pub(super) fn supports(ty: GgufType) -> bool {
    matches!(
        ty,
        GgufType::Iq1S | GgufType::Iq1M | GgufType::Iq2Xs | GgufType::Iq2S | GgufType::Iq3Xxs
    )
}

#[inline(always)]
fn f16_at(block: &[u8], at: usize) -> f32 {
    half::f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
}

#[inline(always)]
fn u16_at(block: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([block[at], block[at + 1]]))
}

#[inline(always)]
fn u32_at(block: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(block[at..at + 4].try_into().expect("4 bytes"))
}

/// Sub-block `ib`'s four 8-value groups (grid magnitudes or IQ1's signed grid, then what each group needs) and its
/// two 16-value scales.
struct Sub {
    grids: [u64; 4],
    signs: [u64; 4],
    // IQ1's per-group offset of + or - 1, as bytes
    shifts: [u64; 4],
    scales: [i16; 2],
}

#[inline(always)]
fn odd(code: usize) -> i16 {
    2 * (code & 0xf) as i16 + 1
}

#[inline(always)]
fn iq1_shift(negative: bool) -> u64 {
    if negative {
        u64::from_le_bytes([0xff; 8])
    } else {
        u64::from_le_bytes([1; 8])
    }
}

/// One grid type's block layout; the kernels are generic over it so each type's decode inlines into its own loop.
trait Grid {
    const BYTES: usize;
    // IQ1's grids hold signed values and an offset instead of magnitudes and signs
    const SHIFTED: bool = false;
    /// The float every integer value of the block is multiplied by.
    fn factor(block: &[u8]) -> f32;
    fn sub(block: &[u8], ib: usize) -> Sub;
}

struct Iq2Xs;
struct Iq2S;
struct Iq3Xxs;
struct Iq1S;
struct Iq1M;

impl Grid for Iq2Xs {
    const BYTES: usize = 74;

    fn factor(block: &[u8]) -> f32 {
        IQ2_FACTOR * f16_at(block, 0)
    }

    #[inline(always)]
    fn sub(block: &[u8], ib: usize) -> Sub {
        let q = |l: usize| u16_at(block, IQ2_XS_QS + 2 * (4 * ib + l));
        let scale = usize::from(block[IQ2_XS_SCALES + ib]);
        Sub {
            grids: std::array::from_fn(|l| IQ2XS_GRID[q(l) & 511]),
            signs: std::array::from_fn(|l| BYTE_SIGNS[usize::from(KSIGNS_IQ2XS[q(l) >> 9])]),
            shifts: [0; 4],
            scales: [odd(scale), odd(scale >> 4)],
        }
    }
}

impl Grid for Iq2S {
    const BYTES: usize = 82;

    fn factor(block: &[u8]) -> f32 {
        IQ2_FACTOR * f16_at(block, 0)
    }

    #[inline(always)]
    fn sub(block: &[u8], ib: usize) -> Sub {
        let qh = usize::from(block[IQ2_S_QH + ib]);
        let scale = usize::from(block[IQ2_S_SCALES + ib]);
        Sub {
            grids: std::array::from_fn(|l| {
                IQ2S_GRID[usize::from(block[IQ2_S_QS + 4 * ib + l]) | ((qh << (8 - 2 * l)) & 0x300)]
            }),
            signs: std::array::from_fn(|l| {
                BYTE_SIGNS[usize::from(block[IQ2_S_SIGNS + 4 * ib + l])]
            }),
            shifts: [0; 4],
            scales: [odd(scale), odd(scale >> 4)],
        }
    }
}

impl Grid for Iq3Xxs {
    const BYTES: usize = 98;

    fn factor(block: &[u8]) -> f32 {
        IQ3_XXS_FACTOR * f16_at(block, 0)
    }

    #[inline(always)]
    fn sub(block: &[u8], ib: usize) -> Sub {
        let aux = u32_at(block, IQ3_XXS_SCALES_AND_SIGNS + 4 * ib);
        let qs = &block[IQ3_XXS_QS + 8 * ib..][..8];
        let scale = odd((aux >> 28) as usize);
        Sub {
            grids: std::array::from_fn(|l| {
                u64::from(IQ3XXS_GRID[usize::from(qs[2 * l])])
                    | (u64::from(IQ3XXS_GRID[usize::from(qs[2 * l + 1])]) << 32)
            }),
            signs: std::array::from_fn(|l| {
                BYTE_SIGNS[usize::from(KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize])]
            }),
            shifts: [0; 4],
            scales: [scale; 2],
        }
    }
}

impl Grid for Iq1S {
    const BYTES: usize = 50;
    const SHIFTED: bool = true;

    fn factor(block: &[u8]) -> f32 {
        IQ1_FACTOR * f16_at(block, 0)
    }

    #[inline(always)]
    fn sub(block: &[u8], ib: usize) -> Sub {
        let qh = u16_at(block, IQ1_S_QH + 2 * ib);
        let scale = 2 * ((qh >> 12) & 7) as i16 + 1;
        Sub {
            grids: std::array::from_fn(|l| {
                IQ1S_GRID[usize::from(block[IQ1_S_QS + 4 * ib + l]) | (((qh >> (3 * l)) & 7) << 8)]
            }),
            signs: [0; 4],
            shifts: [iq1_shift(qh & 0x8000 != 0); 4],
            scales: [scale; 2],
        }
    }
}

impl Grid for Iq1M {
    const BYTES: usize = 56;
    const SHIFTED: bool = true;

    // the f16 scale sits in the top nibbles of the four sub-scale words
    fn factor(block: &[u8]) -> f32 {
        let sc = |k: usize| u16_at(block, IQ1_M_SCALES + 2 * k);
        let bits =
            (sc(0) >> 12) | ((sc(1) >> 8) & 0x00f0) | ((sc(2) >> 4) & 0x0f00) | (sc(3) & 0xf000);
        IQ1_FACTOR * half::f16::from_bits(bits as u16).to_f32()
    }

    #[inline(always)]
    fn sub(block: &[u8], ib: usize) -> Sub {
        let sc = u16_at(block, IQ1_M_SCALES + 2 * (ib / 2)) >> (6 * (ib % 2));
        let qh = [
            usize::from(block[IQ1_M_QH + 2 * ib]),
            usize::from(block[IQ1_M_QH + 2 * ib + 1]),
        ];
        let qs = &block[4 * ib..][..4];
        let index = |l: usize| usize::from(qs[l]) | ((qh[l / 2] << (8 - 4 * (l % 2))) & 0x700);
        Sub {
            grids: std::array::from_fn(|l| IQ1S_GRID[index(l)]),
            signs: [0; 4],
            shifts: std::array::from_fn(|l| iq1_shift(qh[l / 2] & (0x08 << (4 * (l % 2))) != 0)),
            scales: [2 * (sc & 7) as i16 + 1, 2 * ((sc >> 3) & 7) as i16 + 1],
        }
    }
}

// runs `$body` with `$g` naming the grid type of `$ty`
macro_rules! with_grid {
    ($ty:expr, $g:ident => $body:expr) => {
        match $ty {
            GgufType::Iq2Xs => {
                type $g = Iq2Xs;
                $body
            }
            GgufType::Iq2S => {
                type $g = Iq2S;
                $body
            }
            GgufType::Iq3Xxs => {
                type $g = Iq3Xxs;
                $body
            }
            GgufType::Iq1S => {
                type $g = Iq1S;
                $body
            }
            GgufType::Iq1M => {
                type $g = Iq1M;
                $body
            }
            other => unreachable!("grid_dot::supports({other:?}) is false"),
        }
    };
}

impl Sub {
    // the 32 integer values
    fn values(&self, shifted: bool) -> [i8; 32] {
        let mut out = [0i8; 32];
        for l in 0..4 {
            let grid = self.grids[l].to_le_bytes();
            let signs = self.signs[l].to_le_bytes();
            let shifts = self.shifts[l].to_le_bytes();
            for j in 0..8 {
                out[8 * l + j] = if shifted {
                    IQ1_GRID_STEP * grid[j] as i8 + shifts[j] as i8
                } else {
                    grid[j] as i8 * signs[j] as i8
                };
            }
        }
        out
    }
}

fn generic<G: Grid>(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (block, y) in row.chunks_exact(G::BYTES).zip(ys) {
        let q8 = y.qs();
        let mut bsum = 0i32;
        for ib in 0..QK_K / 32 {
            let sub = G::sub(block, ib);
            let values = sub.values(G::SHIFTED);
            for (half, &scale) in sub.scales.iter().enumerate() {
                let dot: i32 = (16 * half..16 * half + 16)
                    .map(|j| i32::from(values[j]) * i32::from(q8[32 * ib + j]))
                    .sum();
                bsum += dot * i32::from(scale);
            }
        }
        sumf += G::factor(block) * y.d() * bsum as f32;
    }
    sumf
}

pub(super) fn vec_dot_generic(ty: GgufType, row: &[u8], ys: &[BlockQ8K]) -> f32 {
    with_grid!(ty, G => generic::<G>(row, ys))
}

pub(super) fn vec_dot(ty: GgufType, row: &[u8], ys: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above
        return with_grid!(ty, G => unsafe { avx2::dot::<G, 1>(row, [ys]) }[0]);
    }
    vec_dot_generic(ty, row, ys)
}

pub(super) fn vec_dot_x4(ty: GgufType, row: &[u8], ys: [&[BlockQ8K]; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    if super::has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above
        return with_grid!(ty, G => unsafe { avx2::dot::<G, 4>(row, ys) });
    }
    ys.map(|y| vec_dot_generic(ty, row, y))
}

#[cfg(target_arch = "x86_64")]
pub(super) mod avx2 {
    use std::arch::x86_64::*;

    use inference_tensor::quantized::k_quants::{BlockQ8K, QK_K};

    use super::super::avx2::{hsum, prefetch_ahead, store_scaled};
    use super::{GgufType, Grid, IQ1_GRID_STEP, Iq1M, Iq1S, Iq2S, Iq2Xs, Iq3Xxs, Sub};

    // a sub-block's 32 signed values and its two scales; without a target feature of its own so it always inlines
    // into the AVX2 loops (a call would pass the vectors through memory)
    #[inline(always)]
    unsafe fn decode<G: Grid>(block: &[u8], ib: usize) -> (__m256i, [i16; 2]) {
        let Sub {
            grids,
            signs,
            shifts,
            scales,
        } = G::sub(block, ib);
        // SAFETY: the caller runs with AVX2 enabled
        unsafe {
            let set =
                |v: [u64; 4]| _mm256_set_epi64x(v[3] as i64, v[2] as i64, v[1] as i64, v[0] as i64);
            let values = if G::SHIFTED {
                // grid bytes are -1, 0 or 1, so their sign applied to the step is the step times them
                let scaled = _mm256_sign_epi8(_mm256_set1_epi8(IQ1_GRID_STEP), set(grids));
                _mm256_add_epi8(scaled, set(shifts))
            } else {
                _mm256_sign_epi8(set(grids), set(signs))
            };
            (values, scales)
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn dot<G: Grid, const N: usize>(row: &[u8], ys: [&[BlockQ8K]; N]) -> [f32; N] {
        let mut acc = [_mm256_setzero_ps(); N];
        for (b, block) in row.chunks_exact(G::BYTES).enumerate() {
            prefetch_ahead(block, 1);
            let mut sumi = [_mm256_setzero_si256(); N];
            let q8: [&[i8; QK_K]; N] = std::array::from_fn(|t| ys[t][b].qs());
            for ib in 0..QK_K / 32 {
                // SAFETY: AVX2 is enabled here
                let (w, [s0, s1]) = unsafe { decode::<G>(block, ib) };
                let scales = _mm256_set_m128i(_mm_set1_epi16(s1), _mm_set1_epi16(s0));
                let magnitude = _mm256_sign_epi8(w, w);
                for t in 0..N {
                    // SAFETY: the 32 values at 32 ib lie inside the block's QK_K
                    let y = unsafe { _mm256_loadu_si256(q8[t][32 * ib..].as_ptr().cast()) };
                    let p16 = _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(y, w));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(p16, scales));
                }
            }
            let factor = G::factor(block);
            for t in 0..N {
                let d = _mm256_set1_ps(factor * ys[t][b].d());
                acc[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        acc.map(|acc| hsum(acc))
    }

    #[target_feature(enable = "avx2,fma")]
    fn decode_rows<G: Grid>(rows: &[u8], w: &mut [i16], d: &mut [f32]) {
        let blocks = rows.chunks_exact(G::BYTES);
        for ((block, w), d) in blocks.zip(w.as_chunks_mut::<QK_K>().0).zip(d) {
            *d = G::factor(block);
            for ib in 0..QK_K / 32 {
                // SAFETY: AVX2 is enabled here
                let (values, [s0, s1]) = unsafe { decode::<G>(block, ib) };
                store_scaled(
                    &mut w[32 * ib..],
                    _mm256_castsi256_si128(values),
                    _mm256_set1_epi16(s0),
                );
                store_scaled(
                    &mut w[32 * ib + 16..],
                    _mm256_extracti128_si256(values, 1),
                    _mm256_set1_epi16(s1),
                );
            }
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(in super::super) fn decode_tile(ty: GgufType, rows: &[u8], w: &mut [i16], d: &mut [f32]) {
        with_grid!(ty, G => decode_rows::<G>(rows, w, d))
    }
}

#[cfg(test)]
mod tests {
    use super::{BYTE_SIGNS, Sub};

    #[test]
    fn iq1_values_are_eight_grid_steps_plus_the_shift() {
        let sub = Sub {
            grids: [u64::from_le_bytes([0xff, 0, 1, 0, 0xff, 1, 1, 0]), 0, 0, 0],
            signs: [0; 4],
            shifts: [
                u64::from_le_bytes([0xff; 8]),
                u64::from_le_bytes([1; 8]),
                0,
                0,
            ],
            scales: [1, 1],
        };
        assert_eq!(sub.values(true)[..8], [-9, -1, 7, -1, -9, 7, 7, -1]);
        assert_eq!(sub.values(true)[8], 1);
        assert_eq!(
            BYTE_SIGNS[0b101].to_le_bytes(),
            [0xff, 1, 0xff, 1, 1, 1, 1, 1]
        );
    }
}
