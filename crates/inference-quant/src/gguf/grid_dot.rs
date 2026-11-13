//! The 1-3 bit i-quants against Q8_K activations: ggml's grid types (IQ1_S, IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS),
//! ik_llama's table types (IQ2_K, IQ2_KS, IQ2_KL, IQ3_K, IQ3_KS) and its trellis types (IQ1_KT, IQ2_KT, IQ3_KT). Each 32-value sub-block decodes to signed bytes
//! under two 16-value integer scales, and each block to one float factor (with the row scale, where the type has one).

use inference_tensor::quantized::k_quants::{BlockQ8K, QK_K};

use super::super::iq_tables::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS};
use super::super::trellis;
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
const F16_ROW_SCALE: usize = 2;
const F32_ROW_SCALE: usize = 4;
// ik's trellis: a group's eight values are eight steps of its state, each a byte sum in -126..=126
const TRELLIS_INDEX_OFFSET: u32 = trellis::IK_INDEX_OFFSET;
const TRELLIS_GROUP: usize = 8;
const TRELLIS_MASK: u32 = trellis::IK.byte_sum().0;
const TRELLIS_BIAS: i32 = trellis::IK.byte_sum().1;
// the state after each of a group's eight steps is its start times these
const TRELLIS_POWERS: [u32; 8] = trellis::IK.powers::<8>();
// ik's CUDA and CPU-GEMM kernels scale IQ2_KT and IQ3_KT rows by these
const IQ2_KT_SCALE: f32 = 1.05;
const IQ3_KT_SCALE: f32 = 1.01;
const KT_SCALES: usize = 4;
const IQ1_KT_QL: usize = 8;
const IQ1_KT_QH: usize = 40;
const IQ3_KT_QH: usize = 68;
// the first half of ik's iq4k_values, the trellis types' sub-block scales
const IQ4K_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// each byte's eight bits as bytes of -1 (bit set) or +1
const BYTE_SIGNS: [u64; 256] = byte_signs();
const IQ2_K_EXTRA: usize = 2;
const IQ2_K_SCALES: usize = 4;
const IQ2_K_QS: usize = 12;
const IQ3_K_EXTRA: usize = 2;
const IQ3_K_SIGNS: usize = 4;
const IQ3_K_SCALES: usize = 6;
const IQ3_K_QS: usize = 14;
const IQ3_K_QH: usize = 78;
const IQ2_KS_SCALES: usize = 2;
const IQ2_KS_QS: usize = 6;
const IQ3_KS_SCALES: usize = 2;
const IQ3_KS_QS: usize = 6;
const IQ3_KS_QH: usize = 70;
const IQ2_KL_SCALES_L: usize = 2;
const IQ2_KL_QS: usize = 6;
const IQ2_KL_QH: usize = 70;
// ik's codebooks (ggml-common.h), each padded to 16 for a byte shuffle; an `extra` bit picks the shifted second half
const IQ2NL_VALUES: [i8; 16] = [-31, -13, 1, 17, -26, -8, 6, 22, 0, 0, 0, 0, 0, 0, 0, 0];
const IQ3NL_VALUES: [i8; 16] = [
    -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
];
// IQ2_KL's 32 codes each name a pair of values; these are the pairs' first and second values
const IQ2KL_FIRST: [i8; 32] = iq2kl_half(0);
const IQ2KL_SECOND: [i8; 32] = iq2kl_half(8);
const IQ2KL_PAIRS: [u16; 32] = [
    0xe9c1, 0x0dc1, 0xc1d8, 0xf6d8, 0x0dd8, 0x2fd8, 0xd8e9, 0xe9e9, 0x01e9, 0x0de9, 0x1ce9, 0xc1f6,
    0x01f6, 0x0df6, 0x2ff6, 0xe901, 0xf601, 0x0101, 0x0d01, 0x1c01, 0xd80d, 0xe90d, 0xf60d, 0x010d,
    0x0d0d, 0xc11c, 0xe91c, 0x011c, 0x1c1c, 0x2f1c, 0xe92f, 0x0d2f,
];

const fn iq2kl_half(shift: u32) -> [i8; 32] {
    let mut out = [0i8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = (IQ2KL_PAIRS[i] >> shift) as u8 as i8;
        i += 1;
    }
    out
}

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
        GgufType::Iq1S
            | GgufType::Iq1M
            | GgufType::Iq2Xs
            | GgufType::Iq2S
            | GgufType::Iq3Xxs
            | GgufType::Iq2K
            | GgufType::Iq2Ks
            | GgufType::Iq2Kl
            | GgufType::Iq3K
            | GgufType::Iq3Ks
            | GgufType::Iq1Kt
            | GgufType::Iq2Kt
            | GgufType::Iq3Kt
    )
}

#[inline(always)]
fn bit(word: usize, at: usize) -> usize {
    (word >> at) & 1
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

/// One type's block layout; the kernels are generic over it so each type's decode inlines into its own loop.
trait Grid {
    const BYTES: usize;
    // the f16 (2) or f32 (4) scale ahead of the row's blocks
    const ROW_SCALE: usize = 0;
    /// The float every integer value of the block is multiplied by, given the row's scale (1 without one).
    fn factor(block: &[u8], row: f32) -> f32;
    /// Sub-block `ib`'s 32 values and its two 16-value scales.
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]);
    /// The same as one vector; the caller runs with AVX2 enabled.
    #[cfg(target_arch = "x86_64")]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]);
}

// the grid types' decode: four grid words per sub-block, then their signs or IQ1's offsets
macro_rules! grid_decode {
    ($shifted:expr) => {
        #[inline(always)]
        fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
            let sub = Self::sub(block, ib);
            (sub.values($shifted), sub.scales)
        }

        #[cfg(target_arch = "x86_64")]
        #[inline(always)]
        unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
            // SAFETY: the caller runs with AVX2 enabled
            unsafe { avx2::grid_values(Self::sub(block, ib), $shifted) }
        }
    };
}

struct Iq2Xs;
struct Iq2S;
struct Iq3Xxs;
struct Iq1S;
struct Iq1M;

impl Grid for Iq2Xs {
    const BYTES: usize = 74;

    fn factor(block: &[u8], _: f32) -> f32 {
        IQ2_FACTOR * f16_at(block, 0)
    }

    grid_decode!(false);
}

impl Iq2Xs {
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

    fn factor(block: &[u8], _: f32) -> f32 {
        IQ2_FACTOR * f16_at(block, 0)
    }

    grid_decode!(false);
}

impl Iq2S {
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

    fn factor(block: &[u8], _: f32) -> f32 {
        IQ3_XXS_FACTOR * f16_at(block, 0)
    }

    grid_decode!(false);
}

impl Iq3Xxs {
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

    fn factor(block: &[u8], _: f32) -> f32 {
        IQ1_FACTOR * f16_at(block, 0)
    }

    grid_decode!(true);
}

impl Iq1S {
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

    // the f16 scale sits in the top nibbles of the four sub-scale words
    fn factor(block: &[u8], _: f32) -> f32 {
        let sc = |k: usize| u16_at(block, IQ1_M_SCALES + 2 * k);
        let bits =
            (sc(0) >> 12) | ((sc(1) >> 8) & 0x00f0) | ((sc(2) >> 4) & 0x0f00) | (sc(3) & 0xf000);
        IQ1_FACTOR * half::f16::from_bits(bits as u16).to_f32()
    }

    grid_decode!(true);
}

impl Iq1M {
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

struct Iq2K;
struct Iq2Ks;
struct Iq2Kl;
struct Iq3K;
struct Iq3Ks;

// 32 two-bit fields at `shift` of 32 bytes, plus each half's table offset, looked up in `table`
#[inline(always)]
fn table_values(
    table: &[i8; 16],
    q: &[u8],
    shift: usize,
    high: Option<(&[u8], usize)>,
    offsets: [usize; 2],
) -> [i8; 32] {
    std::array::from_fn(|j| {
        let mut index = usize::from(q[j] >> shift) & 3;
        if let Some((qh, at)) = high {
            index |= bit(usize::from(qh[j]), at) << 2;
        }
        table[offsets[j / 16] + index]
    })
}

impl Grid for Iq2K {
    const BYTES: usize = 76;

    fn factor(block: &[u8], _: f32) -> f32 {
        f16_at(block, 0)
    }

    #[inline(always)]
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
        let (q, shift, offsets, scales) = Self::parts(block, ib);
        (table_values(&IQ2NL_VALUES, q, shift, None, offsets), scales)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
        let (q, shift, offsets, scales) = Self::parts(block, ib);
        // SAFETY: the caller runs with AVX2 enabled
        (
            unsafe { avx2::table_values(&IQ2NL_VALUES, q, shift, None, offsets) },
            scales,
        )
    }
}

impl Iq2K {
    #[inline(always)]
    fn parts(block: &[u8], ib: usize) -> (&[u8], usize, [usize; 2], [i16; 2]) {
        let extra = u16_at(block, IQ2_K_EXTRA);
        let scale = block[IQ2_K_SCALES + ib];
        (
            &block[IQ2_K_QS + 32 * (ib / 4)..][..32],
            2 * (ib % 4),
            [4 * bit(extra, 2 * ib), 4 * bit(extra, 2 * ib + 1)],
            [i16::from(scale & 0xf) - 8, i16::from(scale >> 4) - 8],
        )
    }
}

impl Grid for Iq3K {
    const BYTES: usize = 110;

    fn factor(block: &[u8], _: f32) -> f32 {
        f16_at(block, 0)
    }

    #[inline(always)]
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
        let (q, shift, qh, offsets, scales) = Self::parts(block, ib);
        (
            table_values(&IQ3NL_VALUES, q, shift, Some((qh, ib)), offsets),
            scales,
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
        let (q, shift, qh, offsets, scales) = Self::parts(block, ib);
        // SAFETY: the caller runs with AVX2 enabled
        let values =
            unsafe { avx2::table_values(&IQ3NL_VALUES, q, shift, Some((qh, ib)), offsets) };
        (values, scales)
    }
}

impl Iq3K {
    #[inline(always)]
    fn parts(block: &[u8], ib: usize) -> (&[u8], usize, &[u8], [usize; 2], [i16; 2]) {
        let extra = u16_at(block, IQ3_K_EXTRA);
        let signs = u16_at(block, IQ3_K_SIGNS);
        let scale = usize::from(block[IQ3_K_SCALES + ib]);
        let signed = |code: usize, at: usize| {
            let magnitude = odd(code);
            if bit(signs, at) != 0 {
                -magnitude
            } else {
                magnitude
            }
        };
        (
            &block[IQ3_K_QS + 32 * (ib / 4)..][..32],
            2 * (ib % 4),
            &block[IQ3_K_QH..][..32],
            [8 * bit(extra, 2 * ib), 8 * bit(extra, 2 * ib + 1)],
            [signed(scale, 2 * ib), signed(scale >> 4, 2 * ib + 1)],
        )
    }
}

impl Grid for Iq2Ks {
    const BYTES: usize = 70;
    const ROW_SCALE: usize = F16_ROW_SCALE;

    fn factor(_: &[u8], row: f32) -> f32 {
        row
    }

    #[inline(always)]
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
        let (q, shift, offset, scale) = Self::parts(block, ib);
        (
            table_values(&IQ2NL_VALUES, q, shift, None, [offset; 2]),
            [scale; 2],
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
        let (q, shift, offset, scale) = Self::parts(block, ib);
        // SAFETY: the caller runs with AVX2 enabled
        let values = unsafe { avx2::table_values(&IQ2NL_VALUES, q, shift, None, [offset; 2]) };
        (values, [scale; 2])
    }
}

impl Iq2Ks {
    // a 64-value group's two halves are sub-blocks `2 g` and `2 g + 1`
    #[inline(always)]
    fn parts(block: &[u8], ib: usize) -> (&[u8], usize, usize, i16) {
        let extra = u16_at(block, 0);
        let (group, half) = (ib / 2, ib % 2);
        let low = usize::from(block[IQ2_KS_SCALES + group] >> (4 * half)) & 0xf;
        let code = low | (bit(extra, 8 + ib) << 4);
        (
            &block[IQ2_KS_QS + 32 * (group / 2)..][..32],
            4 * (group % 2) + 2 * half,
            4 * bit(extra, ib),
            code as i16 - 16,
        )
    }
}

impl Grid for Iq3Ks {
    const BYTES: usize = 102;
    const ROW_SCALE: usize = F16_ROW_SCALE;

    fn factor(_: &[u8], row: f32) -> f32 {
        row
    }

    #[inline(always)]
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
        let (q, shift, qh, offset, scale) = Self::parts(block, ib);
        (
            table_values(&IQ3NL_VALUES, q, shift, Some((qh, ib)), [offset; 2]),
            [scale; 2],
        )
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
        let (q, shift, qh, offset, scale) = Self::parts(block, ib);
        // SAFETY: the caller runs with AVX2 enabled
        let values =
            unsafe { avx2::table_values(&IQ3NL_VALUES, q, shift, Some((qh, ib)), [offset; 2]) };
        (values, [scale; 2])
    }
}

impl Iq3Ks {
    #[inline(always)]
    fn parts(block: &[u8], ib: usize) -> (&[u8], usize, &[u8], usize, i16) {
        let extra = u16_at(block, 0);
        let low = usize::from(block[IQ3_KS_SCALES + ib % 4] >> (4 * (ib / 4))) & 0xf;
        let code = low | (bit(extra, ib) << 4);
        (
            &block[IQ3_KS_QS + 32 * (ib / 4)..][..32],
            2 * (ib % 4),
            &block[IQ3_KS_QH..][..32],
            8 * bit(extra, 8 + ib),
            code as i16 - 16,
        )
    }
}

impl Grid for Iq2Kl {
    const BYTES: usize = 86;
    const ROW_SCALE: usize = F16_ROW_SCALE;

    fn factor(_: &[u8], row: f32) -> f32 {
        row
    }

    #[inline(always)]
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
        let (q, shift, qh, at, scale) = Self::parts(block, ib);
        let values = std::array::from_fn(|j| {
            let code =
                (usize::from(q[j / 2] >> shift) & 0xf) | (bit(usize::from(qh[j / 2]), at) << 4);
            if j % 2 == 0 {
                IQ2KL_FIRST[code]
            } else {
                IQ2KL_SECOND[code]
            }
        });
        (values, [scale; 2])
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
        let (q, shift, qh, at, scale) = Self::parts(block, ib);
        // SAFETY: the caller runs with AVX2 enabled
        (unsafe { avx2::iq2_kl_values(q, shift, qh, at) }, [scale; 2])
    }
}

impl Iq2Kl {
    // sub-block `ib`'s 16 code bytes (each half a 5-bit code with its qh bit) naming its 32 values in pairs
    #[inline(always)]
    fn parts(block: &[u8], ib: usize) -> (&[u8], usize, &[u8], usize, i16) {
        let scales_h = u16_at(block, 0);
        let (group, half) = (ib / 2, ib % 2);
        let low = usize::from(block[IQ2_KL_SCALES_L + ib % 4] >> (4 * (group / 2))) & 0xf;
        let high = (scales_h >> (4 * group + 2 * half)) & 3;
        (
            &block[IQ2_KL_QS + 16 * group..][..16],
            4 * half,
            &block[IQ2_KL_QH..][..16],
            ib,
            (low | (high << 4)) as i16 - 32,
        )
    }
}

struct Iq1Kt;
struct Iq2Kt;
struct Iq3Kt;

// group `g`'s eight trellis values from its start `index`
#[inline(always)]
fn trellis(index: u32) -> [i8; TRELLIS_GROUP] {
    trellis::IK
        .values::<TRELLIS_GROUP>(index + TRELLIS_INDEX_OFFSET)
        .map(|v| v as i8)
}

#[inline(always)]
fn trellis_values(index: impl Fn(usize) -> u32) -> [i8; 32] {
    let groups: [[i8; TRELLIS_GROUP]; 4] = std::array::from_fn(|l| trellis(index(l)));
    std::array::from_fn(|j| groups[j / TRELLIS_GROUP][j % TRELLIS_GROUP])
}

// the scale nibble IQ2_KT and IQ3_KT keep per sub-block in their first four bytes
#[inline(always)]
fn kt_nibble(block: &[u8], ib: usize) -> usize {
    usize::from(block[ib % 4] >> (4 * (ib / 4))) & 0xf
}

impl Grid for Iq2Kt {
    const BYTES: usize = 68;
    const ROW_SCALE: usize = F32_ROW_SCALE;

    fn factor(_: &[u8], row: f32) -> f32 {
        row * IQ2_KT_SCALE
    }

    #[inline(always)]
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
        let scale = i16::from(IQ4K_VALUES[kt_nibble(block, ib)]);
        (trellis_values(|l| Self::index(block, ib, l)), [scale; 2])
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
        let scale = i16::from(IQ4K_VALUES[kt_nibble(block, ib)]);
        // SAFETY: the caller runs with AVX2 enabled
        (
            unsafe { avx2::trellis_values(|l| Self::index(block, ib, l)) },
            [scale; 2],
        )
    }
}

impl Iq2Kt {
    #[inline(always)]
    fn index(block: &[u8], ib: usize, l: usize) -> u32 {
        u16_at(block, KT_SCALES + 2 * (4 * ib + l)) as u32
    }
}

impl Grid for Iq1Kt {
    const BYTES: usize = 56;
    const ROW_SCALE: usize = F32_ROW_SCALE;

    fn factor(_: &[u8], row: f32) -> f32 {
        row
    }

    #[inline(always)]
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
        let scale = i16::from(IQ4K_VALUES[usize::from(block[ib] & 0xf)]);
        (trellis_values(|l| Self::index(block, ib, l)), [scale; 2])
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
        let scale = i16::from(IQ4K_VALUES[usize::from(block[ib] & 0xf)]);
        // SAFETY: the caller runs with AVX2 enabled
        (
            unsafe { avx2::trellis_values(|l| Self::index(block, ib, l)) },
            [scale; 2],
        )
    }
}

impl Iq1Kt {
    // a 13-bit index: a ql byte, a qh nibble, and one bit of the sub-block's scale byte
    #[inline(always)]
    fn index(block: &[u8], ib: usize, l: usize) -> u32 {
        let g = 4 * ib + l;
        let s = u32::from(block[ib]);
        u32::from(block[IQ1_KT_QL + g])
            | ((u32::from(block[IQ1_KT_QH + g % 16]) << (8 - 4 * (g / 16))) & 0xf00)
            | ((s << (8 - l)) & 0x1000)
    }
}

impl Grid for Iq3Kt {
    const BYTES: usize = 100;
    const ROW_SCALE: usize = F32_ROW_SCALE;

    fn factor(_: &[u8], row: f32) -> f32 {
        row * IQ3_KT_SCALE
    }

    // magnitudes of the trellis values, each value's sign bit `ib` of its qh byte
    #[inline(always)]
    fn decode(block: &[u8], ib: usize) -> ([i8; 32], [i16; 2]) {
        let magnitudes = trellis_values(|l| Iq2Kt::index(block, ib, l));
        let qh = &block[IQ3_KT_QH..][..32];
        let values = std::array::from_fn(|j| {
            let m = magnitudes[j].abs();
            if bit(usize::from(qh[j]), ib) != 0 {
                -m
            } else {
                m
            }
        });
        (values, [kt_nibble(block, ib) as i16; 2])
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn decode_avx2(block: &[u8], ib: usize) -> (std::arch::x86_64::__m256i, [i16; 2]) {
        // SAFETY: the caller runs with AVX2 enabled
        let values = unsafe {
            let magnitudes = avx2::trellis_values(|l| Iq2Kt::index(block, ib, l));
            avx2::with_signs(magnitudes, &block[IQ3_KT_QH..], ib)
        };
        (values, [kt_nibble(block, ib) as i16; 2])
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
            GgufType::Iq2K => {
                type $g = Iq2K;
                $body
            }
            GgufType::Iq2Ks => {
                type $g = Iq2Ks;
                $body
            }
            GgufType::Iq2Kl => {
                type $g = Iq2Kl;
                $body
            }
            GgufType::Iq3K => {
                type $g = Iq3K;
                $body
            }
            GgufType::Iq3Ks => {
                type $g = Iq3Ks;
                $body
            }
            GgufType::Iq1Kt => {
                type $g = Iq1Kt;
                $body
            }
            GgufType::Iq2Kt => {
                type $g = Iq2Kt;
                $body
            }
            GgufType::Iq3Kt => {
                type $g = Iq3Kt;
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

// the row's blocks and its scale (1 for a type without one)
#[inline(always)]
fn split_row<G: Grid>(row: &[u8]) -> (&[u8], f32) {
    let scale = match G::ROW_SCALE {
        F16_ROW_SCALE => f16_at(row, 0),
        F32_ROW_SCALE => f32::from_le_bytes(row[..4].try_into().expect("4 bytes")),
        _ => 1.0,
    };
    (&row[G::ROW_SCALE..], scale)
}

fn generic<G: Grid>(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    let (blocks, row_scale) = split_row::<G>(row);
    let mut sumf = 0f32;
    for (block, y) in blocks.chunks_exact(G::BYTES).zip(ys) {
        let q8 = y.qs();
        let mut bsum = 0i32;
        for ib in 0..QK_K / 32 {
            let (values, scales) = G::decode(block, ib);
            for (half, &scale) in scales.iter().enumerate() {
                let dot: i32 = (16 * half..16 * half + 16)
                    .map(|j| i32::from(values[j]) * i32::from(q8[32 * ib + j]))
                    .sum();
                bsum += dot * i32::from(scale);
            }
        }
        sumf += G::factor(block, row_scale) * y.d() * bsum as f32;
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
    use super::{
        GgufType, Grid, IQ1_GRID_STEP, IQ2KL_FIRST, IQ2KL_SECOND, Iq1Kt, Iq1M, Iq1S, Iq2K, Iq2Kl,
        Iq2Ks, Iq2Kt, Iq2S, Iq2Xs, Iq3K, Iq3Ks, Iq3Kt, Iq3Xxs, Sub, TRELLIS_BIAS,
        TRELLIS_INDEX_OFFSET, TRELLIS_MASK, TRELLIS_POWERS, split_row,
    };

    // The decoders below have no target feature of their own so they always inline into the AVX2 loops (a call
    // would pass the vectors through memory); each runs only where its caller has AVX2 enabled.

    #[inline(always)]
    pub(super) unsafe fn grid_values(sub: Sub, shifted: bool) -> (__m256i, [i16; 2]) {
        // SAFETY: the caller runs with AVX2 enabled
        unsafe {
            let set =
                |v: [u64; 4]| _mm256_set_epi64x(v[3] as i64, v[2] as i64, v[1] as i64, v[0] as i64);
            let values = if shifted {
                // grid bytes are -1, 0 or 1, so their sign applied to the step is the step times them
                let scaled = _mm256_sign_epi8(_mm256_set1_epi8(IQ1_GRID_STEP), set(sub.grids));
                _mm256_add_epi8(scaled, set(sub.shifts))
            } else {
                _mm256_sign_epi8(set(sub.grids), set(sub.signs))
            };
            (values, sub.scales)
        }
    }

    // each byte's bit `at` of 32 bytes, as 0 or 1
    #[inline(always)]
    unsafe fn bits_at(bytes: &[u8], at: usize) -> __m256i {
        assert!(bytes.len() >= 32);
        // SAFETY: 32 readable bytes were checked above; the caller runs with AVX2 enabled
        unsafe {
            let v = _mm256_loadu_si256(bytes.as_ptr().cast());
            let shifted = _mm256_srl_epi16(v, _mm_cvtsi32_si128(at as i32));
            _mm256_and_si256(shifted, _mm256_set1_epi8(1))
        }
    }

    #[inline(always)]
    pub(super) unsafe fn table_values(
        table: &[i8; 16],
        q: &[u8],
        shift: usize,
        high: Option<(&[u8], usize)>,
        offsets: [usize; 2],
    ) -> __m256i {
        assert!(q.len() >= 32);
        // SAFETY: 32 readable bytes were checked above, the table holds 16; the caller runs with AVX2 enabled
        unsafe {
            let v = _mm256_loadu_si256(q.as_ptr().cast());
            // a 16-bit shift moves no set bit across a byte the mask keeps, as `shift` is at most 6
            let shifted = _mm256_srl_epi16(v, _mm_cvtsi32_si128(shift as i32));
            let mut index = _mm256_and_si256(shifted, _mm256_set1_epi8(3));
            if let Some((qh, at)) = high {
                index = _mm256_or_si256(index, _mm256_slli_epi16::<2>(bits_at(qh, at)));
            }
            let offset = _mm256_set_m128i(
                _mm_set1_epi8(offsets[1] as i8),
                _mm_set1_epi8(offsets[0] as i8),
            );
            let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(table.as_ptr().cast()));
            _mm256_shuffle_epi8(lut, _mm256_add_epi8(index, offset))
        }
    }

    // 16 five-bit codes looked up in both halves of the pair tables, then interleaved into 32 values
    #[inline(always)]
    pub(super) unsafe fn iq2_kl_values(q: &[u8], shift: usize, qh: &[u8], at: usize) -> __m256i {
        assert!(q.len() >= 16 && qh.len() >= 16);
        // SAFETY: 16 readable bytes of each were checked above, the tables hold 32; AVX2 is enabled by the caller
        unsafe {
            let load = |b: &[u8]| _mm_loadu_si128(b.as_ptr().cast());
            let low = _mm_and_si128(
                _mm_srl_epi16(load(q), _mm_cvtsi32_si128(shift as i32)),
                _mm_set1_epi8(0xf),
            );
            let fifth = _mm_and_si128(
                _mm_srl_epi16(load(qh), _mm_cvtsi32_si128(at as i32)),
                _mm_set1_epi8(1),
            );
            let upper = _mm_cmpeq_epi8(fifth, _mm_set1_epi8(1));
            let look = |table: &[i8; 32]| {
                let lo = _mm_shuffle_epi8(_mm_loadu_si128(table.as_ptr().cast()), low);
                let hi = _mm_shuffle_epi8(_mm_loadu_si128(table[16..].as_ptr().cast()), low);
                _mm_blendv_epi8(lo, hi, upper)
            };
            let (first, second) = (look(&IQ2KL_FIRST), look(&IQ2KL_SECOND));
            _mm256_set_m128i(
                _mm_unpackhi_epi8(first, second),
                _mm_unpacklo_epi8(first, second),
            )
        }
    }

    // four groups' trellis values: each group's eight states from one multiply by the step's powers, their masked
    // bytes summed by two multiply-adds, then packed to bytes in order
    #[inline(always)]
    pub(super) unsafe fn trellis_values(index: impl Fn(usize) -> u32) -> __m256i {
        // SAFETY: the caller runs with AVX2 enabled; the powers hold 8 words
        unsafe {
            let powers = _mm256_loadu_si256(TRELLIS_POWERS.as_ptr().cast());
            let group = |l: usize| {
                let start = _mm256_set1_epi32((index(l) + TRELLIS_INDEX_OFFSET) as i32);
                let states = _mm256_and_si256(
                    _mm256_mullo_epi32(start, powers),
                    _mm256_set1_epi32(TRELLIS_MASK as i32),
                );
                let pairs = _mm256_maddubs_epi16(states, _mm256_set1_epi8(1));
                let sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
                _mm256_sub_epi32(sums, _mm256_set1_epi32(TRELLIS_BIAS))
            };
            let low = _mm256_packs_epi32(group(0), group(1));
            let high = _mm256_packs_epi32(group(2), group(3));
            // the packs interleave the groups' halves per 128-bit lane; this puts each group's eight together
            let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            _mm256_permutevar8x32_epi32(_mm256_packs_epi16(low, high), order)
        }
    }

    // the magnitudes of `values`, negated where bit `at` of each of 32 bytes is set
    #[inline(always)]
    pub(super) unsafe fn with_signs(values: __m256i, bytes: &[u8], at: usize) -> __m256i {
        // SAFETY: the caller runs with AVX2 enabled; bits_at checks the 32 bytes
        unsafe {
            let negate = _mm256_cmpeq_epi8(bits_at(bytes, at), _mm256_set1_epi8(1));
            let magnitudes = _mm256_abs_epi8(values);
            _mm256_sub_epi8(_mm256_xor_si256(magnitudes, negate), negate)
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) fn dot<G: Grid, const N: usize>(row: &[u8], ys: [&[BlockQ8K]; N]) -> [f32; N] {
        let (blocks, row_scale) = split_row::<G>(row);
        let mut acc = [_mm256_setzero_ps(); N];
        for (b, block) in blocks.chunks_exact(G::BYTES).enumerate() {
            prefetch_ahead(block, 1);
            let mut sumi = [_mm256_setzero_si256(); N];
            let q8: [&[i8; QK_K]; N] = std::array::from_fn(|t| ys[t][b].qs());
            for ib in 0..QK_K / 32 {
                // SAFETY: AVX2 is enabled here
                let (w, [s0, s1]) = unsafe { G::decode_avx2(block, ib) };
                let scales = _mm256_set_m128i(_mm_set1_epi16(s1), _mm_set1_epi16(s0));
                let magnitude = _mm256_sign_epi8(w, w);
                for t in 0..N {
                    // SAFETY: the 32 values at 32 ib lie inside the block's QK_K
                    let y = unsafe { _mm256_loadu_si256(q8[t][32 * ib..].as_ptr().cast()) };
                    let p16 = _mm256_maddubs_epi16(magnitude, _mm256_sign_epi8(y, w));
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(p16, scales));
                }
            }
            let factor = G::factor(block, row_scale);
            for t in 0..N {
                let d = _mm256_set1_ps(factor * ys[t][b].d());
                acc[t] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        acc.map(|acc| hsum(acc))
    }

    #[target_feature(enable = "avx2,fma")]
    fn decode_rows<G: Grid>(rows: &[u8], cols: usize, w: &mut [i16], d: &mut [f32]) {
        let row_bytes = G::ROW_SCALE + cols / QK_K * G::BYTES;
        let w_blocks = w.as_chunks_mut::<QK_K>().0.chunks_exact_mut(cols / QK_K);
        for ((row, w), d) in rows
            .chunks_exact(row_bytes)
            .zip(w_blocks)
            .zip(d.chunks_exact_mut(cols / QK_K))
        {
            let (blocks, row_scale) = split_row::<G>(row);
            for ((block, w), d) in blocks.chunks_exact(G::BYTES).zip(w).zip(d) {
                *d = G::factor(block, row_scale);
                for ib in 0..QK_K / 32 {
                    // SAFETY: AVX2 is enabled here
                    let (values, [s0, s1]) = unsafe { G::decode_avx2(block, ib) };
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
    }

    #[target_feature(enable = "avx2,fma")]
    pub(in super::super) fn decode_tile(
        ty: GgufType,
        rows: &[u8],
        cols: usize,
        w: &mut [i16],
        d: &mut [f32],
    ) {
        with_grid!(ty, G => decode_rows::<G>(rows, cols, w, d))
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
