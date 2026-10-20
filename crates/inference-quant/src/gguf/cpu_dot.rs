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
        return unsafe { avx2::iq4_xs(row, ys) };
    }
    iq4_xs_generic(row, ys)
}

fn iq3_s(row: &[u8], ys: &[BlockQ8K]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected above
        return unsafe { avx2::iq3_s(row, ys) };
    }
    iq3_s_generic(row, ys)
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
        IQ4_XS_QS, KVALUES_IQ4NL, f16_at, iq4_xs_sub_scale,
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
    pub(super) fn iq4_xs(row: &[u8], ys: &[BlockQ8K]) -> f32 {
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
    pub(super) fn iq3_s(row: &[u8], ys: &[BlockQ8K]) -> f32 {
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
            println!(
                "{ty:?}: {:.2} GB/s one thread ({acc})",
                (200 * bytes.len()) as f64 / secs / 1e9
            );
        }
    }
}
