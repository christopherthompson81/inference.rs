//! ik_llama.cpp's `IQ*_K` dequantization (`dequantize_row_iq*_k*` in iqk_quantize.cpp) of one row.

use super::kernel::GgufType;

const QK_K: usize = 256;
// ik's value tables (ggml-common.h); each second half is the first shifted for blocks with their `extra` bit set
const IQ2NL_VALUES: [i8; 8] = [-31, -13, 1, 17, -26, -8, 6, 22];
const IQ3NL_VALUES: [i8; 16] = [
    -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
];
const IQ4K_VALUES: [i8; 32] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113, -123, -100, -79, -61,
    -45, -31, -18, -6, 5, 17, 29, 42, 57, 73, 93, 117,
];
const IQ5NL_VALUES: [i8; 64] = [
    -126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6, -1, 5, 11,
    17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121, -124, -112, -101, -90, -81, -72, -63,
    -55, -48, -41, -34, -28, -22, -16, -10, -4, 1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89,
    99, 111, 123,
];
// ik's CUDA and CPU-GEMM kernels read IQ6_K through this table; its reference evaluates the cubic it rounds
const IQ6NL_VALUES: [i8; 128] = [
    -127, -121, -115, -109, -104, -98, -93, -88, -84, -79, -74, -70, -66, -62, -58, -54, -51, -47,
    -44, -40, -37, -34, -31, -28, -25, -22, -19, -16, -13, -11, -8, -5, -2, 0, 3, 6, 9, 12, 14, 17,
    20, 23, 27, 30, 33, 36, 40, 44, 47, 51, 55, 59, 63, 68, 72, 77, 82, 87, 92, 98, 103, 109, 115,
    121, -126, -120, -114, -108, -103, -97, -92, -87, -83, -78, -73, -69, -65, -61, -57, -53, -50,
    -46, -43, -39, -36, -33, -30, -27, -24, -21, -18, -15, -12, -10, -7, -4, -1, 1, 4, 7, 10, 13,
    15, 18, 21, 24, 28, 31, 34, 37, 41, 45, 48, 52, 56, 60, 64, 69, 73, 78, 83, 88, 93, 99, 104,
    110, 116, 122,
];
// Pairs of int8 values, low byte first, indexed by IQ2_KL's 5-bit codes
const IQ2KL_VALUES: [u16; 32] = [
    0xe9c1, 0x0dc1, 0xc1d8, 0xf6d8, 0x0dd8, 0x2fd8, 0xd8e9, 0xe9e9, 0x01e9, 0x0de9, 0x1ce9, 0xc1f6,
    0x01f6, 0x0df6, 0x2ff6, 0xe901, 0xf601, 0x0101, 0x0d01, 0x1c01, 0xd80d, 0xe90d, 0xf60d, 0x010d,
    0x0d0d, 0xc11c, 0xe91c, 0x011c, 0x1c1c, 0x2f1c, 0xe92f, 0x0d2f,
];

fn f16_at(bytes: &[u8], at: usize) -> f32 {
    half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn v(table: &[i8], index: usize) -> f32 {
    f32::from(table[index])
}

/// Dequantizes one row of `y.len()` elements whose bytes (with any leading row scale) are `row`.
pub(super) fn dequantize_row(ty: GgufType, row: &[u8], y: &mut [f32]) {
    let meta = ty.row_scale_bytes();
    let d = match meta {
        2 => f16_at(row, 0),
        4 => f32::from_le_bytes([row[0], row[1], row[2], row[3]]),
        _ => 0.0,
    };
    let size = ty.type_size();
    let blocks = &row[meta..];
    for (i, out) in y.as_chunks_mut::<QK_K>().0.iter_mut().enumerate() {
        let x = &blocks[i * size..][..size];
        match ty {
            GgufType::Iq2K => iq2_k(x, out),
            GgufType::Iq3K => iq3_k(x, out),
            GgufType::Iq4K => iq4_k(x, out),
            GgufType::Iq5K => iq5_k(x, out),
            GgufType::Iq6K => iq6_k(x, out),
            GgufType::Iq4Ks => iq4_ks(x, d, out),
            GgufType::Iq2Ks => iq2_ks(x, d, out),
            GgufType::Iq4Kss => iq4_kss(x, d, out),
            GgufType::Iq5Ks => iq5_ks(x, d, out),
            GgufType::Iq3Ks => iq3_ks(x, d, out),
            GgufType::Iq2Kl => iq2_kl(x, d, out),
            _ => unreachable!("{ty:?} is not an IQ*_K type"),
        }
    }
}

// block_iq2_k: d, extra, scales[8], qs[64]
fn iq2_k(x: &[u8], y: &mut [f32]) {
    let d = f16_at(x, 0);
    let mut extra = u16_at(x, 2);
    let (scales, qs) = (&x[4..12], &x[12..76]);
    for ib32 in 0..8 {
        let dl1 = d * (i32::from(scales[ib32] & 0xf) - 8) as f32;
        let dl2 = d * (i32::from(scales[ib32] >> 4) - 8) as f32;
        let values1 = if extra & 1 != 0 { 4 } else { 0 };
        let values2 = if extra & 2 != 0 { 4 } else { 0 };
        extra >>= 2;
        let (q, shift) = (&qs[32 * (ib32 / 4)..], 2 * (ib32 % 4));
        for j in 0..16 {
            y[32 * ib32 + j] = dl1 * v(&IQ2NL_VALUES, values1 + usize::from((q[j] >> shift) & 3));
            y[32 * ib32 + j + 16] = dl2
                * v(
                    &IQ2NL_VALUES,
                    values2 + usize::from((q[j + 16] >> shift) & 3),
                );
        }
    }
}

// block_iq3_k: d, extra, scales_h, scales_l[8], qs[64], qh[32]
fn iq3_k(x: &[u8], y: &mut [f32]) {
    let d = f16_at(x, 0);
    let mut extra = u16_at(x, 2);
    let mut sh = u16_at(x, 4);
    let (scales_l, qs, qh) = (&x[6..14], &x[14..78], &x[78..110]);
    for ib32 in 0..8 {
        let sign = |bit: u16| if sh & bit != 0 { -1 } else { 1 };
        let dl1 = d * ((2 * i32::from(scales_l[ib32] & 0xf) + 1) * sign(1)) as f32;
        let dl2 = d * ((2 * i32::from(scales_l[ib32] >> 4) + 1) * sign(2)) as f32;
        sh >>= 2;
        let values1 = if extra & 1 != 0 { 8 } else { 0 };
        let values2 = if extra & 2 != 0 { 8 } else { 0 };
        extra >>= 2;
        let (q, shift_l, shift_h) = (&qs[32 * (ib32 / 4)..], 2 * (ib32 % 4), ib32 % 8);
        let index = |j: usize| {
            usize::from((q[j] >> shift_l) & 3) | (usize::from((qh[j] >> shift_h) & 1) << 2)
        };
        for j in 0..16 {
            y[32 * ib32 + j] = dl1 * v(&IQ3NL_VALUES, values1 + index(j));
            y[32 * ib32 + j + 16] = dl2 * v(&IQ3NL_VALUES, values2 + index(j + 16));
        }
    }
}

// block_iq4_k: d, extra, scales_h[4], scales_l[8], qs[128]
fn iq4_k(x: &[u8], y: &mut [f32]) {
    let d = f16_at(x, 0);
    let mut extra = u16_at(x, 2);
    let (scales_h, scales_l, qs) = (&x[4..8], &x[8..16], &x[16..144]);
    for ib in 0..8 {
        let sh = scales_h[ib / 2] >> (4 * (ib % 2));
        let dl1 = d * (i32::from((scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32) as f32;
        let dl2 = d * (i32::from((scales_l[ib] >> 4) | ((sh << 2) & 0x30)) - 32) as f32;
        let values1 = if extra & 1 != 0 { 16 } else { 0 };
        let values2 = if extra & 2 != 0 { 16 } else { 0 };
        extra >>= 2;
        let q = &qs[16 * ib..];
        for j in 0..16 {
            y[32 * ib + j] = dl1 * v(&IQ4K_VALUES, values1 + usize::from(q[j] & 0xf));
            y[32 * ib + j + 16] = dl2 * v(&IQ4K_VALUES, values2 + usize::from(q[j] >> 4));
        }
    }
}

// block_iq5_k: d, extra, scales_h[4], scales_l[8], qs[128], qh[32]
fn iq5_k(x: &[u8], y: &mut [f32]) {
    let d = f16_at(x, 0);
    let mut extra = usize::from(u16_at(x, 2));
    let (sh, sl, qs, qh) = (&x[4..8], &x[8..16], &x[16..144], &x[144..176]);
    for ib64 in 0..4 {
        let scale = |l: u8, h: u8| d * (i32::from(l | (h & 0x30)) - 32) as f32;
        let dl1 = scale(sl[2 * ib64] & 0xf, sh[ib64] << 4);
        let dl2 = scale(sl[2 * ib64] >> 4, sh[ib64] << 2);
        let dl3 = scale(sl[2 * ib64 + 1] & 0xf, sh[ib64]);
        let dl4 = scale(sl[2 * ib64 + 1] >> 4, sh[ib64] >> 2);
        let values = [
            (extra & 1) << 5,
            (extra & 2) << 4,
            (extra & 4) << 3,
            (extra & 8) << 2,
        ];
        let (q, shift) = (&qs[32 * ib64..], 2 * (ib64 % 4));
        let out = &mut y[64 * ib64..];
        for j in 0..16 {
            let h = |k: usize, bit: u8| usize::from((qh[k] >> shift) & bit);
            out[j] = dl1
                * v(
                    &IQ5NL_VALUES,
                    values[0] + (usize::from(q[j] & 0xf) | (h(j, 1) << 4)),
                );
            out[j + 16] = dl2
                * v(
                    &IQ5NL_VALUES,
                    values[1] + (usize::from(q[j + 16] & 0xf) | (h(j + 16, 1) << 4)),
                );
            out[j + 32] = dl3
                * v(
                    &IQ5NL_VALUES,
                    values[2] + (usize::from(q[j] >> 4) | (h(j, 2) << 3)),
                );
            out[j + 48] = dl4
                * v(
                    &IQ5NL_VALUES,
                    values[3] + (usize::from(q[j + 16] >> 4) | (h(j + 16, 2) << 3)),
                );
        }
        extra >>= 4;
    }
}

// block_iq6_k: d, extra, scales[16] (int8), qs[128], qh[64]
fn iq6_k(x: &[u8], y: &mut [f32]) {
    let d = f16_at(x, 0);
    let mut extra = usize::from(u16_at(x, 2));
    let (scales, qs, qh) = (&x[4..20], &x[20..148], &x[148..212]);
    for ib64 in 0..4 {
        let dl = |k: usize| d * f32::from(scales[4 * ib64 + k] as i8);
        let shifted = |bit: usize| if extra & bit != 0 { 64 } else { 0 };
        let (q, h, shift) = (&qs[32 * ib64..], &qh[32 * (ib64 / 2)..], 4 * (ib64 % 2));
        let out = &mut y[64 * ib64..];
        for j in 0..16 {
            let (h1, h2) = (usize::from(h[j] >> shift), usize::from(h[j + 16] >> shift));
            let q1 = usize::from(q[j] & 0xf) | ((h1 & 0x03) << 4);
            let q2 = usize::from(q[j + 16] & 0xf) | ((h2 & 0x03) << 4);
            let q3 = usize::from(q[j] >> 4) | ((h1 & 0x0c) << 2);
            let q4 = usize::from(q[j + 16] >> 4) | ((h2 & 0x0c) << 2);
            out[j] = dl(0) * v(&IQ6NL_VALUES, shifted(1) + q1);
            out[j + 16] = dl(1) * v(&IQ6NL_VALUES, shifted(2) + q2);
            out[j + 32] = dl(2) * v(&IQ6NL_VALUES, shifted(4) + q3);
            out[j + 48] = dl(3) * v(&IQ6NL_VALUES, shifted(8) + q4);
        }
        extra >>= 4;
    }
}

// block_iq4_ks: scales[8], qs[128]; the f32 row scale precedes the blocks
fn iq4_ks(x: &[u8], d: f32, y: &mut [f32]) {
    let (scales, qs) = (&x[..8], &x[8..136]);
    for ib in 0..8 {
        let dl = d * (i32::from(scales[ib] & 254) - 127) as f32;
        let values = usize::from(scales[ib] & 1) << 4;
        let q = &qs[16 * ib..];
        for j in 0..16 {
            y[32 * ib + j] = dl * v(&IQ4K_VALUES, values + usize::from(q[j] & 0xf));
            y[32 * ib + j + 16] = dl * v(&IQ4K_VALUES, values + usize::from(q[j] >> 4));
        }
    }
}

// block_iq2_ks: extra, scales[4], qs[64]; the f16 row scale precedes the blocks
fn iq2_ks(x: &[u8], d: f32, y: &mut [f32]) {
    let mut extra = u16_at(x, 0);
    let (scales, qs) = (&x[2..6], &x[6..70]);
    for ib64 in 0..4 {
        let dl1 = d * (i32::from((scales[ib64] & 0xf) | ((extra >> 4) & 0x10) as u8) - 16) as f32;
        let dl2 = d * (i32::from((scales[ib64] >> 4) | ((extra >> 5) & 0x10) as u8) - 16) as f32;
        let values1 = if extra & 1 != 0 { 4 } else { 0 };
        let values2 = if extra & 2 != 0 { 4 } else { 0 };
        extra >>= 2;
        let (q, shift) = (&qs[32 * (ib64 / 2)..], 4 * (ib64 % 2));
        for j in 0..32 {
            y[64 * ib64 + j] = dl1 * v(&IQ2NL_VALUES, values1 + usize::from((q[j] >> shift) & 3));
            y[64 * ib64 + j + 32] = dl2
                * v(
                    &IQ2NL_VALUES,
                    values2 + usize::from((q[j] >> (shift + 2)) & 3),
                );
        }
    }
}

// block_iq4_kss: 32 u16-pairs; each 32-element block packs its scale in the low bits of its eight u16 codes
fn iq4_kss(x: &[u8], d: f32, y: &mut [f32]) {
    for ib in 0..8 {
        let mut ls = 0i32;
        let mut aux = [0u8; 16];
        for k in 0..8 {
            let q = u16_at(x, 16 * ib + 2 * k);
            let mut a = q & 0xfffe;
            a ^= a >> 1;
            aux[2 * k..2 * k + 2].copy_from_slice(&a.to_le_bytes());
            ls |= i32::from(q & 1) << k;
        }
        let values = ((ls & 1) << 4) as usize;
        let dl = d * ((ls & 254) - 127) as f32;
        for j in 0..16 {
            y[32 * ib + j] = dl * v(&IQ4K_VALUES, values + usize::from(aux[j] & 0xf));
            y[32 * ib + j + 16] = dl * v(&IQ4K_VALUES, values + usize::from(aux[j] >> 4));
        }
    }
}

// block_iq5_ks: scales[8], qs[128], qh[32]; the f32 row scale precedes the blocks
fn iq5_ks(x: &[u8], d: f32, y: &mut [f32]) {
    let (scales, qs, qh) = (&x[..8], &x[8..136], &x[136..168]);
    for ib64 in 0..4 {
        let (s1, s2) = (scales[2 * ib64], scales[2 * ib64 + 1]);
        let dl1 = d * (i32::from(s1 & 254) - 127) as f32;
        let dl2 = d * (i32::from(s2 & 254) - 127) as f32;
        let (values1, values2) = (usize::from(s1 & 1) << 5, usize::from(s2 & 1) << 5);
        let q = &qs[32 * ib64..];
        for j in 0..32 {
            let h = |bit: usize| usize::from((qh[j] >> (2 * ib64 + bit)) & 1) << 4;
            y[64 * ib64 + j] = dl1 * v(&IQ5NL_VALUES, values1 + (usize::from(q[j] & 0xf) | h(0)));
            y[64 * ib64 + j + 32] =
                dl2 * v(&IQ5NL_VALUES, values2 + (usize::from(q[j] >> 4) | h(1)));
        }
    }
}

// block_iq3_ks: extra, scales[4], qs[64], qh[32]; the f16 row scale precedes the blocks
fn iq3_ks(x: &[u8], d: f32, y: &mut [f32]) {
    let extra = u16_at(x, 0);
    let (scales, qs, qh) = (&x[2..6], &x[6..70], &x[70..102]);
    let mut dl = [0f32; 8];
    for j in 0..4 {
        let ls1 = i32::from(scales[j] & 0xf) | (i32::from((extra >> j) & 1) << 4);
        let ls2 = i32::from(scales[j] >> 4) | (i32::from((extra >> (j + 4)) & 1) << 4);
        dl[j] = d * (ls1 - 16) as f32;
        dl[j + 4] = d * (ls2 - 16) as f32;
    }
    for i128 in 0..2 {
        let q = &qs[32 * i128..];
        for ib in 0..4 {
            let block = 4 * i128 + ib;
            let values = usize::from((extra >> (8 + block)) & 1) << 3;
            for j in 0..32 {
                let index =
                    usize::from((q[j] >> (2 * ib)) & 3) | (usize::from((qh[j] >> block) & 1) << 2);
                y[32 * block + j] = dl[block] * v(&IQ3NL_VALUES, values + index);
            }
        }
    }
}

// block_iq2_kl: scales_h, scales_l[4], qs[64], qh[16]; the f16 row scale precedes the blocks
fn iq2_kl(x: &[u8], d: f32, y: &mut [f32]) {
    let scales_h = u32::from(u16_at(x, 0));
    let (scales_l, qs, qh) = (&x[2..6], &x[6..70], &x[70..86]);
    for ib64 in 0..4 {
        let scale = |k: usize| {
            let low = u32::from(scales_l[(2 * ib64 + k) % 4] >> (4 * (ib64 / 2))) & 0xf;
            let high = ((scales_h >> (4 * ib64 + 2 * k)) & 3) << 4;
            d * ((low | high) as i32 - 32) as f32
        };
        let (dl1, dl2) = (scale(0), scale(1));
        let q = &qs[16 * ib64..];
        for j in 0..16 {
            let pair = |code: usize| IQ2KL_VALUES[code].to_le_bytes().map(|b| f32::from(b as i8));
            let val1 =
                pair(usize::from(q[j] & 0xf) | (usize::from((qh[j] >> (2 * ib64)) & 1) << 4));
            let val2 =
                pair(usize::from(q[j] >> 4) | (usize::from((qh[j] >> (2 * ib64 + 1)) & 1) << 4));
            let out = &mut y[64 * ib64..];
            out[2 * j] = dl1 * val1[0];
            out[2 * j + 1] = dl1 * val1[1];
            out[2 * j + 32] = dl2 * val2[0];
            out[2 * j + 33] = dl2 * val2[1];
        }
    }
}
