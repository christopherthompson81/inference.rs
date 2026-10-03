//! ggml's reference dequantization (`dequantize_row_*` in ggml-quants.c) of one IQ1 / IQ2 / IQ3 super-block.

use super::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS,
};

const IQ1S_DELTA: f32 = 0.125;
// Byte layouts of the 256-element blocks, after the leading f16 scale where there is one
const IQ2_XXS_QS: usize = 2;
const IQ2_XS_QS: usize = 2;
const IQ2_XS_SCALES: usize = 66;
const IQ2_S_QS: usize = 2;
const IQ2_S_QH: usize = 66;
const IQ2_S_SCALES: usize = 74;
const IQ3_XXS_QS: usize = 2;
const IQ3_XXS_SCALES_AND_SIGNS: usize = 66;
const IQ3_S_QS: usize = 2;
const IQ3_S_QH: usize = 66;
const IQ3_S_SIGNS: usize = 74;
const IQ3_S_SCALES: usize = 106;
const IQ1_S_QS: usize = 2;
const IQ1_S_QH: usize = 34;
const IQ1_M_QH: usize = 32;
const IQ1_M_SCALES: usize = 48;

fn f16_at(block: &[u8], at: usize) -> f32 {
    half::f16::from_le_bytes([block[at], block[at + 1]]).to_f32()
}

fn u16_at(block: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([block[at], block[at + 1]])
}

fn u32_at(block: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
}

// Eight grid magnitudes with the signs whose bits are set flipped
fn signed_grid(values: &mut [f32], scale: f32, grid: u64, signs: u8) {
    for (j, value) in values.iter_mut().enumerate() {
        let magnitude = f32::from(grid.to_le_bytes()[j]);
        let sign = if signs & KMASK_IQ2XS[j] != 0 {
            -1.0
        } else {
            1.0
        };
        *value = scale * magnitude * sign;
    }
}

pub(super) fn iq2_xxs(block: &[u8], y: &mut [f32]) {
    let d = f16_at(block, 0);
    for ib32 in 0..8 {
        let at = IQ2_XXS_QS + 8 * ib32;
        let (aux0, aux1) = (u32_at(block, at), u32_at(block, at + 4));
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[usize::from(aux0.to_le_bytes()[l])];
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            signed_grid(&mut y[32 * ib32 + 8 * l..][..8], db, grid, signs);
        }
    }
}

pub(super) fn iq2_xs(block: &[u8], y: &mut [f32]) {
    let d = f16_at(block, 0);
    for ib32 in 0..8 {
        let scale = block[IQ2_XS_SCALES + ib32];
        let db = [
            d * (0.5 + f32::from(scale & 0xf)) * 0.25,
            d * (0.5 + f32::from(scale >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let q = u16_at(block, IQ2_XS_QS + 2 * (4 * ib32 + l));
            let grid = IQ2XS_GRID[usize::from(q & 511)];
            let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
            signed_grid(&mut y[32 * ib32 + 8 * l..][..8], db[l / 2], grid, signs);
        }
    }
}

pub(super) fn iq2_s(block: &[u8], y: &mut [f32]) {
    let d = f16_at(block, 0);
    let qs = &block[IQ2_S_QS..IQ2_S_QS + 32];
    let signs = &block[IQ2_S_QS + 32..IQ2_S_QS + 64];
    for ib32 in 0..8 {
        let scale = block[IQ2_S_SCALES + ib32];
        let db = [
            d * (0.5 + f32::from(scale & 0xf)) * 0.25,
            d * (0.5 + f32::from(scale >> 4)) * 0.25,
        ];
        let qh = usize::from(block[IQ2_S_QH + ib32]);
        for l in 0..4 {
            let index = usize::from(qs[4 * ib32 + l]) | ((qh << (8 - 2 * l)) & 0x300);
            signed_grid(
                &mut y[32 * ib32 + 8 * l..][..8],
                db[l / 2],
                IQ2S_GRID[index],
                signs[4 * ib32 + l],
            );
        }
    }
}

pub(super) fn iq3_xxs(block: &[u8], y: &mut [f32]) {
    let d = f16_at(block, 0);
    for ib32 in 0..8 {
        let aux = u32_at(block, IQ3_XXS_SCALES_AND_SIGNS + 4 * ib32);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let qs = &block[IQ3_XXS_QS + 8 * ib32 + 2 * l..];
            let grid = u64::from(IQ3XXS_GRID[usize::from(qs[0])])
                | (u64::from(IQ3XXS_GRID[usize::from(qs[1])]) << 32);
            signed_grid(&mut y[32 * ib32 + 8 * l..][..8], db, grid, signs);
        }
    }
}

pub(super) fn iq3_s(block: &[u8], y: &mut [f32]) {
    let d = f16_at(block, 0);
    for ib32 in 0..8 {
        let scale = block[IQ3_S_SCALES + ib32 / 2];
        let db = d * f32::from(
            1 + 2 * if ib32 % 2 == 0 {
                scale & 0xf
            } else {
                scale >> 4
            },
        );
        let qh = usize::from(block[IQ3_S_QH + ib32]);
        for l in 0..4 {
            let qs = &block[IQ3_S_QS + 8 * ib32 + 2 * l..];
            let index1 = usize::from(qs[0]) | ((qh << (8 - 2 * l)) & 256);
            let index2 = usize::from(qs[1]) | ((qh << (7 - 2 * l)) & 256);
            let grid = u64::from(IQ3S_GRID[index1]) | (u64::from(IQ3S_GRID[index2]) << 32);
            signed_grid(
                &mut y[32 * ib32 + 8 * l..][..8],
                db,
                grid,
                block[IQ3_S_SIGNS + 4 * ib32 + l],
            );
        }
    }
}

// grid holds signed 8-bit values here; each gets `delta` added before the scale
fn shifted_grid(values: &mut [f32], scale: f32, grid: u64, delta: f32) {
    for (j, value) in values.iter_mut().enumerate() {
        *value = scale * (f32::from(grid.to_le_bytes()[j] as i8) + delta);
    }
}

pub(super) fn iq1_s(block: &[u8], y: &mut [f32]) {
    let d = f16_at(block, 0);
    for ib in 0..8 {
        let qh = usize::from(u16_at(block, IQ1_S_QH + 2 * ib));
        let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
        let delta = if qh & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        for l in 0..4 {
            let index = usize::from(block[IQ1_S_QS + 4 * ib + l]) | (((qh >> (3 * l)) & 7) << 8);
            shifted_grid(&mut y[32 * ib + 8 * l..][..8], dl, IQ1S_GRID[index], delta);
        }
    }
}

pub(super) fn iq1_m(block: &[u8], y: &mut [f32]) {
    let sc = |k: usize| usize::from(u16_at(block, IQ1_M_SCALES + 2 * k));
    let scale =
        (sc(0) >> 12) | ((sc(1) >> 8) & 0x00f0) | ((sc(2) >> 4) & 0x0f00) | (sc(3) & 0xf000);
    let d = half::f16::from_bits(scale as u16).to_f32();
    for ib in 0..8 {
        let dl1 = d * (2 * ((sc(ib / 2) >> (6 * (ib % 2))) & 0x7) + 1) as f32;
        let dl2 = d * (2 * ((sc(ib / 2) >> (6 * (ib % 2) + 3)) & 0x7) + 1) as f32;
        let qs = &block[4 * ib..];
        let qh = [
            usize::from(block[IQ1_M_QH + 2 * ib]),
            usize::from(block[IQ1_M_QH + 2 * ib + 1]),
        ];
        let index = [
            usize::from(qs[0]) | ((qh[0] << 8) & 0x700),
            usize::from(qs[1]) | ((qh[0] << 4) & 0x700),
            usize::from(qs[2]) | ((qh[1] << 8) & 0x700),
            usize::from(qs[3]) | ((qh[1] << 4) & 0x700),
        ];
        let delta = |bits: usize| if bits != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
        let deltas = [
            delta(qh[0] & 0x08),
            delta(qh[0] & 0x80),
            delta(qh[1] & 0x08),
            delta(qh[1] & 0x80),
        ];
        for l in 0..4 {
            let dl = if l < 2 { dl1 } else { dl2 };
            shifted_grid(
                &mut y[32 * ib + 8 * l..][..8],
                dl,
                IQ1S_GRID[index[l]],
                deltas[l],
            );
        }
    }
}
