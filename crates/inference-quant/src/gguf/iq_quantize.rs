//! ggml's quantizers for the non-linear 4-bit types (`quantize_iq4_nl` / `quantize_iq4_xs` in ggml-quants.c), with
//! the importance weights an imatrix gives: rows of floats to raw GGUF blocks, as the dequantizers in `raw` read them.

use inference_tensor::Result;
use rayon::prelude::*;

use super::kernel::GgufType;
use super::raw::KVALUES_IQ4NL;

const QK4_NL: usize = 32;
const QK_K: usize = 256;
const BLOCK_IQ4_NL: usize = 2 + QK4_NL / 2;
const BLOCK_IQ4_XS: usize = 2 + 2 + QK_K / 64 + QK_K / 2;
// ggml's floor below which a group is zero
const GROUP_MAX_EPS: f32 = 1e-15;
// the scale search tries this many steps either side of the max-based scale
const SCALE_TRIES: i32 = 7;
// IQ4_XS sub-block scales are 6-bit, offset by 32
const SCALE_OFFSET: i32 = 32;

fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}

// the codebook entry nearest `x` (ggml's `best_index_int8`: binary search, then the closer neighbour)
fn best_index(values: &[i8; 16], x: f32) -> usize {
    if x <= f32::from(values[0]) {
        return 0;
    }
    if x >= f32::from(values[15]) {
        return 15;
    }
    let (mut lo, mut hi) = (0usize, 15usize);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x < f32::from(values[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if x - f32::from(values[lo]) < f32::from(values[hi]) - x {
        lo
    } else {
        hi
    }
}

/// One super-block of `x` (`quantize_row_iq4_nl_impl`): per 32-value block a scale fit to the importance weights;
/// for a super-block of several blocks those scales then share one f16 `d` as 6-bit multiples. Returns `(d, scales
/// code)` with the block's 4-bit indices in `indices`.
fn quantize_super_block(
    x: &[f32],
    quant_weights: Option<&[f32]>,
    indices: &mut [u8],
    scales_h: &mut u16,
    scales_l: &mut [u8],
) -> f32 {
    let values = &KVALUES_IQ4NL;
    let sigma2 = 2.0 * x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let blocks = x.len() / QK4_NL;
    let mut scales = vec![0f32; blocks];
    let mut weight = [0f32; QK4_NL];
    let (mut max_scale, mut amax_scale) = (0f32, 0f32);
    for ib in 0..blocks {
        let xb = &x[ib * QK4_NL..(ib + 1) * QK4_NL];
        let lb = &mut indices[ib * QK4_NL..(ib + 1) * QK4_NL];
        match quant_weights {
            Some(qw) => {
                let qw = &qw[ib * QK4_NL..(ib + 1) * QK4_NL];
                for j in 0..QK4_NL {
                    weight[j] = qw[j] * (sigma2 + xb[j] * xb[j]).sqrt();
                }
            }
            None => {
                for j in 0..QK4_NL {
                    weight[j] = xb[j] * xb[j];
                }
            }
        }
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            scales[ib] = 0.0;
            continue;
        }
        let fit = |inverse: f32, lb: Option<&mut [u8]>| {
            let (mut sumqx, mut sumq2) = (0f32, 0f32);
            let mut lb = lb;
            for j in 0..QK4_NL {
                let l = best_index(values, inverse * xb[j]);
                if let Some(lb) = lb.as_deref_mut() {
                    lb[j] = l as u8;
                }
                let q = f32::from(values[l]);
                sumqx += weight[j] * q * xb[j];
                sumq2 += weight[j] * q * q;
            }
            (sumqx, sumq2)
        };
        let d = -max / f32::from(values[0]);
        let (sumqx, sumq2) = fit(1.0 / d, Some(lb));
        let mut d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
        let mut best = d * sumqx;
        for itry in -SCALE_TRIES..=SCALE_TRIES {
            let inverse = (itry as f32 + f32::from(values[0])) / max;
            let (sumqx, sumq2) = fit(inverse, None);
            if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        if d.abs() > amax_scale {
            amax_scale = d.abs();
            max_scale = d;
        }
    }
    if blocks == 1 {
        // IQ4_NL: the block's own scale, its indices refit to it
        let d = scales[0];
        let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
        for (l, &v) in indices.iter_mut().zip(x) {
            *l = best_index(values, inverse * v) as u8;
        }
        return d;
    }
    let d = -max_scale / SCALE_OFFSET as f32;
    let inverse = if d != 0.0 { 1.0 / d } else { 0.0 };
    *scales_h = 0;
    scales_l.fill(0);
    for ib in 0..blocks {
        let l = nearest_int(inverse * scales[ib]).clamp(-SCALE_OFFSET, SCALE_OFFSET - 1);
        let dl = d * l as f32;
        let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
        let xb = &x[ib * QK4_NL..(ib + 1) * QK4_NL];
        for (j, &v) in xb.iter().enumerate() {
            indices[ib * QK4_NL + j] = best_index(values, idl * v) as u8;
        }
        let code = (l + SCALE_OFFSET) as u8;
        scales_l[ib / 2] |= (code & 0x0F) << (4 * (ib % 2));
        *scales_h |= u16::from(code >> 4) << (2 * ib);
    }
    d
}

// 32 indices to 16 bytes: the first 16 in the low nibbles, the next 16 in the high ones
fn pack_indices(indices: &[u8], qs: &mut [u8]) {
    for (group, out) in indices
        .chunks_exact(QK4_NL)
        .zip(qs.chunks_exact_mut(QK4_NL / 2))
    {
        for j in 0..QK4_NL / 2 {
            out[j] = group[j] | (group[QK4_NL / 2 + j] << 4);
        }
    }
}

fn quantize_iq4_nl_block(x: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    let mut indices = [0u8; QK4_NL];
    let d = quantize_super_block(x, qw, &mut indices, &mut 0, &mut []);
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    pack_indices(&indices, &mut out[2..]);
}

fn quantize_iq4_xs_block(x: &[f32], qw: Option<&[f32]>, out: &mut [u8]) {
    let mut indices = [0u8; QK_K];
    let (mut scales_h, mut scales_l) = (0u16, [0u8; QK_K / 64]);
    let d = quantize_super_block(x, qw, &mut indices, &mut scales_h, &mut scales_l);
    out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    out[2..4].copy_from_slice(&scales_h.to_le_bytes());
    out[4..8].copy_from_slice(&scales_l);
    pack_indices(&indices, &mut out[8..]);
}

/// Whether [`quantize_rows`] writes `ty`.
pub fn can_quantize(ty: GgufType) -> bool {
    matches!(ty, GgufType::Iq4Nl | GgufType::Iq4Xs)
}

/// Rows of `cols` floats as `ty` blocks, each column weighed by `imatrix` (its `cols` importances) when given.
pub fn quantize_rows(
    ty: GgufType,
    cols: usize,
    values: &[f32],
    imatrix: Option<&[f32]>,
) -> Result<Vec<u8>> {
    let (block, size, encode): (usize, usize, fn(&[f32], Option<&[f32]>, &mut [u8])) = match ty {
        GgufType::Iq4Nl => (QK4_NL, BLOCK_IQ4_NL, quantize_iq4_nl_block),
        GgufType::Iq4Xs => (QK_K, BLOCK_IQ4_XS, quantize_iq4_xs_block),
        other => inference_tensor::bail!("{other:?} has no quantizer here"),
    };
    if !cols.is_multiple_of(block) || !values.len().is_multiple_of(cols) {
        inference_tensor::bail!("{ty:?} rows of {cols} values are not whole blocks of {block}");
    }
    if imatrix.is_some_and(|w| w.len() != cols) {
        inference_tensor::bail!(
            "an imatrix of {} entries for rows of {cols}",
            imatrix.unwrap().len()
        );
    }
    let row_bytes = cols / block * size;
    let mut out = vec![0u8; values.len() / cols * row_bytes];
    out.par_chunks_exact_mut(row_bytes)
        .zip(values.par_chunks_exact(cols))
        .for_each(|(row_out, row)| {
            for (i, (x, o)) in row
                .chunks_exact(block)
                .zip(row_out.chunks_exact_mut(size))
                .enumerate()
            {
                encode(x, imatrix.map(|w| &w[i * block..(i + 1) * block]), o);
            }
        });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0
            })
            .collect()
    }

    fn rms_error(ty: GgufType, cols: usize, x: &[f32], imatrix: Option<&[f32]>) -> (f32, Vec<f32>) {
        let bytes = quantize_rows(ty, cols, x, imatrix).unwrap();
        let back = super::super::raw::dequantize_rows(ty, cols, &bytes).unwrap();
        let err = x
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / x.len() as f32;
        (err.sqrt(), back)
    }

    // a 4-bit codebook over uniform values keeps the error near a sixteenth of the range
    #[test]
    fn rows_round_trip_through_the_dequantizers() {
        for ty in [GgufType::Iq4Nl, GgufType::Iq4Xs] {
            let x = values(4 * 512, 7);
            let (err, _) = rms_error(ty, 512, &x, None);
            assert!(err < 0.08, "{ty:?} round trip error {err}");
        }
    }

    #[test]
    fn an_imatrix_lowers_the_error_where_it_weighs() {
        let cols = 256;
        let x = values(64 * cols, 11);
        // the first half of the columns matters far more
        let imatrix = (0..cols)
            .map(|c| if c < cols / 2 { 100.0 } else { 0.01 })
            .collect::<Vec<_>>();
        let weighted_error = |back: &[f32]| {
            x.chunks(cols)
                .zip(back.chunks(cols))
                .map(|(a, b)| {
                    (0..cols)
                        .map(|c| imatrix[c] * (a[c] - b[c]).powi(2))
                        .sum::<f32>()
                })
                .sum::<f32>()
        };
        for ty in [GgufType::Iq4Nl, GgufType::Iq4Xs] {
            let (_, plain) = rms_error(ty, cols, &x, None);
            let (_, weighted) = rms_error(ty, cols, &x, Some(&imatrix));
            assert!(
                weighted_error(&weighted) < weighted_error(&plain),
                "{ty:?}: the imatrix did not lower the weighted error"
            );
        }
    }

    #[test]
    fn a_zero_row_stays_zero() {
        let x = vec![0f32; 256];
        for ty in [GgufType::Iq4Nl, GgufType::Iq4Xs] {
            let (_, back) = rms_error(ty, 256, &x, None);
            assert!(back.iter().all(|v| *v == 0.0), "{ty:?}");
        }
    }
}
