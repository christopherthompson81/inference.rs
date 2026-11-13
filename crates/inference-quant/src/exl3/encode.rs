//! One linear to EXL3 after exllamav3's `quantize_exl3` (`modules/quant/exl3_lib/quantize.py`): the Hessian damped,
//! rotated by the input signs and 128-wide Hadamards and block-LDL factored; the weight regularized (output and input
//! scales, the same rotations, a global scale searched on sampled tiles), encoded by LDLQ (16 rows at a time, each
//! row of tiles through the trellis quantizer with the error so far fed forward through L), rotated back, and its
//! stored scales refit to the encoding in the Hessian metric.

use inference_tensor::{DType, Device, Result, Tensor};

use super::linalg::{cholesky, solve_spd};
use super::quantize::{pack_tile, quantize_tiles};
use super::{Exl3Bits, Exl3Codebook, HAD, TILE, TILE_VALUES, tile_order};

// the share of the mean diagonal added to the Hessian, and again on each retry of a factorization that fails
const SIGMA_REG: f64 = 0.025;
const FACTOR_RETRIES: usize = 10;
// the codebooks' value spread, which the input scales normalize to
const CODEBOOK_SCALE: f64 = 1.24371088;
// LDLQ's rows per span between full error-feedback updates
const SPAN_ROWS: usize = 128;
// the global scale search: tiles a diagonal of this width samples, a coarse grid, then a fine one around its best
const SAMPLE_WIDTH: usize = 3;
const COARSE_SCALES: [f64; 10] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9];
const FINE_STEP: f64 = 0.075;
const MIN_SCALE: f64 = 0.01;
// rounds of alternating output and input scale refits
const REFIT_ROUNDS: usize = 2;
// below these a scale or a denominator counts as zero
const ZERO: f64 = 1e-30;
const SCALE_EPS: f64 = 1e-10;

/// One linear's EXL3 tensors, as a checkpoint stores them.
pub struct Exl3Encoding {
    /// `(in / 16, out / 16, words)` uint16 words (as i16), tile by tile, on the CPU.
    pub trellis: Tensor,
    /// f16 input scales `(in)` and output scales `(out)`.
    pub suh: Tensor,
    pub svh: Tensor,
    /// The output error the encoding leaves, `tr(E H E^T) / tr(W H W^T)` with H undamped.
    pub proxy_error: f64,
}

// LDLQ's drift: the scale search samples tiles scaled up by this, for the error feedback that grows them
fn ldlq_drift(bits: Exl3Bits) -> f64 {
    match bits {
        Exl3Bits::Whole(1) => 1.08,
        Exl3Bits::Half(1) => 1.035,
        Exl3Bits::Whole(2) => 1.018,
        Exl3Bits::Half(2) => 1.009,
        Exl3Bits::Whole(3) => 1.004,
        _ => 1.0,
    }
}

// a deterministic stream of +-1 (xorshift64*), so one seed always flips the same signs
fn signs(len: usize, seed: u64, device: &Device) -> Result<Tensor> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    let values = (0..len)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            if state.wrapping_mul(0x2545F4914F6CDD1D) >> 63 == 0 {
                1f32
            } else {
                -1f32
            }
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, len, device)
}

fn hadamard_matrix(device: &Device) -> Result<Tensor> {
    let scale = 1.0 / (HAD as f32).sqrt();
    let values = (0..HAD * HAD)
        .map(|i| {
            let (r, c) = (i / HAD, i % HAD);
            if (r & c).count_ones() % 2 == 0 {
                scale
            } else {
                -scale
            }
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (HAD, HAD), device)
}

// x's columns rotated in 128-wide blocks (x @ diag(H, H, ...))
fn had_right(x: &Tensor, had: &Tensor) -> Result<Tensor> {
    let (r, c) = x.dims2()?;
    x.contiguous()?
        .reshape((r * c / HAD, HAD))?
        .matmul(had)?
        .reshape((r, c))
}

// x's rows rotated in 128-high blocks (diag(H, H, ...) @ x); the Hadamard is symmetric
fn had_left(x: &Tensor, had: &Tensor) -> Result<Tensor> {
    had_right(&x.t()?, had)?.t()?.contiguous()
}

fn diagonal_indices(n: usize, device: &Device) -> Result<Tensor> {
    let idx = (0..n).map(|i| (i * n + i) as u32).collect::<Vec<_>>();
    Tensor::from_vec(idx, n, device)
}

// `h` with `value` added along its diagonal
fn add_diagonal(h: &Tensor, value: f64) -> Result<Tensor> {
    let n = h.dim(0)?;
    let add = Tensor::full(value as f32, n, h.device())?;
    h.flatten_all()?
        .index_add(&diagonal_indices(n, h.device())?, &add, 0)?
        .reshape((n, n))
}

fn diagonal_mean(h: &Tensor) -> Result<f64> {
    let n = h.dim(0)?;
    let diag = h
        .flatten_all()?
        .index_select(&diagonal_indices(n, h.device())?, 0)?;
    Ok(f64::from(diag.mean_all()?.to_scalar::<f32>()?))
}

// L with each 16-column block multiplied by the inverse of its diagonal block (exllamav3's block LDL: unit
// diagonal blocks, which LDLQ never reads)
fn block_ldl(factor: &Tensor) -> Result<Tensor> {
    let k = factor.dim(0)?;
    let m = k / TILE;
    // the diagonal blocks, gathered whole and inverted on the host (lower triangular, so by substitution)
    let idx = (0..m * TILE_VALUES)
        .map(|i| {
            let (b, r, c) = (i / TILE_VALUES, i / TILE % TILE, i % TILE);
            ((b * TILE + r) * k + b * TILE + c) as u32
        })
        .collect::<Vec<_>>();
    let blocks = factor
        .flatten_all()?
        .index_select(&Tensor::from_vec(idx, m * TILE_VALUES, factor.device())?, 0)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let mut inverses = vec![0f32; m * TILE_VALUES];
    for (block, inv) in blocks
        .chunks(TILE_VALUES)
        .zip(inverses.chunks_mut(TILE_VALUES))
    {
        let l = |r: usize, c: usize| f64::from(block[r * TILE + c]);
        let mut x = [[0f64; TILE]; TILE];
        for c in 0..TILE {
            for r in c..TILE {
                let mut v = if r == c { 1.0 } else { 0.0 };
                for (j, xj) in x.iter().enumerate().take(r).skip(c) {
                    v -= l(r, j) * xj[c];
                }
                x[r][c] = v / l(r, r);
            }
        }
        for (r, row) in x.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                inv[r * TILE + c] = *v as f32;
            }
        }
    }
    let inverses = Tensor::from_vec(inverses, (m, TILE, TILE), factor.device())?;
    factor
        .reshape((k, m, TILE))?
        .transpose(0, 1)?
        .contiguous()?
        .matmul(&inverses)?
        .transpose(0, 1)?
        .contiguous()?
        .reshape((k, k))
}

// the rotated Hessian's block LDL factor, damping its diagonal further until the factorization succeeds
fn factor(h_rot: &Tensor) -> Result<Tensor> {
    let mut h = h_rot.clone();
    for _ in 0..=FACTOR_RETRIES {
        if let Some(factor) = cholesky(&h)? {
            return block_ldl(&factor);
        }
        h = add_diagonal(&h, 2.0 * SIGMA_REG * diagonal_mean(&h)?)?;
    }
    inference_tensor::bail!("the Hessian stays indefinite after {FACTOR_RETRIES} rounds of damping")
}

// root mean square along `dim`, kept
fn rms(x: &Tensor, dim: usize) -> Result<Tensor> {
    x.sqr()?.mean_keepdim(dim)?.sqrt()
}

// values below ZERO in a host vector replaced
fn floor_zeros(v: &mut [f32], with: f32) -> Vec<bool> {
    v.iter_mut()
        .map(|x| {
            let zero = f64::from(x.abs()) < ZERO;
            if zero {
                *x = with;
            }
            zero
        })
        .collect()
}

// the tile cells of `w` (k, n) for tiles at (tile row, tile col), in the quantizer's position order
fn gather_tiles(w: &Tensor, tiles: &[(usize, usize)]) -> Result<Tensor> {
    let n = w.dim(1)?;
    let perm = tile_order();
    let idx = tiles
        .iter()
        .flat_map(|&(tk, tn)| {
            perm.iter()
                .map(move |&cell| ((tk * TILE + cell / TILE) * n + tn * TILE + cell % TILE) as u32)
        })
        .collect::<Vec<_>>();
    let count = tiles.len();
    w.flatten_all()?
        .index_select(&Tensor::from_vec(idx, count * TILE_VALUES, w.device())?, 0)?
        .reshape((count, TILE_VALUES))
}

// the tiles the scale search samples: a wrapped diagonal, every tile row and column at least once, and the tiles of
// highest and lowest energy
fn sample_tiles(w: &Tensor) -> Result<Tensor> {
    let (k, n) = w.dims2()?;
    let (tiles_k, tiles_n) = (k / TILE, n / TILE);
    let diag = tiles_k.max(tiles_n);
    let mut tiles = (0..diag * SAMPLE_WIDTH)
        .map(|i| {
            let (d, wdx) = (i / SAMPLE_WIDTH, i % SAMPLE_WIDTH);
            (d % tiles_k, (d + wdx) % tiles_n)
        })
        .collect::<Vec<_>>();
    let energy = w
        .reshape((tiles_k, TILE, tiles_n, TILE))?
        .sqr()?
        .sum(3)?
        .sum(1)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let extremes = (diag * SAMPLE_WIDTH / 16)
        .max(8)
        .min(energy.len().div_ceil(2));
    let mut order = (0..energy.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| energy[b].total_cmp(&energy[a]));
    for &t in order
        .iter()
        .take(extremes)
        .chain(order.iter().rev().take(extremes))
    {
        tiles.push((t / tiles_n, t % tiles_n));
    }
    gather_tiles(w, &tiles)
}

// the mean square error of quantizing `tiles` scaled by `scale`, measured back at their own scale
fn scaled_error(tiles: &Tensor, scale: f64, bits: Exl3Bits, codebook: Exl3Codebook) -> Result<f64> {
    let (q, _) = quantize_tiles(&(tiles * scale)?, bits, codebook)?;
    Ok(f64::from(
        ((q / scale)? - tiles)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?,
    ))
}

// the global scale whose quantization error over the sampled tiles is least: a coarse grid on a third of the
// sample, a fine one on all of it, refined by a parabola through the best three
fn global_scale(w: &Tensor, bits: Exl3Bits, codebook: Exl3Codebook) -> Result<f64> {
    let tiles = (sample_tiles(w)? * ldlq_drift(bits))?;
    let count = tiles.dim(0)?;
    let third = (0..count).step_by(3).map(|i| i as u32).collect::<Vec<_>>();
    let len = third.len();
    let sub = tiles.index_select(&Tensor::from_vec(third, len, w.device())?, 0)?;
    let mut best = (f64::INFINITY, COARSE_SCALES[0]);
    for &s in &COARSE_SCALES {
        let e = scaled_error(&sub, s, bits, codebook)?;
        if e < best.0 {
            best = (e, s);
        }
    }
    let fine = (0..5)
        .map(|i| best.1 + FINE_STEP * (i as f64 - 2.0))
        .collect::<Vec<_>>();
    let errors = fine
        .iter()
        .map(|&s| scaled_error(&tiles, s, bits, codebook))
        .collect::<Result<Vec<_>>>()?;
    let at = (0..5)
        .min_by(|&a, &b| errors[a].total_cmp(&errors[b]))
        .unwrap_or(2);
    let offset = if at > 0 && at < 4 {
        let (y0, y1, y2) = (errors[at - 1], errors[at], errors[at + 1]);
        let denom = y0 - 2.0 * y1 + y2;
        if denom > 0.0 {
            (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };
    Ok((fine[at] + offset * FINE_STEP).max(MIN_SCALE))
}

// LDLQ over the regularized weight `w` (k, n) with the block LDL factor `l`: the encoding's values (k, n) and each
// tile's states (k / 16, n / 16, 256) on the host
fn ldlq(
    w: &Tensor,
    l: &Tensor,
    bits: Exl3Bits,
    codebook: Exl3Codebook,
) -> Result<(Tensor, Vec<i16>)> {
    let (k, n) = w.dims2()?;
    let tiles_n = n / TILE;
    let device = w.device();
    let perm = tile_order();
    let mut inverse = [0u32; TILE_VALUES];
    for (p, &cell) in perm.iter().enumerate() {
        inverse[cell] = p as u32;
    }
    let to_positions = Tensor::from_vec(perm.map(|c| c as u32).to_vec(), TILE_VALUES, device)?;
    let to_cells = Tensor::from_vec(inverse.to_vec(), TILE_VALUES, device)?;
    let cols = Tensor::arange(0u32, k as u32, device)?.unsqueeze(0)?;
    let mut feedback = Tensor::zeros((k, n), DType::F32, device)?;
    let mut rows_q = vec![None; k / TILE];
    let mut states = vec![0i16; k * n];
    for j in (SPAN_ROWS..=k).rev().step_by(SPAN_ROWS) {
        let i = j - SPAN_ROWS;
        let span_w = w.narrow(0, i, SPAN_ROWS)?;
        // L's rows of the span, each kept left of its own 16-row block (where the factor's values are)
        let starts = Tensor::from_vec(
            (i..j).map(|r| (r / TILE * TILE) as u32).collect::<Vec<_>>(),
            (SPAN_ROWS, 1),
            device,
        )?;
        let span_l =
            (l.narrow(0, i, SPAN_ROWS)? * cols.broadcast_lt(&starts)?.to_dtype(DType::F32)?)?;
        let span_feedback = feedback.narrow(0, i, SPAN_ROWS)?;
        let mut done: Vec<Tensor> = Vec::new();
        for bj in (TILE..=SPAN_ROWS).rev().step_by(TILE) {
            let bi = bj - TILE;
            let mut rows = (span_w.narrow(0, bi, TILE)? + span_feedback.narrow(0, bi, TILE)?)?;
            if bj < SPAN_ROWS {
                let below_q = Tensor::cat(&done.iter().rev().collect::<Vec<_>>(), 0)?;
                let err = (span_w.narrow(0, bj, SPAN_ROWS - bj)? - below_q)?;
                let l_block = span_l
                    .narrow(0, bj, SPAN_ROWS - bj)?
                    .narrow(1, i + bi, TILE)?;
                rows = (rows + l_block.t()?.contiguous()?.matmul(&err)?)?;
            }
            let tiles = rows
                .reshape((TILE, tiles_n, TILE))?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((tiles_n, TILE_VALUES))?
                .index_select(&to_positions, 1)?;
            let (values, tile_states) = quantize_tiles(&tiles, bits, codebook)?;
            let q = values
                .index_select(&to_cells, 1)?
                .reshape((tiles_n, TILE, TILE))?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((TILE, n))?;
            let tile_row = (i + bi) / TILE;
            let s = tile_states
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<i16>()?;
            states[tile_row * tiles_n * TILE_VALUES..(tile_row + 1) * tiles_n * TILE_VALUES]
                .copy_from_slice(&s);
            rows_q[tile_row] = Some(q.clone());
            done.push(q);
        }
        let span_q = Tensor::cat(&done.iter().rev().collect::<Vec<_>>(), 0)?;
        feedback = (feedback + span_l.t()?.contiguous()?.matmul(&(span_w - span_q)?)?)?;
    }
    let rows = rows_q
        .into_iter()
        .map(|r| r.expect("every tile row encoded"))
        .collect::<Vec<_>>();
    Ok((Tensor::cat(&rows, 0)?, states))
}

// tr(E H E^T) / tr(W H W^T) over the columns, for `h` and the weights (k, n)
fn output_error(w: &Tensor, q: &Tensor, h: &Tensor) -> Result<f64> {
    let quadratic = |x: &Tensor| -> Result<f64> {
        Ok(f64::from(
            (x * h.matmul(x)?)?.sum_all()?.to_scalar::<f32>()?,
        ))
    };
    Ok(quadratic(&(w - q)?)? / quadratic(w)?.max(ZERO))
}

/// `weight` `(out, in)` on a CUDA device encoded at `bits` on `codebook` against the input Hessian `hessian`
/// `(in, in)` (x^T x over the calibration rows, over their count); `seed` fixes the sign flips. Both dimensions are
/// multiples of 128.
pub fn encode(
    weight: &Tensor,
    hessian: &Tensor,
    bits: Exl3Bits,
    codebook: Exl3Codebook,
    seed: u64,
) -> Result<Exl3Encoding> {
    let device = weight.device().clone();
    let (n, k) = weight.dims2()?;
    if !k.is_multiple_of(HAD) || !n.is_multiple_of(HAD) || k < SPAN_ROWS {
        inference_tensor::bail!(
            "an EXL3 encoding needs both sides in multiples of {HAD}, not {n} x {k}"
        );
    }
    let had = hadamard_matrix(&device)?;
    let w_orig = weight.to_dtype(DType::F32)?.t()?.contiguous()?;
    let h = hessian
        .to_dtype(DType::F32)?
        .to_device(&device)?
        .contiguous()?;
    let h_mean = diagonal_mean(&h)?;
    let h_damped = add_diagonal(&h, SIGMA_REG * h_mean)?;
    let su_sign = signs(k, seed, &device)?;
    let sv_sign = signs(n, seed ^ 0x5bd1e995, &device)?;

    // H rotated as the inputs will be: P (S H S) P^T
    let h_rot = h_damped.broadcast_mul(&su_sign.unsqueeze(0)?)?;
    let h_rot = had_right(&h_rot, &had)?.broadcast_mul(&su_sign.unsqueeze(1)?)?;
    let l = factor(&had_left(&h_rot, &had)?)?;

    // output scales: each column's rms over their mean (zero columns stay zero)
    let mut out_scales = rms(&w_orig, 0)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let mean = out_scales.iter().map(|&s| f64::from(s)).sum::<f64>() / n as f64;
    if mean > ZERO {
        out_scales
            .iter_mut()
            .for_each(|s| *s = (f64::from(*s) / mean) as f32);
    }
    let zero_cols = floor_zeros(&mut out_scales, 0.1);
    let sv_signs = sv_sign.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let mut sv = sv_signs
        .iter()
        .zip(&out_scales)
        .map(|(s, o)| s * o + SCALE_EPS as f32)
        .collect::<Vec<_>>();
    let mut w = w_orig.broadcast_div(&Tensor::from_vec(sv.clone(), (1, n), &device)?)?;
    for (v, &zero) in sv.iter_mut().zip(&zero_cols) {
        if zero {
            *v = 0.0;
        }
    }
    w = had_right(&w, &had)?;
    // input scales: each row's rms, toward the codebook's spread
    let mut in_scales = rms(&w, 1)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    floor_zeros(&mut in_scales, 0.1);
    let su_signs = su_sign.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let mut su = su_signs
        .iter()
        .zip(&in_scales)
        .map(|(s, i)| (f64::from(s * i) / -CODEBOOK_SCALE + SCALE_EPS) as f32)
        .collect::<Vec<_>>();
    w = had_left(
        &w.broadcast_div(&Tensor::from_vec(su.clone(), (k, 1), &device)?)?,
        &had,
    )?;
    let g = global_scale(&w, bits, codebook)?;
    w = (w * g)?;
    su.iter_mut().for_each(|s| *s = (f64::from(*s) / g) as f32);

    let (q_reg, states) = ldlq(&w, &l, bits, codebook)?;

    // back to the original basis: undo the input rotation and scales, then the output ones
    let mut su_t = Tensor::from_vec(su, (k, 1), &device)?;
    let mut sv_t = Tensor::from_vec(sv, (1, n), &device)?;
    let mut q = had_left(&q_reg, &had)?.broadcast_mul(&su_t)?;
    q = had_right(&q, &had)?.broadcast_mul(&sv_t)?;

    // the stored scales refit to the encoding, output columns in closed form and input rows by a solve, alternated
    let hw = h_damped.matmul(&w_orig)?;
    for _ in 0..REFIT_ROUNDS {
        let num = (&q * &hw)?.sum(0)?;
        let den = (&q * h_damped.matmul(&q)?)?.sum(0)?;
        let c = num
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?
            .iter()
            .zip(den.to_device(&Device::Cpu)?.to_vec1::<f32>()?)
            .map(|(&a, d)| if f64::from(d) > ZERO { a / d } else { 1.0 })
            .collect::<Vec<_>>();
        let c = Tensor::from_vec(c, (1, n), &device)?;
        q = q.broadcast_mul(&c)?;
        sv_t = sv_t.broadcast_mul(&c)?;
        let a = (q.matmul(&q.t()?)? * &h_damped)?;
        let a = add_diagonal(&a, 1e-6 * diagonal_mean(&a)?)?;
        let b = (&q * &hw)?.sum(1)?;
        if let Some(r) = solve_spd(&a, &b)? {
            let r = r
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?
                .iter()
                .map(|&v| if v.is_finite() && v > 0.0 { v } else { 1.0 })
                .collect::<Vec<_>>();
            let r = Tensor::from_vec(r, (k, 1), &device)?;
            q = q.broadcast_mul(&r)?;
            su_t = su_t.broadcast_mul(&r)?;
        }
    }
    let proxy_error = output_error(&w_orig, &q, &h)?;

    let words = bits.words();
    let mut trellis = Vec::with_capacity(k * n / TILE_VALUES * words);
    for tile in states.chunks(TILE_VALUES) {
        let tile = tile.iter().map(|&s| s as u16).collect::<Vec<_>>();
        trellis.extend(pack_tile(&tile, bits).into_iter().map(|w| w as i16));
    }
    Ok(Exl3Encoding {
        trellis: Tensor::from_vec(trellis, (k / TILE, n / TILE, words), &Device::Cpu)?,
        suh: su_t.flatten_all()?.to_dtype(DType::F16)?,
        svh: sv_t.flatten_all()?.to_dtype(DType::F16)?,
        proxy_error,
    })
}

#[cfg(test)]
mod tests {
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::{encode, output_error};
    use crate::exl3::{Exl3Bits, Exl3Codebook, dense_weight};

    // a random linear against a Hessian of correlated inputs: each encoding decodes, through the loader's own
    // reconstruction, to a weight whose output error is the one reported, and more bits leave less of it
    #[test]
    fn encodings_decode_to_their_reported_error() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (n, k, rows) = (256, 256, 2048);
        let w = Tensor::randn(0f32, 0.02f32, (n, k), &device)?;
        let mix = Tensor::randn(0f32, 1f32, (k, k), &device)?;
        let x = Tensor::randn(0f32, 1f32, (rows, k), &device)?.matmul(&mix)?;
        let h = (x.t()?.matmul(&x)? / rows as f64)?;
        let mut errors = Vec::new();
        for (bits, codebook) in [
            (Exl3Bits::Whole(2), Exl3Codebook::ThreeInst),
            (Exl3Bits::Whole(3), Exl3Codebook::Mcg),
            (Exl3Bits::Whole(4), Exl3Codebook::ThreeInst),
            (Exl3Bits::Half(2), Exl3Codebook::Mul1),
        ] {
            let e = encode(&w, &h, bits, codebook, 7)?;
            let decoded = dense_weight(&e.trellis, &e.suh, &e.svh, codebook, DType::F32, &device)?;
            let measured = output_error(&w.t()?.contiguous()?, &decoded.t()?.contiguous()?, &h)?;
            assert!(
                (measured - e.proxy_error).abs() < 0.05 * e.proxy_error + 1e-6,
                "{bits:?}: reported {} measured {measured}",
                e.proxy_error
            );
            errors.push((bits, e.proxy_error));
        }
        assert!(
            errors[2].1 < errors[1].1 && errors[1].1 < errors[0].1,
            "{errors:?}"
        );
        Ok(())
    }
}
