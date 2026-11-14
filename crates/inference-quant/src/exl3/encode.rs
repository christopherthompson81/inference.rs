//! One linear to EXL3 after exllamav3's `quantize_exl3` (`modules/quant/exl3_lib/quantize.py`): the Hessian damped,
//! rotated by the input signs and 128-wide Hadamards and block-LDL factored; the weight regularized (output and input
//! scales, the same rotations, a global scale searched on sampled tiles), encoded by LDLQ (16 rows at a time, each
//! row of tiles through the trellis quantizer with the error so far fed forward through L), rotated back, and its
//! stored scales refit to the encoding in the Hessian metric.

use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;
use tracing::warn;

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
// the columns a full-size pass over a linear takes at once (a multiple of the 128-wide rotation)
const COLUMN_CHUNK: usize = 8192;
// the device memory a batch of experts' LDLQs may hold at once
const EXPERT_BATCH_BYTES: usize = 2 << 30;
// how allocation failures read: the driver's, and cuBLAS's or cuSOLVER's for their workspaces
const OUT_OF_MEMORY: [&str; 3] = ["out of memory", "out_of_memory", "alloc_failed"];
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
    /// The weight `(out, in)` the stored tensors decode to, in f32 at the padded sizes: on the encoding's device, or on
    /// the host for a linear encoded from system memory.
    pub decoded: Tensor,
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

// the mean square error of quantizing `tiles` scaled by each of `scales`, measured back at their own scale: every
// scale's copy in one launch, the quantizer's wave being as wide as a small linear's sample
fn scaled_errors(
    tiles: &Tensor,
    scales: &[f64],
    bits: Exl3Bits,
    codebook: Exl3Codebook,
) -> Result<Vec<f64>> {
    let count = tiles.dim(0)?;
    let factors = Tensor::from_vec(
        scales
            .iter()
            .flat_map(|&s| std::iter::repeat_n(s as f32, count))
            .collect::<Vec<_>>(),
        (scales.len() * count, 1),
        tiles.device(),
    )?;
    let scaled = tiles
        .unsqueeze(0)?
        .repeat((scales.len(), 1, 1))?
        .reshape((scales.len() * count, TILE_VALUES))?;
    let (q, _) = quantize_tiles(&scaled.broadcast_mul(&factors)?, bits, codebook)?;
    let errors = q
        .broadcast_div(&factors)?
        .sub(&scaled)?
        .sqr()?
        .reshape((scales.len(), count * TILE_VALUES))?
        .mean(1)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    Ok(errors.into_iter().map(f64::from).collect())
}

// the global scale whose quantization error over the sampled tiles is least: a coarse grid on a third of the
// sample, a fine one on all of it, refined by a parabola through the best three
fn global_scale(w: &Tensor, bits: Exl3Bits, codebook: Exl3Codebook, work: &Device) -> Result<f64> {
    let tiles = (sample_tiles(w)?.to_device(work)? * ldlq_drift(bits))?;
    let count = tiles.dim(0)?;
    let third = (0..count).step_by(3).map(|i| i as u32).collect::<Vec<_>>();
    let len = third.len();
    let sub = tiles.index_select(&Tensor::from_vec(third, len, work)?, 0)?;
    let coarse = scaled_errors(&sub, &COARSE_SCALES, bits, codebook)?;
    let mut best = (f64::INFINITY, COARSE_SCALES[0]);
    for (&e, &s) in coarse.iter().zip(&COARSE_SCALES) {
        if e < best.0 {
            best = (e, s);
        }
    }
    let fine = (0..5)
        .map(|i| best.1 + FINE_STEP * (i as f64 - 2.0))
        .collect::<Vec<_>>();
    let errors = scaled_errors(&tiles, &fine, bits, codebook)?;
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

// LDLQ over the regularized weights `w` (e, k, n), each with its block LDL factor of `l` (e, k, k), in lockstep: each
// row of tiles of every one through one quantizer launch. The encodings' values (e, k, n), and each one's tile
// states (k / 16, n / 16, 256) on the host, one after another
fn ldlq(
    w: &Tensor,
    l: &Tensor,
    bits: Exl3Bits,
    codebook: Exl3Codebook,
) -> Result<(Tensor, Vec<i16>)> {
    let (e, k, n) = w.dims3()?;
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
    let mut feedback = Tensor::zeros((e, k, n), DType::F32, device)?;
    let mut rows_q = vec![None; k / TILE];
    let mut states = vec![None; k / TILE];
    for j in (SPAN_ROWS..=k).rev().step_by(SPAN_ROWS) {
        let i = j - SPAN_ROWS;
        let span_w = w.narrow(1, i, SPAN_ROWS)?;
        // L's rows of the span, each kept left of its own 16-row block (where the factor's values are)
        let starts = Tensor::from_vec(
            (i..j).map(|r| (r / TILE * TILE) as u32).collect::<Vec<_>>(),
            (SPAN_ROWS, 1),
            device,
        )?;
        let span_l = l
            .narrow(1, i, SPAN_ROWS)?
            .broadcast_mul(&cols.broadcast_lt(&starts)?.to_dtype(DType::F32)?)?;
        let span_feedback = feedback.narrow(1, i, SPAN_ROWS)?;
        let mut done: Vec<Tensor> = Vec::new();
        for bj in (TILE..=SPAN_ROWS).rev().step_by(TILE) {
            let bi = bj - TILE;
            let mut rows = (span_w.narrow(1, bi, TILE)? + span_feedback.narrow(1, bi, TILE)?)?;
            if bj < SPAN_ROWS {
                let below_q = Tensor::cat(&done.iter().rev().collect::<Vec<_>>(), 1)?;
                let err = (span_w.narrow(1, bj, SPAN_ROWS - bj)? - below_q)?;
                let l_block = span_l
                    .narrow(1, bj, SPAN_ROWS - bj)?
                    .narrow(2, i + bi, TILE)?;
                rows = (rows + l_block.transpose(1, 2)?.contiguous()?.matmul(&err)?)?;
            }
            let tiles = rows
                .reshape((e, TILE, tiles_n, TILE))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((e * tiles_n, TILE_VALUES))?
                .index_select(&to_positions, 1)?;
            let (values, tile_states) = quantize_tiles(&tiles, bits, codebook)?;
            let q = values
                .index_select(&to_cells, 1)?
                .reshape((e, tiles_n, TILE, TILE))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((e, TILE, n))?;
            let tile_row = (i + bi) / TILE;
            // read back once at the end, so the rows queue without a sync each
            states[tile_row] = Some(tile_states.reshape((e, tiles_n, TILE_VALUES))?);
            rows_q[tile_row] = Some(q.clone());
            done.push(q);
        }
        let span_q = Tensor::cat(&done.iter().rev().collect::<Vec<_>>(), 1)?;
        feedback = (feedback
            + span_l
                .transpose(1, 2)?
                .contiguous()?
                .matmul(&(span_w - span_q)?)?)?;
    }
    let every = |rows: Vec<Option<Tensor>>| -> Vec<Tensor> {
        rows.into_iter()
            .map(|r| r.expect("every tile row encoded"))
            .collect()
    };
    let states = Tensor::cat(&every(states), 1)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<i16>()?;
    Ok((Tensor::cat(&every(rows_q), 1)?, states))
}

// LDLQ over `w` (e, k, n) a chunk of columns at a time (each column's encoding depends only on its own, given L),
// `w` dropped before the chunks join; states laid out as one `ldlq` over all of them gives them. Each chunk runs on
// L's device and its values join on `w`'s
fn ldlq_columns(
    w: Tensor,
    l: &Tensor,
    bits: Exl3Bits,
    codebook: Exl3Codebook,
) -> Result<(Tensor, Vec<i16>)> {
    let (e, k, n) = w.dims3()?;
    let (work, park) = (l.device().clone(), w.device().clone());
    if n <= COLUMN_CHUNK {
        let (q, states) = ldlq(&w.to_device(&work)?, l, bits, codebook)?;
        return Ok((q.to_device(&park)?, states));
    }
    let (tile_rows, tiles_n) = (k / TILE, n / TILE);
    let mut states = vec![0i16; e * k * n];
    let mut values = Vec::new();
    for (c0, len) in column_chunks(n) {
        let w_c = w.narrow(2, c0, len)?.contiguous()?.to_device(&work)?;
        let (q, chunk) = ldlq(&w_c, l, bits, codebook)?;
        drop(w_c);
        let row = len / TILE * TILE_VALUES;
        for (i, block) in chunk.chunks(row).enumerate() {
            let at = (i * tiles_n + c0 / TILE) * TILE_VALUES;
            states[at..at + row].copy_from_slice(block);
        }
        debug_assert_eq!(chunk.len(), e * tile_rows * row);
        values.push(q.to_device(&park)?);
    }
    drop(w);
    Ok((Tensor::cat(&values, 2)?, states))
}

// tr(E H E^T) / tr(W H W^T) over the columns, for `h` and the weights (k, n), a chunk of columns at a time on `h`'s
// device
fn output_error(w: &Tensor, q: &Tensor, h: &Tensor) -> Result<f64> {
    let quadratic = |x: &Tensor| -> Result<f64> {
        Ok(f64::from(
            (x * h.matmul(x)?)?.sum_all()?.to_scalar::<f32>()?,
        ))
    };
    let (mut err, mut norm) = (0.0, 0.0);
    for (c0, len) in column_chunks(w.dim(1)?) {
        let w_c = w.narrow(1, c0, len)?.contiguous()?.to_device(h.device())?;
        let q_c = q.narrow(1, c0, len)?.contiguous()?.to_device(h.device())?;
        err += quadratic(&(&w_c - q_c)?)?;
        norm += quadratic(&w_c)?;
    }
    Ok(err / norm.max(ZERO))
}

// one linear up to its LDLQ (padded, its Hessian damped and factored), and its weight regularized and scaled. Its
// full-size tensors live on `park`: the device that encodes it (`work`), or the host when they do not fit there
struct Prepared {
    n: usize,
    k: usize,
    work: Device,
    had: Tensor,
    w_orig: Tensor,
    h: Tensor,
    h_damped: Tensor,
    su: Vec<f32>,
    sv: Vec<f32>,
}

fn prepare(
    weight: &Tensor,
    hessian: &Tensor,
    bits: Exl3Bits,
    codebook: Exl3Codebook,
    seed: u64,
    park: &Device,
) -> Result<(Prepared, Tensor, Tensor)> {
    let device = weight.device().clone();
    let (n, k) = weight.dims2()?;
    // zero rows and columns are inert: padded inputs are zero, padded outputs get zero scales
    let (n, k, pad_n, pad_k) = (
        n.next_multiple_of(HAD),
        k.next_multiple_of(HAD),
        n.next_multiple_of(HAD) - n,
        k.next_multiple_of(HAD) - k,
    );
    let weight = weight
        .to_device(park)?
        .pad_with_zeros(0, 0, pad_n)?
        .pad_with_zeros(1, 0, pad_k)?;
    let hessian = hessian
        .pad_with_zeros(0, 0, pad_k)?
        .pad_with_zeros(1, 0, pad_k)?;
    let had = hadamard_matrix(&device)?;
    let had_park = hadamard_matrix(park)?;
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
    drop(h_rot);

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
    let mut w = w_orig.broadcast_div(&Tensor::from_vec(sv.clone(), (1, n), park)?)?;
    for (v, &zero) in sv.iter_mut().zip(&zero_cols) {
        if zero {
            *v = 0.0;
        }
    }
    w = had_right(&w, &had_park)?;
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
        &w.broadcast_div(&Tensor::from_vec(su.clone(), (k, 1), park)?)?,
        &had_park,
    )?;
    let g = global_scale(&w, bits, codebook, &device)?;
    w = (w * g)?;
    su.iter_mut().for_each(|s| *s = (f64::from(*s) / g) as f32);
    Ok((
        Prepared {
            n,
            k,
            work: device,
            had: had_park,
            w_orig,
            h: h.to_device(park)?,
            h_damped: h_damped.to_device(park)?,
            su,
            sv,
        },
        w,
        l,
    ))
}

// the column ranges a full-size pass takes at a time, so no `(k, n)` temporary beyond the weights is ever held
fn column_chunks(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n)
        .step_by(COLUMN_CHUNK)
        .map(move |c| (c, COLUMN_CHUNK.min(n - c)))
}

// a prepared linear's LDLQ values and tile states back to its stored tensors
fn finish(p: Prepared, q_reg: Tensor, states: &[i16], bits: Exl3Bits) -> Result<Exl3Encoding> {
    let Prepared {
        n,
        k,
        work,
        had,
        w_orig,
        h,
        h_damped,
        su,
        sv,
    } = p;
    let device = q_reg.device().clone();
    let h_damped = h_damped.to_device(&work)?;
    let on_work =
        |t: &Tensor, c0: usize, len: usize| t.narrow(1, c0, len)?.contiguous()?.to_device(&work);
    let cpu = |t: Tensor| t.to_device(&Device::Cpu)?.to_vec1::<f32>();
    // back to the original basis: undo the input rotation and scales, then the output ones
    let su0 = Tensor::from_vec(su, (k, 1), &device)?;
    let sv0 = Tensor::from_vec(sv, (1, n), &device)?;
    let mut q = had_left(&q_reg, &had)?.broadcast_mul(&su0)?;
    drop(q_reg);
    q = had_right(&q, &had)?.broadcast_mul(&sv0)?;
    let (mut su_t, mut sv_t) = (su0, sv0);
    // the stored scales refit to the encoding, output columns in closed form and input rows by a solve, alternated
    for _ in 0..REFIT_ROUNDS {
        let mut c = Vec::with_capacity(n);
        for (c0, len) in column_chunks(n) {
            let q_c = on_work(&q, c0, len)?;
            let hw_c = h_damped.matmul(&on_work(&w_orig, c0, len)?)?;
            let num = cpu((&q_c * &hw_c)?.sum(0)?)?;
            let den = cpu((&q_c * h_damped.matmul(&q_c)?)?.sum(0)?)?;
            c.extend(
                num.iter()
                    .zip(den)
                    .map(|(&a, d)| if f64::from(d) > ZERO { a / d } else { 1.0 }),
            );
        }
        let c = Tensor::from_vec(c, (1, n), &device)?;
        q = q.broadcast_mul(&c)?;
        sv_t = sv_t.broadcast_mul(&c)?;
        let mut a = Tensor::zeros((k, k), DType::F32, &work)?;
        let mut b = Tensor::zeros(k, DType::F32, &work)?;
        for (c0, len) in column_chunks(n) {
            let q_c = on_work(&q, c0, len)?;
            let hw_c = h_damped.matmul(&on_work(&w_orig, c0, len)?)?;
            a = (a + q_c.matmul(&q_c.t()?)?)?;
            b = (b + (&q_c * &hw_c)?.sum(1)?)?;
        }
        let a = (a * &h_damped)?;
        let a = add_diagonal(&a, 1e-6 * diagonal_mean(&a)?)?;
        if let Some(r) = solve_spd(&a, &b)? {
            let r = cpu(r)?
                .iter()
                .map(|&v| if v.is_finite() && v > 0.0 { v } else { 1.0 })
                .collect::<Vec<_>>();
            let r = Tensor::from_vec(r, (k, 1), &device)?;
            q = q.broadcast_mul(&r)?;
            su_t = su_t.broadcast_mul(&r)?;
        }
    }
    drop(h_damped);
    let proxy_error = output_error(&w_orig, &q, &h.to_device(&work)?)?;
    drop(w_orig);

    // the weight as the stored tensors decode: the refit scales commute with the rotations, so only their f16
    // rounding separates it from `q`
    let (suh, svh) = (su_t.to_dtype(DType::F16)?, sv_t.to_dtype(DType::F16)?);
    let decoded = q
        .broadcast_mul(&(suh.to_dtype(DType::F32)? / &su_t)?)?
        .broadcast_mul(&(svh.to_dtype(DType::F32)? / &sv_t)?)?;
    drop(q);
    let decoded = decoded.t()?.contiguous()?;

    let words = bits.words();
    let trellis = states
        .par_chunks(TILE_VALUES)
        .flat_map_iter(|tile| {
            let tile = tile.iter().map(|&s| s as u16).collect::<Vec<_>>();
            pack_tile(&tile, bits).into_iter().map(|w| w as i16)
        })
        .collect::<Vec<_>>();
    Ok(Exl3Encoding {
        trellis: Tensor::from_vec(trellis, (k / TILE, n / TILE, words), &Device::Cpu)?,
        suh: suh.flatten_all()?,
        svh: svh.flatten_all()?,
        decoded,
        proxy_error,
    })
}

/// `weight` `(out, in)` on a CUDA device encoded at `bits` on `codebook` against the input Hessian `hessian`
/// `(in, in)` (x^T x over the calibration rows, over their count); `seed` fixes the sign flips. A side not in 128-wide
/// blocks is zero-padded to them, as exllamav3 stores such a linear: the trellis, `suh` and `svh` at the padded sizes.
pub fn encode(
    weight: &Tensor,
    hessian: &Tensor,
    bits: Exl3Bits,
    codebook: Exl3Codebook,
    seed: u64,
) -> Result<Exl3Encoding> {
    Ok(encode_experts(&[(weight, hessian, seed)], bits, codebook)?
        .pop()
        .expect("one encoding a linear"))
}

/// Linears of one shape (a MoE layer's experts of one projection), each `(weight, hessian, seed)` as [`encode`]
/// takes them, encoded together: their LDLQs run in lockstep, every row of tiles of all of them through one quantizer
/// launch, which a small linear's row alone leaves mostly idle. Each encoding is the one [`encode`] gives it. A linear
/// whose encoding does not fit the device's memory is encoded with its full-size tensors in system memory.
pub fn encode_experts(
    linears: &[(&Tensor, &Tensor, u64)],
    bits: Exl3Bits,
    codebook: Exl3Codebook,
) -> Result<Vec<Exl3Encoding>> {
    let Some(&(first, _, _)) = linears.first() else {
        return Ok(Vec::new());
    };
    let (n, k) = first.dims2()?;
    let (n, k) = (n.next_multiple_of(HAD), k.next_multiple_of(HAD));
    // a batch's Hessians (as given and damped), L, weights, feedback and values in f32
    let per_linear = (3 * k * k + 3 * k * n) * 4;
    let batch = (EXPERT_BATCH_BYTES / per_linear).max(1);
    let mut encodings = Vec::with_capacity(linears.len());
    for chunk in linears.chunks(batch) {
        encodings.extend(encode_fitting(chunk, bits, codebook)?);
    }
    Ok(encodings)
}

fn is_out_of_memory(e: &inference_tensor::Error) -> bool {
    let message = e.to_string().to_ascii_lowercase();
    OUT_OF_MEMORY.iter().any(|m| message.contains(m))
}

// `linears` encoded together on their device; a batch out of device memory splits in half, and a linear alone that
// is gets its full-size tensors kept in system memory, a chunk of columns at a time on the device
fn encode_fitting(
    linears: &[(&Tensor, &Tensor, u64)],
    bits: Exl3Bits,
    codebook: Exl3Codebook,
) -> Result<Vec<Exl3Encoding>> {
    let work = linears[0].0.device().clone();
    match encode_batch(linears, bits, codebook, &work) {
        Err(e) if is_out_of_memory(&e) && !work.is_cpu() => {
            if linears.len() > 1 {
                let (a, b) = linears.split_at(linears.len() / 2);
                let mut encodings = encode_fitting(a, bits, codebook)?;
                encodings.extend(encode_fitting(b, bits, codebook)?);
                return Ok(encodings);
            }
            let (n, k) = linears[0].0.dims2()?;
            warn!(
                "Out of device memory encoding a {n}x{k} linear; retrying it from system memory."
            );
            encode_batch(linears, bits, codebook, &Device::Cpu)
        }
        r => r,
    }
}

// `linears` encoded in lockstep on their device, their full-size tensors on `park`
fn encode_batch(
    linears: &[(&Tensor, &Tensor, u64)],
    bits: Exl3Bits,
    codebook: Exl3Codebook,
    park: &Device,
) -> Result<Vec<Exl3Encoding>> {
    let mut prepared = Vec::with_capacity(linears.len());
    let (mut ws, mut ls) = (Vec::new(), Vec::new());
    for &(weight, hessian, seed) in linears {
        let (p, w, l) = prepare(weight, hessian, bits, codebook, seed, park)?;
        prepared.push(p);
        ws.push(w);
        ls.push(l);
    }
    let (n, k) = (prepared[0].n, prepared[0].k);
    // one linear is its own batch, without the copy a stack makes
    let batched = |mut ts: Vec<Tensor>| -> Result<Tensor> {
        if ts.len() == 1 {
            ts.pop().expect("one tensor").unsqueeze(0)
        } else {
            Tensor::stack(&ts, 0)
        }
    };
    let w = batched(ws)?;
    let l = batched(ls)?;
    let (q, states) = ldlq_columns(w, &l, bits, codebook)?;
    drop(l);
    let qs = if prepared.len() == 1 {
        vec![q.squeeze(0)?]
    } else {
        (0..prepared.len())
            .map(|e| q.get(e))
            .collect::<Result<Vec<_>>>()?
    };
    drop(q);
    prepared
        .into_iter()
        .zip(qs)
        .zip(states.chunks(k * n))
        .map(|((p, q), states)| finish(p, q, states, bits))
        .collect()
}

#[cfg(test)]
mod tests {
    use inference_tensor::{DType, Device, Result, Tensor};

    use super::{
        COLUMN_CHUNK, encode, encode_batch, encode_experts, factor, is_out_of_memory, ldlq,
        ldlq_columns, output_error,
    };
    use crate::exl3::{Exl3Bits, Exl3Codebook, dense_weight};

    // a linear wider than a chunk of columns runs its LDLQ a chunk at a time, to what one LDLQ over all of them gives
    #[test]
    fn ldlq_takes_wide_linears_a_chunk_of_columns_at_a_time() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (k, n, rows) = (256, COLUMN_CHUNK + 256, 1024);
        let x = Tensor::randn(0f32, 1f32, (rows, k), &device)?;
        let l =
            factor(&((x.t()?.matmul(&x)? / rows as f64)? + Tensor::eye(k, DType::F32, &device)?)?)?
                .unsqueeze(0)?;
        let w = Tensor::randn(0f32, 1f32, (1, k, n), &device)?;
        let (q, states) = ldlq(&w, &l, Exl3Bits::Whole(3), Exl3Codebook::Mcg)?;
        let (q_c, states_c) = ldlq_columns(w, &l, Exl3Bits::Whole(3), Exl3Codebook::Mcg)?;
        let same = states.iter().zip(&states_c).filter(|(a, b)| a == b).count();
        assert!(
            same * 1000 >= states.len() * 999,
            "{same} of {} states agree",
            states.len()
        );
        let off = (q - q_c)?.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(off < 1e-3, "the values differ by {off} on average");
        Ok(())
    }

    // experts encoded in lockstep get the encodings each gets alone
    #[test]
    fn experts_encode_together_as_each_alone() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (n, k, rows) = (128, 256, 1024);
        let experts = (0..3)
            .map(|_| -> Result<_> {
                let w = Tensor::randn(0f32, 0.02f32, (n, k), &device)?;
                let x = Tensor::randn(0f32, 1f32, (rows, k), &device)?;
                Ok((w, (x.t()?.matmul(&x)? / rows as f64)?))
            })
            .collect::<Result<Vec<_>>>()?;
        let linears = experts
            .iter()
            .zip(11u64..)
            .map(|((w, h), seed)| (w, h, seed))
            .collect::<Vec<_>>();
        let together = encode_experts(&linears, Exl3Bits::Whole(3), Exl3Codebook::Mul1)?;
        for (&(w, h, seed), t) in linears.iter().zip(&together) {
            let alone = encode(w, h, Exl3Bits::Whole(3), Exl3Codebook::Mul1, seed)?;
            let words = |e: &super::Exl3Encoding| e.trellis.flatten_all()?.to_vec1::<i16>();
            assert_eq!(words(&alone)?, words(t)?);
            assert_eq!(alone.proxy_error, t.proxy_error);
        }
        Ok(())
    }

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
            let off = (&decoded - &e.decoded)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let norm = decoded.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                off <= 1e-5 * norm,
                "{bits:?}: the encoder's decode is off the loader's by {off}"
            );
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

    // a linear encoded with its full-size tensors on the host (wide enough to take its columns in chunks) is encoded
    // as well as on the device: the host's rounding moves some of the quantizer's choices, not the error
    #[test]
    fn a_linear_encodes_from_system_memory_as_on_the_device() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (n, k, rows) = (COLUMN_CHUNK + 256, 256, 1024);
        let w = Tensor::randn(0f32, 0.02f32, (n, k), &device)?;
        let x = Tensor::randn(0f32, 1f32, (rows, k), &device)?;
        let h = (x.t()?.matmul(&x)? / rows as f64)?;
        let linear = [(&w, &h, 7u64)];
        let (bits, codebook) = (Exl3Bits::Whole(3), Exl3Codebook::Mul1);
        let on_device = encode_batch(&linear, bits, codebook, &device)?.remove(0);
        let on_host = encode_batch(&linear, bits, codebook, &Device::Cpu)?.remove(0);
        assert!(on_host.decoded.device().is_cpu());
        let words = |e: &super::Exl3Encoding| e.trellis.flatten_all()?.to_vec1::<i16>();
        let (a, b) = (words(&on_device)?, words(&on_host)?);
        let same = a.iter().zip(&b).filter(|(a, b)| a == b).count();
        assert!(
            same * 10 >= a.len() * 9,
            "{same} of {} words agree",
            a.len()
        );
        let (e_d, e_h) = (on_device.proxy_error, on_host.proxy_error);
        assert!(
            (e_d - e_h).abs() <= 1e-3 * e_d,
            "proxy errors {e_d} and {e_h}"
        );
        let off = (on_device.decoded.to_device(&Device::Cpu)? - on_host.decoded)?
            .abs()?
            .mean_all()?
            .to_scalar::<f32>()?;
        let scale = w.abs()?.mean_all()?.to_scalar::<f32>()?;
        assert!(off < 0.05 * scale, "decodes differ by {off} on average");
        Ok(())
    }

    // with the device all but full, a linear larger than what is left encodes from system memory instead of failing
    #[test]
    #[ignore = "fills the device's memory, which the GPU tests beside it share"]
    fn a_linear_too_large_for_the_device_left_encodes_from_system_memory() -> Result<()> {
        const BALLAST: usize = 256 << 20;
        const LEFT: usize = 6;
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (n, k, rows) = (32768, 4096, 4096);
        let w = Tensor::randn(0f32, 0.02f32, (n, k), &device)?;
        let x = Tensor::randn(0f32, 1f32, (rows, k), &device)?;
        let h = (x.t()?.matmul(&x)? / rows as f64)?;
        drop(x);
        let mut ballast = Vec::new();
        loop {
            match Tensor::zeros(BALLAST / 4, DType::F32, &device) {
                Ok(t) => ballast.push(t),
                Err(e) if is_out_of_memory(&e) => break,
                Err(e) => return Err(e),
            }
        }
        ballast.truncate(ballast.len().saturating_sub(LEFT));
        let encoding = encode(&w, &h, Exl3Bits::Whole(4), Exl3Codebook::Mul1, 3)?;
        drop(ballast);
        assert!(
            encoding.decoded.device().is_cpu(),
            "the encoding fit the device"
        );
        assert!(
            encoding.proxy_error < 0.05,
            "proxy error {}",
            encoding.proxy_error
        );
        Ok(())
    }

    // a linear in other widths pads to 128 blocks: its cropped decode carries the reported error, and the kernels
    // layer built at its own widths (input padded, output cut) multiplies as that decode does
    #[test]
    fn padded_linears_encode_and_multiply_at_their_own_widths() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let (n, k, rows) = (200, 176, 1024);
        let w = Tensor::randn(0f32, 0.02f32, (n, k), &device)?;
        let x = Tensor::randn(0f32, 1f32, (rows, k), &device)?;
        let h = (x.t()?.matmul(&x)? / rows as f64)?;
        let e = encode(&w, &h, Exl3Bits::Whole(4), Exl3Codebook::Mcg, 3)?;
        assert_eq!(e.trellis.dims()[..2], [16, 16]);
        let decoded = dense_weight(
            &e.trellis,
            &e.suh,
            &e.svh,
            Exl3Codebook::Mcg,
            DType::F32,
            &device,
        )?
        .narrow(0, 0, n)?
        .narrow(1, 0, k)?
        .contiguous()?;
        let measured = output_error(&w.t()?.contiguous()?, &decoded.t()?.contiguous()?, &h)?;
        assert!(
            (measured - e.proxy_error).abs() < 0.05 * e.proxy_error + 1e-6,
            "reported {} measured {measured}",
            e.proxy_error
        );
        let layer = crate::exl3::Exl3Layer::new_on_device(
            e.trellis.to_device(&device)?,
            e.suh.clone(),
            e.svh.clone(),
            Exl3Codebook::Mcg,
            None,
            DType::F32,
        )?
        .with_features(k, n)?;
        let xs = x.narrow(0, 0, 4)?;
        let actual = crate::QuantMethod::forward(&layer, &xs)?;
        let expected = xs.matmul(&decoded.t()?)?;
        let err = (&actual - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let norm = expected.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(actual.dims(), [4, n]);
        assert!(
            err / norm < 2e-3,
            "the padded layer is off its decode by {}",
            err / norm
        );
        Ok(())
    }
}
