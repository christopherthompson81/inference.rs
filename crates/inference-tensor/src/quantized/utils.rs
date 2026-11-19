use super::k_quants::Quantizer;

// ggml's: the float added to 1.5 * 2^23 rounds to the nearest integer, ties to even
const ROUNDING_MAGIC: f32 = 12_582_912.0;
// ggml's `GROUP_MAX_EPS`: a group whose largest magnitude is below it quantizes to zeros
const GROUP_MAX_EPS: f32 = 1e-15;
// ik_llama.cpp's affine fits treat a range narrower than this as flat
const QKX_FLAT_RANGE: f32 = 1e-10;
// ik_llama.cpp's `make_qp_quants` reads a largest scale below this as all zero
const QP_ZERO_IK: f32 = 1e-16;
// the scale offsets `make_qx_quants` tries, in tenths of a step either side of `nmax`
const QX_SCALE_TRIES: i32 = 9;

/// The scale search of `make_qkx2_quants` / `make_qkx3_quants`: `nstep` scales past the min-max one, from `rmin` in
/// steps of `rdelta`, scored by squared error or, with `use_mad`, absolute error.
#[derive(Clone, Copy)]
pub(super) struct QkxSearch {
    pub rmin: f32,
    pub rdelta: f32,
    pub nstep: usize,
    pub use_mad: bool,
}

/// ggml's search for every imatrix-weighted affine type.
pub(super) const QKX3_SEARCH: QkxSearch = QkxSearch {
    rmin: -0.9,
    rdelta: 0.05,
    nstep: 36,
    use_mad: false,
};

pub(super) fn nearest_int(v: f32) -> i32 {
    let v = v + ROUNDING_MAGIC;
    (v.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// `a * b + c` as `quantizer`'s build computes it: ik_llama.cpp's (gcc, -march=native) fuses it, ggml's base library
/// (no FMA target) rounds the product first.
#[inline(always)]
pub(super) fn madd(quantizer: Quantizer, a: f32, b: f32, c: f32) -> f32 {
    match quantizer {
        Quantizer::IkLlama => a.mul_add(b, c),
        Quantizer::LlamaCpp => a * b + c,
    }
}

/// `a * b - c * d` as `quantizer`'s build computes it: ik_llama.cpp's fuses the first product into the subtraction.
#[inline(always)]
pub(super) fn msub(quantizer: Quantizer, a: f32, b: f32, c: f32, d: f32) -> f32 {
    match quantizer {
        Quantizer::IkLlama => a.mul_add(b, -(c * d)),
        Quantizer::LlamaCpp => a * b - c * d,
    }
}

// ik_llama.cpp's f64 `a * b - c * d`, the first product fused
#[inline(always)]
fn msub64(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a.mul_add(b, -(c * d))
}

// ik_llama.cpp's f64 `a * b + c * d`, the second product fused
#[inline(always)]
fn madd2_64(a: f64, b: f64, c: f64, d: f64) -> f64 {
    c.mul_add(d, a * b)
}

/// `nearest_int(a * b)`, the product fused into the rounding add where `quantizer`'s build fuses it.
#[inline(always)]
pub(super) fn nearest_int_mul(quantizer: Quantizer, a: f32, b: f32) -> i32 {
    let v = madd(quantizer, a, b, ROUNDING_MAGIC);
    (v.to_bits() & 0x007f_ffff) as i32 - 0x0040_0000
}

/// Validates that the input and output are the right size and returns an iterator which maps each
/// input region `xs` to its corresponding output block in `ys`. Each output region is guaranteed
/// to be `T::BLCK_SIZE` long.
pub(super) fn group_for_quantization<'a, 'b, T: super::k_quants::GgmlType>(
    xs: &'b [f32],
    ys: &'a mut [T],
) -> Vec<(&'a mut T, &'b [f32])> {
    let block_size = T::BLCK_SIZE;
    let dtype = T::DTYPE;

    let expected_blocks = xs.len() / block_size;
    let actual_blocks = ys.len();

    // Validate that the input is the right size
    debug_assert_eq!(
        expected_blocks, actual_blocks,
        "quantize {dtype:?}: expected {expected_blocks} blocks but only {actual_blocks} were provided!"
    );

    ys.iter_mut().zip(xs.chunks_exact(block_size)).collect()
}

/// Validates that the input and output are the right size and returns an iterator which maps each
/// input block `xs` to its corresponding output region in `ys`. Each output region is guaranteed
/// to be `T::BLCK_SIZE` long.
pub(super) fn group_for_dequantization<'a, 'b, T: super::k_quants::GgmlType>(
    xs: &'a [T],
    ys: &'b mut [f32],
) -> Vec<(&'a T, &'b mut [f32])> {
    let block_size = T::BLCK_SIZE;
    let dtype = T::DTYPE;

    let actual_output_len = ys.len();
    let expected_output_len = xs.len() * block_size;
    // Validate that the output is the right size
    debug_assert_eq!(
        expected_output_len, actual_output_len,
        "dequantize {dtype:?}: ys (len = {actual_output_len}) does not match the expected length of {expected_output_len}!"
    );

    // Zip the blocks and outputs together
    xs.iter().zip(ys.chunks_exact_mut(block_size)).collect()
}

pub(super) fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// ggml's `make_qx_quants`: the scale (and `ls`, offset by `nmax`) minimizing the weighted error of `x` over
/// `[-nmax, nmax)`, from the plain max scale and nineteen nudges of it; `rmse_type` 0 takes the max scale as it is, a
/// negative one averages the first fit with it. ik_llama.cpp's also tries each nudge from the positive end, then moves
/// single values a step while that lowers the error.
pub(super) unsafe fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: *const f32,
    ls: *mut i8,
    rmse_type: i32,
    qw: *const f32,
    quantizer: Quantizer,
) -> f32 {
    unsafe {
        let x = std::slice::from_raw_parts(x, n);
        let ls = std::slice::from_raw_parts_mut(ls, n);
        let qw = (!qw.is_null()).then(|| std::slice::from_raw_parts(qw, n));
        let q = quantizer;
        let mut max = 0f32;
        let mut amax = 0f32;
        for &v in x {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax < GROUP_MAX_EPS {
            ls.fill(0);
            return 0.;
        }
        let quantize = |iscale: f32, v: f32| nearest_int_mul(q, iscale, v).clamp(-nmax, nmax - 1);
        let iscale = -(nmax as f32) / max;
        if rmse_type == 0 {
            for (l, &v) in ls.iter_mut().zip(x) {
                *l = (nmax + quantize(iscale, v)) as i8;
            }
            return 1.0 / iscale;
        }
        let return_early = rmse_type < 0;
        let rmse_type = rmse_type.abs();
        let weight = |i: usize| match qw {
            Some(qw) => qw[i],
            None => match rmse_type {
                1 => x[i] * x[i],
                2 => 1.0,
                3 => x[i].abs(),
                _ => x[i].abs().sqrt(),
            },
        };
        // the weighted sums of l*x and l^2 for one scale; ik_llama.cpp's build fuses them in the first pass only
        let sums = |iscale: f32, fused: Quantizer| {
            let (mut sumlx, mut suml2) = (0f32, 0f32);
            for (i, &xi) in x.iter().enumerate() {
                let l = quantize(iscale, xi) as f32;
                let w = weight(i);
                sumlx = madd(fused, w * xi, l, sumlx);
                suml2 = madd(fused, w * l, l, suml2);
            }
            (sumlx, suml2)
        };
        let (mut sumlx, mut suml2) = sums(iscale, q);
        for (l, &v) in ls.iter_mut().zip(x) {
            *l = (nmax + quantize(iscale, v)) as i8;
        }
        let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
        if return_early {
            return if suml2 > 0.0 {
                0.5 * (scale + 1.0 / iscale)
            } else {
                1.0 / iscale
            };
        }
        let mut best = scale * sumlx;
        let (mut best_sumlx, mut best_suml2) = (sumlx, suml2);
        let ik = q == Quantizer::IkLlama;
        for is in -QX_SCALE_TRIES..=QX_SCALE_TRIES {
            let tenths = is as f32;
            let candidates = [
                (is != 0 || ik).then(|| -madd(q, 0.1, tenths, nmax as f32) / max),
                ik.then(|| madd(q, 0.1, tenths, (nmax - 1) as f32) / max),
            ];
            for iscale in candidates.into_iter().flatten() {
                (sumlx, suml2) = sums(iscale, Quantizer::LlamaCpp);
                if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
                    for (l, &v) in ls.iter_mut().zip(x) {
                        *l = (nmax + quantize(iscale, v)) as i8;
                    }
                    scale = sumlx / suml2;
                    best = scale * sumlx;
                    (best_sumlx, best_suml2) = (sumlx, suml2);
                }
            }
        }
        if !ik {
            return scale;
        }
        // a value a step up or down along the largest weighted gradient, while that improves the fit
        (sumlx, suml2) = (best_sumlx, best_suml2);
        for _ in 0..n as i32 * (2 * nmax - 1) {
            let (mut abs_gmax, mut gmax) = (0f32, 0f32);
            let mut best_j = None;
            for (j, &xj) in x.iter().enumerate() {
                let l = ls[j] as i32 - nmax;
                let g = scale * weight(j) * madd(q, -scale, l as f32, xj);
                let movable = (g > 0.0 && l < nmax - 1) || (g < 0.0 && l > -nmax);
                if movable && g.abs() > abs_gmax {
                    abs_gmax = g.abs();
                    gmax = g;
                    best_j = Some(j);
                }
            }
            let Some(j) = best_j else { break };
            let w = weight(j);
            let mut l = ls[j] as i32 - nmax;
            // ik_llama.cpp's build rounds `w * x` before the step's sum, and fuses the level's
            let (new_sumlx, new_suml2) = if gmax > 0.0 {
                l += 1;
                (sumlx + w * x[j], madd(q, w, (2 * l - 1) as f32, suml2))
            } else {
                l -= 1;
                (sumlx - w * x[j], madd(q, -w, (2 * l + 1) as f32, suml2))
            };
            if new_suml2 > 0.0 && new_sumlx * new_sumlx > best * new_suml2 {
                (sumlx, suml2) = (new_sumlx, new_suml2);
                scale = sumlx / suml2;
                best = scale * sumlx;
                ls[j] = (l + nmax) as i8;
            } else {
                break;
            }
        }
        scale
    }
}

/// ggml's `make_qkx2_quants`: an affine fit (scale, and the min it subtracts, returned negated as ggml's `the_min`) of
/// `x` onto `[0, nmax]` under `weights`, from the min-max fit and the scales of `search`.
pub(super) fn make_qkx2_quants(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    search: QkxSearch,
    l: &mut [u8],
    quantizer: Quantizer,
) -> (f32, f32) {
    let QkxSearch {
        rmin,
        rdelta,
        nstep,
        use_mad,
    } = search;
    let q = quantizer;
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x = madd(q, weights[i], x[i], sum_x);
    }
    if min > 0.0 {
        min = 0.0;
    }
    let flat = match q {
        Quantizer::LlamaCpp => max == min,
        Quantizer::IkLlama => max - min < QKX_FLAT_RANGE,
    };
    if flat {
        l.fill(0);
        return (0.0, -min);
    }
    let err = |diff: f32| if use_mad { diff.abs() } else { diff * diff };
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0f32;
    for i in 0..n {
        let li = nearest_int_mul(q, iscale, x[i] - min).clamp(0, nmax);
        l[i] = li as u8;
        best_error = madd(
            q,
            weights[i],
            err(madd(q, scale, li as f32, min) - x[i]),
            best_error,
        );
    }
    if nstep == 0 {
        return (scale, -min);
    }
    let mut laux = vec![0i32; n];
    for is in 0..=nstep {
        iscale = (madd(q, rdelta, is as f32, rmin) + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let li = nearest_int_mul(q, iscale, x[i] - min).clamp(0, nmax);
            laux[i] = li;
            let (w, l) = (weights[i], li as f32);
            // gcc shares `w * l` among the three sums, so only the two that multiply it again fuse
            let wl = w * l;
            sum_l += wl;
            sum_l2 = madd(q, wl, l, sum_l2);
            sum_xl = madd(q, wl, x[i], sum_xl);
        }
        let d = msub(q, sum_w, sum_l2, sum_l, sum_l);
        if d > 0.0 {
            // gcc fuses this one's second product: -(c * d - a * b) is that, exactly
            let mut this_scale = -msub(q, sum_x, sum_l, sum_w, sum_xl) / d;
            let mut this_min = msub(q, sum_l2, sum_x, sum_l, sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0f32;
            for i in 0..n {
                let diff = madd(q, this_scale, laux[i] as f32, this_min) - x[i];
                cur_error = madd(q, weights[i], err(diff), cur_error);
            }
            if cur_error < best_error {
                for (l, &a) in l.iter_mut().zip(&laux) {
                    *l = a as u8;
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L165
pub(super) fn make_q3_quants(x: &[f32], nmax: i32, do_rmse: bool, quantizer: Quantizer) -> f32 {
    let q = quantizer;
    let n = x.len();
    let mut l = vec![0i8; n];

    let mut max = 0.0;
    let mut amax = 0.0;
    for &xi in x.iter().take(n) {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }

    if amax < GROUP_MAX_EPS {
        return 0.0;
    }

    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0;
        let mut suml2 = 0.0;
        for i in 0..n {
            let li = nearest_int_mul(q, iscale, x[i]).clamp(-nmax, nmax - 1);
            l[i] = li as i8;
            let (w, lf) = (x[i] * x[i], li as f32);
            sumlx = madd(q, w * x[i], lf, sumlx);
            suml2 = madd(q, w * lf, lf, suml2);
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let li = l[i] as f32;
                let mut slx = madd(q, -(w * x[i]), li, sumlx);
                if slx > 0.0 {
                    let mut sl2 = madd(q, -(w * li), li, suml2);
                    let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                    if new_l != l[i] as i32 {
                        let nf = new_l as f32;
                        slx = madd(q, w * x[i], nf, slx);
                        sl2 = madd(q, w * nf, nf, sl2);
                        if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                            l[i] = new_l as i8;
                            sumlx = slx;
                            suml2 = sl2;
                            n_changed += 1;
                        }
                    }
                }
            }
            if n_changed == 0 {
                break;
            }
        }
        return sumlx / suml2;
    }
    1.0 / iscale
}

/// ik_llama.cpp's `make_qkx3_quants`: ggml's affine fit with its sums in f64 and the squared error from them in closed
/// form, then single values moved a step along the largest residual while the fit improves. (scale, min)
pub(super) fn make_qkx3_quants_ik(
    nmax: i32,
    x: &[f32],
    weights: Option<&[f32]>,
    search: QkxSearch,
    l: &mut [u8],
) -> (f32, f32) {
    let q = Quantizer::IkLlama;
    let QkxSearch {
        rmin,
        rdelta,
        nstep,
        use_mad,
    } = search;
    let n = x.len();
    let wf = |i: usize| weights.map_or(x[i] * x[i], |w| w[i]);
    let mut laux = [0u8; 32];
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = wf(0) as f64;
    let mut sum_x = sum_w * x[0] as f64;
    let mut sum_x2 = sum_w * x[0] as f64 * x[0] as f64;
    for (i, &xi) in x.iter().enumerate().skip(1) {
        min = min.min(xi);
        max = max.max(xi);
        let w = wf(i) as f64;
        let wx = w * xi as f64;
        sum_w += w;
        sum_x += wx;
        sum_x2 = wx.mul_add(xi as f64, sum_x2);
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max - min < QKX_FLAT_RANGE {
        l.fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0f64;
    for i in 0..n {
        let li = nearest_int_mul(q, iscale, x[i] - min).clamp(0, nmax);
        l[i] = li as u8;
        let diff = (scale as f64).mul_add(li as f64, min as f64) - x[i] as f64;
        let diff = if use_mad { diff.abs() } else { diff * diff };
        best_mad = (wf(i) as f64).mul_add(diff, best_mad);
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (madd(q, rdelta, is as f32, rmin) + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f64, 0f64, 0f64);
        for i in 0..n {
            let li = nearest_int_mul(q, iscale, x[i] - min).clamp(0, nmax);
            laux[i] = li as u8;
            let wl = wf(i) as f64 * li as f64;
            sum_l += wl;
            sum_l2 = wl.mul_add(li as f64, sum_l2);
            sum_xl = wl.mul_add(x[i] as f64, sum_xl);
        }
        let d = msub64(sum_w, sum_l2, sum_l, sum_l);
        if d > 0.0 {
            let mut this_scale = -msub64(sum_x, sum_l, sum_w, sum_xl) / d;
            let mut this_min = msub64(sum_l2, sum_x, sum_l, sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mad = if use_mad {
                let mut mad = 0f64;
                for i in 0..n {
                    let diff = this_scale.mul_add(laux[i] as f64, this_min) - x[i] as f64;
                    mad = (wf(i) as f64).mul_add(diff.abs(), mad);
                }
                mad
            } else {
                let m = (-2.0 * this_scale).mul_add(sum_xl, sum_x2);
                let m = (-2.0 * this_min).mul_add(sum_x, m);
                let m = (2.0 * this_scale * this_min).mul_add(sum_l, m);
                let m = (this_scale * this_scale).mul_add(sum_l2, m);
                (this_min * this_min).mul_add(sum_w, m)
            };
            if mad < best_mad {
                l[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale as f32;
                min = this_min as f32;
            }
        }
    }
    if use_mad {
        return (scale, -min);
    }
    let (mut sum_l, mut sum_l2, mut sum_xl) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let li = l[i] as f64;
        let wl = wf(i) as f64 * li;
        sum_l += wl;
        sum_l2 = wl.mul_add(li, sum_l2);
        sum_xl = wl.mul_add(x[i] as f64, sum_xl);
    }
    // the weighted fit gain from the sums, 2 s sum_xl + 2 m sum_x - 2 s m sum_l - s^2 sum_l2 - m^2 sum_w (higher is a
    // smaller error)
    let gain = |s: f64, m: f64, sum_l: f64, sum_l2: f64, sum_xl: f64| {
        let g = madd2_64(2.0 * s, sum_xl, 2.0 * m, sum_x);
        let g = (-2.0 * s * m).mul_add(sum_l, g);
        let g = (-s * s).mul_add(sum_l2, g);
        (-m * m).mul_add(sum_w, g)
    };
    let mut best = gain(scale as f64, min as f64, sum_l, sum_l2, sum_xl);
    let (mut last_j, mut last_dir) = (None, 0);
    for _ in 0..nmax as usize * n {
        let mut gmax = 0f32;
        let (mut best_j, mut dir) = (None, 0i32);
        for j in 0..n {
            let g = madd(q, -scale, l[j] as f32, x[j]) - min;
            if g > 0.0 && (l[j] as i32) < nmax && g > gmax {
                (gmax, best_j, dir) = (g, Some(j), 1);
            } else if g < 0.0 && l[j] > 0 && -g > gmax {
                (gmax, best_j, dir) = (-g, Some(j), -1);
            }
        }
        let Some(j) = best_j else { break };
        if last_j == Some(j) && dir == -last_dir {
            break;
        }
        let w = wf(j) as f64;
        let dirf = dir as f64;
        sum_l = w.mul_add(dirf, sum_l);
        sum_l2 = w.mul_add((2 * l[j] as i32 * dir + 1) as f64, sum_l2);
        sum_xl = (w * x[j] as f64).mul_add(dirf, sum_xl);
        let d = msub64(sum_w, sum_l2, sum_l, sum_l);
        if d <= 0.0 {
            break;
        }
        let mut this_scale = -msub64(sum_x, sum_l, sum_w, sum_xl) / d;
        let mut this_min = msub64(sum_l2, sum_x, sum_l, sum_xl) / d;
        if this_min > 0.0 {
            this_min = 0.0;
            this_scale = sum_xl / sum_l2;
        }
        if this_scale < 0.0 {
            break;
        }
        let score = gain(this_scale, this_min, sum_l, sum_l2, sum_xl);
        if score <= best {
            break;
        }
        best = score;
        scale = this_scale as f32;
        min = this_min as f32;
        l[j] = (l[j] as i32 + dir) as u8;
        (last_j, last_dir) = (Some(j), dir);
    }
    (scale, -min)
}

// https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/ggml/src/ggml-quants.c#L744
/// (scale, min)
pub(super) fn make_qkx3_quants(
    nmax: i32,
    x: &[f32],
    weights: Option<&[f32]>,
    search: QkxSearch,
    l: &mut [u8],
) -> (f32, f32) {
    let QkxSearch {
        rmin,
        rdelta,
        nstep,
        use_mad,
    } = search;
    let n = x.len();
    let mut l_aux: [u8; 32] = [0; 32];

    let mut min_val = x[0];
    let mut max_val = x[0];
    let mut sum_w = match weights {
        Some(w) => w[0],
        None => x[0] * x[0],
    };
    let mut sum_x = sum_w * x[0];

    for i in 1..n {
        if x[i] < min_val {
            min_val = x[i];
        }
        if x[i] > max_val {
            max_val = x[i];
        }
        let w = match weights {
            Some(w) => w[i],
            None => x[i] * x[i],
        };
        sum_w += w;
        sum_x += w * x[i];
    }

    if min_val > 0.0 {
        min_val = 0.0;
    }

    if max_val <= min_val {
        l.fill(0);
        return (0.0, -min_val);
    }

    let mut iscale = nmax as f32 / (max_val - min_val);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0;

    for i in 0..n {
        let l_val = nearest_int(iscale * (x[i] - min_val)).clamp(0, nmax) as u8;
        l[i] = l_val;
        let diff = scale * (l_val as f32) + min_val - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        let w = match weights {
            Some(w) => w[i],
            None => x[i] * x[i],
        };
        best_mad += w * diff;
    }

    if nstep < 1 {
        return (scale, -min_val);
    }

    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max_val - min_val);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0, 0.0, 0.0);

        for i in 0..n {
            let l_val = nearest_int(iscale * (x[i] - min_val)).clamp(0, nmax) as u8;
            l_aux[i] = l_val;
            let w = match weights {
                Some(w) => w[i],
                None => x[i] * x[i],
            };
            sum_l += w * l_val as f32;
            sum_l2 += w * l_val as f32 * l_val as f32;
            sum_xl += w * l_val as f32 * x[i];
        }

        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;

            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }

            let mut mad = 0.0;
            for i in 0..n {
                let diff = this_scale * (l_aux[i] as f32) + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                let w = match weights {
                    Some(w) => w[i],
                    None => x[i] * x[i],
                };
                mad += w * diff;
            }

            if mad < best_mad {
                l.copy_from_slice(&l_aux[..n]);
                best_mad = mad;
                scale = this_scale;
                min_val = this_min;
            }
        }
    }

    (scale, -min_val)
}

// https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/ggml/src/ggml-quants.c#L827
// A negative `x` (IQ2_XXS passes one) gets ggml's mix of types: a level is an int where it is computed and summed,
// and wraps to its uint8_t when stored in and read back from `l`.
pub fn make_qp_quants(
    n: usize,
    nmax: u8,
    x: &[f32],
    l: &mut [u8],
    quant_weights: &[f32],
    quantizer: Quantizer,
) -> f32 {
    assert_eq!(x.len(), n);
    assert_eq!(l.len(), n);
    assert_eq!(quant_weights.len(), n);
    let q = quantizer;
    let nmax_i = i32::from(nmax);

    let max = x.iter().copied().fold(0.0, f32::max);
    let zero = match q {
        Quantizer::LlamaCpp => GROUP_MAX_EPS,
        Quantizer::IkLlama => QP_ZERO_IK,
    };
    if max < zero {
        l.iter_mut().for_each(|li| *li = 0);
        return 0.0;
    }

    let mut iscale = nmax as f32 / max;
    for (xi, li) in x.iter().zip(l.iter_mut()) {
        *li = nearest_int_mul(q, iscale, *xi) as u8;
    }

    // the weighted squared error of `x` against `scale` times its levels
    let mse = |scale: f32, level: &dyn Fn(usize) -> i32| {
        let mut mse = 0f32;
        for (i, (&xi, &w)) in x.iter().zip(quant_weights).enumerate() {
            let diff = madd(q, -scale, level(i) as f32, xi);
            mse = madd(q, w * diff, diff, mse);
        }
        mse
    };
    let mut best_mse = mse(1.0 / iscale, &|i| l[i] as i32);
    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = madd(q, 0.1, is as f32, nmax as f32) / max;
        let candidate = mse(1.0 / iscale_is, &|i| {
            nearest_int_mul(q, iscale_is, x[i]).min(nmax_i)
        });
        if candidate < best_mse {
            best_mse = candidate;
            iscale = iscale_is;
        }
    }

    let mut sumlx = 0.0;
    let mut suml2 = 0.0;
    for ((xi, li), &w) in x.iter().zip(l.iter_mut()).zip(quant_weights.iter()) {
        let li_new = nearest_int_mul(q, iscale, *xi).min(nmax_i);
        *li = li_new as u8;
        sumlx = madd(q, w * xi, li_new as f32, sumlx);
        suml2 = madd(q, w * li_new as f32, li_new as f32, suml2);
    }

    for _ in 0..5 {
        let mut n_changed = 0;
        for ((xi, li), &w) in x.iter().zip(l.iter_mut()).zip(quant_weights.iter()) {
            let lf = *li as f32;
            let mut slx = madd(q, -(w * xi), lf, sumlx);
            let mut sl2 = madd(q, -(w * lf), lf, suml2);
            if slx > 0.0 && sl2 > 0.0 {
                let new_li = nearest_int(xi * sl2 / slx).min(nmax_i);
                if new_li != i32::from(*li) {
                    slx = madd(q, w * xi, new_li as f32, slx);
                    sl2 = madd(q, w * new_li as f32, new_li as f32, sl2);
                    if slx.powi(2) * suml2 > sumlx.powi(2) * sl2 {
                        *li = new_li as u8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }

    match q {
        Quantizer::IkLlama => sumlx / suml2,
        Quantizer::LlamaCpp if suml2 > 0.0 => sumlx / suml2,
        Quantizer::LlamaCpp => 0.0,
    }
}
