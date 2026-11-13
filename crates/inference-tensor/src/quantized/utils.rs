// ggml's: the float added to 1.5 * 2^23 rounds to the nearest integer, ties to even
const ROUNDING_MAGIC: f32 = 12_582_912.0;
// ggml's `GROUP_MAX_EPS`: a group whose largest magnitude is below it quantizes to zeros
const GROUP_MAX_EPS: f32 = 1e-15;
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
/// negative one averages the first fit with it.
pub(super) unsafe fn make_qx_quants(
    n: usize,
    nmax: i32,
    x: *const f32,
    ls: *mut i8,
    rmse_type: i32,
    qw: *const f32,
) -> f32 {
    unsafe {
        let x = std::slice::from_raw_parts(x, n);
        let ls = std::slice::from_raw_parts_mut(ls, n);
        let qw = (!qw.is_null()).then(|| std::slice::from_raw_parts(qw, n));
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
        let mut iscale = -(nmax as f32) / max;
        if rmse_type == 0 {
            for (l, &v) in ls.iter_mut().zip(x) {
                *l = (nmax + nearest_int(iscale * v).clamp(-nmax, nmax - 1)) as i8;
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
        let mut sumlx = 0f32;
        let mut suml2 = 0f32;
        for i in 0..n {
            let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            ls[i] = (l + nmax) as i8;
            let w = weight(i);
            sumlx += w * x[i] * l as f32;
            suml2 += w * l as f32 * l as f32;
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
        for is in -QX_SCALE_TRIES..=QX_SCALE_TRIES {
            if is == 0 {
                continue;
            }
            iscale = -(nmax as f32 + 0.1 * is as f32) / max;
            sumlx = 0.0;
            suml2 = 0.0;
            for (i, &xi) in x.iter().enumerate() {
                let l = nearest_int(iscale * xi).clamp(-nmax, nmax - 1);
                let w = weight(i);
                sumlx += w * xi * l as f32;
                suml2 += w * l as f32 * l as f32;
            }
            if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
                for (l, &v) in ls.iter_mut().zip(x) {
                    *l = (nmax + nearest_int(iscale * v).clamp(-nmax, nmax - 1)) as i8;
                }
                scale = sumlx / suml2;
                best = scale * sumlx;
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
) -> (f32, f32) {
    let QkxSearch {
        rmin,
        rdelta,
        nstep,
        use_mad,
    } = search;
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.fill(0);
        return (0.0, -min);
    }
    let err = |diff: f32| if use_mad { diff.abs() } else { diff * diff };
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = li as u8;
        best_error += weights[i] * err(scale * li as f32 + min - x[i]);
    }
    if nstep == 0 {
        return (scale, -min);
    }
    let mut laux = vec![0i32; n];
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = li;
            let (w, l) = (weights[i], li as f32);
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let cur_error = (0..n)
                .map(|i| weights[i] * err(this_scale * laux[i] as f32 + this_min - x[i]))
                .fold(0f32, |a, v| a + v);
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
pub(super) fn make_q3_quants(x: &[f32], nmax: i32, do_rmse: bool) -> f32 {
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

    if amax == 0.0 {
        return 0.0;
    }

    let iscale = -(nmax as f32) / max;
    if do_rmse {
        let mut sumlx = 0.0;
        let mut suml2 = 0.0;
        for i in 0..n {
            let li = (iscale * x[i]).round() as i32;
            let li = li.clamp(-nmax, nmax - 1);
            l[i] = li as i8;
            let w = x[i] * x[i];
            sumlx += w * x[i] * li as f32;
            suml2 += w * (li * li) as f32;
        }
        for _ in 0..5 {
            let mut n_changed = 0;
            for i in 0..n {
                let w = x[i] * x[i];
                let mut slx = sumlx - w * x[i] * l[i] as f32;
                if slx > 0.0 {
                    let mut sl2 = suml2 - w * (l[i] as i32 * l[i] as i32) as f32;
                    let mut new_l = (x[i] * sl2 / slx).round() as i32;
                    new_l = new_l.clamp(-nmax, nmax - 1);
                    if new_l != l[i] as i32 {
                        slx += w * x[i] * new_l as f32;
                        sl2 += w * (new_l * new_l) as f32;
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
        for li in l.iter_mut() {
            *li += nmax as i8;
        }
        return sumlx / suml2;
    }
    for i in 0..n {
        let li = (iscale * x[i]).round() as i32;
        l[i] = (li.clamp(-nmax, nmax - 1) + nmax) as i8;
    }
    1.0 / iscale
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
pub fn make_qp_quants(n: usize, nmax: u8, x: &[f32], l: &mut [u8], quant_weights: &[f32]) -> f32 {
    assert_eq!(x.len(), n);
    assert_eq!(l.len(), n);
    assert_eq!(quant_weights.len(), n);
    let nmax_i = i32::from(nmax);

    let max = x.iter().copied().fold(0.0, f32::max);
    if max < GROUP_MAX_EPS {
        l.iter_mut().for_each(|li| *li = 0);
        return 0.0;
    }

    let mut iscale = nmax as f32 / max;
    for (xi, li) in x.iter().zip(l.iter_mut()) {
        *li = nearest_int(iscale * xi) as u8;
    }

    let scale = 1.0 / iscale;
    let mut best_mse = x
        .iter()
        .zip(l.iter())
        .zip(quant_weights.iter())
        .map(|((&xi, &li), &w)| {
            let diff = xi - scale * li as f32;
            w * diff * diff
        })
        .sum::<f32>();

    for is in -4..=4 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;

        let mse = x
            .iter()
            .zip(quant_weights.iter())
            .map(|(&xi, &w)| {
                let li = nearest_int(iscale_is * xi).min(nmax_i);
                let diff = xi - scale_is * li as f32;
                w * diff * diff
            })
            .sum::<f32>();

        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    let mut sumlx = 0.0;
    let mut suml2 = 0.0;
    for ((xi, li), &w) in x.iter().zip(l.iter_mut()).zip(quant_weights.iter()) {
        let li_new = nearest_int(iscale * xi).min(nmax_i);
        *li = li_new as u8;
        sumlx += w * xi * li_new as f32;
        suml2 += w * li_new as f32 * li_new as f32;
    }

    for _ in 0..5 {
        let mut n_changed = 0;
        for ((xi, li), &w) in x.iter().zip(l.iter_mut()).zip(quant_weights.iter()) {
            let mut slx = sumlx - w * xi * *li as f32;
            let mut sl2 = suml2 - w * *li as f32 * *li as f32;
            if slx > 0.0 && sl2 > 0.0 {
                let new_li = nearest_int(xi * sl2 / slx).min(nmax_i);
                if new_li != i32::from(*li) {
                    slx += w * xi * new_li as f32;
                    sl2 += w * new_li as f32 * new_li as f32;
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

    if suml2 > 0.0 { sumlx / suml2 } else { 0.0 }
}
