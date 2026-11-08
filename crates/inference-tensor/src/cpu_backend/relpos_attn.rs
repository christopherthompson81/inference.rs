//! Transformer-XL relative-position attention fused on AVX-512 BF16 (`vdpbf16ps`), a query head a key head:
//! score(i, j) = qc_i . k_j + qp_i . p[pos_base - i + j], both query sets scaled beforehand, a softmax over the keys,
//! times v, with no (rows, keys) scores leaving cache. A task takes `R` query rows of a head and walks the keys `C` at
//! a time: content scores against the chunk, position scores against the `R + C - 1` position rows its (row, key)
//! pairs read (one contiguous span), each pair's diagonal entry added, an online softmax, and P V into the rows'
//! outputs. Keys, values and positions are packed as bf16 pairs once a call; `R` and `C` multiples of the panel width
//! keep every span on the same alignment of the positions' panels, so they are packed once too.

/// A head's operands as one contiguous f32 `(heads, rows, d)` block each: content and position queries `(heads, t,
/// d)`, keys and values `(heads, keys, d)`, positions `(heads, p_rows, d)`; pair `(i, j)` reads position row
/// `pos_base - i + j`.
// read by the x86_64 kernel alone
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub struct RelPosAttn<'a> {
    pub qc: &'a [f32],
    pub qp: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub p: &'a [f32],
    pub heads: usize,
    pub t: usize,
    pub keys: usize,
    pub d: usize,
    pub p_rows: usize,
    pub pos_base: usize,
}

/// Whether the kernel takes a head size: whole panels of values, on a CPU that multiplies bf16 pairs.
pub fn supports(d: usize) -> bool {
    d > 0 && d.is_multiple_of(32) && super::gemm_bf16::available()
}

#[cfg(target_arch = "x86_64")]
pub use x86::attention;

/// [`supports`] is false off x86_64, so nothing calls this there.
#[cfg(not(target_arch = "x86_64"))]
pub fn attention(_op: &RelPosAttn) -> Vec<f32> {
    unreachable!("relative-position attention on bf16 runs on x86_64's AVX-512 BF16")
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::RelPosAttn;
    use crate::cpu::erf::fast_exp;
    use crate::cpu_backend::gemm_bf16::row_pairs;
    use rayon::prelude::*;
    use std::arch::x86_64::*;

    // query rows a task takes, and keys a step: multiples of NR (every span then starts on a panel)
    const R: usize = 32;
    const C: usize = 512;
    // a microkernel's rows, and a panel's columns (two 16-lane accumulators)
    const MR: usize = 8;
    const NR: usize = 32;
    const LANES: usize = 16;
    // a span's columns: R + C - 1 position rows, in whole panels
    const SPAN: usize = R + C;
    // position panels before row 0: a last tile's rows past the queries reach up to R - 1 rows early
    const FRONT: usize = 2 * NR;
    // bf16 rounding to nearest even, as `vcvtneps2bf16`
    const BF16_ROUND: u32 = 0x7fff;

    fn bf16_bits(x: f32) -> u32 {
        let bits = x.to_bits();
        (bits + BF16_ROUND + ((bits >> 16) & 1)) >> 16
    }

    fn pair(lo: f32, hi: f32) -> u32 {
        bf16_bits(lo) | bf16_bits(hi) << 16
    }

    // `rows` rows of `d` from `src` (rows past `live` zero) as (rows, d / 2) pairs along d
    fn pack_rows(src: &[f32], live: usize, rows: usize, d: usize) -> Vec<u32> {
        let mut out = vec![0u32; rows * d / 2];
        out.par_chunks_mut(d / 2)
            .take(live)
            .enumerate()
            .for_each(|(r, row)| row_pairs(&src[r * d..][..d], row));
        out
    }

    // NR-column panels over `cols` rows of `src` (row `first + c` for column c, zero outside 0..live), pairs along d:
    // (panels, d / 2, NR)
    fn pack_panels(src: &[f32], live: usize, first: isize, cols: usize, d: usize) -> Vec<u32> {
        let pairs = d / 2;
        let mut out = vec![0u32; cols * pairs];
        out.par_chunks_mut(pairs * NR)
            .enumerate()
            .for_each(|(panel, out)| {
                let mut column = vec![0u32; pairs];
                for j in 0..NR {
                    let row = first + (panel * NR + j) as isize;
                    if row < 0 || row as usize >= live {
                        continue;
                    }
                    row_pairs(&src[row as usize * d..][..d], &mut column);
                    for (p, pair) in column.iter().enumerate() {
                        out[p * NR + j] = *pair;
                    }
                }
            });
        out
    }

    // values as (d / NR, keys / 2, NR): each d-panel's columns, pairs along the keys (zero past `live`)
    fn pack_values(src: &[f32], live: usize, keys: usize, d: usize) -> Vec<u32> {
        let mut out = vec![0u32; d * keys / 2];
        out.par_chunks_mut(keys / 2 * NR)
            .enumerate()
            .for_each(|(panel, out)| {
                for kp in 0..keys / 2 {
                    for j in 0..NR {
                        let at = |key: usize| {
                            if key < live {
                                src[key * d + panel * NR + j]
                            } else {
                                0.0
                            }
                        };
                        out[kp * NR + j] = pair(at(2 * kp), at(2 * kp + 1));
                    }
                }
            });
        out
    }

    // an MR x NR tile: `a`'s MR rows (`a_stride` pairs apart) over `pairs` pairs of panel `b`, into `dst` (`dst_rs`
    // floats a row), added to what it holds where `add`
    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn tile(
        a: *const u32,
        a_stride: usize,
        pairs: usize,
        b: *const u32,
        dst: *mut f32,
        dst_rs: usize,
        add: bool,
    ) {
        unsafe {
            let mut acc = [[_mm512_setzero_ps(); 2]; MR];
            for p in 0..pairs {
                let b0: __m512bh = std::mem::transmute(_mm512_loadu_si512(b.add(p * NR).cast()));
                let b1: __m512bh =
                    std::mem::transmute(_mm512_loadu_si512(b.add(p * NR + LANES).cast()));
                for (r, acc) in acc.iter_mut().enumerate() {
                    let av: __m512bh =
                        std::mem::transmute(_mm512_set1_epi32(*a.add(r * a_stride + p) as i32));
                    acc[0] = _mm512_dpbf16_ps(acc[0], av, b0);
                    acc[1] = _mm512_dpbf16_ps(acc[1], av, b1);
                }
            }
            for (r, acc) in acc.iter().enumerate() {
                let row = dst.add(r * dst_rs);
                let (lo, hi) = if add {
                    (
                        _mm512_add_ps(acc[0], _mm512_loadu_ps(row)),
                        _mm512_add_ps(acc[1], _mm512_loadu_ps(row.add(LANES))),
                    )
                } else {
                    (acc[0], acc[1])
                };
                _mm512_storeu_ps(row, lo);
                _mm512_storeu_ps(row.add(LANES), hi);
            }
        }
    }

    // Cephes' exp on 16 lanes, as `fast_exp` computes it a lane
    #[target_feature(enable = "avx512f")]
    unsafe fn exp16(x: __m512) -> __m512 {
        const LOG2E: f32 = std::f32::consts::LOG2_E;
        const C0: f32 = 0.693_359_4;
        const C1: f32 = -2.121_944_4e-4;
        const POLY: [f32; 5] = [
            0.001_392_034_5,
            0.008_333_45,
            0.041_665_795,
            0.166_665_46,
            0.5,
        ];
        // round to nearest, exceptions suppressed
        const NEAREST: i32 = 0x08;
        let x = _mm512_max_ps(
            _mm512_min_ps(x, _mm512_set1_ps(87.0)),
            _mm512_set1_ps(-87.0),
        );
        let z = _mm512_roundscale_ps::<NEAREST>(_mm512_mul_ps(x, _mm512_set1_ps(LOG2E)));
        let r = _mm512_fnmadd_ps(
            z,
            _mm512_set1_ps(C1),
            _mm512_fnmadd_ps(z, _mm512_set1_ps(C0), x),
        );
        let mut q = _mm512_set1_ps(POLY[0]);
        for c in &POLY[1..] {
            q = _mm512_fmadd_ps(q, r, _mm512_set1_ps(*c));
        }
        let p = _mm512_fmadd_ps(_mm512_mul_ps(r, r), q, r);
        _mm512_scalef_ps(_mm512_add_ps(p, _mm512_set1_ps(1.0)), z)
    }

    // one row's softmax step over a chunk: `x` (C scores) gains `pos` over its `valid` columns, the running max `m`
    // moves to cover them, and the probabilities exp(x - m) (zero past `valid`) go out as bf16 terms `hi` and what
    // they left `lo`, a row of C halves each; returns (the factor the row's earlier output scales by, the row's sum)
    #[target_feature(enable = "avx512f,avx512bf16")]
    unsafe fn row(
        x: *mut f32,
        pos: *const f32,
        valid: usize,
        m: &mut f32,
        hi: *mut u16,
        lo: *mut u16,
    ) -> (f32, f32) {
        unsafe {
            let live = |i: usize| -> __mmask16 {
                let n = valid.saturating_sub(i).min(LANES);
                if n == LANES {
                    u16::MAX
                } else {
                    (1u16 << n) - 1
                }
            };
            let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
            for i in (0..C).step_by(LANES) {
                let k = live(i);
                let v = _mm512_add_ps(_mm512_loadu_ps(x.add(i)), _mm512_loadu_ps(pos.add(i)));
                _mm512_storeu_ps(x.add(i), v);
                vmax = _mm512_mask_max_ps(vmax, k, vmax, v);
            }
            let new_m = m.max(_mm512_reduce_max_ps(vmax));
            let corr = if *m == f32::NEG_INFINITY {
                0.0
            } else {
                fast_exp(*m - new_m)
            };
            *m = new_m;
            let mv = _mm512_set1_ps(new_m);
            let mut vsum = _mm512_setzero_ps();
            for i in (0..C).step_by(LANES) {
                let e = _mm512_maskz_mov_ps(
                    live(i),
                    exp16(_mm512_sub_ps(_mm512_loadu_ps(x.add(i)), mv)),
                );
                vsum = _mm512_add_ps(vsum, e);
                let h = _mm512_cvtneps_pbh(e);
                let back = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(
                    std::mem::transmute::<__m256bh, __m256i>(h),
                )));
                let rest = _mm512_cvtneps_pbh(_mm512_sub_ps(e, back));
                _mm256_storeu_si256(
                    hi.add(i).cast(),
                    std::mem::transmute::<__m256bh, __m256i>(h),
                );
                _mm256_storeu_si256(
                    lo.add(i).cast(),
                    std::mem::transmute::<__m256bh, __m256i>(rest),
                );
            }
            (corr, _mm512_reduce_add_ps(vsum))
        }
    }

    /// `(heads, t, d)` outputs; call only where [`super::supports`] takes `d`.
    pub fn attention(op: &RelPosAttn) -> Vec<f32> {
        let RelPosAttn {
            heads,
            t,
            keys,
            d,
            p_rows,
            pos_base,
            ..
        } = *op;
        let pairs = d / 2;
        let rows = t.next_multiple_of(R);
        let keys_pad = keys.next_multiple_of(C);
        // the span of (row i0 + R - 1, key j0) starts at pos_base - i0 - R + 1 + j0: one residue of NR for every tile
        let shift = (pos_base + 1) % NR;
        let base = shift as isize - FRONT as isize;
        let p_cols = (pos_base + keys_pad + SPAN + FRONT).next_multiple_of(NR);
        fn per_head(x: &[f32], h: usize, n: usize, d: usize) -> &[f32] {
            &x[h * n * d..][..n * d]
        }
        let qc: Vec<Vec<u32>> = (0..heads)
            .map(|h| pack_rows(per_head(op.qc, h, t, d), t, rows, d))
            .collect();
        let qp: Vec<Vec<u32>> = (0..heads)
            .map(|h| pack_rows(per_head(op.qp, h, t, d), t, rows, d))
            .collect();
        let kp: Vec<Vec<u32>> = (0..heads)
            .map(|h| pack_panels(per_head(op.k, h, keys, d), keys, 0, keys_pad, d))
            .collect();
        let vp: Vec<Vec<u32>> = (0..heads)
            .map(|h| pack_values(per_head(op.v, h, keys, d), keys, keys_pad, d))
            .collect();
        let pp: Vec<Vec<u32>> = (0..heads)
            .map(|h| pack_panels(per_head(op.p, h, p_rows, d), p_rows, base, p_cols, d))
            .collect();
        let mut out = vec![0f32; heads * t * d];
        let out_at = out.as_mut_ptr() as usize;
        let tasks: Vec<(usize, usize)> = (0..heads)
            .flat_map(|h| (0..rows).step_by(R).map(move |i0| (h, i0)))
            .collect();
        tasks.into_par_iter().for_each(|(h, i0)| {
            let (qc, qp, kp, vp, pp) = (&qc[h], &qp[h], &kp[h], &vp[h], &pp[h]);
            let mut s = vec![0f32; R * C];
            let mut sp = vec![0f32; R * SPAN];
            // the probabilities as two bf16 terms, p's rounding and what it left: ~16 bits, where one alone moved a
            // transcript's punctuation
            let mut pa = vec![0u32; R * C / 2];
            let mut pa_lo = vec![0u32; R * C / 2];
            let mut o = vec![0f32; R * d];
            let mut m = [f32::NEG_INFINITY; R];
            let mut l = [0f32; R];
            for j0 in (0..keys).step_by(C) {
                let w0 = (pos_base + j0) as isize - (i0 + R - 1) as isize;
                let q0 = ((w0 - base) as usize) / NR;
                // SAFETY: every operand is a packed buffer holding the panels and rows read, and `supports` holds
                unsafe {
                    for g in (0..R).step_by(MR) {
                        let a = qc.as_ptr().add((i0 + g) * pairs);
                        for col_panel in 0..C / NR {
                            let b = kp.as_ptr().add((j0 / NR + col_panel) * pairs * NR);
                            tile(
                                a,
                                pairs,
                                pairs,
                                b,
                                s.as_mut_ptr().add(g * C + col_panel * NR),
                                C,
                                false,
                            );
                        }
                        let a = qp.as_ptr().add((i0 + g) * pairs);
                        for col_panel in 0..SPAN / NR {
                            let b = pp.as_ptr().add((q0 + col_panel) * pairs * NR);
                            tile(
                                a,
                                pairs,
                                pairs,
                                b,
                                sp.as_mut_ptr().add(g * SPAN + col_panel * NR),
                                SPAN,
                                false,
                            );
                        }
                    }
                }
                let valid = C.min(keys - j0);
                for r in 0..R {
                    // SAFETY: row r of the scores, its span's columns from R - 1 - r, and its probabilities' C halves
                    let (corr, sum) = unsafe {
                        row(
                            s.as_mut_ptr().add(r * C),
                            sp.as_ptr().add(r * SPAN + (R - 1 - r)),
                            valid,
                            &mut m[r],
                            pa.as_mut_ptr().cast::<u16>().add(r * C),
                            pa_lo.as_mut_ptr().cast::<u16>().add(r * C),
                        )
                    };
                    for v in &mut o[r * d..][..d] {
                        *v *= corr;
                    }
                    l[r] = l[r] * corr + sum;
                }
                // SAFETY: as above; the values' panels hold keys_pad / 2 pairs, so j0 + C is in range
                unsafe {
                    for g in (0..R).step_by(MR) {
                        for probs in [&pa, &pa_lo] {
                            let a = probs.as_ptr().add(g * C / 2);
                            for dp in 0..d / NR {
                                let b = vp.as_ptr().add((dp * keys_pad / 2 + j0 / 2) * NR);
                                tile(
                                    a,
                                    C / 2,
                                    C / 2,
                                    b,
                                    o.as_mut_ptr().add(g * d + dp * NR),
                                    d,
                                    true,
                                );
                            }
                        }
                    }
                }
            }
            for r in 0..R.min(t.saturating_sub(i0)) {
                let inv = if l[r] > 0.0 { 1.0 / l[r] } else { 0.0 };
                // SAFETY: row (h, i0 + r) of `out`, written by this task alone
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        (out_at as *mut f32).add((h * t + i0 + r) * d),
                        d,
                    )
                };
                for (dst, v) in dst.iter_mut().zip(&o[r * d..][..d]) {
                    *dst = v * inv;
                }
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Tensor};

    // against its definition over the same bf16-rounded inputs, in f64: content and position scores (row i reading
    // position row pos_base - i + j), the softmax and its values; a partial last query tile and key chunk, two
    // heads, head sizes 128 and 32, a run past one key chunk
    #[test]
    fn fused_relative_position_attention_matches_its_definition() -> crate::Result<()> {
        if !super::supports(32) {
            return Ok(());
        }
        // the probabilities' bf16 rounding (2^-8) over the values' sums (worst seen 1.4e-3)
        const TOLERANCE: f64 = 4e-3;
        let round = |t: &Tensor| t.to_dtype(DType::BF16)?.to_dtype(DType::F64);
        for (t, head) in [(150, 128), (70, 32), (600, 64)] {
            let (heads, p_rows, pos_base) = (2, 2 * t - 1, t - 1);
            let randn =
                |rows: usize| Tensor::randn(0f32, 0.5, (1, heads, rows, head), &Device::Cpu);
            let (qc, qp, k, v, p) = (randn(t)?, randn(t)?, randn(t)?, randn(t)?, randn(p_rows)?);
            let got = crate::cpu::with_bf16_matmuls(|| {
                crate::cpu::rel_pos_attention(&qc, &qp, &k, &v, &p, pos_base)
            })?
            .expect("the CPU takes it");
            let ids: Vec<u32> = (0..t)
                .flat_map(|i| (0..t).map(move |j| (pos_base - i + j) as u32))
                .collect();
            let ids =
                Tensor::from_vec(ids, (1, 1, t, t), &Device::Cpu)?.repeat((1, heads, 1, 1))?;
            let content = round(&qc)?.matmul(&round(&k)?.t()?)?;
            let position = round(&qp)?
                .matmul(&round(&p)?.t()?)?
                .contiguous()?
                .gather(&ids, 3)?;
            let probs = crate::nn::ops::softmax_last_dim(&(content + position)?)?;
            let want = probs.matmul(&round(&v)?)?.flatten_all()?.to_vec1::<f64>()?;
            let got = got.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()?;
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0f64, f64::max);
            assert!(worst < TOLERANCE, "t {t} head {head}: {worst}");
        }
        // outside the scope the CPU keeps the composite
        let x = Tensor::zeros((1, 1, 4, 32), DType::F32, &Device::Cpu)?;
        let p = Tensor::zeros((1, 1, 7, 32), DType::F32, &Device::Cpu)?;
        assert!(crate::cpu::rel_pos_attention(&x, &x, &x, &x, &p, 3)?.is_none());
        Ok(())
    }
}
