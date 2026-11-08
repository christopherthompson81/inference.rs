//! [`relpos_attn`](super::relpos_attn)'s fused relative-position attention in f32 on AVX2 and FMA, for a CPU without
//! bf16 pairs: the same walk (a task takes `R` query rows of a head, the keys `C` at a time, content and position
//! scores into cache, each pair's diagonal entry added, an online softmax, P V into the rows' outputs), its products
//! a 6 x 16 FMA tile over f32 panels. Every product keeps f32, so its outputs differ from the composite's (scores
//! written out, a softmax pass, the values' matmul) only by the order of the sums and the exp's ~1e-7.

use super::relpos_attn::RelPosAttn;

/// Whether the kernel takes a head size: whole panels of values, on a CPU with AVX2 and FMA.
pub fn supports(d: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        d > 0
            && d.is_multiple_of(16)
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = d;
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::attention;

/// [`supports`] is false off x86_64, so nothing calls this there.
#[cfg(not(target_arch = "x86_64"))]
pub fn attention(_op: &RelPosAttn) -> Vec<f32> {
    unreachable!("the f32 relative-position attention runs on x86_64's AVX2")
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::RelPosAttn;
    use crate::cpu::erf::fast_exp;
    use rayon::prelude::*;
    use std::arch::x86_64::*;

    // query rows a task takes (a multiple of MR and NR, so every span starts on a panel), and keys a step
    const R: usize = 48;
    const C: usize = 512;
    // a tile's rows, and a panel's columns (two 8-lane accumulators): 12 accumulators, two panel loads, a broadcast
    const MR: usize = 6;
    const NR: usize = 16;
    const LANES: usize = 8;
    // a span's columns: R + C - 1 position rows, in whole panels
    const SPAN: usize = R + C;
    // position panels before row 0: a last tile's rows past the queries reach up to R - 1 rows early
    const FRONT: usize = 4 * NR;
    // keys a P V tile takes, its 16 KB panel reused by every row group (an i7-10700K's layer: 128 keys 739 ms, 256
    // 634-664, 512 642)
    const PV_DEPTH: usize = 256;

    // a head's `d` columns of an operand whose rows are `width` apart
    #[derive(Clone, Copy)]
    struct Head<'a> {
        src: &'a [f32],
        width: usize,
        at: usize,
        d: usize,
    }

    impl Head<'_> {
        fn row(&self, r: usize) -> &[f32] {
            &self.src[r * self.width + self.at..][..self.d]
        }
    }

    // `rows` query rows, each (x + bias) scale, rows past `live` zero
    fn pad_rows(head: Head, bias: &[f32], scale: f32, live: usize, rows: usize) -> Vec<f32> {
        let d = head.d;
        let bias = &bias[head.at..][..d];
        let mut out = vec![0f32; rows * d];
        out.par_chunks_mut(d)
            .take(live)
            .enumerate()
            .for_each(|(r, row)| {
                for ((y, x), b) in row.iter_mut().zip(head.row(r)).zip(bias) {
                    *y = (x + b) * scale;
                }
            });
        out
    }

    // NR-column panels over `cols` rows of `head` (row `first + c` for column c, zero outside 0..live): (panels, d, NR)
    fn pack_panels(head: Head, live: usize, first: isize, cols: usize) -> Vec<f32> {
        let d = head.d;
        let mut out = vec![0f32; cols * d];
        out.par_chunks_mut(d * NR)
            .enumerate()
            .for_each(|(panel, out)| {
                for j in 0..NR {
                    let row = first + (panel * NR + j) as isize;
                    if row < 0 || row as usize >= live {
                        continue;
                    }
                    for (dd, value) in head.row(row as usize).iter().enumerate() {
                        out[dd * NR + j] = *value;
                    }
                }
            });
        out
    }

    // values as (d / NR, keys, NR): each d-panel's columns, key by key (zero past `live`)
    fn pack_values(head: Head, live: usize, keys: usize) -> Vec<f32> {
        let d = head.d;
        let mut out = vec![0f32; d * keys];
        out.par_chunks_mut(keys * NR)
            .enumerate()
            .for_each(|(panel, out)| {
                for key in 0..live {
                    out[key * NR..][..NR].copy_from_slice(&head.row(key)[panel * NR..][..NR]);
                }
            });
        out
    }

    // an MR x NR tile: `a`'s MR rows (`a_stride` floats apart) over `depth` columns of panel `b` (`depth` rows of NR),
    // into `dst` (`dst_rs` floats a row), added to what it holds where `add`
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tile(
        a: *const f32,
        a_stride: usize,
        depth: usize,
        b: *const f32,
        dst: *mut f32,
        dst_rs: usize,
        add: bool,
    ) {
        unsafe {
            let mut acc = [[_mm256_setzero_ps(); 2]; MR];
            for p in 0..depth {
                let b0 = _mm256_loadu_ps(b.add(p * NR));
                let b1 = _mm256_loadu_ps(b.add(p * NR + LANES));
                for (r, acc) in acc.iter_mut().enumerate() {
                    let av = _mm256_broadcast_ss(&*a.add(r * a_stride + p));
                    acc[0] = _mm256_fmadd_ps(av, b0, acc[0]);
                    acc[1] = _mm256_fmadd_ps(av, b1, acc[1]);
                }
            }
            for (r, acc) in acc.iter().enumerate() {
                let row = dst.add(r * dst_rs);
                let (lo, hi) = if add {
                    (
                        _mm256_add_ps(acc[0], _mm256_loadu_ps(row)),
                        _mm256_add_ps(acc[1], _mm256_loadu_ps(row.add(LANES))),
                    )
                } else {
                    (acc[0], acc[1])
                };
                _mm256_storeu_ps(row, lo);
                _mm256_storeu_ps(row.add(LANES), hi);
            }
        }
    }

    // Cephes' exp on 8 lanes, as `fast_exp` computes it a lane
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp8(x: __m256) -> __m256 {
        const LOG2E: f32 = std::f32::consts::LOG2_E;
        const C0: f32 = 0.693_359_4;
        const C1: f32 = -2.121_944_4e-4;
        const ROUND_MAGIC: f32 = 12_582_912.0;
        const POLY: [f32; 5] = [
            0.001_392_034_5,
            0.008_333_45,
            0.041_665_795,
            0.166_665_46,
            0.5,
        ];
        let x = _mm256_max_ps(
            _mm256_min_ps(x, _mm256_set1_ps(87.0)),
            _mm256_set1_ps(-87.0),
        );
        let magic = _mm256_set1_ps(ROUND_MAGIC);
        let z = _mm256_sub_ps(
            _mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(LOG2E)), magic),
            magic,
        );
        let r = _mm256_sub_ps(
            _mm256_sub_ps(x, _mm256_mul_ps(z, _mm256_set1_ps(C0))),
            _mm256_mul_ps(z, _mm256_set1_ps(C1)),
        );
        let mut q = _mm256_set1_ps(POLY[0]);
        for c in &POLY[1..] {
            q = _mm256_add_ps(_mm256_mul_ps(q, r), _mm256_set1_ps(*c));
        }
        let p = _mm256_add_ps(r, _mm256_mul_ps(_mm256_mul_ps(r, r), q));
        let e = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_cvtps_epi32(z),
            _mm256_set1_epi32(127),
        )));
        _mm256_mul_ps(e, _mm256_add_ps(_mm256_set1_ps(1.0), p))
    }

    // one row's softmax step over a chunk: `x` (C scores) gains `pos` over its `valid` columns, the running max `m`
    // moves to cover them, and the probabilities exp(x - m) (zero past `valid`) go to `probs`; returns (the factor
    // the row's earlier output scales by, the row's sum)
    #[target_feature(enable = "avx2,fma")]
    unsafe fn row(
        x: *const f32,
        pos: *const f32,
        valid: usize,
        m: &mut f32,
        probs: *mut f32,
    ) -> (f32, f32) {
        unsafe {
            let whole = valid / LANES * LANES;
            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
            for i in (0..whole).step_by(LANES) {
                let v = _mm256_add_ps(_mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(pos.add(i)));
                _mm256_storeu_ps(probs.add(i), v);
                vmax = _mm256_max_ps(vmax, v);
            }
            let mut lanes = [0f32; LANES];
            _mm256_storeu_ps(lanes.as_mut_ptr(), vmax);
            let mut chunk_max = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for i in whole..valid {
                let v = *x.add(i) + *pos.add(i);
                *probs.add(i) = v;
                chunk_max = chunk_max.max(v);
            }
            let new_m = m.max(chunk_max);
            let corr = if *m == f32::NEG_INFINITY {
                0.0
            } else {
                fast_exp(*m - new_m)
            };
            *m = new_m;
            let mv = _mm256_set1_ps(new_m);
            let mut vsum = _mm256_setzero_ps();
            for i in (0..whole).step_by(LANES) {
                let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(probs.add(i)), mv));
                _mm256_storeu_ps(probs.add(i), e);
                vsum = _mm256_add_ps(vsum, e);
            }
            _mm256_storeu_ps(lanes.as_mut_ptr(), vsum);
            let mut sum: f32 = lanes.iter().sum();
            for i in whole..valid {
                let e = fast_exp(*probs.add(i) - new_m);
                *probs.add(i) = e;
                sum += e;
            }
            for i in valid..C {
                *probs.add(i) = 0.0;
            }
            (corr, sum)
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
        let rows = t.next_multiple_of(R);
        let keys_pad = keys.next_multiple_of(C);
        // the span of (row i0 + R - 1, key j0) starts at pos_base - i0 - R + 1 + j0: one residue of NR for every tile
        let shift = (pos_base + 1) % NR;
        let base = shift as isize - FRONT as isize;
        let p_cols = (pos_base + keys_pad + SPAN + FRONT).next_multiple_of(NR);
        let width = heads * d;
        let head = |src, h: usize| Head {
            src,
            width,
            at: h * d,
            d,
        };
        let qc: Vec<Vec<f32>> = (0..heads)
            .map(|h| pad_rows(head(op.q, h), op.bias_u, op.scale, t, rows))
            .collect();
        let qp: Vec<Vec<f32>> = (0..heads)
            .map(|h| pad_rows(head(op.q, h), op.bias_v, op.scale, t, rows))
            .collect();
        let kp: Vec<Vec<f32>> = (0..heads)
            .map(|h| pack_panels(head(op.k, h), keys, 0, keys_pad))
            .collect();
        let vp: Vec<Vec<f32>> = (0..heads)
            .map(|h| pack_values(head(op.v, h), keys, keys_pad))
            .collect();
        let pp: Vec<Vec<f32>> = (0..heads)
            .map(|h| pack_panels(head(op.p, h), p_rows, base, p_cols))
            .collect();
        let mut out = vec![0f32; t * width];
        let out_at = out.as_mut_ptr() as usize;
        let tasks: Vec<(usize, usize)> = (0..heads)
            .flat_map(|h| (0..rows).step_by(R).map(move |i0| (h, i0)))
            .collect();
        let run = || {
            tasks.par_iter().for_each(|&(h, i0)| {
                let (qc, qp, kp, vp, pp) = (&qc[h], &qp[h], &kp[h], &vp[h], &pp[h]);
                let mut s = vec![0f32; R * C];
                let mut sp = vec![0f32; R * SPAN];
                let mut probs = vec![0f32; R * C];
                let mut o = vec![0f32; R * d];
                let mut m = [f32::NEG_INFINITY; R];
                let mut l = [0f32; R];
                for j0 in (0..keys).step_by(C) {
                    let w0 = (pos_base + j0) as isize - (i0 + R - 1) as isize;
                    let q0 = ((w0 - base) as usize) / NR;
                    // SAFETY: every operand is a padded buffer holding the rows and panels read, and `supports` holds
                    // a panel outer, the row groups inner: each 8 KB panel stays in L1 for all of them, where a chunk's
                    // 67 panels streamed past L2 once a group
                    unsafe {
                        for col_panel in 0..C / NR {
                            let b = kp.as_ptr().add((j0 / NR + col_panel) * d * NR);
                            for g in (0..R).step_by(MR) {
                                let a = qc.as_ptr().add((i0 + g) * d);
                                let dst = s.as_mut_ptr().add(g * C + col_panel * NR);
                                tile(a, d, d, b, dst, C, false);
                            }
                        }
                        for col_panel in 0..SPAN / NR {
                            let b = pp.as_ptr().add((q0 + col_panel) * d * NR);
                            for g in (0..R).step_by(MR) {
                                let a = qp.as_ptr().add((i0 + g) * d);
                                let dst = sp.as_mut_ptr().add(g * SPAN + col_panel * NR);
                                tile(a, d, d, b, dst, SPAN, false);
                            }
                        }
                    }
                    let valid = C.min(keys - j0);
                    for r in 0..R {
                        // SAFETY: row r of the scores, its span's columns from R - 1 - r, and its C probabilities
                        let (corr, sum) = unsafe {
                            row(
                                s.as_ptr().add(r * C),
                                sp.as_ptr().add(r * SPAN + (R - 1 - r)),
                                valid,
                                &mut m[r],
                                probs.as_mut_ptr().add(r * C),
                            )
                        };
                        for v in &mut o[r * d..][..d] {
                            *v *= corr;
                        }
                        l[r] = l[r] * corr + sum;
                    }
                    // SAFETY: as above; the values' panels hold keys_pad rows, so j0 + C is in range
                    unsafe {
                        for kk in (0..C).step_by(PV_DEPTH) {
                            for dp in 0..d / NR {
                                let b = vp.as_ptr().add((dp * keys_pad + j0 + kk) * NR);
                                for g in (0..R).step_by(MR) {
                                    let a = probs.as_ptr().add(g * C + kk);
                                    let dst = o.as_mut_ptr().add(g * d + dp * NR);
                                    tile(a, C, PV_DEPTH, b, dst, d, true);
                                }
                            }
                        }
                    }
                }
                for r in 0..R.min(t.saturating_sub(i0)) {
                    let inv = if l[r] > 0.0 { 1.0 / l[r] } else { 0.0 };
                    // SAFETY: head h's columns of row i0 + r of `out`, written by this task alone
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            (out_at as *mut f32).add((i0 + r) * width + h * d),
                            d,
                        )
                    };
                    for (dst, v) in dst.iter_mut().zip(&o[r * d..][..d]) {
                        *dst = v * inv;
                    }
                }
            });
        };
        // the tiles fill FMA slots a sibling thread can use, as gemm's do (`smt_gemm_pool`)
        match crate::utils::smt_gemm_pool() {
            Some(pool) => pool.install(run),
            None => run(),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Tensor};

    // against its definition in f64: content and position scores (row i reading position row pos_base - i + j) of the
    // biased, scaled queries, the softmax and its values; operands laid out as projections (heads side by side in a
    // row), a partial last query tile and key chunk, head sizes 128, 32 and 16, a run past one key chunk
    #[test]
    fn f32_relative_position_attention_matches_its_definition() -> crate::Result<()> {
        if !super::supports(16) {
            return Ok(());
        }
        // f32 sums against f64 ones, and the exp's ~1e-7
        const TOLERANCE: f64 = 1e-5;
        const SCALE: f32 = 0.3;
        for (t, head) in [(150, 128), (70, 32), (600, 16)] {
            let (heads, p_rows, pos_base) = (2, 2 * t - 1, t - 1);
            let width = heads * head;
            let randn = |rows: usize| Tensor::randn(0f32, 0.5, (1, rows, width), &Device::Cpu);
            let (q, k, v, p) = (randn(t)?, randn(t)?, randn(t)?, randn(p_rows)?);
            let (bias_u, bias_v) = (randn(1)?, randn(1)?);
            let slice = |x: &Tensor| x.flatten_all()?.to_vec1::<f32>();
            let (qv, kv, vv, pv) = (slice(&q)?, slice(&k)?, slice(&v)?, slice(&p)?);
            let (buv, bvv) = (slice(&bias_u)?, slice(&bias_v)?);
            let got = super::attention(&crate::cpu_backend::relpos_attn::RelPosAttn {
                q: &qv,
                bias_u: &buv,
                bias_v: &bvv,
                scale: SCALE,
                k: &kv,
                v: &vv,
                p: &pv,
                heads,
                t,
                keys: t,
                d: head,
                p_rows,
                pos_base,
            });
            let split = |x: &Tensor| -> crate::Result<Tensor> {
                let rows = x.dim(1)?;
                x.reshape((1, rows, heads, head))?
                    .transpose(1, 2)?
                    .to_dtype(DType::F64)
            };
            let scaled = |bias: &Tensor| split(&(q.broadcast_add(bias)? * f64::from(SCALE))?);
            let ids: Vec<u32> = (0..t)
                .flat_map(|i| (0..t).map(move |j| (pos_base - i + j) as u32))
                .collect();
            let ids =
                Tensor::from_vec(ids, (1, 1, t, t), &Device::Cpu)?.repeat((1, heads, 1, 1))?;
            let content = scaled(&bias_u)?.matmul(&split(&k)?.t()?)?;
            let position = scaled(&bias_v)?
                .matmul(&split(&p)?.t()?)?
                .contiguous()?
                .gather(&ids, 3)?;
            let probs = crate::nn::ops::softmax_last_dim(&(content + position)?)?;
            let want = probs
                .matmul(&split(&v)?)?
                .transpose(1, 2)?
                .flatten_all()?
                .to_vec1::<f64>()?;
            let worst = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (f64::from(*g) - w).abs())
                .fold(0f64, f64::max);
            assert!(worst < TOLERANCE, "t {t} head {head}: {worst}");
        }
        Ok(())
    }
}
