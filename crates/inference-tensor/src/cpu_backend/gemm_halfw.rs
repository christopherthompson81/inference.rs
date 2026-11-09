//! `x` (f32 rows) times a 16-bit `(n, k)` weight's transpose (f16 or bf16 bits) on AVX2 and FMA, for a CPU without
//! bf16 pairs: the weight widens to f32 as it packs (a 64-column block, `KC` deep at a time, in L1-sized panels), and
//! the products and their sums stay f32, the f32 layer's over its rounded weight. A task takes `MC` rows by `NC`
//! columns.

/// How a weight's 16 bits widen to f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    F16,
    Bf16,
}

/// Whether the kernel takes the product: whole panels of columns, on a CPU with AVX2 and FMA.
pub fn supports(n: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        n.is_multiple_of(x86::NC)
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = n;
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::gemm;

/// [`supports`] is false off x86_64, so nothing calls this there.
#[cfg(not(target_arch = "x86_64"))]
pub fn gemm(
    _x: &[f32],
    _w: &[u16],
    _half: Half,
    _rows: usize,
    _n: usize,
    _k: usize,
    _out: &mut [f32],
) {
    unreachable!("the bf16-weight gemm runs on x86_64's AVX2")
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Half;
    use crate::cpu_backend::fma_tile::{MR, NR, tile};
    use rayon::prelude::*;

    // a task's rows (a multiple of MR) and columns (whole panels), and the depth a packed block holds: a 16 KB panel
    // of KC x NR floats stays in L1 while the task's MC rows pass over it
    const MC: usize = 192;
    pub(super) const NC: usize = 64;
    const KC: usize = 256;

    fn widen(bits: u16, half: Half) -> f32 {
        match half {
            Half::F16 => half::f16::from_bits(bits).to_f32(),
            Half::Bf16 => f32::from_bits(u32::from(bits) << 16),
        }
    }

    /// `out` (`rows x n`, written whole) = `x` (`rows x k`, row-major) times `w` (`n x k` 16-bit values of `half`,
    /// row-major) transposed; call only where [`super::supports`] takes `n`.
    pub fn gemm(
        x: &[f32],
        w: &[u16],
        half: Half,
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) {
        let out_at = out.as_mut_ptr() as usize;
        let tasks: Vec<(usize, usize)> = (0..rows)
            .step_by(MC)
            .flat_map(|r0| (0..n).step_by(NC).map(move |c0| (r0, c0)))
            .collect();
        let run = || {
            tasks.par_iter().for_each(|&(r0, c0)| {
                let live = MC.min(rows - r0);
                let mut panels = vec![0f32; NC * KC];
                // a partial last row group's rows, zero past the live ones, and its tile's outputs
                let mut pad_a = vec![0f32; MR * KC];
                let mut pad_out = [0f32; MR * NC];
                for k0 in (0..k).step_by(KC) {
                    let depth = KC.min(k - k0);
                    // panel q holds columns c0 + q NR.., depth rows of NR
                    for (q, panel) in panels
                        .as_chunks_mut::<{ KC * NR }>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        for j in 0..NR {
                            let src = &w[(c0 + q * NR + j) * k + k0..][..depth];
                            for (p, bits) in src.iter().enumerate() {
                                panel[p * NR + j] = widen(*bits, half);
                            }
                        }
                    }
                    let add = k0 > 0;
                    for g in (0..live).step_by(MR) {
                        let whole = g + MR <= live;
                        let (a, a_stride) = if whole {
                            (x[(r0 + g) * k + k0..].as_ptr(), k)
                        } else {
                            for r in 0..MR {
                                let row = &mut pad_a[r * KC..][..depth];
                                if g + r < live {
                                    row.copy_from_slice(&x[(r0 + g + r) * k + k0..][..depth]);
                                } else {
                                    row.fill(0.0);
                                }
                            }
                            (pad_a.as_ptr(), KC)
                        };
                        for q in 0..NC / NR {
                            let b = panels[q * KC * NR..].as_ptr();
                            // SAFETY: `a` holds MR rows of `depth` (`a_stride` apart), `b` a panel of depth x NR, and the
                            // destination MR rows of NR (`out`'s rows of this task's columns, or the padded tile's)
                            unsafe {
                                if whole {
                                    let dst = (out_at as *mut f32).add((r0 + g) * n + c0 + q * NR);
                                    tile(a, a_stride, depth, b, dst, n, add);
                                } else {
                                    tile(
                                        a,
                                        a_stride,
                                        depth,
                                        b,
                                        pad_out.as_mut_ptr().add(q * NR),
                                        NC,
                                        false,
                                    );
                                }
                            }
                        }
                        if !whole {
                            for r in 0..live - g {
                                // SAFETY: row r0 + g + r of `out`, this task's columns alone
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        (out_at as *mut f32).add((r0 + g + r) * n + c0),
                                        NC,
                                    )
                                };
                                for (d, v) in dst.iter_mut().zip(&pad_out[r * NC..][..NC]) {
                                    *d = if add { *d + v } else { *v };
                                }
                            }
                        }
                    }
                }
            });
        };
        // the tiles fill FMA slots a sibling thread can use (`smt_gemm_pool`)
        match crate::utils::smt_gemm_pool() {
            Some(pool) => pool.install(run),
            None => run(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Half;
    use crate::{DType, Device, Tensor};

    // a weight rounded to `half`, and its bits
    fn rounded(w: &Tensor, half: Half) -> crate::Result<(Tensor, Vec<u16>)> {
        Ok(match half {
            Half::F16 => {
                let w = w.to_dtype(DType::F16)?;
                let bits = w.flatten_all()?.to_vec1::<half::f16>()?;
                (w, bits.into_iter().map(half::f16::to_bits).collect())
            }
            Half::Bf16 => {
                let w = w.to_dtype(DType::BF16)?;
                let bits = w.flatten_all()?.to_vec1::<half::bf16>()?;
                (w, bits.into_iter().map(half::bf16::to_bits).collect())
            }
        })
    }

    // against the f32 product over the rounded weight, f16 and bf16: a partial last row group and task, a depth past one
    // block and a ragged last one, two column tasks
    #[test]
    fn a_half_weight_product_matches_the_rounded_weights() -> crate::Result<()> {
        if !super::supports(128) {
            return Ok(());
        }
        // f32 sums in a different order, over k up to 600
        const TOLERANCE: f32 = 1e-3;
        for half in [Half::F16, Half::Bf16] {
            for (rows, n, k) in [(101usize, 128usize, 600usize), (5, 64, 40), (200, 192, 256)] {
                let x = Tensor::randn(0f32, 1., (rows, k), &Device::Cpu)?;
                let (w, bits) = rounded(&Tensor::randn(0f32, 1., (n, k), &Device::Cpu)?, half)?;
                let mut out = vec![f32::NAN; rows * n];
                let xs = x.flatten_all()?.to_vec1::<f32>()?;
                super::gemm(&xs, &bits, half, rows, n, k, &mut out);
                let want = x
                    .matmul(&w.to_dtype(DType::F32)?.t()?)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let worst = out
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| (g - w).abs())
                    .fold(0f32, f32::max);
                assert!(worst < TOLERANCE, "{half:?} {rows} x {n} x {k}: {worst}");
            }
        }
        Ok(())
    }
}
