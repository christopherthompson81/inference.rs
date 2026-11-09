//! The f32 FMA tile the AVX2 kernels multiply with: `MR` rows of `a` against an `NR`-column panel of `b`, 12 8-lane
//! accumulators, two panel loads and a broadcast a step.

use std::arch::x86_64::*;

// a tile's rows, and a panel's columns (two 8-lane accumulators)
pub(crate) const MR: usize = 6;
pub(crate) const NR: usize = 16;
const LANES: usize = 8;

/// An MR x NR tile: `a`'s MR rows (`a_stride` floats apart) over `depth` columns of panel `b` (`depth` rows of NR),
/// into `dst` (`dst_rs` floats a row), added to what it holds where `add`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn tile(
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
