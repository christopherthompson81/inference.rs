//! f32 matmuls from bf16 inputs on AVX-512 BF16 (`vdpbf16ps`: 32 multiply-adds an instruction, twice f32 FMA's), the
//! products exact in f32 and the sums in f32. Both operands are packed once a call: A's rows as bf16 pairs along k, B
//! as panels of `NR` columns whose 16-lane loads hold each column's pair. A task takes a block of rows over a group of
//! panels, each `MR` x `NR` tile summing the whole of k in registers.

#[cfg(target_arch = "x86_64")]
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// a tile's rows (each a broadcast pair a step) and columns (two 16-lane accumulators)
#[cfg(target_arch = "x86_64")]
const MR: usize = 6;
#[cfg(target_arch = "x86_64")]
const NR: usize = 32;
#[cfg(target_arch = "x86_64")]
const LANES: usize = 16;
// rows and panels a task takes at most (a panel stays in cache over its rows' tiles), fewer where the product would
// leave threads idle: TASKS_PER_THREAD a thread at least
#[cfg(target_arch = "x86_64")]
const MC: usize = 96;
#[cfg(target_arch = "x86_64")]
const PANELS_PER_TASK: usize = 4;
#[cfg(target_arch = "x86_64")]
const TASKS_PER_THREAD: usize = 4;

/// Set to keep CPU f32 matmuls in f32 where a model would let them take bf16 inputs (for accuracy, or A/B runs).
pub const NO_BF16_ENV: &str = "INFERENCE_CPU_NO_BF16";
/// Set to let every model's CPU forward take bf16 matmul inputs, to measure one not yet opted in.
pub const BF16_ENV: &str = "INFERENCE_CPU_BF16";

/// Whether [`BF16_ENV`] lets every model take bf16 matmul inputs.
pub fn forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| std::env::var_os(BF16_ENV).is_some())
}

std::thread_local! {
    // whether this thread's matmuls run inside `with_bf16_matmuls`
    static IN_SCOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread's large f32 matmuls take bf16 inputs: inside [`with_bf16_matmuls`], on a CPU that multiplies
/// bf16 pairs, with [`NO_BF16_ENV`] unset.
pub fn enabled() -> bool {
    IN_SCOPE.get() && allowed()
}

/// Whether a scope may take bf16 matmul inputs here: a CPU that multiplies bf16 pairs, with [`NO_BF16_ENV`] unset.
pub fn allowed() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOWED.get_or_init(|| available() && std::env::var_os(NO_BF16_ENV).is_none())
}

/// Whether a model measured under bf16 may keep bf16 weights on a CPU without bf16 pairs (widened as its products pack
/// them): [`NO_BF16_ENV`] unset.
pub fn weights_allowed() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOWED.get_or_init(|| std::env::var_os(NO_BF16_ENV).is_none())
}

/// Whether this thread is inside a model's bf16 scope ([`with_bf16_matmuls`]) with [`NO_BF16_ENV`] unset, whatever
/// the CPU multiplies: bf16-held weights that widen to f32 for the products need no bf16 pairs.
pub fn in_scope() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    IN_SCOPE.get() && *ALLOWED.get_or_init(|| std::env::var_os(NO_BF16_ENV).is_none())
}

/// Runs `f` with this thread's large f32 CPU matmuls taking bf16 inputs, where the CPU multiplies bf16 pairs and
/// [`NO_BF16_ENV`] is unset: a model measured to hold under them opts in around its forward (others keep f32).
pub fn with_bf16_matmuls<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            IN_SCOPE.set(self.0);
        }
    }
    let _restore = Restore(IN_SCOPE.replace(true));
    f()
}

/// Whether this CPU multiplies bf16 pairs (`avx512bf16`).
pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// An operand's elements and strides: element (r, c) at `data[r * rs + c * cs]`.
// read by the x86_64 kernel alone
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub struct Strided<'a> {
    pub data: &'a [f32],
    pub rs: usize,
    pub cs: usize,
}

/// B's elements: f32, rounded to bf16 as they are packed, or bf16 already (a weight kept so, `Linear::to_half`).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub enum BOperand<'a> {
    F32(Strided<'a>),
    Bf16 {
        data: &'a [u16],
        rs: usize,
        cs: usize,
    },
}

// round to nearest even, as `vcvtneps2bf16` does
#[cfg(target_arch = "x86_64")]
fn bf16_bits(x: f32) -> u32 {
    half::bf16::from_f32(x).to_bits() as u32
}

// `src` (k values) as k / 2 pairs into `out` (an odd k's last half zero): 16 values an instruction, a scalar tail
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_pairs(src: &[f32], out: &mut [u32]) {
    let whole = src.len() / LANES * LANES;
    // SAFETY: callers run only where `available`; each step reads LANES values and writes LANES / 2 pairs in range
    unsafe { row_pairs_avx512(&src[..whole], out.as_mut_ptr()) };
    for (p, pair) in out.iter_mut().enumerate().skip(whole / 2) {
        let hi = src.get(2 * p + 1).map_or(0, |&x| bf16_bits(x));
        *pair = bf16_bits(src[2 * p]) | hi << 16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn row_pairs_avx512(src: &[f32], out: *mut u32) {
    unsafe {
        for (i, chunk) in src.as_chunks::<LANES>().0.iter().enumerate() {
            let halves = _mm512_cvtneps_pbh(_mm512_loadu_ps(chunk.as_ptr()));
            _mm256_storeu_si256(
                out.add(i * LANES / 2).cast(),
                std::mem::transmute::<__m256bh, __m256i>(halves),
            );
        }
    }
}

/// `dst` (`batch` products of `m` x `n`, row-major, one after another) = `a` (`m` x `k`) `b` (`k` x `n`) through
/// bf16 inputs, product `i`'s operands `a_skip` and `b_skip` elements on; call only where [`available`].
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &Strided,
    a_skip: usize,
    b: &BOperand,
    b_skip: usize,
    dst: &mut [f32],
) {
    let pairs = k.div_ceil(2);
    let rows = m.next_multiple_of(MR);
    let panels = n.div_ceil(NR);
    // A as (batch, rows, pairs) u32s, its padding rows and odd k's last half zero
    let mut ap = vec![0u32; batch * rows * pairs];
    ap.par_chunks_mut(pairs).enumerate().for_each(|(i, row)| {
        let (bi, r) = (i / rows, i % rows);
        if r >= m {
            return;
        }
        if a.cs == 1 {
            row_pairs(&a.data[bi * a_skip + r * a.rs..][..k], row);
            return;
        }
        let at = |kk: usize| a.data[bi * a_skip + r * a.rs + kk * a.cs];
        for (p, pair) in row.iter_mut().enumerate() {
            let hi = if 2 * p + 1 < k {
                bf16_bits(at(2 * p + 1))
            } else {
                0
            };
            *pair = bf16_bits(at(2 * p)) | hi << 16;
        }
    });
    // B as (batch, panels) panels of (pairs, NR) u32s, its padding columns zero
    let mut bp = vec![0u32; batch * panels * pairs * NR];
    bp.par_chunks_mut(pairs * NR)
        .enumerate()
        .for_each(|(i, out)| {
            let (bi, panel) = (i / panels, i % panels);
            let b = match b {
                BOperand::F32(b) => b,
                &BOperand::Bf16 { data, rs, cs } => {
                    for j in 0..NR.min(n - panel * NR) {
                        let at = bi * b_skip + (panel * NR + j) * cs;
                        let half = |kk: usize| {
                            if kk < k {
                                u32::from(data[at + kk * rs])
                            } else {
                                0
                            }
                        };
                        for p in 0..pairs {
                            out[p * NR + j] = half(2 * p) | half(2 * p + 1) << 16;
                        }
                    }
                    return;
                }
            };
            // a column contiguous along k (a transposed weight's rows) converts a vector at a time
            let mut column = vec![0u32; if b.rs == 1 { pairs } else { 0 }];
            for j in 0..NR.min(n - panel * NR) {
                let col = panel * NR + j;
                if b.rs == 1 {
                    row_pairs(&b.data[bi * b_skip + col * b.cs..][..k], &mut column);
                    for (p, pair) in column.iter().enumerate() {
                        out[p * NR + j] = *pair;
                    }
                    continue;
                }
                let at = |kk: usize| b.data[bi * b_skip + kk * b.rs + col * b.cs];
                for p in 0..pairs {
                    let hi = if 2 * p + 1 < k {
                        bf16_bits(at(2 * p + 1))
                    } else {
                        0
                    };
                    out[p * NR + j] = bf16_bits(at(2 * p)) | hi << 16;
                }
            }
        });
    let dst_at = dst.as_mut_ptr() as usize;
    // smaller tasks until every thread has a few: a 128-column product has 4 panels
    let want = rayon::current_num_threads() * TASKS_PER_THREAD;
    let (mut mc, mut pc) = (MC, PANELS_PER_TASK);
    let count = |mc: usize, pc: usize| batch * rows.div_ceil(mc) * panels.div_ceil(pc);
    while count(mc, pc) < want && (pc > 1 || mc > MR) {
        if pc > 1 {
            pc /= 2;
        } else {
            mc = (mc / 2).max(MR).next_multiple_of(MR);
        }
    }
    let tasks: Vec<(usize, usize, usize)> = (0..batch)
        .flat_map(|bi| {
            (0..rows)
                .step_by(mc)
                .flat_map(move |r0| (0..panels).step_by(pc).map(move |p0| (bi, r0, p0)))
        })
        .collect();
    tasks.into_par_iter().for_each(|(bi, r0, p0)| {
        for panel in p0..(p0 + pc).min(panels) {
            let cols = NR.min(n - panel * NR);
            let b_panel = &bp[(bi * panels + panel) * pairs * NR..][..pairs * NR];
            for r in (r0..(r0 + mc).min(rows)).step_by(MR) {
                let live = MR.min(m.saturating_sub(r));
                if live == 0 {
                    break;
                }
                // SAFETY: the tile's rows are r..r + live < m and its columns panel * NR + cols <= n, each written by
                // this task alone; `available` was checked by the caller
                unsafe {
                    tile(
                        &ap[(bi * rows + r) * pairs..][..MR * pairs],
                        pairs,
                        b_panel,
                        (dst_at as *mut f32).add(bi * m * n + r * n + panel * NR),
                        n,
                        live,
                        cols,
                    )
                };
            }
        }
    });
}

/// [`available`] is false off x86_64, so nothing calls this there.
#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    _batch: usize,
    _m: usize,
    _n: usize,
    _k: usize,
    _a: &Strided,
    _a_skip: usize,
    _b: &BOperand,
    _b_skip: usize,
    _dst: &mut [f32],
) {
    unreachable!("bf16 matmuls run on x86_64's AVX-512 BF16")
}

// one MR x NR tile over every pair: `a` holds MR rows of `pairs`, `b` a panel; `live` rows and `cols` columns stored
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn tile(
    a: &[u32],
    pairs: usize,
    b: &[u32],
    dst: *mut f32,
    dst_rs: usize,
    live: usize,
    cols: usize,
) {
    unsafe {
        let mut acc = [[_mm512_setzero_ps(); 2]; MR];
        let (a, b) = (a.as_ptr(), b.as_ptr());
        for p in 0..pairs {
            let b0: __m512bh = std::mem::transmute(_mm512_loadu_si512(b.add(p * NR).cast()));
            let b1: __m512bh =
                std::mem::transmute(_mm512_loadu_si512(b.add(p * NR + LANES).cast()));
            for (r, acc) in acc.iter_mut().enumerate() {
                let av: __m512bh =
                    std::mem::transmute(_mm512_set1_epi32(*a.add(r * pairs + p) as i32));
                acc[0] = _mm512_dpbf16_ps(acc[0], av, b0);
                acc[1] = _mm512_dpbf16_ps(acc[1], av, b1);
            }
        }
        let lo = if cols >= LANES {
            u16::MAX
        } else {
            (1u16 << cols) - 1
        };
        let hi = if cols >= NR {
            u16::MAX
        } else if cols > LANES {
            (1u16 << (cols - LANES)) - 1
        } else {
            0
        };
        for (r, acc) in acc.iter().enumerate().take(live) {
            let row = dst.add(r * dst_rs);
            _mm512_mask_storeu_ps(row, lo, acc[0]);
            _mm512_mask_storeu_ps(row.add(LANES), hi, acc[1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // a model's bf16 scope holds on any CPU, apart from whether its products take bf16 pairs
    #[test]
    fn the_scope_holds_whatever_the_cpu_multiplies() {
        assert!(!super::in_scope());
        assert!(super::with_bf16_matmuls(super::in_scope));
        assert!(!super::in_scope());
    }

    // a large f32 matmul takes bf16 inputs inside the scope alone: outside it the product is f32's, inside it the
    // rounded inputs'
    #[test]
    fn only_a_scoped_matmul_takes_bf16_inputs() -> crate::Result<()> {
        if !available() {
            return Ok(());
        }
        use crate::{DType, Device, Tensor};
        // 2^26 multiply-adds over 256 columns, the smallest product the path takes
        let (m, n, k) = (512, 256, 512);
        let a = Tensor::randn(0f32, 1., (m, k), &Device::Cpu)?;
        let b = Tensor::randn(0f32, 1., (k, n), &Device::Cpu)?;
        let rounded = |t: &Tensor| t.to_dtype(DType::BF16)?.to_dtype(DType::F32);
        let outside = a.matmul(&b)?;
        let inside = with_bf16_matmuls(|| a.matmul(&b))?;
        let f32_ref = a
            .to_dtype(DType::F64)?
            .matmul(&b.to_dtype(DType::F64)?)?
            .to_dtype(DType::F32)?;
        let bf16_ref = rounded(&a)?
            .to_dtype(DType::F64)?
            .matmul(&rounded(&b)?.to_dtype(DType::F64)?)?
            .to_dtype(DType::F32)?;
        let off = |x: &Tensor, y: &Tensor| -> crate::Result<f32> {
            (x - y)?.abs()?.max_all()?.to_scalar::<f32>()
        };
        assert!(off(&outside, &f32_ref)? < 1e-3, "outside the scope: f32");
        assert!(
            off(&inside, &bf16_ref)? < 1e-3,
            "inside the scope: bf16 inputs"
        );
        assert!(off(&inside, &outside)? > 1e-2, "the scope changed nothing");
        Ok(())
    }

    // a row's pairs from 16 values an instruction equal the scalar rounding's, a partial vector and odd k included
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vector_row_pairs_match_scalar_ones() {
        if !available() {
            return;
        }
        for k in [1usize, 15, 16, 17, 33, 100] {
            let src: Vec<f32> = (0..k)
                .map(|i| ((i * 2654435761) % 1000) as f32 / 37.0 - 13.0)
                .collect();
            let mut got = vec![0u32; k.div_ceil(2)];
            row_pairs(&src, &mut got);
            let want: Vec<u32> = (0..k.div_ceil(2))
                .map(|p| {
                    bf16_bits(src[2 * p]) | src.get(2 * p + 1).map_or(0, |&x| bf16_bits(x)) << 16
                })
                .collect();
            assert_eq!(got, want, "k {k}");
        }
    }

    // against f32 sums over the same bf16-rounded inputs: edge tiles (rows, columns, odd k) and strided operands
    #[test]
    fn bf16_products_match_f32_sums_of_rounded_inputs() {
        if !available() {
            return;
        }
        const TOLERANCE: f32 = 1e-5;
        let round = |x: f32| half::bf16::from_f32(x).to_f32();
        let value = |i: usize| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0;
        for (m, n, k) in [(13, 45, 33), (6, 32, 2), (100, 70, 257), (1, 1, 1)] {
            let a: Vec<f32> = (0..m * k).map(value).collect();
            // b transposed in memory: (k x n) at strides (1, k)
            let b: Vec<f32> = (0..n * k).map(|i| value(i + 7)).collect();
            let mut got = vec![0f32; m * n];
            gemm(
                1,
                m,
                n,
                k,
                &Strided {
                    data: &a,
                    rs: k,
                    cs: 1,
                },
                0,
                &BOperand::F32(Strided {
                    data: &b,
                    rs: 1,
                    cs: k,
                }),
                0,
                &mut got,
            );
            for r in 0..m {
                for c in 0..n {
                    let want: f32 = (0..k)
                        .map(|kk| round(a[r * k + kk]) * round(b[c * k + kk]))
                        .sum();
                    let g = got[r * n + c];
                    assert!(
                        (g - want).abs() <= TOLERANCE * want.abs().max(1.0) * (k as f32).sqrt(),
                        "{m}x{n}x{k} ({r}, {c}): {g} vs {want}"
                    );
                }
            }
        }
    }
}
