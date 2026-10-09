//! A direct AVX2/FMA f32 conv1d whose register tile reads the padded input in place, writing no im2col buffer.

use std::sync::OnceLock;

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::conv::ParamsConv1D;
use crate::{Layout, Result};

// output channels and time steps of the register tile: 6 x 2 ymm accumulators
const MR: usize = 6;
const NR: usize = 16;
// time steps per task: every channel block runs over this window while its input slice sits in L2
const TIME_BLOCK: usize = 4 * NR;
// below this kernel width im2col's single GEMM is as fast
const MIN_KERNEL: usize = 5;

pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Whether the direct kernel takes this conv: f32 handled by the caller, stride 1, one group, a wide kernel.
pub fn applies(p: &ParamsConv1D) -> bool {
    available() && p.stride == 1 && p.groups == 1 && p.k_size >= MIN_KERNEL && p.c_in > 0
}

#[derive(Clone, Copy)]
struct Out(*mut f32);
// SAFETY: each task writes only its own (batch, time block, channel blocks) tiles of the output.
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    // a method, so closures capture the Sync wrapper rather than its raw pointer field
    fn at(&self, offset: usize) -> *mut f32 {
        self.0.wrapping_add(offset)
    }
}

/// `out[b, o, t] = sum_i sum_k w[o, i, k] * x[b, i, t + k * dilation - padding]`, as (b, c_out, l_out) contiguous.
pub fn conv1d(
    src: &[f32],
    l: &Layout,
    kernel: &[f32],
    kernel_l: &Layout,
    p: &ParamsConv1D,
) -> Result<Vec<f32>> {
    debug_assert!(
        applies(p),
        "the direct conv1d was called outside the shapes it takes"
    );
    let (b, c_in, l_in) = l.shape().dims3()?;
    let (s0, s1, s2) = (l.stride()[0], l.stride()[1], l.stride()[2]);
    let (k0, k1, k2) = (
        kernel_l.stride()[0],
        kernel_l.stride()[1],
        kernel_l.stride()[2],
    );
    let (src, kernel) = (&src[l.start_offset()..], &kernel[kernel_l.start_offset()..]);
    let (c_out, k_size, dilation, l_out) = (p.c_out, p.k_size, p.dilation, p.l_out());
    let blocks = c_out.div_ceil(MR);
    // [block][i][k][r], zero past c_out
    let mut packed = vec![0f32; blocks * c_in * k_size * MR];
    packed
        .par_chunks_mut(c_in * k_size * MR)
        .enumerate()
        .for_each(|(block, dst)| {
            for i in 0..c_in {
                for k in 0..k_size {
                    for r in 0..MR.min(c_out - block * MR) {
                        dst[(i * k_size + k) * MR + r] =
                            kernel[(block * MR + r) * k0 + i * k1 + k * k2];
                    }
                }
            }
        });
    // each row padded on both sides, with NR spare steps so a tile's 16-wide loads never leave the buffer
    let row_len = l_in + 2 * p.padding + NR;
    let mut padded = vec![0f32; b * c_in * row_len];
    padded
        .par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(row, dst)| {
            let base = (row / c_in) * s0 + (row % c_in) * s1;
            for (t, d) in dst[p.padding..p.padding + l_in].iter_mut().enumerate() {
                *d = src[base + t * s2];
            }
        });
    let mut out = vec![0f32; b * c_out * l_out];
    let dst = Out(out.as_mut_ptr());
    let time_blocks = l_out.div_ceil(TIME_BLOCK);
    // too few (batch, time block) tasks to fill the pool: split each one's channel blocks as well
    let chunks = rayon::current_num_threads()
        .div_ceil(b * time_blocks)
        .clamp(1, blocks.max(1));
    let per_chunk = blocks.div_ceil(chunks);
    (0..b * time_blocks * chunks)
        .into_par_iter()
        .for_each(|task| {
            let (batch, tb, chunk) = (
                task / (time_blocks * chunks),
                task / chunks % time_blocks,
                task % chunks,
            );
            let input = &padded[batch * c_in * row_len..(batch + 1) * c_in * row_len];
            for block in chunk * per_chunk..blocks.min((chunk + 1) * per_chunk) {
                let w = &packed[block * c_in * k_size * MR..(block + 1) * c_in * k_size * MR];
                let rows = MR.min(c_out - block * MR);
                for t0 in (tb * TIME_BLOCK..l_out.min((tb + 1) * TIME_BLOCK)).step_by(NR) {
                    let cols = NR.min(l_out - t0);
                    let o = dst.at((batch * c_out + block * MR) * l_out + t0);
                    // SAFETY: applies() held, so AVX2 and FMA exist; reads stay in `input` and `w`, writes in this tile
                    unsafe {
                        tile(
                            input.as_ptr().add(t0),
                            row_len,
                            w.as_ptr(),
                            c_in,
                            k_size,
                            dilation,
                            o,
                            l_out,
                            rows,
                            cols,
                        )
                    };
                }
            }
        });
    Ok(out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn tile(
    x: *const f32,
    row_len: usize,
    w: *const f32,
    c_in: usize,
    k_size: usize,
    dilation: usize,
    out: *mut f32,
    out_rs: usize,
    rows: usize,
    cols: usize,
) {
    let mut acc = [[_mm256_setzero_ps(); 2]; MR];
    for i in 0..c_in {
        let row = unsafe { x.add(i * row_len) };
        let wi = unsafe { w.add(i * k_size * MR) };
        for k in 0..k_size {
            let (x0, x1) = unsafe {
                (
                    _mm256_loadu_ps(row.add(k * dilation)),
                    _mm256_loadu_ps(row.add(k * dilation + 8)),
                )
            };
            let wk = unsafe { wi.add(k * MR) };
            for (r, a) in acc.iter_mut().enumerate() {
                let v = unsafe { _mm256_broadcast_ss(&*wk.add(r)) };
                a[0] = _mm256_fmadd_ps(v, x0, a[0]);
                a[1] = _mm256_fmadd_ps(v, x1, a[1]);
            }
        }
    }
    if rows == MR && cols == NR {
        for (r, a) in acc.iter().enumerate() {
            let p = unsafe { out.add(r * out_rs) };
            unsafe {
                _mm256_storeu_ps(p, a[0]);
                _mm256_storeu_ps(p.add(8), a[1]);
            }
        }
        return;
    }
    let mut scratch = [0f32; MR * NR];
    for (r, a) in acc.iter().enumerate() {
        unsafe {
            _mm256_storeu_ps(scratch.as_mut_ptr().add(r * NR), a[0]);
            _mm256_storeu_ps(scratch.as_mut_ptr().add(r * NR + 8), a[1]);
        }
    }
    for r in 0..rows {
        for c in 0..cols {
            unsafe { *out.add(r * out_rs + c) = scratch[r * NR + c] };
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile(
    _: *const f32,
    _: usize,
    _: *const f32,
    _: usize,
    _: usize,
    _: usize,
    _: *mut f32,
    _: usize,
    _: usize,
    _: usize,
) {
    unreachable!("the direct conv runs only where available() is true")
}
