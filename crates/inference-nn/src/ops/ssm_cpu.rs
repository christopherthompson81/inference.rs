//! Mamba-2's prefill on the CPU in two passes over plain slices: the causal depthwise convolution, and the selective
//! scan with each head's state kept in cache across the sequence. The tensor-op loop it replaces ran a dozen ops per
//! timestep over the whole state, each one split across the thread pool.

use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

/// The scan's inputs, as `selective_scan_cuda` takes them (all f32, contiguous after the call's copies):
/// `x` `(batch, seq, heads, head_dim)`, `dt` `(batch, seq, heads)` before bias and softplus, `a` `(heads,)` (already
/// negated), `b` and `c` `(batch, seq, groups, state)` (heads split evenly over the groups), `d` and `dt_bias`
/// `(heads,)`.
pub struct SelectiveScan<'a> {
    pub x: &'a Tensor,
    pub dt: &'a Tensor,
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub c: &'a Tensor,
    pub d: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub dt_min: f32,
    pub dt_max: f32,
}

// partial sums the dot with C keeps apart, as the AVX2 lanes do
const DOT_LANES: usize = 8;

/// One state row's step: `row = row * decay + dtx * b`, then `row . c`.
#[inline(always)]
fn state_row_step(row: &mut [f32], b: &[f32], c: &[f32], decay: f32, dtx: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2_fma() && row.len().is_multiple_of(DOT_LANES) {
        // SAFETY: AVX2 and FMA were detected, and the row is whole lanes
        return unsafe { state_row_step_avx2(row, b, c, decay, dtx) };
    }
    let mut lanes = [0f32; DOT_LANES];
    for (i, ((sn, &bn), &cn)) in row.iter_mut().zip(b).zip(c).enumerate() {
        *sn = *sn * decay + dtx * bn;
        lanes[i % DOT_LANES] += *sn * cn;
    }
    lanes.iter().sum()
}

#[cfg(target_arch = "x86_64")]
fn avx2_fma() -> bool {
    static DETECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DETECTED.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn state_row_step_avx2(row: &mut [f32], b: &[f32], c: &[f32], decay: f32, dtx: f32) -> f32 {
    use std::arch::x86_64::*;
    let (vdecay, vdtx) = (_mm256_set1_ps(decay), _mm256_set1_ps(dtx));
    let mut acc = _mm256_setzero_ps();
    for i in (0..row.len()).step_by(DOT_LANES) {
        // SAFETY: `row`, `b` and `c` hold whole lanes from `i`
        unsafe {
            let s = _mm256_loadu_ps(row.as_ptr().add(i));
            // the update as a multiply and an add, as the scalar and CUDA scans round it
            let s = _mm256_add_ps(
                _mm256_mul_ps(s, vdecay),
                _mm256_mul_ps(vdtx, _mm256_loadu_ps(b.as_ptr().add(i))),
            );
            _mm256_storeu_ps(row.as_mut_ptr().add(i), s);
            acc = _mm256_fmadd_ps(s, _mm256_loadu_ps(c.as_ptr().add(i)), acc);
        }
    }
    let mut lanes = [0f32; DOT_LANES];
    // SAFETY: `lanes` holds eight floats
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    lanes.iter().sum()
}

fn f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// `y` `(batch, seq, heads, head_dim)`, with `state` `(batch, heads, head_dim, state)` advanced over the sequence: per
/// step `dt = clamp(softplus(dt + dt_bias))`, `state = state * exp(dt * a) + dt * b * x`, `y = state . c + d * x`.
pub fn selective_scan_cpu(scan: SelectiveScan<'_>, state: &mut Tensor) -> Result<Tensor> {
    let (batch, seq, heads, head_dim) = scan.x.dims4()?;
    let (groups, d_state) = (scan.b.dim(2)?, scan.b.dim(3)?);
    let heads_per_group = heads / groups;
    let (x, dt, a, b, c, d, dt_bias) = (
        f32_vec(scan.x)?,
        f32_vec(scan.dt)?,
        f32_vec(scan.a)?,
        f32_vec(scan.b)?,
        f32_vec(scan.c)?,
        f32_vec(scan.d)?,
        f32_vec(scan.dt_bias)?,
    );
    let mut states = f32_vec(state)?;
    let mut y = vec![0f32; batch * seq * heads * head_dim];
    let head_state = head_dim * d_state;
    // one item per (batch, head): its state block and its outputs, strided through `y`
    let mut outputs: Vec<Vec<f32>> = vec![vec![0f32; seq * head_dim]; batch * heads];
    states
        .par_chunks_mut(head_state)
        .zip(outputs.par_iter_mut())
        .enumerate()
        .for_each(|(item, (s, out))| {
            let (bi, h) = (item / heads, item % heads);
            for t in 0..seq {
                let step = (bi * seq + t) * heads + h;
                let raw = dt[step] + dt_bias[h];
                let dt_t = (1.0 + raw.exp()).ln().clamp(scan.dt_min, scan.dt_max);
                let decay = (dt_t * a[h]).exp();
                let group_step = (bi * seq + t) * groups + h / heads_per_group;
                let bt = &b[group_step * d_state..(group_step + 1) * d_state];
                let ct = &c[group_step * d_state..(group_step + 1) * d_state];
                let xt = &x[step * head_dim..(step + 1) * head_dim];
                for (p, row) in s.chunks_exact_mut(d_state).enumerate() {
                    out[t * head_dim + p] =
                        state_row_step(row, bt, ct, decay, dt_t * xt[p]) + d[h] * xt[p];
                }
            }
        });
    for (item, out) in outputs.iter().enumerate() {
        let (bi, h) = (item / heads, item % heads);
        for t in 0..seq {
            let at = ((bi * seq + t) * heads + h) * head_dim;
            y[at..at + head_dim].copy_from_slice(&out[t * head_dim..(t + 1) * head_dim]);
        }
    }
    *state =
        Tensor::from_vec(states, state.shape().clone(), &Device::Cpu)?.to_dtype(state.dtype())?;
    Tensor::from_vec(y, (batch, seq, heads, head_dim), &Device::Cpu)
}

/// The causal depthwise convolution over `padded` `(batch, channels, kernel - 1 + seq)` with `weight`
/// `(channels, kernel)`, as `(batch, seq, channels)` in `padded`'s dtype: output step `t` sums taps `t..t + kernel`.
pub fn causal_conv1d_cpu(padded: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (batch, channels, width) = padded.dims3()?;
    let kernel = weight.dim(1)?;
    let seq = width + 1 - kernel;
    let (input, weight) = (f32_vec(padded)?, f32_vec(weight)?);
    let mut out = vec![0f32; batch * seq * channels];
    // each output row (one step's channels) reads every channel's window at that step
    out.par_chunks_mut(channels)
        .enumerate()
        .for_each(|(row, dst)| {
            let (bi, t) = (row / seq, row % seq);
            for (ch, o) in dst.iter_mut().enumerate() {
                let src = &input[(bi * channels + ch) * width + t..][..kernel];
                let w = &weight[ch * kernel..(ch + 1) * kernel];
                // an index loop: the zipped sum stayed an out-of-line call per channel in the dev profile
                let mut sum = 0f32;
                for j in 0..kernel {
                    sum += src[j] * w[j];
                }
                *o = sum;
            }
        });
    Tensor::from_vec(out, (batch, seq, channels), &Device::Cpu)?.to_dtype(padded.dtype())
}
