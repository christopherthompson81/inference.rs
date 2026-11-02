//! Mamba-2's prefill on the CPU in two passes over plain slices: the causal depthwise convolution, and the selective
//! scan with each head's state kept in cache across the sequence. The tensor-op loop it replaces ran a dozen ops per
//! timestep over the whole state, each one split across the thread pool.

use inference_tensor::{DType, Device, Result, Tensor};
use rayon::prelude::*;

/// The scan's inputs, as `selective_scan_cuda` takes them (all f32, contiguous after the call's copies):
/// `x` `(batch, seq, heads, head_dim)`, `dt` `(batch, seq, heads)` before bias and softplus, `a` `(heads,)` (already
/// negated), `b` and `c` `(batch, seq, heads, state)` (expanded from groups), `d` and `dt_bias` `(heads,)`.
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

fn f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// `y` `(batch, seq, heads, head_dim)`, with `state` `(batch, heads, head_dim, state)` advanced over the sequence: per
/// step `dt = clamp(softplus(dt + dt_bias))`, `state = state * exp(dt * a) + dt * b * x`, `y = state . c + d * x`.
pub fn selective_scan_cpu(scan: SelectiveScan<'_>, state: &mut Tensor) -> Result<Tensor> {
    let (batch, seq, heads, head_dim) = scan.x.dims4()?;
    let d_state = scan.b.dim(3)?;
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
                let bt = &b[step * d_state..(step + 1) * d_state];
                let ct = &c[step * d_state..(step + 1) * d_state];
                let xt = &x[step * head_dim..(step + 1) * head_dim];
                for (p, row) in s.chunks_exact_mut(d_state).enumerate() {
                    let dtx = dt_t * xt[p];
                    let mut acc = 0f32;
                    for ((sn, &bn), &cn) in row.iter_mut().zip(bt).zip(ct) {
                        *sn = *sn * decay + dtx * bn;
                        acc += *sn * cn;
                    }
                    out[t * head_dim + p] = acc + d[h] * xt[p];
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
                *o = src.iter().zip(w).map(|(x, w)| x * w).sum();
            }
        });
    Tensor::from_vec(out, (batch, seq, channels), &Device::Cpu)?.to_dtype(padded.dtype())
}
