//! RWKV-7's WKV recurrence (a generalized delta rule with a decay per key channel).

use inference_tensor::{DType, Result, Tensor};

/// One WKV-7 pass, every input f32 `(batch, seq, heads, head_size)`: the receptance `r`, the decay `w` (already
/// `exp(-0.6065 sigmoid(..))`), `k`, `v`, and the state update's `a` and `b` (RWKV-7's `-kk` and `kk * a`).
#[derive(Clone, Copy)]
pub struct Wkv7<'a> {
    pub r: &'a Tensor,
    pub w: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub a: &'a Tensor,
    pub b: &'a Tensor,
}

/// Advances `state` (f32 `(batch, heads, head_size, head_size)`, value rows by key columns) over the sequence in
/// place and returns the output `(batch, seq, heads, head_size)`: per head `S = S diag(w) + v k^T + (S a) b^T`,
/// `y = S r`.
pub fn rwkv7_wkv(wkv: Wkv7<'_>, state: &Tensor) -> Result<Tensor> {
    if wkv.r.device().is_cuda() {
        #[cfg(feature = "cuda")]
        return crate::cuda::rwkv::rwkv7_wkv_cuda(wkv, state);
    }
    rwkv7_wkv_cpu(wkv, state)
}

fn rwkv7_wkv_cpu(wkv: Wkv7<'_>, state: &Tensor) -> Result<Tensor> {
    let (batch, seq, heads, n) = wkv.r.dims4()?;
    let read =
        |t: &Tensor| -> Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };
    let [r, w, k, v, a, b] = [wkv.r, wkv.w, wkv.k, wkv.v, wkv.a, wkv.b].map(read);
    let (r, w, k, v, a, b) = (r?, w?, k?, v?, a?, b?);
    let mut s = read(state)?;
    let mut y = vec![0f32; batch * seq * heads * n];
    let (states, out) = (
        super::ssm_cpu::SharedMut(s.as_mut_ptr()),
        super::ssm_cpu::SharedMut(y.as_mut_ptr()),
    );
    let channels = heads * n;
    inference_tensor::utils::barrier_pool().execute_chunked(batch * heads, |items| {
        for item in items {
            let (bi, h) = (item / heads, item % heads);
            // SAFETY: each (batch, head) owns its state block and its output columns
            let s = unsafe { states.slice(item * n * n, n * n) };
            for t in 0..seq {
                let at = (bi * seq + t) * channels + h * n;
                let (rt, wt, kt, at_, bt) = (
                    &r[at..at + n],
                    &w[at..at + n],
                    &k[at..at + n],
                    &a[at..at + n],
                    &b[at..at + n],
                );
                // SAFETY: as above
                let yt = unsafe { out.slice(at, n) };
                for i in 0..n {
                    let row = &mut s[i * n..(i + 1) * n];
                    let sa = row.iter().zip(at_).map(|(s, a)| s * a).sum::<f32>();
                    let vi = v[at + i];
                    let mut out = 0f32;
                    for j in 0..n {
                        row[j] = row[j] * wt[j] + vi * kt[j] + sa * bt[j];
                        out += row[j] * rt[j];
                    }
                    yt[i] = out;
                }
            }
        }
    });
    state.inplace_op1(&WriteState(&s))?;
    Tensor::from_vec(y, (batch, seq, heads, n), wkv.r.device())
}

/// The token shift's lerps over `x` `(batch, seq, hidden)`: for each of `mixes` `(n, hidden)`,
/// `x + (prev - x) * (mix + extra)` with `prev` the previous token's `x` and `last` `(batch, hidden)` before the first,
/// `extra` (RWKV-6's data-dependent part, `(n, batch, seq, hidden)`) zero when absent. Returns the mixes `(n, batch,
/// seq, hidden)` and the sequence's last token `(batch, hidden)`.
pub fn rwkv_token_shift(
    x: &Tensor,
    last: &Tensor,
    mixes: &Tensor,
    extra: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    if x.device().is_cuda() {
        #[cfg(feature = "cuda")]
        return crate::cuda::rwkv::rwkv_token_shift_cuda(x, last, mixes, extra);
    }
    let s = x.dim(1)?;
    let last_in = last.unsqueeze(1)?.to_dtype(x.dtype())?;
    let prev = if s == 1 {
        last_in
    } else {
        Tensor::cat(&[last_in, x.narrow(1, 0, s - 1)?], 1)?
    };
    let shifted = (prev - x)?;
    let mixes = mixes.to_dtype(x.dtype())?;
    let mixed = (0..mixes.dim(0)?)
        .map(|m| {
            let mix = match extra {
                Some(extra) => extra
                    .get(m)?
                    .to_dtype(x.dtype())?
                    .broadcast_add(&mixes.get(m)?)?,
                None => mixes.get(m)?,
            };
            x + shifted.broadcast_mul(&mix)?
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        Tensor::stack(&mixed, 0)?,
        x.narrow(1, s - 1, 1)?.squeeze(1)?,
    ))
}

/// One RWKV-6 time-mix's inputs between its projections and the output projection, `(batch, seq, hidden)` in the
/// activations' dtype: the receptance, `k`, `v`, the decay before its exps (`time_decay` plus its LoRA) and the gate
/// before its SiLU; `u` (`time_first`) and the group norm's weight and bias are `(hidden,)`.
#[derive(Clone, Copy)]
pub struct TimeMix6<'a> {
    pub r: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub w: &'a Tensor,
    pub g: &'a Tensor,
    pub u: &'a Tensor,
    pub norm_w: &'a Tensor,
    pub norm_b: &'a Tensor,
    pub norm_eps: f64,
    pub heads: usize,
    pub head_dim: usize,
}

/// RWKV-6's time-mix between its projections and the output projection: `w = exp(-exp(w))`, per head
/// `y = r (S + diag(u) k^T v)` and `S = diag(w) S + k^T v` over `state` (f32 `(batch, heads, key, value)`, advanced
/// in place), the group norm per head, times `silu(g)`. On CUDA one kernel.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn rwkv6_time_mix(tm: TimeMix6<'_>, state: &Tensor) -> Result<Tensor> {
    if tm.r.device().is_cuda() {
        #[cfg(feature = "cuda")]
        return crate::cuda::rwkv::rwkv6_time_mix_cuda(tm, state);
    }
    let (batch, seq, h) = tm.r.dims3()?;
    let (heads, n) = (tm.heads, tm.head_dim);
    let read =
        |t: &Tensor| -> Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };
    let [r, k, v, w, g, u, norm_w, norm_b] =
        [tm.r, tm.k, tm.v, tm.w, tm.g, tm.u, tm.norm_w, tm.norm_b].map(read);
    let (r, k, v, w, g) = (r?, k?, v?, w?, g?);
    let (u, norm_w, norm_b) = (u?, norm_w?, norm_b?);
    let mut s = read(state)?;
    let mut y = vec![0f32; batch * seq * h];
    let (states, out) = (
        super::ssm_cpu::SharedMut(s.as_mut_ptr()),
        super::ssm_cpu::SharedMut(y.as_mut_ptr()),
    );
    inference_tensor::utils::barrier_pool().execute_chunked(batch * heads, |items| {
        for item in items {
            let (bi, head) = (item / heads, item % heads);
            // SAFETY: each (batch, head) owns its state block and its output columns
            let s = unsafe { states.slice(item * n * n, n * n) };
            let c0 = head * n;
            for t in 0..seq {
                let at = (bi * seq + t) * h + c0;
                // SAFETY: as above
                let yt = unsafe { out.slice(at, n) };
                let decay = (0..n).map(|i| (-w[at + i].exp()).exp()).collect::<Vec<_>>();
                yt.fill(0.0);
                for i in 0..n {
                    let (ri, ki, ui) = (r[at + i], k[at + i], u[c0 + i]);
                    let row = &mut s[i * n..(i + 1) * n];
                    for j in 0..n {
                        let kv = ki * v[at + j];
                        yt[j] += ri * (row[j] + ui * kv);
                        row[j] = row[j] * decay[i] + kv;
                    }
                }
                let mean = yt.iter().sum::<f32>() / n as f32;
                let var = yt.iter().map(|y| (y - mean).powi(2)).sum::<f32>() / n as f32;
                let inv = 1.0 / (var + tm.norm_eps as f32).sqrt();
                for j in 0..n {
                    let c = c0 + j;
                    let gate = g[at + j];
                    let normed = (yt[j] - mean) * inv * norm_w[c] + norm_b[c];
                    yt[j] = normed * gate / (1.0 + (-gate).exp());
                }
            }
        }
    });
    state.inplace_op1(&WriteState(&s))?;
    Tensor::from_vec(y, (batch, seq, h), tm.r.device())?.to_dtype(tm.r.dtype())
}

/// One time-mix's inputs between its projections and `o_proj`, `(batch, seq, hidden)` in the activations' dtype:
/// the receptance, the decay LoRA's raw output, `k`, `v`, the rate LoRA's raw output, the gate, and past the first
/// layer the first layer's `v` with the value-mix LoRA's raw output. The vectors are `(hidden,)` (`r_k` per head).
#[derive(Clone, Copy)]
pub struct TimeMix7<'a> {
    pub r: &'a Tensor,
    pub w: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub a: &'a Tensor,
    pub g: &'a Tensor,
    pub v_mix: Option<(&'a Tensor, &'a Tensor)>,
    pub k_k: &'a Tensor,
    pub k_a: &'a Tensor,
    pub r_k: &'a Tensor,
    pub norm_w: &'a Tensor,
    pub norm_b: &'a Tensor,
    pub norm_eps: f64,
    pub heads: usize,
    pub head_dim: usize,
}

// fla's `w = -exp(-0.5) * sigmoid(..)`, the decay `exp(w)` in (0.545, 1)
const DECAY_SCALE: f64 = -0.606_530_659_712_633_4;
// F.normalize's floor on a head's key norm
const KK_NORM_EPS: f64 = 1e-12;

/// RWKV-7's time-mix between its projections and `o_proj`: the decay, rate and value mix, `kk` and `k`, the WKV
/// recurrence over `state` (advanced in place), the group norm per head, the bonus and the gate. Returns the input to
/// `o_proj` in the activations' dtype; on CUDA one kernel.
pub fn rwkv7_time_mix(tm: TimeMix7<'_>, state: &Tensor) -> Result<Tensor> {
    if tm.r.device().is_cuda() {
        #[cfg(feature = "cuda")]
        return crate::cuda::rwkv::rwkv7_time_mix_cuda(tm, state);
    }
    use inference_tensor::D;
    use inference_tensor::nn::ops::sigmoid;
    let (b, s, h) = tm.r.dims3()?;
    let f32 = |t: &Tensor| t.to_dtype(DType::F32);
    let (r, k, g) = (f32(tm.r)?, f32(tm.k)?, f32(tm.g)?);
    let w = (sigmoid(&f32(tm.w)?)? * DECAY_SCALE)?.exp()?;
    let a = sigmoid(&f32(tm.a)?)?;
    let mut v = f32(tm.v)?;
    if let Some((first, mix)) = tm.v_mix {
        v = (&v + (f32(first)? - &v)?.mul(&sigmoid(&f32(mix)?)?)?)?;
    }
    let vector = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all();
    let heads = |t: &Tensor| t.reshape((b, s, tm.heads, tm.head_dim));
    let kk = heads(&k.broadcast_mul(&vector(tm.k_k)?)?)?;
    let norm = kk
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .maximum(KK_NORM_EPS)?;
    let kk = kk.broadcast_div(&norm)?;
    // k (1 + (a - 1) k_a)
    let k = (&k + (&a - 1.0)?.broadcast_mul(&vector(tm.k_a)?)?.mul(&k)?)?;
    let (r, w, k, v, a) = (heads(&r)?, heads(&w)?, heads(&k)?, heads(&v)?, heads(&a)?);
    let y = rwkv7_wkv(
        Wkv7 {
            r: &r.contiguous()?,
            w: &w.contiguous()?,
            k: &k.contiguous()?,
            v: &v.contiguous()?,
            a: &kk.neg()?.contiguous()?,
            b: &kk.mul(&a)?.contiguous()?,
        },
        state,
    )?;
    let mean = y.mean_keepdim(D::Minus1)?;
    let centered = y.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let y = centered
        .broadcast_div(&(var + tm.norm_eps)?.sqrt()?)?
        .reshape((b, s, h))?
        .broadcast_mul(&vector(tm.norm_w)?)?
        .broadcast_add(&vector(tm.norm_b)?)?;
    // the bonus: each head's (r . k r_k) times its value
    let r_k = vector(tm.r_k)?.reshape((tm.heads, tm.head_dim))?;
    let rk = r.mul(&k)?.broadcast_mul(&r_k)?.sum_keepdim(D::Minus1)?;
    let y = (y + rk.broadcast_mul(&v)?.reshape((b, s, h))?)?.mul(&g)?;
    y.to_dtype(tm.r.dtype())
}

// copies the advanced state back into the caller's tensor
struct WriteState<'a>(&'a [f32]);

impl inference_tensor::InplaceOp1 for WriteState<'_> {
    fn name(&self) -> &'static str {
        "rwkv7-state-write"
    }

    fn cpu_fwd(
        &self,
        storage: &mut inference_tensor::CpuStorage,
        layout: &inference_tensor::Layout,
    ) -> Result<()> {
        let inference_tensor::CpuStorage::F32(data) = storage else {
            inference_tensor::bail!("rwkv7 state must be f32");
        };
        if !layout.is_contiguous() {
            inference_tensor::bail!("rwkv7 state must be contiguous");
        }
        data[layout.start_offset()..][..self.0.len()].copy_from_slice(self.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::Device;

    // the recurrence written out for one head, against the op over two heads and two sequences
    #[test]
    fn the_cpu_recurrence_follows_the_update() -> Result<()> {
        const B: usize = 2;
        const T: usize = 5;
        const H: usize = 2;
        const N: usize = 4;
        let rand = |shape: &[usize]| Tensor::randn(0f32, 0.5, shape, &Device::Cpu);
        let inputs = [0; 6].map(|_| rand(&[B, T, H, N]).unwrap());
        let w = (inputs[1].abs()?.neg()?.exp())?;
        let state = rand(&[B, H, N, N])?;
        let initial = state.flatten_all()?.to_vec1::<f32>()?;
        let wkv = Wkv7 {
            r: &inputs[0],
            w: &w,
            k: &inputs[2],
            v: &inputs[3],
            a: &inputs[4],
            b: &inputs[5],
        };
        let y = rwkv7_wkv_cpu(wkv, &state)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let flat = [
            &inputs[0], &w, &inputs[2], &inputs[3], &inputs[4], &inputs[5],
        ]
        .map(|t| t.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let [r, w, k, v, a, b] = &flat;
        let (bi, h) = (1, 1);
        let mut s = initial[(bi * H + h) * N * N..][..N * N].to_vec();
        for t in 0..T {
            let at = (bi * T + t) * H * N + h * N;
            let sa = (0..N)
                .map(|i| (0..N).map(|j| a[at + j] * s[i * N + j]).sum::<f32>())
                .collect::<Vec<_>>();
            for i in 0..N {
                let mut want = 0f32;
                for j in 0..N {
                    s[i * N + j] =
                        s[i * N + j] * w[at + j] + v[at + i] * k[at + j] + sa[i] * b[at + j];
                    want += s[i * N + j] * r[at + j];
                }
                assert!(
                    (y[at + i] - want).abs() < 1e-5,
                    "step {t} row {i}: {} vs {want}",
                    y[at + i]
                );
            }
        }
        let advanced = state.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            advanced[(bi * H + h) * N * N..][..N * N]
                .iter()
                .zip(&s)
                .all(|(x, y)| (x - y).abs() < 1e-5)
        );
        Ok(())
    }
}
