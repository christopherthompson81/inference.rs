//! Kimi Delta Attention's recurrence (a gated delta rule with a decay per key channel) and its gated output norm.

use inference_tensor::{DType, Result, Tensor};

use super::rwkv::WriteState;
use super::ssm_cpu::SharedMut;

// fla's L2 norm adds this under the square root
const L2_EPS: f32 = 1e-6;
// softplus is the identity past this, as torch and fla take it
const SOFTPLUS_THRESHOLD: f32 = 20.0;

/// One KDA layer's inputs in the activations' dtype: `q`, `k`, `v` after their convs and the decay LoRA's output `f`
/// before its bias, all `(batch, seq, heads * head_dim)`, the output gate before its sigmoid (the same shape) and
/// `b` before its sigmoid `(batch, seq, heads)`; `dt_bias` `(heads * head_dim)`, `a_log` `(heads)` and the output
/// norm's weight `(head_dim)`. `slots` names each sequence's state row (the hybrid pool's checkout), else row = seq.
#[derive(Clone, Copy)]
pub struct Kda<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub f: &'a Tensor,
    pub gate: &'a Tensor,
    pub b: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub a_log: &'a Tensor,
    pub norm_w: &'a Tensor,
    pub norm_eps: f64,
    pub heads: usize,
    pub head_dim: usize,
    pub slots: Option<&'a Tensor>,
}

/// KDA over `state` (f32 `(rows, heads, key, value)`, advanced in place): per head q and k L2-normed, q scaled by
/// `head_dim^-0.5`, the decay `exp(-exp(a_log) softplus(f + dt_bias))` per key channel, `beta = sigmoid(b)`, then
/// `S = diag(w) S`, `S += k (beta (v - k^T S))`, `y = q^T S`; returns `rmsnorm(y) norm_w sigmoid(gate)` per head in
/// the activations' dtype. On CUDA one kernel.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn kda(kda: Kda<'_>, state: &Tensor) -> Result<Tensor> {
    if kda.q.device().is_cuda() {
        #[cfg(feature = "cuda")]
        return crate::cuda::kda::kda_cuda(kda, state);
    }
    let (batch, seq, h) = kda.q.dims3()?;
    let (heads, n) = (kda.heads, kda.head_dim);
    let read =
        |t: &Tensor| -> Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };
    let [q, k, v, f, gate, b, dt_bias, a_log, norm_w] = [
        kda.q,
        kda.k,
        kda.v,
        kda.f,
        kda.gate,
        kda.b,
        kda.dt_bias,
        kda.a_log,
        kda.norm_w,
    ]
    .map(read);
    let (q, k, v, f, gate, b) = (q?, k?, v?, f?, gate?, b?);
    let (dt_bias, a_log, norm_w) = (dt_bias?, a_log?, norm_w?);
    let rows = match kda.slots {
        Some(slots) => slots
            .to_vec1::<u32>()?
            .into_iter()
            .map(|r| r as usize)
            .collect(),
        None => (0..batch).collect::<Vec<_>>(),
    };
    let mut s = read(state)?;
    let mut y = vec![0f32; batch * seq * h];
    let (states, out) = (SharedMut(s.as_mut_ptr()), SharedMut(y.as_mut_ptr()));
    let scale = (n as f32).powf(-0.5);
    let eps = kda.norm_eps as f32;
    inference_tensor::utils::barrier_pool().execute_chunked(batch * heads, |items| {
        for item in items {
            let (bi, head) = (item / heads, item % heads);
            // SAFETY: each (sequence, head) owns its state block and its output columns
            let s = unsafe { states.slice((rows[bi] * heads + head) * n * n, n * n) };
            let c0 = head * n;
            let rate = a_log[head].exp();
            for t in 0..seq {
                let token = bi * seq + t;
                let at = token * h + c0;
                let norm = |x: &[f32]| (x.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
                let (qn, kn) = (norm(&q[at..at + n]), norm(&k[at..at + n]));
                let beta = 1.0 / (1.0 + (-b[token * heads + head]).exp());
                // SAFETY: as above
                let yt = unsafe { out.slice(at, n) };
                yt.fill(0.0);
                let mut kv = vec![0f32; n];
                for i in 0..n {
                    let x = f[at + i] + dt_bias[c0 + i];
                    let softplus = if x > SOFTPLUS_THRESHOLD {
                        x
                    } else {
                        x.exp().ln_1p()
                    };
                    let w = (-rate * softplus).exp();
                    let ki = k[at + i] / kn;
                    let row = &mut s[i * n..(i + 1) * n];
                    for (j, sij) in row.iter_mut().enumerate() {
                        *sij *= w;
                        kv[j] += ki * *sij;
                    }
                }
                for i in 0..n {
                    let (qi, ki) = (q[at + i] / qn * scale, k[at + i] / kn);
                    let row = &mut s[i * n..(i + 1) * n];
                    for j in 0..n {
                        row[j] += ki * beta * (v[at + j] - kv[j]);
                        yt[j] += qi * row[j];
                    }
                }
                let inv = 1.0 / (yt.iter().map(|y| y * y).sum::<f32>() / n as f32 + eps).sqrt();
                for j in 0..n {
                    yt[j] *= inv * norm_w[j] / (1.0 + (-gate[at + j]).exp());
                }
            }
        }
    });
    state.inplace_op1(&WriteState(&s))?;
    Tensor::from_vec(y, (batch, seq, h), kda.q.device())?.to_dtype(kda.q.dtype())
}
