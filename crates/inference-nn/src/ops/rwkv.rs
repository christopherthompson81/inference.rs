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
