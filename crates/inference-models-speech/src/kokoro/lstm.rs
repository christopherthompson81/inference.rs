use inference_tensor::nn::{VarBuilder, ops};
use inference_tensor::{DType, IndexOp, Result, Tensor};

/// One direction of a single-layer PyTorch LSTM (gates i, f, g, o).
#[derive(Debug, Clone)]
struct Direction {
    w_ih_t: Tensor,
    w_hh_t: Tensor,
    bias: Tensor,
    // the recurrent weights (hidden x 4 hidden) on the host, for the CPU's step loop
    w_hh_host: Option<Vec<f32>>,
}

impl Direction {
    fn new(in_dim: usize, hidden: usize, suffix: &str, vb: &VarBuilder) -> Result<Self> {
        let gates = 4 * hidden;
        let w_ih = vb.get((gates, in_dim), &format!("weight_ih_l0{suffix}"))?;
        let w_hh = vb.get((gates, hidden), &format!("weight_hh_l0{suffix}"))?;
        let bias = (vb.get(gates, &format!("bias_ih_l0{suffix}"))?
            + vb.get(gates, &format!("bias_hh_l0{suffix}"))?)?;
        let w_hh_t = w_hh.t()?.contiguous()?;
        let w_hh_host = (vb.device().is_cpu() && w_hh_t.dtype() == DType::F32)
            .then(|| w_hh_t.flatten_all()?.to_vec1::<f32>())
            .transpose()?;
        Ok(Self {
            w_ih_t: w_ih.t()?.contiguous()?,
            w_hh_t,
            bias,
            w_hh_host,
        })
    }

    /// `xs` is (1, T, in); returns (1, T, hidden), run backwards in time when `reverse`.
    fn run(&self, xs: &Tensor, reverse: bool) -> Result<Tensor> {
        let hidden = self.w_hh_t.dim(0)?;
        let steps = xs.dim(1)?;
        let gx = xs.i(0)?.matmul(&self.w_ih_t)?.broadcast_add(&self.bias)?;
        if let Some(w_hh) = self.w_hh_host.as_deref()
            && xs.device().is_cpu()
        {
            let out = host_steps(
                &gx.flatten_all()?.to_vec1::<f32>()?,
                w_hh,
                hidden,
                steps,
                reverse,
            );
            return Tensor::from_vec(out, (1, steps, hidden), xs.device());
        }
        let mut h = Tensor::zeros((1, hidden), xs.dtype(), xs.device())?;
        let mut c = h.clone();
        let mut outs = vec![None; steps];
        for step in 0..steps {
            let t = if reverse { steps - 1 - step } else { step };
            let g = (gx.narrow(0, t, 1)? + h.matmul(&self.w_hh_t)?)?;
            let i = ops::sigmoid(&g.narrow(1, 0, hidden)?)?;
            let f = ops::sigmoid(&g.narrow(1, hidden, hidden)?)?;
            let cell = g.narrow(1, 2 * hidden, hidden)?.tanh()?;
            let o = ops::sigmoid(&g.narrow(1, 3 * hidden, hidden)?)?;
            c = ((f * &c)? + (i * cell)?)?;
            h = (o * c.tanh()?)?;
            outs[t] = Some(h.clone());
        }
        let outs = outs.into_iter().flatten().collect::<Vec<_>>();
        Tensor::cat(&outs, 0)?.unsqueeze(0)
    }
}

// One tensor op per gate per step costs more than the step itself on the CPU, so the recurrence runs as plain loops.
fn host_steps(gx: &[f32], w_hh: &[f32], hidden: usize, steps: usize, reverse: bool) -> Vec<f32> {
    let gates = 4 * hidden;
    let sigmoid = |x: f32| 1. / (1. + (-x).exp());
    let (mut h, mut c) = (vec![0f32; hidden], vec![0f32; hidden]);
    let mut out = vec![0f32; steps * hidden];
    let mut g = vec![0f32; gates];
    for step in 0..steps {
        let t = if reverse { steps - 1 - step } else { step };
        g.copy_from_slice(&gx[t * gates..(t + 1) * gates]);
        for (k, &hk) in h.iter().enumerate() {
            for (gj, wj) in g.iter_mut().zip(&w_hh[k * gates..(k + 1) * gates]) {
                *gj += hk * wj;
            }
        }
        for j in 0..hidden {
            let (i, f) = (sigmoid(g[j]), sigmoid(g[hidden + j]));
            let (cell, o) = (g[2 * hidden + j].tanh(), sigmoid(g[3 * hidden + j]));
            c[j] = f * c[j] + i * cell;
            h[j] = o * c[j].tanh();
        }
        out[t * hidden..(t + 1) * hidden].copy_from_slice(&h);
    }
    out
}

/// A bidirectional single-layer LSTM over (1, T, in), concatenating both directions to (1, T, 2 * hidden).
#[derive(Debug, Clone)]
pub struct BiLstm {
    forward: Direction,
    backward: Direction,
}

impl BiLstm {
    pub fn new(in_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            forward: Direction::new(in_dim, hidden, "", &vb)?,
            backward: Direction::new(in_dim, hidden, "_reverse", &vb)?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // only the host loops run side by side; a GPU's ops from two threads would interleave on its one stream
        let (fwd, bwd) = if xs.device().is_cpu() {
            rayon::join(
                || self.forward.run(xs, false),
                || self.backward.run(xs, true),
            )
        } else {
            (self.forward.run(xs, false), self.backward.run(xs, true))
        };
        Tensor::cat(&[fwd?, bwd?], 2)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use inference_tensor::{DType, Device, Tensor};

    use super::*;

    // The CPU's host loop against the per-step tensor ops it replaces, both directions
    #[test]
    fn host_steps_match_the_tensor_recurrence() -> Result<()> {
        let (in_dim, hidden, steps) = (5, 3, 7);
        let dev = Device::Cpu;
        let mut weights = HashMap::new();
        for suffix in ["", "_reverse"] {
            let mut put = |name: &str, shape: &[usize]| -> Result<()> {
                weights.insert(
                    format!("{name}{suffix}"),
                    Tensor::randn(0f32, 0.5, shape, &dev)?,
                );
                Ok(())
            };
            put("weight_ih_l0", &[4 * hidden, in_dim])?;
            put("weight_hh_l0", &[4 * hidden, hidden])?;
            put("bias_ih_l0", &[4 * hidden])?;
            put("bias_hh_l0", &[4 * hidden])?;
        }
        let lstm = BiLstm::new(
            in_dim,
            hidden,
            VarBuilder::from_tensors(weights, DType::F32, &dev),
        )?;
        assert!(lstm.forward.w_hh_host.is_some());
        let mut tensor_only = lstm.clone();
        tensor_only.forward.w_hh_host = None;
        tensor_only.backward.w_hh_host = None;
        let x = Tensor::randn(0f32, 1., (1, steps, in_dim), &dev)?;
        let diff = (lstm.forward(&x)? - tensor_only.forward(&x)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "{diff}");
        Ok(())
    }
}
