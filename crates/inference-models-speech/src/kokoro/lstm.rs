use inference_tensor::nn::{VarBuilder, ops};
use inference_tensor::{IndexOp, Result, Tensor};

/// One direction of a single-layer PyTorch LSTM (gates i, f, g, o).
#[derive(Debug, Clone)]
struct Direction {
    w_ih_t: Tensor,
    w_hh_t: Tensor,
    bias: Tensor,
}

impl Direction {
    fn new(in_dim: usize, hidden: usize, suffix: &str, vb: &VarBuilder) -> Result<Self> {
        let gates = 4 * hidden;
        let w_ih = vb.get((gates, in_dim), &format!("weight_ih_l0{suffix}"))?;
        let w_hh = vb.get((gates, hidden), &format!("weight_hh_l0{suffix}"))?;
        let bias = (vb.get(gates, &format!("bias_ih_l0{suffix}"))?
            + vb.get(gates, &format!("bias_hh_l0{suffix}"))?)?;
        Ok(Self {
            w_ih_t: w_ih.t()?.contiguous()?,
            w_hh_t: w_hh.t()?.contiguous()?,
            bias,
        })
    }

    /// `xs` is (1, T, in); returns (1, T, hidden), run backwards in time when `reverse`.
    fn run(&self, xs: &Tensor, reverse: bool) -> Result<Tensor> {
        let hidden = self.w_hh_t.dim(0)?;
        let steps = xs.dim(1)?;
        let gx = xs.i(0)?.matmul(&self.w_ih_t)?.broadcast_add(&self.bias)?;
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
        Tensor::cat(
            &[self.forward.run(xs, false)?, self.backward.run(xs, true)?],
            2,
        )
    }
}
