use inference_tensor::nn::{Embedding, Linear, Module, VarBuilder, embedding, linear, ops};
use inference_tensor::{D, IndexOp, Result, Tensor};

use super::config::{HeadKind, ParakeetConfig};

/// One emitted token: its id and the encoder frames it spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Emission {
    pub token: u32,
    pub frame: usize,
    pub frames: usize,
}

// one layer of PyTorch's LSTM (gates i, f, g, o), stepped a token at a time
struct LstmLayer {
    w_ih_t: Tensor,
    w_hh_t: Tensor,
    bias: Tensor,
}

impl LstmLayer {
    fn new(hidden: usize, layer: usize, vb: &VarBuilder) -> Result<Self> {
        let gates = 4 * hidden;
        let get = |name: &str, shape: (usize, usize)| vb.get(shape, &format!("{name}_l{layer}"));
        let bias = (vb.get(gates, &format!("bias_ih_l{layer}"))?
            + vb.get(gates, &format!("bias_hh_l{layer}"))?)?;
        Ok(Self {
            w_ih_t: get("weight_ih", (gates, hidden))?.t()?.contiguous()?,
            w_hh_t: get("weight_hh", (gates, hidden))?.t()?.contiguous()?,
            bias,
        })
    }

    /// `x`, `h` and `c` are `(1, hidden)`; returns the new `(h, c)`.
    fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> Result<(Tensor, Tensor)> {
        let hidden = h.dim(1)?;
        let g = (x.matmul(&self.w_ih_t)? + h.matmul(&self.w_hh_t)?)?.broadcast_add(&self.bias)?;
        let i = ops::sigmoid(&g.narrow(1, 0, hidden)?)?;
        let f = ops::sigmoid(&g.narrow(1, hidden, hidden)?)?;
        let cell = g.narrow(1, 2 * hidden, hidden)?.tanh()?;
        let o = ops::sigmoid(&g.narrow(1, 3 * hidden, hidden)?)?;
        let c = ((f * c)? + (i * cell)?)?;
        let h = (o * c.tanh()?)?;
        Ok((h, c))
    }
}

/// The transducer's prediction network: embedding, stacked LSTM, projection; state carried across tokens.
pub(super) struct Predictor {
    embedding: Embedding,
    layers: Vec<LstmLayer>,
    projector: Linear,
}

struct PredictorState {
    h: Vec<Tensor>,
    c: Vec<Tensor>,
    output: Tensor,
}

impl Predictor {
    fn new(cfg: &ParakeetConfig, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let lstm = vb.pp("lstm");
        let layers = (0..cfg.num_decoder_layers.unwrap_or(1))
            .map(|l| LstmLayer::new(hidden, l, &lstm))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embedding: embedding(cfg.vocab_size, hidden, vb.pp("embedding"))?,
            layers,
            projector: linear(hidden, hidden, vb.pp("decoder_projector"))?,
        })
    }

    fn start(&self, token: u32, like: &Tensor) -> Result<PredictorState> {
        let zero = Tensor::zeros(
            (1, self.projector.weight().dim(0)?),
            like.dtype(),
            like.device(),
        )?;
        let state = PredictorState {
            h: vec![zero.clone(); self.layers.len()],
            c: vec![zero.clone(); self.layers.len()],
            output: zero,
        };
        self.advance(token, state)
    }

    fn advance(&self, token: u32, state: PredictorState) -> Result<PredictorState> {
        let ids = Tensor::new(&[token], state.output.device())?;
        let mut x = self.embedding.forward(&ids)?;
        let (mut h, mut c) = (Vec::new(), Vec::new());
        for (layer, (h0, c0)) in self.layers.iter().zip(state.h.iter().zip(&state.c)) {
            let (h1, c1) = layer.step(&x, h0, c0)?;
            x = h1.clone();
            h.push(h1);
            c.push(c1);
        }
        Ok(PredictorState {
            h,
            c,
            output: self.projector.forward(&x)?,
        })
    }
}

/// Turns the encoder's frames into token emissions: a CTC head, or a transducer's predictor and joint network.
pub(super) enum Head {
    Ctc {
        projection: Linear,
        blank: u32,
    },
    Transducer {
        encoder_projector: Linear,
        predictor: Predictor,
        joint: Linear,
        vocab: usize,
        blank: u32,
        durations: Vec<usize>,
        max_symbols_per_step: usize,
    },
}

impl Head {
    /// `vb` is the checkpoint's root; `hidden` the encoder's width.
    pub(super) fn new(
        cfg: &ParakeetConfig,
        kind: HeadKind,
        hidden: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if kind == HeadKind::Ctc {
            let w = vb
                .pp("ctc_head")
                .get((cfg.vocab_size, hidden, 1), "weight")?
                .squeeze(2)?;
            let b = vb.pp("ctc_head").get(cfg.vocab_size, "bias")?;
            let blank = cfg.pad_token_id.unwrap_or(cfg.vocab_size as u32 - 1);
            return Ok(Self::Ctc {
                projection: Linear::new(w, Some(b)),
                blank,
            });
        }
        let dec = cfg.decoder_hidden_size.unwrap_or(hidden);
        let durations = if kind == HeadKind::Tdt {
            cfg.durations.clone()
        } else {
            Vec::new()
        };
        Ok(Self::Transducer {
            encoder_projector: linear(hidden, dec, vb.pp("encoder_projector"))?,
            predictor: Predictor::new(cfg, dec, vb.pp("decoder"))?,
            joint: linear(
                dec,
                cfg.vocab_size + durations.len(),
                vb.pp("joint").pp("head"),
            )?,
            vocab: cfg.vocab_size,
            blank: cfg.blank_token_id.unwrap_or(cfg.vocab_size as u32 - 1),
            durations,
            max_symbols_per_step: cfg.max_symbols_per_step,
        })
    }

    /// Greedy decoding of `(1, frames, hidden)` encoder output into emissions in frame order.
    pub(super) fn decode(&self, encoded: &Tensor) -> Result<Vec<Emission>> {
        match self {
            Self::Ctc { projection, blank } => ctc_greedy(projection, *blank, encoded),
            Self::Transducer {
                encoder_projector,
                predictor,
                joint,
                vocab,
                blank,
                durations,
                max_symbols_per_step,
            } => {
                let frames = encoder_projector.forward(encoded)?.squeeze(0)?;
                let search = Transducer {
                    predictor,
                    joint,
                    vocab: *vocab,
                    blank: *blank,
                    durations,
                    max_symbols_per_step: *max_symbols_per_step,
                };
                search.greedy(&frames)
            }
        }
    }
}

fn ctc_greedy(projection: &Linear, blank: u32, encoded: &Tensor) -> Result<Vec<Emission>> {
    let best = projection
        .forward(encoded)?
        .squeeze(0)?
        .argmax(D::Minus1)?
        .to_vec1::<u32>()?;
    let mut out: Vec<Emission> = Vec::new();
    let mut previous = None;
    for (frame, &token) in best.iter().enumerate() {
        if token != blank && previous != Some(token) {
            out.push(Emission {
                token,
                frame,
                frames: 1,
            });
        }
        previous = Some(token);
    }
    Ok(out)
}

struct Transducer<'a> {
    predictor: &'a Predictor,
    joint: &'a Linear,
    vocab: usize,
    blank: u32,
    durations: &'a [usize],
    max_symbols_per_step: usize,
}

impl Transducer<'_> {
    // RNN-T advances a frame on blank; TDT by the predicted duration (a blank's 0 taken as 1). Either way, too many
    // symbols at one frame force a step, as NeMo's greedy search does
    fn greedy(&self, frames: &Tensor) -> Result<Vec<Emission>> {
        let total = frames.dim(0)?;
        let mut state = self.predictor.start(self.blank, frames)?;
        let mut out = Vec::new();
        let (mut frame, mut symbols) = (0usize, 0usize);
        while frame < total {
            let logits = self
                .joint
                .forward(
                    &frames
                        .i(frame..frame + 1)?
                        .broadcast_add(&state.output)?
                        .relu()?,
                )?
                .squeeze(0)?;
            let token = logits
                .narrow(0, 0, self.vocab)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            let mut step = if self.durations.is_empty() {
                usize::from(token == self.blank)
            } else {
                let d = logits
                    .narrow(0, self.vocab, self.durations.len())?
                    .argmax(0)?
                    .to_scalar::<u32>()?;
                self.durations[d as usize]
            };
            if token == self.blank {
                step = step.max(1);
            } else {
                // a TDT token spans its duration; an RNN-T token, which predicts none, one frame
                let frames = if self.durations.is_empty() { 1 } else { step };
                out.push(Emission {
                    token,
                    frame,
                    frames,
                });
                state = self.predictor.advance(token, state)?;
                symbols += 1;
                if step == 0 && symbols >= self.max_symbols_per_step {
                    step = 1;
                }
            }
            if step > 0 {
                symbols = 0;
            }
            frame += step;
        }
        Ok(out)
    }
}
