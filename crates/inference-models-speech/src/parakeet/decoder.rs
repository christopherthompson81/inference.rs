use std::collections::HashMap;

use inference_tensor::nn::{Embedding, Linear, Module, VarBuilder, embedding, linear, ops};
use inference_tensor::{D, Error, IndexOp, Result, Tensor};

use super::config::{HeadKind, ParakeetConfig};

// a label NeMo marks a finished slot's carried-over candidate with
const NO_LABEL: i64 = -1;
// NeMo's `hash_text`, a 64-bit linear congruential step per label
const HASH_MULTIPLIER: i64 = 6364136223846793005;
const HASH_INCREMENT: i64 = 1;

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

#[derive(Clone)]
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

    // `tokens.len()` rows, each fed its token from the zero state
    fn start(&self, tokens: &[u32], like: &Tensor) -> Result<PredictorState> {
        let zero = Tensor::zeros(
            (tokens.len(), self.projector.weight().dim(0)?),
            like.dtype(),
            like.device(),
        )?;
        let state = PredictorState {
            h: vec![zero.clone(); self.layers.len()],
            c: vec![zero.clone(); self.layers.len()],
            output: zero,
        };
        self.advance(tokens, state)
    }

    // each row of `state` fed its row's token
    fn advance(&self, tokens: &[u32], state: PredictorState) -> Result<PredictorState> {
        let ids = Tensor::new(tokens, state.output.device())?;
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

    /// Decoding of `(1, frames, hidden)` encoder output into emissions in frame order: greedy, or a transducer
    /// beam search over `beam` hypotheses (NeMo's MALSD for TDT, its default search for RNN-T).
    pub(super) fn decode(&self, encoded: &Tensor, beam: Option<usize>) -> Result<Vec<Emission>> {
        match self {
            Self::Ctc { .. } if beam.is_some_and(|b| b > 1) => Err(Error::Msg(
                "beam search decodes transducer (RNN-T, TDT) heads; this one is CTC".into(),
            )),
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
                match beam {
                    Some(beam) if beam > 1 => search.beam(&frames, beam),
                    _ => search.greedy(&frames),
                }
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
        let mut state = self.predictor.start(&[self.blank], frames)?;
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
                state = self.predictor.advance(&[token], state)?;
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

// a beam hypothesis; `state` is the predictor's before `pending`, its last token, which `settle` feeds
#[derive(Clone)]
struct Hypothesis {
    score: f64,
    tokens: Vec<u32>,
    emissions: Vec<Emission>,
    state: PredictorState,
    pending: Option<u32>,
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln();
    logits.iter().map(|l| l - max - sum).collect()
}

// indices of the `k` largest, largest first; ties to the lower index
fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    order.truncate(k);
    order
}

fn best(hyps: &[Hypothesis]) -> usize {
    (0..hyps.len())
        .max_by(|&a, &b| hyps[a].score.total_cmp(&hyps[b].score))
        .expect("a non-empty beam")
}

impl Transducer<'_> {
    // the predictor output after `hyp`'s tokens, from the cache NeMo keeps per token sequence
    fn settle(
        &self,
        hyp: &mut Hypothesis,
        cache: &mut HashMap<Vec<u32>, PredictorState>,
    ) -> Result<()> {
        let Some(token) = hyp.pending.take() else {
            return Ok(());
        };
        hyp.state = match cache.get(&hyp.tokens) {
            Some(state) => state.clone(),
            None => {
                let state = self.predictor.advance(&[token], hyp.state.clone())?;
                cache.insert(hyp.tokens.clone(), state.clone());
                state
            }
        };
        Ok(())
    }

    // the joint's logits for one frame and predictor output, on the host
    fn logits(&self, frames: &Tensor, frame: usize, state: &PredictorState) -> Result<Vec<f32>> {
        self.joint
            .forward(
                &frames
                    .i(frame..frame + 1)?
                    .broadcast_add(&state.output)?
                    .relu()?,
            )?
            .squeeze(0)?
            .to_dtype(inference_tensor::DType::F32)?
            .to_vec1::<f32>()
    }

    // NeMo's best hypothesis: highest score per symbol, the start blank counted
    fn finish(mut hyps: Vec<Hypothesis>) -> Vec<Emission> {
        let normed = |h: &Hypothesis| h.score / (h.tokens.len() + 1) as f64;
        hyps.sort_by(|a, b| normed(b).total_cmp(&normed(a)));
        hyps.into_iter().next().expect("a non-empty beam").emissions
    }

    // RNN-T takes NeMo's default beam search, TDT its MALSD: NeMo's default TDT search drops speech on long audio
    fn beam(&self, frames: &Tensor, beam_size: usize) -> Result<Vec<Emission>> {
        if !self.durations.is_empty() {
            return self.malsd(frames, beam_size);
        }
        let start = Hypothesis {
            score: 0.0,
            tokens: Vec::new(),
            emissions: Vec::new(),
            state: self.predictor.start(&[self.blank], frames)?,
            pending: None,
        };
        self.rnnt_beam(frames, beam_size, start)
    }

    // NeMo's `BeamRNNTInfer.default_beam_search`; its scores are Python floats
    fn rnnt_beam(
        &self,
        frames: &Tensor,
        beam_size: usize,
        start: Hypothesis,
    ) -> Result<Vec<Emission>> {
        let tokens = self.vocab - 1;
        let beam = beam_size.min(tokens);
        let beam_k = beam.min(tokens - 1);
        let mut cache = HashMap::new();
        let mut kept = vec![start];
        for frame in 0..frames.dim(0)? {
            let mut hyps = std::mem::take(&mut kept);
            loop {
                let mut hyp = hyps.swap_remove(best(&hyps));
                self.settle(&mut hyp, &mut cache)?;
                let logp = log_softmax(&self.logits(frames, frame, &hyp.state)?);
                let non_blank: Vec<f32> = logp[..tokens].to_vec();
                for token in top_k(&non_blank, beam_k) {
                    let mut next = hyp.clone();
                    next.score += f64::from(logp[token]);
                    next.tokens.push(token as u32);
                    next.emissions.push(Emission {
                        token: token as u32,
                        frame,
                        frames: 1,
                    });
                    next.pending = Some(token as u32);
                    hyps.push(next);
                }
                let mut blank = hyp;
                blank.score += f64::from(logp[self.blank as usize]);
                kept.push(blank);
                let open = hyps[best(&hyps)].score;
                if kept.iter().filter(|h| h.score > open).count() >= beam {
                    kept.retain(|h| h.score > open);
                    kept.sort_by(|a, b| a.score.total_cmp(&b.score));
                    break;
                }
            }
            // hypotheses only lengthen, so no later one asks for a state shorter than every kept one's
            let shortest = kept.iter().map(|h| h.tokens.len()).min().unwrap_or(0);
            cache.retain(|tokens: &Vec<u32>, _| tokens.len() >= shortest);
        }
        Ok(Self::finish(kept))
    }

    // NeMo's `ModifiedALSDBatchedTDTComputer` without fusion models; its scores are float32
    fn malsd(&self, frames: &Tensor, beam: usize) -> Result<Vec<Emission>> {
        let Some(last_frame) = frames.dim(0)?.checked_sub(1) else {
            return Ok(Vec::new());
        };
        let blank = self.blank as usize;
        let n_durations = self.durations.len();
        let mut slots: Vec<Slot> = (0..beam)
            .map(|b| Slot {
                score: if b == 0 { 0.0 } else { f32::NEG_INFINITY },
                last_label: NO_LABEL,
                ..Slot::default()
            })
            .collect();
        let mut state = self.predictor.start(&vec![self.blank; beam], frames)?;
        while slots.iter().any(|s| s.next_frame <= last_frame) {
            let at: Vec<u32> = slots
                .iter()
                .map(|s| s.next_frame.min(last_frame) as u32)
                .collect();
            let at = Tensor::new(at.as_slice(), frames.device())?;
            let logits = self
                .joint
                .forward(
                    &frames
                        .index_select(&at, 0)?
                        .broadcast_add(&state.output)?
                        .relu()?,
                )?
                .to_dtype(inference_tensor::DType::F32)?
                .to_vec2::<f32>()?;
            // per slot: its top `beam` (label, duration) pairs, then all `beam * beam` candidates
            let mut candidates: Vec<(f32, usize, i64, usize)> = Vec::with_capacity(beam * beam);
            for (b, slot) in slots.iter().enumerate() {
                let logp = log_softmax(&logits[b][..self.vocab]);
                let durations_logp = log_softmax(&logits[b][self.vocab..]);
                let pairs: Vec<f32> = logp
                    .iter()
                    .flat_map(|l| durations_logp.iter().map(move |d| l + d))
                    .collect();
                let top = top_k(&pairs, beam);
                let active = slot.next_frame <= last_frame;
                let forced = active && slot.symbols_here >= self.max_symbols_per_step;
                for (k, &pair) in top.iter().enumerate() {
                    let (mut label, mut d) = ((pair / n_durations) as i64, pair % n_durations);
                    // a blank always advances
                    if label == blank as i64 && self.durations[d] == 0 {
                        d = self.min_advance();
                    }
                    let mut score = slot.score + pairs[pair];
                    if !active {
                        // a finished slot keeps its place at its own score, once
                        (score, label) = (
                            if k == 0 {
                                slot.score
                            } else {
                                f32::NEG_INFINITY
                            },
                            NO_LABEL,
                        );
                    } else if forced {
                        // past `max_symbols` at one frame, the slot's only move is blank
                        let best_duration = durations_logp
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        score = if k == 0 {
                            slot.score + (logp[blank] + best_duration)
                        } else {
                            f32::NEG_INFINITY
                        };
                        label = blank as i64;
                        if self.durations[d] == 0 {
                            d = self.min_advance();
                        }
                    }
                    candidates.push((score, b, label, d));
                }
            }
            let scores: Vec<f32> = candidates.iter().map(|c| c.0).collect();
            let chosen = top_k(&scores, beam);
            let previous = slots;
            slots = chosen
                .iter()
                .map(|&c| {
                    let (score, source, label, d) = candidates[c];
                    previous[source].extended(score, label, self.durations[d], blank as i64)
                })
                .collect();
            recombine(&mut slots);
            // the predictor runs each slot's new label from its source's state; a blank or finished slot keeps it
            let sources: Vec<u32> = chosen.iter().map(|&c| candidates[c].1 as u32).collect();
            let sources = Tensor::new(sources.as_slice(), frames.device())?;
            let carried = PredictorState {
                h: state
                    .h
                    .iter()
                    .map(|h| h.index_select(&sources, 0))
                    .collect::<Result<_>>()?,
                c: state
                    .c
                    .iter()
                    .map(|c| c.index_select(&sources, 0))
                    .collect::<Result<_>>()?,
                output: state.output.index_select(&sources, 0)?,
            };
            let labels: Vec<u32> = chosen
                .iter()
                .map(|&c| match candidates[c].2 {
                    l if l >= 0 => l as u32,
                    _ => self.blank,
                })
                .collect();
            let keep: Vec<u8> = labels.iter().map(|&l| u8::from(l == self.blank)).collect();
            let advanced = self.predictor.advance(&labels, carried.clone())?;
            let keep = Tensor::new(keep.as_slice(), frames.device())?.unsqueeze(1)?;
            let pick = |kept: &Tensor, new: &Tensor| {
                keep.broadcast_as(kept.shape())?.where_cond(kept, new)
            };
            state = PredictorState {
                h: carried
                    .h
                    .iter()
                    .zip(&advanced.h)
                    .map(|(k, n)| pick(k, n))
                    .collect::<Result<_>>()?,
                c: carried
                    .c
                    .iter()
                    .zip(&advanced.c)
                    .map(|(k, n)| pick(k, n))
                    .collect::<Result<_>>()?,
                output: pick(&carried.output, &advanced.output)?,
            };
        }
        // NeMo's best: highest score per emitted token, plus one
        let norm = |s: &Slot| s.score / (s.emissions.len() + 1) as f32;
        let best = (1..slots.len()).fold(0, |best, b| {
            if norm(&slots[b]) > norm(&slots[best]) {
                b
            } else {
                best
            }
        });
        Ok(slots.swap_remove(best).emissions.as_ref().clone())
    }

    // the duration a blank takes in place of zero: NeMo takes index 1, the shortest non-zero for durations 0..n
    fn min_advance(&self) -> usize {
        (0..self.durations.len())
            .filter(|&i| self.durations[i] > 0)
            .min_by_key(|&i| self.durations[i])
            .expect("a TDT head has a non-zero duration")
    }
}

// one MALSD slot, its emissions shared with the slots branched from it
#[derive(Clone, Default)]
struct Slot {
    score: f32,
    emissions: std::rc::Rc<Vec<Emission>>,
    next_frame: usize,
    symbols_here: usize,
    last_label: i64,
    hash: i64,
}

impl Slot {
    // `self` extended by `label` (a token, blank, or none for a carried-over finished slot) lasting `duration`
    fn extended(&self, score: f32, label: i64, duration: usize, blank: i64) -> Slot {
        let token = label >= 0 && label != blank;
        let advance = if label >= 0 { duration } else { 0 };
        let mut emissions = self.emissions.clone();
        if token {
            std::rc::Rc::make_mut(&mut emissions).push(Emission {
                token: label as u32,
                frame: self.next_frame,
                frames: duration,
            });
        }
        Slot {
            score,
            emissions,
            next_frame: self.next_frame + advance,
            symbols_here: if label >= 0 && advance > 0 {
                0
            } else {
                self.symbols_here + usize::from(token)
            },
            last_label: if token { label } else { self.last_label },
            hash: if token {
                self.hash
                    .wrapping_mul(HASH_MULTIPLIER)
                    .wrapping_add(HASH_INCREMENT)
                    .wrapping_add(label)
            } else {
                self.hash
            },
        }
    }

    fn same_path(&self, other: &Slot) -> bool {
        self.hash == other.hash
            && self.last_label == other.last_label
            && self.emissions.len() == other.emissions.len()
            && self.next_frame == other.next_frame
    }
}

// NeMo's `recombine_hyps_`: of slots on the same path the best keeps the log-sum-exp of their scores, the others drop
fn recombine(slots: &mut [Slot]) {
    let n = slots.len();
    let mut merged = vec![f32::NEG_INFINITY; n];
    for j in 0..n {
        let same: Vec<usize> = (0..n).filter(|&k| slots[j].same_path(&slots[k])).collect();
        // torch's argmax: the first of the equal maxima
        let keeper = same
            .iter()
            .copied()
            .reduce(|a, b| {
                if slots[b].score > slots[a].score {
                    b
                } else {
                    a
                }
            })
            .expect("a slot is on its own path");
        if keeper == j {
            let max = slots[j].score;
            merged[j] = if max == f32::NEG_INFINITY {
                max
            } else {
                max + same
                    .iter()
                    .map(|&k| (slots[k].score - max).exp())
                    .sum::<f32>()
                    .ln()
            };
        }
    }
    for (slot, score) in slots.iter_mut().zip(merged) {
        slot.score = score;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(score: f32, next_frame: usize, hash: i64) -> Slot {
        Slot {
            score,
            next_frame,
            hash,
            last_label: NO_LABEL,
            ..Slot::default()
        }
    }

    #[test]
    fn slots_on_one_path_merge_into_the_first_best() {
        let (a, b) = (-1.0f32, -2.0f32);
        let mut slots = vec![
            slot(b, 3, 7),
            slot(a, 3, 7),
            slot(a, 4, 7),
            slot(-0.5, 3, 9),
        ];
        recombine(&mut slots);
        let merged = a + (1.0 + (b - a).exp()).ln();
        assert_eq!(slots[0].score, f32::NEG_INFINITY);
        assert!((slots[1].score - merged).abs() < 1e-6);
        // another frame, or another transcript, is another path
        assert_eq!(slots[2].score, a);
        assert_eq!(slots[3].score, -0.5);
    }
}
