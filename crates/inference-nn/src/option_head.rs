//! vllm-decision2's option head (llama.cpp's `vllm-decision2` architecture): over the normed hidden states of a
//! prompt that ends each option's span and then the question's span, an option scores the scaled dot product of
//! its key with the question's query plus a small MLP over both.
// prompt positions and the head's width stay far below what the casts lose
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Module};
use inference_tensor::{DType, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::decision::JointLayout;

/// The head's width, from its key projection, and its LayerNorms' epsilon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptionHeadConfig {
    pub head_dim: usize,
    pub norm_eps: f64,
}

pub struct OptionHead {
    option_norm: LayerNorm,
    query_norm: LayerNorm,
    key: Tensor,
    query: Tensor,
    option_mlp: Tensor,
    option_mlp_b: Tensor,
    query_mlp: Tensor,
    scorer_out: Tensor,
}

impl OptionHead {
    /// The head under llama.cpp's `decision.*` names, held in f32 (the converter writes it so).
    pub fn new(cfg: &OptionHeadConfig, hidden_size: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let vb = vb.set_dtype(DType::F32);
        let norm = |name: &str| -> Result<LayerNorm> {
            let vb = vb.pp(name);
            Ok(LayerNorm::new(
                vb.get(hidden_size, "weight")?,
                vb.get(hidden_size, "bias")?,
                cfg.norm_eps,
            ))
        };
        let proj = |name: &str| vb.pp(name).get((cfg.head_dim, hidden_size), "weight");
        Ok(Self {
            option_norm: norm("option_norm")?,
            query_norm: norm("query_norm")?,
            key: proj("key")?,
            query: proj("query")?,
            option_mlp: proj("option_mlp")?,
            option_mlp_b: vb.pp("option_mlp").get(cfg.head_dim, "bias")?,
            query_mlp: proj("query_mlp")?,
            scorer_out: vb.pp("scorer_out").get((1, cfg.head_dim), "weight")?,
        })
    }

    /// One score per option of `layout`, read at the last token of each option span against the last token of its
    /// one question span, from the prompt's normed hidden states `[tokens, hidden]`.
    pub fn forward(&self, hidden: &Tensor, layout: &JointLayout) -> Result<Vec<f32>> {
        let [(_, question)] = layout.questions.as_slice() else {
            inference_tensor::bail!("the option head reads one question a prompt");
        };
        let device = self.key.device();
        let hidden = hidden.to_device(device)?.to_dtype(DType::F32)?;
        let last = |span: &std::ops::Range<usize>| span.end.saturating_sub(1) as u32;
        let rows = layout
            .options
            .iter()
            .map(|(_, span)| last(span))
            .collect::<Vec<_>>();
        let options = hidden.index_select(&Tensor::new(rows, device)?, 0)?;
        let query = hidden.narrow(0, last(question) as usize, 1)?;
        let options = self.option_norm.forward(&options)?;
        let query = self.query_norm.forward(&query)?;
        let head_dim = self.key.dim(0)? as f64;
        let keys = options.matmul(&self.key.t()?)?;
        let q = query.matmul(&self.query.t()?)?;
        let bilinear = (keys.broadcast_mul(&q)?.sum(1)? / head_dim.sqrt())?;
        let mlp = options
            .matmul(&self.option_mlp.t()?)?
            .broadcast_add(&self.option_mlp_b)?
            .broadcast_add(&query.matmul(&self.query_mlp.t()?)?)?
            .gelu_erf()?;
        let nonlinear = mlp.matmul(&self.scorer_out.t()?)?.squeeze(1)?;
        (bilinear + nonlinear)?.to_vec1::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use inference_tensor::Device;

    use super::*;
    use crate::decision::QuestionKind;

    const HIDDEN: usize = 6;
    const HEAD_DIM: usize = 4;

    fn random(shape: (usize, usize), seed: f32) -> Tensor {
        let n = shape.0 * shape.1;
        let values = (0..n)
            .map(|i| ((i as f32 + seed) * 0.37).sin())
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
    }

    fn head() -> OptionHead {
        let norm = || {
            LayerNorm::new(
                Tensor::ones(HIDDEN, DType::F32, &Device::Cpu).unwrap(),
                Tensor::zeros(HIDDEN, DType::F32, &Device::Cpu).unwrap(),
                1e-5,
            )
        };
        OptionHead {
            option_norm: norm(),
            query_norm: norm(),
            key: random((HEAD_DIM, HIDDEN), 1.0),
            query: random((HEAD_DIM, HIDDEN), 2.0),
            option_mlp: random((HEAD_DIM, HIDDEN), 3.0),
            option_mlp_b: Tensor::zeros(HEAD_DIM, DType::F32, &Device::Cpu).unwrap(),
            query_mlp: random((HEAD_DIM, HIDDEN), 4.0),
            scorer_out: random((1, HEAD_DIM), 5.0),
        }
    }

    // options at 1..3 and 3..5, the question at 5..7: the head reads rows 2, 4 and 6
    fn layout() -> JointLayout {
        JointLayout {
            tokens: vec![0; 7],
            questions: vec![(QuestionKind::Choice, 5..7)],
            options: vec![(0, 1..3), (0, 3..5)],
        }
    }

    fn with_row(hidden: &Tensor, row: usize) -> Tensor {
        let mut rows = hidden.to_vec2::<f32>().unwrap();
        rows[row].iter_mut().for_each(|v| *v += 1.0);
        Tensor::new(rows, &Device::Cpu).unwrap()
    }

    #[test]
    fn an_option_scores_from_the_end_of_its_span_and_the_question() {
        let (head, layout) = (head(), layout());
        let hidden = random((7, HIDDEN), 0.0);
        let scores = head.forward(&hidden, &layout).unwrap();
        assert_eq!(scores.len(), 2);
        // rows inside a span but not at its end are not read
        for row in [0, 1, 3, 5] {
            assert_eq!(
                head.forward(&with_row(&hidden, row), &layout).unwrap(),
                scores
            );
        }
        let moved = head.forward(&with_row(&hidden, 2), &layout).unwrap();
        assert!(moved[0] != scores[0] && moved[1] == scores[1]);
        let both = head.forward(&with_row(&hidden, 6), &layout).unwrap();
        assert!(both[0] != scores[0] && both[1] != scores[1]);
    }
}
