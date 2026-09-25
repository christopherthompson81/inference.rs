use candle_core::{DType, Result, Tensor};
use serde::Serialize;

use super::config::LABELS;

#[derive(Debug, Clone, Serialize)]
pub struct LayoutDetection {
    pub class_id: usize,
    pub label: &'static str,
    pub score: f32,
    /// `[x1, y1, x2, y2]` in original image pixels.
    pub bbox: [f32; 4],
    /// Position in predicted reading order among the kept detections.
    pub reading_order: usize,
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

/// HF `_get_order_seqs`: rank of every query in the pairwise-vote reading order.
pub fn order_ranks(order_logits: &[Vec<f32>]) -> Vec<usize> {
    let q = order_logits.len();
    let s = |i: usize, j: usize| sigmoid(order_logits[i][j]);
    let votes: Vec<f32> = (0..q)
        .map(|j| {
            let before: f32 = (0..j).map(|i| s(i, j)).sum();
            let after: f32 = (j + 1..q).map(|i| 1. - s(j, i)).sum();
            before + after
        })
        .collect();
    let mut ptr: Vec<usize> = (0..q).collect();
    ptr.sort_by(|&a, &b| votes[a].total_cmp(&votes[b]));
    let mut rank = vec![0; q];
    for (r, &p) in ptr.iter().enumerate() {
        rank[p] = r;
    }
    rank
}

pub struct PostprocessArgs {
    pub threshold: f32,
    /// `(width, height)` of the source image.
    pub orig_size: (u32, u32),
}

/// Post-process a single batch item (`logits: (q, c)`, `boxes: (q, 4)`, `order_logits: (q, q)`).
pub fn postprocess(
    logits: &Tensor,
    boxes: &Tensor,
    order_logits: &Tensor,
    args: &PostprocessArgs,
) -> Result<Vec<LayoutDetection>> {
    let logits = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let boxes = boxes.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let order = order_ranks(&order_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?);
    let q = logits.len();
    let c = logits.first().map_or(0, |r| r.len());

    let mut flat: Vec<(f32, usize)> = logits
        .iter()
        .flatten()
        .enumerate()
        .map(|(i, &l)| (sigmoid(l), i))
        .collect();
    flat.sort_by(|a, b| b.0.total_cmp(&a.0));
    flat.truncate(q);

    let (w, h) = (args.orig_size.0 as f32, args.orig_size.1 as f32);
    let mut kept: Vec<(usize, LayoutDetection)> = flat
        .into_iter()
        .filter(|(score, _)| *score >= args.threshold)
        .map(|(score, i)| {
            let (qi, class_id) = (i / c, i % c);
            let [cx, cy, bw, bh] = [boxes[qi][0], boxes[qi][1], boxes[qi][2], boxes[qi][3]];
            let bbox = [
                (cx - 0.5 * bw) * w,
                (cy - 0.5 * bh) * h,
                (cx + 0.5 * bw) * w,
                (cy + 0.5 * bh) * h,
            ];
            let det = LayoutDetection {
                class_id,
                label: LABELS.get(class_id).copied().unwrap_or("unknown"),
                score,
                bbox,
                reading_order: 0,
            };
            (order[qi], det)
        })
        .collect();
    kept.sort_by_key(|(rank, _)| *rank);
    Ok(kept
        .into_iter()
        .enumerate()
        .map(|(i, (_, mut d))| {
            d.reading_order = i;
            d
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(q: usize, v: f32) -> Vec<Vec<f32>> {
        vec![vec![v; q]; q]
    }

    #[test]
    fn order_ranks_follow_pairwise_votes() {
        // s(i, j) ~ 1 for i < j means every earlier query precedes every later one
        assert_eq!(order_ranks(&constant(5, 50.)), vec![0, 1, 2, 3, 4]);
        assert_eq!(order_ranks(&constant(5, -50.)), vec![4, 3, 2, 1, 0]);
    }
}
