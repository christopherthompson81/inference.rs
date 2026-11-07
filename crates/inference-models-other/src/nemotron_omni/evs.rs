//! Efficient Video Sampling, the reference's `compute_retention_mask`: of a video's `T x N` projected tokens, keep the
//! first tubelet's and those least like the same position one tubelet earlier, `max(N, int(T * N * (1 - rate)))` in
//! all. The reference reshapes each tubelet to a square grid first, which only it needs; the comparison is per position.

use inference_tensor::{D, DType, Result, Tensor};

// what the reference gives the first tubelet's tokens, above any `1 - cosine`, so they are always kept
const FIRST_TUBELET: f32 = 255.0;
// torch's `cosine_similarity` eps
const COSINE_EPS: f64 = 1e-8;

/// Each tubelet's kept positions, ascending, from the `(tubelets, tokens, hidden)` embeddings of one video.
pub fn retained(embeds: &Tensor, rate: f64) -> Result<Vec<Vec<usize>>> {
    let (tubelets, tokens, _) = embeds.dims3()?;
    let similarity = if tubelets > 1 {
        let embeds = embeds.to_dtype(DType::F32)?;
        let (later, earlier) = (
            embeds.narrow(0, 1, tubelets - 1)?,
            embeds.narrow(0, 0, tubelets - 1)?,
        );
        let dot = (&later * &earlier)?.sum(D::Minus1)?;
        let norms =
            (later.sqr()?.sum(D::Minus1)?.sqrt()? * earlier.sqr()?.sum(D::Minus1)?.sqrt()?)?;
        (dot / norms.clamp(COSINE_EPS, f64::MAX)?)?
            .flatten_all()?
            .to_vec1::<f32>()?
    } else {
        Vec::new()
    };
    Ok(keep(&similarity, tubelets, tokens, rate))
}

/// The kept positions of each tubelet given the cosine of every later tubelet's tokens to the one before, row-major.
pub fn keep(similarity: &[f32], tubelets: usize, tokens: usize, rate: f64) -> Vec<Vec<usize>> {
    let dissimilarity = std::iter::repeat_n(FIRST_TUBELET, tokens)
        .chain(similarity.iter().map(|s| 1.0 - s))
        .collect::<Vec<_>>();
    // Python's `int(T * N * (1 - q))`, the product taken left to right
    let budget = ((tubelets * tokens) as f64 * (1.0 - rate)) as usize;
    let count = tokens.max(budget).min(tubelets * tokens);
    let mut order = (0..dissimilarity.len()).collect::<Vec<_>>();
    // torch's stable descending argsort
    order.sort_by(|&a, &b| dissimilarity[b].total_cmp(&dissimilarity[a]));
    let mut kept = vec![Vec::new(); tubelets];
    for &position in &order[..count] {
        kept[position / tokens].push(position % tokens);
    }
    for positions in &mut kept {
        positions.sort_unstable();
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::keep;

    // three tubelets of four tokens at rate 0.5: six kept, the first tubelet's four and the two that changed most
    // (0.9 and the earlier of the 0.5 tie, as a stable sort keeps them)
    #[test]
    fn the_first_tubelet_stays_and_the_most_changed_tokens_follow() {
        let similarity = [0.9, 0.5, 0.95, 0.99, 0.1, 0.5, 0.97, 0.98];
        let kept = keep(&similarity, 3, 4, 0.5);
        assert_eq!(kept, [vec![0, 1, 2, 3], vec![1], vec![0]]);
        // never fewer than one tubelet's worth
        let kept = keep(&similarity, 3, 4, 0.99);
        assert_eq!(kept.iter().map(Vec::len).sum::<usize>(), 4);
    }
}
